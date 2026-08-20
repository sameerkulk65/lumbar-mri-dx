
import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

sys.path.insert(0, ".")
from src.model import build_model

console = Console()


def export_torchscript(model, save_path="outputs/exports/lumbar_model.pt", image_size=512):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x, heads={"segmentation", "classification", "morphology"})
            return out["segmentation"], out["classification"], out["morphology"]

    wrapper = Wrapper(model)
    wrapper.eval()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, strict=False)
    traced.save(save_path)
    size_mb = Path(save_path).stat().st_size / 1e6
    console.print("[green]TorchScript exported:[/green] {} -- {:.1f} MB".format(save_path, size_mb))
    return save_path


class ClinicalInferenceEngine:
    CONDITIONS = [
        "Spinal canal stenosis",
        "Left neural foraminal narrowing",
        "Right neural foraminal narrowing",
        "Left subarticular stenosis",
        "Right subarticular stenosis",
    ]
    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    GRADES = ["Normal/Mild", "Moderate", "Severe"]
    # Disc-morphology taxonomy -- distinct from CONDITIONS/GRADES above.
    # L5/S1 labeled "Sacral" per the project brief (the sacrum is one fused
    # bone with no discs of its own beyond the L5/S1 junction).
    MORPH_LEVELS  = ["L3/L4", "L4/L5", "L5/S1 (Sacral)"]
    MORPH_CLASSES = [
        "Normal", "Degenerated", "Bulging", "Herniated",
        "Thinning", "Disc Degeneration with Osteophyte formation",
    ]
    SEG_LABELS = {
        0: "Background",  1: "L1 vertebra", 2: "L2 vertebra",
        3: "L3 vertebra", 4: "L4 vertebra", 5: "L5 vertebra",
        6: "L1 IVD",      7: "L2 IVD",      8: "L3 IVD",
        9: "L4 IVD",      10: "L5 IVD",     11: "Spinal canal",
    }

    def __init__(self, model_path):
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        console.print("[green]Engine loaded:[/green] {}".format(model_path))

    def preprocess(self, img_array):
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=0)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            img_array = img_array.transpose(2, 0, 1)
        img = img_array.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.tensor(img[None], dtype=torch.float32)

    def predict(self, img_array):
        t0  = time.time()
        inp = self.preprocess(img_array)
        with torch.no_grad():
            seg_logits, cls_logits, morph_logits = self.model(inp)
        ms = (time.time() - t0) * 1000

        seg_mask  = seg_logits[0].argmax(0).numpy().astype(np.uint8)
        cls_probs = torch.softmax(cls_logits, dim=-1)[0].numpy()
        grades    = cls_probs.argmax(-1)

        findings = {}
        for i, cond in enumerate(self.CONDITIONS):
            for j, level in enumerate(self.LEVELS):
                key = "{}_{}".format(cond, level)
                g   = grades[i * 5 + j]
                findings[key] = {
                    "grade":      self.GRADES[g],
                    "confidence": float(cls_probs[i * 5 + j, g]),
                    "probs":      cls_probs[i * 5 + j].tolist(),
                }

        seg_summary = {}
        for label_id, label_name in self.SEG_LABELS.items():
            px = int((seg_mask == label_id).sum())
            if px > 0:
                seg_summary[label_name] = px

        morph_probs = torch.softmax(morph_logits, dim=-1)[0].numpy()
        morph_types = morph_probs.argmax(-1)
        morphology  = {}
        for j, level in enumerate(self.MORPH_LEVELS):
            m = morph_types[j]
            morphology[level] = {
                "type":       self.MORPH_CLASSES[m],
                "confidence": float(morph_probs[j, m]),
                "probs":      morph_probs[j].tolist(),
            }

        return {
            "findings":    findings,
            "seg_mask":    seg_mask,
            "seg_summary": seg_summary,
            "morphology":  morphology,
            "latency_ms":  round(ms, 1),
        }


def to_fhir_report(result, patient_id="P001", study_id="S001"):
    snomed = {
        "Normal/Mild": {"code": "17621005", "display": "Normal"},
        "Moderate":    {"code": "6736007",  "display": "Moderate severity"},
        "Severe":      {"code": "24484000", "display": "Severe"},
    }
    observations = []
    for finding_name, info in result["findings"].items():
        grade = info["grade"]
        observations.append({
            "resourceType": "Observation",
            "status":       "preliminary",
            "code":         {"text": finding_name.replace("_", " ")},
            "valueCodeableConcept": {
                "coding": [{"system": "http://snomed.info/sct", **snomed[grade]}]
            },
            "component": [{
                "code":         {"text": "confidence"},
                "valueDecimal": round(info["confidence"], 4),
            }],
        })

    # Disc-morphology findings -- no verified SNOMED mapping for this
    # taxonomy, so these carry a plain text code rather than a guessed one.
    for level, info in result.get("morphology", {}).items():
        observations.append({
            "resourceType": "Observation",
            "status":       "preliminary",
            "code":         {"text": "Disc morphology {}".format(level)},
            "valueCodeableConcept": {"text": info["type"]},
            "component": [{
                "code":         {"text": "confidence"},
                "valueDecimal": round(info["confidence"], 4),
            }],
        })

    severe   = [k for k, v in result["findings"].items() if v["grade"] == "Severe"]
    moderate = [k for k, v in result["findings"].items() if v["grade"] == "Moderate"]
    abnormal_morph = [
        "{} ({})".format(lvl, v["type"])
        for lvl, v in result.get("morphology", {}).items() if v["type"] != "Normal"
    ]

    conclusion = "AI-assisted preliminary findings. "
    if severe:
        conclusion += "Severe: {}. ".format(", ".join(severe[:2]))
    if moderate:
        conclusion += "Moderate: {}. ".format(", ".join(moderate[:2]))
    if abnormal_morph:
        conclusion += "Disc morphology: {}. ".format(", ".join(abnormal_morph))
    conclusion += "Radiologist review required."

    return {
        "resourceType": "DiagnosticReport",
        "status":       "preliminary",
        "subject":      {"reference": "Patient/{}".format(patient_id)},
        "issued":       datetime.utcnow().isoformat() + "Z",
        "imagingStudy": [{"reference": "ImagingStudy/{}".format(study_id)}],
        "result":       observations,
        "conclusion":   conclusion,
        "extension":    [{"url": "inference-latency-ms", "valueDecimal": result["latency_ms"]}],
    }


def print_results(result):
    table = Table(title="Clinical Findings", show_lines=True)
    table.add_column("Condition",  style="cyan",  min_width=38)
    table.add_column("Grade",      style="white", justify="center")
    table.add_column("Confidence", style="green", justify="right")
    colors = {"Normal/Mild": "green", "Moderate": "yellow", "Severe": "red"}
    for finding, info in result["findings"].items():
        g = info["grade"]
        c = colors[g]
        table.add_row(
            finding.replace("_", " "),
            "[{c}]{g}[/{c}]".format(c=c, g=g),
            "{:.1f}%".format(info["confidence"] * 100),
        )
    console.print(table)
    console.print("Latency: {} ms".format(result["latency_ms"]))

    seg_table = Table(title="Segmentation Summary", show_lines=True)
    seg_table.add_column("Structure", style="cyan")
    seg_table.add_column("Pixels",    style="green", justify="right")
    for struct, px in result["seg_summary"].items():
        seg_table.add_row(struct, str(px))
    console.print(seg_table)

    morph_table = Table(title="Disc Morphology", show_lines=True)
    morph_table.add_column("Level",      style="cyan")
    morph_table.add_column("Type",       style="white")
    morph_table.add_column("Confidence", style="green", justify="right")
    for level, info in result.get("morphology", {}).items():
        morph_table.add_row(
            level, info["type"], "{:.1f}%".format(info["confidence"] * 100))
    console.print(morph_table)


if __name__ == "__main__":
    console.rule("[bold cyan]Step 8 - Deploy[/bold cyan]")

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    console.rule("Building model")
    model = build_model(cfg)
    model.eval()

    console.rule("TorchScript export")
    model_path = export_torchscript(
        model,
        save_path="outputs/exports/lumbar_model.pt",
        image_size=cfg["data"]["image_size"],
    )

    console.rule("Loading inference engine")
    engine = ClinicalInferenceEngine(model_path)

    console.rule("Running inference")
    dummy_img = (np.random.randn(512, 512, 3) * 127 + 128).clip(0, 255).astype(np.uint8)
    result    = engine.predict(dummy_img)

    console.rule("Results")
    print_results(result)

    console.rule("FHIR report")
    fhir      = to_fhir_report(result, patient_id="P001", study_id="S001")
    fhir_path = Path("outputs/exports/fhir_report.json")
    fhir_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fhir_path, "w", encoding="utf-8") as f:
        json.dump(fhir, f, indent=2)
    console.print("[green]FHIR saved:[/green] {}".format(fhir_path))
    console.print("Observations : {}".format(len(fhir["result"])))
    console.print("Conclusion   : {}".format(fhir["conclusion"]))

    console.print("")
    console.print("[bold green]Step 8 COMPLETE - Full pipeline working end to end.[/bold green]")
    console.print("")
    console.print("When SPIDER download finishes run full training:")
    console.print("  python src/train.py --mode train")
