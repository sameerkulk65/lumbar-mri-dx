"""
src/model.py
Unified LumbarDiagnosticModel — assembles encoder + all three heads
into a single nn.Module with one forward pass.
"""

import torch
import torch.nn as nn
import yaml
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, ".")
from src.backbone import SpineEncoder, init_weights, count_parameters
from src.heads import DetectionHead, SegmentationHead, ClassificationHead, UncertaintyWrapper

console = Console()


# ── UNIFIED MODEL ────────────────────────────────────────────────────────

class LumbarDiagnosticModel(nn.Module):
    """
    Full multi-task lumbosacral MRI diagnostic model.

    Single forward pass returns:
      detection     : list of 4 dicts (one per FPN level)
                      each with keys cls_logits, box_pred, centerness
      segmentation  : [B, num_seg_classes, H, W]
      classification: [B, num_tasks, num_grades]

    Args:
        cfg: parsed config.yaml dict
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        # Encoder — backbone + FPN/PAN neck
        self.encoder = SpineEncoder(
            backbone_name=m["backbone"],
            fpn_channels=m["fpn_channels"],
            pretrained=m["pretrained"],
        )

        # Task heads
        self.det_head = DetectionHead(
            in_channels=m["fpn_channels"],
            num_classes=m["num_det_classes"],
        )
        self.seg_head = SegmentationHead(
            fpn_channels=m["fpn_channels"],
            num_classes=m["num_seg_classes"],
        )
        self.cls_head = ClassificationHead(
            fpn_channels=m["fpn_channels"],
            num_tasks=m["num_cls_tasks"],
            num_grades=m["num_grades"],
        )
        # Disc-morphology head (Normal/Degenerated/Bulging/Herniated/
        # Thinning/Osteophyte) -- a different taxonomy from cls_head's
        # RSNA severity grades, covering the lowest disc levels incl. L5/S1.
        self.morph_head = ClassificationHead(
            fpn_channels=m["fpn_channels"],
            num_tasks=m["num_morph_levels"],
            num_grades=m["num_morph_classes"],
        )

        # Uncertainty wrapper around classification head
        self.cls_uncertain = UncertaintyWrapper(self.cls_head, n_passes=20)

        # Initialise neck and head weights (backbone keeps pretrained)
        init_weights(self.encoder.neck)
        init_weights(self.det_head)
        init_weights(self.seg_head)
        init_weights(self.cls_head)
        init_weights(self.morph_head)

        console.print("[green]LumbarDiagnosticModel built.[/green]")

    def forward(self, x: torch.Tensor, heads=None) -> dict:
        """
        Args:
            x     : [B, 3, H, W] normalised MRI image tensor
            heads : optional set of {"detection","segmentation",
                    "classification","morphology"} to compute. Defaults to
                    all four (training/full-inference behaviour). Pass a
                    smaller set to skip unused heads' compute -- e.g.
                    GradCAM/uncertainty only need "classification".
        Returns:
            dict with the requested keys among: detection, segmentation,
            classification, morphology
        """
        if heads is None:
            heads = {"detection", "segmentation", "classification", "morphology"}
        fpn_maps = self.encoder(x)

        out = {}
        if "detection" in heads:
            out["detection"] = self.det_head(fpn_maps)
        if "segmentation" in heads:
            out["segmentation"] = self.seg_head(fpn_maps)
        if "classification" in heads:
            out["classification"] = self.cls_head(fpn_maps)
        if "morphology" in heads:
            out["morphology"] = self.morph_head(fpn_maps)
        return out

    def predict(self, x: torch.Tensor, uncertainty: bool = False) -> dict:
        """
        Inference-time forward pass.
        If uncertainty=True, runs MC-Dropout on classification head.

        Args:
            x          : [B, 3, H, W]
            uncertainty: whether to compute epistemic uncertainty
        Returns:
            dict with softmax probabilities and optional uncertainty estimates
        """
        self.eval()
        with torch.no_grad():
            fpn_maps = self.encoder(x)
            det_out  = self.det_head(fpn_maps)
            seg_out  = self.seg_head(fpn_maps)

        # Segmentation — argmax to class indices
        seg_pred = seg_out.argmax(dim=1)            # [B, H, W]

        # Detection — sigmoid on class logits
        det_pred = []
        for level in det_out:
            det_pred.append({
                "cls_probs":  level["cls_logits"].sigmoid(),
                "box_pred":   level["box_pred"],
                "centerness": level["centerness"].sigmoid(),
                "stride":     level["stride"],
            })

        # Classification
        if uncertainty:
            cls_mean, cls_std = self.cls_uncertain.predict_with_uncertainty(fpn_maps)
            cls_probs = cls_mean.softmax(dim=-1)
            return {
                "detection":       det_pred,
                "seg_mask":        seg_pred,
                "cls_probs":       cls_probs,
                "cls_uncertainty": cls_std,
            }
        else:
            with torch.no_grad():
                cls_out   = self.cls_head(fpn_maps)
            cls_probs = cls_out.softmax(dim=-1)
            return {
                "detection":  det_pred,
                "seg_mask":   seg_pred,
                "cls_probs":  cls_probs,
            }

    def freeze_backbone(self):
        """Freeze backbone weights — useful for head-only fine-tuning."""
        for param in self.encoder.backbone.parameters():
            param.requires_grad = False
        console.print("[yellow]Backbone frozen.[/yellow]")

    def unfreeze_backbone(self):
        """Unfreeze all weights for full fine-tuning."""
        for param in self.encoder.backbone.parameters():
            param.requires_grad = True
        console.print("[green]Backbone unfrozen.[/green]")

    def summary(self):
        """Print a rich parameter summary table."""
        table = Table(title="Model Summary", show_lines=True)
        table.add_column("Component",   style="cyan",    min_width=28)
        table.add_column("Params (M)",  style="green",   justify="right")
        table.add_column("Trainable",   style="yellow",  justify="right")

        components = {
            "Backbone (MobileViT-v2)": self.encoder.backbone,
            "FPN + PAN neck":          self.encoder.neck,
            "Detection head":          self.det_head,
            "Segmentation head":       self.seg_head,
            "Classification head":     self.cls_head,
        }

        total_params = 0
        total_train  = 0
        for name, module in components.items():
            p = count_parameters(module)
            total_params += p["total_M"]
            total_train  += p["trainable_M"]
            table.add_row(
                name,
                str(p["total_M"]),
                str(p["trainable_M"]),
            )

        table.add_row(
            "[bold]TOTAL[/bold]",
            "[bold]{}[/bold]".format(round(total_params, 2)),
            "[bold]{}[/bold]".format(round(total_train, 2)),
        )
        console.print(table)


# ── MODEL FACTORY ─────────────────────────────────────────────────────────

def build_model(cfg: dict) -> LumbarDiagnosticModel:
    """Build and return the full model from config."""
    model = LumbarDiagnosticModel(cfg)
    return model


def load_checkpoint(model: LumbarDiagnosticModel,
                    ckpt_path: str,
                    strict: bool = True) -> LumbarDiagnosticModel:
    """Load saved weights into model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Strip 'model.' prefix added by Lightning if present
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
    console.print(f"[green]Checkpoint loaded:[/green] {ckpt_path}")
    return model


def save_checkpoint(model: LumbarDiagnosticModel,
                    save_path: str,
                    extra: dict = None):
    """Save model weights to disk."""
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, save_path)
    console.print(f"[green]Checkpoint saved:[/green] {save_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from pathlib import Path

    console.rule("[bold cyan]Step 4 — Unified Model[/bold cyan]")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Build model
    model = build_model(cfg)
    model.summary()

    # Test standard forward pass
    console.rule("Forward pass — training mode")
    model.train()
    dummy = torch.randn(2, 3, 512, 512)
    out   = model(dummy)

    console.print("\n[green]Training outputs:[/green]")
    console.print("  detection levels : {}".format(len(out["detection"])))
    console.print("  segmentation     : {}".format(list(out["segmentation"].shape)))
    console.print("  classification   : {}".format(list(out["classification"].shape)))

    # Test inference predict()
    console.rule("Inference — predict() without uncertainty")
    pred = model.predict(dummy, uncertainty=False)
    console.print("\n[green]Inference outputs:[/green]")
    console.print("  seg_mask  : {}".format(list(pred["seg_mask"].shape)))
    console.print("  cls_probs : {}".format(list(pred["cls_probs"].shape)))
    console.print("  cls sum   : {:.4f}  (should be ~1.0 per task)".format(
        pred["cls_probs"][0, 0].sum().item()))

    # Test inference with MC-Dropout uncertainty
    console.rule("Inference — predict() with uncertainty")
    pred_u = model.predict(dummy, uncertainty=True)
    console.print("\n[green]Uncertainty outputs:[/green]")
    console.print("  cls_probs       : {}".format(list(pred_u["cls_probs"].shape)))
    console.print("  cls_uncertainty : {}".format(list(pred_u["cls_uncertainty"].shape)))
    console.print("  avg uncertainty : {:.4f}".format(
        pred_u["cls_uncertainty"].mean().item()))

    # Test freeze / unfreeze
    console.rule("Backbone freeze / unfreeze")
    model.freeze_backbone()
    trainable_frozen = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    console.print("  Trainable params (frozen backbone): {} M".format(
        round(trainable_frozen / 1e6, 2)))

    model.unfreeze_backbone()
    trainable_full = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    console.print("  Trainable params (full model)     : {} M".format(
        round(trainable_full / 1e6, 2)))

    # Save a test checkpoint
    console.rule("Checkpoint save / load")
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, "outputs/checkpoints/test_checkpoint.pt",
                    extra={"epoch": 0, "val_loss": 999.0})
    model2 = build_model(cfg)
    load_checkpoint(model2, "outputs/checkpoints/test_checkpoint.pt")
    console.print("[green]Checkpoint round-trip OK.[/green]")

    console.print("")
    console.print("[bold green]Step 4 COMPLETE.[/bold green]")
    console.print("Next: python src/losses.py")
