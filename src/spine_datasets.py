import logging
import random
import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import yaml
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_transforms(cfg: dict, split: str = "train"):
    size = cfg["data"]["image_size"]
    if split == "train":
        aug = cfg["augmentation"]["train"]
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=aug["horizontal_flip_p"]),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-aug["rotation_limit"], aug["rotation_limit"]),
                p=0.5,
            ),
            A.ElasticTransform(p=aug["elastic_p"]),
            A.GridDistortion(p=0.2),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=aug["clahe_p"]),
            A.GaussNoise(p=aug["noise_p"]),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})


# SPIDER mask label remapping
# Original: 0=bg, 1-7=vertebrae, 100=canal, 201-207=IVDs
# Remapped: 0-15 contiguous
SPIDER_REMAP = {
    0: 0,   1: 1,   2: 2,   3: 3,   4: 4,
    5: 5,   6: 6,   7: 7,
    100: 8,
    201: 9,  202: 10, 203: 11, 204: 12,
    205: 13, 206: 14, 207: 15,
}
SPIDER_REMAP_ARRAY = np.zeros(256, dtype=np.int64)
for k, v in SPIDER_REMAP.items():
    if k < 256:
        SPIDER_REMAP_ARRAY[k] = v


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Remap SPIDER mask values to contiguous 0-15."""
    out = np.zeros_like(mask, dtype=np.int64)
    for k, v in SPIDER_REMAP.items():
        out[mask == k] = v
    return out


class SPIDERDataset(Dataset):
    """
    SPIDER lumbar MRI segmentation dataset.
    Each HF study has a variable number of slices (8-143).
    We build a flat index of (study_idx, slice_idx) pairs.
    Returns (img_tensor [3,H,W], target_dict).
    """
    NUM_CLASSES = 16

    LABEL_MAP = {
        0: "background",    1: "L1_vertebra",  2: "L2_vertebra",
        3: "L3_vertebra",   4: "L4_vertebra",  5: "L5_vertebra",
        6: "S1_vertebra",   7: "other_vertebra",
        8: "spinal_canal",
        9: "L1L2_IVD",     10: "L2L3_IVD",   11: "L3L4_IVD",
        12: "L4L5_IVD",    13: "L5S1_IVD",   14: "IVD_14",
        15: "IVD_15",
    }

    def __init__(
        self,
        hf_split: str = "train",
        transform=None,
        cache_dir: str = "./data/.cache",
        flat_indices: Optional[List[int]] = None,
    ):
        console.print("[cyan]Loading SPIDER ({} split)...[/cyan]".format(hf_split))
        self.raw       = load_dataset(
            "cdoswald/SPIDER", name="default", split=hf_split,
            trust_remote_code=True, cache_dir=cache_dir,
        )
        self.transform = transform

        # Build flat index: list of (study_idx, slice_idx)
        self._pairs: List[Tuple[int, int]] = []
        for study_i in range(len(self.raw)):
            n = len(self.raw[study_i]["image"])
            for slice_j in range(n):
                self._pairs.append((study_i, slice_j))

        total = len(self._pairs)

        # Apply subset
        if flat_indices is not None:
            self._pairs = [self._pairs[i] for i in flat_indices]

        console.print(
            "[green]SPIDER:[/green] {} studies | {} total slices | {} used".format(
                len(self.raw), total, len(self._pairs)))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        study_idx, slice_idx = self._pairs[idx]
        sample   = self.raw[study_idx]

        # Image: 16-bit TIFF -> normalise to uint8
        img_pil = sample["image"][slice_idx]
        img_f32 = np.array(img_pil, dtype=np.float32)
        lo, hi  = img_f32.min(), img_f32.max()
        if hi > lo:
            img_u8 = ((img_f32 - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img_f32, dtype=np.uint8)

        # Mask: remap original labels to 0-15
        mask_pil = sample["mask"][slice_idx]
        mask_raw = np.array(mask_pil, dtype=np.int64)
        mask     = remap_mask(mask_raw)

        img_3ch  = np.stack([img_u8, img_u8, img_u8], axis=-1)  # H x W x 3

        if self.transform:
            out    = self.transform(image=img_3ch, mask=mask.astype(np.uint8))
            img_t  = out["image"]
            mask_t = torch.as_tensor(out["mask"], dtype=torch.long)
        else:
            img_t  = torch.tensor(img_3ch, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask_t = torch.tensor(mask, dtype=torch.long)

        return img_t, {
            "source":       "spider",
            "seg_mask":     mask_t,
            "grades":       torch.full((25,), -1, dtype=torch.long),
            "morph_labels": torch.full((3,), -1, dtype=torch.long),
        }


class RSNASpineDataset(Dataset):
    CONDITIONS = [
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ]
    LEVELS     = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
    GRADE      = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    NUM_TASKS  = 25

    def __init__(self, rsna_dir, split="train", val_fraction=0.15,
                 transform=None, image_size=512, seed=42):
        self.root      = Path(rsna_dir)
        self.transform = transform
        self.size      = image_size
        self.meta      = pd.read_csv(self.root / "train.csv")
        self.desc      = pd.read_csv(self.root / "train_series_descriptions.csv")
        self._build_label_matrix()
        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(self.meta))
        n_val   = int(len(indices) * val_fraction)
        self.indices = (indices[:n_val] if split == "val" else indices[n_val:]).tolist()
        console.print("[green]RSNA:[/green] {} studies ({})".format(len(self.indices), split))

    def _build_label_matrix(self):
        cols = ["{}_{}".format(c, l) for c in self.CONDITIONS for l in self.LEVELS]
        for col in cols:
            if col not in self.meta.columns:
                self.meta[col] = "Normal/Mild"
            self.meta[col] = self.meta[col].fillna("Normal/Mild")
        gm = self.GRADE
        self.label_matrix = (
            self.meta[cols].map(lambda x: gm.get(str(x), 0))
            .values.astype(np.int64))

    def _load_image(self, study_id):
        study_dir = self.root / "train_images" / str(study_id)
        if not study_dir.exists():
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        dirs = list(study_dir.iterdir())
        if not dirs:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        dcm_files = sorted(dirs[0].glob("*.dcm"))
        if not dcm_files:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        ds  = pydicom.dcmread(str(dcm_files[len(dcm_files) // 2]))
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = cv2.resize(arr.astype(np.uint8), (self.size, self.size))
        return np.stack([arr, arr, arr], axis=-1)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row      = self.meta.iloc[real_idx]
        img      = self._load_image(row.study_id)
        labels   = torch.tensor(self.label_matrix[real_idx], dtype=torch.long)
        if self.transform:
            out   = self.transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
            img_t = out["image"]
        else:
            img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img_t, {
            "source":       "rsna",
            "seg_mask":     torch.zeros(512, 512, dtype=torch.long),
            "grades":       labels,
            "morph_labels": torch.full((3,), -1, dtype=torch.long),
        }


# Keyword -> morph class, checked in this priority order (first match wins).
# NOTE: tuned against the *expected* structure of the Sudirman "Radiologists
# Notes" companion dataset (free-text per-level observations). The Colab
# notebook's inspection cell prints real note samples before training starts
# -- adjust these keyword lists there if real phrasing differs.
MORPH_CLASSES = {
    "Normal": 0, "Degenerated": 1, "Bulging": 2,
    "Herniated": 3, "Thinning": 4,
    "Disc Degeneration with Osteophyte formation": 5,
}
MORPH_KEYWORDS = [
    ("Herniated",  ("herniat", "extrusion", "sequestr")),
    ("Bulging",    ("bulg", "protrusion")),
    ("Disc Degeneration with Osteophyte formation", ("osteophyte", "spondylosis")),
    ("Thinning",   ("thin", "height loss", "reduced height")),
    ("Degenerated", ("degenerat", "desiccat", "black disc")),
]
MORPH_LEVELS = ["l3_l4", "l4_l5", "l5_s1"]

# The real "Radiologists Report.xlsx" (confirmed via Mendeley's file API,
# 2026-08) is ONE free-text note per patient covering all levels together
# (columns: "Patient ID", "Clinician's Notes") -- not one column per level
# as originally assumed. These patterns split a note into per-level
# segments (from one level mention to the next) before keyword-matching
# each segment separately. Levels never mentioned in a note default to
# "Normal" (absence-implies-normal is the common radiology-report
# convention, but isn't universally true -- see the notebook's class-
# distribution sanity check before trusting this on a new dataset).
LEVEL_PATTERNS = {
    "l3_l4": re.compile(r"l\s*3\s*[-/]\s*l?\s*4", re.I),
    "l4_l5": re.compile(r"l\s*4\s*[-/]\s*l?\s*5", re.I),
    "l5_s1": re.compile(r"l\s*5\s*[-/]\s*s?\s*1", re.I),
}


def parse_morph_note(note: str) -> int:
    """Map one free-text radiologist note to a MORPH_CLASSES index."""
    text = str(note).lower()
    for name, keywords in MORPH_KEYWORDS:
        if any(kw in text for kw in keywords):
            return MORPH_CLASSES[name]
    return MORPH_CLASSES["Normal"]


def split_note_by_level(note: str) -> dict:
    """Split one whole-patient note into {level: text} segments, each
    running from that level's mention to the next level mention (or end
    of note). Levels not mentioned are simply absent from the result."""
    text = str(note)
    matches = []
    for level, pat in LEVEL_PATTERNS.items():
        for m in pat.finditer(text):
            matches.append((m.start(), level))
    matches.sort()
    if not matches:
        return {}
    segments = {level: [] for level in LEVEL_PATTERNS}
    for i, (pos, level) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        segments[level].append(text[pos:end])
    return {level: " ".join(parts) for level, parts in segments.items() if parts}


class SudirmanDiscDataset(Dataset):
    """
    Sudirman et al. "Lumbar Spine MRI Dataset" (Mendeley, CC BY 4.0) +
    companion "Radiologists Notes" -- disc-morphology labels (Normal/
    Degenerated/Bulging/Herniated/Thinning/Osteophyte) for the lowest 3
    disc levels (L3/L4, L4/L5, L5/S1 -- "sacral" per the project brief).

    Expects, under `root`:
      images/<batch-folder>/<patient_id, zero-padded>/<series>/*.ima
                                                  -- from k57fr854j2
      Radiologists Report.xlsx (sheet "Sheet1",
      columns "Patient ID", "Clinician's Notes") -- from s6bgczr8s2

    Both the notes-file and images-folder structures above are confirmed
    against the real Mendeley download (2026-08). `_load_image` picks
    the T1-sagittal series specifically (a sagittal slice shows the
    whole spine, covering all 3 morph levels in one image) and falls
    back to any series found if no sagittal folder exists for a patient.
    """

    def __init__(self, root, split="train", val_fraction=0.15,
                 transform=None, image_size=512, seed=42):
        self.root      = Path(root)
        self.transform = transform
        self.size      = image_size
        self.notes     = pd.read_excel(
            self.root / "Radiologists Report.xlsx", sheet_name="Sheet1")

        note_col = "Clinician's Notes"
        segments = self.notes[note_col].map(split_note_by_level)
        self.morph_matrix = np.stack([
            segments.map(
                lambda segs, lvl=level: parse_morph_note(segs[lvl])
                if lvl in segs else MORPH_CLASSES["Normal"]
            ).astype(np.int64).values
            for level in MORPH_LEVELS
        ], axis=1)   # [N, 3]

        rng     = np.random.default_rng(seed)
        indices = rng.permutation(len(self.notes))
        n_val   = int(len(indices) * val_fraction)
        self.indices = (indices[:n_val] if split == "val" else indices[n_val:]).tolist()
        console.print("[green]Sudirman:[/green] {} studies ({})".format(
            len(self.indices), split))

    def _load_image(self, study_id):
        # Real layout (confirmed 2026-08 against the actual Mendeley
        # download): images/<batch-folder>/<patient_id, zero-padded>/
        # <series-name>/*.ima -- one level deeper than the original
        # images/<study_id>/*.dcm guess, with several MRI series per
        # patient (localizer, T2 axial, T1 sagittal, T1 axial). We
        # specifically want the T1 SAGITTAL series: a sagittal slice
        # shows the whole spine in one image, which is what a single
        # representative image per patient needs to cover L3/L4, L4/L5,
        # and L5/S1 together. Siemens' .ima extension is DICOM-compatible
        # -- pydicom reads it the same as .dcm.
        study_id_str = str(study_id).zfill(4)
        patient_dirs = [d for d in self.root.glob(f"images/**/{study_id_str}") if d.is_dir()]
        if not patient_dirs:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        patient_dir = patient_dirs[0]

        sag_dirs = [d for d in patient_dir.rglob("*") if d.is_dir() and "sag" in d.name.lower()]
        search_root = sag_dirs[0] if sag_dirs else patient_dir

        candidates = sorted(search_root.glob("*.ima")) or sorted(search_root.rglob("*.ima"))
        candidates = candidates or sorted(search_root.glob("*.dcm")) or sorted(search_root.rglob("*.dcm"))
        if candidates:
            ds  = pydicom.dcmread(str(candidates[len(candidates) // 2]))
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            arr = cv2.resize(arr.astype(np.uint8), (self.size, self.size))
            return np.stack([arr, arr, arr], axis=-1)
        jpgs = sorted(patient_dir.rglob("*.jpg"))
        if jpgs:
            img = cv2.imread(str(jpgs[len(jpgs) // 2]), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.size, self.size))
            return np.stack([img, img, img], axis=-1)
        return np.zeros((self.size, self.size, 3), dtype=np.uint8)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row      = self.notes.iloc[real_idx]
        study_id = row["Patient ID"] if "Patient ID" in self.notes.columns else real_idx
        img      = self._load_image(study_id)
        morph    = torch.tensor(self.morph_matrix[real_idx], dtype=torch.long)
        if self.transform:
            out   = self.transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
            img_t = out["image"]
        else:
            img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img_t, {
            "source":       "sudirman",
            "seg_mask":     torch.full((512, 512), -1, dtype=torch.long),
            "grades":       torch.full((25,), -1, dtype=torch.long),
            "morph_labels": morph,
        }


def spine_collate_fn(batch):
    """Collate (img, target_dict) pairs into batched tensors."""
    imgs    = torch.stack([b[0] for b in batch])
    keys    = batch[0][1].keys()
    targets = {}
    for key in keys:
        vals = [b[1][key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            targets[key] = torch.stack(vals)
        else:
            targets[key] = vals
    return imgs, targets


def build_dataloaders(cfg: dict):
    train_tfm      = build_transforms(cfg, "train")
    val_tfm        = build_transforms(cfg, "val")
    bs             = cfg["training"]["batch_size"]
    nw             = cfg["data"]["num_workers"]
    seed           = cfg["project"]["seed"]
    train_datasets = []
    val_datasets   = []

    # SPIDER
    try:
        # Load once to get the full flat index
        probe       = SPIDERDataset(cache_dir=cfg["data"]["cache_dir"])
        total       = len(probe)
        all_idx     = list(range(total))
        random.seed(seed)
        random.shuffle(all_idx)
        n_val       = int(total * 0.15)
        val_idx     = all_idx[:n_val]
        trn_idx     = all_idx[n_val:]

        trn_spider  = SPIDERDataset(
            cache_dir=cfg["data"]["cache_dir"],
            transform=train_tfm, flat_indices=trn_idx)
        val_spider  = SPIDERDataset(
            cache_dir=cfg["data"]["cache_dir"],
            transform=val_tfm,   flat_indices=val_idx)

        train_datasets.append(trn_spider)
        val_datasets.append(val_spider)
        console.print("[green]SPIDER added to dataloaders.[/green]")
    except Exception as e:
        console.print("[red]SPIDER failed:[/red] {}".format(e))
        import traceback; traceback.print_exc()

    # RSNA
    rsna_dir = Path(cfg["data"]["rsna_dir"])
    if (rsna_dir / "train.csv").exists():
        try:
            train_datasets.append(RSNASpineDataset(
                str(rsna_dir), split="train", transform=train_tfm,
                image_size=cfg["data"]["image_size"]))
            val_datasets.append(RSNASpineDataset(
                str(rsna_dir), split="val", transform=val_tfm,
                image_size=cfg["data"]["image_size"]))
            console.print("[green]RSNA added to dataloaders.[/green]")
        except Exception as e:
            console.print("[red]RSNA failed:[/red] {}".format(e))
    else:
        console.print("[yellow]RSNA not found - skipping.[/yellow]")

    # Sudirman (disc morphology -- Normal/Degenerated/Bulging/Herniated/
    # Thinning/Osteophyte, incl. the L5/S1 "sacral" level)
    sudirman_dir = Path(cfg["data"]["sudirman_dir"])
    if (sudirman_dir / "radiologist_notes.csv").exists():
        try:
            train_datasets.append(SudirmanDiscDataset(
                str(sudirman_dir), split="train", transform=train_tfm,
                image_size=cfg["data"]["image_size"]))
            val_datasets.append(SudirmanDiscDataset(
                str(sudirman_dir), split="val", transform=val_tfm,
                image_size=cfg["data"]["image_size"]))
            console.print("[green]Sudirman added to dataloaders.[/green]")
        except Exception as e:
            console.print("[red]Sudirman failed:[/red] {}".format(e))
    else:
        console.print("[yellow]Sudirman not found - skipping.[/yellow]")

    if not train_datasets:
        raise RuntimeError("No datasets loaded.")

    train_ds = ConcatDataset(train_datasets)
    val_ds   = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=False, drop_last=True,
        collate_fn=spine_collate_fn)
    val_loader   = DataLoader(
        val_ds,   batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=False,
        collate_fn=spine_collate_fn)

    console.print("[bold]Dataloaders:[/bold] {} train  {} val".format(
        len(train_ds), len(val_ds)))
    return train_loader, val_loader