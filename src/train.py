"""
src/train.py
PyTorch Lightning training module + trainer setup.
Handles: training loop, validation, logging, checkpointing,
         learning rate scheduling, and early stopping.
"""

import sys
import yaml
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
from rich.console import Console

sys.path.insert(0, ".")
from src.model import build_model
from src.losses import MultiTaskLoss
from src.spine_datasets import build_dataloaders

console = Console()


# ── METRICS HELPERS ───────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor,
               num_classes: int = 12, ignore_index: int = 0) -> torch.Tensor:
    """Mean Dice score across foreground classes."""
    pred_cls = pred.argmax(dim=1)           # [B, H, W]
    scores   = []
    for c in range(1, num_classes):         # skip background (0)
        if c == ignore_index:
            continue
        p = (pred_cls == c).float()
        t = (target   == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        scores.append((2 * inter + 1) / (denom + 1))
    if not scores:
        return torch.tensor(0.0)
    return torch.stack(scores).mean()


def cls_accuracy(pred: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """Per-task grade accuracy, ignoring sentinel -1 labels."""
    pred_cls    = pred.argmax(dim=-1)       # [B, num_tasks]
    valid       = (target >= 0)
    if valid.sum() == 0:
        return torch.tensor(0.0)
    correct = (pred_cls[valid] == target[valid]).float()
    return correct.mean()


# morph_accuracy shares cls_accuracy's exact logic (per-task argmax vs.
# sentinel-masked target) -- same helper, kept as an alias for readability
# in logs/call sites.
morph_accuracy = cls_accuracy


# ── LIGHTNING MODULE ──────────────────────────────────────────────────────

class LumbarLightningModule(L.LightningModule):
    """
    Lightning module wrapping the full diagnostic model.

    Handles:
      - Training + validation steps
      - Multi-task loss with learnable uncertainty weighting
      - Dice score (segmentation) and accuracy (classification) metrics
      - Cosine annealing LR schedule with linear warmup
      - Gradient clipping
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg         = cfg
        self.model       = build_model(cfg)
        self.loss_fn     = MultiTaskLoss(cfg, use_auto_weight=True)
        self.save_hyperparameters(cfg)

        self.num_seg_cls = cfg["model"]["num_seg_classes"]
        self.lr          = cfg["training"]["learning_rate"]
        self.wd          = cfg["training"]["weight_decay"]
        self.epochs      = cfg["training"]["epochs"]

    # ── FORWARD ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)

    # ── TRAINING STEP ────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        preds         = self.model(imgs)

        total_loss, breakdown = self.loss_fn(preds, targets)

        # Metrics
        dice = dice_score(
            preds["segmentation"],
            targets["seg_mask"].long(),
            self.num_seg_cls,
        )
        acc = cls_accuracy(
            preds["classification"],
            targets["grades"],
        )
        morph_acc = morph_accuracy(
            preds["morphology"],
            targets["morph_labels"],
        )

        # Log everything
        self.log("train/loss",       total_loss,          prog_bar=True)
        self.log("train/loss_det",   breakdown["loss_det"])
        self.log("train/loss_seg",   breakdown["loss_seg"])
        self.log("train/loss_cls",   breakdown["loss_cls"])
        self.log("train/loss_morph", breakdown["loss_morph"])
        self.log("train/dice",       dice,                prog_bar=True)
        self.log("train/acc",        acc,                 prog_bar=True)
        self.log("train/morph_acc",  morph_acc,           prog_bar=True)

        if "sigma_det" in breakdown:
            self.log("sigma/det",   breakdown["sigma_det"])
            self.log("sigma/seg",   breakdown["sigma_seg"])
            self.log("sigma/cls",   breakdown["sigma_cls"])
            self.log("sigma/morph", breakdown["sigma_morph"])

        return total_loss

    # ── VALIDATION STEP ──────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        preds         = self.model(imgs)

        total_loss, breakdown = self.loss_fn(preds, targets)

        dice = dice_score(
            preds["segmentation"],
            targets["seg_mask"].long(),
            self.num_seg_cls,
        )
        acc = cls_accuracy(
            preds["classification"],
            targets["grades"],
        )
        morph_acc = morph_accuracy(
            preds["morphology"],
            targets["morph_labels"],
        )

        self.log("val/loss",       total_loss, prog_bar=True, sync_dist=True)
        self.log("val/dice",       dice,       prog_bar=True, sync_dist=True)
        self.log("val/acc",        acc,        prog_bar=True, sync_dist=True)
        self.log("val/morph_acc",  morph_acc,  prog_bar=True, sync_dist=True)
        self.log("val/loss_seg",   breakdown["loss_seg"],   sync_dist=True)
        self.log("val/loss_cls",   breakdown["loss_cls"],   sync_dist=True)
        self.log("val/loss_morph", breakdown["loss_morph"], sync_dist=True)

        return total_loss

    # ── OPTIMIZER + SCHEDULER ────────────────────────────────────────────

    def configure_optimizers(self):
        # Separate param groups: lower LR for pretrained backbone
        backbone_params = list(self.model.encoder.backbone.parameters())
        other_params    = [p for p in self.model.parameters()
                           if not any(p is bp for bp in backbone_params)]
        other_params   += list(self.loss_fn.parameters())

        param_groups = [
            {"params": backbone_params, "lr": self.lr * 0.1},
            {"params": other_params,    "lr": self.lr},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.wd,
        )

        # Cosine annealing with linear warmup
        warmup_epochs = 5
        scheduler     = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.epochs - warmup_epochs,
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "monitor":   "val/loss",
            },
        }


# ── TRAINER FACTORY ───────────────────────────────────────────────────────

def build_trainer(cfg: dict, extra_callbacks: list = None) -> L.Trainer:
    output_dir = Path(cfg["project"]["output_dir"])
    log_dir    = Path(cfg["project"]["log_dir"])

    callbacks = [
        # Save best checkpoint by val/dice
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="best-{epoch:02d}-{val_dice:.3f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        # Stop if val/loss does not improve for 15 epochs
        EarlyStopping(
            monitor="val/loss",
            patience=15,
            mode="min",
            verbose=True,
        ),
        # Log LR every epoch
        LearningRateMonitor(logging_interval="epoch"),
        # Pretty progress bar
        RichProgressBar(),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    logger = CSVLogger(
        save_dir=str(log_dir),
        name="lumbar_mri_dx",
    )

    # Use CPU (no GPU detected) — switch to gpu if available
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices     = 1

    trainer = L.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator=accelerator,
        devices=devices,
        precision="32",             # use 32 on CPU; switch to 16-mixed on GPU
        gradient_clip_val=cfg["training"]["gradient_clip"],
        log_every_n_steps=5,
        val_check_interval=1.0,     # validate once per epoch
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=True,
    )

    console.print(
        "[green]Trainer built:[/green] "
        "{} epochs · {} · precision=32".format(
            cfg["training"]["epochs"], accelerator.upper()))
    return trainer


# ── MAIN TRAINING ENTRY POINT ─────────────────────────────────────────────

def train(cfg: dict, resume_ckpt: str = None, init_from: str = None,
          extra_callbacks: list = None):
    """
    Args:
        resume_ckpt     : full Lightning resume (same architecture, incl.
                          optimizer/epoch state) -- for continuing an
                          interrupted run.
        init_from       : partial weight init only (`strict=False`) -- for
                          starting a *new* run whose architecture has grown
                          (e.g. the morphology head added after `last.ckpt`
                          was saved). Encoder/det/seg/cls start from the old
                          weights; new params start random; optimizer/epoch
                          state is NOT restored.
        extra_callbacks : additional Lightning callbacks appended to the
                          default set (e.g. a Colab-side checkpoint-export
                          callback) -- see build_trainer().
    """
    console.rule("[bold cyan]Starting Training[/bold cyan]")

    # Data
    console.print("Loading datasets...")
    train_loader, val_loader = build_dataloaders(cfg)

    # Module
    module = LumbarLightningModule(cfg)

    if init_from:
        console.print("Partial-loading weights from {}...".format(init_from))
        ckpt = torch.load(init_from, map_location="cpu")
        full_state = ckpt.get("state_dict", ckpt)
        state = {k: v for k, v in full_state.items() if k.startswith("model.")}
        missing, unexpected = module.load_state_dict(state, strict=False)
        console.print("  matched   : {} tensors".format(len(state) - len(unexpected)))
        console.print("  new/random: {}".format(
            [k for k in missing if k.startswith("model.morph_head")][:3] +
            (["..."] if len(missing) > 3 else [])))

    # Trainer
    trainer = build_trainer(cfg, extra_callbacks=extra_callbacks)

    # Fit
    console.print("Training started. Logs -> {}".format(cfg["project"]["log_dir"]))
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt,
    )

    console.print("")
    console.print("[bold green]Training complete.[/bold green]")
    best = trainer.checkpoint_callback.best_model_path
    console.print("Best checkpoint: {}".format(best))
    return best


# ── SMOKE TEST (runs 2 batches without full dataset download) ─────────────

def smoke_test(cfg: dict):
    """
    Quick sanity check — runs 2 train steps + 1 val step
    using synthetic data. Does not require the dataset download.
    """
    console.rule("[bold cyan]Step 6 — Train Loop Smoke Test[/bold cyan]")

    from torch.utils.data import DataLoader, TensorDataset

    B  = cfg["training"]["batch_size"]
    H  = W = cfg["data"]["image_size"]
    T  = cfg["model"]["num_cls_tasks"]
    M  = cfg["model"]["num_morph_levels"]
    NG = cfg["model"]["num_morph_classes"]

    def make_loader(n_batches):
        imgs     = torch.randn(n_batches * B, 3, H, W)
        seg_masks = torch.cat([
            torch.randint(0, 12, (n_batches * B // 2, H, W)),
            torch.full((n_batches * B // 2, H, W), -1),
        ], dim=0)
        grades = torch.cat([
            torch.full((n_batches * B // 2, T), -1),
            torch.randint(0, 3, (n_batches * B // 2, T)),
        ], dim=0)
        # Third synthetic "source" -- covers the morph head like SPIDER/RSNA
        # cover seg/grades above -- every sample gets a random morph label
        # so the smoke test exercises loss_morph regardless of split size.
        morph_labels = torch.randint(0, NG, (n_batches * B, M))

        class SyntheticDS(torch.utils.data.Dataset):
            def __len__(self): return len(imgs)
            def __getitem__(self, i):
                return imgs[i], {
                    "seg_mask":     seg_masks[i],
                    "grades":       grades[i],
                    "morph_labels": morph_labels[i],
                    "source":       "spider" if seg_masks[i][0,0] >= 0 else "rsna",
                }

        return DataLoader(SyntheticDS(), batch_size=B, shuffle=False)

    train_loader = make_loader(4)   # 4 train batches
    val_loader   = make_loader(2)   # 2 val batches

    module  = LumbarLightningModule(cfg)

    # Quick 3-epoch trainer
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        precision="32",
        gradient_clip_val=cfg["training"]["gradient_clip"],
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[RichProgressBar()],
    )

    trainer.fit(module, train_loader, val_loader)

    console.print("")
    console.print("[bold green]Smoke test PASSED.[/bold green]")
    console.print("Model trains correctly end-to-end on synthetic data.")
    console.print("")
    console.print("When SPIDER download finishes, run full training with:")
    console.print("  python src/train.py --mode train")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "train"],
        default="smoke",
        help="smoke = quick synthetic test, train = full training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to fully resume from (same architecture)",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to checkpoint to partially load weights from (strict=False) "
             "-- use when the architecture grew, e.g. after adding morph_head",
    )
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "smoke":
        smoke_test(cfg)
    else:
        train(cfg, resume_ckpt=args.resume, init_from=args.init_from)
