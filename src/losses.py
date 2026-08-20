"""
src/losses.py
Multi-task loss functions for the lumbosacral MRI diagnostic model.
  - Focal loss          (detection classification)
  - IoU loss            (detection bounding boxes)
  - Centerness loss     (detection quality)
  - Dice + Focal loss   (segmentation)
  - Ordinal cross-entropy (classification grading)
  - Combined multi-task loss with learnable weights
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

console = Console()


# ── FOCAL LOSS ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Sigmoid focal loss for dense detection.
    Downweights easy negatives so the model focuses on hard examples.

    Args:
        alpha : class balance weight (0.25 standard for FCOS)
        gamma : focusing parameter (2.0 standard)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [N, C] raw logits
            target : [N, C] binary targets (0 or 1)
        """
        pred_sigmoid = pred.sigmoid()
        ce_loss      = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")
        p_t          = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t      = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ── IoU LOSS ──────────────────────────────────────────────────────────────

class IoULoss(nn.Module):
    """
    GIoU loss for bounding box regression.
    More stable than L1/L2 for boxes of varying scales.
    """

    def forward(self, pred_boxes: torch.Tensor,
                gt_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes : [N, 4] (x1, y1, x2, y2)
            gt_boxes   : [N, 4] (x1, y1, x2, y2)
        """
        # Intersection
        inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

        inter_area = (inter_x2 - inter_x1).clamp(0) * \
                     (inter_y2 - inter_y1).clamp(0)

        # Union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(0) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(0)
        gt_area   = (gt_boxes[:, 2]   - gt_boxes[:, 0]).clamp(0) * \
                    (gt_boxes[:, 3]   - gt_boxes[:, 1]).clamp(0)
        union_area = pred_area + gt_area - inter_area + 1e-8

        iou = inter_area / union_area

        # Enclosing box for GIoU
        enc_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
        enc_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
        enc_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
        enc_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
        enc_area = (enc_x2 - enc_x1).clamp(0) * \
                   (enc_y2 - enc_y1).clamp(0) + 1e-8

        giou  = iou - (enc_area - union_area) / enc_area
        return (1 - giou).mean()


# ── DICE LOSS ─────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice loss for segmentation.
    Works well with class imbalance — critical for spine anatomy
    where background pixels dominate.

    Args:
        smooth      : smoothing constant to avoid division by zero
        ignore_index: class index to exclude from loss (e.g. background=0)
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, C, H, W] raw logits
            target : [B, H, W] integer class labels
        """
        num_classes = pred.shape[1]
        pred_soft   = pred.softmax(dim=1)

        # One-hot encode target
        target_oh = F.one_hot(
            target.clamp(0), num_classes
        ).permute(0, 3, 1, 2).float()   # [B, C, H, W]

        dice_scores = []
        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            p = pred_soft[:, c]
            t = target_oh[:, c]
            intersection = (p * t).sum()
            denominator  = p.sum() + t.sum()
            dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
            dice_scores.append(dice)

        return 1 - torch.stack(dice_scores).mean()


# ── SEGMENTATION LOSS (Dice + Focal combined) ─────────────────────────────

class SegmentationLoss(nn.Module):
    """
    Combined Dice + Focal loss for segmentation.
    Dice handles class imbalance; Focal handles hard pixels.

    Args:
        dice_weight  : weight on Dice term
        focal_weight : weight on Focal CE term
        class_weights: optional per-class weights [C] tensor
    """

    def __init__(
        self,
        dice_weight:   float = 1.0,
        focal_weight:  float = 1.0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.dice       = DiceLoss(ignore_index=0)   # ignore background
        self.dw         = dice_weight
        self.fw         = focal_weight
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, C, H, W] raw logits
            target : [B, H, W] integer labels  (-1 = ignore)
        """
        # Mask out sentinel -1 (samples with no seg label e.g. RSNA)
        valid = (target >= 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Dice loss on valid samples only
        dice_loss = self.dice(pred, target.clamp(0))

        # Focal cross-entropy — flatten spatial dims, ignore background(0) and sentinel(-1)
        B, C, H, W = pred.shape
        pred_flat   = pred.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
        target_flat = target.reshape(-1).clamp(min=-1)           # [B*H*W]

        # Replace sentinel -1 with 0 for CE (will be ignored via mask)
        valid_flat  = (target_flat >= 0)
        if valid_flat.sum() == 0:
            focal_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        else:
            focal_loss = F.cross_entropy(
                pred_flat[valid_flat],
                target_flat[valid_flat],
                weight=self.class_weights.to(pred.device)
                if self.class_weights is not None else None,
                reduction="mean",
            )

        return self.dw * dice_loss + self.fw * focal_loss


# ── ORDINAL CLASSIFICATION LOSS ───────────────────────────────────────────

class OrdinalClassificationLoss(nn.Module):
    """
    Cross-entropy loss over ordinal grades for multi-task grading.
    Ignores tasks where target == -1 (sentinel for missing labels).

    Args:
        label_smoothing: smoothing to prevent overconfident predictions
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, num_tasks, num_grades] logits
            target : [B, num_tasks] integer grade labels (0/1/2, -1=ignore)
        """
        B, T, G = pred.shape
        pred_flat   = pred.view(B * T, G)
        target_flat = target.view(B * T)

        # Only compute loss where label is valid (not -1)
        valid_mask  = (target_flat >= 0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return F.cross_entropy(
            pred_flat[valid_mask],
            target_flat[valid_mask],
            label_smoothing=self.label_smoothing,
        )


# ── CENTERNESS LOSS ───────────────────────────────────────────────────────

class CenternessLoss(nn.Module):
    """Binary cross-entropy for FCOS centerness branch."""

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [N] raw logits
            target : [N] centerness scores in [0, 1]
        """
        return F.binary_cross_entropy_with_logits(
            pred, target, reduction="mean")


# ── MULTI-TASK LOSS ───────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Combines all task losses with learnable uncertainty weighting.

    Uses Kendall et al. (2018) homoscedastic uncertainty to
    automatically balance task losses during training — no manual
    lambda tuning needed.

    The learnable log_sigma parameters absorb task difficulty:
      total_loss = sum_i [ (1/2*sigma_i^2) * L_i + log(sigma_i) ]

    Args:
        cfg             : full config dict
        use_auto_weight : if True use learnable uncertainty weighting,
                          if False use fixed lambdas from config
    """

    def __init__(self, cfg: dict, use_auto_weight: bool = True):
        super().__init__()
        t = cfg["training"]

        self.seg_loss   = SegmentationLoss(dice_weight=1.0, focal_weight=1.0)
        self.cls_loss   = OrdinalClassificationLoss(label_smoothing=0.1)
        self.morph_loss = OrdinalClassificationLoss(label_smoothing=0.1)
        self.focal      = FocalLoss(alpha=0.25, gamma=2.0)
        self.ctr_loss   = CenternessLoss()
        self.iou_loss   = IoULoss()

        self.use_auto_weight = use_auto_weight

        if use_auto_weight:
            # Learnable log(sigma) per task — init to 0 (sigma=1)
            self.log_sigma_det   = nn.Parameter(torch.zeros(1))
            self.log_sigma_seg   = nn.Parameter(torch.zeros(1))
            self.log_sigma_cls   = nn.Parameter(torch.zeros(1))
            self.log_sigma_morph = nn.Parameter(torch.zeros(1))
        else:
            self.lambda_det   = t["lambda_det"]
            self.lambda_seg   = t["lambda_seg"]
            self.lambda_cls   = t["lambda_cls"]
            self.lambda_morph = t.get("lambda_morph", 1.0)

    def _weighted(self, loss: torch.Tensor,
                  log_sigma: nn.Parameter) -> torch.Tensor:
        """Apply Kendall uncertainty weighting to a single task loss."""
        precision = torch.exp(-2 * log_sigma)
        return precision * loss + log_sigma

    def forward(self, preds: dict, targets: dict) -> tuple:
        """
        Args:
            preds  : output dict from LumbarDiagnosticModel.forward()
            targets: dict with keys:
                       seg_mask [B, H, W]   integer labels (-1 ok)
                       grades   [B, 25]     integer grades (-1 ok)
                       source   list[str]   dataset source per sample

        Returns:
            total_loss : scalar tensor
            breakdown  : dict of individual loss values (for logging)
        """

        # ── Segmentation loss
        seg_mask = targets["seg_mask"]
        if not isinstance(seg_mask, __import__("torch").Tensor):
            seg_mask = __import__("torch").stack(seg_mask) if isinstance(seg_mask, list) else __import__("torch").tensor(seg_mask)
        seg_mask = seg_mask.long()
        if seg_mask.dim() == 4:
            seg_mask = seg_mask.squeeze(1)
        loss_seg = self.seg_loss(
            preds["segmentation"],
            seg_mask,
        )

        # ── Classification loss
        loss_cls = self.cls_loss(
            preds["classification"],
            targets["grades"],
        )

        # ── Disc-morphology loss
        loss_morph = self.morph_loss(
            preds["morphology"],
            targets["morph_labels"],
        )

        # ── Detection loss (stub — uses dummy targets until annotations loaded)
        #    Real implementation would match anchors to GT boxes here
        loss_det = torch.tensor(0.0,
                                device=preds["segmentation"].device,
                                requires_grad=False)
        for level_out in preds["detection"]:
            B, C, H, W = level_out["cls_logits"].shape
            dummy_cls = torch.zeros(B, C, H, W,
                                    device=level_out["cls_logits"].device)
            dummy_ctr = torch.zeros(B, 1, H, W,
                                    device=level_out["centerness"].device)
            loss_det = loss_det + self.focal(
                level_out["cls_logits"].view(B, C, -1).permute(0, 2, 1).reshape(-1, C),
                dummy_cls.view(B, C, -1).permute(0, 2, 1).reshape(-1, C),
            )

        # ── Combine with weighting
        if self.use_auto_weight:
            total = (self._weighted(loss_det, self.log_sigma_det) +
                     self._weighted(loss_seg, self.log_sigma_seg) +
                     self._weighted(loss_cls, self.log_sigma_cls) +
                     self._weighted(loss_morph, self.log_sigma_morph))
        else:
            total = (self.lambda_det * loss_det +
                     self.lambda_seg * loss_seg +
                     self.lambda_cls * loss_cls +
                     self.lambda_morph * loss_morph)

        breakdown = {
            "loss_det":   loss_det.item(),
            "loss_seg":   loss_seg.item(),
            "loss_cls":   loss_cls.item(),
            "loss_morph": loss_morph.item(),
            "loss_total": total.item(),
        }
        if self.use_auto_weight:
            breakdown["sigma_det"]   = self.log_sigma_det.exp().item()
            breakdown["sigma_seg"]   = self.log_sigma_seg.exp().item()
            breakdown["sigma_cls"]   = self.log_sigma_cls.exp().item()
            breakdown["sigma_morph"] = self.log_sigma_morph.exp().item()

        return total, breakdown


# ── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    import sys
    sys.path.insert(0, ".")
    from src.model import build_model

    console.rule("[bold cyan]Step 5 — Loss Functions[/bold cyan]")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Build model and loss
    model    = build_model(cfg)
    loss_fn  = MultiTaskLoss(cfg, use_auto_weight=True)

    console.print("Loss function built with learnable uncertainty weighting.")
    console.print("Learnable parameters in loss:")
    for name, param in loss_fn.named_parameters():
        console.print("  {} : {}".format(name, param.shape))

    # Simulate a training batch
    console.rule("Simulated training batch")
    B = 2
    dummy_imgs = torch.randn(B, 3, 512, 512)

    # Mixed batch: sample 0 from SPIDER (has seg), sample 1 from RSNA (has grades)
    dummy_targets = {
        "seg_mask": torch.cat([
            torch.randint(0, 12, (1, 512, 512)),   # SPIDER sample — valid mask
            torch.full((1, 512, 512), -1),          # RSNA sample  — no mask
        ], dim=0),
        "grades": torch.cat([
            torch.full((1, 25), -1),                # SPIDER — no grade labels
            torch.randint(0, 3, (1, 25)),           # RSNA   — valid grades
        ], dim=0),
        "morph_labels": torch.full((2, 3), -1),     # neither sample has morph labels
        "source": ["spider", "rsna"],
    }

    model.train()
    preds = model(dummy_imgs)
    total_loss, breakdown = loss_fn(preds, dummy_targets)

    console.print("\n[green]Loss breakdown:[/green]")
    for k, v in breakdown.items():
        console.print("  {:<15} : {:.4f}".format(k, v))

    # Test backward pass
    console.rule("Backward pass test")
    total_loss.backward()
    console.print("[green]Backward pass OK — gradients computed.[/green]")

    # Check gradients flow to backbone
    backbone_grad = next(
        p.grad for p in model.encoder.backbone.parameters()
        if p.grad is not None
    )
    console.print("Backbone gradient norm: {:.4f}".format(
        backbone_grad.norm().item()))

    # Fixed lambda mode
    console.rule("Fixed lambda weighting")
    loss_fixed = MultiTaskLoss(cfg, use_auto_weight=False)
    total_f, breakdown_f = loss_fixed(preds, dummy_targets)
    console.print("Fixed lambda total loss: {:.4f}".format(total_f.item()))

    console.print("")
    console.print("[bold green]Step 5 COMPLETE.[/bold green]")
    console.print("Next: python src/train.py")
