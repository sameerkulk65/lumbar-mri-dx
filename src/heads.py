"""
src/heads.py
Three parallel task heads:
  1. Detection head   — anchor-free FCOS bounding boxes
  2. Segmentation head — lightweight U-decoder pixel masks
  3. Classification head — multi-label ordinal grading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

console = Console()


# ── SHARED BUILDING BLOCKS ───────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DepthwiseSeparable(nn.Module):
    """Depthwise separable conv — lightweight replacement for 3x3 conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


# ── HEAD 1: DETECTION (FCOS-style anchor-free) ───────────────────────────

class DetectionHead(nn.Module):
    """
    Anchor-free detection head applied to each FPN level independently.
    Predicts per-pixel:
      - class logits  : [B, num_classes, H, W]
      - box distances : [B, 4, H, W]  (left, right, top, bottom)
      - centerness    : [B, 1, H, W]

    All FPN levels share the same head weights (as in FCOS).
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 8,
                 num_convs: int = 4):
        super().__init__()

        # Shared tower for cls and reg branches
        cls_tower, reg_tower = [], []
        for _ in range(num_convs):
            cls_tower.append(DepthwiseSeparable(in_channels, in_channels))
            reg_tower.append(DepthwiseSeparable(in_channels, in_channels))

        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        # Output predictors
        self.cls_pred        = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred        = nn.Conv2d(in_channels, 4, 1)
        self.centerness_pred = nn.Conv2d(in_channels, 1, 1)

        # Scale factors per FPN level (learnable)
        self.scales = nn.Parameter(torch.ones(4))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialise cls bias for focal loss (prior prob = 0.01)
        import math
        bias_value = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(self, fpn_outputs: list) -> list:
        results = []
        for i, feat in enumerate(fpn_outputs):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_logits  = self.cls_pred(cls_feat)
            # exp with learnable scale — keeps box distances positive
            box_pred    = (self.reg_pred(reg_feat) * self.scales[i]).exp()
            centerness  = self.centerness_pred(reg_feat)

            results.append({
                "cls_logits": cls_logits,   # [B, num_classes, H, W]
                "box_pred":   box_pred,     # [B, 4, H, W]
                "centerness": centerness,   # [B, 1, H, W]
                "stride":     2 ** (i + 2),
            })
        return results


# ── HEAD 2: SEGMENTATION (Lightweight U-decoder) ─────────────────────────

class SegmentationHead(nn.Module):
    """
    Lightweight U-decoder that fuses all FPN levels at P2 resolution
    then upsamples to full image resolution.

    Output: [B, num_classes, H, W] — same spatial size as input image.
    """

    def __init__(self, fpn_channels: int = 256, num_classes: int = 12):
        super().__init__()

        # Upsamplers for P3, P4, P5 → P2 resolution
        self.up3 = nn.Upsample(scale_factor=2,  mode="bilinear", align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4,  mode="bilinear", align_corners=False)
        self.up5 = nn.Upsample(scale_factor=8,  mode="bilinear", align_corners=False)

        # Fuse all 4 FPN levels (4 x fpn_channels → 256)
        self.fuse = nn.Sequential(
            ConvBnRelu(fpn_channels * 4, 256, kernel=1, padding=0),
            ConvBnRelu(256, 128),
            ConvBnRelu(128, 128),
        )

        # Final upsampling to full resolution + prediction
        self.upsample_final = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False)

        self.predict = nn.Sequential(
            ConvBnRelu(128, 64),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, fpn_outputs: list) -> torch.Tensor:
        p2, p3, p4, p5 = fpn_outputs

        # Bring everything to P2 spatial resolution
        p3_up = self.up3(p3)
        p4_up = self.up4(p4)
        p5_up = self.up5(p5)

        # Handle any rounding mismatches in spatial dims
        target_h, target_w = p2.shape[-2], p2.shape[-1]
        p3_up = F.interpolate(p3_up, size=(target_h, target_w), mode="nearest")
        p4_up = F.interpolate(p4_up, size=(target_h, target_w), mode="nearest")
        p5_up = F.interpolate(p5_up, size=(target_h, target_w), mode="nearest")

        # Concatenate and fuse
        fused = self.fuse(torch.cat([p2, p3_up, p4_up, p5_up], dim=1))

        # Upsample x4 to restore to input image resolution
        out = self.upsample_final(fused)
        return self.predict(out)    # [B, num_classes, H, W]


# ── HEAD 3: CLASSIFICATION (Multi-label ordinal grading) ─────────────────

class ClassificationHead(nn.Module):
    """
    Multi-task ordinal grading head.
    25 independent tasks (5 conditions x 5 disc levels),
    each predicting 3 ordinal grades: Normal/Mild, Moderate, Severe.

    Output: [B, num_tasks, num_grades]
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        num_tasks: int = 25,
        num_grades: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_tasks  = num_tasks
        self.num_grades = num_grades

        # Global average pool P4 and P5, concatenate for richer context
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fpn_channels * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_tasks * num_grades),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fpn_outputs: list) -> torch.Tensor:
        # Use P4 and P5 — highest semantic content
        p4, p5 = fpn_outputs[2], fpn_outputs[3]

        p4_pooled = self.gap(p4)    # [B, C, 1, 1]
        p5_pooled = self.gap(p5)    # [B, C, 1, 1]

        # Concatenate along channel dim then flatten
        combined = torch.cat([p4_pooled, p5_pooled], dim=1)  # [B, 2C, 1, 1]

        out = self.classifier(combined)                 # [B, num_tasks*num_grades]
        return out.view(-1, self.num_tasks, self.num_grades)  # [B, 25, 3]


# ── UNCERTAINTY HEAD (MC-Dropout wrapper) ────────────────────────────────

class UncertaintyWrapper(nn.Module):
    """
    Wraps any head to enable Monte Carlo Dropout inference.
    Call model.train() to activate dropout during inference,
    then average N forward passes to get mean + std (epistemic uncertainty).
    """

    def __init__(self, head: nn.Module, n_passes: int = 20):
        super().__init__()
        self.head     = head
        self.n_passes = n_passes

    def forward(self, fpn_outputs: list) -> torch.Tensor:
        return self.head(fpn_outputs)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, fpn_outputs: list
    ) -> tuple:
        self.head.train()   # keep dropout active
        preds = torch.stack(
            [self.head(fpn_outputs) for _ in range(self.n_passes)], dim=0
        )
        self.head.eval()
        mean = preds.mean(0)
        std  = preds.std(0)
        return mean, std    # both [B, num_tasks, num_grades]


# ── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    import sys
    sys.path.insert(0, ".")
    from src.backbone import SpineEncoder

    console.rule("[bold cyan]Step 3 — Task Heads[/bold cyan]")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    fpn_ch    = cfg["model"]["fpn_channels"]
    n_det_cls = cfg["model"]["num_det_classes"]
    n_seg_cls = cfg["model"]["num_seg_classes"]
    n_tasks   = cfg["model"]["num_cls_tasks"]
    n_grades  = cfg["model"]["num_grades"]

    # Build encoder + all three heads
    console.print("Building encoder and heads...")
    encoder  = SpineEncoder(cfg["model"]["backbone"], fpn_ch,
                            cfg["model"]["pretrained"])
    det_head = DetectionHead(fpn_ch, n_det_cls)
    seg_head = SegmentationHead(fpn_ch, n_seg_cls)
    cls_head = ClassificationHead(fpn_ch, n_tasks, n_grades)
    cls_unc  = UncertaintyWrapper(cls_head, n_passes=10)

    # Parameter counts
    def params_M(m):
        return round(sum(p.numel() for p in m.parameters()) / 1e6, 2)

    console.print("\nParameter counts:")
    console.print("  Encoder (backbone+FPN) : {} M".format(params_M(encoder)))
    console.print("  Detection head         : {} M".format(params_M(det_head)))
    console.print("  Segmentation head      : {} M".format(params_M(seg_head)))
    console.print("  Classification head    : {} M".format(params_M(cls_head)))
    total = params_M(encoder) + params_M(det_head) + params_M(seg_head) + params_M(cls_head)
    console.print("  TOTAL                  : {} M".format(round(total, 2)))

    # Forward pass
    console.print("\nRunning forward pass [2, 3, 512, 512]...")
    encoder.eval(); det_head.eval(); seg_head.eval(); cls_head.eval()
    dummy = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        fpn_maps = encoder(dummy)

        det_out  = det_head(fpn_maps)
        seg_out  = seg_head(fpn_maps)
        cls_out  = cls_head(fpn_maps)

    console.print("\n[green]Detection outputs (per FPN level):[/green]")
    for i, d in enumerate(det_out):
        console.print("  P{}: cls={} box={} ctr={}".format(
            i+2,
            list(d["cls_logits"].shape),
            list(d["box_pred"].shape),
            list(d["centerness"].shape),
        ))

    console.print("\n[green]Segmentation output:[/green]")
    console.print("  {}  (should be [2, 12, 512, 512])".format(list(seg_out.shape)))

    console.print("\n[green]Classification output:[/green]")
    console.print("  {}  (should be [2, 25, 3])".format(list(cls_out.shape)))

    # Uncertainty test
    console.print("\n[green]Uncertainty estimation (10 MC passes):[/green]")
    cls_unc.head.train()
    mean, std = cls_unc.predict_with_uncertainty(fpn_maps)
    console.print("  Mean shape : {}".format(list(mean.shape)))
    console.print("  Std  shape : {}".format(list(std.shape)))
    console.print("  Avg uncertainty: {:.4f}".format(std.mean().item()))

    console.print("")
    console.print("[bold green]Step 3 COMPLETE.[/bold green]")
    console.print("Next: python src/model.py")
