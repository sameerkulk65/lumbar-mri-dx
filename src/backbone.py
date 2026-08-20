"""
src/backbone.py
Lightweight backbone (MobileViT-v2) + FPN + PAN neck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from rich.console import Console

console = Console()


# ── LIGHTWEIGHT BACKBONE ─────────────────────────────────────────────────

class LightweightBackbone(nn.Module):
    """
    MobileViT-v2 feature extractor via timm.
    Returns 4 feature maps at strides 4, 8, 16, 32 (P2-P5).
    Falls back to efficientnet_b3 if mobilevitv2 is unavailable.
    """

    def __init__(self, model_name: str = "mobilevitv2_100", pretrained: bool = True):
        super().__init__()
        try:
            self.encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
            console.print(f"[green]Backbone:[/green] {model_name} loaded")
        except Exception:
            console.print(
                f"[yellow]{model_name} unavailable, falling back to "
                f"efficientnet_b3[/yellow]")
            self.encoder = timm.create_model(
                "efficientnet_b3",
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
            console.print("[green]Backbone:[/green] efficientnet_b3 loaded")

        self.out_channels = self.encoder.feature_info.channels()
        console.print(
            "Feature channels: {}".format(self.out_channels))

    def forward(self, x: torch.Tensor):
        return self.encoder(x)      # list of 4 feature maps


# ── FPN + PAN NECK ────────────────────────────────────────────────────────

class FPNWithPAN(nn.Module):
    """
    Feature Pyramid Network (top-down) + Path Aggregation Network (bottom-up).
    Takes 4 backbone feature maps, outputs 4 enriched maps all at fpn_channels.
    """

    def __init__(self, in_channels: list, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1x1 convolutions to unify channel dims
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])

        # Smooth 3x3 after top-down merge
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

        # PAN bottom-up downsampling convolutions
        self.pan_down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, features: list) -> list:
        # Step 1 — lateral projections
        laterals = [l(f) for l, f in zip(self.lateral, features)]

        # Step 2 — top-down FPN merge
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="nearest",
            )

        # Step 3 — smooth after merge
        fpn_out = [s(l) for s, l in zip(self.smooth, laterals)]

        # Step 4 — bottom-up PAN merge
        pan_out = [fpn_out[0]]
        for i in range(len(fpn_out) - 1):
            pan_out.append(fpn_out[i + 1] + self.pan_down[i](pan_out[i]))

        return pan_out      # [P2, P3, P4, P5]


# ── COMBINED ENCODER ─────────────────────────────────────────────────────

class SpineEncoder(nn.Module):
    """
    Full encoder: backbone + FPN/PAN neck.
    Drop-in feature extractor for all task heads.
    """

    def __init__(
        self,
        backbone_name: str = "mobilevitv2_100",
        fpn_channels: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = LightweightBackbone(backbone_name, pretrained)
        self.neck     = FPNWithPAN(self.backbone.out_channels, fpn_channels)
        self.out_channels = fpn_channels

    def forward(self, x: torch.Tensor) -> list:
        features = self.backbone(x)
        return self.neck(features)


# ── WEIGHT INITIALISATION ─────────────────────────────────────────────────

def init_weights(module: nn.Module):
    """Kaiming init for conv layers, constant init for BN."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ── PARAMETER COUNT UTILITY ───────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_M":     round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "frozen_M":    round((total - trainable) / 1e6, 2),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    console.rule("[bold cyan]Step 2 — Backbone + FPN/PAN[/bold cyan]")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    backbone_name = cfg["model"]["backbone"]
    fpn_channels  = cfg["model"]["fpn_channels"]
    pretrained    = cfg["model"]["pretrained"]

    console.print("Building SpineEncoder...")
    encoder = SpineEncoder(
        backbone_name=backbone_name,
        fpn_channels=fpn_channels,
        pretrained=pretrained,
    )

    # Apply weight init to neck only (backbone keeps pretrained weights)
    init_weights(encoder.neck)
    console.print("[green]Weight initialisation applied to FPN/PAN neck.[/green]")

    # Parameter count
    params = count_parameters(encoder)
    console.print("Parameters:")
    console.print("  Total     : {} M".format(params["total_M"]))
    console.print("  Trainable : {} M".format(params["trainable_M"]))
    console.print("  Frozen    : {} M".format(params["frozen_M"]))

    # Forward pass test
    console.print("\nRunning forward pass with dummy input [2, 3, 512, 512]...")
    encoder.eval()
    dummy = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        fpn_maps = encoder(dummy)

    console.print("[green]FPN output shapes:[/green]")
    for i, fmap in enumerate(fpn_maps):
        console.print("  P{}: {}".format(i + 2, list(fmap.shape)))

    # Verify all strides are correct
    expected_strides = [4, 8, 16, 32]
    input_size = 512
    console.print("\nStride verification:")
    for i, fmap in enumerate(fpn_maps):
        actual_stride = input_size // fmap.shape[-1]
        expected      = expected_strides[i]
        status        = "OK" if actual_stride == expected else "CHECK"
        console.print("  P{}: stride {} — {}".format(
            i + 2, actual_stride, status))

    console.print("")
    console.print("[bold green]Step 2 COMPLETE.[/bold green]")
    console.print("Next: python src/heads.py")
