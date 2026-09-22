"""
model_multiplane.py — Shared-encoder, 3-head UNet for coronal / sagittal /
full_body_coronal spine segmentation.

Architecture
────────────
DoubleConv / Down / Up blocks and Kaiming init are UNCHANGED, copied
verbatim from model_spine.py (and identical to the fetal dual-head
model.py's blocks). Only the output stage is extended: instead of one
out_conv, there are three — one per plane — following the same pattern
model.py used to go from 1 head to 2 (out_conv + calv_conv), just
extended from 2 heads to 3.

All three heads are always computed on every forward pass; the training
loop (train_multiplane.py) picks which head's logits to use for loss,
based on each sample's `plane` tag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Seg_Config
Config = Seg_Config.Spine

# ─────────────────────────────────────────────────────────────────────────────
#  Building blocks — UNCHANGED from model_spine.py / model.py
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2 with optional dropout."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downsampling with maxpool then double conv."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling then double conv."""
    def __init__(self, in_ch, out_ch, bilinear=True, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, dropout)
        else:
            self.up   = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 dimensions if needed
        dy = x2.shape[2] - x1.shape[2]
        dx = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Shared-encoder, 3-head UNet
# ─────────────────────────────────────────────────────────────────────────────

class UNetMultiPlane(nn.Module):
    """
    Shared encoder + decoder (unchanged architecture) with three independent
    1x1-conv output heads:

        out_conv_coronal      — (B, NUM_CLASSES_PER_PLANE["coronal"],           H, W)
        out_conv_sagittal     — (B, NUM_CLASSES_PER_PLANE["sagittal"],          H, W)
        out_conv_full_body    — (B, NUM_CLASSES_PER_PLANE["full_body_coronal"], H, W)

    forward(x) always computes all three heads and returns a dict keyed by
    plane name; the caller (train/infer script) selects the relevant head
    per sample.
    """

    def __init__(
        self,
        in_channels:   int = Config.IN_CHANNELS,
        base_features: int = Config.BASE_FEATURES,
        bilinear:      bool = Config.BILINEAR,
        dropout:       float = Config.DROPOUT,
        num_classes_per_plane: dict = None,
    ):
        super().__init__()
        f      = base_features
        factor = 2 if bilinear else 1

        ncp = num_classes_per_plane or Config.NUM_CLASSES_PER_PLANE

        # ── Shared encoder ────────────────────────────────────────────────
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,      f * 2)
        self.down2 = Down(f * 2,  f * 4)
        self.down3 = Down(f * 4,  f * 8)
        self.down4 = Down(f * 8,  f * 16 // factor, dropout=dropout)

        # ── Shared decoder ────────────────────────────────────────────────
        self.up1 = Up(f * 16, f * 8  // factor, bilinear, dropout=dropout)
        self.up2 = Up(f * 8,  f * 4  // factor, bilinear)
        self.up3 = Up(f * 4,  f * 2  // factor, bilinear)
        self.up4 = Up(f * 2,  f,                bilinear)

        # ── Three independent output heads (1x1 conv, same pattern the
        # fetal model.py used to add calv_conv alongside out_conv) ────────
        self.out_conv_coronal            = nn.Conv2d(f, ncp["coronal"],              kernel_size=1)
        self.out_conv_sagittal_exclusive = nn.Conv2d(f, ncp["sagittal_exclusive"],    kernel_size=1)
        # +1 input channel: a normalized vertical-position map, concatenated
        # only here so the region head can tell cervical (top) apart from
        # sacral (bottom) — the other heads/encoder are untouched.
        self.out_conv_sagittal_regions   = nn.Conv2d(f + 1, Config.NUM_REGIONS_PER_PLANE["sagittal"], kernel_size=1)
        self.out_conv_full_body          = nn.Conv2d(f, ncp["full_body_coronal"],     kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init — unchanged from model_spine.py / model.py."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x : (B, 1, H, W) normalized grayscale images

        Returns dict:
            "coronal"           : (B, NUM_CLASSES_PER_PLANE["coronal"],           H, W) raw logits
            "sagittal"          : (B, NUM_CLASSES_PER_PLANE["sagittal"],          H, W) raw logits
            "full_body_coronal" : (B, NUM_CLASSES_PER_PLANE["full_body_coronal"], H, W) raw logits

        All three heads are always computed — the caller decides which
        head's output to use per sample (via that sample's `plane` tag).
        """
        # ── Shared encoder ────────────────────────────────────────────────
        x1 = self.inc(x)      # (B, 64,   H,    W)
        x2 = self.down1(x1)   # (B, 128,  H/2,  W/2)
        x3 = self.down2(x2)   # (B, 256,  H/4,  W/4)
        x4 = self.down3(x3)   # (B, 512,  H/8,  W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16)  (bilinear -> 512)

        # ── Shared decoder ────────────────────────────────────────────────
        d = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        d = self.up2(d,  x3)  # (B, 256, H/4, W/4)
        d = self.up3(d,  x2)  # (B, 128, H/2, W/2)
        d = self.up4(d,  x1)  # (B, 64,  H,   W)   — shared feature map

        # ── Three independent heads ───────────────────────────────────────
        # Normalized vertical position map (0 at top, 1 at bottom), same
        # for every sample — gives the region head absolute position along
        # the spine, which local texture alone can't provide.
        B, _, H, W = d.shape
        ypos = torch.linspace(0, 1, H, device=d.device, dtype=d.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        d_region = torch.cat([d, ypos], dim=1)

        return {
            "coronal":             self.out_conv_coronal(d),
            "sagittal_exclusive":  self.out_conv_sagittal_exclusive(d),
            "sagittal_regions":    self.out_conv_sagittal_regions(d_region),
            "full_body_coronal":   self.out_conv_full_body(d),
        }


def build_model() -> UNetMultiPlane:
    """Factory function to build the model."""
    return UNetMultiPlane()


# ─────────────────────────────────────────────────────────────────────────────
#  Quick shape check  →  python -m model.model_multiplane
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    x     = torch.randn(2, 1, 512, 512)
    out   = model(x)

    for plane, logits in out.items():
        print(f"{plane:20s}: {logits.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {total_params:,}")