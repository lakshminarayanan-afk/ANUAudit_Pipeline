import torch
import torch.nn as nn
import torch.nn.functional as F
from source_codes.seg.UNET_SoftMAk.config import Config

# from config import Config

# ─────────────────────────────────────────────────────────────────────────────
#  Building blocks  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2  with optional dropout."""
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
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
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
        dy = x2.shape[2] - x1.shape[2]
        dx = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Dual-head UNet
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Shared encoder + decoder with two independent output heads:

        out_conv  — softmax head  → (B, NUM_CLASSES,      H, W)
        calv_conv — sigmoid head  → (B, NUM_CALV_CLASSES, H, W)

    Parameters are identical to the original single-head UNet except
    the final layer is replaced by the two heads above.
    """

    def __init__(self):
        super().__init__()
        f        = Config.BASE_FEATURES        # 64
        bilinear = Config.BILINEAR
        drop     = Config.DROPOUT
        factor   = 2 if bilinear else 1

        # ── Shared encoder ────────────────────────────────────────────────
        self.inc   = DoubleConv(Config.IN_CHANNELS, f)
        self.down1 = Down(f,      f * 2)
        self.down2 = Down(f * 2,  f * 4)
        self.down3 = Down(f * 4,  f * 8)
        self.down4 = Down(f * 8,  f * 16 // factor, dropout=drop)

        # ── Shared decoder ────────────────────────────────────────────────
        self.up1 = Up(f * 16, f * 8  // factor, bilinear, dropout=drop)
        self.up2 = Up(f * 8,  f * 4  // factor, bilinear)
        self.up3 = Up(f * 4,  f * 2  // factor, bilinear)
        self.up4 = Up(f * 2,  f,                bilinear)

        # ── Head 1: inner structures (softmax, 23 classes) ─────────────
        self.out_conv = nn.Conv2d(f, Config.NUM_CLASSES, kernel_size=1)

        # ── Head 2: calvarium (sigmoid, 2 independent binary channels) ────
        # A small dedicated head — two extra conv layers before the 1×1
        # gives the head a bit of its own representational capacity
        # without adding much cost (~0.5% extra params).
        self.calv_conv = nn.Sequential(
            nn.Conv2d(f, f // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f // 2, Config.NUM_CALV_CLASSES, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
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
            x : (B, 1, H, W)  normalised grayscale

        Returns dict:
            "inner" : (B, 23, H, W)  raw logits — apply softmax for probs
            "calv"  : (B,  2, H, W)  raw logits — apply sigmoid for probs
        """
        # ── Shared encoder ────────────────────────────────────────────────
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ── Shared decoder ────────────────────────────────────────────────
        d  = self.up1(x5, x4)
        d  = self.up2(d,  x3)
        d  = self.up3(d,  x2)
        d  = self.up4(d,  x1)   # (B, f, H, W)  — shared feature map

        # ── Two independent heads ─────────────────────────────────────────
        logits_inner = self.out_conv(d)   # (B, 23, H, W)
        logits_calv  = self.calv_conv(d)  # (B,  2, H, W)

        return {"inner": logits_inner, "calv": logits_calv}


def build_model() -> UNet:
    return UNet()