"""
model_nft.py — standalone NFT model definition.

Extracted from train_new.py: contains only what is needed to *construct*
the NFT segmentation network for inference — the NFTUNet class and the
build_model() factory. All training-only machinery (dataloaders, losses,
metrics, schedulers, class-weighting, overlay logging, and the train/eval
loops) has been removed.

The backbone blocks (DoubleConv / Down / Up) are imported from
model.model_vanilla — unchanged — so the architecture stays byte-for-byte
identical to the one the checkpoint was trained with and weights load cleanly.

Usage
-----
    from model_nft import build_model          # if this file is on sys.path
    # or
    from model.model_nft import build_model    # if placed in the model/ package

    model = build_model()
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
"""

import os
import sys

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
#  Make the project's `model` package importable regardless of whether this
#  file sits in <project>/model/, <project>/train/, or <project>/ itself.
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.abspath(os.path.join(_HERE, "..")), _HERE):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)



# Number of output classes for the NFT head: background / nft.
# (Matches NFTConfig.NUM_CLASSES from the training project.)
NFT_NUM_CLASSES = 2


# ─────────────────────────────────────────────────────────────────────────────
#  Model  —  single-head UNet, 2 classes (background / nft).
#  Reuses the DoubleConv / Down / Up blocks from model_vanilla so the
#  backbone stays identical to the rest of the codebase; only the number
#  of output channels differs (2 instead of the 22-structure NUM_CLASSES).
# ─────────────────────────────────────────────────────────────────────────────


"""
model.py  —  Baseline UNet for multi-structure fetal head segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


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


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        f        = Config.BASE_FEATURES
        bilinear = Config.BILINEAR
        drop     = Config.DROPOUT
        factor   = 2 if bilinear else 1

        self.inc   = DoubleConv(Config.IN_CHANNELS, f)
        self.down1 = Down(f,      f * 2)
        self.down2 = Down(f * 2,  f * 4)
        self.down3 = Down(f * 4,  f * 8)
        self.down4 = Down(f * 8,  f * 16 // factor, dropout=drop)

        self.up1 = Up(f * 16, f * 8  // factor, bilinear, dropout=drop)
        self.up2 = Up(f * 8,  f * 4  // factor, bilinear)
        self.up3 = Up(f * 4,  f * 2  // factor, bilinear)
        self.up4 = Up(f * 2,  f,                bilinear)

        self.out_conv = nn.Conv2d(f, Config.NUM_CLASSES, kernel_size=1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.out_conv(x)   # raw logits (B, 33, H, W)


def build_model() -> UNet:
    return UNet()




class NFTUNet(nn.Module):
    def __init__(
        self,
        in_channels:   int   = 1,
        num_classes:   int   = NFT_NUM_CLASSES,
        base_features: int   = 64,
        bilinear:      bool  = True,
        dropout:       float = 0.2,
    ):
        super().__init__()
        f      = base_features
        factor = 2 if bilinear else 1

        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,      f * 2)
        self.down2 = Down(f * 2,  f * 4)
        self.down3 = Down(f * 4,  f * 8)
        self.down4 = Down(f * 8,  f * 16 // factor, dropout=dropout)

        self.up1 = Up(f * 16, f * 8  // factor, bilinear, dropout=dropout)
        self.up2 = Up(f * 8,  f * 4  // factor, bilinear)
        self.up3 = Up(f * 4,  f * 2  // factor, bilinear)
        self.up4 = Up(f * 2,  f,                bilinear)

        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.out_conv(x)   # raw logits (B, 2, H, W)


def build_model() -> NFTUNet:
    return NFTUNet()