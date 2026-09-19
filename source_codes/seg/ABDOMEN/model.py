from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from source_codes.seg.ABDOMEN.config import Config_abd


# ─────────────────────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, dropout)
        else:
            self.up   = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dy = x2.shape[2] - x1.shape[2]
        dx = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Squeeze-and-Excitation  (channel attention)
#  Helps the decoder focus on relevant feature channels for small structures.
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


# ─────────────────────────────────────────────────────────────────────────────
#  ASPP  (Atrous Spatial Pyramid Pooling)
# ─────────────────────────────────────────────────────────────────────────────

class ASPPConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        return F.interpolate(self.net(x), size=size,
                             mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Multi-scale context at the bottleneck.
    Branches: 1×1 conv  +  3×3 dilated (r=2,4,6)  +  global avg-pool.
    At 512×512 input, bottleneck is 32×32 — small dilations are better.
    """

    def __init__(self, in_ch: int, aspp_ch: int = 256,
                 dilations: tuple[int, ...] = (2, 4, 6),
                 dropout: float = 0.3):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, aspp_ch, 1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.ModuleList(
            [ASPPConv(in_ch, aspp_ch, d) for d in dilations]
        )
        self.pool_br = ASPPPooling(in_ch, aspp_ch)

        n_branches = 1 + len(dilations) + 1
        self.project = nn.Sequential(
            nn.Conv2d(aspp_ch * n_branches, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = (
            [self.conv1x1(x)]
            + [d(x) for d in self.dilated]
            + [self.pool_br(x)]
        )
        return self.project(torch.cat(parts, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Deep Supervision auxiliary head
#  Attached to decoder level up2 (1/4 resolution).
#  Only used during training — disabled at inference.
# ─────────────────────────────────────────────────────────────────────────────

class AuxHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        out = self.head(x)
        return F.interpolate(out, size=target_size,
                             mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
#  UNet
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    UNet encoder-decoder with:
      • ASPP at the bottleneck
      • SE (squeeze-excitation) attention on skip connections
      • Deep supervision auxiliary head from decoder level up2
      • Deeper 3-conv classification head
      • Output: (B, NUM_CHANNELS, H, W) raw logits  — all sigmoid channels

    Channels 0–12 = 13 structures,  channel 13 = skin line.

    During training, forward() returns (main_logits, aux_logits).
    During eval,     forward() returns  main_logits only.
    """

    def __init__(self):
        super().__init__()
        f      = Config_abd.BASE_FEATURES    # 64
        bilin  = Config_abd.BILINEAR
        drop   = Config_abd.DROPOUT
        factor = 2 if bilin else 1

        # ── Encoder ──────────────────────────────────────────────────────
        self.inc   = DoubleConv(Config_abd.IN_CHANNELS, f)
        self.down1 = Down(f,      f * 2)
        self.down2 = Down(f * 2,  f * 4)
        self.down3 = Down(f * 4,  f * 8)
        self.down4 = Down(f * 8,  f * 16 // factor, dropout=drop)

        # ── SE on skip connections ────────────────────────────────────────
        self.se1 = SEBlock(f)
        self.se2 = SEBlock(f * 2)
        self.se3 = SEBlock(f * 4)
        self.se4 = SEBlock(f * 8)

        # ── Bottleneck: ASPP ─────────────────────────────────────────────
        bn_ch     = f * 16 // factor   # 512
        self.aspp = ASPP(bn_ch, aspp_ch=bn_ch // 2,
                         dilations=(2, 4, 6), dropout=drop)

        # ── Decoder ──────────────────────────────────────────────────────
        self.up1 = Up(f * 16, f * 8  // factor, bilin, dropout=drop)
        self.up2 = Up(f * 8,  f * 4  // factor, bilin)
        self.up3 = Up(f * 4,  f * 2  // factor, bilin)
        self.up4 = Up(f * 2,  f,                bilin)

        # ── Deep supervision from up2 output (1/4 scale) ─────────────────
        self.aux_head = AuxHead(f * 4 // factor, Config_abd.NUM_CHANNELS)

        # ── Main classification head ──────────────────────────────────────
        # Three conv layers: two 3×3 refiners + final 1×1 projection.
        self.head = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(f // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f // 2, Config_abd.NUM_CHANNELS, 1),
            # Raw logits — sigmoid applied in loss / inference
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  normalised grayscale

        Returns
        -------
        Training   : (main_logits, aux_logits)  both (B, NUM_CHANNELS, H, W)
        Inference  : main_logits                     (B, NUM_CHANNELS, H, W)
        """
        H, W = x.shape[2], x.shape[3]

        # Encoder
        x1 = self.inc(x)          # (B,  f, H,    W   )
        x2 = self.down1(x1)       # (B, 2f, H/2,  W/2 )
        x3 = self.down2(x2)       # (B, 4f, H/4,  W/4 )
        x4 = self.down3(x3)       # (B, 8f, H/8,  W/8 )
        x5 = self.down4(x4)       # (B, 8f, H/16, W/16)  (bilinear factor=2)

        # SE on skip connections
        x1 = self.se1(x1)
        x2 = self.se2(x2)
        x3 = self.se3(x3)
        x4 = self.se4(x4)

        # Bottleneck
        x5 = self.aspp(x5)

        # Decoder
        d1 = self.up1(x5, x4)    # (B, 4f, H/8,  W/8 )
        d2 = self.up2(d1, x3)    # (B, 2f, H/4,  W/4 )  ← aux branch here
        d3 = self.up3(d2, x2)    # (B,  f, H/2,  W/2 )
        d4 = self.up4(d3, x1)    # (B,  f, H,    W   )

        main = self.head(d4)     # (B, NUM_CHANNELS, H, W)

        if self.training:
            aux = self.aux_head(d2, (H, W))   # (B, NUM_CHANNELS, H, W)
            return main, aux

        return main


def build_model() -> UNet:
    return UNet()


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 1, 512, 512)

    model.train()
    main, aux = model(dummy)
    print(f"Train  — main: {main.shape}  aux: {aux.shape}")

    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"Eval   — out:  {out.shape}")

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n:,}")