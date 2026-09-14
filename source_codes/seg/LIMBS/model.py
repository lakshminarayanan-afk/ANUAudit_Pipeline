r"""
model.py
--------
Everything below (SwinBlock, SwinEncoder, FPN, UNetDecoder, AgreementHead, MultiTaskSwinUNet,
etc.) is your existing, active code. The segmentation path (encoder -> FPN -> decoder -> seg_head)
is byte-for-byte unchanged. Search "H/4 CHANGE" to find every touch point for this pass.

H/4 CHANGE (this pass)
-------------------------------------------------------------------------------------------------
Goal: move the CCDC auxiliary branch off the full-resolution decoder output (dec_out, which was
OOM'ing) onto the H/4, W/4 feature that UNetDecoder already computes internally as an intermediate
step (the output of block2, right before it gets upsampled to native resolution). This is not a
new tensor - it was already being computed and then discarded every forward pass; we just keep a
reference to it.

Concretely:
    - UNetDecoder.forward now returns (dec_out, feat_h4) instead of just dec_out.
        dec_out  : (B, decoder_ch[3]=32, H,  W ) - UNCHANGED, still feeds seg_head exactly as before.
        feat_h4  : (B, decoder_ch[2]=64, H/4, W/4) - the block2 output, before final upsample/final_up.
    - CCDCProjectionHead's forward() now takes a full (B, C, H/4, W/4) map (1x1 convs, so this is
      the same per-pixel computation whether called on a full map or a gathered pixel list -
      but it now only ever touches ~1/16th the spatial elements of the old dense call).
    - MultiTaskSwinUNet.forward projects+normalizes ccdc_embedding right after computing it
      (cheap now, since it's H/4 x W/4), instead of deferring the projection to loss.py.
    - outputs["ccdc_embedding"] replaces outputs["decoder_features"] as the CCDC-facing key.
      shape: (B, CCDC_PROJ_DIM, H/4, W/4), already L2-normalized, still attached to the autograd
      graph.

The segmentation branch (encoder, FPN, decoder's native-res path, seg_head, AgreementHead,
proj_head/SupCon pooled embedding) is completely untouched by this change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ----------------------------------------------------------------------------
# Basic building blocks (UNCHANGED - verbatim from your existing model.py)
# ----------------------------------------------------------------------------

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        rpb = rpb.permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        window_size = self.window_size
        shift_size = self.shift_size
        if min(Hp, Wp) <= window_size:
            shift_size = 0
            window_size = min(Hp, Wp)

        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            attn_mask = self._compute_mask(Hp, Wp, window_size, shift_size, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, Hp, Wp)

        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    @staticmethod
    def _compute_mask(H, W, window_size, shift_size, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        if pad_r or pad_b:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        x = self.proj(x)
        _, _, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, Hp, Wp


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, downsample=True, drop_path=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size, shift_size=0 if i % 2 == 0 else window_size // 2,
                      drop_path=drop_path)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        feat_before_down = x
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, feat_before_down


class SwinEncoder(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size=4)
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=dims[i], depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                downsample=(i < self.num_layers - 1),
            ))
        self.out_channels = dims

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        feats = []
        for layer in self.layers:
            x, H, W, feat_before_down = layer(x, H, W)
            n = feat_before_down.shape[1]
            h = w = int(math.sqrt(n))
            feats.append(feat_before_down)
        return feats, x


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, feats):
        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        outs = [o_conv(l) for o_conv, l in zip(self.output_convs, laterals)]
        return outs


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, num_groups=8):
        super().__init__()
        g = min(num_groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    H/4 CHANGE: forward() now returns a tuple (dec_out, feat_h4) instead of just dec_out.

        dec_out : (B, decoder_ch[3], H, W)     - UNCHANGED. Still the native-resolution tensor
                                                  that seg_head consumes, computed exactly as
                                                  before (block2 -> interpolate to input_hw ->
                                                  final_up).
        feat_h4 : (B, decoder_ch[2], H/4, W/4) - NEW return value. This is simply block2's output
                                                  BEFORE the interpolate-to-native-res step - a
                                                  tensor that was already being computed on every
                                                  forward pass and then discarded. No new
                                                  computation is introduced to obtain it.

    block2's output is at H/4, W/4 because DecoderBlock upsamples its input to match its skip
    connection's spatial size, and block2's skip is p1 (the highest-resolution FPN level, which is
    H/4, W/4 relative to the native input - patch_embed already downsamples by 4x before any FPN
    level is formed).
    """

    def __init__(self, pyramid_ch=256, decoder_ch=(256, 128, 64, 32), num_groups=8):
        super().__init__()
        self.block4 = DecoderBlock(pyramid_ch, pyramid_ch, decoder_ch[0], num_groups)
        self.block3 = DecoderBlock(decoder_ch[0], pyramid_ch, decoder_ch[1], num_groups)
        self.block2 = DecoderBlock(decoder_ch[1], pyramid_ch, decoder_ch[2], num_groups)
        g_final = min(num_groups, decoder_ch[3])
        while decoder_ch[3] % g_final != 0:
            g_final -= 1
        self.final_up = nn.Sequential(
            nn.Conv2d(decoder_ch[2], decoder_ch[3], 3, padding=1),
            nn.GroupNorm(g_final, decoder_ch[3]),
            nn.ReLU(inplace=True),
        )
        self.out_ch = decoder_ch[3]
        # H/4 CHANGE: channel count of block2's output - this is what CCDCProjectionHead expects.
        self.h4_ch = decoder_ch[2]

    def forward(self, pyramid, input_hw):
        p1, p2, p3, p4 = pyramid
        x = self.block4(p4, p3)
        x = self.block3(x, p2)
        x = self.block2(x, p1)
        # H/4 CHANGE: capture block2's output here, before it is touched further. This is exactly
        # H/4, W/4 relative to the native input.
        feat_h4 = x
        x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)  # UNCHANGED
        x = self.final_up(x)                                                       # UNCHANGED
        return x, feat_h4   # H/4 CHANGE: tuple return; native-res value (x) is identical to before


class AgreementHead(nn.Module):
    def __init__(self, encoder_ch, decoder_ch, num_bone_classes):
        super().__init__()
        joint_dim = encoder_ch + decoder_ch
        hidden = joint_dim // 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_bone_classes),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, encoder_pooled, decoder_pooled):
        joint = torch.cat([encoder_pooled, decoder_pooled], dim=1)
        logits = self.mlp(joint)
        return logits, self.gate


# ----------------------------------------------------------------------------
# ============================== CCDC (H/4) ====================================
# Lightweight pixel-level projection head for CCDC-style contrastive learning.
#
# H/4 CHANGE: this head is now called ONCE per forward pass on the full (B, C, H/4, W/4) feature
# map (16x fewer spatial elements than the old native-resolution call that OOM'd), rather than on
# a small set of pre-gathered pixel rows. Because conv1/conv2 are 1x1 convolutions, this is the
# identical per-pixel linear computation either way - only the calling convention changed back to
# a dense spatial map, which is now cheap because the map itself is small.
# ----------------------------------------------------------------------------

class CCDCProjectionHead(nn.Module):
    def __init__(self, decoder_ch, ccdc_proj_dim=64, num_groups=8):
        super().__init__()
        g = min(num_groups, decoder_ch)
        while decoder_ch % g != 0:
            g -= 1
        self.conv1 = nn.Conv2d(decoder_ch, decoder_ch, kernel_size=1)
        self.norm = nn.GroupNorm(g, decoder_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(decoder_ch, ccdc_proj_dim, kernel_size=1)

    def forward(self, feat_h4):
        """
        feat_h4: (B, decoder_ch, H/4, W/4) - the H/4-resolution decoder feature (block2's output).
        Returns: (B, ccdc_proj_dim, H/4, W/4).

        NEVER call this on the full native-resolution (B, decoder_ch, H, W) map - that dense call
        (with GroupNorm over ~2M spatial positions at 1080x1920) is what originally OOM'd.
        """
        x = self.conv1(feat_h4)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

# ============================ END CCDC (H/4) ==================================
# ----------------------------------------------------------------------------


class MultiTaskSwinUNet(nn.Module):
    """
    Segmentation head -> per-pixel logits over `num_seg_classes`, with a shared per-class bias
        injected from the agreement bottleneck before argmax (gated - see AgreementHead).
        UNCHANGED: still native resolution, still fed by dec_out exactly as before.
    Agreement head -> "which bone" logits over `num_bone_classes`. UNCHANGED.
    Projection head (`proj_head`) -> pooled, L2-normalized image-level embedding for the existing
        SupCon loss. UNCHANGED.
    CCDC projection head (`ccdc_proj_head`) -> H/4 CHANGE: now applied to the H/4, W/4 decoder
        feature (`feat_h4`) instead of the native-resolution one. Output is projected and
        L2-normalized here in forward(), producing outputs["ccdc_embedding"] with shape
        (B, ccdc_proj_dim, H/4, W/4). This replaces the old outputs["decoder_features"] key.
    """

    def __init__(self, in_chans=1, num_seg_classes=5, num_bone_classes=4,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, fpn_channels=256, proj_dim=128, ccdc_proj_dim=64):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_bone_classes = num_bone_classes

        self.encoder = SwinEncoder(in_chans, embed_dim, depths, num_heads, window_size)
        enc_ch = self.encoder.out_channels

        self.fpn = FPN(enc_ch, out_channels=fpn_channels)
        self.decoder = UNetDecoder(pyramid_ch=fpn_channels)

        self.seg_head = nn.Conv2d(self.decoder.out_ch, num_seg_classes, kernel_size=1)

        deepest_ch = enc_ch[-1]
        decoder_ch = self.decoder.out_ch

        self.agreement_head = AgreementHead(
            encoder_ch=deepest_ch, decoder_ch=decoder_ch, num_bone_classes=num_bone_classes
        )

        joint_dim = deepest_ch + decoder_ch
        self.proj_head = nn.Sequential(
            nn.Linear(joint_dim, deepest_ch),
            nn.GELU(),
            nn.Linear(deepest_ch, proj_dim),
        )

        # H/4 CHANGE: input channels = self.decoder.h4_ch (block2's output channels, e.g. 64),
        # NOT self.decoder.out_ch (32, the native-res dec_out channels used before). Output dim
        # (ccdc_proj_dim) is unchanged - still driven by Config.CCDC_PROJ_DIM.
        self.ccdc_proj_head = CCDCProjectionHead(self.decoder.h4_ch, ccdc_proj_dim=ccdc_proj_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        feats_flat = []
        hw_list = []
        cur, h, w = self.encoder.patch_embed(x)
        for layer in self.encoder.layers:
            stage_h, stage_w = h, w
            cur, h, w, feat_before_down = layer(cur, h, w)
            feats_flat.append(feat_before_down)
            hw_list.append((stage_h, stage_w))

        feats = []
        for f, (fh, fw) in zip(feats_flat, hw_list):
            Bc, N, Cc = f.shape
            feats.append(f.transpose(1, 2).reshape(Bc, Cc, fh, fw))

        deepest = cur
        encoder_pooled = deepest.mean(dim=1)

        pyramid = self.fpn(feats)
        # H/4 CHANGE: decoder now returns (dec_out, feat_h4). dec_out is identical to what it was
        # before; feat_h4 is the new H/4, W/4 tensor used only by the CCDC branch below.
        dec_out, ccdc_feat_h4 = self.decoder(pyramid, input_hw=(H, W))

        decoder_pooled = dec_out.mean(dim=(2, 3))  # UNCHANGED

        joint_pooled = torch.cat([encoder_pooled, decoder_pooled], dim=1)

        proj = self.proj_head(joint_pooled)
        proj = F.normalize(proj, dim=-1)

        bone_logits, gate = self.agreement_head(encoder_pooled, decoder_pooled)

        seg_logits = self.seg_head(dec_out)  # UNCHANGED: native resolution, same dec_out as before

        bg_logits = seg_logits[:, :1, :, :]
        fg_logits = seg_logits[:, 1:, :, :] + gate * bone_logits.unsqueeze(-1).unsqueeze(-1)
        seg_logits = torch.cat([bg_logits, fg_logits], dim=1)

        # H/4 CHANGE: CCDC projection now runs here, on the small H/4 x W/4 feature map (cheap),
        # instead of being deferred to loss.py and run on gathered native-resolution pixel rows.
        ccdc_embedding = self.ccdc_proj_head(ccdc_feat_h4)      # (B, ccdc_proj_dim, H/4, W/4)
        ccdc_embedding = F.normalize(ccdc_embedding, dim=1)      # per-pixel L2 norm over channels

        return {
            "seg_logits": seg_logits,           # UNCHANGED: (B, num_seg_classes, H, W)
            "bone_logits": bone_logits,         # UNCHANGED
            "embedding": proj,                  # UNCHANGED: pooled SupCon embedding
            "ccdc_embedding": ccdc_embedding,   # H/4 CHANGE: (B, ccdc_proj_dim, H/4, W/4),
                                                 # already projected + L2-normalized. Replaces the
                                                 # old "decoder_features" key.
            "pooled_features": encoder_pooled,  # UNCHANGED
            "agreement_gate": gate.detach(),    # UNCHANGED
        }


if __name__ == "__main__":
    model = MultiTaskSwinUNet(in_chans=1, num_seg_classes=5, num_bone_classes=4, ccdc_proj_dim=32)
    x = torch.randn(2, 1, 517, 663)
    out = model(x)
    print("seg_logits:", out["seg_logits"].shape)
    print("bone_logits:", out["bone_logits"].shape)
    print("embedding:", out["embedding"].shape)
    print("ccdc_embedding:", out["ccdc_embedding"].shape)
    print("agreement_gate (should start at 0.0):", out["agreement_gate"].item())