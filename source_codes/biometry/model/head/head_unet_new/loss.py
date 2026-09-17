# # loss.py  —  Combined CrossEntropy + Tversky for softmax multi-class segmentation.
# #
# # Targets throughout: (B, H, W) int64 class-index masks.
# # Class 0 = background.  Classes 1–24 = foreground structures.
# #
# # Class weights are computed from pixel frequencies on the training set.
# # Background (class 0) always gets weight = 1.0 so it acts as an anchor.
# # Foreground weights are scaled by inverse-frequency so rare structures
# # get higher loss contribution.

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple
# from config.config import Config


# # ─────────────────────────────────────────────────────────────────────────────
# #  Tversky Loss  (softmax version, class-index targets)
# # ─────────────────────────────────────────────────────────────────────────────

# class TverskyLoss(nn.Module):
#     """
#     Tversky loss for multi-class softmax segmentation.

#     alpha = FP weight  (penalises false positives)
#     beta  = FN weight  (penalises false negatives / missed detections)

#     Setting alpha < beta (e.g. 0.3 / 0.7) biases the model towards recall,
#     which is usually desirable in medical segmentation.
#     """

#     def __init__(self, smooth: float = 1e-6):
#         super().__init__()
#         self.alpha  = Config.TVERSKY_ALPHA
#         self.beta   = Config.TVERSKY_BETA
#         self.smooth = smooth

#     def forward(
#         self,
#         logits:        torch.Tensor,                    # (B, C, H, W)
#         targets:       torch.Tensor,                    # (B, H, W) int64
#         class_weights: Optional[torch.Tensor] = None,  # (C,) on same device
#     ) -> torch.Tensor:

#         probs = torch.softmax(logits, dim=1)    # (B, C, H, W)
#         B, C, H, W = probs.shape

#         # One-hot encode targets  →  (B, C, H, W)
#         targets_oh = (
#             F.one_hot(targets.long(), num_classes=C)  # (B, H, W, C)
#             .permute(0, 3, 1, 2)                       # (B, C, H, W)
#             .float()
#         )

#         p = probs.view(B, C, -1)        # (B, C, N)
#         t = targets_oh.view(B, C, -1)   # (B, C, N)

#         TP = (p * t).sum(2)                    # (B, C)
#         FP = (p * (1.0 - t)).sum(2)            # (B, C)
#         FN = ((1.0 - p) * t).sum(2)            # (B, C)

#         tversky_per_class = 1.0 - (TP + self.smooth) / (
#             TP + self.alpha * FP + self.beta * FN + self.smooth
#         )                                       # (B, C)

#         # Average over batch → (C,)
#         tversky_per_class = tversky_per_class.mean(0)

#         if class_weights is not None:
#             tversky_per_class = tversky_per_class * class_weights

#         return tversky_per_class.mean()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Combined loss
# # ─────────────────────────────────────────────────────────────────────────────

# class CombinedSegLoss(nn.Module):
#     """
#     loss = ce_weight * CrossEntropyLoss  +  tv_weight * TverskyLoss

#     Both terms accept (B, H, W) int64 targets and (B, C, H, W) logits.
#     class_weights (C,) is passed directly to F.cross_entropy — no in-place
#     mutation of module attributes, so this is DDP-safe.
#     """

#     def __init__(self):
#         super().__init__()
#         self.tversky         = TverskyLoss(smooth=Config.LOSS_SMOOTH)
#         self.ce_weight       = Config.CE_WEIGHT
#         self.tv_weight       = Config.TV_WEIGHT
#         self.label_smoothing = 0.05   # helps generalisation slightly

#     def forward(
#         self,
#         logits:        torch.Tensor,
#         targets:       torch.Tensor,
#         class_weights: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:

#         targets = targets.long()

#         ce_loss = F.cross_entropy(
#             logits,
#             targets,
#             weight          = class_weights,
#             label_smoothing = self.label_smoothing,
#         )

#         tv_loss = self.tversky(logits, targets, class_weights)

#         return self.ce_weight * ce_loss + self.tv_weight * tv_loss


# # ─────────────────────────────────────────────────────────────────────────────
# #  Weight strategy
# # ─────────────────────────────────────────────────────────────────────────────

# def _apply_weight_strategy(
#     freq: torch.Tensor,
#     strategy: str,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     """
#     Given per-class pixel frequencies (values in [0, 1]), return raw weights.

#     "inv_smooth" :  weight = 1 / (freq^alpha + eps)
#         alpha=0.7 gives ~10× spread for typical fetal US class distributions.
#         Large, common structures  (freq ~0.3–0.5) → weight ~2–3
#         Medium structures         (freq ~0.05)    → weight ~7
#         Small/rare structures     (freq ~0.005)   → weight ~25  (capped later)
#     """
#     if strategy == "inv_smooth":
#         alpha = Config.CLASS_WEIGHT_ALPHA
#         return 1.0 / (freq ** alpha + eps)

#     elif strategy == "sqrt_inv":
#         return 1.0 / (torch.sqrt(freq + eps))

#     elif strategy == "mid_peak":
#         f     = freq / (freq.max() + eps)
#         peak  = 0.15
#         sigma = 0.10
#         return torch.exp(-((f - peak) ** 2) / (2 * sigma ** 2)) + eps

#     else:
#         raise ValueError(f"Unknown CLASS_WEIGHT_STRATEGY: {strategy!r}")


# # ─────────────────────────────────────────────────────────────────────────────
# #  Class-weight computation
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_class_weights(
#     dataloader,
#     num_classes: int,
#     device:      torch.device,
#     max_batches: int = 100,
# ) -> torch.Tensor:
#     """
#     Scans up to max_batches of training data to estimate per-class pixel
#     frequencies, then returns class weights (C,) on `device`.

#     Background (class 0) is always given weight = 1.0.
#     Foreground classes are weighted by inverse frequency, clipped to
#     [CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX] and normalised so mean ≈ 1.
#     Hard/thin structures are additionally boosted by a fixed multiplier.

#     Expects batch["mask"] to be (B, H, W) int64 class-index masks.
#     """
#     pixel_count = torch.zeros(num_classes, dtype=torch.float64)
#     tot_pixels  = 0

#     print("  Scanning training data for class frequencies …")
#     for i, batch in enumerate(dataloader):
#         m = batch["mask"]
#         if m.ndim == 4:
#             m = m.argmax(dim=1)   # safety: handle accidental one-hot input
#         m = m.long()

#         B, H, W     = m.shape
#         tot_pixels += B * H * W

#         counts      = torch.bincount(m.view(-1), minlength=num_classes).float()
#         pixel_count += counts.double()

#         if i >= max_batches - 1:
#             break

#     freq = (pixel_count / (tot_pixels + 1e-8)).float()   # (C,)

#     # ── Step 1: inverse frequency weights for foreground ─────────────────
#     fg_freq = freq[1:]   # (C-1,)
#     raw_w   = _apply_weight_strategy(fg_freq, Config.CLASS_WEIGHT_STRATEGY)

#     # Normalise so mean ≈ 1
#     raw_w = raw_w / (raw_w.mean() + 1e-8)

#     # Clip to avoid instability
#     fg_w = torch.clamp(raw_w, Config.CLASS_WEIGHT_MIN, Config.CLASS_WEIGHT_MAX)

#     # Re-normalise after clipping
#     fg_w = fg_w / (fg_w.mean() + 1e-8)

#     # ── Step 2: assemble full weight vector (background = 1.0) ───────────
#     class_w      = torch.ones(num_classes, dtype=torch.float32)
#     class_w[1:]  = fg_w

#     # ── Step 3: manually boost thin/hard structures ───────────────────────
#     boost_structures = {
#         "inner calvarium":                      3.0
#     }
#     for name, boost in boost_structures.items():
#         if name in Config.STRUCTURE_TO_IDX:
#             idx = Config.STRUCTURE_TO_IDX[name]   # 1-based
#             class_w[idx] = class_w[idx] * boost

#     # Re-normalise foreground after boost so mean stays ≈ 1
#     class_w[1:] = class_w[1:] / (class_w[1:].mean() + 1e-8)

#     _print_weight_table(num_classes, freq, class_w)

#     return class_w.to(device)


# def _print_weight_table(
#     num_classes: int,
#     freq:        torch.Tensor,
#     class_w:     torch.Tensor,
# ):
#     fg_w = class_w[1:]
#     print(f"\n  {'Idx':>3}  {'Structure':47}  {'freq%':>6}  {'cls_w':>7}")
#     print("  " + "─" * 70)

#     # Sort foreground by frequency (ascending = rarest first)
#     order = freq[1:].argsort().tolist()
#     for i in order:
#         name   = Config.STRUCTURES[i] if i < len(Config.STRUCTURES) else f"class_{i+1}"
#         bar    = "█" * max(1, int(class_w[i + 1].item() * 5))
#         print(
#             f"  [{i+1:2d}]  {name:47}  "
#             f"{freq[i+1]*100:5.2f}%  "
#             f"{class_w[i+1]:7.3f}  {bar}"
#         )

#     print(f"  [ 0]  {'background':47}  {freq[0]*100:5.2f}%  {class_w[0]:7.3f}")
#     print(
#         f"\n  Foreground class_w range: "
#         f"[{fg_w.min():.3f}, {fg_w.max():.3f}]  "
#         f"spread = {fg_w.max()/fg_w.min():.1f}×"
#     )


# def build_loss() -> CombinedSegLoss:
#     return CombinedSegLoss()




# loss.py  —  Combined CrossEntropy + Tversky for softmax multi-class segmentation.
#
# Targets throughout: (B, H, W) int64 class-index masks.
# Class 0 = background.  Classes 1–22 = foreground structures.
#
# Class weights are computed from pixel frequencies on the training set.
# Background (class 0) always gets weight = 1.0 so it acts as an anchor.
# Foreground weights are scaled by inverse-frequency so rare structures
# get higher loss contribution.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config.config import Config


# ─────────────────────────────────────────────────────────────────────────────
#  Tversky Loss  (softmax version, class-index targets)
# ─────────────────────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky loss for multi-class softmax segmentation.

    alpha = FP weight  (penalises false positives)
    beta  = FN weight  (penalises false negatives / missed detections)

    Setting alpha < beta (e.g. 0.3 / 0.7) biases the model towards recall,
    which is usually desirable in medical segmentation.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.alpha  = Config.TVERSKY_ALPHA
        self.beta   = Config.TVERSKY_BETA
        self.smooth = smooth

    def forward(
        self,
        logits:        torch.Tensor,                    # (B, C, H, W)
        targets:       torch.Tensor,                    # (B, H, W) int64
        class_weights: Optional[torch.Tensor] = None,  # (C,) on same device
    ) -> torch.Tensor:

        probs = torch.softmax(logits, dim=1)    # (B, C, H, W)
        B, C, H, W = probs.shape

        # One-hot encode targets  →  (B, C, H, W)
        targets_oh = (
            F.one_hot(targets.long(), num_classes=C)  # (B, H, W, C)
            .permute(0, 3, 1, 2)                       # (B, C, H, W)
            .float()
        )

        p = probs.view(B, C, -1)        # (B, C, N)
        t = targets_oh.view(B, C, -1)   # (B, C, N)

        TP = (p * t).sum(2)                    # (B, C)
        FP = (p * (1.0 - t)).sum(2)            # (B, C)
        FN = ((1.0 - p) * t).sum(2)            # (B, C)

        tversky_per_class = 1.0 - (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )                                       # (B, C)

        # Average over batch → (C,)
        tversky_per_class = tversky_per_class.mean(0)

        if class_weights is not None:
            tversky_per_class = tversky_per_class * class_weights

        return tversky_per_class.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class CombinedSegLoss(nn.Module):
    """
    loss = ce_weight * CrossEntropyLoss  +  tv_weight * TverskyLoss

    Both terms accept (B, H, W) int64 targets and (B, C, H, W) logits.
    class_weights (C,) is passed directly to F.cross_entropy — no in-place
    mutation of module attributes, so this is DDP-safe.
    """

    def __init__(self):
        super().__init__()
        self.tversky         = TverskyLoss(smooth=Config.LOSS_SMOOTH)
        self.ce_weight       = Config.CE_WEIGHT
        self.tv_weight       = Config.TV_WEIGHT
        self.label_smoothing = 0.05

    def forward(
        self,
        logits:        torch.Tensor,
        targets:       torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        targets = targets.long()

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight          = class_weights,
            label_smoothing = self.label_smoothing,
        )

        tv_loss = self.tversky(logits, targets, class_weights)

        return self.ce_weight * ce_loss + self.tv_weight * tv_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Weight strategy
# ─────────────────────────────────────────────────────────────────────────────

def _apply_weight_strategy(
    freq:     torch.Tensor,
    strategy: str,
    eps:      float = 1e-8,
) -> torch.Tensor:
    """
    Given per-class pixel frequencies (values in [0, 1]), return raw weights.

    "inv_smooth" :  weight = 1 / (freq^alpha + eps)
    """
    if strategy == "inv_smooth":
        alpha = Config.CLASS_WEIGHT_ALPHA
        return 1.0 / (freq ** alpha + eps)

    elif strategy == "sqrt_inv":
        return 1.0 / (torch.sqrt(freq + eps))

    elif strategy == "mid_peak":
        f     = freq / (freq.max() + eps)
        peak  = 0.15
        sigma = 0.10
        return torch.exp(-((f - peak) ** 2) / (2 * sigma ** 2)) + eps

    else:
        raise ValueError(f"Unknown CLASS_WEIGHT_STRATEGY: {strategy!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  Class-weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    dataloader,
    num_classes: int,
    device:      torch.device,
    max_batches: int = 100,
    mask_key:    str = "mask",          # ← NEW: which batch key holds the mask
) -> torch.Tensor:
    """
    Scans up to max_batches of training data to estimate per-class pixel
    frequencies, then returns class weights (C,) on `device`.

    Parameters
    ----------
    dataloader  : training DataLoader
    num_classes : total number of classes including background (= Config.NUM_CLASSES)
    device      : target device for the returned tensor
    max_batches : how many batches to scan (default 100)
    mask_key    : key in the batch dict that contains the (B, H, W) int64
                  class-index mask.  Use "mask" for single-head datasets and
                  "mask_inner" for dual-head datasets.

    Background (class 0) is always given weight = 1.0.
    Foreground classes are weighted by inverse frequency, clipped to
    [CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX] and normalised so mean ≈ 1.
    """
    pixel_count = torch.zeros(num_classes, dtype=torch.float64)
    tot_pixels  = 0

    print(f"  Scanning training data for class frequencies (mask_key='{mask_key}') …")

    for i, batch in enumerate(dataloader):
        m = batch[mask_key]

        # Safety: handle accidental one-hot input
        if m.ndim == 4:
            m = m.argmax(dim=1)

        m = m.long()

        B, H, W     = m.shape
        tot_pixels += B * H * W

        counts      = torch.bincount(m.view(-1), minlength=num_classes).float()
        pixel_count += counts.double()

        if i >= max_batches - 1:
            break

    freq = (pixel_count / (tot_pixels + 1e-8)).float()   # (C,)

    # ── Step 1: inverse frequency weights for foreground ─────────────────
    fg_freq = freq[1:]   # (C-1,)
    raw_w   = _apply_weight_strategy(fg_freq, Config.CLASS_WEIGHT_STRATEGY)

    # Normalise so mean ≈ 1
    raw_w = raw_w / (raw_w.mean() + 1e-8)

    # Clip to avoid instability
    fg_w = torch.clamp(raw_w, Config.CLASS_WEIGHT_MIN, Config.CLASS_WEIGHT_MAX)

    # Re-normalise after clipping
    fg_w = fg_w / (fg_w.mean() + 1e-8)

    # ── Step 2: assemble full weight vector (background = 1.0) ───────────
    class_w     = torch.ones(num_classes, dtype=torch.float32)
    class_w[1:] = fg_w

    # ── Step 3: manually boost thin/hard structures ───────────────────────
    # Note: calvarium is now a separate sigmoid head and no longer in
    # STRUCTURE_TO_IDX, so we only boost inner structures here.
    boost_structures = {
        "csp (cavum septum pellucidum)":         2.0,
        "arrow sign":                            2.0,
        "pillars of fornix":                     2.0,
        "cisternae magna":                       1.5,
        "cerebellar vermis":                     1.5,
        "anterior horn of lateral ventricle 1":  1.5,
        "anterior horn of lateral ventricles 2": 1.5,
    }
    for name, boost in boost_structures.items():
        if name in Config.STRUCTURE_TO_IDX:
            idx           = Config.STRUCTURE_TO_IDX[name]   # 1-based
            class_w[idx]  = class_w[idx] * boost

    # Re-normalise foreground after boost so mean stays ≈ 1
    class_w[1:] = class_w[1:] / (class_w[1:].mean() + 1e-8)

    _print_weight_table(num_classes, freq, class_w)

    return class_w.to(device)


def _print_weight_table(
    num_classes: int,
    freq:        torch.Tensor,
    class_w:     torch.Tensor,
):
    fg_w = class_w[1:]
    print(f"\n  {'Idx':>3}  {'Structure':47}  {'freq%':>6}  {'cls_w':>7}")
    print("  " + "─" * 70)

    # Sort foreground by frequency (ascending = rarest first)
    order = freq[1:].argsort().tolist()
    for i in order:
        name = Config.STRUCTURES[i] if i < len(Config.STRUCTURES) else f"class_{i+1}"
        bar  = "█" * max(1, int(class_w[i + 1].item() * 5))
        print(
            f"  [{i+1:2d}]  {name:47}  "
            f"{freq[i+1]*100:5.2f}%  "
            f"{class_w[i+1]:7.3f}  {bar}"
        )

    print(f"  [ 0]  {'background':47}  {freq[0]*100:5.2f}%  {class_w[0]:7.3f}")
    print(
        f"\n  Foreground class_w range: "
        f"[{fg_w.min():.3f}, {fg_w.max():.3f}]  "
        f"spread = {fg_w.max()/fg_w.min():.1f}×" 
    )


def build_loss() -> CombinedSegLoss:
    return CombinedSegLoss()