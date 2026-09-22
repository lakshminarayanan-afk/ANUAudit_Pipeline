# loss_multiplane.py — Per-plane CE + Tversky loss for the 3-head spine model.
#
# The actual loss math (TverskyLoss, CombinedSegLoss, the inv_smooth weight
# strategy) is REUSED UNCHANGED from loss.py — copied verbatim below rather
# than importing, since loss.py lives in the fetal-head project's own
# model/ package and this project intentionally stays separate from it
# (per the "keep fetal-brain files completely separate" instruction).
#
# What's new here is only the multi-plane plumbing:
#   - PlaneConfigView: adapts config_multiplane.Config (which keys
#     STRUCTURES / STRUCTURE_TO_IDX by plane) into the flat per-plane
#     shape loss.py's functions already expect via their `cfg` parameter.
#   - compute_class_weights_multiplane: loss.py's compute_class_weights
#     assumes every batch in the dataloader belongs to one dataset/plane.
#     A multiplane batch mixes all 3 planes, so this does ONE pass over
#     the loader and buckets pixel counts by each sample's `plane` tag,
#     then applies loss.py's existing weight-strategy math separately
#     per plane.
#   - build_losses_per_plane: one CombinedSegLoss instance per plane.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from config.config import Config as MultiPlaneConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Tversky Loss  (softmax version, class-index targets)
#  — UNCHANGED from loss.py —
# ─────────────────────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky loss for multi-class softmax segmentation.

    alpha = FP weight  (penalises false positives)
    beta  = FN weight  (penalises false negatives / missed detections)

    Setting alpha < beta (e.g. 0.3 / 0.7) biases the model towards recall,
    which is usually desirable in medical segmentation.
    """

    def __init__(self, smooth: float = 1e-6, cfg=None):
        super().__init__()
        cfg = cfg or MultiPlaneConfig
        self.alpha  = cfg.TVERSKY_ALPHA
        self.beta   = cfg.TVERSKY_BETA
        self.smooth = smooth

    def forward(
        self,
        logits:        torch.Tensor,                    # (B, C, H, W)
        targets:       torch.Tensor,                    # (B, H, W) int64
        class_weights: Optional[torch.Tensor] = None,  # (C,) on same device
    ) -> torch.Tensor:

        probs = torch.softmax(logits, dim=1)    # (B, C, H, W)
        B, C, H, W = probs.shape

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

        tversky_per_class = tversky_per_class.mean(0)

        if class_weights is not None:
            tversky_per_class = tversky_per_class * class_weights

        return tversky_per_class.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Combined loss  — UNCHANGED from loss.py —
# ─────────────────────────────────────────────────────────────────────────────

class CombinedSegLoss(nn.Module):
    """
    loss = ce_weight * CrossEntropyLoss  +  tv_weight * TverskyLoss
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or MultiPlaneConfig
        self.tversky         = TverskyLoss(smooth=cfg.LOSS_SMOOTH, cfg=cfg)
        self.ce_weight       = cfg.CE_WEIGHT
        self.tv_weight       = cfg.TV_WEIGHT
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
#  Weight strategy  — UNCHANGED from loss.py —
# ─────────────────────────────────────────────────────────────────────────────

def _apply_weight_strategy(
    freq:     torch.Tensor,
    strategy: str,
    eps:      float = 1e-8,
    cfg=None,
) -> torch.Tensor:
    cfg = cfg or MultiPlaneConfig
    if strategy == "inv_smooth":
        alpha = cfg.CLASS_WEIGHT_ALPHA
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


def _print_weight_table(
    num_classes:  int,
    freq:         torch.Tensor,
    class_w:      torch.Tensor,
    structures:   list,
    plane_label:  str = "",
):
    fg_w = class_w[1:]
    print(f"\n  ── {plane_label} class weights ──")
    print(f"  {'Idx':>3}  {'Structure':47}  {'freq%':>6}  {'cls_w':>7}")
    print("  " + "─" * 70)

    order = freq[1:].argsort().tolist()
    for i in order:
        name = structures[i] if i < len(structures) else f"class_{i+1}"
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


# ─────────────────────────────────────────────────────────────────────────────
#  PlaneConfigView — adapts config_multiplane.Config (plane-keyed dicts)
#  into the flat per-plane shape that loss.py's functions expect via `cfg`.
#  No loss math changes here — purely a view/adapter.
# ─────────────────────────────────────────────────────────────────────────────

class PlaneConfigView:
    """A read-only, flat view of MultiPlaneConfig for a single plane."""

    def __init__(self, plane: str, base_cfg=MultiPlaneConfig):
        self.plane            = plane
        self.STRUCTURES       = base_cfg.STRUCTURES[plane]
        self.STRUCTURE_TO_IDX = base_cfg.STRUCTURE_TO_IDX[plane]
        self.NUM_CLASSES      = base_cfg.NUM_CLASSES_PER_PLANE[plane]

        # Shared hyperparameters — identical across planes, passed through.
        self.CE_WEIGHT             = base_cfg.CE_WEIGHT
        self.TV_WEIGHT             = base_cfg.TV_WEIGHT
        self.LOSS_SMOOTH           = base_cfg.LOSS_SMOOTH
        self.TVERSKY_ALPHA         = base_cfg.TVERSKY_ALPHA
        self.TVERSKY_BETA          = base_cfg.TVERSKY_BETA
        self.CLASS_WEIGHT_STRATEGY = base_cfg.CLASS_WEIGHT_STRATEGY
        self.CLASS_WEIGHT_ALPHA    = base_cfg.CLASS_WEIGHT_ALPHA
        self.CLASS_WEIGHT_MIN      = base_cfg.CLASS_WEIGHT_MIN
        self.CLASS_WEIGHT_MAX      = base_cfg.CLASS_WEIGHT_MAX


def build_losses_per_plane(base_cfg=MultiPlaneConfig, region_pos_weight=None) -> Dict[str, nn.Module]:
    """One loss module per plane. Sagittal gets the combined
    exclusive-CE/Tversky + sigmoid-region loss; others unchanged."""
    losses = {}
    for plane in base_cfg.PLANES:
        if plane == "sagittal":
            losses[plane] = SagittalCombinedLoss(cfg=base_cfg, region_pos_weight=region_pos_weight)
        else:
            losses[plane] = CombinedSegLoss(cfg=PlaneConfigView(plane, base_cfg))
    return losses


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-plane class-weight computation
#
#  ── ASSUMPTION (flagged): no manual `boost_structures` overrides are
#  applied here, unlike the fetal loss.py (which boosted calvarium/CSP/
#  arrow sign by fixed multipliers). Nothing was specified for spine.
#  The mask report flagged these as very rare and worth a look if
#  automatic inverse-frequency weighting isn't enough:
#    coronal            -> "Coronal spine- Cervical Region"
#    full_body_coronal  -> "Gall Bladder", "Diaphragm"
#  Add a per-plane boost dict here if you want that; structure is ready
#  for it (see the commented block below).
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights_multiplane(
    dataloader,
    base_cfg=MultiPlaneConfig,
    device: torch.device = torch.device("cpu"),
    max_batches: int = 200,
    mask_key: str = "mask",
    plane_key: str = "plane",
) -> Dict[str, torch.Tensor]:
    """
    Scans up to max_batches of the (mixed-plane) training dataloader,
    accumulating per-class pixel frequencies SEPARATELY per plane (since
    a batch may contain samples from more than one plane), then applies
    the same inv_smooth weighting math as loss.py's compute_class_weights
    — independently for each plane's own class count.

    Returns: {plane: class_weights (NUM_CLASSES_PER_PLANE[plane],) tensor on device}
    """
    planes = base_cfg.PLANES
    def _n_classes(p):
        return base_cfg.NUM_CLASSES_PER_PLANE["sagittal_exclusive"] if p == "sagittal" \
            else base_cfg.NUM_CLASSES_PER_PLANE[p]

    def _n_classes(p):
        return base_cfg.NUM_CLASSES_PER_PLANE["sagittal_exclusive"] if p == "sagittal" \
            else base_cfg.NUM_CLASSES_PER_PLANE[p]

    planes = base_cfg.PLANES
    pixel_counts = {
        p: torch.zeros(_n_classes(p), dtype=torch.float64)
        for p in planes
    }
    tot_pixels = {p: 0 for p in planes}

    print(f"  Scanning training data for per-plane class frequencies "
          f"(mask_key='{mask_key}') …")

    for i, batch in enumerate(dataloader):
        masks  = batch[mask_key]          # (B, H, W) int64 — plane-local class indices
        planes_in_batch = batch[plane_key]  # list[str], length B

        for p in set(planes_in_batch):
            idx = [j for j, pl in enumerate(planes_in_batch) if pl == p]
            if not idx:
                continue
            m = masks[idx].long()
            B, H, W = m.shape
            tot_pixels[p] += B * H * W
            counts = torch.bincount(
                m.reshape(-1), minlength=_n_classes(p)
            ).float()
            pixel_counts[p] += counts.double()

        if i >= max_batches - 1:
            break

    class_weights: Dict[str, torch.Tensor] = {}

    for p in planes:
        num_classes = _n_classes(p)
        view        = PlaneConfigView("sagittal_exclusive" if p == "sagittal" else p, base_cfg)

        if tot_pixels[p] == 0:
            print(f"  [WARN] No samples seen for plane '{p}' in the first "
                  f"{max_batches} batches — using uniform weights (all 1.0).")
            class_weights[p] = torch.ones(num_classes, dtype=torch.float32).to(device)
            continue

        freq = (pixel_counts[p] / (tot_pixels[p] + 1e-8)).float()   # (C,)

        fg_freq = freq[1:]
        raw_w   = _apply_weight_strategy(fg_freq, view.CLASS_WEIGHT_STRATEGY, cfg=view)
        raw_w   = raw_w / (raw_w.mean() + 1e-8)
        fg_w    = torch.clamp(raw_w, view.CLASS_WEIGHT_MIN, view.CLASS_WEIGHT_MAX)
        fg_w    = fg_w / (fg_w.mean() + 1e-8)

        class_w      = torch.ones(num_classes, dtype=torch.float32)
        class_w[1:]  = fg_w

        # ── Optional manual boosts go here, e.g.:
        # boost_structures = {"Coronal spine- Cervical Region": 2.0}
        # for name, boost in boost_structures.items():
        #     idx = view.STRUCTURE_TO_IDX.get(name)
        #     if idx is not None and idx < num_classes:
        #         class_w[idx] *= boost
        # class_w[1:] = class_w[1:] / (class_w[1:].mean() + 1e-8)
        
        boost_structures = {
            "coronal": {
                "Coronal spine- Cervical Region": 4.0,
                "Coronal spine- Sacral Region":   3.0,
                "Coronal spine- Thoracic Region": 3.0,
                "Coronal spine- Lumbar Region":   3.0,
            }
        }
        if p in boost_structures:
            for name, boost in boost_structures[p].items():
                idx = view.STRUCTURE_TO_IDX.get(name)
                if idx is not None and idx < num_classes:
                    class_w[idx] *= boost
            class_w[1:] = class_w[1:] / (class_w[1:].mean() + 1e-8)

            
        _print_weight_table(num_classes, freq, class_w, view.STRUCTURES, plane_label=p)

        class_weights[p] = class_w.to(device)

    return class_weights

def compute_region_pos_weight(
    dataloader,
    num_regions: int = 4,
    max_batches: int = 200,
    device: torch.device = torch.device("cpu"),
    region_key: str = "region_mask",
) -> torch.Tensor:
    """Scans training data for sagittal region-mask positive/negative pixel
    counts, returns BCEWithLogitsLoss-compatible pos_weight per region."""
    pos_count = torch.zeros(num_regions, dtype=torch.float64)
    tot_pixels = 0

    print(f"  Scanning training data for sagittal region frequencies…")

    for i, batch in enumerate(dataloader):
        if region_key not in batch:
            continue
        regions = batch[region_key]  # (B, R, H, W)
        pos_count += regions.sum(dim=(0, 2, 3)).double()
        tot_pixels += regions.shape[0] * regions.shape[2] * regions.shape[3]
        if i >= max_batches - 1:
            break

    if tot_pixels == 0:
        print("  [WARN] No region_mask data found — using pos_weight=1.0")
        return torch.ones(num_regions, dtype=torch.float32).to(device)

    neg_count  = tot_pixels - pos_count
    pos_weight = (neg_count / (pos_count + 1e-6)).float()
    pos_weight = torch.clamp(pos_weight, max=15.0)   # prevent destabilizing shared encoder
    print(f"  Region pos_weight (clipped): {pos_weight.tolist()}")
    return pos_weight.to(device)


def build_loss(cfg=None) -> CombinedSegLoss:
    """Single-plane factory, kept for API parity with loss.py's build_loss."""
    return CombinedSegLoss(cfg=cfg or MultiPlaneConfig)

class RegionBCELoss(nn.Module):
    def __init__(self, cfg=None, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            pos_weight = pos_weight.view(-1, 1, 1)  # (R,) -> (R, 1, 1) for broadcast over (B, R, H, W)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.bce(logits, targets)


class SagittalCombinedLoss(nn.Module):
    def __init__(self, cfg=None, region_pos_weight=None):
        super().__init__()
        cfg = cfg or MultiPlaneConfig
        self.excl_loss   = CombinedSegLoss(cfg=PlaneConfigView("sagittal_exclusive", cfg))
        self.region_loss = RegionBCELoss(cfg, pos_weight=region_pos_weight)
        self.region_w    = cfg.REGION_LOSS_WEIGHT

    def forward(self, logits_excl, targets_excl, logits_region, targets_region, class_weights=None):
        l1 = self.excl_loss(logits_excl, targets_excl, class_weights)
        l2 = self.region_loss(logits_region, targets_region)
        return l1 + self.region_w * l2