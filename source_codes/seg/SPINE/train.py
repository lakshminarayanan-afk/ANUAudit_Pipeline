import os
import random
import time
import multiprocessing as mp
from pathlib import Path



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
sys.path.append('/mnt/data4tb/anusha/Prathicksha_Spine')

from config.config import Config
from dataset.dataset import build_dataloaders
from model.model import build_model
from model.loss import build_losses_per_plane, compute_class_weights_multiplane, compute_region_pos_weight


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler  — UNCHANGED from train_spine.py
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, n_epochs: int):
    sched  = Config.LR_SCHEDULER
    warmup = Config.WARMUP_EPOCHS

    if sched == "cosine":
        main = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs - warmup), eta_min=1e-7
        )
    elif sched == "plateau":
        main = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif sched == "step":
        main = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        raise ValueError(f"Unknown LR_SCHEDULER: {sched}")

    if warmup > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, main], milestones=[warmup]
        )
    return main


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics  — UNCHANGED from train_spine.py (one instance built per plane)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """Compute Dice, IoU, precision, recall per class."""

    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device      = device
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64, device=self.device
        )

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds   : (B, H, W) int64 class indices
        targets : (B, H, W) int64 class indices
        """
        preds   = preds.flatten().to(self.device)
        targets = targets.flatten().to(self.device)

        # Build confusion matrix
        # confusion[i, j] = number of pixels with GT=i predicted as j
        indices = targets * self.num_classes + preds
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes ** 2
        ).view(self.num_classes, self.num_classes)

    def compute(self):
        c = self.confusion.float()
        smooth = 1e-8

        # Per-class metrics
        tp = torch.diag(c)                          # (C,)
        fp = c.sum(0) - tp                          # (C,)
        fn = c.sum(1) - tp                          # (C,)

        dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        iou       = (tp + smooth) / (tp + fp + fn + smooth)
        precision = (tp + smooth) / (tp + fp + smooth)
        recall    = (tp + smooth) / (tp + fn + smooth)

        # Count how many classes are actually present in GT
        gt_present_count = (c.sum(1) > 0).long()

        # Mean over classes that are present
        present_mask = gt_present_count > 0
        n_present    = present_mask.sum().item()

        if n_present > 0:
            mean_dice      = dice[present_mask].mean().item()
            mean_iou       = iou[present_mask].mean().item()
            mean_precision = precision[present_mask].mean().item()
            mean_recall    = recall[present_mask].mean().item()
        else:
            mean_dice = mean_iou = mean_precision = mean_recall = 0.0

        return {
            "dice_per_class":      dice.cpu().numpy(),
            "iou_per_class":       iou.cpu().numpy(),
            "precision_per_class": precision.cpu().numpy(),
            "recall_per_class":    recall.cpu().numpy(),
            "mean_dice":           mean_dice,
            "mean_iou":            mean_iou,
            "mean_precision":      mean_precision,
            "mean_recall":         mean_recall,
            "n_present_classes":   n_present,
            "gt_present_count":    gt_present_count.cpu().numpy(),
        }


class BinaryRegionMetrics:
    """Per-channel independent Dice/IoU for sigmoid multi-label regions."""
    def __init__(self, num_regions: int, device: torch.device):
        self.num_regions = num_regions
        self.device      = device
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_regions, device=self.device)
        self.fp = torch.zeros(self.num_regions, device=self.device)
        self.fn = torch.zeros(self.num_regions, device=self.device)
        self.gt_present = torch.zeros(self.num_regions, device=self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds, targets = preds.to(self.device), targets.to(self.device)
        for r in range(self.num_regions):
            p, g = preds[:, r].reshape(-1), targets[:, r].reshape(-1)
            self.tp[r] += (p * g).sum()
            self.fp[r] += (p * (1 - g)).sum()
            self.fn[r] += ((1 - p) * g).sum()
            self.gt_present[r] += (g.sum() > 0).float()

    def compute(self):
        smooth = 1e-8
        dice = (2 * self.tp + smooth) / (2 * self.tp + self.fp + self.fn + smooth)
        iou  = (self.tp + smooth) / (self.tp + self.fp + self.fn + smooth)
        mask = self.gt_present > 0
        n    = mask.sum().item()
        return {
            "dice_per_region": dice.cpu().numpy(),
            "iou_per_region":  iou.cpu().numpy(),
            "mean_dice": dice[mask].mean().item() if n else 0.0,
            "mean_iou":  iou[mask].mean().item()  if n else 0.0,
            "n_present_regions": n,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization helpers
#  MULTIPLANE ADAPTATION: original _per_class_bar used a single flat
#  Config.NUM_CLASSES / Config.STRUCTURES. Those are now dicts keyed by
#  plane, so the bar-chart helper takes a `plane` arg and looks up that
#  plane's own structure list / class count. Logic inside is unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def _colors_for_plane(plane: str):
    n = Config.NUM_CLASSES_PER_PLANE[plane]
    return [mcolors.hsv_to_rgb((i / n, 0.8, 0.92)) for i in range(n)]


def _per_class_bar(plane: str, dice_scores, present_counts, title):
    fg_dice    = dice_scores[1:]
    fg_present = present_counts[1:]
    fg_names = Config.STRUCTURES['sagittal_exclusive'] if plane == 'sagittal' else Config.STRUCTURES[plane]
    n          = len(fg_names)

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))
    y = np.arange(n)

    display  = np.where(np.isnan(fg_dice), 0.0, fg_dice)
    colours, hatches = [], []
    for i in range(n):
        if np.isnan(fg_dice[i]) or fg_present[i] == 0:
            colours.append("#aaaaaa"); hatches.append("//")
        elif fg_dice[i] >= 0.7:
            colours.append("#2ecc71"); hatches.append("")
        elif fg_dice[i] >= 0.4:
            colours.append("#e67e22"); hatches.append("")
        else:
            colours.append("#e74c3c"); hatches.append("")

    for i in range(n):
        ax.barh(y[i], display[i], color=colours[i],
                hatch=hatches[i], edgecolor="white", linewidth=0.4)
        label = f"{display[i]:.3f}" if not np.isnan(fg_dice[i]) else "absent"
        ax.text(min(display[i] + 0.01, 1.0), y[i], label, va="center", fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(fg_names, fontsize=7)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Dice Score")
    ax.set_title(title, fontsize=11, fontweight="bold")

    valid = display[~np.isnan(fg_dice)]
    if len(valid):
        ax.axvline(valid.mean(), color="steelblue", linestyle="--",
                   linewidth=1.2, label=f"mean={valid.mean():.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Train / eval loops
#  MULTIPLANE ADAPTATION: model(images) now returns a dict of logits (one
#  per plane) for every sample in the batch. Each sample's loss/metrics are
#  routed to ONLY its own plane's head, using the `plane` tag from the
#  dataloader. Per-plane losses present in a batch are averaged (per
#  config_multiplane.py's documented rule), not summed, so batch
#  composition doesn't change gradient magnitude.
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterions, scaler, device, class_weights,
):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    last_print = time.time()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)   # (B,1,H,W)
        masks  = batch["mask"].to(device, non_blocking=True)    # (B,H,W) int64
        planes = batch["plane"]                                  # list[str], len B

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=Config.MIXED_PRECISION):
            outputs = model(images)   # dict: plane -> (B, C_plane, H, W) logits

            regions = batch["region_mask"].to(device, non_blocking=True)  # (B,4,H,W)

            plane_losses = []
            for plane in Config.PLANES:
                idx = [i for i, p in enumerate(planes) if p == plane]
                if not idx:
                    continue
                idx_t   = torch.tensor(idx, device=device)
                masks_p = masks.index_select(0, idx_t)
                cw      = class_weights[plane] if class_weights is not None else None

                if plane == "sagittal":
                    logits_excl   = outputs["sagittal_exclusive"].index_select(0, idx_t)
                    logits_region = outputs["sagittal_regions"].index_select(0, idx_t)
                    regions_p     = regions.index_select(0, idx_t)
                    loss_p = criterions[plane](logits_excl, masks_p, logits_region, regions_p, cw)
                else:
                    logits_p = outputs[plane].index_select(0, idx_t)
                    loss_p   = criterions[plane](logits_p, masks_p, cw)

                plane_losses.append(loss_p)
            # Average the per-plane losses PRESENT in this batch (not sum) —
            # see config_multiplane.py's cross-plane batch loss note.
            loss = torch.stack(plane_losses).mean()

        if loss.requires_grad:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()

        now = time.time()
        if now - last_print >= 5.0:
            avg_so_far = total_loss / (batch_idx + 1)
            print(f"    [batch {batch_idx+1}/{n_batches}] avg_loss={avg_so_far:.4f}", flush=True)
            last_print = now

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterions, device, class_weights, metrics_trackers):
    model.eval()
    total_loss = 0.0
    for tracker in metrics_trackers.values():
        tracker.reset()

    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        masks   = batch["mask"].to(device, non_blocking=True)
        regions = batch["region_mask"].to(device, non_blocking=True)
        planes  = batch["plane"]

        with autocast("cuda", enabled=Config.MIXED_PRECISION):
            outputs = model(images)

            plane_losses = []
            for plane in Config.PLANES:
                idx = [i for i, p in enumerate(planes) if p == plane]
                if not idx:
                    continue
                idx_t   = torch.tensor(idx, device=device)
                masks_p = masks.index_select(0, idx_t)
                cw      = class_weights[plane] if class_weights is not None else None

                if plane == "sagittal":
                    logits_excl   = outputs["sagittal_exclusive"].index_select(0, idx_t)
                    logits_region = outputs["sagittal_regions"].index_select(0, idx_t)
                    regions_p     = regions.index_select(0, idx_t)
                    loss_p = criterions[plane](logits_excl, masks_p, logits_region, regions_p, cw)
                    plane_losses.append(loss_p)

                    preds_p = torch.argmax(logits_excl, dim=1)
                    metrics_trackers[plane].update(preds_p, masks_p)

                    region_preds = (torch.sigmoid(logits_region) > 0.5).float()
                    metrics_trackers["sagittal_regions"].update(region_preds, regions_p)
                else:
                    logits_p = outputs[plane].index_select(0, idx_t)
                    loss_p   = criterions[plane](logits_p, masks_p, cw)
                    plane_losses.append(loss_p)

                    preds_p = torch.argmax(logits_p, dim=1)
                    metrics_trackers[plane].update(preds_p, masks_p)

            loss = torch.stack(plane_losses).mean()

        total_loss += loss.item()

    metrics_per_plane = {p: metrics_trackers[p].compute() for p in Config.PLANES}
    metrics_per_plane["sagittal_regions"] = metrics_trackers["sagittal_regions"].compute()
    return total_loss / len(loader), metrics_per_plane

# ─────────────────────────────────────────────────────────────────────────────
#  Main training function
#  MULTIPLANE ADAPTATION: checkpoint selection uses the mean of the three
#  planes' mean_dice (equal-weighted average, consistent with the
#  "average not sum" rule used for batch loss combination). Everything
#  else — optimizer, scaler, early stopping, checkpoint cadence, test-set
#  evaluation at the end — follows train_spine.py exactly.
# ─────────────────────────────────────────────────────────────────────────────

def train():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(Config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading multiplane spine dataset (coronal / sagittal / full_body_coronal)...")
    train_loader, val_loader, test_loader = build_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nBuilding shared-encoder 3-head UNet...")
    model = build_model().to(device)

    # Freeze everything except the sagittal region head
    for name, param in model.named_parameters():
        param.requires_grad = "out_conv_sagittal_regions" in name

    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_p:,}")

    # ── Class weights (per plane) ─────────────────────────────────────────
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        print("\nComputing per-plane class weights...")
        class_weights = compute_class_weights_multiplane(
            train_loader, Config, device, mask_key="mask", plane_key="plane"
        )

    # ── Loss (per plane) & Optimizer ─────────────────────────────────────

    region_pos_weight = compute_region_pos_weight(
        train_loader, num_regions=Config.NUM_REGIONS_PER_PLANE["sagittal"], device=device
    )
    criterions = build_losses_per_plane(Config, region_pos_weight=region_pos_weight)
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = build_scheduler(optimizer, Config.EPOCHS)
    scaler    = GradScaler("cuda")

    def _n_classes(p):
        return Config.NUM_CLASSES_PER_PLANE["sagittal_exclusive"] if p == "sagittal" \
            else Config.NUM_CLASSES_PER_PLANE[p]

    metrics_trackers = {
        p: SegmentationMetrics(_n_classes(p), device)
        for p in Config.PLANES
    }
    metrics_trackers["sagittal_regions"] = BinaryRegionMetrics(
        Config.NUM_REGIONS_PER_PLANE["sagittal"], device
    )

    # ── Resume from checkpoint if present ───────────────────────────────
    start_epoch    = 1
    best_dice      = 0.0
    best_epoch     = 0
    patience_count = 0

    last_ckpt_path = Path(Config.CHECKPOINT_DIR) / "last.pth"
    old_best_ckpt = "./checkpoint_new/best_model.pth"

    if last_ckpt_path.exists():
        # This new fine-tuning run already started and crashed/stopped —
        # resume it fully (model + optimizer + scheduler + epoch state).
        print(f"\nResuming interrupted fine-tune run from: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch    = ckpt["epoch"] + 1
        best_dice      = ckpt.get("best_dice", 0.0)
        best_epoch     = ckpt.get("epoch", 0)
        patience_count = ckpt.get("patience_count", 0)
        print(f"  Resumed at epoch {start_epoch}, best_dice so far={best_dice:.4f}")

    elif Path(old_best_ckpt).exists():
        # First time running this fine-tune — warm-start weights only.
        # out_conv_sagittal_regions changed shape (added ypos channel), so
        # its old weights can't be reused — skip just that layer and keep
        # its fresh random init; load everything else normally.
        print(f"\nWarm-starting model weights from: {old_best_ckpt}")
        ckpt = torch.load(old_best_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"]
        filtered = {k: v for k, v in state_dict.items() if "out_conv_sagittal_regions" not in k}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"  Loaded weights (was epoch {ckpt.get('epoch','?')}, "
              f"mean_dice={ckpt.get('mean_dice','?')})")
        print(f"  Skipped (fresh init): {[m for m in missing if 'out_conv_sagittal_regions' in m]}")

    else:
        print("\nNo prior checkpoint found — starting fresh.")

    # Freeze everything except the sagittal region head
    for name, param in model.named_parameters():
        param.requires_grad = "out_conv_sagittal_regions" in name

    # ── Training loop ─────────────────────────────────────────────────────
    log_path = Path(Config.LOG_DIR) / "train_log.csv"

    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write(
                "epoch,train_loss,val_loss,"
                + ",".join(f"{p}_dice" for p in Config.PLANES)
                + ",mean_dice,lr\n"
            )

    hdr = (f"{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
           + "  ".join(f"{p[:8]:>8}" for p in Config.PLANES)
           + f"  {'mDice':>8}  {'LR':>10}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterions,
            scaler, device, class_weights
        )

        va_loss, metrics_per_plane = evaluate(
            model, val_loader, criterions, device, class_weights, metrics_trackers
        )

        plane_dice = {p: metrics_per_plane[p]["mean_dice"] for p in Config.PLANES}
        region_dice = metrics_per_plane["sagittal_regions"]["mean_dice"]
        # Only the region head is being trained right now — select checkpoints
        # by region dice alone, since the other 3 planes are frozen and can't
        # change (using their mean would mask whether the region head is
        # actually improving).
        mean_dice  = region_dice
        lr         = optimizer.param_groups[0]["lr"]

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(va_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t0

        # ── Logging ───────────────────────────────────────────────────────
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{tr_loss:.4f},{va_loss:.4f},"
                + ",".join(f"{plane_dice[p]:.4f}" for p in Config.PLANES)
                + f",{mean_dice:.4f},{lr:.2e}\n"
            )

        marker = " ✓" if mean_dice > best_dice else ""
        dice_str = "  ".join(f"{plane_dice[p]:>8.4f}" for p in Config.PLANES)
        print(f"{epoch:>4}  {tr_loss:>8.4f}  {va_loss:>8.4f}  {dice_str}  "
              f"{mean_dice:>8.4f}  {lr:>10.2e}{marker}  ({elapsed:.1f}s)")

        # ── Checkpoint ────────────────────────────────────────────────────
        ckpt_dict = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),
            "best_dice":        best_dice,
            "patience_count":   patience_count,
            "val_loss":         va_loss,
            "mean_dice":        mean_dice,
            "plane_dice":       plane_dice,
        }

        if mean_dice > best_dice:
            best_dice      = mean_dice
            best_epoch     = epoch
            patience_count = 0
            ckpt_dict["best_dice"]      = best_dice
            ckpt_dict["patience_count"] = patience_count
            torch.save(ckpt_dict, Config.BEST_MODEL)
            print(f"  → Saved best model (mean Dice={mean_dice:.4f})")
        else:
            patience_count += 1
            ckpt_dict["patience_count"] = patience_count

        # always save latest state — this is what resume loads
        torch.save(ckpt_dict, last_ckpt_path)

        if epoch % 10 == 0:
            p = Path(Config.CHECKPOINT_DIR) / f"epoch_{epoch:04d}.pth"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, p)

        if patience_count >= Config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping after epoch {epoch}.")
            break

    print("=" * len(hdr))
    print(f"Best mean Dice = {best_dice:.4f}  (epoch {best_epoch})")

    # ── Test ──────────────────────────────────────────────────────────────
    print("\nLoading best model for testing...")
    ckpt = torch.load(Config.BEST_MODEL, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    te_loss, te_metrics_per_plane = evaluate(
        model, test_loader, criterions, device, class_weights, metrics_trackers
    )

    print(f"\nTest loss: {te_loss:.4f}")
    for p in Config.PLANES:
        m = te_metrics_per_plane[p]
        print(f"\n[{p}]")
        print(f"  mDice: {m['mean_dice']:.4f}")
        print(f"  mIoU:  {m['mean_iou']:.4f}")
        print(f"  mPrec: {m['mean_precision']:.4f}")
        print(f"  mRec:  {m['mean_recall']:.4f}")
        structs = Config.STRUCTURES['sagittal_exclusive'] if p == 'sagittal' else Config.STRUCTURES[p]
        fg_present_count = m['gt_present_count'][1:]
        n_fg_present = int((fg_present_count > 0).sum())
        print(f"  Classes present: {n_fg_present}/{len(structs)}")

    rm = te_metrics_per_plane["sagittal_regions"]
    print(f"\n[sagittal_regions]")
    print(f"  mDice: {rm['mean_dice']:.4f}")
    print(f"  mIoU:  {rm['mean_iou']:.4f}")
    print(f"  Regions present: {rm['n_present_regions']}/4")

    te_mean_dice = sum(te_metrics_per_plane[p]["mean_dice"] for p in Config.PLANES) / len(Config.PLANES)
    print(f"\nTest combined mean Dice (equal-weighted across planes): {te_mean_dice:.4f}\n")

    # Save one test-metrics chart per plane
    for p in Config.PLANES:
        m = te_metrics_per_plane[p]
        structs = Config.STRUCTURES['sagittal_exclusive'] if p == 'sagittal' else Config.STRUCTURES[p]
        fg_present_count = m['gt_present_count'][1:]
        n_fg_present = int((fg_present_count > 0).sum())
        fig = _per_class_bar(
            p,
            m["dice_per_class"],
            m["gt_present_count"],
            title=f"[{p}] Test Dice per class [{n_fg_present}/{len(structs)} present]",
        )
        chart_path = Path(Config.LOG_DIR) / f"test_dice_per_class_{p}.png"
        plt.savefig(str(chart_path), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Test metrics chart saved to: {chart_path}")


if __name__ == "__main__":
    train()