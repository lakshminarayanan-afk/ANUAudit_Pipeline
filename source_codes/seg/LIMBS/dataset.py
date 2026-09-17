"""
dataset.py
----------
Reads from the plane-based layout defined in config.py:

    Config.DATASET_DIR/
        TRAIN/<PLANE>/images/*.png    TRAIN/<PLANE>/masks/*.npz
        VAL/<PLANE>/images/*.png      VAL/<PLANE>/masks/*.npz
        TEST/<PLANE>/images/*.png     TEST/<PLANE>/masks/*.npz

    <PLANE> in {HU, RU, FE, TF}. BoneUSDataset pools all 4 planes into one dataset by default so a
    single model learns to segment + discriminate all 4 bone groups (matches the architecture: one
    shared encoder/decoder, one seg head, one "which bone" classification+contrastive head).

Each mask .npz contains (see config.py docstring for full detail):
    mask_structures : (12, H, W) uint8, values in {0, 255}
    mask_artifacts  : ( 8, H, W) uint8, values in {0, 255}  [only used if shadow_as_ignore=True]
    structure_names : (12,) str
    artifact_names  : ( 8,) str

Only the 6 bone channels (Config.BONE_STRUCTURES) become segmentation targets; everything else in
structure_names (Placenta, Hand, Thigh, Femur Length, Foot, Lower Leg) is ignored on purpose - the
model has to learn to use context on its own rather than being hand-fed soft-tissue landmarks.

Radius+Ulna merge into one class, Tibia+Fibula merge into one class (Config.BONE_NAME_TO_CLASS /
Config.LABEL_MAP). Overlap between bone channels at the same pixel is resolved deterministically via
Config.STRUCTURE_LABEL_PRIORITY (earlier name in the list wins).

NO RESIZING is performed anywhere. Images stay at native resolution. Because a batch needs uniform
tensor shape, the custom `collate_fn` pads every sample in a batch up to the max H/W *in that batch*
(padding, not resizing) using reflect-pad for images and a constant ignore value for masks so padded
pixels never contribute to the loss.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
# import albumentations as A

from source_codes.seg.LIMBS.config import Config
# from LIMBS.config import Config

# Re-exported for convenience so existing imports (`from dataset import LABEL_MAP, ...`) keep working.
LABEL_MAP = Config.LABEL_MAP
NUM_SEG_CLASSES = Config.NUM_SEG_CLASSES
NUM_BONE_CLASSES = Config.NUM_BONE_CLASSES
IGNORE_INDEX = Config.IGNORE_INDEX


def get_train_augmentations():
    # NOTE: no Resize / RandomResizedCrop here on purpose. Only augmentations that preserve
    # native resolution and pixel spacing (important for ultrasound geometry).
    return A.Compose([
        A.HorizontalFlip(p=Config.AUG_HFLIP),
        A.RandomBrightnessContrast(brightness_limit=Config.AUG_BRIGHTNESS_LIMIT,
                                    contrast_limit=Config.AUG_CONTRAST_LIMIT,
                                    p=Config.AUG_BRIGHTNESS_CONTRAST),
        A.GaussNoise(std_range=(0.02, 0.1), p=Config.AUG_GAUSS_NOISE),
        A.CLAHE(clip_limit=2.0, p=Config.AUG_CLAHE),
        A.Affine(translate_percent=Config.AUG_AFFINE_TRANSLATE,
                  rotate=(-Config.AUG_AFFINE_ROTATE_LIMIT, Config.AUG_AFFINE_ROTATE_LIMIT),
                  scale=Config.AUG_AFFINE_SCALE,
                  shear=(-Config.AUG_AFFINE_SHEAR, Config.AUG_AFFINE_SHEAR),
                  border_mode=cv2.BORDER_REFLECT_101,
                  # FIX (VERIFIED ISSUE #4): this was hardcoded to p=0.0, silently disabling
                  # affine augmentation regardless of Config.AUG_AFFINE. Use the existing
                  # configured probability instead.
                  p=Config.AUG_AFFINE
                  ),
    ])


def get_val_augmentations():
    return A.Compose([])  # no-op: native resolution, no augmentation at eval time


def _to_channel_first(arr, names):
    """Normalize a (C,H,W) or (H,W,C) array to (C,H,W) using len(names) to disambiguate."""
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D structures array, got shape {arr.shape}")
    if arr.shape[0] == len(names):
        return arr
    if arr.shape[-1] == len(names):
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Could not match structures array shape {arr.shape} to {len(names)} names")


def load_bone_mask_from_npz(npz_path, shadow_as_ignore=False):
    """
    Returns a single-channel (H, W) uint8 mask with values in Config.LABEL_MAP (0..4), plus
    Config.IGNORE_INDEX for pixels excluded via shadow_as_ignore.
    """
    data = np.load(npz_path, allow_pickle=True)
    structures = data["mask_structures"]
    structure_names = [str(n) for n in data["structure_names"]]
    structures = _to_channel_first(structures, structure_names)

    H, W = structures.shape[1], structures.shape[2]
    combined = np.zeros((H, W), dtype=np.uint8)

    name_to_idx = {name: i for i, name in enumerate(structure_names)}

    # Write in REVERSE priority order so the highest-priority name (first in the list) is
    # written last and therefore wins wherever two bone channels overlap.
    for name in reversed(Config.STRUCTURE_LABEL_PRIORITY):
        if name not in Config.BONE_NAME_TO_CLASS:
            continue  # not one of the 6 bone structures we train on
        if name in name_to_idx:
            class_idx = Config.BONE_NAME_TO_CLASS[name]
            channel = structures[name_to_idx[name]] > Config.MASK_THRESHOLD
            combined[channel] = class_idx

    if shadow_as_ignore and "mask_artifacts" in data and "artifact_names" in data:
        artifacts = data["mask_artifacts"]
        artifact_names = [str(n) for n in data["artifact_names"]]
        artifacts = _to_channel_first(artifacts, artifact_names)
        art_name_to_idx = {name: i for i, name in enumerate(artifact_names)}
        for class_idx, shadow_names in Config.BONE_CLASS_TO_SHADOW_NAMES.items():
            for shadow_name in shadow_names:
                if shadow_name in art_name_to_idx:
                    shadow_region = artifacts[art_name_to_idx[shadow_name]] > Config.MASK_THRESHOLD
                    # only blank out shadow pixels that aren't already a confident bone label
                    combined[shadow_region & (combined == 0)] = IGNORE_INDEX

    return combined


# class BoneUSDataset(Dataset):
#     def __init__(self, split="train", planes=None, pad_multiple=None, transform=None,
#                  shadow_as_ignore=False):
#         """
#         split: "train" | "val" | "test" -> selects Config.DATASET_DIR/<SPLIT>/<plane>/...
#         planes: list of plane codes to pool together, defaults to Config.PLANES (all 4: HU/RU/FE/TF).
#         pad_multiple: defaults to Config.PAD_MULTIPLE (32).
#         shadow_as_ignore: if True, pixels under the matching bone's shadowing/artifact channel that
#                       aren't already labeled as bone get IGNORE_INDEX instead of background. Off by
#                       default - turn on if shadowing regions are confusing the segmentation loss.
#         """
#         self.split = split
#         self.planes = planes or Config.PLANES
#         self.pad_multiple = pad_multiple or Config.PAD_MULTIPLE
#         self.shadow_as_ignore = shadow_as_ignore

#         self.samples = []

#         for plane in self.planes:
#             img_dir = Config.images_dir(split, plane)

#             if not img_dir.exists():
#                 continue

#     # ---------------------------------------------------------
#     # UNSEEN / ONLINE INFERENCE
#     # No GT masks required.
#     # ---------------------------------------------------------
#             if split.lower() == "online":
#                 for img_path in sorted(img_dir.glob("*")):
#                     self.samples.append(
#                         (str(img_path), None, plane)
#             )

#     # ---------------------------------------------------------
#     # NORMAL TRAIN / VAL / TEST
#     # GT masks are required as before.
#     # ---------------------------------------------------------
#             else:
#                 mask_dir = Config.masks_dir(split, plane)

#                 for img_path in sorted(img_dir.glob("*")):
#                     mask_path = mask_dir / (img_path.stem + ".npz")

#                     if mask_path.exists():
#                         self.samples.append(
#                             (str(img_path), str(mask_path), plane)
#                 )

#         if len(self.samples) == 0:
#             raise RuntimeError(
#                 f"No samples found for split='{split}', planes={self.planes}. "
#                 f"Check Config.DATASET_DIR = {Config.DATASET_DIR}"
#             )

#         if transform is None:
#             transform = get_train_augmentations() if split == "train" else get_val_augmentations()
#         self.transform = transform
#         # for plane in self.planes:

#         #     img_dir = Config.images_dir(split, plane)
#         #     mask_dir = Config.masks_dir(split, plane)

#         #     print(f"\nPlane: {plane}")
#         #     print(f"Image dir: {img_dir}")
#         #     print(f"Mask dir : {mask_dir}")
#         #     print(f"Image dir exists? {img_dir.exists()}")
#         #     print(f"Mask dir exists? {mask_dir.exists()}")

#         #     if not img_dir.exists():
#         #         continue

#         #     imgs = list(img_dir.glob("*"))
#         #     print(f"Found {len(imgs)} image files")

#         #     for img_path in imgs:
#         #         mask_path = mask_dir / (img_path.stem + ".npz")

#         #         if mask_path.exists():
#         #             self.samples.append((str(img_path), str(mask_path), plane))
#         #         else:
#         #             print(f"Missing mask: {mask_path}")

#     def __len__(self):
#         return len(self.samples)

#     def _pad_to_multiple(self, image, mask):
#         h, w = image.shape[:2]
#         pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
#         pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
#         if pad_h == 0 and pad_w == 0:
#             return image, mask
#         image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
#         mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
#         return image, mask

#     def _derive_bone_label(self, mask):
#         # "which bone" label = dominant non-background, non-ignore class present in the mask.
#         vals, counts = np.unique(mask[mask != IGNORE_INDEX], return_counts=True)
#         fg = [(v, c) for v, c in zip(vals, counts) if v != 0]
#         if not fg:
#             # no foreground bone found in this crop/sample -> shouldn't normally happen if
#             # your dataset only contains images that show one of the 6 bones.
#             return 0
#         dominant_val = max(fg, key=lambda vc: vc[1])[0]
#         return int(dominant_val) - 1  # shift so femur=0, humerus=1, radius_ulna=2, tibia_fibula=3

#     def __getitem__(self, idx):
#         img_path, mask_path, plane = self.samples[idx]

#         image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             raise FileNotFoundError(f"Could not read image: {img_path}")

#         # ---------------------------------------------------------
#         # ONLINE / UNSEEN DATA: no GT mask
#         # ---------------------------------------------------------
#         if mask_path is None:
#             mask = np.zeros(image.shape[:2], dtype=np.int64)

#             if self.transform is not None:
#                 # Apply image-only transformation.
#                 # We still pass a dummy mask because the existing
#                 # Albumentations pipeline expects image + mask.
#                 augmented = self.transform(image=image, mask=mask)
#                 image, mask = augmented["image"], augmented["mask"]

#             image, mask = self._pad_to_multiple(image, mask)

#             # No meaningful GT bone label exists for unseen data.
#             bone_label = 0

#         # ---------------------------------------------------------
#         # NORMAL TRAIN / VAL / TEST DATA: use real GT mask
#         # ---------------------------------------------------------
#         else:
#             mask = load_bone_mask_from_npz(
#                 mask_path,
#                 shadow_as_ignore=self.shadow_as_ignore
#             )

#             if mask.shape != image.shape[:2]:
#                 raise ValueError(
#                     f"Image/mask size mismatch for {img_path}: "
#                     f"image {image.shape[:2]} vs mask {mask.shape}"
#                 )

#             if self.transform is not None:
#                 augmented = self.transform(image=image, mask=mask)
#                 image, mask = augmented["image"], augmented["mask"]

#             image, mask = self._pad_to_multiple(image, mask)

#             bone_label = self._derive_bone_label(mask)

#         # ---------------------------------------------------------
#         # Common image processing
#         # ---------------------------------------------------------
#         image = image.astype(np.float32) / 255.0
#         image = (image - 0.5) / 0.5

#         image_t = torch.from_numpy(image).unsqueeze(0).float()
#         mask_t = torch.from_numpy(mask.astype(np.int64))
#         bone_label_t = torch.tensor(bone_label, dtype=torch.long)

#         return {
#             "image": image_t,
#             "mask": mask_t,
#             "bone_label": bone_label_t,
#             "plane": plane,
#             "orig_size": (image.shape[0], image.shape[1]),
#             "path": img_path,
#         }

def collate_fn(batch):
    """
    Pads every sample in the batch to the max (H, W) found in THIS batch (not a fixed global size).
    This keeps things at native resolution as much as possible - the only padding beyond each
    sample's own pad_multiple rounding is the extra needed to match the largest sample in the batch.
    """
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    images, masks, bone_labels, planes, orig_sizes, paths = [], [], [], [], [], []
    for item in batch:
        img = item["image"]
        msk = item["mask"]
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
            msk = torch.nn.functional.pad(msk, (0, pad_w, 0, pad_h), mode="constant", value=IGNORE_INDEX)
        images.append(img)
        masks.append(msk)
        bone_labels.append(item["bone_label"])
        planes.append(item["plane"])
        orig_sizes.append(item["orig_size"])
        paths.append(item["path"])

    return {
        "image": torch.stack(images, dim=0),
        "mask": torch.stack(masks, dim=0),
        "bone_label": torch.stack(bone_labels, dim=0),
        "plane": planes,
        "orig_size": orig_sizes,
        "path": paths,
    }


if __name__ == "__main__":
    ds = BoneUSDataset(split="train")
    print(f"{len(ds)} samples found across planes {ds.planes}")
    sample = ds[0]
    print("image:", sample["image"].shape, "mask:", sample["mask"].shape,
          "bone_label:", sample["bone_label"], "plane:", sample["plane"])
    print("unique mask values:", torch.unique(sample["mask"]))