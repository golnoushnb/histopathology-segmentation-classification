# TASK 1: TISSUE SEGMENTATION — FINAL PIPELINES

# 0. INSTALL DEPENDENCIES
import subprocess
import sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip_install(
    "rasterio",
    "geopandas",
    "shapely",
    "albumentations",
    "tifffile",
    "opencv-python-headless",
)


# 1. IMPORTS
import os
import json
import glob
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from affine import Affine
from shapely.geometry import shape
from rasterio.features import rasterize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch.amp import autocast, GradScaler
    AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_NEW_API = False

import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# 2. CONFIG
class Config:
    DRIVE_ROOT = "/content/drive/MyDrive"

    DATASET_ROOT = os.path.join(DRIVE_ROOT, "Dataset_Splits")

    TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
    VAL_DIR   = os.path.join(DATASET_ROOT, "validation")
    TEST_DIR  = os.path.join(DATASET_ROOT, "test")

    CACHE_DIR = os.path.join(
        DRIVE_ROOT, "Uni of Edi ", "Computer Vision",
        "Assignment - Task 1", "masks_cache"
    )
    SAVE_DIR = os.path.join(
        DRIVE_ROOT, "Uni of Edi ", "Computer Vision",
        "Assignment - Task 1", "checkpoints"
    )

    PATCH_SIZE  = 512
    TILE_STRIDE = 384
    # more patches per image for better class coverage
    TRAIN_PATCHES_PER_IMAGE = 12

    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE   = 1
    NUM_WORKERS      = 2

    CLASS_MAP: Dict[str, int] = {
        "Tissue Tumor":            1,
        "Tissue Stroma":           2,
        "Tissue Blood Vessel":     0,
        "Tissue Epidermis":        0,
        "Tissue White Background": 0,
        "Tissue Necrosis":         0,
        "tissue_tumor":            1,
        "tissue_stroma":           2,
        "tissue_blood_vessel":     0,
        "tissue_epidermis":        0,
        "tissue_white_background": 0,
        "tissue_necrosis":         0,
    }

    NUM_CLASSES = 3
    SEED        = 42

cfg = Config()

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)


class UNetTrainConfig:
    NUM_EPOCHS   = 120
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    CE_WEIGHT     = 0.5
    DICE_WEIGHT   = 0.5
    LR_PATIENCE   = 4
    LR_FACTOR     = 0.5
    LR_MIN        = 1e-6
    EARLY_STOP_PATIENCE = 30
    USE_AMP       = True
    GRAD_CLIP     = 1.0
    CKPT_PATH     = os.path.join(cfg.SAVE_DIR, "unet_best_patch.pth")

tcfg = UNetTrainConfig()


class AutoencoderTrainConfig:
    AE_EPOCHS       = 50
    AE_LR           = 1e-3
    AE_WEIGHT_DECAY  = 1e-4

    SEG_EPOCHS      = 100
    SEG_LR          = 1e-3          # decoder only (encoder frozen)
    SEG_WEIGHT_DECAY = 1e-4

    CE_WEIGHT   = 0.5
    DICE_WEIGHT = 0.5

    LR_PATIENCE = 4
    LR_FACTOR   = 0.5
    LR_MIN      = 1e-6
    EARLY_STOP_PATIENCE = 30

    USE_AMP   = True
    GRAD_CLIP = 1.0

    AE_CKPT_PATH  = os.path.join(cfg.SAVE_DIR, "autoencoder_stage1_patch.pth")
    SEG_CKPT_PATH = os.path.join(cfg.SAVE_DIR, "autoencoder_seg_best_patch.pth")

aecfg = AutoencoderTrainConfig()


# 3. GOOGLE DRIVE
def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        log.info("Google Drive mounted at /content/drive")
    except ImportError:
        log.warning("Not running in Colab — skipping Drive mount.")


# 4. DATA / MASK UTILITIES
def load_tif(tif_path: str) -> np.ndarray:
    img = tifffile.imread(tif_path)

    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    return img


def geojson_to_mask(
    geojson_path: str,
    image_height: int,
    image_width: int,
    class_map: Dict[str, int],
) -> np.ndarray:
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    with open(geojson_path, "r") as f:
        gj = json.load(f)

    features = gj.get("features", [])
    if not features:
        log.warning(f"No features found in {geojson_path}")
        return mask

    transform     = Affine(1, 0, 0, 0, 1, 0)
    priority_order = [0, 2, 1]
    shapes_by_class: Dict[int, list] = {0: [], 1: [], 2: []}

    for feat in features:
        props          = feat.get("properties", {})
        classification = props.get("classification", {})
        raw_name = (
            classification.get("name", "")
            if isinstance(classification, dict)
            else str(classification)
        )
        label = class_map.get(raw_name, 0)

        geom = shape(feat["geometry"])
        if geom.is_empty or not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_empty:
            shapes_by_class[label].append(geom)

    for cls_id in priority_order:
        geoms = shapes_by_class[cls_id]
        if not geoms:
            continue

        burned = rasterize(
            [(g, cls_id) for g in geoms],
            out_shape=(image_height, image_width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        if cls_id == 0:
            mask[mask == 0] = burned[mask == 0]
        else:
            mask[burned == cls_id] = cls_id

    return mask


def find_pairs(split_dir: str) -> List[Tuple[str, str]]:
    img_dir    = os.path.join(split_dir, "image")
    tissue_dir = os.path.join(split_dir, "tissue")

    if not os.path.isdir(img_dir):
        log.error(f"Expected image/ subfolder not found in {split_dir}")
        return []
    if not os.path.isdir(tissue_dir):
        log.error(f"Expected tissue/ subfolder not found in {split_dir}")
        return []

    tif_files   = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    stem_to_tif = {Path(t).stem: t for t in tif_files}

    pairs = []
    geojson_files = sorted(
        glob.glob(os.path.join(tissue_dir, "*.geojson"))
        + glob.glob(os.path.join(tissue_dir, "*.GeoJSON"))
    )

    for gj in geojson_files:
        gj_stem  = Path(gj).stem
        img_stem = gj_stem[:-len("_tissue")] if gj_stem.endswith("_tissue") else gj_stem

        if img_stem in stem_to_tif:
            pairs.append((stem_to_tif[img_stem], gj))
        else:
            log.warning(f"No matching tif for {Path(gj).name}")

    log.info(f"Paired {len(pairs)}/{len(tif_files)} images with tissue labels in {split_dir}")
    return pairs


def get_or_create_mask(
    tif_path: str,
    geojson_path: str,
    cache_dir: str,
    class_map: Dict[str, int],
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    stem       = Path(tif_path).stem
    cache_path = os.path.join(cache_dir, stem + "_mask.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    img  = load_tif(tif_path)
    h, w = img.shape[:2]
    mask = geojson_to_mask(geojson_path, h, w, class_map)
    np.save(cache_path, mask)
    return mask


def precompute_all_masks(
    pairs: List[Tuple[str, str]],
    cache_dir: str,
    class_map: Dict[str, int],
):
    log.info(f"Precomputing {len(pairs)} masks -> {cache_dir}")
    for i, (tif_path, gj_path) in enumerate(pairs):
        get_or_create_mask(tif_path, gj_path, cache_dir, class_map)
        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{len(pairs)} done")
    log.info("Mask precomputation complete.")


# 5. TRANSFORMS
# No ImageNet normalisation anywhere.
# All three pipelines (seg train, recon train, recon eval) keep
# images in [0,1] so the encoder sees the same distribution in
# Stage 1 and Stage 2.

def get_patch_train_transforms() -> A.Compose:
    """Segmentation training patches — [0,1] output."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # strong geometric augmentation
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ToTensorV2(),
    ])


def get_reconstruction_train_transforms(patch_size: int) -> A.Compose:
    """Autoencoder Stage 1 training — [0,1] output, matches seg pipeline."""
    return A.Compose([
        A.Resize(patch_size, patch_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.03,
            p=0.4,
        ),
        ToTensorV2(),
    ])


def get_reconstruction_eval_transforms(patch_size: int) -> A.Compose:
    """Autoencoder Stage 1 validation — [0,1] output."""
    return A.Compose([
        A.Resize(patch_size, patch_size),
        ToTensorV2(),
    ])


# 6. DATASETS
# patch sampling order and weights.
# Tumor sampled 5x more than Other. Now: 40% Other, 30% each.
PATCH_SAMPLE_CLASS_ORDER   = [0, 1, 2]
PATCH_SAMPLE_CLASS_WEIGHTS = [4.0, 3.0, 3.0]


class TissueSegPatchDataset(Dataset):
    """Patch-based training dataset with stratified class sampling."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        cache_dir: str,
        class_map: Dict[str, int],
        patch_size: int = 512,
        patches_per_image: int = 12,
        transform: Optional[A.Compose] = None,
    ):
        self.pairs             = pairs
        self.cache_dir         = cache_dir
        self.class_map         = class_map
        self.patch_size        = patch_size
        self.patches_per_image = patches_per_image
        self.transform         = transform

        self.index_map = [
            img_idx
            for img_idx in range(len(self.pairs))
            for _ in range(self.patches_per_image)
        ]

    def __len__(self):
        return len(self.index_map)

    def _pad_if_needed(self, image, mask):
        h, w  = image.shape[:2]
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)

        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask  = np.pad(mask,  ((0, pad_h), (0, pad_w)),          mode="constant", constant_values=0)

        return image, mask

    def _sample_patch_coords(self, mask):
        h, w = mask.shape
        ps   = self.patch_size

        if h <= ps or w <= ps:
            return 0, 0

        #class sampling
        target_class = random.choices(
            PATCH_SAMPLE_CLASS_ORDER,
            weights=PATCH_SAMPLE_CLASS_WEIGHTS,
            k=1,
        )[0]

        ys, xs = np.where(mask == target_class)

        if len(xs) == 0:
            ys, xs = np.where(mask != 255)

        if len(xs) > 0:
            idx    = random.randint(0, len(xs) - 1)
            cx, cy = xs[idx], ys[idx]
            x0 = max(0, min(cx - ps // 2, w - ps))
            y0 = max(0, min(cy - ps // 2, h - ps))
            return x0, y0

        return random.randint(0, w - ps), random.randint(0, h - ps)

    def __getitem__(self, idx):
        img_idx           = self.index_map[idx]
        tif_path, gj_path = self.pairs[img_idx]

        image = load_tif(tif_path)
        mask  = get_or_create_mask(tif_path, gj_path, self.cache_dir, self.class_map)

        image, mask = self._pad_if_needed(image, mask)

        x0, y0 = self._sample_patch_coords(mask)
        ps     = self.patch_size

        image_patch = image[y0:y0+ps, x0:x0+ps]
        mask_patch  = mask[y0:y0+ps,  x0:x0+ps]

        if self.transform is not None:
            augmented   = self.transform(image=image_patch, mask=mask_patch)
            image_patch = augmented["image"]
            mask_patch  = augmented["mask"]
        else:
            # manual path also keeps [0,1]
            image_patch = torch.from_numpy(image_patch.transpose(2, 0, 1)).float() / 255.0
            mask_patch  = torch.from_numpy(mask_patch)

        # ToTensorV2 on uint8 produces [0,255] int tensor — divide to [0,1]
        if image_patch.dtype == torch.uint8:
            image_patch = image_patch.float() / 255.0
        elif image_patch.max() > 1.0:
            image_patch = image_patch.float() / 255.0

        return image_patch, mask_patch.long()


class TissueSegFullImageDataset(Dataset):
    """Full-image dataset for tiled validation/test inference."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        cache_dir: str,
        class_map: Dict[str, int],
    ):
        self.pairs     = pairs
        self.cache_dir = cache_dir
        self.class_map = class_map

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tif_path, gj_path = self.pairs[idx]
        image = load_tif(tif_path)
        mask  = get_or_create_mask(tif_path, gj_path, self.cache_dir, self.class_map)
        return image, mask, Path(tif_path).stem


class TissueReconDataset(Dataset):
    """Autoencoder Stage 1 dataset — [0,1] images, no label needed."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        patch_size: int = 512,
        transform: Optional[A.Compose] = None,
    ):
        self.pairs      = pairs
        self.patch_size = patch_size
        self.transform  = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tif_path, _ = self.pairs[idx]
        image       = load_tif(tif_path)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image     = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # ensure [0,1] regardless of path taken
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.is_floating_point() and image.max() > 1.0:
            image = image / 255.0

        return image, torch.tensor(0, dtype=torch.long)


# 7. DATALOADERS
def build_segmentation_dataloaders(cfg: Config, cache_masks: bool = True):
    train_pairs = find_pairs(cfg.TRAIN_DIR)
    val_pairs   = find_pairs(cfg.VAL_DIR)
    test_pairs  = find_pairs(cfg.TEST_DIR)

    assert len(train_pairs) > 0, "No training pairs found."

    if cache_masks:
        for split_pairs, split_name in [
            (train_pairs, "train"),
            (val_pairs,   "validation"),
            (test_pairs,  "test"),
        ]:
            split_cache = os.path.join(cfg.CACHE_DIR, split_name)
            precompute_all_masks(split_pairs, split_cache, cfg.CLASS_MAP)

    train_cache = os.path.join(cfg.CACHE_DIR, "train")
    val_cache   = os.path.join(cfg.CACHE_DIR, "validation")
    test_cache  = os.path.join(cfg.CACHE_DIR, "test")

    train_ds = TissueSegPatchDataset(
        train_pairs, train_cache, cfg.CLASS_MAP,
        patch_size=cfg.PATCH_SIZE,
        patches_per_image=cfg.TRAIN_PATCHES_PER_IMAGE,
        transform=get_patch_train_transforms(),
    )
    val_ds  = TissueSegFullImageDataset(val_pairs,  val_cache,  cfg.CLASS_MAP)
    test_ds = TissueSegFullImageDataset(test_pairs, test_cache, cfg.CLASS_MAP)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    print(
        f"Segmentation loaders ready - train patches={len(train_ds)}, "
        f"val images={len(val_ds)}, test images={len(test_ds)}"
    )
    return train_loader, val_loader, test_loader


def build_reconstruction_dataloaders(cfg: Config):
    train_pairs = find_pairs(cfg.TRAIN_DIR)
    val_pairs   = find_pairs(cfg.VAL_DIR)

    train_ds = TissueReconDataset(
        train_pairs, patch_size=cfg.PATCH_SIZE,
        transform=get_reconstruction_train_transforms(cfg.PATCH_SIZE),
    )
    val_ds = TissueReconDataset(
        val_pairs, patch_size=cfg.PATCH_SIZE,
        transform=get_reconstruction_eval_transforms(cfg.PATCH_SIZE),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    print(f"Reconstruction loaders ready - train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader



# 8. MODELS
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# larger base_channels (48 vs 32) for more capacity
class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 48):
        super().__init__()
        c = base_channels
        f = [c, c*2, c*4, c*8, c*16]

        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(f[3], f[4])

        self.up4  = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f[3] + f[3], f[3])

        self.up3  = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f[2] + f[2], f[2])

        self.up2  = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f[1] + f[1], f[1])

        self.up1  = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f[0] + f[0], f[0])

        self.output_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b  = self.bottleneck(self.pool(s4))

        d4 = self.dec4(torch.cat([self.up4(b),  s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.output_conv(d1)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 48):
        super().__init__()
        c = base_channels
        f = [c, c*2, c*4, c*8, c*16]

        self.enc1       = DoubleConv(in_channels, f[0])
        self.enc2       = DoubleConv(f[0], f[1])
        self.enc3       = DoubleConv(f[1], f[2])
        self.enc4       = DoubleConv(f[2], f[3])
        self.bottleneck = DoubleConv(f[3], f[4])
        self.pool       = nn.MaxPool2d(2, 2)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b  = self.bottleneck(self.pool(s4))
        return b, [s1, s2, s3, s4]


class ReconDecoder(nn.Module):
    def __init__(self, base_channels: int = 48):
        super().__init__()
        c = base_channels
        f = [c, c*2, c*4, c*8, c*16]

        self.up4  = nn.ConvTranspose2d(f[4], f[3], 2, 2)
        self.dec4 = DoubleConv(f[3], f[3])

        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.dec3 = DoubleConv(f[2], f[2])

        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.dec2 = DoubleConv(f[1], f[1])

        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.dec1 = DoubleConv(f[0], f[0])

        self.output_conv = nn.Sequential(
            nn.Conv2d(f[0], 3, kernel_size=1),
            nn.Sigmoid(),   # output in [0,1] matching input
        )

    def forward(self, b):
        x = self.dec4(self.up4(b))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.output_conv(x)


class Autoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 48):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = ReconDecoder(base_channels)

    def forward(self, x):
        b, _ = self.encoder(x)
        return self.decoder(b)


class SegDecoder(nn.Module):
    def __init__(self, num_classes: int = 3, base_channels: int = 48):
        super().__init__()
        c = base_channels
        f = [c, c*2, c*4, c*8, c*16]

        self.up4  = nn.ConvTranspose2d(f[4], f[3], 2, 2)
        self.dec4 = DoubleConv(f[3] + f[3], f[3])

        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.dec3 = DoubleConv(f[2] + f[2], f[2])

        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.dec2 = DoubleConv(f[1] + f[1], f[1])

        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.dec1 = DoubleConv(f[0] + f[0], f[0])

        self.output_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, bottleneck, skips):
        s1, s2, s3, s4 = skips
        d4 = self.dec4(torch.cat([self.up4(bottleneck), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.output_conv(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 9. LOSSES / METRICS

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        dims         = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        denom        = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        # no label_smoothing as it hurts boundary pixels
        self.ce          = nn.CrossEntropyLoss(weight=class_weights)
        self.dice        = DiceLoss()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def compute_class_weights_from_masks(
    pairs: List[Tuple[str, str]],
    cache_dir: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    FIX 12: Compute weights from raw cached masks, not from the patch
    loader. The patch loader is biased by the stratified sampler, so
    computing weights from it double-corrects for class imbalance.
    Reading masks directly gives the true pixel distribution.

    FIX 2: Raw inverse-frequency, no sqrt dampening, capped at 6x.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)

    for tif_path, _ in pairs:
        stem       = Path(tif_path).stem
        cache_path = os.path.join(cache_dir, stem + "_mask.npy")
        if not os.path.exists(cache_path):
            continue
        mask = np.load(cache_path)
        for cls in range(num_classes):
            counts[cls] += int((mask == cls).sum())

    counts  = torch.clamp(counts, min=1.0)
    total   = counts.sum()
    weights = total / (num_classes * counts)

    # Cap at 6x so no single class wildly dominates
    weights = torch.clamp(weights, max=6.0)
    weights = weights / weights.sum() * num_classes

    print(
        f"True pixel class weights  - Other: {weights[0]:.3f} | "
        f"Tumor: {weights[1]:.3f} | Stroma: {weights[2]:.3f}"
    )
    return weights.float().to(device)


def harmonic_mean_dice(dice_other: float, dice_tumor: float, dice_stroma: float) -> float:
    scores = [max(s, 1e-6) for s in [dice_other, dice_tumor, dice_stroma]]
    return len(scores) / sum(1.0 / s for s in scores)


def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    preds       = logits.argmax(dim=1)
    dice_scores = []

    for cls in range(num_classes):
        pred_cls     = (preds == cls).float()
        target_cls   = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union        = pred_cls.sum() + target_cls.sum()

        if union == 0:
            dice_scores.append(torch.tensor(1.0, device=logits.device))
        else:
            dice_scores.append((2 * intersection + 1e-5) / (union + 1e-5))

    d_other  = dice_scores[0].item()
    d_tumor  = dice_scores[1].item()
    d_stroma = dice_scores[2].item()

    return {
        "dice_other":  d_other,
        "dice_tumor":  d_tumor,
        "dice_stroma": d_stroma,
        # checkpoint on arithmetic mean; harmonic shown for info only
        "mean_dice":   float(np.mean([d_other, d_tumor, d_stroma])),
        "harm_dice":   harmonic_mean_dice(d_other, d_tumor, d_stroma),
        "pixel_acc":   (preds == targets).float().mean().item(),
    }


# 10. INFERENCE HELPERS
def amp_autocast(enabled=True):
    if AMP_NEW_API:
        return autocast("cuda", enabled=enabled and torch.cuda.is_available())
    return autocast(enabled=enabled and torch.cuda.is_available())


def patch_to_tensor(patch_uint8: np.ndarray) -> torch.Tensor:
    """
    FIX 10: convert patch to [0,1] float tensor with no ImageNet normalisation,
    consistent with how training patches are prepared.
    """
    patch = patch_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)


@torch.no_grad()
def predict_full_image_tiled(
    model, image_np: np.ndarray, device,
    num_classes=3, tile_size=512, stride=384,
):
    h, w  = image_np.shape[:2]
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h > 0 or pad_w > 0:
        image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    H, W       = image_np.shape[:2]
    logits_sum = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    count_map  = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    ys = list(range(0, max(1, H - tile_size + 1), stride))
    xs = list(range(0, max(1, W - tile_size + 1), stride))
    if ys[-1] != H - tile_size:
        ys.append(H - tile_size)
    if xs[-1] != W - tile_size:
        xs.append(W - tile_size)

    model.eval()

    for y0 in ys:
        for x0 in xs:
            patch        = image_np[y0:y0+tile_size, x0:x0+tile_size]
            patch_tensor = patch_to_tensor(patch).to(device)
            with amp_autocast(enabled=False):
                logits = model(patch_tensor)[0]
            logits_sum[:, y0:y0+tile_size, x0:x0+tile_size] += logits
            count_map[:,  y0:y0+tile_size, x0:x0+tile_size] += 1.0

    return (logits_sum / count_map)[:, :h, :w].unsqueeze(0)


@torch.no_grad()
def predict_full_image_tiled_ae(
    encoder, seg_decoder, image_np: np.ndarray, device,
    num_classes=3, tile_size=512, stride=384,
):
    h, w  = image_np.shape[:2]
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h > 0 or pad_w > 0:
        image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    H, W       = image_np.shape[:2]
    logits_sum = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    count_map  = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    ys = list(range(0, max(1, H - tile_size + 1), stride))
    xs = list(range(0, max(1, W - tile_size + 1), stride))
    if ys[-1] != H - tile_size:
        ys.append(H - tile_size)
    if xs[-1] != W - tile_size:
        xs.append(W - tile_size)

    encoder.eval()
    seg_decoder.eval()

    for y0 in ys:
        for x0 in xs:
            patch        = image_np[y0:y0+tile_size, x0:x0+tile_size]
            patch_tensor = patch_to_tensor(patch).to(device)
            b, skips     = encoder(patch_tensor)
            logits       = seg_decoder(b, skips)[0]
            logits_sum[:, y0:y0+tile_size, x0:x0+tile_size] += logits
            count_map[:,  y0:y0+tile_size, x0:x0+tile_size] += 1.0

    return (logits_sum / count_map)[:, :h, :w].unsqueeze(0)


# 11. TRAINING HELPERS
def save_checkpoint(payload: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    print(f"Checkpoint saved -> {path}")


def plot_unet_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("UNet Combined Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_dice"],   label="Train")
    axes[1].plot(history["val_dice"],     label="Val")
    axes[1].plot(history["ema_val_dice"], label="EMA Val", linestyle="--")
    axes[1].axhline(y=0.467, color="r", linestyle="--", label="Baseline")
    axes[1].set_title("UNet Arithmetic Mean Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_dice_other"],  label="Other")
    axes[2].plot(history["val_dice_tumor"],  label="Tumor")
    axes[2].plot(history["val_dice_stroma"], label="Stroma")
    axes[2].set_title("UNet Val Dice per Class")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ae_stage1_curves(history: dict, save_path: str):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train MSE")
    plt.plot(history["val_loss"],   label="Val MSE")
    plt.title("Stage 1 Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ae_stage2_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("AE Stage 2 Combined Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_dice"],   label="Train")
    axes[1].plot(history["val_dice"],     label="Val")
    axes[1].plot(history["ema_val_dice"], label="EMA Val", linestyle="--")
    axes[1].axhline(y=0.467, color="r", linestyle="--", label="Baseline")
    axes[1].set_title("AE Stage 2 Arithmetic Mean Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_dice_other"],  label="Other")
    axes[2].plot(history["val_dice_tumor"],  label="Tumor")
    axes[2].plot(history["val_dice_stroma"], label="Stroma")
    axes[2].set_title("AE Stage 2 Val Dice per Class")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# 12. UNET TRAINING
def train_unet_one_epoch(model, loader, criterion, optimiser, scaler, device, num_classes):
    model.train()
    total_loss  = 0.0
    metrics_sum = {
        "dice_other": 0, "dice_tumor": 0, "dice_stroma": 0,
        "mean_dice": 0, "harm_dice": 0, "pixel_acc": 0,
    }

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        with amp_autocast(enabled=tcfg.USE_AMP):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.GRAD_CLIP)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            m = compute_batch_metrics(logits.detach(), masks, num_classes)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


@torch.no_grad()
def validate_tiled_unet(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss  = 0.0
    metrics_sum = {
        "dice_other": 0.0, "dice_tumor": 0.0, "dice_stroma": 0.0,
        "mean_dice": 0.0, "harm_dice": 0.0, "pixel_acc": 0.0,
    }
    n = len(loader)

    for image_np, mask_np, _ in loader:
        image_np = image_np[0].numpy()
        mask_np  = mask_np[0].numpy()

        logits  = predict_full_image_tiled(
            model, image_np, device=device,
            num_classes=num_classes,
            tile_size=cfg.PATCH_SIZE, stride=cfg.TILE_STRIDE,
        )
        targets = torch.from_numpy(mask_np).long().unsqueeze(0).to(device)
        loss    = criterion(logits, targets)
        total_loss += loss.item()

        m = compute_batch_metrics(logits, targets, num_classes)
        for k in metrics_sum:
            metrics_sum[k] += m[k]

    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training UNet on: {device}")

    train_loader, val_loader, _ = build_segmentation_dataloaders(cfg)

    model = UNet(in_channels=3, num_classes=cfg.NUM_CLASSES, base_channels=48).to(device)
    print(f"UNet trainable parameters: {count_parameters(model):,}")

    train_pairs = find_pairs(cfg.TRAIN_DIR)
    train_cache = os.path.join(cfg.CACHE_DIR, "train")
    # compute weights from raw masks, not from biased patch loader
    class_weights = compute_class_weights_from_masks(
        train_pairs, train_cache, cfg.NUM_CLASSES, device
    )
    criterion = CombinedLoss(class_weights, tcfg.CE_WEIGHT, tcfg.DICE_WEIGHT).to(device)

    optimiser = optim.Adam(model.parameters(), lr=tcfg.LEARNING_RATE, weight_decay=tcfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="max",
        factor=tcfg.LR_FACTOR, patience=tcfg.LR_PATIENCE, min_lr=tcfg.LR_MIN,
    )
    scaler = (
        GradScaler("cuda", enabled=tcfg.USE_AMP and torch.cuda.is_available())
        if AMP_NEW_API else
        GradScaler(enabled=tcfg.USE_AMP and torch.cuda.is_available())
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "val_dice_other": [], "val_dice_tumor": [], "val_dice_stroma": [],
        "val_harm_dice": [], "ema_val_dice": [],
    }

    best_val_dice     = 0.0
    epochs_no_improve = 0
    # EMA smoothing for LR scheduler — absorbs spikes from small val set
    ema_val_dice = 0.0
    EMA_ALPHA    = 0.25

    for epoch in range(1, tcfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_metrics = train_unet_one_epoch(
            model, train_loader, criterion, optimiser, scaler, device, cfg.NUM_CLASSES
        )
        val_loss, val_metrics = validate_tiled_unet(
            model, val_loader, criterion, device, cfg.NUM_CLASSES
        )

        # smooth val signal before handing to scheduler
        ema_val_dice = EMA_ALPHA * val_metrics["mean_dice"] + (1 - EMA_ALPHA) * ema_val_dice
        scheduler.step(ema_val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_metrics["mean_dice"])
        history["val_dice"].append(val_metrics["mean_dice"])
        history["val_dice_other"].append(val_metrics["dice_other"])
        history["val_dice_tumor"].append(val_metrics["dice_tumor"])
        history["val_dice_stroma"].append(val_metrics["dice_stroma"])
        history["val_harm_dice"].append(val_metrics["harm_dice"])
        history["ema_val_dice"].append(ema_val_dice)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{tcfg.NUM_EPOCHS} | "
            f"Time: {elapsed:.0f}s | "
            f"LR: {optimiser.param_groups[0]['lr']:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_metrics['mean_dice']:.4f} "
            f"(EMA {ema_val_dice:.4f} | HM {val_metrics['harm_dice']:.4f}) "
            f"[Other: {val_metrics['dice_other']:.3f} "
            f"Tumor: {val_metrics['dice_tumor']:.3f} "
            f"Stroma: {val_metrics['dice_stroma']:.3f}]"
        )

        # checkpoint on arithmetic mean Dice
        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            save_checkpoint({
                "epoch": epoch,
                "val_dice": best_val_dice,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
            }, tcfg.CKPT_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= tcfg.EARLY_STOP_PATIENCE:
            print("Early stopping triggered for UNet.")
            break

    print(f"Best UNet validation Arithmetic-Mean Dice: {best_val_dice:.4f}")
    plot_unet_curves(history, os.path.join(cfg.SAVE_DIR, "unet_training_curves_patch.png"))
    return model, history


# 13. AUTOENCODER TRAINING
def train_autoencoder_stage1(train_loader, val_loader, device):
    print("STAGE 1 : Autoencoder pretraining")
    model = Autoencoder(in_channels=3, base_channels=48).to(device)
    print(f"Autoencoder parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=aecfg.AE_LR, weight_decay=aecfg.AE_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="min",
        factor=aecfg.LR_FACTOR, patience=aecfg.LR_PATIENCE, min_lr=aecfg.LR_MIN,
    )
    scaler = (
        GradScaler("cuda", enabled=aecfg.USE_AMP and torch.cuda.is_available())
        if AMP_NEW_API else
        GradScaler(enabled=aecfg.USE_AMP and torch.cuda.is_available())
    )

    history           = {"train_loss": [], "val_loss": []}
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, aecfg.AE_EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with amp_autocast(enabled=aecfg.USE_AMP):
                recon = model(images)
                loss  = criterion(recon, images)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), aecfg.GRAD_CLIP)
            scaler.step(optimiser)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with amp_autocast(enabled=aecfg.USE_AMP):
                    recon = model(images)
                    loss  = criterion(recon, images)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{aecfg.AE_EPOCHS} | "
            f"Time: {elapsed:.0f}s | "
            f"LR: {optimiser.param_groups[0]['lr']:.2e} | "
            f"Train MSE: {train_loss:.5f} | "
            f"Val MSE: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
            }, aecfg.AE_CKPT_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= aecfg.EARLY_STOP_PATIENCE:
            print("Early stopping Stage 1.")
            break

    print(f"Best Stage 1 val MSE: {best_val_loss:.5f}")
    plot_ae_stage1_curves(history, os.path.join(cfg.SAVE_DIR, "stage1_curves_patch.png"))
    return model


@torch.no_grad()
def validate_tiled_ae(encoder, seg_decoder, loader, criterion, device, num_classes):
    total_loss  = 0.0
    metrics_sum = {
        "dice_other": 0.0, "dice_tumor": 0.0, "dice_stroma": 0.0,
        "mean_dice": 0.0, "harm_dice": 0.0, "pixel_acc": 0.0,
    }
    n = len(loader)

    for image_np, mask_np, _ in loader:
        image_np = image_np[0].numpy()
        mask_np  = mask_np[0].numpy()

        logits  = predict_full_image_tiled_ae(
            encoder, seg_decoder, image_np, device=device,
            num_classes=num_classes,
            tile_size=cfg.PATCH_SIZE, stride=cfg.TILE_STRIDE,
        )
        targets = torch.from_numpy(mask_np).long().unsqueeze(0).to(device)
        loss    = criterion(logits, targets)
        total_loss += loss.item()

        m = compute_batch_metrics(logits, targets, num_classes)
        for k in metrics_sum:
            metrics_sum[k] += m[k]

    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


def train_autoencoder_stage2(autoencoder, train_loader, val_loader, device):
    """
    Encoder is FROZEN as required by the coursework specification.
    Only the segmentation decoder is trained.
    """
    print("STAGE 2 : Fixed encoder + segmentation decoder (coursework spec)")

    encoder     = autoencoder.encoder.to(device)
    seg_decoder = SegDecoder(num_classes=cfg.NUM_CLASSES, base_channels=48).to(device)

    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = count_parameters(seg_decoder)
    print(f"Frozen encoder params:    {enc_params:,}")
    print(f"Trainable decoder params: {dec_params:,}")

    train_pairs = find_pairs(cfg.TRAIN_DIR)
    train_cache = os.path.join(cfg.CACHE_DIR, "train")
    # weights from raw masks
    class_weights = compute_class_weights_from_masks(
        train_pairs, train_cache, cfg.NUM_CLASSES, device
    )
    criterion = CombinedLoss(class_weights, aecfg.CE_WEIGHT, aecfg.DICE_WEIGHT).to(device)

    # Only decoder parameters are optimised
    optimiser = optim.Adam(
        seg_decoder.parameters(),
        lr=aecfg.SEG_LR,
        weight_decay=aecfg.SEG_WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimiser, mode="max",
        factor=aecfg.LR_FACTOR, patience=aecfg.LR_PATIENCE, min_lr=aecfg.LR_MIN,
    )
    scaler = (
        GradScaler("cuda", enabled=aecfg.USE_AMP and torch.cuda.is_available())
        if AMP_NEW_API else
        GradScaler(enabled=aecfg.USE_AMP and torch.cuda.is_available())
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "val_dice_other": [], "val_dice_tumor": [], "val_dice_stroma": [],
        "val_harm_dice": [], "ema_val_dice": [],
    }

    best_val_dice     = 0.0
    epochs_no_improve = 0
    ema_val_dice      = 0.0
    EMA_ALPHA         = 0.25

    for epoch in range(1, aecfg.SEG_EPOCHS + 1):
        t0 = time.time()

        seg_decoder.train()
        # encoder stays eval/frozen throughout
        encoder.eval()

        train_loss    = 0.0
        train_metrics = {
            "dice_other": 0, "dice_tumor": 0, "dice_stroma": 0,
            "mean_dice": 0, "harm_dice": 0, "pixel_acc": 0,
        }

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            optimiser.zero_grad(set_to_none=True)

            # Encoder is frozen — no gradients needed
            with torch.no_grad():
                bottleneck, skips = encoder(images)

            with amp_autocast(enabled=aecfg.USE_AMP):
                logits = seg_decoder(bottleneck, skips)
                loss   = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(seg_decoder.parameters(), aecfg.GRAD_CLIP)
            scaler.step(optimiser)
            scaler.update()

            train_loss += loss.item()

            with torch.no_grad():
                m = compute_batch_metrics(logits.detach(), masks, cfg.NUM_CLASSES)
                for k in train_metrics:
                    train_metrics[k] += m[k]

        val_loss, val_metrics = validate_tiled_ae(
            encoder, seg_decoder, val_loader, criterion, device, cfg.NUM_CLASSES
        )

        n_train       = len(train_loader)
        train_loss   /= n_train
        train_metrics = {k: v / n_train for k, v in train_metrics.items()}

        ema_val_dice = EMA_ALPHA * val_metrics["mean_dice"] + (1 - EMA_ALPHA) * ema_val_dice
        scheduler.step(ema_val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_metrics["mean_dice"])
        history["val_dice"].append(val_metrics["mean_dice"])
        history["val_dice_other"].append(val_metrics["dice_other"])
        history["val_dice_tumor"].append(val_metrics["dice_tumor"])
        history["val_dice_stroma"].append(val_metrics["dice_stroma"])
        history["val_harm_dice"].append(val_metrics["harm_dice"])
        history["ema_val_dice"].append(ema_val_dice)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{aecfg.SEG_EPOCHS} | "
            f"Time: {elapsed:.0f}s | "
            f"LR: {optimiser.param_groups[0]['lr']:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_metrics['mean_dice']:.4f} "
            f"(EMA {ema_val_dice:.4f} | HM {val_metrics['harm_dice']:.4f}) "
            f"[Other: {val_metrics['dice_other']:.3f} "
            f"Tumor: {val_metrics['dice_tumor']:.3f} "
            f"Stroma: {val_metrics['dice_stroma']:.3f}]"
        )

        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            save_checkpoint({
                "epoch": epoch,
                "val_dice": best_val_dice,
                "encoder_state":     encoder.state_dict(),
                "seg_decoder_state": seg_decoder.state_dict(),
                "optimiser_state":   optimiser.state_dict(),
            }, aecfg.SEG_CKPT_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= aecfg.EARLY_STOP_PATIENCE:
            print("Early stopping Stage 2.")
            break

    print(f"Best AE Stage 2 validation Arithmetic-Mean Dice: {best_val_dice:.4f}")
    plot_ae_stage2_curves(history, os.path.join(cfg.SAVE_DIR, "stage2_curves_patch.png"))
    return encoder, seg_decoder, history


def visualise_reconstructions(autoencoder, val_loader, device, n=4):
    autoencoder.eval()
    images, _ = next(iter(val_loader))
    images    = images[:n].to(device)

    with torch.no_grad():
        recon = autoencoder(images)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, i].set_title(f"Recon {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def train_autoencoder_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Autoencoder pipeline on: {device}")

    recon_train_loader, recon_val_loader  = build_reconstruction_dataloaders(cfg)
    seg_train_loader,   seg_val_loader, _ = build_segmentation_dataloaders(cfg)

    autoencoder = train_autoencoder_stage1(recon_train_loader, recon_val_loader, device)

    print("Visualising Stage 1 reconstructions:")
    visualise_reconstructions(autoencoder, recon_val_loader, device, n=4)

    encoder, seg_decoder, history = train_autoencoder_stage2(
        autoencoder, seg_train_loader, seg_val_loader, device
    )

    return autoencoder, encoder, seg_decoder, history


# 14. EVALUATION / VISUALISATION
def mask_to_rgb(mask_np):
    colour = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    colour[mask_np == 0] = [0.6, 0.6, 0.6]
    colour[mask_np == 1] = [0.9, 0.1, 0.1]
    colour[mask_np == 2] = [0.1, 0.3, 0.9]
    return colour


def _global_eval(model_name, predict_fn, loader, device, num_classes=3):
    """Shared global-accumulation evaluation used for both UNet and AE."""
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    pred_totals   = torch.zeros(num_classes, dtype=torch.float64)
    target_totals = torch.zeros(num_classes, dtype=torch.float64)
    union_totals  = torch.zeros(num_classes, dtype=torch.float64)
    correct_px    = 0.0
    total_px      = 0.0

    for image_np, mask_np, _ in loader:
        image_np = image_np[0].numpy()
        mask_np  = mask_np[0].numpy()

        logits = predict_fn(image_np)
        preds  = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        correct_px += (preds == mask_np).sum()
        total_px   += mask_np.size

        for cls in range(num_classes):
            p = preds == cls
            t = mask_np == cls
            intersections[cls] += np.logical_and(p, t).sum()
            pred_totals[cls]   += p.sum()
            target_totals[cls] += t.sum()
            union_totals[cls]  += np.logical_or(p, t).sum()

    dice_scores, iou_scores = [], []
    for cls in range(num_classes):
        dice_scores.append(
            float((2 * intersections[cls] + 1e-5) / (pred_totals[cls] + target_totals[cls] + 1e-5))
        )
        iou_scores.append(
            float((intersections[cls] + 1e-5) / (union_totals[cls] + 1e-5))
        )

    results = {
        "dice_other":         dice_scores[0],
        "dice_tumor":         dice_scores[1],
        "dice_stroma":        dice_scores[2],
        "mean_dice":          float(np.mean(dice_scores)),
        "harmonic_mean_dice": harmonic_mean_dice(*dice_scores),
        "iou_other":          iou_scores[0],
        "iou_tumor":          iou_scores[1],
        "iou_stroma":         iou_scores[2],
        "mean_iou":           float(np.mean(iou_scores)),
        "pixel_acc":          correct_px / total_px,
    }

    print(f"\n{'='*60}")
    print(f"{model_name} - Global tiled results")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"{k:<22}: {v:.4f}")

    return results


def evaluate_global_tiled(model_name, model, loader, device, num_classes=3):
    model.eval()
    return _global_eval(
        model_name,
        lambda img: predict_full_image_tiled(
            model, img, device=device, num_classes=num_classes,
            tile_size=cfg.PATCH_SIZE, stride=cfg.TILE_STRIDE,
        ),
        loader, device, num_classes,
    )


def evaluate_global_tiled_ae(model_name, encoder, seg_decoder, loader, device, num_classes=3):
    encoder.eval()
    seg_decoder.eval()
    return _global_eval(
        model_name,
        lambda img: predict_full_image_tiled_ae(
            encoder, seg_decoder, img, device=device, num_classes=num_classes,
            tile_size=cfg.PATCH_SIZE, stride=cfg.TILE_STRIDE,
        ),
        loader, device, num_classes,
    )


def save_qualitative_tiled(model_name, model, loader, save_dir, device, max_items=4):
    legend_patches = [
        mpatches.Patch(color=[0.6, 0.6, 0.6], label="Other (0)"),
        mpatches.Patch(color=[0.9, 0.1, 0.1], label="Tumor (1)"),
        mpatches.Patch(color=[0.1, 0.3, 0.9], label="Stroma (2)"),
    ]

    items = []
    for i, (image_np, mask_np, stem) in enumerate(loader):
        if i >= max_items:
            break
        items.append((image_np[0].numpy(), mask_np[0].numpy(), stem[0]))

    n         = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (image_np, mask_np, stem) in enumerate(items):
        logits  = predict_full_image_tiled(
            model, image_np, device=device, num_classes=cfg.NUM_CLASSES,
            tile_size=cfg.PATCH_SIZE, stride=cfg.TILE_STRIDE,
        )
        pred_np = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Input: {stem}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_to_rgb(mask_np))
        axes[i, 1].set_title("Ground Truth", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_to_rgb(pred_np))
        axes[i, 2].set_title("Prediction", fontsize=9)
        axes[i, 2].axis("off")

    axes[0, 2].legend(handles=legend_patches, loc="upper right", fontsize=8)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_tiled_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {path}")


def save_qualitative_tiled_ae(model_name, encoder, seg_decoder, loader, save_dir, device, max_items=4):
    legend_patches = [
        mpatches.Patch(color=[0.6, 0.6, 0.6], label="Other (0)"),
        mpatches.Patch(color=[0.9, 0.1, 0.1], label="Tumor (1)"),
        mpatches.Patch(color=[0.1, 0.3, 0.9], label="Stroma (2)"),
    ]

    items = []
    for i, (image_np, mask_np, stem) in enumerate(loader):
        if i >= max_items:
            break
        items.append((image_np[0].numpy(), mask_np[0].numpy(), stem[0]))

    n         = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (image_np, mask_np, stem) in enumerate(items):
        logits  = predict_full_image_tiled_ae(
            encoder, seg_decoder, image_np, device=device,
            num_classes=cfg.NUM_CLASSES,
            tile_size=cfg.PATCH_SIZE, stride=cfg.TILE_STRIDE,
        )
        pred_np = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Input: {stem}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_to_rgb(mask_np))
        axes[i, 1].set_title("Ground Truth", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_to_rgb(pred_np))
        axes[i, 2].set_title("Prediction", fontsize=9)
        axes[i, 2].axis("off")

    axes[0, 2].legend(handles=legend_patches, loc="upper right", fontsize=8)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_tiled_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {path}")


def load_trained_models_for_evaluation(device):
    unet_ckpt = torch.load(tcfg.CKPT_PATH, map_location=device)
    unet      = UNet(in_channels=3, num_classes=cfg.NUM_CLASSES, base_channels=48).to(device)
    unet.load_state_dict(unet_ckpt["model_state"])
    unet.eval()

    seg_ckpt    = torch.load(aecfg.SEG_CKPT_PATH, map_location=device)
    encoder     = Encoder(in_channels=3, base_channels=48).to(device)
    seg_decoder = SegDecoder(num_classes=cfg.NUM_CLASSES, base_channels=48).to(device)
    encoder.load_state_dict(seg_ckpt["encoder_state"])
    seg_decoder.load_state_dict(seg_ckpt["seg_decoder_state"])
    encoder.eval()
    seg_decoder.eval()

    print(f"UNet loaded   - epoch {unet_ckpt['epoch']} | val Dice {unet_ckpt['val_dice']:.4f}")
    print(f"AE+Seg loaded - epoch {seg_ckpt['epoch']}  | val Dice {seg_ckpt['val_dice']:.4f}")

    return unet, encoder, seg_decoder


def run_full_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = build_segmentation_dataloaders(cfg)

    unet, encoder, seg_decoder = load_trained_models_for_evaluation(device)

    unet_results = evaluate_global_tiled(
        "UNet", unet, test_loader, device, cfg.NUM_CLASSES
    )
    ae_results = evaluate_global_tiled_ae(
        "Autoencoder + SegDecoder", encoder, seg_decoder, test_loader, device, cfg.NUM_CLASSES
    )

    print(f"\nBaseline mean Dice (arithmetic): 0.4670")
    print(f"UNet     mean Dice (arithmetic): {unet_results['mean_dice']:.4f} | harmonic: {unet_results['harmonic_mean_dice']:.4f}")
    print(f"AE+Seg   mean Dice (arithmetic): {ae_results['mean_dice']:.4f} | harmonic: {ae_results['harmonic_mean_dice']:.4f}")

    save_qualitative_tiled("UNet_validation",          unet,    val_loader,  cfg.SAVE_DIR, device)
    save_qualitative_tiled_ae("Autoencoder_validation", encoder, seg_decoder, val_loader,  cfg.SAVE_DIR, device)
    save_qualitative_tiled("UNet_test",                unet,    test_loader, cfg.SAVE_DIR, device)
    save_qualitative_tiled_ae("Autoencoder_test",       encoder, seg_decoder, test_loader, cfg.SAVE_DIR, device)

    return unet_results, ae_results


# 15. MAIN
if __name__ == "__main__":
    mount_drive()

    print("\nBuilding loaders / caching masks...")
    train_loader, val_loader, test_loader = build_segmentation_dataloaders(cfg)
    recon_train_loader, recon_val_loader  = build_reconstruction_dataloaders(cfg)

    print("TRAIN UNET ")
    unet_model, unet_history = train_unet()

    print("TRAIN AUTOENCODER PIPELINE")
    autoencoder, encoder, seg_decoder, ae_history = train_autoencoder_pipeline()

    print("FULL EVALUATION ")
    unet_results, ae_results = run_full_evaluation()

    print("\nAll done.")