import os
import json
import copy
import random
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import tifffile as tiff
from PIL import Image
from shapely.geometry import shape

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from tqdm.auto import tqdm



# Paths
ROOT_DIR = "/content/drive/MyDrive/computer_vision_project"
RAW_ROOT = os.path.join(ROOT_DIR, "data", "Dataset_Splits")
TASK2_TEST_DIR = os.path.join(ROOT_DIR, "data", "Task2_Test_Set")

TRAIN_IMG_DIR = os.path.join(RAW_ROOT, "train", "image")
TRAIN_NUC_DIR = os.path.join(RAW_ROOT, "train", "nuclei")
VAL_IMG_DIR   = os.path.join(RAW_ROOT, "validation", "image")
VAL_NUC_DIR   = os.path.join(RAW_ROOT, "validation", "nuclei")

OUT_DIR       = os.path.join(ROOT_DIR, "data", "processed_task2_v2")
TRAIN_OUT     = os.path.join(OUT_DIR, "train")
VAL_OUT       = os.path.join(OUT_DIR, "val")
CONTRA_OUT    = os.path.join(OUT_DIR, "contrastive")

CKPT_DIR      = os.path.join(ROOT_DIR, "checkpoints_task2_v2")
FIG_DIR       = os.path.join(ROOT_DIR, "figures_task2_v2")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Data and training configuration
SEED = 42
PATCH_SIZE = 100
N_TRAIN = 2500
N_VAL = 700

# Build the contrastive set from the remaining training nuclei, keeping classes balanced.
USE_MAX_BALANCED_CONTRASTIVE_SET = True
MANUAL_N_CONTRASTIVE = 1500  # ignored if USE_MAX_BALANCED_CONTRASTIVE_SET=True

BATCH_SIZE = 64
CONTRASTIVE_BATCH_SIZE = 128
NUM_WORKERS = 2

SUP_EPOCHS = 60
SUP_PATIENCE = 12

SUPCON_EPOCHS = 80
SUPCON_SAVE_EVERY = 10
SUPCON_TEMPERATURE = 0.07

PROBE_EPOCHS = 12           # short probe to select best encoder checkpoint
PROBE_PATIENCE = 4

FROZEN_EPOCHS = 50          # final frozen-head training after checkpoint selection
FROZEN_PATIENCE = 10

LEARNING_RATE_SUP = 1e-3
LEARNING_RATE_SUPCON = 1e-3
LEARNING_RATE_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
DROPOUT = 0.2

CLASS_MAP = {
    "nuclei_tumor": "tumor",
    "nuclei_lymphocyte": "lymphocyte",
    "nuclei_histiocyte": "histiocyte",
}
VALID_CLASSES = set(CLASS_MAP.keys())
CLASS_TO_IDX = {"tumor": 0, "lymphocyte": 1, "histiocyte": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES = ["tumor", "lymphocyte", "histiocyte"]

# ImageNet normalization used with ResNet18.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

SUP_CKPT = os.path.join(CKPT_DIR, "resnet18_supervised_best.pth")
SUPCON_FINAL_ENCODER_CKPT = os.path.join(CKPT_DIR, "supcon_encoder_final.pth")
SUPCON_SELECTED_ENCODER_CKPT = os.path.join(CKPT_DIR, "supcon_encoder_selected_by_probe.pth")
SUPCON_FROZEN_CKPT = os.path.join(CKPT_DIR, "supcon_frozen_best.pth")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

USE_AMP = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)



# DATASET EXTRACTION
def get_paired_files(img_dir, nuc_dir):
    img_files = {
        Path(f).stem: os.path.join(img_dir, f)
        for f in os.listdir(img_dir) if f.endswith(".tif")
    }

    nuc_files = {}
    for f in os.listdir(nuc_dir):
        if f.endswith(".geojson"):
            stem = Path(f).stem
            base = stem.replace("_nuclei", "")
            nuc_files[base] = os.path.join(nuc_dir, f)

    stems = sorted(set(img_files) & set(nuc_files))
    print(f"Found {len(stems)} paired slides in {img_dir}")
    return [(img_files[s], nuc_files[s]) for s in stems]


def parse_nuclei(geojson_path):
    with open(geojson_path, "r") as f:
        gj = json.load(f)

    nuclei = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        cls_name = props.get("classification", {}).get("name", "")
        if cls_name not in VALID_CLASSES:
            continue

        geom = shape(feat["geometry"])
        cx, cy = geom.centroid.x, geom.centroid.y
        nuclei.append((int(round(cx)), int(round(cy)), CLASS_MAP[cls_name]))

    return nuclei

# Compute the overlap between the requested patch window and the valid image region,
# then place the crop onto a zero-padded canvas.
def extract_patch(img_array, cx, cy, patch_size=100):
    h, w = img_array.shape[:2]
    half = patch_size // 2

    x1_src = max(cx - half, 0)
    x2_src = min(cx + half, w)
    y1_src = max(cy - half, 0)
    y2_src = min(cy + half, h)

    x1_dst = x1_src - (cx - half)
    y1_dst = y1_src - (cy - half)
    x2_dst = x1_dst + (x2_src - x1_src)
    y2_dst = y1_dst + (y2_src - y1_src)

    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    crop = img_array[y1_src:y2_src, x1_src:x2_src]

    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    elif crop.ndim == 3 and crop.shape[2] == 1:
        crop = np.repeat(crop, 3, axis=2)
    elif crop.ndim == 3 and crop.shape[2] > 3:
        crop = crop[:, :, :3]

    if crop.dtype != np.uint8:
        crop = crop.astype(np.uint8)

    canvas[y1_dst:y2_dst, x1_dst:x2_dst] = crop
    return canvas


def save_patch(patch_array, out_dir, class_label, idx):
    cls_dir = os.path.join(out_dir, class_label)
    os.makedirs(cls_dir, exist_ok=True)
    Image.fromarray(patch_array).save(os.path.join(cls_dir, f"{idx:06d}.png"))


def count_patches(root):
    counts = {}
    if not os.path.isdir(root):
        return counts
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path) if f.endswith(".png")])
    return counts


def collect_nuclei(paired_files, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()

    all_nuclei = defaultdict(list)
    for img_path, nuc_path in tqdm(paired_files, desc="Parsing GeoJSONs"):
        stem = Path(img_path).stem
        nuclei = parse_nuclei(nuc_path)
        for cx, cy, label in nuclei:
            # Unique identifier used to exclude nuclei already assigned to another split.
            uid = (stem, cx, cy)
            if uid not in exclude_ids:
                all_nuclei[label].append((img_path, cx, cy, uid))

    return all_nuclei

def sample_and_save_nuclei(all_nuclei, out_dir, n_per_class):
    sampled = {}
    for lbl, items in all_nuclei.items():
        items = items.copy()
        random.shuffle(items)
        sampled[lbl] = items[:n_per_class]
        if len(items) < n_per_class:
            print(f"Warning: {lbl} only has {len(items)} samples, requested {n_per_class}")
    # Group crops by source slide so each image is read only once.
    img_to_crops = defaultdict(list)
    for lbl, items in sampled.items():
        for img_path, cx, cy, uid in items:
            img_to_crops[img_path].append((cx, cy, lbl, uid))

    counters = defaultdict(int)
    used_ids = set()
    for img_path, crops in tqdm(img_to_crops.items(), desc=f"Extracting patches -> {Path(out_dir).name}"):
        img_array = tiff.imread(img_path)
        for cx, cy, lbl, uid in crops:
            patch = extract_patch(img_array, cx, cy, PATCH_SIZE)
            save_patch(patch, out_dir, lbl, counters[lbl])
            counters[lbl] += 1
            used_ids.add(uid)

    print(f"Saved to {out_dir}")
    for lbl in CLASS_NAMES:
        print(f"  {lbl}: {counters.get(lbl, 0)}")
    return used_ids


def build_task2_datasets():
    for split_path in [TRAIN_OUT, VAL_OUT, CONTRA_OUT]:
        if os.path.exists(split_path):
            shutil.rmtree(split_path)

    train_pairs = get_paired_files(TRAIN_IMG_DIR, TRAIN_NUC_DIR)
    val_pairs = get_paired_files(VAL_IMG_DIR, VAL_NUC_DIR)

    # Build the supervised training set from the official training split.
    train_pool = collect_nuclei(train_pairs)
    for lbl in CLASS_NAMES:
        print(f"Train pool {lbl}: {len(train_pool[lbl])}")
    train_used_ids = sample_and_save_nuclei(train_pool, TRAIN_OUT, N_TRAIN)

    # Build the validation set from the official validation split.
    val_pool = collect_nuclei(val_pairs)
    for lbl in CLASS_NAMES:
        print(f"Val pool {lbl}: {len(val_pool[lbl])}")
    _ = sample_and_save_nuclei(val_pool, VAL_OUT, N_VAL)

    # Build the contrastive set from the remaining training nuclei to avoid overlap.
    contrastive_pool = collect_nuclei(train_pairs, exclude_ids=train_used_ids)
    remaining = {lbl: len(contrastive_pool[lbl]) for lbl in CLASS_NAMES}
    print("Remaining after train extraction:", remaining)

    if USE_MAX_BALANCED_CONTRASTIVE_SET:
        n_contrastive = min(remaining.values())
    else:
        n_contrastive = min(MANUAL_N_CONTRASTIVE, min(remaining.values()))

    print(f"Using contrastive set size = {n_contrastive} per class")
    _ = sample_and_save_nuclei(contrastive_pool, CONTRA_OUT, n_contrastive)

    print("\nFinal patch counts")
    for split_name, split_path in [("Train", TRAIN_OUT), ("Val", VAL_OUT), ("Contrastive", CONTRA_OUT)]:
        print(split_name, count_patches(split_path))


# DATASETS AND TRANSFORMS
class RandomRotate90:
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, angle)


class NucleiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TwoViewTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        # Generate two independently augmented views of the same image.
        return self.base_transform(img), self.base_transform(img)


class ContrastiveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # SupCon requires two augmented views of each patch.
        self.transform = TwoViewTransform(transform)
        self.samples = []
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} contrastive samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        v1, v2 = self.transform(img)
        return v1, v2, label


class Task2NPYTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for fname in sorted(os.listdir(root_dir)):
            if not fname.endswith(".npy"):
                continue
            lower = fname.lower()
            # The official test filenames encode both class label and subtype.
            if "_nuclei_tumor_" in lower:
                label = CLASS_TO_IDX["tumor"]
            elif "_nuclei_lymphocyte_" in lower:
                label = CLASS_TO_IDX["lymphocyte"]
            elif "_nuclei_histiocyte_" in lower:
                label = CLASS_TO_IDX["histiocyte"]
            else:
                continue

            if "primary" in lower:
                subtype = "primary"
            elif "metastatic" in lower:
                subtype = "metastatic"
            else:
                subtype = "unknown"

            self.samples.append((os.path.join(root_dir, fname), label, subtype, fname))

        print(f"Loaded {len(self.samples)} official Task 2 test samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subtype, fname = self.samples[idx]
        arr = np.load(path)
        # Ensure all test samples are loaded as 3-channel uint8 images.
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, subtype, fname


def make_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # No augmentation is applied during validation or testing.
    eval_transform = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Contrastive augmentation: a bit stronger than supervised training,
    # but still stain-safe for pathology patches.
    supcon_transform = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(PATCH_SIZE, scale=(0.80, 1.0), ratio=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, eval_transform, supcon_transform


def make_loaders():
    train_tf, eval_tf, supcon_tf = make_transforms()

    train_dataset = NucleiDataset(TRAIN_OUT, transform=train_tf)
    val_dataset = NucleiDataset(VAL_OUT, transform=eval_tf)
    contrastive_dataset = ContrastiveDataset(CONTRA_OUT, transform=supcon_tf)
    test_dataset = Task2NPYTestDataset(TASK2_TEST_DIR, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    contrastive_loader = DataLoader(
        contrastive_dataset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader, contrastive_loader, test_loader, test_dataset


# MODELS
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model):
    return sum(p.numel() for p in model.parameters())


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.2):
        super().__init__()
        backbone = models.resnet18(weights=None)
        # Use ResNet18 as a feature extractor (remove final FC layer).
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # Linear classification head.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)

    def get_features(self, x):
        # Return flattened feature representations for analysis.
        feat = self.encoder(x)
        return feat.squeeze(-1).squeeze(-1)


class SupConEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # Projection head used only during contrastive pretraining
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)
        z = self.projector(feat)
        # Return normalized embeddings for contrastive loss.
        return F.normalize(z, dim=1)

    def get_features(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)


class FrozenEncoderClassifier(nn.Module):
    def __init__(self, encoder_state_dict=None, num_classes=3, dropout=0.2):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        if encoder_state_dict is not None:
            self.encoder.load_state_dict(encoder_state_dict)
        # Freeze encoder parameters so only the classifier head is trained.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        # Keep encoder in eval mode to disable BatchNorm updates.
        self.encoder.eval()
        return self

    def forward(self, x):
        # Prevent gradient computation through the frozen encoder.
        with torch.no_grad():
            feat = self.encoder(x)
        return self.classifier(feat)

    def get_features(self, x):
        # Extract features without tracking gradients.
        with torch.no_grad():
            feat = self.encoder(x)
        return feat.squeeze(-1).squeeze(-1)


# LOSSES AND TRAINING HELPERS
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size, n_views, feat_dim = features.shape
        features = features.view(batch_size * n_views, feat_dim)
        features = F.normalize(features, dim=1)
        # Repeat labels across both augmented views so positives include same-class pairs.
        labels = labels.contiguous().view(-1, 1)
        labels = labels.repeat_interleave(n_views, dim=0)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        logits = torch.matmul(features, features.T) / self.temperature
        # Remove self comparisons from the similarity matrix.
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * n_views, device=features.device)
        mask = mask * logits_mask

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Average log-probability over positive pairs for each anchor.
        pos_count = mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count
        loss = -mean_log_prob_pos.mean()
        return loss


class EarlyStopper:
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0

    def step(self, value):
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True

        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False, True

        self.num_bad_epochs += 1
        should_stop = self.num_bad_epochs >= self.patience
        return should_stop, False


def train_one_epoch_classifier(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total = 0
    preds_all, labels_all = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds_all.extend(logits.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / total, acc


def evaluate_classifier(model, loader, criterion=None, with_metadata=False):
    model.eval()
    total_loss = 0.0
    total = 0
    preds_all, labels_all = [], []
    subtype_all, fname_all = [], []

    with torch.no_grad():
        for batch in loader:
            if with_metadata:
                # Test batches also return subtype and filename for subtype-level analysis.
                images, labels, subtypes, fnames = batch
            else:
                images, labels = batch
                subtypes, fnames = None, None

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(images)
                loss = criterion(logits, labels) if criterion is not None else None

            if loss is not None:
                total_loss += loss.item() * images.size(0)
            total += images.size(0)

            preds = logits.argmax(1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            if with_metadata:
                subtype_all.extend(subtypes)
                fname_all.extend(fnames)

    acc = accuracy_score(labels_all, preds_all)
    avg_loss = total_loss / total if criterion is not None else None

    if with_metadata:
        return avg_loss, acc, preds_all, labels_all, subtype_all, fname_all
    return avg_loss, acc, preds_all, labels_all


def train_supervised_model(train_loader, val_loader):
    model = ResNet18Classifier(num_classes=3, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_SUP, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUP_EPOCHS, eta_min=1e-6)
    stopper = EarlyStopper(patience=SUP_PATIENCE, mode="max")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    # Save the best model according to validation accuracy.
    best_state = None

    print("\nTraining end-to-end supervised classifier")
    print("Trainable params:", count_trainable_params(model))

    for epoch in range(1, SUP_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch_classifier(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate_classifier(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        should_stop, improved = stopper.step(val_acc)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, SUP_CKPT)

        print(
            f"[Supervised] Epoch {epoch:03d}/{SUP_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            + ("  *best" if improved else "")
        )
        if should_stop:
            print("Early stopping triggered for supervised model.")
            break

    model.load_state_dict(best_state)
    return model, history


def pretrain_supcon(contrastive_loader):
    model = SupConEncoder(proj_dim=128).to(device)
    criterion = SupConLoss(temperature=SUPCON_TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_SUPCON, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS, eta_min=1e-6)

    saved_encoder_paths = []
    history = {"loss": []}

    print("\nPretraining SupCon encoder")
    print("Trainable params:", count_trainable_params(model))

    for epoch in range(1, SUPCON_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        # Encode two augmented views of each patch and compute the SupCon loss.

        for view1, view2, labels in contrastive_loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                z1 = model(view1)
                z2 = model(view2)
                feats = torch.stack([z1, z2], dim=1)
                loss = criterion(feats, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * view1.size(0)
            total += view1.size(0)

        epoch_loss /= total
        history["loss"].append(epoch_loss)
        scheduler.step()

        print(f"[SupCon] Epoch {epoch:03d}/{SUPCON_EPOCHS} | loss={epoch_loss:.4f}")

        # Save encoder snapshots for later probe-based checkpoint selection.
        if epoch % SUPCON_SAVE_EVERY == 0 or epoch == SUPCON_EPOCHS:
            encoder_path = os.path.join(CKPT_DIR, f"supcon_encoder_epoch_{epoch:03d}.pth")
            torch.save(model.encoder.state_dict(), encoder_path)
            saved_encoder_paths.append(encoder_path)

    torch.save(model.encoder.state_dict(), SUPCON_FINAL_ENCODER_CKPT)
    return saved_encoder_paths, history


def train_frozen_head_once(encoder_state_dict, train_loader, val_loader, max_epochs=20, patience=5, save_path=None):
    model = FrozenEncoderClassifier(
        encoder_state_dict=encoder_state_dict,
        num_classes=3,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # Only the classifier head is optimized; the encoder remains frozen.
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    stopper = EarlyStopper(patience=patience, mode="max")

    best_state = None
    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch_classifier(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate_classifier(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        should_stop, improved = stopper.step(val_acc)
        # Keep the best head according to validation accuracy.
        if improved:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save(best_state, save_path)

        if should_stop:
            break

    model.load_state_dict(best_state)
    return model, history, best_val_acc


def select_best_supcon_encoder(saved_encoder_paths, train_loader, val_loader):
    print("\nSelecting best SupCon encoder using downstream frozen-head validation accuracy")
    best_encoder_path = None
    best_val_acc = -1.0
    results = []

    # Evaluate each saved encoder by training a short linear probe on top.
    for encoder_path in saved_encoder_paths:
        enc_state = torch.load(encoder_path, map_location=device)
        _, _, val_acc = train_frozen_head_once(
            encoder_state_dict=enc_state,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=PROBE_EPOCHS,
            patience=PROBE_PATIENCE,
            save_path=None,
        )
        results.append((encoder_path, val_acc))
        print(f"Probe {Path(encoder_path).name}: val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_encoder_path = encoder_path

    print(f"Selected encoder checkpoint: {best_encoder_path}")
    print(f"Best probe validation accuracy: {best_val_acc:.4f}")

    selected_state = torch.load(best_encoder_path, map_location=device)
    torch.save(selected_state, SUPCON_SELECTED_ENCODER_CKPT)
    return best_encoder_path, results


def train_final_frozen_model(selected_encoder_path, train_loader, val_loader):
    print("\nTraining final frozen-head classifier using selected SupCon encoder")
    enc_state = torch.load(selected_encoder_path, map_location=device)
    model, history, best_val_acc = train_frozen_head_once(
        encoder_state_dict=enc_state,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=FROZEN_EPOCHS,
        patience=FROZEN_PATIENCE,
        save_path=SUPCON_FROZEN_CKPT,
    )
    print(f"Frozen-head best val accuracy: {best_val_acc:.4f}")
    return model, history


# ANALYSIS HELPERS
def plot_history(history, title_prefix, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title_prefix} loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].axhline(0.7083, color="red", linestyle="--", label="baseline")
    axes[1].set_title(f"{title_prefix} accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_supcon_loss(history, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(history["loss"])
    plt.title("SupCon pretraining loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_cls_report(y_true, y_pred, header):
    print("\n" + "=" * 72)
    print(header)
    print("=" * 72)
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Tumor", "Lymphocyte", "Histiocyte"],
        digits=4,
    ))


def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Tumor", "Lymphocyte", "Histiocyte"])
    ax.set_yticklabels(["Tumor", "Lymphocyte", "Histiocyte"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def subtype_report(y_true, y_pred, subtypes, subtype_name):
    idx = [i for i, s in enumerate(subtypes) if s == subtype_name]
    if len(idx) == 0:
        print(f"No samples for subtype={subtype_name}")
        return
    y_true_sub = [y_true[i] for i in idx]
    y_pred_sub = [y_pred[i] for i in idx]
    print_cls_report(y_true_sub, y_pred_sub, f"{subtype_name.upper()} ({len(idx)} samples)")

def extract_features(model, loader):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            # Ignore any extra metadata fields and keep only images and labels.
            images, lbls = batch[:2]
            images = images.to(device, non_blocking=True)
            feat = model.get_features(images)
            feats.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def knn_eval(features, labels, k=5):
    # Evaluate feature quality with a simple cosine k-NN classifier.
    x = normalize(features)
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(x_train, y_train)
    return knn.score(x_test, y_test)


def latent_space_summary(model, loader, name):
    feats, labels = extract_features(model, loader)
    feats_norm = normalize(feats)

    # Silhouette on a stratified subsample for speed.
    max_n = min(1000, len(labels))
    if len(labels) > max_n:
        idx_all = np.arange(len(labels))
        _, idx_sub = train_test_split(idx_all, test_size=max_n, random_state=SEED, stratify=labels)
        feats_sub = feats_norm[idx_sub]
        labels_sub = labels[idx_sub]
    else:
        feats_sub = feats_norm
        labels_sub = labels

    sil = silhouette_score(feats_sub, labels_sub)
    # Use k-NN accuracy as a simple measure of class separability in feature space.
    knn_acc = knn_eval(feats, labels, k=5)

    print(f"{name}: silhouette={sil:.4f}, knn_acc={knn_acc:.4f}")
    return feats, labels, sil, knn_acc


def plot_tsne(features, labels, title, save_path):
    # Project features to 2D for qualitative visualization.
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    emb = tsne.fit_transform(features)
    colors = {0: "tab:red", 1: "tab:green", 2: "goldenrod"}

    plt.figure(figsize=(7, 6))
    for cls_idx, cls_name in IDX_TO_CLASS.items():
        mask = labels == cls_idx
        plt.scatter(
            emb[mask, 0], emb[mask, 1], s=10, alpha=0.45,
            label=cls_name, c=colors[cls_idx]
        )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_sample_patches(save_path):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for col, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(TRAIN_OUT, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(".png")]
        samples = random.sample(files, 3)
        for row, fname in enumerate(samples):
            img = Image.open(os.path.join(cls_dir, fname))
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(cls.capitalize())
    plt.suptitle("Training patches (3 per class)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_class_distribution(test_dataset, save_path):
    split_counts = {}
    for split_name, split_path in [("Train", TRAIN_OUT), ("Val", VAL_OUT), ("Contrastive", CONTRA_OUT)]:
        counts = count_patches(split_path)
        split_counts[split_name] = {
            "tumor": counts.get("tumor", 0),
            "lymphocyte": counts.get("lymphocyte", 0),
            "histiocyte": counts.get("histiocyte", 0),
        }

    # The official test set counts are inferred from filename-derived labels.    
    test_counts = defaultdict(int)
    for _, lbl, _, _ in test_dataset.samples:
        test_counts[IDX_TO_CLASS[lbl]] += 1
    split_counts["Test"] = dict(test_counts)

    splits = list(split_counts.keys())
    tumor = [split_counts[s]["tumor"] for s in splits]
    lymph = [split_counts[s]["lymphocyte"] for s in splits]
    histo = [split_counts[s]["histiocyte"] for s in splits]

    x = np.arange(len(splits))
    width = 0.25

    plt.figure(figsize=(9, 4))
    plt.bar(x - width, tumor, width, label="Tumor")
    plt.bar(x, lymph, width, label="Lymphocyte")
    plt.bar(x + width, histo, width, label="Histiocyte")
    plt.xticks(x, splits)
    plt.ylabel("Number of patches")
    plt.title("Class distribution across splits")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    print("=" * 80)
    print("TASK 2: NUCLEI CLASSIFICATION")
    print("=" * 80)

    build_task2_datasets()
    train_loader, val_loader, contrastive_loader, test_loader, test_dataset = make_loaders()

    save_sample_patches(os.path.join(FIG_DIR, "sample_patches.png"))
    save_class_distribution(test_dataset, os.path.join(FIG_DIR, "class_distribution.png"))

   
    # Model A: end-to-end supervised
    supervised_model, sup_history = train_supervised_model(train_loader, val_loader)
    plot_history(sup_history, "Supervised", os.path.join(FIG_DIR, "supervised_history.png"))

  
    # Model B: SupCon pretrain + frozen linear classifier
    saved_encoder_paths, supcon_history = pretrain_supcon(contrastive_loader)
    plot_supcon_loss(supcon_history, os.path.join(FIG_DIR, "supcon_loss.png"))

    selected_encoder_path, probe_results = select_best_supcon_encoder(saved_encoder_paths, train_loader, val_loader)
    frozen_model, frozen_history = train_final_frozen_model(selected_encoder_path, train_loader, val_loader)
    plot_history(frozen_history, "SupCon Frozen", os.path.join(FIG_DIR, "supcon_frozen_history.png"))

    # Load best weights explicitly for final evaluation
    supervised_model.load_state_dict(torch.load(SUP_CKPT, map_location=device))
    frozen_model.load_state_dict(torch.load(SUPCON_FROZEN_CKPT, map_location=device))

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Validation
    _, sup_val_acc, sup_val_preds, sup_val_labels = evaluate_classifier(supervised_model, val_loader, criterion)
    _, frozen_val_acc, frozen_val_preds, frozen_val_labels = evaluate_classifier(frozen_model, val_loader, criterion)

    # test
    _, sup_test_acc, sup_test_preds, sup_test_labels, sup_types, sup_fnames = evaluate_classifier(
        supervised_model, test_loader, criterion, with_metadata=True
    )
    _, frozen_test_acc, frozen_test_preds, frozen_test_labels, frozen_types, frozen_fnames = evaluate_classifier(
        frozen_model, test_loader, criterion, with_metadata=True
    )

    sup_val_f1 = f1_score(sup_val_labels, sup_val_preds, average="macro")
    frozen_val_f1 = f1_score(frozen_val_labels, frozen_val_preds, average="macro")
    sup_test_f1 = f1_score(sup_test_labels, sup_test_preds, average="macro")
    frozen_test_f1 = f1_score(frozen_test_labels, frozen_test_preds, average="macro")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Baseline accuracy                             : 0.7083")
    print(f"Supervised trainable params                   : {count_trainable_params(supervised_model):,}")
    print(f"Supervised total params                       : {count_all_params(supervised_model):,}")
    print(f"Frozen-head trainable params                  : {count_trainable_params(frozen_model):,}")
    print(f"Frozen-head total params                      : {count_all_params(frozen_model):,}")
    print("-" * 80)
    print(f"Supervised validation acc / macro F1          : {sup_val_acc:.4f} / {sup_val_f1:.4f}")
    print(f"SupCon Frozen validation acc / macro F1       : {frozen_val_acc:.4f} / {frozen_val_f1:.4f}")
    print("-" * 80)
    print(f"Supervised official test acc / macro F1       : {sup_test_acc:.4f} / {sup_test_f1:.4f}")
    print(f"SupCon Frozen official test acc / macro F1    : {frozen_test_acc:.4f} / {frozen_test_f1:.4f}")
    print("=" * 80)

    print_cls_report(sup_val_labels, sup_val_preds, "Supervised - Validation")
    print_cls_report(frozen_val_labels, frozen_val_preds, "SupCon Frozen - Validation")
    print_cls_report(sup_test_labels, sup_test_preds, "Supervised - Official Test")
    print_cls_report(frozen_test_labels, frozen_test_preds, "SupCon Frozen - Official Test")

    plot_confusion(sup_val_labels, sup_val_preds, "Supervised - Validation", os.path.join(FIG_DIR, "cm_sup_val.png"))
    plot_confusion(frozen_val_labels, frozen_val_preds, "SupCon Frozen - Validation", os.path.join(FIG_DIR, "cm_frozen_val.png"))
    plot_confusion(sup_test_labels, sup_test_preds, "Supervised - Official Test", os.path.join(FIG_DIR, "cm_sup_test.png"))
    plot_confusion(frozen_test_labels, frozen_test_preds, "SupCon Frozen - Official Test", os.path.join(FIG_DIR, "cm_frozen_test.png"))

    print("\nSubtype analysis - Supervised")
    subtype_report(sup_test_labels, sup_test_preds, sup_types, "primary")
    subtype_report(sup_test_labels, sup_test_preds, sup_types, "metastatic")

    print("\nSubtype analysis - SupCon Frozen")
    subtype_report(frozen_test_labels, frozen_test_preds, frozen_types, "primary")
    subtype_report(frozen_test_labels, frozen_test_preds, frozen_types, "metastatic")

    sup_feats, sup_lbls, sup_sil, sup_knn = latent_space_summary(supervised_model, test_loader, "Supervised")
    frozen_feats, frozen_lbls, frozen_sil, frozen_knn = latent_space_summary(frozen_model, test_loader, "SupCon Frozen")

    plot_tsne(sup_feats, sup_lbls, "t-SNE - Supervised features", os.path.join(FIG_DIR, "tsne_supervised.png"))
    plot_tsne(frozen_feats, frozen_lbls, "t-SNE - SupCon frozen features", os.path.join(FIG_DIR, "tsne_supcon_frozen.png"))

    print("\nLatent-space summary")
    print(f"Supervised: silhouette={sup_sil:.4f}, knn_acc={sup_knn:.4f}")
    print(f"SupCon Frozen: silhouette={frozen_sil:.4f}, knn_acc={frozen_knn:.4f}")

    with open(os.path.join(CKPT_DIR, "probe_results.txt"), "w") as f:
        for path, acc in probe_results:
            f.write(f"{Path(path).name}\t{acc:.4f}\n")

    print("\nSaved checkpoints to:", CKPT_DIR)
    print("Saved figures to:", FIG_DIR)


if __name__ == "__main__":
    main()
