# Computer Vision Coursework 2025/2026

## Overview

This repository contains code for two tasks on the PUMA histopathology dataset:

- **Task 1 (task1.py):** Tissue segmentation using a UNet and an autoencoder pre-training pipeline.
- **Task 2 (task2.py):** Nuclei classification using a supervised ResNet18 and a supervised contrastive learning approach.

## Environment

All experiments were run on **Google Colab** using an **NVIDIA A100 GPU** with the following setup:

- Python 3.10
- CUDA 12.1
- PyTorch 2.5.1

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The PUMA dataset should be placed in Google Drive with the following structure:

```
Drive/MyDrive/Dataset_Splits/
    train/
        image/       (.tif files)
        tissue/      (.GeoJSON files)
        nuclei/      (.GeoJSON files)
    validation/
        image/
        tissue/
        nuclei/
    test/
        image/
        tissue/
        nuclei/
```

Paths are configured separately in each script and assume Google Drive is mounted at `/content/drive`. **Note:** Task 1 uses `/content/drive/MyDrive/Dataset_Splits`, while Task 2 expects `/content/drive/MyDrive/computer_vision_project/data/Dataset_Splits`. Adjust the paths to match your Drive layout before running. Task 2 also expects a separate `Task2_Test_Set` folder.

## Running

### Task 1: Tissue Segmentation

Run `task1.py` in Google Colab (for example, upload it and execute `%run task1.py` in a notebook). The script will:

1. Mount Google Drive and cache rasterised masks as `.npy` files.
2. Train the UNet end to end (120 epochs).
3. Train the autoencoder Stage 1 (reconstruction, 50 epochs) then Stage 2 (frozen encoder segmentation, up to 100 epochs).
4. Evaluate both models on the test set using tiled inference and save qualitative results.

Checkpoints and figures are saved to `Drive/MyDrive/Uni of Edi /Computer Vision/Assignment - Task 1/checkpoints/`.

### Task 2: Nuclei Classification

Mount Google Drive in Colab first (`from google.colab import drive; drive.mount("/content/drive")`), then run `task2.py` (for example, with `%run task2.py` in a notebook). The script will:

1. Extract nucleus patches from the training and validation splits.
2. Train a supervised ResNet18 classifier (60 epochs).
3. Train a supervised contrastive encoder (80 epochs) then a frozen head classifier.
4. Evaluate both models on the provided test set and produce metrics and visualisations.

## External Code

- [PyTorch](https://pytorch.org/) for all model implementations and training.
- [Albumentations](https://albumentations.ai/) for data augmentation.
- [Rasterio](https://rasterio.readthedocs.io/) for rasterising GeoJSON polygon annotations to pixel masks.
- [Shapely](https://shapely.readthedocs.io/) for geometry processing.
- [tifffile](https://github.com/cgohlke/tifffile) for reading .tif whole slide images.
- [scikit-learn](https://scikit-learn.org/) for evaluation metrics and t-SNE.

All model architectures (UNet, Encoder, Decoders, ResNet18 classifier, contrastive pipeline) were implemented from scratch by the authors. No pretrained weights were used.

## Approximate Training Times (A100 GPU)

| Component             | Time       |
|-----------------------|------------|
| Task 1: UNet          | ~45 min    |
| Task 1: AE Stage 1   | ~3 min     |
| Task 1: AE Stage 2   | ~30 min    |
| Task 2: Supervised    | ~10 min    |
| Task 2: SupCon        | ~15 min    |

## Report

See [coursework-report.pdf](coursework-report.pdf) for the accompanying coursework report.
