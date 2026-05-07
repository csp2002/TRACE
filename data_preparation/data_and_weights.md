# Data and Pretrained Weights

This repository does **not** redistribute datasets or pretrained checkpoints. This document covers (a) how to obtain and convert each dataset, and (b) where to place each set of pretrained weights so the training, testing and simulation code can find them.

## 1. Datasets

| Dataset | Source | Notes |
|---|---|---|
| **KiTS** | [https://kits-challenge.org/kits23/](https://kits-challenge.org/kits23/) | KiTS23 challenge data; kidney tumors |
| **LiTS** | [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094) | LiTS challenge; liver tumors |
| **MSD-Pancreas** | [http://medicaldecathlon.com/](http://medicaldecathlon.com/) | Medical Segmentation Decathlon, Task 07 |
| **MSD-Colon** | [http://medicaldecathlon.com/](http://medicaldecathlon.com/) | Medical Segmentation Decathlon, Task 10 |

After downloading, you should have one folder per dataset containing volumetric NIfTI files (`*.nii.gz`) for both CT volumes and segmentation masks.

## 2. Data preprocessing pipeline

```
Raw NIfTI volumes
     │  data_preparation/nifti_to_2d.py
     ▼
2D_data/<dataset>/<split>/{CT,Mask}/<patient>/*.png
     │  data_preparation/extract_middle_slice.py    (writes annotation_dict_middle.json)
     │  data_preparation/extract_neighbor_slice.py  (writes annotation_dict_neighbor.json)
     ▼
Ready for training (tumor_seg/train.py) and simulation (tumor_seg/simulation.py)
```

### 2.1 Step 1: NIfTI → 2D PNG slices

```bash
python -m data_preparation.nifti_to_2d \
    --src   /path/to/raw/<dataset>/ \
    --dst   ./2D_data/<dataset>/ \
    --data  <dataset>
```

Per-dataset preprocessing details:

- Re-orient volumes to a consistent axis order (depth, height, width).
- Construct **binary** tumor masks by retaining only the dataset-specific tumor label and discarding organ labels.
- Apply dataset-specific **intensity clipping** based on the 0.5th and 99.5th percentiles of foreground voxel intensities (per-dataset clip ranges are given in the supplement of the paper).
- Apply min–max normalisation to map intensities into `[0, 1]`.
- Extract 2D axial slices and retain only those that contain foreground tumor pixels.
- Save each retained slice as a grayscale PNG together with its binary mask, organised by case.

The resulting layout is:

```
2D_data/
└── <dataset>/                e.g. kits, lits, pancreas, colon
    ├── train/
    │   ├── CT/<patient_id>/<slice_idx>.png
    │   └── Mask/<patient_id>/<slice_idx>.png
    └── test/
        ├── CT/<patient_id>/<slice_idx>.png
        └── Mask/<patient_id>/<slice_idx>.png
```

### 2.2 Step 2: Reference slice annotations

The simulation framework needs to know which slice acts as the "reference" for each case. Two reference protocols are supported:

```bash
# Middle-slice protocol: pick the slice with the largest GT mask
python -m data_preparation.extract_middle_slice \
    --root ./2D_data --datasets kits lits pancreas colon

# Neighbor-slice protocol: middle slice + previous-slice fallbacks
python -m data_preparation.extract_neighbor_slice \
    --root ./2D_data --datasets kits lits pancreas colon
```

These produce JSON files (`annotation_dict_middle.json`, `annotation_dict_neighbor.json`) consumed by `tumor_seg/simulation.py` and the `+TRACE` training pipeline.

## 3. ImageNet-pretrained backbone weights (for the 7 conventional models)

Required to initialise the 7 conventional model backbones at training time. Download each file and place it at the listed path **relative to the repository root**. All commands assume you run from the repo root.

| Backbone | File | Source | Place at |
|---|---|---|---|
| **TransUNet** | `R50+ViT-B_16.npz` | Google ViT release: `https://console.cloud.google.com/storage/vit_models/imagenet21k/` (download `R50+ViT-B_16.npz`) | `./model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz` |
| **SwinUNet** | `swin_tiny_patch4_window7_224.pth` | Microsoft Swin Transformer: `https://github.com/microsoft/Swin-Transformer` → "Swin-T (ImageNet-1k pretrained)" | `./tumor_seg/networks/swin_tiny_patch4_window7_224.pth` |
| **MedFormer / AttentionUNet / UNet++ / FATNet / H2Former** | `resnet34.pth` | torchvision: `https://download.pytorch.org/models/resnet34-b627a593.pth` (rename to `resnet34.pth` after download) | `./tumor_seg/networks/resnet34.pth` |

Quick command sketch:

```bash
mkdir -p ./model/vit_checkpoint/imagenet21k ./tumor_seg/networks

# TransUNet ImageNet-21k init (manual download from Google Cloud Storage; needs a browser)
# Place R50+ViT-B_16.npz under ./model/vit_checkpoint/imagenet21k/

# SwinUNet (manual download from Microsoft's release page)
# Place swin_tiny_patch4_window7_224.pth under ./tumor_seg/networks/

# ResNet-34 (direct curl)
curl -L -o ./tumor_seg/networks/resnet34.pth \
    https://download.pytorch.org/models/resnet34-b627a593.pth
```

## 4. Foundation-model weights (for MedSAM and MedSAM2)

| Backbone | File | Source | Place at |
|---|---|---|---|
| **MedSAM** | `medsam_vit_b.pth` | Wang Lab MedSAM: `https://github.com/bowang-lab/MedSAM` → "Pre-trained weights" | `./foundation_models/medsam/medsam_vit_b.pth` |
| **MedSAM2** | `MedSAM2_latest.pt` | Wang Lab MedSAM2: `https://github.com/bowang-lab/MedSAM2` → "Checkpoints" | `./foundation_models/medsam2/checkpoints/MedSAM2_latest.pt` |

For up-to-date download URLs, see the upstream README of each project.

## 5. Trained TRACE checkpoints

We do **not** redistribute the trained TRACE checkpoints (~7 conventional backbones × 4 datasets × 2 variants for the 7 conventional models alone, plus MedSAM and MedSAM2 variants). Recreate them by running `tumor_seg/train.py` per the Quick Start in the top-level README. Expect ~150 epochs per `(model, dataset)` on a single A6000-class GPU.

The `tumor_seg/simulation.py` driver expects checkpoints under `./checkpoints/<exp_subdir>/...`; refer to the dispatch logic inside `simulation.py` for the exact subdirectory naming convention (`TU_{dataset}224`, `medformer_middle_{dataset}224`, `medformer_ours_neighbor_{dataset}224`, etc.).
