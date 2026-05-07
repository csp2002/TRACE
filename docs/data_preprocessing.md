# Data Preprocessing Guideline

This repository does **not** redistribute any datasets. The TRACE evaluation uses 5 tumor CT datasets, four of which are publicly available; the in-house brain-tumor cohort is not redistributable. This document describes how to obtain each dataset and convert it into the 2D slice format expected by the training, testing and simulation code.

## 1. Datasets

### 1.1 Publicly available

| Dataset | Source | Notes |
|---|---|---|
| **KiTS** | [https://kits-challenge.org/kits23/](https://kits-challenge.org/kits23/) | KiTS23 challenge data; kidney tumors |
| **LiTS** | [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094) | LiTS challenge; liver tumors |
| **MSD-Pancreas** | [http://medicaldecathlon.com/](http://medicaldecathlon.com/) | Medical Segmentation Decathlon, Task 07 |
| **MSD-Colon** | [http://medicaldecathlon.com/](http://medicaldecathlon.com/) | Medical Segmentation Decathlon, Task 10 |

After downloading, you should have one folder per dataset containing volumetric NIfTI files (`*.nii.gz`) for both CT volumes and segmentation masks.

### 1.2 In-house cohort ("Local")

The "Local" dataset used in our paper is a private brain-tumor MRI/CT cohort collected under an IRB-approved protocol. We are unable to redistribute it. Researchers wishing to reproduce the in-house results should obtain similar data through their own institutional channels.

## 2. Pipeline overview

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

## 3. Step 1: NIfTI → 2D PNG slices

```bash
python -m data_preparation.nifti_to_2d \
    --src   /path/to/raw/<dataset>/ \
    --dst   ./2D_data/<dataset>/ \
    --data  <dataset>
```

**Per-dataset preprocessing details** (consult `data_preparation/nifti_to_2d.py` and `Dataset_statistic.py`-style helpers in the original paper):

- Re-orient volumes to a consistent axis order (depth, height, width).
- Construct **binary** tumor masks by retaining only the dataset-specific tumor label and discarding organ labels.
- Apply dataset-specific **intensity clipping** based on the 0.5th and 99.5th percentiles of foreground voxel intensities (see Table S1 of the paper for the per-dataset clip ranges).
- Apply min–max normalisation to map intensities into `[0, 1]`.
- Extract 2D axial slices and retain only those that contain foreground tumor pixels.
- Save each retained slice as a grayscale PNG together with its binary mask, organised by case.

The resulting layout is:

```
2D_data/
└── <dataset>/                e.g. kits, lits, pancreas, colon, local
    ├── train/
    │   ├── CT/<patient_id>/<slice_idx>.png
    │   └── Mask/<patient_id>/<slice_idx>.png
    └── test/
        ├── CT/<patient_id>/<slice_idx>.png
        └── Mask/<patient_id>/<slice_idx>.png
```

## 4. Step 2: Reference slice annotations

The simulation framework needs to know which slice acts as the "reference" for each case. Two reference protocols are supported:

```bash
# Middle-slice protocol: pick the slice with the largest GT mask
python -m data_preparation.extract_middle_slice \
    --root ./2D_data --datasets kits lits pancreas colon local

# Neighbor-slice protocol: middle slice + previous-slice fallbacks
python -m data_preparation.extract_neighbor_slice \
    --root ./2D_data --datasets kits lits pancreas colon local
```

These produce JSON files (`annotation_dict_middle.json`, `annotation_dict_neighbor.json`) consumed by `tumor_seg/simulation.py` and the `+TRACE` training pipeline.

## 5. Pretrained backbone weights

The 7 CNN backbones rely on ImageNet-pretrained checkpoints:

| Backbone | Checkpoint | Where to put it |
|---|---|---|
| TransUNet | `R50+ViT-B_16` (Google original) | `./tumor_seg/networks/<filename>.pth` (place yourself) |
| SwinUNet | `swin_tiny_patch4_window7_224.pth` | `./tumor_seg/networks/swin_tiny_patch4_window7_224.pth` |
| Other CNNs | ResNet-34 (`resnet34.pth`) | `./tumor_seg/networks/resnet34.pth` |

These are **not** included in the repository. Download from the upstream repositories (TransUNet, Swin-UNet, torchvision) and place them under `tumor_seg/networks/`. The exact filenames the code expects can be found by grep'ing for `.pth` in `tumor_seg/networks/`.

For **MedSAM** and **MedSAM2** vendored under `third_party/`, follow each vendor's own README (`third_party/medsam/README.md` and `third_party/medsam2/README.md`) to obtain the upstream pretrained weights.

## 6. Trained TRACE checkpoints

We do **not** redistribute the trained TRACE checkpoints (~7 backbones × 5 datasets × 2 variants). Recreate them by training each backbone according to `tumor_seg/train.py`'s CLI; expect O(150 epochs) per (model, dataset) on a single A6000-class GPU.

## 7. Patient exclusion

The original paper excludes one patient (an outlier in the in-house cohort) from all simulations. The `--exclude-patients` flag of `tumor_seg/simulation.py` and `tumor_seg/edit_workload.py` accepts a quoted string of comma-separated patient IDs to drop at runtime; the released defaults pass an empty string.
