# Data and Pretrained Weights

Dataset acquisition + conversion, and pretrained-weight placement.

## 1. Datasets

| Dataset | Source | Notes |
|---|---|---|
| **KiTS** | [https://kits-challenge.org/kits23/](https://kits-challenge.org/kits23/) | KiTS23 challenge data; kidney tumors |
| **LiTS** | [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094) | LiTS challenge; liver tumors |
| **MSD-Pancreas** | [http://medicaldecathlon.com/](http://medicaldecathlon.com/) | Medical Segmentation Decathlon, Task 07 |
| **MSD-Colon** | [http://medicaldecathlon.com/](http://medicaldecathlon.com/) | Medical Segmentation Decathlon, Task 10 |


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
    --data        <dataset>                 # one of: kits, lits, pancreas, colon
    --data_prefix /path/to/raw/<dataset>/   # raw NIfTI root for that dataset
    --save_folder ./2D_data/<dataset>/      # optional; defaults to ./2D_data/<dataset>
```

Train / test split is read from [`data_preparation/splits/<dataset>/split.pkl`](splits/) — pancreas / LiTS / colon from [3DSAM-adapter](https://github.com/med-air/3DSAM-adapter), kits from the [KiTS23](https://github.com/neheller/kits23) release. Each entry maps `case_id -> (img_path, seg_path)`, paths relative to `--data_prefix`.

Resulting layout:

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

TRACE and the simulation framework needs a "reference" slice per case. Two protocols:

```bash
# Middle-slice protocol: pick the slice with the largest GT mask
python -m data_preparation.extract_middle_slice \
    --root ./2D_data --datasets kits lits pancreas colon

# Neighbor-slice protocol: middle slice + previous-slice fallbacks
python -m data_preparation.extract_neighbor_slice \
    --root ./2D_data --datasets kits lits pancreas colon
```

Output: `annotation_dict_middle.json` / `annotation_dict_neighbor.json`.

## 3. Foundation-model weights (for MedSAM and MedSAM2)

| Backbone | File | Source | 
|---|---|---|---|
| **MedSAM** | `medsam_vit_b.pth` |  `https://github.com/bowang-lab/MedSAM`
| **MedSAM2** | `MedSAM2_latest.pt` |  `https://github.com/bowang-lab/MedSAM2` 



