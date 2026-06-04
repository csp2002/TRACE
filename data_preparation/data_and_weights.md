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
# Default example — converts MSD-Colon (extract Task10_Colon.tar into the repo root first).
python -m data_preparation.nifti_to_2d \
    --data        colon                 # one of: kits | lits | pancreas | colon
    --data_prefix Task10_Colon          # raw NIfTI root; defaults to the canonical extracted folder name
                                        # (kits23 / LITS17 / Task07_Pancreas / Task10_Colon)
    --save_folder ./2D_data/colon       # output root; defaults to ./2D_data/<dataset>
    --splits      train val test        # subset of {train, val, test}; default is all three
```

The split is read from the bundled [`data_preparation/splits/<dataset>/split.pkl`](splits/). Each entry maps `case_id -> (img_path, seg_path)`, with both paths relative to `--data_prefix`. The paths inside each split.pkl assume the dataset has been extracted into the canonical folder for its source archive — `Task10_Colon/` (MSD), `Task07_Pancreas/` (MSD), `LITS17/` (CodaLab LiTS17, both Training_Batch zips extracted into a single directory), `kits23/` (the KiTS23 dataset directory). Override via `--data_prefix` if you put the data elsewhere.

Resulting layout:

```
2D_data/
└── <dataset>/                e.g. kits, lits, pancreas, colon
    ├── train/
    │   ├── CT/<patient_id>/<slice_idx>.png
    │   └── Mask/<patient_id>/<slice_idx>.png
    ├── val/
    │   ├── CT/<patient_id>/<slice_idx>.png
    │   └── Mask/<patient_id>/<slice_idx>.png
    └── test/
        ├── CT/<patient_id>/<slice_idx>.png
        └── Mask/<patient_id>/<slice_idx>.png
```

### 2.2 Step 2: Reference slice annotations

TRACE and the simulation framework needs a "reference" slice per case. Two protocols:

```bash
# Default — build reference-slice JSONs for the MSD-Colon 2D output produced above.
# Middle-slice protocol: pick the slice with the largest GT mask
python -m data_preparation.extract_middle_slice --root ./2D_data --datasets colon

# Neighbor-slice protocol: middle slice + previous-slice fallbacks
python -m data_preparation.extract_neighbor_slice --root ./2D_data --datasets colon

# To process more than one dataset in a single call, list them all after `--datasets`, e.g.
#   --datasets kits lits pancreas colon
```

Output: `annotation_dict_middle.json` / `annotation_dict_neighbor.json`, one pair per `<dataset>/<split>/` directory under `--root`.

## 3. Foundation-model weights (for MedSAM and MedSAM2)

| Backbone | File | Source | 
|---|---|---|---|
| **MedSAM** | `medsam_vit_b.pth` |  `https://github.com/bowang-lab/MedSAM`
| **MedSAM2** | `MedSAM2_latest.pt` |  `https://github.com/bowang-lab/MedSAM2` 



