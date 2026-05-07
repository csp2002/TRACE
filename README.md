# TRACE: Reference-Conditioned Iterative Refinement for Tumor Segmentation

> Code for **TRACE**, a plug-and-play refinement module that conditions on a single annotated reference slice and iteratively refines 2D tumor segmentations. Evaluated on 9 segmentation backbones across 5 tumor CT datasets.

## 1. What this repository provides

- **Training, testing and simulation code** for 9 backbones × 2 variants (baseline / +TRACE)
- **Slice-by-slice acceptance simulation** (`tumor_seg/simulation.py`) replicating the clinician–AI accept/reject workflow
- **Edit-workload comparison** (`tumor_seg/edit_workload.py`) measuring expected manual editing effort

What it does **not** include:
- Datasets (preprocessing instructions in `docs/data_preprocessing.md`)
- Trained checkpoints (train them via the scripts here, or contact the authors)
- Manuscript / supplementary material
- HCI study analysis code, surveys, or clinician annotations
- Visualization scripts and result CSVs from the paper

## 2. Models

| Family | Backbones |
|---|---|
| **7 traditional 2D CNNs** | H2Former, UNet++, AttentionUNet, TransUNet, SwinUNet, FATNet, MedFormer |
| **2 foundation models** | MedSAM, MedSAM2 |

Each backbone has a baseline variant (no reference conditioning) and a `+TRACE` variant. The TRACE module is defined in three locations to match each family's input resolution and feature geometry:

| Family | TRACE class | Input |
|---|---|---|
| 7 CNNs | `tumor_seg/networks/vit_seg_modeling.py` (`class TRACE`) | 224 × 224 |
| MedSAM | `third_party/medsam/My_utils.py` (`class TRACE`) | 1024 × 1024 |
| MedSAM2 | `third_party/medsam2/training/model/trace.py` (`class TRACE`) | 512 × 512 |

The three implementations share the same conceptual design (dual ResNet-18 encoders + confidence-pyramid decoder + iterative refinement); only tensor dimensions differ.

## 3. Repository layout

```
tumor-seg-trace/
├── data_preparation/
│   ├── nifti_to_2d.py
│   ├── extract_middle_slice.py
│   └── extract_neighbor_slice.py
├── tumor_seg/                       # 7-CNN code + simulation
│   ├── networks/
│   ├── datasets/
│   ├── train.py
│   ├── test.py
│   ├── simulation.py
│   ├── simulation_other_metrics.py
│   ├── edit_workload.py
│   └── middle_finetune.py
├── third_party/
│   ├── medsam/                      # MedSAM (vendored) + TRACE training/inference
│   └── medsam2/                     # MedSAM2 (vendored) + TRACE training/inference
├── scripts/                         # Bash launchers (run_simulation_*, run_edit_workload_*)
├── docs/
│   └── data_preprocessing.md
├── 3DSAM.yaml                       # Conda env for 7-CNN + simulation
├── medsam.yaml                      # Conda env for MedSAM/MedSAM2
├── pyproject.toml
├── requirements.txt
├── LICENSE                          # Apache-2.0
└── CITATION.cff
```

## 4. Installation

```bash
git clone <this-repo> tumor-seg-trace
cd tumor-seg-trace

# Environment for 7 CNN backbones + simulation framework
conda env create -f 3DSAM.yaml
conda activate 3DSAM
pip install -e .

# (Optional) Environment for MedSAM / MedSAM2
conda env create -f medsam.yaml
```

## 5. Data preparation

This repository does **not** redistribute datasets. See [`docs/data_preprocessing.md`](docs/data_preprocessing.md) for download links and the conversion pipeline producing the expected `2D_data/<dataset>/<split>/{CT,Mask}/<patient>/*.png` layout, plus the JSON reference-slice annotations (`annotation_dict_middle.json`, `annotation_dict_neighbor.json`).

Pretrained ImageNet checkpoints for the 7 CNN backbones (TransUNet, SwinUNet, ResNet-34) must be placed under `tumor_seg/networks/`. MedSAM and MedSAM2 vendor weights are obtained from their respective upstream repositories.

## 6. Quick start

### 6.1 Train

```bash
# Baseline TransUNet on KiTS
python -m tumor_seg.train --exp_name transunet --dataset kits

# TransUNet + TRACE (uses neighbor reference protocol)
python -m tumor_seg.train --exp_name transunet_ours --dataset kits
```

The `--exp_name` switch maps to baseline (e.g. `transunet`) vs. ours (e.g. `transunet_ours`). See `tumor_seg/train.py` for the full list (`medformer`, `attention_unet`, `unetpp`, `swin_unet`, `fat_net`, `h2former` — each with an `_ours` counterpart).

### 6.2 Simulation (rejection-rate sweep)

```bash
# Run all dataset/option combinations for one backbone
bash scripts/run_simulation_TransUNet.sh

# Or invoke directly
python -m tumor_seg.simulation \
    --model_name TransUNet \
    --dataset kits \
    --option 3 \
    --thresholds 0.70 0.75 0.80 0.85 0.90 0.95
```

Simulation options (`--option`):
1. **Neighbor reference** — previous slice's prediction conditions current slice
2. **Last GT reference** — last rejected slice's GT conditions current slice (centred)
3. **Conditional neighbor** — use prediction if accepted, GT if rejected

### 6.3 Edit workload comparison

```bash
bash scripts/run_edit_workload_TransUNet.sh
# or:
python -m tumor_seg.edit_workload --model_name TransUNet
```

Compares Mode A (no AI) vs Mode B (AI-assisted) edit effort using FN-rate, FP-rate, Hausdorff-95 and Dice-difference metrics.

### 6.4 Other accept/reject metrics

`tumor_seg/simulation_other_metrics.py` is a variant of `simulation.py` supporting Surface DSC, FN/FP rate and HD95 as the accept/reject criterion in addition to Dice. It uses the same CLI; see `scripts/run_sim_v4_local_TransUNet_metrics.sh` for an example.

## 7. Vendor code attribution

`third_party/medsam/` is derived from [Wang Lab's MedSAM](https://github.com/bowang-lab/MedSAM); `third_party/medsam2/` is derived from [Wang Lab's MedSAM2](https://github.com/bowang-lab/MedSAM2). We retain each vendor's `LICENSE` and `README.md` and add the `class TRACE` integration on top. Substantive modifications are concentrated in:

- `third_party/medsam/My_utils.py` — adds `class TRACE` and `MedSAM_v6554` (= MedSAM + TRACE)
- `third_party/medsam/train_one_gpu.py`, `test_ref_finetune.py` — TRACE-aware training/testing
- `third_party/medsam2/training/model/trace.py` — TRACE for MedSAM2
- `third_party/medsam2/training/model/medsam2_v6554.py` — MedSAM2 + TRACE
- `third_party/medsam2/train_medsam2_2d_v6554.py`, `test_medsam2_2d_v6554.py` — TRACE-aware training/testing

## 8. Citation

```bibtex
@article{trace2026,
  title  = {TBD},
  author = {TBD},
  year   = {2026},
}
```

## 9. License

This repository is released under the [Apache-2.0 License](LICENSE). Vendored components retain their original licenses (see `third_party/*/LICENSE`).
