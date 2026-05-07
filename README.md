# TRACE: Tumor contouring Refinement via Annotation-Conditioned Enhancement

## 1. Repository layout

```
tumor-seg-trace/
├── data_preparation/
│   ├── nifti_to_2d.py
│   ├── extract_middle_slice.py
│   ├── extract_neighbor_slice.py
│   └── data_and_weights.md          # dataset + checkpoint preparation guide
├── tumor_seg/                       # 7 conventional models + simulation
│   ├── networks/
│   ├── datasets/
│   ├── train.py
│   ├── test.py
│   ├── simulation.py
│   ├── simulation_other_metrics.py
│   ├── edit_workload.py
│   └── middle_finetune.py
├── foundation_models/
│   ├── medsam/                      # MedSAM (vendored) + TRACE training/inference
│   └── medsam2/                     # MedSAM2 (vendored) + TRACE training/inference
├── TRACE.yaml                       # Conda env for the 7 conventional models + simulation
├── medsam.yaml                      # Conda env for MedSAM/MedSAM2
├── pyproject.toml
├── requirements.txt
└── LICENSE                          # Apache-2.0
```

## 2. Installation

```bash
git clone <this-repo> tumor-seg-trace
cd tumor-seg-trace

# Environment for the 7 conventional models + simulation framework
conda env create -f TRACE.yaml
conda activate TRACE
pip install -e .
```

For the **MedSAM** and **MedSAM2** environments, please follow the installation instructions in their original GitHub repositories: [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) and [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2).

## 3. Data and pretrained weights

This repository does **not** redistribute datasets or pretrained checkpoints. See [`data_preparation/data_and_weights.md`](data_preparation/data_and_weights.md) for:

- Download links for the four publicly available CT datasets (KiTS, LiTS, MSD-Pancreas, MSD-Colon) and notes on the in-house "Local" cohort.
- The full preprocessing pipeline (NIfTI → 2D PNG slices → reference-slice annotations).
- Where to place the ImageNet-pretrained backbone weights (TransUNet R50+ViT-B, SwinUNet, ResNet-34) and the MedSAM / MedSAM2 vendor weights.
- Notes on trained TRACE checkpoint layout and patient exclusion.

## 4. Quick start

We evaluate TRACE on **9 segmentation backbones**, each in a baseline variant and a `+TRACE` variant:

| Family | Backbones |
|---|---|
| 7 conventional models | H2Former, UNet++, AttentionUNet, TransUNet, SwinUNet, FATNet, MedFormer |
| 2 foundation models | MedSAM, MedSAM2 |

### 4.1 Train

```bash
# Baseline TransUNet on KiTS
python -m tumor_seg.train --exp_name transunet --dataset kits

# TransUNet + TRACE (uses neighbor reference protocol)
python -m tumor_seg.train --exp_name transunet_ours --dataset kits
```

The `--exp_name` switch maps to baseline (e.g. `transunet`) vs. ours (e.g. `transunet_ours`). See `tumor_seg/train.py` for the full list (`medformer`, `attention_unet`, `unetpp`, `swin_unet`, `fat_net`, `h2former` — each with an `_ours` counterpart). Foundation-model training entry points live under `foundation_models/medsam/` and `foundation_models/medsam2/`.

### 4.2 Workflow simulation

Two parallel components implement our slice-by-slice clinician–AI collaboration simulation. They share the same `--model_name` / `--dataset` flags and the same checkpoint layout.

#### 4.2.1 Rejection-rate sweep

```bash
python -m tumor_seg.simulation \
    --model_name TransUNet \
    --dataset kits \
    --option 3 \
    --thresholds 0.70 0.75 0.80 0.85 0.90 0.95
```

Iterate over the 5 datasets (`kits`, `lits`, `pancreas`, `colon`, `local`) and the 9 backbones (`TransUNet`, `MedFormer`, `AttentionUNet`, `UNetPlusPlus`, `SwinUnet`, `FAT_Net`, `H2Former`, `MedSAM`, `MedSAM2`) to reproduce all reported simulation results.

Reference-protocol options (`--option`):
1. **Neighbor reference** — previous slice's prediction conditions current slice
2. **Last GT reference** — last rejected slice's GT conditions current slice (centred)
3. **Conditional neighbor** — use prediction if accepted, GT if rejected

A second entry point, `tumor_seg/simulation_other_metrics.py`, supports Surface DSC, FN/FP rate and HD95 as the accept/reject criterion in addition to Dice; it shares the CLI of `simulation.py` plus a `--metric` flag.

#### 4.2.2 Edit workload comparison

```bash
python -m tumor_seg.edit_workload --model_name TransUNet
```

Compares Mode A (no AI) vs Mode B (AI-assisted) edit effort using FN-rate, FP-rate, Hausdorff-95 and Dice-difference metrics.

## 5. Acknowledgements

This repository builds on [TransUNet](https://github.com/Beckschen/TransUNet), [MedSAM](https://github.com/bowang-lab/MedSAM) and [MedSAM2](https://github.com/bowang-lab/MedSAM2). We thank their authors for releasing their code.

## 6. License

This repository is released under the [Apache-2.0 License](LICENSE). Vendored components retain their original licenses (see `foundation_models/*/LICENSE`).
