# TRACE: Tumor contouring Refinement via Annotation-Conditioned Enhancement

## 1. Repository layout

```
TRACE/
├── data_preparation/
│   ├── nifti_to_2d.py
│   ├── extract_middle_slice.py
│   ├── extract_neighbor_slice.py
│   └── data_and_weights.md          # dataset + checkpoint preparation guide
├── tumor_seg/                       # 7 conventional models 
│   ├── networks/
│   ├── datasets/
│   ├── train.py
│   ├── test.py
│   ├── simulation.py
│   ├── simulation_other_metrics.py
│   └── edit_workload.py
├── foundation_models/
│   ├── medsam/                      # MedSAM + TRACE training/inference
│   └── medsam2/                     # MedSAM2 + TRACE training/inference
├── TRACE.yaml                       # Conda env for the 7 conventional models 
├── medsam.yaml                      # Conda env for MedSAM/MedSAM2
└── LICENSE                          # Apache-2.0
```

## 2. Installation

```bash
git clone https://github.com/csp2002/TRACE.git
cd TRACE
conda env create -f TRACE.yaml
conda activate TRACE
```

For the **MedSAM** and **MedSAM2** environments, please follow the installation instructions in their original GitHub repositories: [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) and [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2).

## 3. Data and pretrained weights

Please see [`data_preparation/data_and_weights.md`](data_preparation/data_and_weights.md) for details.

## 4. Quick start

We evaluate TRACE on diverse segmentation backbones, each in a baseline variant and a `+TRACE` variant:

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

The `--exp_name` switch maps to baseline (e.g. `transunet`) vs. ours (e.g. `transunet_ours`). See `tumor_seg/train.py` for the full list (`medformer`, `attention_unet`, `unetpp`, `swin_unet`, `FAT_Net`, `H2Former` — each with an `_ours` counterpart). Foundation-model training entry points live under `foundation_models/medsam/` and `foundation_models/medsam2/`.

Trained checkpoints are written to `./checkpoints/<exp_subdir>/...`; the test and simulation drivers below load from the same location.

### 4.2 Test (per-slice Dice and IoU)

```bash
# Baseline TransUNet on KiTS test set
python -m tumor_seg.test --exp_name transunet --dataset kits

# TransUNet + TRACE on KiTS test set
python -m tumor_seg.test --exp_name transunet_ours --dataset kits --test_ref neighbor
```

`tumor_seg/test.py` reports mean Dice and IoU over all test slices. Pass `--is_save` to additionally dump the predicted masks to `--test_save_dir`. The exact `--exp_name` strings and reference-protocol flags follow the same convention as `train.py`.

### 4.3 Workflow simulation

Two parallel components implement our slice-by-slice clinician–AI collaboration simulation. They share the same `--model_name` / `--dataset` flags and the same checkpoint layout.

#### 4.3.1 Rejection-rate sweep

```bash
python -m tumor_seg.simulation \
    --model_name TransUNet \
    --dataset kits \
    --thresholds 0.70 0.75 0.80 0.85 0.90 0.95
```

For each slice in the test volume the AI's prediction is compared against the clinician's ground truth at the chosen Dice threshold; if the metric falls below the threshold the slice is "rejected" and the clinician's mask is used as the reference for the next slice, otherwise the AI prediction itself is the reference. The driver sweeps the threshold list, computes the rejection rate for both Mode A (no reference) and Mode B (TRACE-conditioned), and writes a JSON curve.

A second entry point, `tumor_seg/simulation_other_metrics.py`, supports Surface DSC, FN/FP rate and HD95 as the accept/reject criterion in addition to Dice; it shares the CLI of `simulation.py` plus a `--metric` flag.

#### 4.3.2 Edit workload comparison

```bash
python -m tumor_seg.edit_workload --model_name TransUNet
```

Compares Mode A (no AI) vs Mode B (AI-assisted) edit effort using FN-rate, FP-rate, Hausdorff-95 and Dice-difference metrics.

## 5. Acknowledgements

This repository builds on [TransUNet](https://github.com/Beckschen/TransUNet), [MedSAM](https://github.com/bowang-lab/MedSAM) and [MedSAM2](https://github.com/bowang-lab/MedSAM2). We thank their authors for releasing their code.

## 6. License

This repository is released under the [Apache-2.0 License](LICENSE). Vendored components retain their original licenses (see `foundation_models/*/LICENSE`).
