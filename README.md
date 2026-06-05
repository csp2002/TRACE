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
└── LICENSE                          # Apache-2.0
```

## 2. Installation

```bash
git clone https://github.com/csp2002/TRACE.git
cd TRACE
conda env create -f TRACE.yaml
conda activate TRACE
```

> **Note on the reference environment.** All experiments in this repository were run on a Linux server with 8× NVIDIA RTX A6000 GPUs (CUDA 11.3, Python 3.9, PyTorch 1.12.1). The pinned versions in `TRACE.yaml` reflect that exact setup. On a different host (different CUDA driver, GPU generation, or OS), some pinned versions may not be installable verbatim — relaxing a few minor-version pins is expected and usually safe.

For the **MedSAM** and **MedSAM2** environments, please follow the installation instructions in their original GitHub repositories: [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) and [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2).

> **Extra packages for the TRACE add-on.** TRACE's refinement module imports `timm` and `einops`, which are *not* part of the stock MedSAM/MedSAM2 environments (vanilla MedSAM/MedSAM2 do not need them). After creating each foundation-model env, also run `pip install timm einops`. To run the workflow-simulation drivers (`tumor_seg/simulation*.py`, `tumor_seg/edit_workload.py`) with `--model_name MedSAM` or `MedSAM2`, additionally `pip install ml-collections` (imported by the conventional backbones those drivers load).

## 3. Data and pretrained weights

Please see [`data_preparation/data_and_weights.md`](data_preparation/data_and_weights.md) for details.

## 4. Quick start

We evaluate TRACE on diverse segmentation backbones, each in a baseline variant and a `+TRACE` variant:

| Family | Backbones |
|---|---|
| 7 conventional models | H2Former, UNet++, AttentionUNet, TransUNet, SwinUNet, FATNet, MedFormer |
| 2 foundation models | MedSAM, MedSAM2 |

### 4.1 Train

**7 conventional models** (`tumor_seg/train.py`):

```bash
# Baseline TransUNet on Colon — no reference slice
python -m tumor_seg.train --exp_name transunet --dataset colon

# TransUNet + TRACE on Colon — neighbor reference (matches the simulation setup below)
python -m tumor_seg.train --exp_name transunet_ours --dataset colon --ref neighbor
```

`--exp_name` selects the model: pick from `transunet`, `medformer`, `attention_unet`, `unetpp`, `swin_unet`, `FAT_Net`, `H2Former` (baseline) or any of those with `_ours` appended (+TRACE). `--ref` (used only by `_ours`) is `middle` or `neighbor`.

**MedSAM** (`foundation_models/medsam/train_one_gpu.py`). First activate the MedSAM env you installed from [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) (not the `TRACE` env above):

```bash
conda activate medsam   # name of the env you created from upstream MedSAM
cd foundation_models/medsam
# Baseline MedSAM on Colon — neighbor box prompt
python train_one_gpu.py --data colon --ref neighbor
# MedSAM + TRACE on Colon — neighbor reference
python train_one_gpu.py --data colon --ref neighbor --use_trace
```

**MedSAM2** (`foundation_models/medsam2/train_medsam2_2d.py`). First activate the MedSAM2 env you installed from [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2):

```bash
conda activate medsam2  # name of the env you created from upstream MedSAM2
cd foundation_models/medsam2
# Baseline MedSAM2 on Colon — neighbor box prompt
python train_medsam2_2d.py --data colon --ref_type neighbor --checkpoint checkpoints/MedSAM2_latest.pt
# MedSAM2 + TRACE on Colon — neighbor reference
python train_medsam2_2d.py --data colon --ref_type neighbor --use_trace --checkpoint checkpoints/MedSAM2_latest.pt
```

All training writes checkpoints to subdirectories whose names encode the variant (baseline vs `+TRACE`) and the reference protocol. The test and simulation drivers below load from the same locations.

### 4.2 Test

```bash
# Baseline TransUNet on Colon test set
python -m tumor_seg.test --exp_name transunet --dataset colon

# TransUNet + TRACE on Colon test set — must match the `--ref` used at training
python -m tumor_seg.test --exp_name transunet_ours --dataset colon --train_ref neighbor --test_ref neighbor
```

`tumor_seg/test.py` reports mean Dice and IoU over all test slices. `--train_ref` selects which trained checkpoint to load (matches the training `--ref`); `--test_ref` selects which reference protocol to use at test time. Pass `--is_save` to additionally dump the predicted masks to `--test_save_dir`.

### 4.3 Workflow simulation

Two parallel components implement our slice-by-slice clinician–AI collaboration simulation. They share the same `--model_name` / `--dataset` flags and the same checkpoint layout.

#### 4.3.1 Rejection-rate sweep

```bash
python -m tumor_seg.simulation \
    --model_name TransUNet \
    --dataset colon \
    --thresholds 0.70 0.75 0.80 0.85 0.90 0.95
```

Each slice's AI prediction is checked against GT at the Dice threshold. The driver sweeps thresholds and reports the rejection rate for **Mode A** (baseline alone) vs **Mode B** (baseline + TRACE).

A second entry point, `tumor_seg/simulation_other_metrics.py`, supports Surface DSC, FN/FP rate and HD95 as the accept/reject criterion in addition to Dice; it shares the CLI of `simulation.py` plus a `--metric` flag.

#### 4.3.2 Edit workload comparison

```bash
python -m tumor_seg.edit_workload --model_name TransUNet --datasets colon
```

Compares per-slice edit effort (treat every slice as 'rejected') between **Mode A** (baseline alone) and **Mode B** (baseline + TRACE with neighbor reference) via FN-rate, FP-rate, HD-95 and Dice-difference between the AI prediction and GT.

The examples above use `--model_name TransUNet`, runnable in the `TRACE` env; for `MedSAM` / `MedSAM2`, switch to the matching conda env (which additionally need `pip install timm einops ml-collections` — see the install note in Section 2).

## 5. Acknowledgements

This repository builds on [TransUNet](https://github.com/Beckschen/TransUNet), [MedSAM](https://github.com/bowang-lab/MedSAM) and [MedSAM2](https://github.com/bowang-lab/MedSAM2). We thank their authors for releasing their code.

## 6. License

This repository is released under the [Apache-2.0 License](LICENSE). Vendored components retain their original licenses (see `foundation_models/*/LICENSE`).
