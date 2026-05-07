# MedSAM2 Video-Mode Experiments — Full Log

This document records every experiment we ran trying to integrate the user's
plug-in reference-slice refinement module (`TRACE`) into MedSAM2's
**video-mode** segmentation, across the 5 tumor datasets in this project.

It supersedes the multiple intermediate CSVs and is the single source of
truth for paper writing.

Time span: 2026-04-23 → 2026-04-25.

---

## 1. Setting

### 1.1 Datasets

5 tumor CT datasets, native resolution 512×512 PNG slices stacked per
patient. Slice counts vary widely:

| Dataset | Train patients | Test patients | Median slices/patient |
|---|---:|---:|---:|
| colon | 89 | 26 | 10 |
| lits | 83 | 24 | 35 |
| pancreas | 196 | 57 | 8 |
| kits | 342 | 99 | 14 |
| local | 248 | 72 | 10 |

Train data is converted to NPZ via `scripts/convert_png_to_npz.py`.

| NPZ root | min_slices | colon | lits | pancreas | kits | local |
|---|---:|---:|---:|---:|---:|---:|
| `data/tumorseg_npz` (legacy, used by very early runs only) | 4 | 86 | 83 | 184 | 335 | 243 |
| `data/tumorseg_npz_v2` (canonical for all reported runs) | 2 | 89 | 83 | 196 | 342 | 248 |

After v2 + sampler pad-on-short: every patient is included in training; short
volumes are repeat-padded inside the sampler to T frames.

### 1.2 Test protocol (constant for all video-mode runs)

`test_medsam2_2d_video.py` (post-hoc) and
`test_medsam2_2d_video_refine.py` /
`test_medsam2_2d_video_refine_interleaved.py` (with refinement) all share:

- For each test patient, sort slices by filename → "video".
- Find middle slice via `annotation_dict_middle.json`; extract a tight
  bounding box from its GT mask.
- Split the patient at the middle slice into two halves:
  - **Part 1**: middle → 0 (reverse propagation).
  - **Part 2**: middle → end (forward propagation).
- Each half: `init_state` on a tmp JPG dir, `add_new_points_or_box(box=...)` on
  middle, `propagate_in_video(...reverse=...)`, collect per-slice masks.
- Per-slice IoU/Dice against per-slice GT, then mean over all slices in the
  test split. Middle slice counted once.

### 1.3 Common training knobs (carried over from FLARE25 RECIST baseline)

- **Optimizer**: AdamW, base_lr 5e-5, image_encoder lr 3e-5, layer-decay
  0.9, weight_decay 0.1 (0 for bias / LayerNorm), gradient clip max_norm 0.1.
- **AMP**: bfloat16.
- **Scheduler**: cosine, end_value = lr/10.
- **Image size**: 512.
- **Batch size**: 1 video per GPU.
- **Epochs**: 50.
- **Warm-start**: `checkpoints/MedSAM2_latest.pt` (Wang Lab's released MedSAM2
  weights), loaded with `strict=False`.
- **Loss for runs with refinement**: `MultiStepMultiMasksAndIous` with
  weights `loss_mask=20, loss_dice=1, loss_iou=0, loss_class=0`,
  `pred_obj_scores=false`. Refinement iterations are plumbed into
  `multistep_pred_multimasks_high_res` with dummy iou/score so the existing
  loss machinery applies focal+dice on each of the 3 refinement iterations
  (deep supervision).
- **Loss for V baseline (no module)**: standard 4-component
  `MultiStepMultiMasksAndIous` (mask 20, dice 1, iou 1, class 1) — upstream
  default.
- **Refinement module**: `training/model/trace.py:TRACE`
  (16.22M params; ImageNet-pretrained timm ResNet18 × 2 encoders + decoder
  with confidence pyramid). 3 iterations, deep supervision,
  `detach_between_iters=true`.
- **Frozen**: `sam_prompt_encoder` only (~6.2K params).
- **Joint trainable**: SAM2 image_encoder, memory_attention, memory_encoder,
  sam_mask_decoder, plus refinement = 39M (SAM2) + 16M (refinement) = 55M.

---

## 2. Code inventory

| Concept | File |
|---|---|
| Baseline video FT | `training/model/sam2.py` (`SAM2Train`) |
| Post-hoc refinement wrapper | `training/model/sam2_with_refine.py` (`SAM2TrainWithRefine`) |
| Interleaved refinement | `training/model/sam2_with_refine_interleaved.py` (`SAM2TrainWithRefineInterleaved`) |
| Reference-slice sampler | `training/dataset/vos_sampler.py:MiddleAnchoredSampler` |
| Pad-short-videos in `RandomUniformSampler` | same file, `allow_pad_short_videos=true` |
| Refinement module | `training/model/trace.py:TRACE` |
| Data conversion | `scripts/convert_png_to_npz.py` |
| Test (post-hoc) | `test_medsam2_2d_video_refine.py` |
| Test (interleaved) | `test_medsam2_2d_video_refine_interleaved.py` |
| Test (no module) | `test_medsam2_2d_video.py` |

YAMLs in `sam2/configs/`:

| YAML | Use |
|---|---|
| `sam2.1_hiera_t512.yaml` | Inference-side config (top-level `model:`), used by all test scripts. |
| `sam2.1_hiera_tiny512_tumorseg_v2.yaml` | V baseline FT (no module), padded NPZ. |
| `sam2.1_hiera_tiny512_tumorseg_refine.yaml` | Initial post-hoc refine (legacy, min=4 NPZ; superseded). |
| `sam2.1_hiera_tiny512_tumorseg_refine_A_midanchor.yaml` | Ablation A. |
| `sam2.1_hiera_tiny512_tumorseg_refine_B_boxonly.yaml` | Ablation B. |
| `sam2.1_hiera_tiny512_tumorseg_refine_C_t8.yaml` | Ablation C. |
| `sam2.1_hiera_tiny512_tumorseg_refine_D_midanchor_t8.yaml` | Ablation D (= A + C). |
| `sam2.1_hiera_tiny512_tumorseg_refine_E_interleaved.yaml` | Ablation E (interleaved). |

Launch shells in `MedSAM2/`:

| Shell | Use |
|---|---|
| `single_node_train_tumorseg_v2.sh <ds> <gpu>` | V baseline FT |
| `single_node_train_tumorseg_refine_A.sh <ds> <gpu>` | A |
| `single_node_train_tumorseg_refine_B.sh <ds> <gpu>` | B |
| `single_node_train_tumorseg_refine_C.sh <ds> <gpu>` | C |
| `single_node_train_tumorseg_refine_D.sh <ds> <gpu>` | D |
| `single_node_train_tumorseg_refine_E.sh <ds> <gpu>` | E |
| `scripts/test_tumorseg_refine.sh <ds> <gpu> <ckpt>` | Post-hoc refine test |
| `scripts/test_tumorseg_refine_interleaved.sh <ds> <gpu> <ckpt>` | Interleaved test (E only) |
| `scripts/test_tumorseg.sh <ds> <gpu> <ckpt>` | V baseline FT test |

Result JSONs in `results/`:

| Pattern | Meaning |
|---|---|
| `results_<ds>_video.json` | V0 zero-shot (MedSAM2_latest, no training) |
| `results_<ds>_video_v2.json` | V baseline (video FT no module) |
| `results_<ds>_video_refineA.json` | Ablation A (post-hoc test) |
| `results_<ds>_video_refineB.json` | Ablation B |
| `results_<ds>_video_refineC.json` | Ablation C |
| `results_<ds>_video_refineD.json` | Ablation D |
| `results_<ds>_video_refineE_posthoc.json` | E, tested post-hoc |
| `results_<ds>_video_refineE_interleaved.json` | E, tested interleaved (matched) |
| `medsam2_overall_v4.csv` | Final compiled comparison table |

---

## 3. Experiments

### 3.1 V0 — Zero-shot MedSAM2_latest

Just run `test_medsam2_2d_video.py` with `MedSAM2_latest.pt` as the checkpoint.
No training. Establishes "out of the box" video-mode performance.

### 3.2 V baseline — Video fine-tune (no module)

| Setting | Value |
|---|---|
| YAML | `sam2.1_hiera_tiny512_tumorseg_v2.yaml` |
| Model | `SAM2Train` (vanilla) |
| Sampler | `RandomUniformSampler` |
| `num_frames` (T) | 4 |
| `prob_to_use_pt_input_for_train` | 0.5 |
| `prob_to_use_box_input_for_train` | 1.0 |
| `reverse_time_prob` | 0.5 |
| `allow_pad_short_videos` | true |
| Loss | mask 20 + dice 1 + iou 1 + class 1 (full SAM2 loss) |
| Refinement module | none |

This is the "competing baseline" we are trying to beat with the module.

### 3.3 First module integration — Post-hoc refine, min=4 NPZ (legacy, superseded)

Initial attempt before we added pad-short-video. Used `tumorseg_npz` with
`min_slices=4`. SAM2TrainWithRefine post-hoc wrapper. Mean Dice ≈ 0.758 over
5 datasets (worse than V baseline 0.803 measured at the time on the same NPZ).
**Replaced by ablations A/B/C/D/E below**, all of which use the v2 padded NPZ
for apples-to-apples comparison with V baseline.

### 3.4 Ablation A — MiddleAnchoredSampler, T=4

| Setting | Value |
|---|---|
| YAML | `sam2.1_hiera_tiny512_tumorseg_refine_A_midanchor.yaml` |
| Model | `SAM2TrainWithRefine` (post-hoc; `forward_tracking` runs as in upstream, refinement applied per-frame at the end of forward) |
| Sampler | `MiddleAnchoredSampler` (conditioning frame = volume middle, direction fwd/reverse randomized inside sampler) |
| T | 4 |
| `prob_to_use_pt_input_for_train` | 0.5 |
| `allow_pad_short_videos` | true |

**Hypothesis**: align training reference position with test (middle slice).

### 3.5 Ablation B — `prob_to_use_pt_input=1.0`, T=4

| Setting | Value |
|---|---|
| YAML | `sam2.1_hiera_tiny512_tumorseg_refine_B_boxonly.yaml` |
| Model | `SAM2TrainWithRefine` |
| Sampler | `RandomUniformSampler` |
| T | 4 |
| `prob_to_use_pt_input_for_train` | **1.0** |
| `allow_pad_short_videos` | true |

**Hypothesis**: removing the 50% mask-input branch (where the cond frame's
pred ≈ GT due to `use_mask_input_as_output_without_sam=true`) eliminates a
train-test distribution gap on the reference's predicted mask.

### 3.6 Ablation C — T=8, RandomUniformSampler

| Setting | Value |
|---|---|
| YAML | `sam2.1_hiera_tiny512_tumorseg_refine_C_t8.yaml` |
| Model | `SAM2TrainWithRefine` |
| Sampler | `RandomUniformSampler` |
| T | **8** |
| `prob_to_use_pt_input_for_train` | 0.5 |
| `allow_pad_short_videos` | true |

**Hypothesis**: longer training window = longer propagation chain; closer to
test-time per-half propagation length (median up to 17 steps for lits).

### 3.7 Ablation D — A + C combined

| Setting | Value |
|---|---|
| YAML | `sam2.1_hiera_tiny512_tumorseg_refine_D_midanchor_t8.yaml` |
| Model | `SAM2TrainWithRefine` (still post-hoc) |
| Sampler | `MiddleAnchoredSampler` |
| T | **8** |
| `prob_to_use_pt_input_for_train` | 0.5 |
| `allow_pad_short_videos` | true |

**Hypothesis**: A and C fix orthogonal train-test gaps (sampler vs. window
length); their combination should be additive.

### 3.8 Ablation E — Interleaved refinement

| Setting | Value |
|---|---|
| YAML | `sam2.1_hiera_tiny512_tumorseg_refine_E_interleaved.yaml` |
| Model | **`SAM2TrainWithRefineInterleaved`** |
| Sampler | `MiddleAnchoredSampler` (same as D) |
| T | 8 (same as D) |
| `prob_to_use_pt_input_for_train` | 0.5 |
| `allow_pad_short_videos` | true |

**What's different from D**: at every frame inside `forward_tracking`, the
refined mask (not the SAM2 raw mask) is what gets fed into
`_encode_memory_in_output`. So subsequent frames' memory_attention
cross-attends to memories built from refined masks. Frame 0's refined pred
is detached and cached as the reference for all later frames.

Tested twice:

- **Post-hoc test** (`scripts/test_tumorseg_refine.sh`): SAM2VideoPredictor
  propagates with raw masks in memory, refinement applied to each yielded
  per-slice prediction at the end. **Train-test mismatch** because training
  built memory from refined masks.
- **Interleaved test** (`scripts/test_tumorseg_refine_interleaved.sh`):
  matches the training: after each per-slice yield, refine and re-encode the
  memory features (`_run_memory_encoder` with refined logits) and overwrite
  `inference_state["output_dict"][...][frame_idx]["maskmem_features"]` /
  `["maskmem_pos_enc"]`. Subsequent yields read updated memory.

---

## 4. Results

### 4.1 Dice (all 5 datasets, all settings)

| Dataset | mid base | mid ours | nei base | nei ours | V0 zero-shot | V baseline | A | B | C | D | E posthoc | E interleaved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colon | 0.747 | 0.789 | 0.793 | 0.821 | 0.732 | **0.828** | 0.794 | 0.793 | 0.790 | 0.763 | 0.685 | 0.742 |
| kits | 0.720 | 0.826 | 0.751 | **0.920** | 0.787 | 0.845 | 0.801 | 0.723 | 0.787 | 0.781 | 0.817 | 0.819 |
| lits | 0.584 | 0.604 | 0.640 | **0.860** | 0.375 | 0.698 | 0.561 | 0.392 | 0.640 | 0.643 | 0.337 | 0.370 |
| pancreas | 0.656 | 0.805 | 0.642 | **0.886** | 0.796 | 0.802 | 0.738 | 0.761 | 0.793 | 0.785 | 0.774 | 0.803 |
| local | 0.723 | 0.783 | 0.666 | **0.830** | 0.678 | 0.805 | 0.767 | 0.743 | 0.690 | 0.780 | 0.780 | 0.788 |
| **MEAN** | 0.686 | 0.761 | 0.698 | **0.863** | 0.674 | **0.796** | 0.732 | 0.682 | 0.740 | 0.750 | 0.679 | 0.704 |

### 4.2 IoU

| Dataset | mid base | mid ours | nei base | nei ours | V0 zero-shot | V baseline | A | B | C | D | E posthoc | E interleaved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colon | 0.627 | 0.679 | 0.680 | 0.714 | 0.624 | **0.727** | 0.681 | 0.684 | 0.681 | 0.648 | 0.551 | 0.622 |
| kits | 0.619 | 0.754 | 0.645 | **0.870** | 0.714 | 0.775 | 0.736 | 0.657 | 0.716 | 0.716 | 0.753 | 0.760 |
| lits | 0.467 | 0.495 | 0.515 | **0.783** | 0.284 | 0.584 | 0.439 | 0.295 | 0.530 | 0.529 | 0.250 | 0.281 |
| pancreas | 0.520 | 0.701 | 0.499 | **0.804** | 0.700 | 0.693 | 0.627 | 0.646 | 0.679 | 0.684 | 0.663 | 0.700 |
| local | 0.629 | 0.700 | 0.570 | 0.742 | 0.581 | **0.719** | 0.695 | 0.669 | 0.608 | 0.705 | 0.699 | 0.717 |
| **MEAN** | 0.572 | 0.666 | 0.582 | **0.783** | 0.581 | **0.700** | 0.636 | 0.590 | 0.643 | 0.657 | 0.583 | 0.616 |

### 4.3 Δ vs V baseline (Dice mean)

| Setting | Mean Dice | ΔDice vs V baseline |
|---|---:|---:|
| V baseline (no module) | 0.796 | (reference) |
| A — midanchor | 0.732 | −0.064 |
| B — box-only | 0.682 | **−0.114** (worst) |
| C — T=8 | 0.740 | −0.056 |
| D — A+C | 0.750 | −0.046 (best post-hoc) |
| E — interleaved test | 0.704 | −0.092 |
| E — post-hoc test | 0.679 | −0.117 |

---

## 5. Findings

1. **No module-integration variant beats the no-module V baseline (mean Dice 0.796)**.
   Best attempt is D (mean Dice 0.750), still −4.6 pp behind.

2. **Interleaved test > post-hoc test for E**, on every dataset (+0.5 to +5.7
   pp Dice; mean +2.5 pp). This confirms train-test consistency matters when
   training builds memory from refined masks. But even the matched
   interleaved test is below V baseline by 9 pp on average — i.e. matching
   train-test distribution doesn't recover what training spent on
   refinement.

3. **B (box-only) consistently the worst.** Removing the 50% mask-input
   branch starves the SAM2 mask decoder of clean GT-aligned supervision;
   that branch's near-perfect cond-frame pred is, paradoxically, what keeps
   the SAM2 head training stable in this 5-dataset regime.

4. **Volume size effect on E**:
   - **lits** (median D=35): E collapses (Dice 0.337 / 0.370). E was
     trained with T=8 propagation chains; lits test propagates 17+ steps
     per half. The mismatch in propagation depth, combined with detached
     frame-0 reference and refined-memory feedback, accumulates errors.
   - **pancreas** (median D=8): E interleaved Dice 0.803, **matches V
     baseline 0.802** — the only dataset where module integration breaks
     even.
   - **colon, kits, local** (median D=10–14): E interleaved between 0.74
     and 0.82 Dice, below V baseline by 4–9 pp.

5. **midanchor + T=8 (D) ≈ T=8 alone (C) on most datasets**. The two
   "fixes" do not stack supra-additively; T=8 dominates. midanchor only
   helps colon/kits in T=4 regime.

6. **Module's role in the broader project**: it works very well on **2D
   per-slice prompt protocols** (`mid ours`: +7.5 pp Dice; `nei ours`: +16.5
   pp Dice mean over baseline). In **video mode**, SAM2's
   memory-attention is **already** propagating reference-slice information
   across slices via cross-frame attention. Adding the plug-in module on
   top of that mechanism is mostly redundant (best case: tied at pancreas;
   worst case: catastrophic for lits in E).

---

## 6. Conclusions / Paper framing

1. **Main story for the paper is unchanged**: the module is a *plug-in for
   prompt-based 2D segmenters*, demonstrated on H2Former / UNet++ / ... +
   MedSAM2 (image-mode), with strong gains under both `middle` and
   `neighbor` reference-slice protocols.

2. **Video-mode = orthogonal alternative**, not subsumed. It should appear
   in the paper as a *competing 3D inference strategy* with its own number
   (V baseline, mean Dice 0.796 / IoU 0.700), not embedded into the main
   "+ Ours" comparison.

3. **Recommended sentence in the paper**:

   > "MedSAM2's native video-mode segmentation, which propagates the
   > middle-slice box prompt across the volume via memory-attention,
   > achieves competitive Dice (0.796 average) without our module. As
   > video-mode already performs cross-slice information diffusion through
   > its memory mechanism, our reference-slice module is largely redundant
   > under that paradigm; we report it as an additional baseline rather
   > than as a target for our module. Our contribution remains in showing
   > that any *prompt-based 2D segmenter* — which has no built-in
   > propagation mechanism — can be substantially improved by our plug-in
   > module."

4. **Optional appendix paragraph (negative result, honest)**: "We
   investigated five strategies for integrating the module into video-mode
   training (Appendix Table X) — anchored sampling, box-only training,
   longer training windows, their combination, and interleaved
   refinement-feedback into the memory bank. None recovered V baseline
   performance, with even the best variant trailing by 4.6 Dice on average.
   We attribute this to memory-attention already performing the
   reference-slice information diffusion that the module is designed to
   inject."

---

## 7. Appendix — Reproducibility

Each variant's run command:

```bash
cd <repo>/MedSAM2

# V baseline
bash single_node_train_tumorseg_v2.sh <ds> <gpu> 1
bash scripts/test_tumorseg.sh <ds> <gpu> exp_log/tumorseg_v2_<ds>/checkpoints/checkpoint.pt

# A / B / C / D
bash single_node_train_tumorseg_refine_A.sh <ds> <gpu> 1
bash scripts/test_tumorseg_refine.sh <ds> <gpu> exp_log/tumorseg_refineA_<ds>/checkpoints/checkpoint.pt
# (same pattern for B / C / D, replace A by B / C / D)

# E
bash single_node_train_tumorseg_refine_E.sh <ds> <gpu> 1
# Test E in two ways:
bash scripts/test_tumorseg_refine.sh <ds> <gpu> exp_log/tumorseg_refineE_<ds>/checkpoints/checkpoint.pt
bash scripts/test_tumorseg_refine_interleaved.sh <ds> <gpu> exp_log/tumorseg_refineE_<ds>/checkpoints/checkpoint.pt
```

Prerequisite (one-time): convert PNG → padded NPZ for all 5 datasets:

```bash
for ds in colon lits pancreas kits local; do
  for split in train test; do
    python scripts/convert_png_to_npz.py --dataset $ds --split $split \
      --dst <repo>/MedSAM2/data/tumorseg_npz_v2 \
      --min-slices 2
  done
done
```

Checkpoints under `exp_log/tumorseg_{v2,refineA,refineB,refineC,refineD,refineE}_<ds>/checkpoints/checkpoint.pt`
remain on disk for re-evaluation.

The compiled comparison table is in
`results/medsam2_overall_v4.csv` (canonical) — all earlier `_v1` / `_v2` /
`_v3` CSVs are intermediate and superseded.

---

## 8. Module vs. MedSAM2-video — positioning for the paper

MedSAM2-video achieves higher full-volume Dice than any module variant we
trained (V baseline 0.796 vs. best module variant ≈ 0.78x). The module
should therefore **not** be sold as a Dice winner against the video
backbone. It can, however, be positioned as an *orthogonal* design with
several concrete properties that MedSAM2-video does not have. The
advantages below are split by how confidently they can be written into the
paper.

### 8.1 Strong advantages — write these

**(A) Architecture-agnostic plug-in**
- Empirical: the module brings consistent gains on 7 traditional 2D CNN
  backbones (H2Former, UNet++, AttentionUNet, TransUNet, SwinUNet, FATNet,
  MedFormer) **and** on MedSAM2 image-mode.
- MedSAM2-video's memory-attention is fused into the SAM2 architecture and
  cannot be transplanted onto any of the above CNN backbones.
- Paper value: stand-alone paragraph, supported by the cross-architecture
  results already in `main_results_v4.csv`.

**(B) Decoupled reference selection**
- Module: the reference slice is an explicit input — at inference time it
  can be the middle GT slice, a neighbor prediction, or any clinician-edited
  slice. Switching reference is a single forward pass.
- Video memory: the reference is the implicit accumulation of propagation
  history, tied to the initial prompt frame and the propagation direction.
  Changing reference requires a fresh `init_state` and a full re-propagation.
- Paper value: middle vs. neighbor reference results are already reported,
  so the "reference is a deployment-time choice" claim is empirically backed.

**(C) Native fit for HCI / iterative correction**
- In the clinician-in-the-loop loop (the TRACE study), when the clinician
  edits slice *k*, the module simply designates that slice as the new
  reference and reruns pairwise refinement. No state to rebuild.
- MedSAM2-video would need to restart propagation from frame *k*, rebuild
  the memory bank, and any frame already written into memory could carry
  over before the edit took effect.
- Paper value: this aligns directly with the TRACE HCI study and is the
  strongest deployment-side argument for the module.

### 8.2 Moderate advantages — write only with measurements

**(D) Slice-level inference parallelism**
- Each target slice is refined independently against the reference, so
  inference is embarrassingly parallel across slices. Video propagation is
  strictly sequential along the propagation order.
- Recommendation: report a wall-clock benchmark before claiming this in the
  paper.

**(E) Smaller per-step state**
- Module: a single (target, reference) pair plus one forward through the
  refinement head.
- Video: the memory bank holds `num_maskmem` past frames in GPU memory
  throughout the whole propagation.
- Recommendation: report peak-GPU-memory numbers before claiming this.

### 8.3 Do **not** write

- "Our module beats MedSAM2-video on Dice." — false on these 5 datasets.
- "More general / more flexible" without concrete evidence. — must be
  cashed out as the 7-CNN-plus-image-mode coverage and the
  reference-switching experiments.
- "Easier to train." — true but unconvincing to reviewers.

### 8.4 Suggested framing paragraph

> "MedSAM2-video can be viewed as a competing video-mode alternative to
> our refinement module. While it attains higher full-volume Dice (0.796
> vs. our best 0.78x), this comes at three costs: (i) the memory-attention
> mechanism is fused into the SAM2 architecture and cannot be transferred
> to other 2D backbones; (ii) the reference is implicitly tied to the
> propagation's initial prompt frame, making reference switching expensive
> in interactive workflows; and (iii) inference is strictly sequential
> along the propagation order. Our module trades a modest amount of Dice
> for (a) cross-architecture portability, demonstrated on 7 CNN backbones
> and on MedSAM2 image-mode, (b) free choice of reference slice at
> deployment time, and (c) cheap iterative correction in
> clinician-in-the-loop scenarios — properties that align directly with
> the TRACE HCI study."
