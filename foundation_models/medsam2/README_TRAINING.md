# MedSAM2 2D Training and Testing Guide

This guide describes how to train and test MedSAM2 for 2D medical-image segmentation.

## Files

1. **`training/dataset/medsam2_2d_dataset.py`**: dataset class; supports both 'middle' and 'neighbor' reference-slice modes.
2. **`train_medsam2_2d.py`**: training script.
3. **`test_medsam2_2d.py`**: test script.

## Data preparation

Make sure your data directory looks like this:

```
2D_data/
├── colon/
│   ├── train/
│   │   ├── CT/
│   │   │   └── colon_001/
│   │   │       ├── 001.png
│   │   │       └── ...
│   │   ├── Mask/
│   │   │   └── colon_001/
│   │   │       ├── 001.png
│   │   │       └── ...
│   │   ├── annotation_dict_middle.json
│   │   └── annotation_dict_neighbor.json
│   └── test/
│       ├── CT/
│       ├── Mask/
│       ├── annotation_dict_middle.json
│       └── annotation_dict_neighbor.json
├── kits/
├── lits/
├── pancreas/
└── colon/
```

### annotation_dict format

**middle mode** (`annotation_dict_middle.json`):
```json
{
  "colon_001": {
    "middle_filename": "050.png",
    "ct_path": "/path/to/CT/colon_001/050.png",
    "mask_path": "/path/to/Mask/colon_001/050.png"
  }
}
```

**neighbor mode** (`annotation_dict_neighbor.json`):
```json
{
  "colon_001": {
    "001.png": {
      "ct_path": "/path/to/CT/colon_001/001.png",
      "mask_path": "/path/to/Mask/colon_001/001.png",
      "ref_filename": "002.png",
      "ref_ct_path": "/path/to/CT/colon_001/002.png",
      "ref_mask_path": "/path/to/Mask/colon_001/002.png"
    }
  }
}
```

## Training

### Basic usage

```bash
cd <repo>/foundation_models/medsam2
python train_medsam2_2d.py \
    --data colon \
    --ref_type middle \
    --checkpoint checkpoints/MedSAM2_latest.pt \
    --batch_size 4 \
    --num_epochs 80 \
    --lr 1e-4 \
    --device cuda:0
```

### CLI flags

- `--data`: dataset name (kits, pancreas, lits, colon)
- `--ref_type`: reference-slice mode (middle, neighbor)
- `--checkpoint`: pretrained-model path
- `--cfg`: model config file (default: sam2/configs/sam2.1/sam2.1_hiera_t.yaml)
- `--batch_size`: batch size (default: 4)
- `--num_epochs`: number of training epochs (default: 80)
- `--lr`: learning rate (default: 1e-4)
- `--weight_decay`: weight decay (default: 0.01)
- `--image_size`: image size (default: 512)
- `--device`: device (default: cuda:0)
- `--use_amp`: enable mixed-precision training
- `--work_dir`: output directory (default: ./work_dir)
- `--resume`: path to a checkpoint for resuming training

### Training examples

**Using the middle slice as the reference:**
```bash
python train_medsam2_2d.py \
    --data colon \
    --ref_type middle \
    --checkpoint checkpoints/MedSAM2_latest.pt \
    --batch_size 4 \
    --num_epochs 80 \
    --lr 1e-4 \
    --device cuda:0
```

**Using the neighbor slice as the reference:**
```bash
python train_medsam2_2d.py \
    --data colon \
    --ref_type neighbor \
    --checkpoint checkpoints/MedSAM2_latest.pt \
    --batch_size 4 \
    --num_epochs 80 \
    --lr 1e-4 \
    --device cuda:0
```

**Resume training:**
```bash
python train_medsam2_2d.py \
    --data colon \
    --ref_type middle \
    --checkpoint checkpoints/MedSAM2_latest.pt \
    --resume work_dir/MedSAM2-2D-colon-middle/20231201-120000/latest.pth \
    --device cuda:0
```

### Training outputs

Training writes the following layout under `work_dir`:

```
work_dir/
└── MedSAM2-2D-{dataset}-{ref_type}/
    └── {timestamp}/
        ├── latest.pth          # latest checkpoint
        ├── best.pth            # best checkpoint
        ├── loss_curves.png     # loss curves
        ├── args.json           # training arguments
        └── train_script.py     # backup of the training script
```

## Testing

### Basic usage

```bash
cd <repo>/foundation_models/medsam2
python test_medsam2_2d.py \
    --data colon \
    --ref_type middle \
    --checkpoint work_dir/MedSAM2-2D-colon-middle/20231201-120000/best.pth \
    --device cuda:0
```

### CLI flags

- `--data`: dataset name (kits, pancreas, lits, colon)
- `--ref_type`: reference-slice mode (middle, neighbor)
- `--checkpoint`: model checkpoint path (training checkpoint or pretrained model)
- `--cfg`: model config file (default: sam2/configs/sam2.1/sam2.1_hiera_t.yaml)
- `--device`: device (default: cuda:0)
- `--image_size`: image size (default: 512)

### Test outputs

Results are written to `results/results_{dataset}_{ref_type}.json`:

```json
{
  "dataset": "colon",
  "ref_type": "middle",
  "checkpoint": "/path/to/checkpoint.pth",
  "avg_iou": 0.85,
  "avg_dice": 0.90,
  "num_samples": 100
}
```

## Training on each of the four datasets

You can train one model per dataset:

```bash
# Colon dataset
python train_medsam2_2d.py --data colon --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt

# KiTS dataset
python train_medsam2_2d.py --data kits --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt

# LiTS dataset
python train_medsam2_2d.py --data lits --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt

# Pancreas dataset
python train_medsam2_2d.py --data pancreas --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt
```

The neighbor reference mode is invoked the same way.

## Notes

1. **Memory usage**: training builds one SAM2ImagePredictor per image per batch, which can be memory-heavy. Reduce `batch_size` if you hit OOM.

2. **Training speed**: routing through the predictor is slower than calling the model's internals directly. For faster training, consider MedSAM2's full training framework.

3. **Image size**: default is 512x512. The dataset class resizes raw inputs as needed.

4. **Box prompt**: training extracts the bounding box from the reference slice's GT mask, matching MedSAM's training recipe.

5. **Loss**: a combination of Dice loss and BCE loss.

## Comparison with the MedSAM code

This implementation derives from MedSAM's training code (`MedSAM/train_one_gpu.py` and `MedSAM/My_utils.py`); key differences:

1. **Model**: MedSAM2 (SAM2) replaces MedSAM (SAM).
2. **Image size**: MedSAM uses 1024x1024; MedSAM2 uses 512x512.
3. **Inference**: MedSAM calls the model's forward directly; MedSAM2 goes through SAM2ImagePredictor.
4. **Data format**: MedSAM expects float32 in [0, 1]; MedSAM2 expects uint8 in [0, 255].

## Troubleshooting

1. **Module not found**: run the script from the MedSAM2 directory or set PYTHONPATH correctly.

2. **CUDA OOM**: reduce `batch_size` or shrink the image size.

3. **Annotation file missing**: make sure the training/test data directory contains `annotation_dict_middle.json` or `annotation_dict_neighbor.json`.

4. **Checkpoint loading failed**: training checkpoints must contain a "model" key. Pretrained models can be loaded directly.
