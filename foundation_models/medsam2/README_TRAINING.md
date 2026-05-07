# MedSAM2 2D训练和测试指南

本指南说明如何使用MedSAM2进行2D医学图像分割的训练和测试。

## 文件说明

1. **`training/dataset/medsam2_2d_dataset.py`**: 数据集类，支持middle和neighbor两种参考slice模式
2. **`train_medsam2_2d.py`**: 训练脚本
3. **`test_medsam2_2d.py`**: 测试脚本

## 数据准备

确保你的数据目录结构如下：

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
└── local/
```

### annotation_dict格式

**middle模式** (`annotation_dict_middle.json`):
```json
{
  "colon_001": {
    "middle_filename": "050.png",
    "ct_path": "/path/to/CT/colon_001/050.png",
    "mask_path": "/path/to/Mask/colon_001/050.png"
  }
}
```

**neighbor模式** (`annotation_dict_neighbor.json`):
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

## 训练

### 基本用法

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

### 参数说明

- `--data`: 数据集名称 (kits, pancreas, lits, colon, local)
- `--ref_type`: 参考slice类型 (middle, neighbor)
- `--checkpoint`: 预训练模型路径
- `--cfg`: 模型配置文件（默认: sam2/configs/sam2.1/sam2.1_hiera_t.yaml）
- `--batch_size`: 批次大小（默认: 4）
- `--num_epochs`: 训练轮数（默认: 80）
- `--lr`: 学习率（默认: 1e-4）
- `--weight_decay`: 权重衰减（默认: 0.01）
- `--image_size`: 图像尺寸（默认: 512）
- `--device`: 设备（默认: cuda:0）
- `--use_amp`: 使用混合精度训练
- `--work_dir`: 输出目录（默认: ./work_dir）
- `--resume`: 恢复训练的checkpoint路径

### 训练示例

**使用middle slice作为参考**:
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

**使用neighbor slice作为参考**:
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

**恢复训练**:
```bash
python train_medsam2_2d.py \
    --data colon \
    --ref_type middle \
    --checkpoint checkpoints/MedSAM2_latest.pt \
    --resume work_dir/MedSAM2-2D-colon-middle/20231201-120000/latest.pth \
    --device cuda:0
```

### 训练输出

训练过程中会在`work_dir`目录下创建以下结构：

```
work_dir/
└── MedSAM2-2D-{dataset}-{ref_type}/
    └── {timestamp}/
        ├── latest.pth          # 最新checkpoint
        ├── best.pth            # 最佳checkpoint
        ├── loss_curves.png     # Loss曲线
        ├── args.json           # 训练参数
        └── train_script.py     # 训练脚本备份
```

## 测试

### 基本用法

```bash
cd <repo>/foundation_models/medsam2
python test_medsam2_2d.py \
    --data colon \
    --ref_type middle \
    --checkpoint work_dir/MedSAM2-2D-colon-middle/20231201-120000/best.pth \
    --device cuda:0
```

### 参数说明

- `--data`: 数据集名称 (kits, pancreas, lits, colon, local)
- `--ref_type`: 参考slice类型 (middle, neighbor)
- `--checkpoint`: 模型checkpoint路径（可以是训练checkpoint或预训练模型）
- `--cfg`: 模型配置文件（默认: sam2/configs/sam2.1/sam2.1_hiera_t.yaml）
- `--device`: 设备（默认: cuda:0）
- `--image_size`: 图像尺寸（默认: 512）

### 测试输出

测试结果会保存到`results/results_{dataset}_{ref_type}.json`:

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

## 五个数据集的训练

你可以分别为每个数据集训练模型：

```bash
# Colon数据集
python train_medsam2_2d.py --data colon --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt

# Kits数据集
python train_medsam2_2d.py --data kits --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt

# Lits数据集
python train_medsam2_2d.py --data lits --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt

# Pancreas数据集
python train_medsam2_2d.py --data pancreas --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt

# Local数据集
python train_medsam2_2d.py --data local --ref_type middle --checkpoint checkpoints/MedSAM2_latest.pt
```

同样，你也可以使用neighbor模式训练。

## 注意事项

1. **内存使用**: 训练过程中每个batch会为每张图像创建一个SAM2ImagePredictor实例，这可能会占用较多内存。如果遇到OOM错误，可以减小batch_size。

2. **训练速度**: 由于使用了predictor进行训练，训练速度可能不如直接使用模型内部方法。如果需要更快的训练速度，可以考虑使用MedSAM2的完整训练框架。

3. **图像尺寸**: 默认使用512x512。如果原始图像尺寸不同，数据集类会自动调整。

4. **Box Prompt**: 训练时从参考slice的GT Mask中提取bounding box作为prompt，这与MedSAM的训练方式一致。

5. **Loss函数**: 使用Dice Loss + BCE Loss的组合。

## 与MedSAM代码的对比

本实现参考了MedSAM的训练代码（`MedSAM/train_one_gpu.py`和`MedSAM/My_utils.py`），主要区别：

1. **模型**: 使用MedSAM2 (SAM2) 替代MedSAM (SAM)
2. **图像尺寸**: MedSAM使用1024x1024，MedSAM2使用512x512
3. **推理方式**: MedSAM直接调用模型forward，MedSAM2使用SAM2ImagePredictor
4. **数据格式**: MedSAM使用float32 [0,1]，MedSAM2使用uint8 [0,255]

## 故障排除

1. **找不到模块**: 确保你在MedSAM2目录下运行脚本，或者正确设置了PYTHONPATH。

2. **CUDA OOM**: 减小batch_size或使用更小的图像尺寸。

3. **找不到annotation文件**: 确保训练/测试数据目录下有对应的annotation_dict_middle.json或annotation_dict_neighbor.json文件。

4. **Checkpoint加载失败**: 如果是训练checkpoint，确保checkpoint中包含"model"键。如果是预训练模型，直接使用checkpoint路径即可。
