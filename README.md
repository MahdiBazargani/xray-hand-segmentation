# Hand X-ray Segmentation

Deep learning pipeline for segmenting hands in X-ray images using U-Net with attention gates.

## Features

- ✅ 2D U-Net with Attention Gates
- ✅ Albumentations-based augmentation
- ✅ Patch-based training with sliding window inference
- ✅ Mixed precision training (AMP)
- ✅ CPU/GPU compatible
- ✅ Deep supervision support
- ✅ Test-time augmentation (TTA)
- ✅ TensorBoard logging
- ✅ Multiple loss functions (Dice, BCE, Focal, Tversky)

## Dataset Structure

```
dataset/
│
├── train/
│   ├── images/
│   │   ├── 1405.png
│   │   ├── 1407.png
│   │   └── ...
│   │
│   └── masks/
│       ├── 1405_mask.png
│       ├── 1407_mask.png
│       └── ...
│
└── val/
    ├── images/
    │   ├── 1377.png
    │   └── ...
    │
    └── masks/
        ├── 1377_mask.png
        └── ...
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Basic training:
```bash
python train.py \
    --data_dir ./dataset \
    --output_dir ./outputs \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4
```

Advanced training with options:
```bash
python train.py \
    --data_dir ./dataset \
    --output_dir ./outputs \
    --model unet_attention \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --loss bce_dice \
    --deep_supervision \
    --image_size 512 \
    --patch_size 512 \
    --base_channels 64 \
    --dropout 0.1
```

Training on CPU:
```bash
python train.py --data_dir ./dataset --output_dir ./outputs --cpu
```

### Testing

Basic inference:
```bash
python test.py \
    --input_dir ./test_images \
    --output_dir ./predictions \
    --checkpoint ./outputs/best_model.pth
```

With sliding window for large images:
```bash
python test.py \
    --input_dir ./test_images \
    --output_dir ./predictions \
    --checkpoint ./outputs/best_model.pth \
    --use_sliding_window \
    --patch_size 512 \
    --stride 256
```

With test-time augmentation:
```bash
python test.py \
    --input_dir ./test_images \
    --output_dir ./predictions \
    --checkpoint ./outputs/best_model.pth \
    --use_tta \
    --n_tta 5
```

With post-processing and overlay visualization:
```bash
python test.py \
    --input_dir ./test_images \
    --output_dir ./predictions \
    --checkpoint ./outputs/best_model.pth \
    --post_process \
    --save_overlay
```

## Training Parameters

### Data Parameters
- `--data_dir`: Path to dataset directory (required)
- `--output_dir`: Output directory (default: ./outputs)
- `--image_size`: Image size for training (default: 512)
- `--patch_size`: Patch size for patch-based training (default: 512)

### Model Parameters
- `--model`: Model architecture (choices: unet, unet_attention; default: unet_attention)
- `--base_channels`: Base number of channels (default: 64)
- `--dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `--epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--optimizer`: Optimizer (choices: adam, adamw, sgd; default: adamw)
- `--scheduler`: LR scheduler (choices: cosine, step, poly, none; default: cosine)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--loss`: Loss function (choices: dice, bce, bce_dice, focal, tversky; default: bce_dice)
- `--deep_supervision`: Enable deep supervision

### Other Parameters
- `--num_workers`: Number of data loader workers (default: 4)
- `--val_interval`: Validation interval in epochs (default: 1)
- `--save_interval`: Save interval in epochs (default: 10)
- `--patience`: Early stopping patience (default: 20)
- `--cpu`: Force CPU usage

## Test Parameters

### Required Parameters
- `--input_dir`: Path to directory containing test images
- `--output_dir`: Path to output directory for masks
- `--checkpoint`: Path to model checkpoint

### Model Parameters
- `--model`: Model architecture (default: unet_attention)
- `--base_channels`: Base number of channels (default: 64)

### Inference Parameters
- `--resize_to`: Resize images to this size (None = keep original)
- `--use_sliding_window`: Use sliding window inference for large images
- `--patch_size`: Patch size for sliding window (default: 512)
- `--stride`: Stride for sliding window (default: 256)
- `--use_tta`: Use test-time augmentation
- `--n_tta`: Number of TTA iterations 1-5 (default: 3)
- `--threshold`: Threshold for binary mask (default: 0.5)
- `--post_process`: Apply morphological post-processing
- `--save_overlay`: Save overlay visualization

## Project Structure

```
.
├── dataset.py              # Dataset classes
├── augmentations.py        # Albumentations pipelines
├── train.py               # Training script
├── test.py                # Inference script
├── requirements.txt       # Dependencies
├── models/
│   └── unet.py           # U-Net architecture
└── utils/
    ├── losses.py         # Loss functions
    └── metrics.py        # Evaluation metrics
```

## Augmentation Pipeline

### Training Augmentations
- Geometric: ShiftScaleRotate, ElasticTransform, GridDistortion
- Intensity: CLAHE, RandomBrightnessContrast, RandomGamma
- Noise: GaussNoise, GaussianBlur, MotionBlur
- Dropout: CoarseDropout
- Flip: HorizontalFlip, VerticalFlip

### Validation Augmentations
- Minimal preprocessing (resize + pad only)

## Loss Functions

Available loss functions:
- **Dice Loss**: Optimizes Dice coefficient
- **BCE Loss**: Binary cross-entropy
- **BCE+Dice Loss**: Combined loss (default)
- **Focal Loss**: Handles class imbalance
- **Tversky Loss**: Generalization of Dice
- **Deep Supervision Loss**: Multi-scale supervision

## Metrics

- Dice Score (Sørensen-Dice coefficient)
- IoU Score (Jaccard Index)
- Pixel Accuracy
- Precision
- Recall

## Tips for Best Results

1. **Large Images**: Use `--use_sliding_window` in test.py for images > 1024px
2. **Better Accuracy**: Enable `--use_tta` for test-time augmentation
3. **Clean Results**: Use `--post_process` to remove small artifacts
4. **Deep Supervision**: Add `--deep_supervision` in training for better feature learning
5. **GPU Memory**: Reduce `--batch_size` or `--patch_size` if OOM errors occur
6. **CPU Training**: Use `--cpu` flag if no GPU available (slower)

## Citation

If you use this code, please cite the original datasets:
- RSNA Pediatric Bone Age Challenge (2017)
- Hand Masks Dataset: https://zenodo.org/records/7611677

## License

MIT License