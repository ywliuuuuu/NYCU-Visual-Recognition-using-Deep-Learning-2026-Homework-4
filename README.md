# NYCU Visual Recognition using Deep Learning 2026 — Homework 4

* **Student ID**: 313553049
* **Name**: 劉怡妏

## Introduction

This repository contains the implementation for HW4: **All-in-One Image Restoration** (rain + snow removal) using a single unified model.

We adopt **PromptIR** (Potlapalli et al., NeurIPS 2023), a Restormer-style Transformer encoder-decoder with learnable Prompt Blocks in the decoder that dynamically encode degradation-specific information. The model is trained from scratch on 3,200 image pairs (1,600 rain + 1,600 snow) using patch-based training with mixed-precision (fp16) and cosine annealing scheduling.

## File Structure

```
.
├── train_hw4.py        # Training script (PyTorch Lightning)
├── inference_hw4.py    # Inference → pred.npz
├── dataset_hw4.py      # RainSnowDataset (train/val split)
├── net/
│   └── model.py        # PromptIR architecture
├── utils/              # Schedulers, image utilities, SSIM
├── requirements.txt
└── train_ckpt_hw4/     # Saved checkpoints (not included in repo)
```

## Environment Setup

```bash
# Install PyTorch with CUDA (adjust version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_hw4.py \
  --data_dir path/to/hw4_release_dataset/train \
  --epochs 150 \
  --batch_size 4 \
  --num_workers 4
```

To resume from a checkpoint:

```bash
python train_hw4.py --data_dir ... --epochs 150 --resume train_ckpt_hw4/last.ckpt
```

### Inference

```bash
python inference_hw4.py \
  --test_dir path/to/test/degraded \
  --ckpt train_ckpt_hw4/epoch=148-val_psnr=29.27.ckpt \
  --output pred.npz \
  --tta
```

`--tta` enables horizontal flip test-time augmentation (recommended).

## Performance Snapshot
![Leaderboard Screenshot](assets/leaderboard.png)