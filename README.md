# Region-Aware Image Enhancement using OneFormer Guidance

This project implements a region-aware image enhancement framework that leverages semantic segmentation guidance from a pretrained OneFormer model and a UNet-based enhancement network.

Semantic information is extracted offline and used as fixed region masks during training and inference, enabling efficient and stable enhancement without running a segmentation model in the training loop.

The framework supports:
- ADE20K (synthetic degradations)
- LOL / paired low-light datasets
- Baseline (RGB-only) and region-aware variants


## Key Idea

1. Use OneFormer to segment images into semantic regions.
2. Group fine-grained classes into three coarse regions:
   - Human
   - Vegetation
   - Structure
3. Precompute region masks offline.
4. Concatenate RGB input with region masks.
5. Train a UNet to enhance degraded images.

This design:
- avoids segmentation noise during training
- reduces computational cost
- keeps the enhancement network fully CNN-based


## Project Structure

scripts/
├── degrade_ade20k.py        # Generate degraded ADE20K images
├── precompute_masks.py     # Offline OneFormer mask generation
├── region_masks.py         # Semantic grouping logic
├── plot_curves.py          # Plot PSNR / SSIM / Loss curves
├── visualize_results.py    # Save qualitative results
├── README.txt              # Short command reference

src/
├── datasets/
│   ├── PairedMaskDataset.py   # Region-aware dataset (RGB + masks)
│   └── PairedRGBDataset.py    # Baseline RGB-only dataset
├── models/
│   ├── unet.py                # Enhancement network
│   └── oneformer_wrapper.py   # OneFormer inference (offline only)
├── train/
│   ├── train_unet.py          # Unified training script
│   └── README.txt
└── utils/
    ├── metrics.py             # PSNR / SSIM
    └── perceptual_loss.py     # VGG-based perceptual loss


## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- opencv-python
- pillow
- matplotlib
- tqdm
- transformers

GPU (CUDA) is strongly recommended, especially for mask precomputation.


## Pipeline Overview

### Step 1 — Generate Degraded ADE20K Images

Create synthetic low-quality inputs (low-light, noise, blur, JPEG).

python scripts/degrade_ade20k.py \
  --ade_root /path/to/ADEChallengeData2016 \
  --split training \
  --max_images 600


### Step 2 — Precompute Semantic Region Masks (Offline)

Run OneFormer once to generate semantic masks.

python scripts/precompute_masks.py \
  --out_root /path/to/oneformer_masks \
  --splits \
    train:/path/to/Dataset/train/input,\
    val:/path/to/Dataset/val/input

Each image produces a .npy file of shape:
(3, H, W)


### Step 3 — Train the Enhancement Network

A single training script supports all configurations.

ADE20K – Baseline
python src/train/train_unet.py \
  --train_input Dataset/train/input \
  --train_target Dataset/train/target \
  --val_input Dataset/val/input \
  --val_target Dataset/val/target \
  --ckpt_dir checkpoints_ade_baseline \
  --plot_dir plots_ade_baseline


ADE20K – Region-Aware
python src/train/train_unet.py \
  --use_masks \
  --use_perceptual \
  --train_input Dataset/train/input \
  --train_target Dataset/train/target \
  --train_masks /path/to/oneformer_masks/train \
  --val_input Dataset/val/input \
  --val_target Dataset/val/target \
  --val_masks /path/to/oneformer_masks/val \
  --ckpt_dir checkpoints_ade \
  --plot_dir plots_ade


### Step 4 — Visualize Results

Input (degraded) | Output (enhanced) | Ground Truth

python scripts/visualize_results.py \
  --dataset ade \
  --ckpt checkpoints_ade/unet_epoch_18.pt \
  --data_root Dataset \
  --out_dir outputs/visuals


### Step 5 — Plot Training Curves

python scripts/plot_curves.py \
  --plot_dir plots_ade \
  --out_dir figures_ade \
  --dataset ADE20K \
  --plot_train_val


## Design Choices & Notes

- Semantic segmentation is fully offline.
- The enhancement network never sees raw class IDs.
- Region grouping is intentionally coarse to reduce noise.
- Perceptual loss is used only in region-aware training.
- The same pipeline works for ADE20K and LOL.


## Reproducibility

All scripts are parameterized, dataset-agnostic, and runnable locally or on Google Colab.

See scripts/README.txt for a concise command reference.


## Summary

This project demonstrates a clean separation between preprocessing (segmentation),
training (enhancement), and evaluation (metrics and visualization).

The result is an efficient, interpretable, and reproducible
region-aware image enhancement framework.
