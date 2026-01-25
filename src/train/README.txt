This training script (train_unet.py) runs baseline 
and region-aware training for both ADE20K and LOL.

--------------------------------------------------
ADE – Baseline
--------------------------------------------------
python train_unet.py \
  --train_input /content/Datasets/train/input \
  --train_target /content/Datasets/train/target \
  --val_input /content/Datasets/val/input \
  --val_target /content/Datasets/val/target \
  --ckpt_dir checkpoints_ade_baseline \
  --plot_dir plots_ade_baseline


--------------------------------------------------
ADE – Region-aware
--------------------------------------------------
python train_unet.py \
  --use_masks \
  --use_perceptual \
  --train_input /content/Datasets/train/input \
  --train_target /content/Datasets/train/target \
  --train_masks /content/drive/MyDrive/ade20k_oneformer_masks/train \
  --val_input /content/Datasets/val/input \
  --val_target /content/Datasets/val/target \
  --val_masks /content/drive/MyDrive/ade20k_oneformer_masks/val \
  --ckpt_dir checkpoints_ade \
  --plot_dir plots_ade


--------------------------------------------------
LOL – Baseline
--------------------------------------------------
python train_unet.py \
  --train_input data/train/input \
  --train_target data/train/target \
  --val_input data/val/input \
  --val_target data/val/target \
  --ckpt_dir checkpoints_lol_baseline \
  --plot_dir plots_lol_baseline


--------------------------------------------------
LOL – Region-aware
--------------------------------------------------
python train_unet.py \
  --use_masks \
  --use_perceptual \
  --train_input data/train/input \
  --train_target data/train/target \
  --train_masks /content/drive/MyDrive/oneformer_masks/train \
  --val_input data/val/input \
  --val_target data/val/target \
  --val_masks /content/drive/MyDrive/oneformer_masks/val \
  --ckpt_dir checkpoints_lol \
  --plot_dir plots_lol
