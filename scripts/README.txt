README.txt
==========

This project uses 4 scripts for data preparation, mask generation,
visualization, and plotting. The same pipeline works for ADE20K and LOL.

--------------------------------------------------
1) Degrade ADE20K Images
--------------------------------------------------

Generate degraded inputs (low-light, noise, blur, JPEG) from ADE20K.

Training set:
python degrade_ade20k.py \
  --ade_root /content/Datasets/ADEChallengeData2016 \
  --split training \
  --max_images 600

Validation set:
python degrade_ade20k.py \
  --ade_root /content/Datasets/ADEChallengeData2016 \
  --split validation \
  --max_images 100

Output structure:
Dataset/
├── train/input
├── train/target
├── val/input
└── val/target


--------------------------------------------------
2) Precompute OneFormer Semantic Masks
--------------------------------------------------

Masks are grouped into 3 regions:
human, vegetation, structure.
Each image produces a .npy file of shape (3, H, W).

ADE20K:
python precompute_masks.py \
  --out_root /content/drive/MyDrive/ade20k_oneformer_masks \
  --splits \
    train:/content/Datasets/train/input,\
    val:/content/Datasets/val/input,\
    test:/content/Datasets/test/input

LOL / Paired dataset:
python precompute_masks.py \
  --out_root /content/drive/MyDrive/oneformer_masks \
  --splits \
    train:data/train/input,\
    val:data/val/input


--------------------------------------------------
3) Plot Training Curves
--------------------------------------------------

Generate Loss, PSNR, and SSIM plots.

ADE20K:
python plot_curves.py \
  --plot_dir /content/drive/MyDrive/oneformer_ade/plots \
  --out_dir figures_ade \
  --dataset ADE20K \
  --plot_train_val

LOL:
python plot_curves.py \
  --plot_dir /content/drive/MyDrive/oneformer/plots \
  --out_dir figures_lol \
  --dataset LOL


--------------------------------------------------
4) Visualize Results (Input | Output | GT)
--------------------------------------------------

Save qualitative comparison images.

ADE20K:
python visualize_results.py \
  --dataset ade \
  --ckpt /content/drive/MyDrive/checkpoints_ade/unet_epoch_18.pt \
  --data_root /content/Datasets \
  --out_dir outputs/visuals

LOL / Paired dataset:
python visualize_results.py \
  --dataset paired \
  --ckpt /content/drive/MyDrive/checkpoints/unet_epoch_20.pt \
  --input_dir data/val/input \
  --target_dir data/val/target \
  --mask_dir /content/drive/MyDrive/oneformer_masks/val \
  --out_dir outputs/visuals


--------------------------------------------------
Notes
--------------------------------------------------
- CUDA GPU is recommended.
- Mask precomputation is done once.

