from PIL import Image
import os
import torchvision.transforms as T
import torch
import numpy as np
import cv2


class ADEPairedMaskDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, mask_dir, size=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.mask_dir = mask_dir
        self.size = size

        self.inputs = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        fname = self.inputs[idx]

        x = Image.open(os.path.join(self.input_dir, fname)).convert("RGB")

        base = fname.split("_")[0]
        y = Image.open(os.path.join(
            self.target_dir, base + ".png")).convert("RGB")

        masks = np.load(os.path.join(
            self.mask_dir, os.path.splitext(fname)[0] + ".npy"))

        if self.size is not None:
            x = x.resize(self.size, Image.BILINEAR)
            y = y.resize(self.size, Image.BILINEAR)
            masks = np.stack([
                cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST)
                for m in masks
            ])

        x = torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.array(y)).permute(2, 0, 1).float() / 255.0
        masks = torch.from_numpy(masks).float()

        x = torch.cat([x, masks], dim=0)  # (3+K, H, W)

        return x, y
