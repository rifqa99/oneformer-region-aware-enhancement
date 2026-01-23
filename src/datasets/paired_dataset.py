from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T
import numpy as np
import torch


class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, mask_dir, size=None):
        self.mask_dir = mask_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.files = sorted(os.listdir(input_dir))
        self.transform = T.ToTensor()
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        inp = Image.open(os.path.join(self.input_dir, name)).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, name)).convert("RGB")
        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(name)[0] + ".npy"
        )
        masks = np.load(mask_path)                 # (3, H, W)
        masks = torch.from_numpy(masks).float()

        if self.size:
            inp = inp.resize(self.size)
            tgt = tgt.resize(self.size)

        img = self.transform(inp)
        x = torch.cat([img, masks], dim=0)   # (6, H, W)
        return x, self.transform(tgt)
