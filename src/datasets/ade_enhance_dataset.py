import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset


class ADEEnhancementDataset(Dataset):
    def __init__(self, root, split="train"):
        """root: /content/Dataset
        split: train | val | test  """

        self.split = split
        self.input_dir = os.path.join(root, split, "input")
        self.target_dir = os.path.join(root, split, "target")
        self.seg_dir = os.path.join(root, split, "seg")

        self.inputs = sorted(glob(os.path.join(self.input_dir, "*.png")))
        self.targets = sorted(glob(os.path.join(self.target_dir, "*.png")))

        if split != "test":
            self.segs = sorted(glob(os.path.join(self.seg_dir, "*.png")))
        else:
            self.segs = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp_path = self.inputs[idx]

        # extract base id: 0001_2.png → 0001.png
        base = os.path.basename(inp_path).split("_")[0] + ".png"
        tgt_path = os.path.join(self.target_dir, base)

        inp = cv2.imread(inp_path)[:, :, ::-1]
        tgt = cv2.imread(tgt_path)[:, :, ::-1]

        inp = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
        tgt = torch.from_numpy(tgt).permute(2, 0, 1).float() / 255.0

        if self.split != "test":
            seg_path = os.path.join(self.seg_dir, base)
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            seg = torch.from_numpy(seg).long()
            return inp, seg, tgt
        else:
            return inp, tgt
