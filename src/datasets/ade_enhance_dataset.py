
from torch.utils.data import DataLoader
import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset

IMG_SIZE = (512, 512)  # (W, H)


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

        # map 0001_2.png → 0001.png
        base = os.path.basename(inp_path).split("_")[0] + ".png"
        tgt_path = os.path.join(self.target_dir, base)

        # ---------- input (RGB) ----------
        inp = cv2.imread(inp_path)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        inp = torch.from_numpy(inp).permute(
            2, 0, 1).float() / 255.0   # (3,H,W)

        # ---------- target (RGB) ----------
        tgt = cv2.imread(tgt_path)
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
        tgt = cv2.resize(tgt, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        tgt = torch.from_numpy(tgt).permute(
            2, 0, 1).float() / 255.0   # (3,H,W)

        if self.split != "test":
            # ---------- segmentation (single channel) ----------
            seg_path = os.path.join(self.seg_dir, base)
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if seg is None:
                raise FileNotFoundError(f"Missing seg file: {seg_path}")

            seg = cv2.resize(seg, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            seg = torch.from_numpy(seg).long()                          # (H,W)

            return inp, seg, tgt   # ✅ ORDER MATTERS
        else:
            return inp, tgt


ds = ADEEnhancementDataset("/content/Datasets", split="train")
dl = DataLoader(ds, batch_size=2, shuffle=True)

x, seg, y = next(iter(dl))
print(x.shape, seg.shape, y.shape)
