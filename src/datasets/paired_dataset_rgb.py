from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T


class PairedImageDatasetRGB(Dataset):
    def __init__(self, input_dir, target_dir, size=None):
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

        if self.size:
            inp = inp.resize(self.size)
            tgt = tgt.resize(self.size)

        return self.transform(inp), self.transform(tgt)
