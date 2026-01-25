from src.datasets.PairedMaskDataset import PairedImageDataset
from torch.utils.data import DataLoader
from src.models.unet import UNet
from src.utils.concat_inputs import concat_image_and_masks
import torchvision.transforms as T
from src.utils.region_masks import build_region_masks
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


from src.models.oneformer_wrapper import OneFormerWrapper

# Load model
model = OneFormerWrapper(device="cpu")

# Load image
img = Image.open("data/sample/school.jfif").convert("RGB")

# Predict segmentation
seg = model.predict(img)  # (H, W)
print("Unique class IDs:", torch.unique(seg))


# Convert to numpy
seg_np = seg.cpu().numpy()

SEMANTIC_GROUPS = {
    "human": ["person"],
    "vegetation": ["plant", "tree"],
    "structure": ["wall", "building"],
}

id2label = model.model.config.id2label

masks = build_region_masks(seg, id2label, SEMANTIC_GROUPS)

for k, v in masks.items():
    print(k, v.sum().item())

for name, mask in masks.items():
    plt.figure(figsize=(4, 4))
    plt.imshow(mask.cpu().numpy(), cmap="gray")
    plt.title(name)
    plt.axis("off")
    plt.savefig(f"outputs/mask_{name}.png", bbox_inches="tight")
    plt.close()

print("Saved region masks to outputs/")

to_tensor = T.ToTensor()
img_t = to_tensor(img)          # (3, H, W)

# concatenate image and masks
input_tensor = concat_image_and_masks(img_t, masks)
print("Input shape:", input_tensor.shape)


# Fake target (for smoke test only)
target = img_t.unsqueeze(0)  # (1, 3, H, W)

model_enh = UNet(in_channels=6, out_channels=3)
optimizer = torch.optim.Adam(model_enh.parameters(), lr=1e-3)
loss_fn = torch.nn.L1Loss()

model_enh.train()
inp = input_tensor.unsqueeze(0)  # (1, 6, H, W)
out = model_enh(inp)
_, _, h, w = out.shape
target_c = target[:, :, :h, :w]
loss = loss_fn(out, target_c)

loss.backward()
optimizer.step()

print("Smoke loss:", loss.item())

ds = PairedImageDataset(
    "data/train/input",
    "data/train/target",
    size=(512, 512)
)

dl = DataLoader(ds, batch_size=2, shuffle=True)

x, y = next(iter(dl))
print(x.shape, y.shape)
