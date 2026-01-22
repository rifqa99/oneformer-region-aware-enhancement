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
