import torch


def concat_image_and_masks(image, masks):
    """
    image: (3, H, W) tensor
    masks: dict of (H, W) tensors
    """
    mask_tensors = [m.unsqueeze(0) for m in masks.values()]
    return torch.cat([image] + mask_tensors, dim=0)
