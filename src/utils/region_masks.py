import torch


def build_region_masks(seg_map, id2label, semantic_groups):
    """
    seg_map: (H, W) tensor with class IDs
    id2label: dict {id: class_name}
    semantic_groups: dict {region: [keywords]}
    """
    masks = {}

    for region, keywords in semantic_groups.items():
        mask = torch.zeros_like(seg_map, dtype=torch.float32)

        for class_id, class_name in id2label.items():
            class_name = class_name.lower()
            if any(k in class_name for k in keywords):
                mask[seg_map == class_id] = 1.0

        masks[region] = mask

    return masks
