import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


class OneFormerWrapper:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        )
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        # image: PIL Image
        inputs = self.processor(
            image,
            task_inputs=["semantic"],
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        return seg  # (H, W) class ids
