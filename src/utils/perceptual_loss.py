import torch
import torch.nn as nn
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(3, 8, 15), device="cuda"):
        super().__init__()
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.layers = layers
        self.blocks = nn.ModuleList()
        prev = 0
        for l in layers:
            self.blocks.append(vgg[prev:l].eval())
            prev = l
        for b in self.blocks:
            for p in b.parameters():
                p.requires_grad = False
        self.to(device)

    def forward(self, x, y):
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.mean(torch.abs(x - y))
        return loss
