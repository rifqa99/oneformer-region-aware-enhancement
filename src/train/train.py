import torch
from torch import nn


def main():
    x = torch.randn(2, 3, 256, 256)
    y_hat = x  # placeholder
    loss = nn.L1Loss()(y_hat, x)
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
