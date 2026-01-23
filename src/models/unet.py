import torch
import torch.nn as nn
from src.utils.metrics import psnr, ssim

EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = (512, 512)


def center_crop(tensor, target):
    _, _, h, w = target.shape
    return tensor[:, :, :h, :w]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        m = self.mid(self.pool2(d2))

        u2 = self.up2(m)
        d2_c = center_crop(d2, u2)
        u2 = self.conv2(torch.cat([u2, d2_c], dim=1))

        u1 = self.up1(u2)
        d1_c = center_crop(d1, u1)
        u1 = self.conv1(torch.cat([u1, d1_c], dim=1))

        return self.out(u1)


p = psnr(out, tgt_img[:, :, :h, :w]).item()
s = ssim(out, tgt_img[:, :, :h, :w]).item()
val_psnr += p
val_ssim += s
val_psnr, val_ssim = 0.0, 0.0
val_psnr /= max(1, len(val_dl))
val_ssim /= max(1, len(val_dl))
print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")
