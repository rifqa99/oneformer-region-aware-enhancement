import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))


def ssim(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = pred.mean(dim=(-2, -1))
    mu_y = target.mean(dim=(-2, -1))

    sigma_x = ((pred - mu_x[..., None, None]) ** 2).mean(dim=(-2, -1))
    sigma_y = ((target - mu_y[..., None, None]) ** 2).mean(dim=(-2, -1))
    sigma_xy = ((pred - mu_x[..., None, None]) *
                (target - mu_y[..., None, None])).mean(dim=(-2, -1))

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )

    return ssim_map.mean()
