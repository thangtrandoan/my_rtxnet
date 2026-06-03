import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = F.mse_loss(pred, target).item()
    return 10.0 * math.log10(1.0 / max(mse, eps))


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    channels = pred.shape[1]
    kernel_1d = torch.arange(11, device=pred.device, dtype=pred.dtype) - 5
    kernel_1d = torch.exp(-(kernel_1d * kernel_1d) / (2 * 1.5 * 1.5))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel = kernel_2d.expand(channels, 1, 11, 11).contiguous()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.conv2d(pred, kernel, groups=channels)
    mu_y = F.conv2d(target, kernel, groups=channels)
    sigma_x = F.conv2d(pred * pred, kernel, groups=channels) - mu_x * mu_x
    sigma_y = F.conv2d(target * target, kernel, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(pred * target, kernel, groups=channels) - mu_x * mu_y

    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().item()
