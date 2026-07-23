"""
Loss function for 3DGS training: combines L1 (robust to noise) and SSIM
(preserves image structure - important for fine BTS equipment detail such as
antennas, cables, mounting brackets).

total_loss = l1_weight * L1 + dssim_weight * (1 - SSIM)

lambda_dssim passed into compute_loss() comes from configs/loss.yaml:
loss.dssim.weight (loss.l1.weight is also in the config, but the original
3DGS formula only uses a single lambda_dssim and implicitly assumes
l1_weight = 1 - lambda_dssim; if you want 2 independent weights, use
compute_loss_weighted() below).

The ssim()/gaussian window implementation matches the standard formula used
in utils/loss_utils.py of graphdeco-inria/gaussian-splatting (window_size=11,
sigma=1.5, C1=0.01**2, C2=0.03**2).
"""
import torch
import torch.nn.functional as F
from math import exp


def l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


def _gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11):
    """img1, img2: (B or none,3,H,W) or (3,H,W), values in [0,1]."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compute_loss(rendered, gt, lambda_dssim=0.2):
    """Standard 3DGS formula: total = (1-lambda)*L1 + lambda*(1-SSIM)."""
    l1 = l1_loss(rendered, gt)
    s = ssim(rendered, gt)
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - s)
    return total, {"l1": l1.item(), "ssim": s.item()}


def compute_loss_weighted(rendered, gt, l1_weight=0.8, dssim_weight=0.2):
    """Variant using the two independent weights loss.l1.weight /
    loss.dssim.weight from configs/loss.yaml, instead of assuming they
    always sum to 1."""
    l1 = l1_loss(rendered, gt)
    s = ssim(rendered, gt)
    total = l1_weight * l1 + dssim_weight * (1.0 - s)
    return total, {"l1": l1.item(), "ssim": s.item()}
