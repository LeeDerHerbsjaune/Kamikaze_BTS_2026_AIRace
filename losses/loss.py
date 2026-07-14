"""
Loss function cho training 3DGS: kết hợp L1 (robust với noise) và SSIM
(giữ cấu trúc/structure của ảnh, quan trọng cho chi tiết thiết bị nhỏ trên trạm BTS
như anten, dây cáp, giá đỡ).

total_loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
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
    """img1, img2: (B or none,3,H,W) hoặc (3,H,W), giá trị trong [0,1]."""
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
    l1 = l1_loss(rendered, gt)
    s = ssim(rendered, gt)
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - s)
    return total, {"l1": l1.item(), "ssim": s.item()}
