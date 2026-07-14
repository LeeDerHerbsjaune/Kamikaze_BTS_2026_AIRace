"""
Tiện ích chuyển đổi màu RGB <-> hệ số Spherical Harmonics (SH).
Dùng để biểu diễn màu view-dependent cho từng Gaussian (giống 3DGS gốc).
"""
import torch

C0 = 0.28209479177387814


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """Chuyển màu RGB [0,1] sang hệ số SH bậc 0 (DC component)."""
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """Chuyển ngược hệ số SH bậc 0 sang RGB [0,1]."""
    return sh * C0 + 0.5


def num_sh_bases(degree: int) -> int:
    """Số lượng hệ số SH ứng với bậc `degree` (bao gồm cả DC)."""
    return (degree + 1) ** 2
