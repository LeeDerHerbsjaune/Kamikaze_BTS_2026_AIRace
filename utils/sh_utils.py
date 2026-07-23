"""
RGB <-> Spherical Harmonics (SH) coefficient conversion utilities.
Used to represent view-dependent color per Gaussian (same as the reference
3DGS implementation's utils/sh_utils.py - C0 is the degree-0 SH basis
constant, and only the DC term is converted here since that's all
GaussianModel.create_from_pcd() needs at initialization).
"""
import torch

C0 = 0.28209479177387814


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] color to the degree-0 (DC) SH coefficient."""
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """Inverse of RGB2SH: degree-0 SH coefficient back to RGB [0,1]."""
    return sh * C0 + 0.5


def num_sh_bases(degree: int) -> int:
    """Number of SH coefficients for a given `degree` (DC term included)."""
    return (degree + 1) ** 2
