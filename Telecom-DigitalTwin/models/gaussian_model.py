"""
GaussianModel: holds the learnable 3D Gaussian point set (position, scale,
rotation, opacity, SH coefficients for view-dependent color).
Pure data container - all optimizer/densify logic lives in Trainer, not here
(cross-checked against the reference scene/gaussian_model.py from
graphdeco-inria/gaussian-splatting: activations, densify formulas and the
create_from_pcd initialization order all match; see inline notes below for
the couple of places we deliberately deviate, e.g. no CUDA simple_knn
extension available so nearest-neighbor distance is approximated in pure
PyTorch).
"""
import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData, PlyElement

from utils.sh_utils import RGB2SH, SH2RGB, num_sh_bases
from utils.general_utils import inverse_sigmoid, strip_lowerdiag
from utils.geometry import build_scaling_rotation


class GaussianModel:
    def __init__(self, sh_degree: int, device="cuda"):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.device = device

        self._xyz = torch.empty(0, device=device)
        self._features_dc = torch.empty(0, device=device)
        self._features_rest = torch.empty(0, device=device)
        self._scaling = torch.empty(0, device=device)
        self._rotation = torch.empty(0, device=device)
        self._opacity = torch.empty(0, device=device)

        # Resized to their real shape by Trainer once the point count is known.
        self.xyz_gradient_accum = torch.empty(0, device=device)
        self.denom = torch.empty(0, device=device)
        self.max_radii2D = torch.empty(0, device=device)

        self.spatial_lr_scale = 1.0

    # ---------------- Activations ----------------
    # Same activation functions as the reference GaussianModel.setup_functions():
    # exp for scaling, sigmoid for opacity, normalize for rotation quaternion.
    @property
    def scaling(self):
        return torch.exp(self._scaling)

    @property
    def rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def xyz(self):
        return self._xyz

    @property
    def opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def get_covariance(self):
        L = build_scaling_rotation(self.scaling, self.rotation)
        actual_cov = L @ L.transpose(1, 2)
        return strip_lowerdiag(actual_cov)

    def oneup_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ---------------- Init from point cloud (SfM or random) ----------------
    def create_from_pcd(self, xyz: np.ndarray, rgb: np.ndarray, spatial_lr_scale: float,
                         opacity_init: float = 0.1, scale_init_factor: float = 1.0):
        """opacity_init/scale_init_factor default to configs/gaussian.yaml
        (model.init.opacity, model.init.scale_factor) - train.py passes these
        in from config instead of hard-coding them. `scale_init_factor=1.0`
        reproduces the reference's exact formula (no extra multiplier there);
        it's a knob we added on top, not present in the original repo.

        `rgb` must already be normalized to [0,1] - RGB2SH() below assumes
        that range (dataloader/bts_dataset.py is responsible for converting
        the raw uint8 [0,255] COLMAP point color before calling this)."""
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        fused_color = RGB2SH(torch.tensor(rgb, dtype=torch.float32, device=self.device))

        n = fused_point_cloud.shape[0]
        # features: (N, 3 RGB channels, num_sh_bases). Only the degree-0 (DC)
        # SH coefficient is set from the observed color; higher-order bands
        # are left at 0 (torch.zeros() default) and get learned as training
        # progresses and active_sh_degree increases (see oneup_sh_degree()).
        # Matches reference: `features[:, :3, 0] = fused_color; features[:, 3:, 1:] = 0.0`.
        features = torch.zeros((n, 3, num_sh_bases(self.max_sh_degree)), device=self.device)
        features[:, :, 0] = fused_color

        # Reference computes `dist2 = distCUDA2(points)` via the compiled
        # `simple_knn` CUDA extension, which returns the MEAN OF SQUARED
        # distances to each point's 3 nearest neighbors (not the square of
        # the mean distance - these differ by Jensen's inequality). We don't
        # ship that CUDA extension, so `_nn_dist_squared()` below approximates
        # it in pure PyTorch via `torch.cdist` + topk, matching that same
        # "mean of squared distances" semantics.
        dist2 = torch.clamp_min(self._nn_dist_squared(fused_point_cloud), 1e-7)
        scales = torch.log(torch.sqrt(dist2) * scale_init_factor)[..., None].repeat(1, 3)
        rots = torch.zeros((n, 4), device=self.device)
        rots[:, 0] = 1.0

        opacities = inverse_sigmoid(opacity_init * torch.ones((n, 1), device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((n,), device=self.device)

    @staticmethod
    def _nn_dist_squared(points, k=3):
        """Pure-PyTorch stand-in for the reference's CUDA `distCUDA2`
        (from the `simple_knn` submodule): for each point, the mean of the
        SQUARED Euclidean distances to its `k` nearest neighbors (k=3,
        matching the reference). Chunked over `cdist` to bound peak memory
        for large point clouds; not as fast as the CUDA kernel but gives the
        same statistic."""
        from torch import cdist
        chunks = []
        chunk_size = 4096
        for i in range(0, points.shape[0], chunk_size):
            d = cdist(points[i:i + chunk_size], points)
            knn = d.topk(k + 1, largest=False).values[:, 1:]  # drop self (distance 0)
            chunks.append(knn.pow(2).mean(dim=1))
        return torch.cat(chunks)

    # ---------------- Checkpoint ----------------
    def capture(self) -> dict:
        """Full snapshot for Trainer to checkpoint - sufficient to resume
        training (not just to render), so keep spatial_lr_scale and
        max_radii2D too. (Reference returns a tuple incl. optimizer state
        dict via GaussianModel.capture()/restore(); we keep the same idea
        but store optimizer state on the Trainer side instead, and use a
        dict here for clarity.)"""
        return {
            "xyz": self._xyz.detach().cpu(),
            "f_dc": self._features_dc.detach().cpu(),
            "f_rest": self._features_rest.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "opacity": self._opacity.detach().cpu(),
            "active_sh_degree": self.active_sh_degree,
            "spatial_lr_scale": self.spatial_lr_scale,
            "max_radii2D": self.max_radii2D.detach().cpu(),
        }

    def restore(self, state: dict):
        self._xyz = nn.Parameter(state["xyz"].to(self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(state["f_dc"].to(self.device).requires_grad_(True))
        self._features_rest = nn.Parameter(state["f_rest"].to(self.device).requires_grad_(True))
        self._scaling = nn.Parameter(state["scaling"].to(self.device).requires_grad_(True))
        self._rotation = nn.Parameter(state["rotation"].to(self.device).requires_grad_(True))
        self._opacity = nn.Parameter(state["opacity"].to(self.device).requires_grad_(True))
        self.active_sh_degree = state["active_sh_degree"]
        self.spatial_lr_scale = state.get("spatial_lr_scale", 1.0)

        n = self._xyz.shape[0]
        default_radii = torch.zeros((n,), device=self.device)
        self.max_radii2D = state.get("max_radii2D", default_radii).to(self.device)

    def save_ply(self, path: str):
        """Export the point cloud (position + DC color) to .ply for viewing
        in an external viewer (viser/SuperSplat/CloudCompare). Reference's
        save_ply() also stores normals (zeros) and full SH; we only keep
        position + DC color here since that's all downstream viewers used
        in this project actually need."""
        xyz = self._xyz.detach().cpu().numpy()
        rgb = (SH2RGB(self._features_dc[:, 0, :].detach().cpu())
               .clamp(0, 1).numpy() * 255).astype(np.uint8)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate([xyz, rgb], axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
