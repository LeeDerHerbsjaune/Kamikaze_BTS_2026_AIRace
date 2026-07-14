"""
GaussianModel: biểu diễn scene bằng tập điểm 3D Gaussian có thể học được
(vị trí, scale, rotation, opacity, hệ số SH cho màu view-dependent).
Chỉ là data container thuần - toàn bộ optimizer/densify logic thuộc về Trainer.
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

        # Các biến này sẽ được khởi tạo kích thước thật bên phía Trainer
        self.xyz_gradient_accum = torch.empty(0, device=device)
        self.denom = torch.empty(0, device=device)
        self.max_radii2D = torch.empty(0, device=device)

        self.spatial_lr_scale = 1.0

    # ---------------- Activations ----------------
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

    # ---------------- Init từ point cloud (SfM hoặc random) ----------------
    def create_from_pcd(self, xyz: np.ndarray, rgb: np.ndarray, spatial_lr_scale: float,
                         opacity_init: float = 0.1, scale_init_factor: float = 1.0):
        """opacity_init/scale_init_factor mặc định khớp configs/gaussian.yaml
        (model.init.opacity, model.init.scale_factor) - train.py truyền trực
        tiếp 2 giá trị này từ config thay vì hard-code."""
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        fused_color = RGB2SH(torch.tensor(rgb, dtype=torch.float32, device=self.device))

        n = fused_point_cloud.shape[0]
        # features: (N, 3 kênh RGB, num_sh_bases). Chỉ set hệ số SH bậc 0 (DC,
        # index cuối = 0) bằng màu quan sát được; các bậc cao hơn (index 1:)
        # để 0 vì torch.zeros() đã khởi tạo sẵn - model sẽ tự học dần trong
        # lúc train khi active_sh_degree tăng lên (xem oneup_sh_degree()).
        features = torch.zeros((n, 3, num_sh_bases(self.max_sh_degree)), device=self.device)
        features[:, :, 0] = fused_color

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
        from torch import cdist
        chunks = []
        chunk_size = 4096
        for i in range(0, points.shape[0], chunk_size):
            d = cdist(points[i:i + chunk_size], points)
            knn = d.topk(k + 1, largest=False).values[:, 1:]
            chunks.append(knn.mean(dim=1) ** 2)
        return torch.cat(chunks)

    # ---------------- Checkpoint ----------------
    def capture(self) -> dict:
        """Snapshot đầy đủ để Trainer lưu checkpoint - đủ để resume training
        (không chỉ để render), nên giữ cả spatial_lr_scale và max_radii2D."""
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
        """Xuất point cloud (vị trí + màu DC) ra .ply để xem trong viewer
        (viser/supersplat/CloudCompare)."""
        xyz = self._xyz.detach().cpu().numpy()
        rgb = (SH2RGB(self._features_dc[:, 0, :].detach().cpu())
               .clamp(0, 1).numpy() * 255).astype(np.uint8)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate([xyz, rgb], axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
