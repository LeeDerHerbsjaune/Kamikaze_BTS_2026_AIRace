"""
GaussianModel: biểu diễn scene bằng tập điểm 3D Gaussian có thể học được
(vị trí, scale, rotation, opacity, hệ số SH cho màu view-dependent).
Bao gồm cả logic densify (clone/split) và prune dùng trong training loop.
"""
import torch
import torch.nn as nn
import numpy as np

from utils.sh_utils import RGB2SH, num_sh_bases
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_lowerdiag


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R


def build_scaling_rotation(scaling, rotation):
    L = torch.zeros((scaling.shape[0], 3, 3), device=scaling.device)
    R = build_rotation(rotation)
    L[:, 0, 0] = scaling[:, 0]
    L[:, 1, 1] = scaling[:, 1]
    L[:, 2, 2] = scaling[:, 2]
    L = R @ L
    return L


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

        self.xyz_gradient_accum = torch.empty(0, device=device)
        self.denom = torch.empty(0, device=device)
        self.max_radii2D = torch.empty(0, device=device)

        self.optimizer = None
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
    def create_from_pcd(self, xyz: np.ndarray, rgb: np.ndarray, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        fused_color = RGB2SH(torch.tensor(rgb, dtype=torch.float32, device=self.device))

        n = fused_point_cloud.shape[0]
        features = torch.zeros((n, 3, num_sh_bases(self.max_sh_degree)), device=self.device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # scale khởi tạo dựa trên khoảng cách tới các điểm lân cận (đơn giản hoá: dùng std cục bộ)
        dist2 = torch.clamp_min(self._nn_dist_squared(fused_point_cloud), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((n, 4), device=self.device)
        rots[:, 0] = 1.0

        opacities = inverse_sigmoid(0.1 * torch.ones((n, 1), device=self.device))

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
        """Xấp xỉ khoảng cách bình phương tới k điểm gần nhất (dùng cho khởi tạo scale)."""
        from torch import cdist
        chunks = []
        chunk_size = 4096
        for i in range(0, points.shape[0], chunk_size):
            d = cdist(points[i:i + chunk_size], points)
            knn = d.topk(k + 1, largest=False).values[:, 1:]
            chunks.append(knn.mean(dim=1) ** 2)
        return torch.cat(chunks)

    # ---------------- Optimizer setup ----------------
    def setup_training(self, cfg):
        opt_cfg = cfg["optimizer"]
        self.percent_dense = cfg["model"]["percent_dense"]
        n = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device=self.device)
        self.denom = torch.zeros((n, 1), device=self.device)

        params = [
            {"params": [self._xyz], "lr": opt_cfg["position_lr_init"] * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": opt_cfg["feature_lr"], "name": "f_dc"},
            {"params": [self._features_rest], "lr": opt_cfg["feature_lr"] / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": opt_cfg["opacity_lr"], "name": "opacity"},
            {"params": [self._scaling], "lr": opt_cfg["scaling_lr"], "name": "scaling"},
            {"params": [self._rotation], "lr": opt_cfg["rotation_lr"], "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        self.xyz_scheduler = get_expon_lr_func(
            lr_init=opt_cfg["position_lr_init"] * self.spatial_lr_scale,
            lr_final=opt_cfg["position_lr_final"] * self.spatial_lr_scale,
            lr_delay_mult=opt_cfg["position_lr_delay_mult"],
            max_steps=opt_cfg["position_lr_max_steps"],
        )

    def update_learning_rate(self, iteration):
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = self.xyz_scheduler(iteration)

    # ---------------- Densify & Prune ----------------
    def add_densification_stats(self, viewspace_point_grad, visibility_filter):
        self.xyz_gradient_accum[visibility_filter] += torch.norm(
            viewspace_point_grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom.clamp_min(1)
        grads = torch.nan_to_num(grads, nan=0.0)

        self._densify_and_clone(grads, max_grad, extent)
        self._densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_points_vs | big_points_ws
        self._prune_points(prune_mask)
        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, grad_threshold, extent):
        selected = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected = selected & (self.scaling.max(dim=1).values <= self.percent_dense * extent)

        new_xyz = self._xyz[selected]
        new_f_dc = self._features_dc[selected]
        new_f_rest = self._features_rest[selected]
        new_opacities = self._opacity[selected]
        new_scaling = self._scaling[selected]
        new_rotation = self._rotation[selected]
        self._append_points(new_xyz, new_f_dc, new_f_rest, new_opacities, new_scaling, new_rotation)

    def _densify_and_split(self, grads, grad_threshold, extent, N=2):
        n_init = self._xyz.shape[0]
        padded_grad = torch.zeros((n_init,), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected = padded_grad >= grad_threshold
        selected = selected & (self.scaling.max(dim=1).values > self.percent_dense * extent)

        stds = self.scaling[selected].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected].repeat(N, 1)
        new_scaling = torch.log(self.scaling[selected].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected].repeat(N, 1)
        new_f_dc = self._features_dc[selected].repeat(N, 1, 1)
        new_f_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)

        self._append_points(new_xyz, new_f_dc, new_f_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat([selected, torch.zeros(new_xyz.shape[0], dtype=bool, device=self.device)])
        self._prune_points(prune_filter)

    def _append_points(self, new_xyz, new_f_dc, new_f_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz, "f_dc": new_f_dc, "f_rest": new_f_rest,
            "opacity": new_opacities, "scaling": new_scaling, "rotation": new_rotation,
        }
        for group in self.optimizer.param_groups:
            extension = d[group["name"]]
            group["params"][0] = nn.Parameter(
                torch.cat([group["params"][0], extension], dim=0).requires_grad_(True))
            setattr(self, f"_{'xyz' if group['name']=='xyz' else group['name']}", group["params"][0])

        n = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device=self.device)
        self.denom = torch.zeros((n, 1), device=self.device)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(new_xyz.shape[0], device=self.device)])

    def _prune_points(self, mask):
        valid = ~mask
        for group in self.optimizer.param_groups:
            group["params"][0] = nn.Parameter(group["params"][0][valid].requires_grad_(True))
            setattr(self, f"_{'xyz' if group['name']=='xyz' else group['name']}", group["params"][0])
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]

    def reset_opacity(self):
        new_opacity = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                group["params"][0] = nn.Parameter(new_opacity.requires_grad_(True))
                self._opacity = group["params"][0]

    # ---------------- Checkpoint ----------------
    def save_ply(self, path):
        """Xuất point cloud (vị trí + màu DC) ra .ply để xem trong viewer (viser/supersplat/CloudCompare)."""
        from plyfile import PlyData, PlyElement
        xyz = self._xyz.detach().cpu().numpy()
        from utils.sh_utils import SH2RGB
        rgb = (SH2RGB(self._features_dc[:, 0, :].detach().cpu()).clamp(0, 1).numpy() * 255).astype(np.uint8)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate([xyz, rgb], axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def capture(self):
        return {
            "xyz": self._xyz.detach().cpu(), "f_dc": self._features_dc.detach().cpu(),
            "f_rest": self._features_rest.detach().cpu(), "scaling": self._scaling.detach().cpu(),
            "rotation": self._rotation.detach().cpu(), "opacity": self._opacity.detach().cpu(),
            "active_sh_degree": self.active_sh_degree,
        }

    def restore(self, state):
        self._xyz = nn.Parameter(state["xyz"].to(self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(state["f_dc"].to(self.device).requires_grad_(True))
        self._features_rest = nn.Parameter(state["f_rest"].to(self.device).requires_grad_(True))
        self._scaling = nn.Parameter(state["scaling"].to(self.device).requires_grad_(True))
        self._rotation = nn.Parameter(state["rotation"].to(self.device).requires_grad_(True))
        self._opacity = nn.Parameter(state["opacity"].to(self.device).requires_grad_(True))
        self.active_sh_degree = state["active_sh_degree"]
