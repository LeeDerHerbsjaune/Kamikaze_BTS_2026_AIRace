"""
Trainer: điều phối toàn bộ training loop -
Load Batch -> Render -> Compute Loss -> Backward -> Optimizer Step -> Densify -> Prune

Tất cả truy cập config trong file này dùng utils.config_loader.cfg_get() với
dotted-path khớp chính xác cấu trúc trong configs/*.yaml. Xem README.md /
CHANGELOG.md để biết mapping đầy đủ giữa key config và nơi dùng.
"""
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from renderer.gaussian_renderer import render
from losses.loss import compute_loss
from evaluation.metrics import evaluate_dataset
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.geometry import build_rotation
from utils.config_loader import cfg_get

# Mapping tường minh optimizer param-group name -> attribute thật trên GaussianModel.
# KHÔNG suy luận bằng string formatting (f"_{group['name']}") - đó là bug đã gây
# lệch shape giữa _xyz và _features_dc/_features_rest sau lần densify đầu tiên.
_PARAM_TO_ATTR = {
    "xyz": "_xyz",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "opacity": "_opacity",
    "scaling": "_scaling",
    "rotation": "_rotation",
}

# Mapping tên param-group -> flag bật/tắt trong configs/gaussian.yaml (model.optimize.*)
_PARAM_TO_OPTIMIZE_FLAG = {
    "xyz": "position",
    "f_dc": "sh",
    "f_rest": "sh",
    "opacity": "opacity",
    "scaling": "scale",
    "rotation": "rotation",
}


class Trainer:
    def __init__(self, cfg, gaussians, dataset, logger):
        self.cfg = cfg
        self.gaussians = gaussians
        self.dataset = dataset
        self.logger = logger

        bg_color = cfg_get(cfg, "renderer.background.color", [0, 0, 0])
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device=gaussians.device)
        self.backend = cfg_get(cfg, "renderer.backend", "gsplat")

        self.checkpoint_dir = cfg_get(cfg, "training.checkpoint_dir") or cfg_get(
            cfg, "training.checkpoint.directory", "./outputs/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_cams = list(dataset.train_cameras)
        if not self.train_cams:
            raise ValueError("dataset.train_cameras rỗng - không có gì để train.")

        # QUAN TRỌNG: gaussians.create_from_pcd(...) phải chạy XONG trước khi tạo
        # Trainer(...), vì setup_training cần gaussians.xyz đã có kích thước thật.
        self.setup_training(cfg)

    def train(self):
        cfg = self.cfg
        n_iters = cfg_get(cfg, "training.iterations", 30000)
        log_interval = cfg_get(cfg, "training.log_interval", 50)
        eval_interval = cfg_get(cfg, "training.eval_interval", 2000)
        save_interval = cfg_get(cfg, "training.save_interval", 5000)

        sh_up_interval = cfg_get(cfg, "sh.increase_interval", 1000)
        lambda_dssim = cfg_get(cfg, "loss.dssim.weight", 0.2)
        use_lpips_eval = cfg_get(cfg, "evaluation.use_lpips", True)

        densify_enabled = cfg_get(cfg, "densify.enabled", True)
        densify_from_iter = cfg_get(cfg, "densify.from_iter", 500)
        densify_until_iter = cfg_get(cfg, "densify.until_iter", 15000)
        densify_interval = cfg_get(cfg, "densify.interval", 100)
        densify_grad_threshold = cfg_get(cfg, "densify.grad_threshold", 0.0002)
        clone_factor = cfg_get(cfg, "densify.clone_factor", 2)
        split_factor = cfg_get(cfg, "densify.split_factor", 2)
        min_world_size_ratio = cfg_get(cfg, "densify.min_world_size_ratio", 1.0e-5)
        max_gaussians = cfg_get(cfg, "densify.max_gaussians", None)

        min_opacity = cfg_get(cfg, "pruning.opacity.min", 0.005)
        opacity_reset_interval = cfg_get(cfg, "pruning.opacity.reset_interval", 3000)
        max_screen_size = cfg_get(cfg, "pruning.size.max_screen", 20)

        prune_after_iter = cfg_get(cfg, "pruning.schedule.after_iter") or densify_from_iter
        prune_interval = cfg_get(cfg, "pruning.schedule.interval") or densify_interval
        min_visible_count = cfg_get(cfg, "pruning.schedule.min_visible_count", None)

        pbar = tqdm(range(1, n_iters + 1), desc="Training")
        cam_pool = []

        for iteration in pbar:
            self.update_learning_rate(iteration)

            if iteration % sh_up_interval == 0:
                self.gaussians.oneup_sh_degree()

            if not cam_pool:
                cam_pool = self.train_cams.copy()
                random.shuffle(cam_pool)
            cam = cam_pool.pop()

            # -------- Render --------
            out = render(cam, self.gaussians, self.bg_color, backend=self.backend)
            rendered = out["render"]
            gt = cam.image.to(rendered.device)

            # -------- Loss --------
            loss, loss_parts = compute_loss(rendered, gt, lambda_dssim)

            # -------- Backward --------
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # -------- Densification stats --------
            with torch.no_grad():
                if densify_enabled and iteration < densify_until_iter:
                    if out["viewspace_points"].grad is not None:
                        self.gaussians.max_radii2D[out["visibility_filter"]] = torch.max(
                            self.gaussians.max_radii2D[out["visibility_filter"]],
                            out["radii"][out["visibility_filter"]])
                        self.add_densification_stats(
                            out["viewspace_points"].grad, out["visibility_filter"])

            # -------- Optimizer step --------
            self.optimizer.step()

            # -------- Densify / Prune theo lịch chính --------
            with torch.no_grad():
                if (densify_enabled
                        and iteration > densify_from_iter
                        and iteration < densify_until_iter
                        and iteration % densify_interval == 0):
                    extent = self._scene_extent()
                    self.densify_and_prune(
                        densify_grad_threshold, min_opacity, extent, max_screen_size,
                        clone_factor=clone_factor, split_factor=split_factor,
                        min_world_size_ratio=min_world_size_ratio,
                        max_gaussians=max_gaussians)

                # -------- Prune nhẹ theo lịch riêng, độc lập với densify --------
                if (iteration > prune_after_iter and iteration % prune_interval == 0):
                    self.prune_low_quality(min_opacity, min_visible_count=min_visible_count)

                if opacity_reset_interval and iteration % opacity_reset_interval == 0:
                    self.reset_opacity()

            if iteration % log_interval == 0:
                pbar.set_postfix(loss=loss.item(), n_gaussians=self.gaussians.xyz.shape[0])
                self.logger.log_scalar("train/loss", loss.item(), iteration)
                self.logger.log_scalar("train/n_gaussians", self.gaussians.xyz.shape[0], iteration)

            if iteration % eval_interval == 0 and self.dataset.eval_cameras:
                metrics = evaluate_dataset(self.dataset.eval_cameras, self.gaussians,
                                           render, self.bg_color,
                                           use_lpips=use_lpips_eval)
                if metrics:
                    self.logger.log_dict("eval", metrics, iteration)

            if iteration % save_interval == 0 or iteration == n_iters:
                self.save_checkpoint(iteration)

        self.save_checkpoint(n_iters, name="last")

    def _scene_extent(self):
        from utils.camera_utils import cameras_extent
        return cameras_extent(self.train_cams)

    def save_checkpoint(self, iteration, name=None):
        name = name or f"iter_{iteration}"
        path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        torch.save({"iteration": iteration, "gaussians": self.gaussians.capture()}, path)
        self.logger.log_text(f"Đã lưu checkpoint: {path}")

    # ---------------- Optimizer setup ----------------
    def setup_training(self, cfg):
        self.percent_dense = cfg_get(cfg, "model.percent_dense", 0.01)
        n = self.gaussians.xyz.shape[0]

        self.gaussians.xyz_gradient_accum = torch.zeros((n, 1), device=self.gaussians.device)
        self.gaussians.denom = torch.zeros((n, 1), device=self.gaussians.device)

        pos_lr_init = cfg_get(cfg, "optimizer.position.lr_init", 0.00016)
        pos_lr_final = cfg_get(cfg, "optimizer.position.lr_final", 0.0000016)
        pos_lr_delay_mult = cfg_get(cfg, "optimizer.position.lr_delay_mult", 0.01)
        pos_lr_max_steps = cfg_get(cfg, "optimizer.position.lr_max_steps", 30000)
        feature_lr = cfg_get(cfg, "optimizer.feature.lr", 0.0025)
        opacity_lr = cfg_get(cfg, "optimizer.opacity.lr", 0.05)
        scaling_lr = cfg_get(cfg, "optimizer.scaling.lr", 0.005)
        rotation_lr = cfg_get(cfg, "optimizer.rotation.lr", 0.001)

        all_groups = [
            {"params": [self.gaussians._xyz], "lr": pos_lr_init * self.gaussians.spatial_lr_scale, "name": "xyz"},
            {"params": [self.gaussians._features_dc], "lr": feature_lr, "name": "f_dc"},
            {"params": [self.gaussians._features_rest], "lr": feature_lr / 20.0, "name": "f_rest"},
            {"params": [self.gaussians._opacity], "lr": opacity_lr, "name": "opacity"},
            {"params": [self.gaussians._scaling], "lr": scaling_lr, "name": "scaling"},
            {"params": [self.gaussians._rotation], "lr": rotation_lr, "name": "rotation"},
        ]

        # Tôn trọng model.optimize.{position,rotation,scale,opacity,sh} trong
        # configs/gaussian.yaml: param-group nào bị tắt thì requires_grad=False
        # và KHÔNG đưa vào optimizer (tránh Adam vẫn cấp state/step vô ích).
        params = []
        for group in all_groups:
            flag_key = _PARAM_TO_OPTIMIZE_FLAG[group["name"]]
            enabled = cfg_get(cfg, f"model.optimize.{flag_key}", True)
            group["params"][0].requires_grad_(enabled)
            if enabled:
                params.append(group)
            else:
                self.logger.log_text(f"[optimize] param-group '{group['name']}' bị tắt "
                                      f"(model.optimize.{flag_key}=false), sẽ không được học.")

        if not params:
            raise ValueError("model.optimize.* tắt hết toàn bộ param-group - không có gì để train.")

        # densify/prune (_append_points, _prune_points) giả định TẤT CẢ 6 thuộc
        # tính (_xyz, _features_dc/_rest, _opacity, _scaling, _rotation) luôn
        # cùng số điểm N và được resize đồng bộ qua optimizer.param_groups. Nếu
        # 1 param-group bị tắt (model.optimize.*=false) nó sẽ KHÔNG được resize
        # cùng, dẫn tới lệch shape ngay lần densify/prune đầu tiên. Vì vậy chỉ
        # cho phép tắt optimize khi densify cũng bị tắt.
        if len(params) < len(all_groups) and cfg_get(cfg, "densify.enabled", True):
            raise ValueError(
                "Không thể tắt riêng lẻ model.optimize.* khi densify.enabled=true: "
                "densify/prune cần resize đồng bộ cả 6 thuộc tính Gaussian. "
                "Hãy set densify.enabled=false nếu muốn cố định 1 số thuộc tính.")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        self.xyz_scheduler = get_expon_lr_func(
            lr_init=pos_lr_init * self.gaussians.spatial_lr_scale,
            lr_final=pos_lr_final * self.gaussians.spatial_lr_scale,
            lr_delay_mult=pos_lr_delay_mult,
            max_steps=pos_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = self.xyz_scheduler(iteration)

    # ---------------- Densify & Prune ----------------
    def add_densification_stats(self, viewspace_point_grad, visibility_filter):
        self.gaussians.xyz_gradient_accum[visibility_filter] += torch.norm(
            viewspace_point_grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.gaussians.denom[visibility_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                          clone_factor=2, split_factor=2,
                          min_world_size_ratio=1.0e-5, max_gaussians=None):
        grads = self.gaussians.xyz_gradient_accum / self.gaussians.denom.clamp_min(1)
        grads = torch.nan_to_num(grads, nan=0.0)

        if max_gaussians is not None and self.gaussians._xyz.shape[0] >= max_gaussians:
            self._prune_points(self._extra_prune_mask(min_opacity, max_screen_size,
                                                        extent, min_world_size_ratio))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        self._densify_and_clone(grads, max_grad, extent, clone_factor)
        self._densify_and_split(grads, max_grad, extent, split_factor)

        prune_mask = self._extra_prune_mask(min_opacity, max_screen_size, extent, min_world_size_ratio)
        self._prune_points(prune_mask)

        if max_gaussians is not None and self.gaussians._xyz.shape[0] > max_gaussians:
            n_excess = self.gaussians._xyz.shape[0] - max_gaussians
            _, idx_sorted = torch.sort(self.gaussians.opacity.squeeze(-1))
            excess_mask = torch.zeros(self.gaussians._xyz.shape[0], dtype=torch.bool, device=self.gaussians.device)
            excess_mask[idx_sorted[:n_excess]] = True
            self._prune_points(excess_mask)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extra_prune_mask(self, min_opacity, max_screen_size, extent, min_world_size_ratio):
        prune_mask = (self.gaussians.opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.gaussians.max_radii2D > max_screen_size
            big_points_ws = self.gaussians.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_points_vs | big_points_ws
        if min_world_size_ratio:
            tiny_points = self.gaussians.scaling.max(dim=1).values < min_world_size_ratio * extent
            prune_mask = prune_mask | tiny_points
        return prune_mask

    def prune_low_quality(self, min_opacity, min_visible_count=None):
        """Prune độc lập theo lịch riêng (pruning.schedule.interval/after_iter)."""
        prune_mask = (self.gaussians.opacity < min_opacity).squeeze(-1)
        if min_visible_count is not None:
            rarely_seen = self.gaussians.denom.squeeze(-1) < min_visible_count
            prune_mask = prune_mask | rarely_seen
        if prune_mask.any():
            self._prune_points(prune_mask)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, grad_threshold, extent, clone_factor=2):
        selected = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected = selected & (self.gaussians.scaling.max(dim=1).values <= self.percent_dense * extent)

        n_copies = max(1, clone_factor - 1)
        new_xyz = self.gaussians._xyz[selected].repeat(n_copies, 1)
        new_f_dc = self.gaussians._features_dc[selected].repeat(n_copies, 1, 1)
        new_f_rest = self.gaussians._features_rest[selected].repeat(n_copies, 1, 1)
        new_opacities = self.gaussians._opacity[selected].repeat(n_copies, 1)
        new_scaling = self.gaussians._scaling[selected].repeat(n_copies, 1)
        new_rotation = self.gaussians._rotation[selected].repeat(n_copies, 1)
        self._append_points(new_xyz, new_f_dc, new_f_rest, new_opacities, new_scaling, new_rotation)

    def _densify_and_split(self, grads, grad_threshold, extent, split_factor=2):
        N = max(2, split_factor)
        n_init = self.gaussians._xyz.shape[0]
        padded_grad = torch.zeros((n_init,), device=self.gaussians.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected = padded_grad >= grad_threshold
        selected = selected & (self.gaussians.scaling.max(dim=1).values > self.percent_dense * extent)

        stds = self.gaussians.scaling[selected].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.gaussians.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.gaussians._rotation[selected]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.gaussians._xyz[selected].repeat(N, 1)
        new_scaling = torch.log(self.gaussians.scaling[selected].repeat(N, 1) / (0.8 * N))
        new_rotation = self.gaussians._rotation[selected].repeat(N, 1)
        new_f_dc = self.gaussians._features_dc[selected].repeat(N, 1, 1)
        new_f_rest = self.gaussians._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self.gaussians._opacity[selected].repeat(N, 1)

        self._append_points(new_xyz, new_f_dc, new_f_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat([selected, torch.zeros(new_xyz.shape[0], dtype=torch.bool, device=self.gaussians.device)])
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
            setattr(self.gaussians, _PARAM_TO_ATTR[group["name"]], group["params"][0])

        n = self.gaussians._xyz.shape[0]
        self.gaussians.xyz_gradient_accum = torch.zeros((n, 1), device=self.gaussians.device)
        self.gaussians.denom = torch.zeros((n, 1), device=self.gaussians.device)
        self.gaussians.max_radii2D = torch.cat([self.gaussians.max_radii2D, torch.zeros(new_xyz.shape[0], device=self.gaussians.device)])

    def _prune_points(self, mask):
        valid = ~mask
        for group in self.optimizer.param_groups:
            group["params"][0] = nn.Parameter(group["params"][0][valid].requires_grad_(True))
            setattr(self.gaussians, _PARAM_TO_ATTR[group["name"]], group["params"][0])
        self.gaussians.xyz_gradient_accum = self.gaussians.xyz_gradient_accum[valid]
        self.gaussians.denom = self.gaussians.denom[valid]
        self.gaussians.max_radii2D = self.gaussians.max_radii2D[valid]

    def reset_opacity(self):
        new_opacity = inverse_sigmoid(torch.min(self.gaussians.opacity, torch.ones_like(self.gaussians.opacity) * 0.01))
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                group["params"][0] = nn.Parameter(new_opacity.requires_grad_(True))
                self.gaussians._opacity = group["params"][0]
