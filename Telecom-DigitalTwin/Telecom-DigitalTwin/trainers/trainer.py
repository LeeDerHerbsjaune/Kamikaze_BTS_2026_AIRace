"""
Trainer: điều phối toàn bộ training loop -
Load Batch -> Render -> Compute Loss -> Backward -> Optimizer Step -> Densify -> Prune
đúng như sơ đồ workflow, cộng thêm validation định kỳ và lưu checkpoint.
"""
import os
import random
import torch
from tqdm import tqdm

from renderer.gaussian_renderer import render
from losses.loss import compute_loss
from evaluation.metrics import evaluate_dataset


class Trainer:
    def __init__(self, cfg, gaussians, dataset, logger):
        self.cfg = cfg
        self.gaussians = gaussians
        self.dataset = dataset
        self.logger = logger

        self.bg_color = torch.tensor(cfg["renderer"]["bg_color"], dtype=torch.float32,
                                     device=gaussians.device)
        self.backend = cfg["renderer"]["backend"]

        self.checkpoint_dir = cfg["train"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_cams = list(dataset.train_cameras)

    def train(self):
        cfg = self.cfg
        n_iters = cfg["optimizer"]["iterations"]
        densify_cfg = cfg["densify"]

        pbar = tqdm(range(1, n_iters + 1), desc="Training")
        cam_pool = []

        for iteration in pbar:
            self.gaussians.update_learning_rate(iteration)

            if iteration % cfg["train"]["sh_up_interval"] == 0:
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
            loss, loss_parts = compute_loss(rendered, gt, cfg["loss"]["lambda_dssim"])

            # -------- Backward --------
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # -------- Densification stats (trước optimizer step, cần grad còn giữ) --------
            with torch.no_grad():
                if iteration < densify_cfg["densify_until_iter"]:
                    if out["viewspace_points"].grad is not None:
                        self.gaussians.max_radii2D[out["visibility_filter"]] = torch.max(
                            self.gaussians.max_radii2D[out["visibility_filter"]],
                            out["radii"][out["visibility_filter"]])
                        self.gaussians.add_densification_stats(
                            out["viewspace_points"].grad, out["visibility_filter"])

            # -------- Optimizer step --------
            self.gaussians.optimizer.step()

            # -------- Densify / Prune --------
            with torch.no_grad():
                if (iteration > densify_cfg["densify_from_iter"]
                        and iteration < densify_cfg["densify_until_iter"]
                        and iteration % densify_cfg["densification_interval"] == 0):
                    extent = self._scene_extent()
                    self.gaussians.densify_and_prune(
                        densify_cfg["densify_grad_threshold"],
                        densify_cfg["min_opacity"], extent,
                        densify_cfg["max_screen_size"])

                if iteration % densify_cfg["opacity_reset_interval"] == 0:
                    self.gaussians.reset_opacity()

            if iteration % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix(loss=loss.item(), n_gaussians=self.gaussians.xyz.shape[0])
                self.logger.log_scalar("train/loss", loss.item(), iteration)
                self.logger.log_scalar("train/n_gaussians", self.gaussians.xyz.shape[0], iteration)

            if iteration % cfg["train"]["eval_interval"] == 0 and self.dataset.eval_cameras:
                metrics = evaluate_dataset(self.dataset.eval_cameras, self.gaussians,
                                           render, self.bg_color,
                                           use_lpips=cfg["loss"]["use_lpips_eval"])
                self.logger.log_dict("eval", metrics, iteration)

            if iteration % cfg["train"]["save_interval"] == 0 or iteration == n_iters:
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
