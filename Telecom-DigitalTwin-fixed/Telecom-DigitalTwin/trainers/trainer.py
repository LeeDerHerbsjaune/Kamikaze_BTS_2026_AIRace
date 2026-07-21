"""
Trainer: orchestrates the full training loop -
Load Batch -> Render -> Compute Loss -> Backward -> Optimizer Step -> Densify -> Prune

All config access in this file uses utils.config_loader.cfg_get() with a
dotted path matching the exact structure in configs/*.yaml. See README.md /
CHANGELOG.md for the full mapping between config keys and where they're used.

Cross-checked against the reference training loop in
graphdeco-inria/gaussian-splatting/train.py. Matches: SH degree increase
every N iters, densify_and_clone/split thresholds and formulas, the
add_densification_stats accumulation, periodic opacity reset. Two
deliberate deviations, documented inline where they occur:
  1. `optimizer.step()` runs BEFORE densify_and_prune here, whereas the
     reference runs it AFTER (see train() below for why).
  2. `max_screen_size` pruning is gated the same way as the reference
     (`size_threshold = 20 if iteration > opacity_reset_interval else None`)
     to avoid pruning by screen size before Gaussians have had a chance to
     converge - see `train()`.
"""
import os
import random
import time
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

from renderer.gaussian_renderer import render
from losses.loss import compute_loss
from evaluation.metrics import evaluate_dataset
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.geometry import build_rotation
from utils.config_loader import cfg_get

# Explicit mapping of optimizer param-group name -> real GaussianModel attribute.
# Do NOT infer this via string formatting (f"_{group['name']}") - that was a
# bug that caused a shape mismatch between _xyz and _features_dc/_features_rest
# after the first densify call.
_PARAM_TO_ATTR = {
    "xyz": "_xyz",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "opacity": "_opacity",
    "scaling": "_scaling",
    "rotation": "_rotation",
}

# Param-group name -> corresponding on/off flag in configs/gaussian.yaml (model.optimize.*)
_PARAM_TO_OPTIMIZE_FLAG = {
    "xyz": "position",
    "f_dc": "sh",
    "f_rest": "sh",
    "opacity": "opacity",
    "scaling": "scale",
    "rotation": "rotation",
}


class Trainer:
    def __init__(self, cfg, gaussians, dataset, logger,
                 resume_iteration: int = 0, resume_optimizer_state: dict = None):
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

        # Whether to include the Adam optimizer state (exp_avg/exp_avg_sq) in
        # every saved checkpoint, so a later --resume can restore momentum
        # instead of restarting Adam from zero. Off by default since it
        # roughly doubles+ checkpoint size (Adam keeps 2 extra tensors per
        # parameter) - not worth the disk cost unless you actually expect to
        # resume (e.g. training on a session-limited host like Kaggle).
        self.save_optimizer_state = cfg_get(cfg, "training.save_optimizer_state", False)

        self.train_cams = list(dataset.train_cameras)
        if not self.train_cams:
            raise ValueError("dataset.train_cameras is empty - nothing to train on.")

        # Eval history (iteration, metrics dict) - used to write the final
        # training summary to notes.md (see log_summary in train()).
        self.eval_history = []

        # IMPORTANT: gaussians.create_from_pcd(...) (or gaussians.restore(...)
        # when resuming) must have already run before constructing
        # Trainer(...), since setup_training needs gaussians.xyz to already
        # have its real size.
        self.setup_training(cfg)

        # ---------------- Resume ----------------
        # Continue the iteration counter from where the checkpoint left off
        # (NOT restart at 1), so the lr schedule / sh-degree schedule /
        # densify-prune-reset schedule - all driven by the ABSOLUTE iteration
        # number below - stay consistent with the Gaussian state that was
        # actually restored.
        self.start_iteration = resume_iteration + 1
        if resume_optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(resume_optimizer_state)
                self.logger.log_text(
                    "Restored optimizer state (Adam momentum) from checkpoint - resuming seamlessly.")
            except Exception as e:
                self.logger.log_text(
                    f"WARNING: could not restore optimizer state ({e}) - state doesn't match "
                    f"the current param-group structure (e.g. different Gaussian count from "
                    f"densify), falling back to a freshly-initialized optimizer. Gaussian "
                    f"weights themselves are unaffected.")
        elif resume_iteration > 0:
            self.logger.log_text(
                "Resuming without a saved optimizer state (checkpoint predates "
                "training.save_optimizer_state=true, or it was off when that checkpoint was "
                "saved) - Adam momentum restarts from zero. Gaussian weights are unaffected; "
                "expect only a brief, minor dip/wobble in loss for the first few hundred "
                "iterations after resume.")

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

        if self.start_iteration > n_iters:
            self.logger.log_text(
                f"start_iteration ({self.start_iteration}) > training.iterations ({n_iters}) - "
                f"the resumed checkpoint is already past the configured target, nothing to "
                f"train further. Raise 'training.iterations' in the config if you want to "
                f"continue training this run.")
            return

        pbar = tqdm(range(self.start_iteration, n_iters + 1), desc="Training")
        cam_pool = []
        train_start_time = time.time()
        last_log_time = train_start_time
        n_gaussians_before_run = self.gaussians.xyz.shape[0]

        self.logger.log_text(
            f"Starting training: {n_iters} iterations"
            f"{f' (resuming from iteration {self.start_iteration})' if self.start_iteration > 1 else ''}, "
            f"{n_gaussians_before_run} initial Gaussians, "
            f"{len(self.train_cams)} train views, {len(self.dataset.eval_cameras)} eval views, "
            f"backend={self.backend}, densify={'on' if densify_enabled else 'off'}.")

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
            # NOTE on ordering vs. the reference: the official train.py calls
            # optimizer.step() AFTER densify_and_prune()/reset_opacity() in
            # the same iteration. Because densify/reset always replace
            # optimizer param tensors with freshly-created nn.Parameter
            # objects (via cat_tensors_to_optimizer/_prune_optimizer/
            # replace_tensor_to_optimizer), those fresh tensors have
            # `.grad = None` - so on iterations where densify/reset actually
            # fires, the reference's optimizer.step() call right after is
            # effectively a no-op (this iteration's gradient is discarded).
            # This is a long-standing, accepted quirk of the reference
            # codebase (happens roughly once every densify.interval steps),
            # not something we've seen documented as intentional, but it's
            # been the de-facto behavior since the paper's release. We
            # deliberately step the optimizer BEFORE densify/prune instead,
            # so every computed gradient is actually applied - arguably more
            # correct, at the cost of not being bit-for-bit identical to the
            # reference's training dynamics.
            self.optimizer.step()

            # -------- Densify / Prune on the main schedule --------
            with torch.no_grad():
                if (densify_enabled
                        and iteration > densify_from_iter
                        and iteration < densify_until_iter
                        and iteration % densify_interval == 0):
                    extent = self._scene_extent()
                    n_before = self.gaussians.xyz.shape[0]
                    # Matches reference: `size_threshold = 20 if iteration >
                    # opacity_reset_interval else None` - screen-size pruning
                    # is disabled until Gaussians have survived at least one
                    # opacity reset, so recently-created (still-converging)
                    # Gaussians aren't pruned away purely for looking large
                    # on screen before they've had a chance to shrink/settle.
                    screen_size_active = max_screen_size if iteration > opacity_reset_interval else None
                    self.densify_and_prune(
                        densify_grad_threshold, min_opacity, extent, screen_size_active,
                        clone_factor=clone_factor, split_factor=split_factor,
                        min_world_size_ratio=min_world_size_ratio,
                        max_gaussians=max_gaussians)
                    n_after = self.gaussians.xyz.shape[0]
                    self.logger.log_text(
                        f"[densify @ iter {iteration}] n_gaussians: {n_before} -> {n_after} "
                        f"({'+' if n_after >= n_before else ''}{n_after - n_before})")

                # -------- Light-weight prune on its own schedule, independent of densify --------
                if (iteration > prune_after_iter and iteration % prune_interval == 0):
                    self.prune_low_quality(min_opacity, min_visible_count=min_visible_count)

                if opacity_reset_interval and iteration % opacity_reset_interval == 0:
                    self.reset_opacity()

            if iteration % log_interval == 0:
                now = time.time()
                elapsed_since_last = now - last_log_time
                it_per_sec = log_interval / elapsed_since_last if elapsed_since_last > 0 else None
                last_log_time = now
                current_lr = next((g["lr"] for g in self.optimizer.param_groups if g["name"] == "xyz"), None)

                pbar.set_postfix(loss=loss.item(), n_gaussians=self.gaussians.xyz.shape[0])
                self.logger.log_scalar("train/loss", loss.item(), iteration)
                self.logger.log_scalar("train/n_gaussians", self.gaussians.xyz.shape[0], iteration)
                self.logger.log_progress(iteration, n_iters, loss.item(), self.gaussians.xyz.shape[0],
                                         it_per_sec=it_per_sec, lr=current_lr)

            if iteration % eval_interval == 0 and self.dataset.eval_cameras:
                # Free cached CUDA memory before eval: training accumulates
                # freed-but-retained blocks (PyTorch's caching allocator
                # keeps them around for faster reuse). Right before eval
                # (which also runs the fairly heavy LPIPS/VGG network) is a
                # good point to release that cache and reduce fragmentation.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    metrics = evaluate_dataset(self.dataset.eval_cameras, self.gaussians,
                                               render, self.bg_color,
                                               use_lpips=use_lpips_eval)
                    if metrics:
                        self.logger.log_dict("eval", metrics, iteration)
                        self.eval_history.append((iteration, metrics))
                except torch.cuda.OutOfMemoryError as e:
                    # Running out of VRAM during eval should NOT kill the
                    # whole training run (which may have already run for
                    # tens of minutes) - skip this eval round, clear the
                    # cache, and keep training. If this keeps happening,
                    # consider lowering 'dataset.images.resolution' or
                    # setting 'evaluation.use_lpips: false' in the config.
                    self.logger.log_text(
                        f"[eval @ iter {iteration}] WARNING: out of VRAM during evaluate "
                        f"(likely LPIPS/VGG memory usage) - SKIPPING this eval round, "
                        f"continuing training. If this repeats, lower "
                        f"'dataset.images.resolution' or set 'evaluation.use_lpips: false' "
                        f"in the config. Error detail: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if iteration % save_interval == 0 or iteration == n_iters:
                self.save_checkpoint(iteration)

        self.save_checkpoint(n_iters, name="last")

        total_time = time.time() - train_start_time
        summary_lines = [
            f"Experiment: {cfg.get('experiment_name', '-')}",
            f"Total iterations: {n_iters}",
            f"Training time: {datetime.timedelta(seconds=int(total_time))}",
            f"Gaussians: {n_gaussians_before_run} (initial) -> {self.gaussians.xyz.shape[0]} (final)",
            f"Final loss: {loss.item():.4f}",
        ]
        if self.eval_history:
            last_iter, last_metrics = self.eval_history[-1]
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in last_metrics.items())
            summary_lines.append(f"Last eval (iter {last_iter}): {metrics_str}")
            best_psnr_iter, best_psnr_metrics = max(
                self.eval_history, key=lambda x: x[1].get("psnr", float("-inf")))
            summary_lines.append(
                f"Best PSNR: {best_psnr_metrics.get('psnr', float('nan')):.4f} at iter {best_psnr_iter}")
        else:
            summary_lines.append("No successful eval runs (check eval_cameras is non-empty, or repeated OOM).")
        self.logger.log_summary("Training summary", summary_lines)

    def _scene_extent(self):
        from utils.camera_utils import cameras_extent
        return cameras_extent(self.train_cams)

    def save_checkpoint(self, iteration, name=None):
        name = name or f"iter_{iteration}"
        path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        payload = {"iteration": iteration, "gaussians": self.gaussians.capture()}
        if self.save_optimizer_state:
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, path)
        self.logger.log_text(
            f"Saved checkpoint: {path}" + (" (with optimizer state)" if self.save_optimizer_state else ""))

        note = f"{self.gaussians.xyz.shape[0]} Gaussians"
        if self.eval_history and self.eval_history[-1][0] <= iteration:
            last_metrics = self.eval_history[-1][1]
            note += ", latest eval: " + ", ".join(f"{k}={v:.4f}" for k, v in last_metrics.items())
        self.logger.log_artifact("checkpoint", path, iteration=iteration, note=note)

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

        # Respect model.optimize.{position,rotation,scale,opacity,sh} from
        # configs/gaussian.yaml: any disabled param-group gets
        # requires_grad=False and is excluded from the optimizer (so Adam
        # doesn't waste state/steps on it).
        params = []
        for group in all_groups:
            flag_key = _PARAM_TO_OPTIMIZE_FLAG[group["name"]]
            enabled = cfg_get(cfg, f"model.optimize.{flag_key}", True)
            group["params"][0].requires_grad_(enabled)
            if enabled:
                params.append(group)
            else:
                self.logger.log_text(f"[optimize] param-group '{group['name']}' disabled "
                                      f"(model.optimize.{flag_key}=false), will not be learned.")

        if not params:
            raise ValueError("model.optimize.* disabled every param-group - nothing to train.")

        # densify/prune (_append_points, _prune_points) assume ALL 6
        # Gaussian attributes (_xyz, _features_dc/_rest, _opacity, _scaling,
        # _rotation) always have the same point count N and get resized in
        # lockstep via optimizer.param_groups. If one param-group is disabled
        # (model.optimize.*=false) it will NOT be resized along with the
        # others, causing a shape mismatch on the very first densify/prune.
        # So partial optimize-flag disabling is only allowed when densify is
        # also disabled.
        if len(params) < len(all_groups) and cfg_get(cfg, "densify.enabled", True):
            raise ValueError(
                "Cannot disable individual model.optimize.* flags while "
                "densify.enabled=true: densify/prune needs to resize all 6 "
                "Gaussian attributes in lockstep. Set densify.enabled=false "
                "if you want to freeze some attributes.")

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
    # Formulas cross-checked against GaussianModel.densify_and_clone/
    # densify_and_split/densify_and_prune in the reference scene/gaussian_model.py:
    # thresholds, clone/split selection masks, and the split scale formula
    # `log(scaling / (0.8*N))` all match exactly.
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
        # NOTE: this tiny-point removal criterion is NOT present in the
        # reference implementation - it's an addition of ours to clean up
        # degenerate near-zero-size Gaussians after densify/split, gated by
        # densify.min_world_size_ratio (0 disables it entirely).
        if min_world_size_ratio:
            tiny_points = self.gaussians.scaling.max(dim=1).values < min_world_size_ratio * extent
            prune_mask = prune_mask | tiny_points
        return prune_mask

    def prune_low_quality(self, min_opacity, min_visible_count=None):
        """Independent light-weight prune on its own schedule
        (pruning.schedule.interval/after_iter) - not present in the
        reference, added so pruning can run more often than the main
        densify cycle without the clone/split overhead."""
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
            old_param = group["params"][0]
            # IMPORTANT: Adam's internal state (exp_avg/exp_avg_sq) is keyed
            # by the PARAMETER OBJECT ITSELF. Simply replacing
            # group["params"][0] with a brand-new nn.Parameter (as done
            # previously) leaves the old parameter's state entry orphaned in
            # self.optimizer.state - it is never freed since nothing
            # references old_param anymore except that dict, but Python
            # can't garbage-collect it because the dict itself still holds
            # it. Over ~100 densify events this leaks an ever-growing set of
            # CUDA tensors, eventually causing an OOM unrelated to the
            # current Gaussian count. Must explicitly move the state to the
            # new parameter (extending with zeros for the new points).
            stored_state = self.optimizer.state.get(old_param, None)
            new_param = nn.Parameter(torch.cat([old_param, extension], dim=0).requires_grad_(True))
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    [stored_state["exp_avg"], torch.zeros_like(extension)], dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    [stored_state["exp_avg_sq"], torch.zeros_like(extension)], dim=0)
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = stored_state
            group["params"][0] = new_param
            setattr(self.gaussians, _PARAM_TO_ATTR[group["name"]], group["params"][0])

        n = self.gaussians._xyz.shape[0]
        self.gaussians.xyz_gradient_accum = torch.zeros((n, 1), device=self.gaussians.device)
        self.gaussians.denom = torch.zeros((n, 1), device=self.gaussians.device)
        self.gaussians.max_radii2D = torch.cat([self.gaussians.max_radii2D, torch.zeros(new_xyz.shape[0], device=self.gaussians.device)])

    def _prune_points(self, mask):
        valid = ~mask
        for group in self.optimizer.param_groups:
            old_param = group["params"][0]
            # Same leak as in _append_points (see comment there): must
            # explicitly carry Adam's state over to the new parameter
            # object, indexed by the same `valid` mask, instead of leaving
            # the old (larger) state tensors orphaned in optimizer.state.
            stored_state = self.optimizer.state.get(old_param, None)
            new_param = nn.Parameter(old_param[valid].requires_grad_(True))
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid]
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = stored_state
            group["params"][0] = new_param
            setattr(self.gaussians, _PARAM_TO_ATTR[group["name"]], group["params"][0])
        self.gaussians.xyz_gradient_accum = self.gaussians.xyz_gradient_accum[valid]
        self.gaussians.denom = self.gaussians.denom[valid]
        self.gaussians.max_radii2D = self.gaussians.max_radii2D[valid]

    def reset_opacity(self):
        new_opacity = inverse_sigmoid(torch.min(self.gaussians.opacity, torch.ones_like(self.gaussians.opacity) * 0.01))
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                old_param = group["params"][0]
                # Same leak pattern as _append_points/_prune_points (see
                # comments there). Shape doesn't change here, but the old
                # parameter object's Adam state would still be orphaned if
                # not migrated. Also reset exp_avg/exp_avg_sq to zero (not
                # just copy over) since the opacity values just changed
                # discontinuously - keeping stale momentum from before the
                # reset would fight the new values.
                stored_state = self.optimizer.state.get(old_param, None)
                new_param = nn.Parameter(new_opacity.requires_grad_(True))
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(new_opacity)
                    stored_state["exp_avg_sq"] = torch.zeros_like(new_opacity)
                    del self.optimizer.state[old_param]
                    self.optimizer.state[new_param] = stored_state
                group["params"][0] = new_param
                self.gaussians._opacity = group["params"][0]