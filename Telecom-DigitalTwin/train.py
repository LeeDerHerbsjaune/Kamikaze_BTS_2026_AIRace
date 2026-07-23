"""
train.py - Main entry point, wiring the full workflow:

Competition Dataset -> configs/default.yaml -> train.py
    -> Load Config / Init Logger / Set Seed
    -> Build Dataset -> Preprocessing -> Build Gaussian Model
    -> Build Renderer -> Build Loss -> Build Optimizer -> Trainer
    -> Training Loop -> Validation -> Save Checkpoint
    -> Inference -> Novel View Images

Run:
    python train.py --config configs/default.yaml
"""
import argparse
import os

# IMPORTANT: this env var must be set BEFORE `import torch` (here or in any
# module imported below) - PyTorch's CUDA allocator only reads this config
# at initialization. Enabling expandable_segments significantly reduces OOM
# caused by memory fragmentation (it doesn't reduce total VRAM needs, but
# uses already-allocated memory more efficiently) - matches the suggestion
# in the original CUDA OOM error message. Both env var names are set since
# PyTorch has renamed this across versions.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from utils.general_utils import set_seed
from utils.logger import Logger
from utils.config_loader import load_config, cfg_get
from dataloader.bts_dataset import BTSDataset
from models.gaussian_model import GaussianModel
from trainers.trainer import Trainer
from visualization.viz_utils import plot_camera_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_root", type=str, default=None,
                         help="Override dataset.root from the config (handy on Kaggle/Colab "
                              "when the dataset path changes between runs and you don't "
                              "want to edit the yaml).")
    args = parser.parse_args()

    # ---------- Load Config ----------
    cfg = load_config(args.config)
    if args.data_root:
        cfg.setdefault("dataset", {})["root"] = args.data_root

    output_root = cfg.get("output_root", "outputs")
    exp_dir = os.path.join(output_root, cfg["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)

    # Override the output directories by experiment_name (overrides the
    # static values in configs/train.yaml and configs/inference.yaml, which
    # are just defaults).
    cfg.setdefault("training", {})["checkpoint_dir"] = os.path.join(exp_dir, "checkpoints")
    cfg.setdefault("inference", {})["output_dir"] = os.path.join(exp_dir, "novel_views")

    # ---------- Init Logger ----------
    logger = Logger(log_dir=exp_dir)
    logger.log_text(f"Starting experiment: {cfg['experiment_name']}")

    # ---------- Set Random Seed ----------
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.log_text("WARNING: no CUDA found, running on CPU will be very slow for 3DGS.")

    # ---------- Build Dataset (+ Preprocessing inside) ----------
    logger.log_text("Loading dataset (drone images + camera poses)...")
    dataset = BTSDataset(cfg, device=device)
    logger.log_text(f"Train views: {len(dataset.train_cameras)}, "
                     f"Eval views: {len(dataset.eval_cameras)}, "
                     f"Target novel views: {len(dataset.target_cameras)}")

    plot_camera_trajectory(dataset.train_cameras, dataset.eval_cameras, dataset.target_cameras,
                            save_path=os.path.join(exp_dir, "camera_trajectory.png"))
    logger.log_artifact("camera_trajectory", os.path.join(exp_dir, "camera_trajectory.png"),
                        note=f"{len(dataset.train_cameras)} train, {len(dataset.eval_cameras)} eval, "
                             f"{len(dataset.target_cameras)} target")

    # ---------- Build Gaussian Model ----------
    logger.log_text("Initializing Gaussian Model from the point cloud...")
    sh_degree = cfg_get(cfg, "model.sh_degree", 3)
    gaussians = GaussianModel(sh_degree, device=device)

    init_position = cfg_get(cfg, "model.init.position", "pointcloud")
    opacity_init = cfg_get(cfg, "model.init.opacity", 0.1)
    scale_init_factor = cfg_get(cfg, "model.init.scale_factor", 1.0)

    if dataset.point_cloud_xyz is not None and init_position == "pointcloud":
        xyz, rgb = dataset.point_cloud_xyz, dataset.point_cloud_rgb
    else:
        # No SfM point cloud available (e.g. using the nerf_transforms
        # format), or model.init.position="random" was explicitly requested
        # in the config.
        n = cfg_get(cfg, "preprocessing.initialize_pointcloud.random_points", 100000)
        xyz = (np.random.rand(n, 3) * 2 - 1) * 2.0
        rgb = np.random.rand(n, 3)
        logger.log_text(f"Not using an SfM point cloud (init_position='{init_position}'), "
                         f"initializing {n} random points.")

    from utils.camera_utils import cameras_extent
    spatial_lr_scale = cameras_extent(dataset.train_cameras)
    gaussians.create_from_pcd(xyz, rgb, spatial_lr_scale,
                               opacity_init=opacity_init, scale_init_factor=scale_init_factor)
    logger.log_text(f"Initialized {gaussians.xyz.shape[0]} Gaussians")

    # ---------- Trainer (builds renderer/loss/optimizer internally) ----------
    trainer = Trainer(cfg, gaussians, dataset, logger)

    # ---------- Training Loop + Validation + Save Checkpoint ----------
    trainer.train()

    ply_path = os.path.join(exp_dir, "point_cloud_final.ply")
    gaussians.save_ply(ply_path)
    logger.log_artifact("point_cloud", ply_path, note=f"{gaussians.xyz.shape[0]} Gaussians")
    logger.log_text("Training complete.")

    # ---------- Inference -> Novel View Images ----------
    logger.log_text("Starting target novel view rendering...")
    cfg["inference"]["checkpoint"] = os.path.join(cfg["training"]["checkpoint_dir"], "last.pth")
    run_inference_from_cfg(cfg, gaussians, dataset, logger)

    logger.close()


def run_inference_from_cfg(cfg, gaussians, dataset, logger):
    """Render directly with the model that was just trained (no need to
    reload the checkpoint from disk)."""
    import torch
    from renderer.gaussian_renderer import render
    from inference.render_novel_views import tensor_to_pil, _save_video

    if not dataset.target_cameras:
        logger.log_text("No target novel views (dataset.target_views.file is empty/missing) - skipping inference.")
        logger.log_summary("Inference summary", [
            "SKIPPED - no target novel views (check dataset.target_views.file in the config).",
        ])
        return

    bg_color = torch.tensor(cfg_get(cfg, "renderer.background.color", [0, 0, 0]),
                            dtype=torch.float32, device=gaussians.device)
    backend = cfg_get(cfg, "renderer.backend", "gsplat")
    out_dir = cfg["inference"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    render_video = cfg_get(cfg, "inference.render.video.enabled", True)
    fps = cfg_get(cfg, "inference.render.video.fps", 24)

    frames = []
    for cam in dataset.target_cameras:
        with torch.no_grad():
            out = render(cam, gaussians, bg_color, backend=backend)
        pil_img = tensor_to_pil(out["render"])
        pil_img.save(os.path.join(out_dir, f"{cam.image_name}.png"))
        frames.append(pil_img)

    logger.log_text(f"Rendered {len(dataset.target_cameras)} novel view images to {out_dir}")
    logger.log_artifact("novel_view_images", out_dir, note=f"{len(dataset.target_cameras)} images")

    if render_video and frames:
        video_path = os.path.join(out_dir, "novel_views.mp4")
        _save_video([np.array(f) for f in frames], video_path, fps)
        logger.log_artifact("novel_view_video", video_path, note=f"{fps} fps, {len(frames)} frames")

    logger.log_summary("Inference summary", [
        f"Novel views rendered: {len(dataset.target_cameras)}",
        f"Output directory: {out_dir}",
        f"Video preview: {'yes' if render_video and frames else 'not created (render_video=false or no frames)'}",
    ])


if __name__ == "__main__":
    main()
