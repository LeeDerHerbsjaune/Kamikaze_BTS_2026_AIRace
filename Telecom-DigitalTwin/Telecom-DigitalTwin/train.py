"""
train.py - Entry point chính, nối đúng workflow:

Competition Dataset -> configs/default.yaml -> train.py
    -> Load Config / Init Logger / Set Seed
    -> Build Dataset -> Preprocessing -> Build Gaussian Model
    -> Build Renderer -> Build Loss -> Build Optimizer -> Trainer
    -> Training Loop -> Validation -> Save Checkpoint
    -> Inference -> Novel View Images

Chạy:
    python train.py --config configs/default.yaml
"""
import argparse
import os
import numpy as np
import torch

from utils.general_utils import set_seed
from utils.logger import Logger
from utils.config_loader import load_config
from datasets.bts_dataset import BTSDataset
from models.gaussian_model import GaussianModel
from trainers.trainer import Trainer
from visualization.viz_utils import plot_camera_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # ---------- Load Config ----------
    cfg = load_config(args.config)
    exp_dir = os.path.join("outputs", cfg["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    cfg["train"]["checkpoint_dir"] = os.path.join(exp_dir, "checkpoints")
    cfg["inference"]["output_dir"] = os.path.join(exp_dir, "novel_views")

    # ---------- Init Logger ----------
    logger = Logger(log_dir=exp_dir)
    logger.log_text(f"Bắt đầu experiment: {cfg['experiment_name']}")

    # ---------- Set Random Seed ----------
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.log_text("CẢNH BÁO: không tìm thấy CUDA, chạy trên CPU sẽ rất chậm với 3DGS.")

    # ---------- Build Dataset (+ Preprocessing bên trong) ----------
    logger.log_text("Đang load dataset (ảnh drone + camera pose)...")
    dataset = BTSDataset(cfg, device=device)
    logger.log_text(f"Train views: {len(dataset.train_cameras)}, "
                     f"Eval views: {len(dataset.eval_cameras)}, "
                     f"Target novel views: {len(dataset.target_cameras)}")

    plot_camera_trajectory(dataset.train_cameras, dataset.eval_cameras, dataset.target_cameras,
                            save_path=os.path.join(exp_dir, "camera_trajectory.png"))

    # ---------- Build Gaussian Model ----------
    logger.log_text("Khởi tạo Gaussian Model từ point cloud...")
    gaussians = GaussianModel(cfg["model"]["sh_degree"], device=device)

    if dataset.point_cloud_xyz is not None:
        xyz, rgb = dataset.point_cloud_xyz, dataset.point_cloud_rgb
    else:
        # Không có SfM point cloud (vd dùng nerf_transforms format) -> khởi tạo random trong bbox scene
        n = cfg["preprocessing"]["num_random_points"]
        xyz = (np.random.rand(n, 3) * 2 - 1) * 2.0
        rgb = np.random.rand(n, 3)
        logger.log_text(f"Không tìm thấy SfM point cloud, khởi tạo random {n} điểm.")

    from utils.camera_utils import cameras_extent
    spatial_lr_scale = cameras_extent(dataset.train_cameras)
    gaussians.create_from_pcd(xyz, rgb, spatial_lr_scale)
    logger.log_text(f"Số Gaussian khởi tạo: {gaussians.xyz.shape[0]}")

    # ---------- Build Renderer / Loss / Optimizer (setup) ----------
    gaussians.setup_training(cfg)  # optimizer + LR schedule

    # ---------- Trainer ----------
    trainer = Trainer(cfg, gaussians, dataset, logger)

    # ---------- Training Loop + Validation + Save Checkpoint ----------
    trainer.train()

    gaussians.save_ply(os.path.join(exp_dir, "point_cloud_final.ply"))
    logger.log_text("Huấn luyện hoàn tất.")

    # ---------- Inference -> Novel View Images ----------
    logger.log_text("Bắt đầu render novel views mục tiêu...")
    from inference.render_novel_views import main as run_inference
    cfg["inference"]["checkpoint"] = os.path.join(cfg["train"]["checkpoint_dir"], "last.pth")
    run_inference_from_cfg(cfg, gaussians, dataset, logger)


def run_inference_from_cfg(cfg, gaussians, dataset, logger):
    """Render trực tiếp bằng model vừa train xong (không cần load lại checkpoint từ disk)."""
    import torch
    from renderer.gaussian_renderer import render
    from inference.render_novel_views import tensor_to_pil, _save_video
    import os

    bg_color = torch.tensor(cfg["renderer"]["bg_color"], dtype=torch.float32, device=gaussians.device)
    out_dir = cfg["inference"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for cam in dataset.target_cameras:
        with torch.no_grad():
            out = render(cam, gaussians, bg_color, backend=cfg["renderer"]["backend"])
        pil_img = tensor_to_pil(out["render"])
        pil_img.save(os.path.join(out_dir, f"{cam.image_name}.png"))
        frames.append(pil_img)

    logger.log_text(f"Đã render {len(dataset.target_cameras)} novel view images vào {out_dir}")
    if cfg["inference"]["render_video"] and frames:
        import numpy as np
        _save_video([np.array(f) for f in frames], os.path.join(out_dir, "novel_views.mp4"),
                    cfg["inference"]["fps"])


if __name__ == "__main__":
    main()
