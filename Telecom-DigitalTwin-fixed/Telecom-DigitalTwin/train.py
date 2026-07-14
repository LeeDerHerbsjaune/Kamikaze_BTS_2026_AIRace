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
from utils.config_loader import load_config, cfg_get
from datasets.bts_dataset import BTSDataset
from models.gaussian_model import GaussianModel
from trainers.trainer import Trainer
from visualization.viz_utils import plot_camera_trajectory


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override dataset root"
    )
    args = parser.parse_args()
    
    # ---------- Load Config ----------
    cfg = load_config(args.config)
    # Override dataset root
    if args.data_root is not None:
        cfg["dataset"]["root"] = args.data_root 
    exp_dir = os.path.join("outputs", cfg["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    # Override thư mục output theo experiment_name (đè lên giá trị tĩnh trong
    # configs/train.yaml và configs/inference.yaml, vốn chỉ là default).
    cfg.setdefault("training", {})["checkpoint_dir"] = os.path.join(exp_dir, "checkpoints")
    cfg.setdefault("inference", {})["output_dir"] = os.path.join(exp_dir, "novel_views")

    # ---------- Init Logger ----------
    logger = Logger(log_dir=exp_dir)
    logger.log_text(f"Bắt đầu experiment: {cfg['experiment_name']}")

    # ---------- Set Random Seed ----------
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
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
    sh_degree = cfg_get(cfg, "model.sh_degree", 3)
    gaussians = GaussianModel(sh_degree, device=device)

    init_position = cfg_get(cfg, "model.init.position", "pointcloud")
    opacity_init = cfg_get(cfg, "model.init.opacity", 0.1)
    scale_init_factor = cfg_get(cfg, "model.init.scale_factor", 1.0)

    if dataset.point_cloud_xyz is not None and init_position == "pointcloud":
        xyz, rgb = dataset.point_cloud_xyz, dataset.point_cloud_rgb
    else:
        # Không có SfM point cloud (vd dùng nerf_transforms format), hoặc
        # model.init.position="random" được yêu cầu tường minh trong config.
        n = cfg_get(cfg, "preprocessing.initialize_pointcloud.random_points", 100000)
        xyz = (np.random.rand(n, 3) * 2 - 1) * 2.0
        rgb = np.random.rand(n, 3)
        logger.log_text(f"Không dùng SfM point cloud (init_position='{init_position}'), "
                         f"khởi tạo random {n} điểm.")

    from utils.camera_utils import cameras_extent
    spatial_lr_scale = cameras_extent(dataset.train_cameras)
    gaussians.create_from_pcd(xyz, rgb, spatial_lr_scale,
                               opacity_init=opacity_init, scale_init_factor=scale_init_factor)
    logger.log_text(f"Số Gaussian khởi tạo: {gaussians.xyz.shape[0]}")

    # ---------- Trainer (build renderer/loss/optimizer bên trong) ----------
    trainer = Trainer(cfg, gaussians, dataset, logger)

    # ---------- Training Loop + Validation + Save Checkpoint ----------
    trainer.train()

    gaussians.save_ply(os.path.join(exp_dir, "point_cloud_final.ply"))
    logger.log_text("Huấn luyện hoàn tất.")

    # ---------- Inference -> Novel View Images ----------
    logger.log_text("Bắt đầu render novel views mục tiêu...")
    cfg["inference"]["checkpoint"] = os.path.join(cfg["training"]["checkpoint_dir"], "last.pth")
    run_inference_from_cfg(cfg, gaussians, dataset, logger)

    logger.close()


def run_inference_from_cfg(cfg, gaussians, dataset, logger):
    """Render trực tiếp bằng model vừa train xong (không cần load lại checkpoint từ disk)."""
    import torch
    from renderer.gaussian_renderer import render
    from inference.render_novel_views import tensor_to_pil, _save_video

    if not dataset.target_cameras:
        logger.log_text("Không có target novel views (dataset.target_views.file rỗng/không tồn tại) - bỏ qua bước inference.")
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

    logger.log_text(f"Đã render {len(dataset.target_cameras)} novel view images vào {out_dir}")
    if render_video and frames:
        _save_video([np.array(f) for f in frames], os.path.join(out_dir, "novel_views.mp4"), fps)


if __name__ == "__main__":
    main()
