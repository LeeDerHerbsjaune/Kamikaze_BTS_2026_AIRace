"""
Inference: load checkpoint đã train, render novel view mục tiêu do đề bài
cung cấp (không có ground truth) và lưu ra ảnh PNG (+ tuỳ chọn dựng video).
Đây là bước cuối trong workflow: Inference -> Novel View Images.

Nguồn pose mục tiêu (dataset.target_views.file trong config) hỗ trợ 2 định
dạng, tự nhận diện theo đuôi file (xem dataloader/bts_dataset.py):
  - target_poses.json : R/T/FoVx/FoVy tường minh.
  - test_poses.csv     : quaternion + intrinsics tường minh - dùng cho bộ
    test/submission do đề bài cấp riêng để chấm điểm.
Không cần đổi gì ở file này để đọc CSV - BTSDataset tự lo phần đó. File này
chỉ thêm tuỳ chọn đóng gói `submission.zip` sau khi render xong (bật qua
`inference.output.zip: true` trong config) cho tiện nộp bài.
"""
import os
import argparse
import zipfile

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.gaussian_model import GaussianModel
from dataloader.bts_dataset import BTSDataset
from renderer.gaussian_renderer import render
from utils.config_loader import load_config, cfg_get


def tensor_to_pil(img_tensor):
    arr = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main(cfg_path):
    cfg = load_config(cfg_path)

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    gaussians = GaussianModel(cfg_get(cfg, "model.sh_degree", 3), device=device)

    checkpoint_path = cfg_get(cfg, "inference.checkpoint")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại '{checkpoint_path}' (inference.checkpoint). "
            f"Chạy training trước, hoặc sửa lại đường dẫn trong config.")

    ckpt = torch.load(checkpoint_path, map_location=device)
    gaussians.restore(ckpt["gaussians"])
    print(f"Đã load checkpoint tại iteration {ckpt['iteration']}, "
          f"số Gaussian: {gaussians.xyz.shape[0]}")

    dataset = BTSDataset(cfg, device=device)
    bg_color = torch.tensor(cfg_get(cfg, "renderer.background.color", [0, 0, 0]),
                            dtype=torch.float32, device=device)

    out_dir = cfg_get(cfg, "inference.output_dir") or cfg_get(
        cfg, "inference.output.directory", "./outputs/novel_views")
    os.makedirs(out_dir, exist_ok=True)

    if not dataset.target_cameras:
        print("Cảnh báo: dataset.target_views.file không tìm thấy hoặc rỗng - "
              "không có novel view nào để render.")
        return

    render_images = cfg_get(cfg, "inference.render.images", True)
    render_video = cfg_get(cfg, "inference.render.video.enabled", True)
    fps = cfg_get(cfg, "inference.render.video.fps", 24)
    backend = cfg_get(cfg, "renderer.backend", "gsplat")

    frames = []
    for cam in tqdm(dataset.target_cameras, desc="Rendering novel views"):
        with torch.no_grad():
            out = render(cam, gaussians, bg_color, backend=backend)
        pil_img = tensor_to_pil(out["render"])
        if render_images:
            save_path = os.path.join(out_dir, f"{cam.image_name}.png")
            pil_img.save(save_path)
        frames.append(np.array(pil_img))

    print(f"Đã render {len(dataset.target_cameras)} ảnh novel view vào {out_dir}")

    if render_video and frames:
        _save_video(frames, os.path.join(out_dir, "novel_views.mp4"), fps)

    if render_images:
        _maybe_zip_submission(out_dir, cfg)


def _save_video(frames, path, fps):
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        print(f"Đã lưu video preview: {path}")
    except ImportError:
        print("Cần cài `imageio`/`imageio-ffmpeg` để xuất video preview, bỏ qua bước này.")


def _maybe_zip_submission(out_dir, cfg):
    """Đóng gói toàn bộ ảnh .png vừa render trong `out_dir` thành 1 file
    submission.zip (nằm cùng cấp với out_dir), tiện nộp bài trực tiếp.
    Bật qua config: inference.output.zip: true (mặc định false, không đổi
    hành vi cũ nếu người dùng chưa thêm key này)."""
    zip_enabled = cfg_get(cfg, "inference.output.zip", False)
    if not zip_enabled:
        return

    png_files = sorted(f for f in os.listdir(out_dir) if f.lower().endswith(".png"))
    if not png_files:
        print("[submission] Không có file .png nào trong output_dir để đóng gói, bỏ qua.")
        return

    zip_name = cfg_get(cfg, "inference.output.zip_name", "submission.zip")
    zip_path = os.path.join(os.path.dirname(out_dir.rstrip(os.sep)) or out_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in png_files:
            zf.write(os.path.join(out_dir, fname), arcname=fname)

    print(f"[submission] Đã đóng gói {len(png_files)} ảnh vào: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
