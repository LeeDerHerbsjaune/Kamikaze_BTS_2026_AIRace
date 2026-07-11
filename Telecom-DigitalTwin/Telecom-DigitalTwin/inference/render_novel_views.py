"""
Inference: load checkpoint đã train, render 20-50 novel view mục tiêu do đề bài
cung cấp (không có ground truth) và lưu ra ảnh PNG (+ tuỳ chọn dựng video).
Đây là bước cuối trong workflow: Inference -> Novel View Images.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.gaussian_model import GaussianModel
from datasets.bts_dataset import BTSDataset
from renderer.gaussian_renderer import render
from utils.config_loader import load_config


def tensor_to_pil(img_tensor):
    arr = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main(cfg_path):
    cfg = load_config(cfg_path)

    device = cfg["device"]
    gaussians = GaussianModel(cfg["model"]["sh_degree"], device=device)

    ckpt = torch.load(cfg["inference"]["checkpoint"], map_location=device)
    gaussians.restore(ckpt["gaussians"])
    print(f"Đã load checkpoint tại iteration {ckpt['iteration']}, "
          f"số Gaussian: {gaussians.xyz.shape[0]}")

    dataset = BTSDataset(cfg, device=device)
    bg_color = torch.tensor(cfg["renderer"]["bg_color"], dtype=torch.float32, device=device)

    out_dir = cfg["inference"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for cam in tqdm(dataset.target_cameras, desc="Rendering novel views"):
        with torch.no_grad():
            out = render(cam, gaussians, bg_color, backend=cfg["renderer"]["backend"])
        pil_img = tensor_to_pil(out["render"])
        save_path = os.path.join(out_dir, f"{cam.image_name}.png")
        pil_img.save(save_path)
        frames.append(np.array(pil_img))

    print(f"Đã render {len(dataset.target_cameras)} ảnh novel view vào {out_dir}")

    if cfg["inference"]["render_video"] and frames:
        _save_video(frames, os.path.join(out_dir, "novel_views.mp4"), cfg["inference"]["fps"])


def _save_video(frames, path, fps):
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        print(f"Đã lưu video preview: {path}")
    except ImportError:
        print("Cần cài `imageio`/`imageio-ffmpeg` để xuất video preview, bỏ qua bước này.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
