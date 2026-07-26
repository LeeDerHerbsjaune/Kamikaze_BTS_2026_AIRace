"""
Inference: loads a trained checkpoint, renders the 20-50 target novel views
provided by the challenge (no ground truth) and saves them as PNGs (+
optionally builds a preview video). This is the last step of the workflow:
Inference -> Novel View Images.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.gaussian_model import GaussianModel
from dataloader.bts_dataset import BTSDataset
from renderer.gaussian_renderer import render
from utils.config_loader import load_config, cfg_get
from utils.logger import Logger


def tensor_to_pil(img_tensor):
    arr = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main(cfg_path):
    cfg = load_config(cfg_path)

    # Uses the same Logger/notes.md as train.py - placed in the output
    # directory (not checkpoint_dir) so results are recorded even when
    # render.sh runs INDEPENDENTLY of train.py (e.g. re-rendering from an
    # older checkpoint without retraining from scratch).
    out_dir = cfg_get(cfg, "inference.output_dir") or cfg_get(
        cfg, "inference.output.directory", "./outputs/novel_views")
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(log_dir=out_dir, use_tensorboard=False)

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    gaussians = GaussianModel(cfg_get(cfg, "model.sh_degree", 3), device=device)

    checkpoint_path = cfg_get(cfg, "inference.checkpoint")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}' (inference.checkpoint). "
            f"Run training first, or fix the path in the config.")

    ckpt = torch.load(checkpoint_path, map_location=device)
    gaussians.restore(ckpt["gaussians"])
    logger.log_text(f"Loaded checkpoint '{checkpoint_path}' at iteration {ckpt['iteration']}, "
                     f"{gaussians.xyz.shape[0]} Gaussians")

    dataset = BTSDataset(cfg, device=device)
    bg_color = torch.tensor(cfg_get(cfg, "renderer.background.color", [0, 0, 0]),
                            dtype=torch.float32, device=device)

    if not dataset.target_cameras:
        logger.log_text("Warning: dataset.target_views.file not found or empty - "
                        "no novel views to render.")
        logger.log_summary("Render summary", ["SKIPPED - no target novel views."])
        logger.close()
        return

    render_images = cfg_get(cfg, "inference.render.images", True)
    render_video = cfg_get(cfg, "inference.render.video.enabled", True)
    fps = cfg_get(cfg, "inference.render.video.fps", 24)
    backend = cfg_get(cfg, "renderer.backend", "gsplat")

    frames = []
    alpha_stats = []
    for cam in tqdm(dataset.target_cameras, desc="Rendering novel views"):
        with torch.no_grad():
            out = render(cam, gaussians, bg_color, backend=backend)
        pil_img = tensor_to_pil(out["render"])
        if render_images:
            save_path = os.path.join(out_dir, f"{cam.image_name}.png")
            pil_img.save(save_path)
        frames.append(np.array(pil_img))
        if out.get("alpha") is not None:
            alpha_stats.append(out["alpha"].mean().item())

    logger.log_text(f"Rendered {len(dataset.target_cameras)} novel view images to {out_dir}")
    logger.log_artifact("novel_view_images", out_dir, iteration=ckpt["iteration"],
                        note=f"{len(dataset.target_cameras)} images")

    if alpha_stats:
        mean_alpha = sum(alpha_stats) / len(alpha_stats)
        min_alpha = min(alpha_stats)
        # Mean alpha (accumulated opacity per pixel) close to 0 means most
        # rays never hit enough Gaussian coverage and are showing
        # background color (black by default) instead of scene content -
        # the render looks dark/underexposed NOT because colors are wrong,
        # but because there's almost nothing opaque there to show a color
        # for. This is a completely different root cause (and fix) than
        # e.g. insufficient training or an SH color bug, both of which
        # would still show reasonably high alpha.
        logger.log_text(f"Mean alpha (opacity coverage) across novel views: {mean_alpha:.4f} "
                        f"(min: {min_alpha:.4f}). Low values (<0.3) strongly suggest these "
                        f"cameras are pointed at regions with sparse/no reconstructed "
                        f"geometry, or are positioned much farther from the scene than "
                        f"train cameras - not an undertrained-model or color issue.")
        if mean_alpha < 0.3:
            logger.log_text("WARNING: low alpha coverage detected - this is very likely why "
                            "novel-view renders look mostly dark/black. Check "
                            "dataset.target_views.pose_convention, verify normalize_scene is "
                            "applied consistently, and confirm these target camera positions "
                            "actually overlap the photographed area (not just plausible in "
                            "isolation).")

    if render_video and frames:
        video_path = os.path.join(out_dir, "novel_views.mp4")
        _save_video(frames, video_path, fps)
        logger.log_artifact("novel_view_video", video_path, iteration=ckpt["iteration"],
                            note=f"{fps} fps, {len(frames)} frames")

    logger.log_summary("Render summary", [
        f"Checkpoint: {checkpoint_path} (iteration {ckpt['iteration']})",
        f"Novel views rendered: {len(dataset.target_cameras)}",
        f"Output directory: {out_dir}",
    ])
    logger.close()


def _save_video(frames, path, fps):
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        print(f"Saved preview video: {path}")
    except ImportError:
        print("Need `imageio`/`imageio-ffmpeg` installed to export a preview video, skipping this step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
