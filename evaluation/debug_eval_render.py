"""
Debug script: render 1 eval camera (CÓ ảnh ground truth thật, cùng hệ toạ độ
lúc train - KHÔNG đi qua test_poses.csv) và lưu so sánh cạnh nhau với ảnh gốc.

Mục đích: tách bạch 2 khả năng gây ra kết quả render đen/lấm tấm:
  - Nếu ảnh render ra RÕ, gần giống ảnh gốc  -> model đã train tốt, bug nằm
    ở việc đọc/convert pose trong test_poses.csv (dataloader/bts_dataset.py
    _load_target_views_csv) hoặc ở chính file CSV do đề bài cấp.
  - Nếu ảnh render ra CŨNG đen/lấm tấm như target views -> bug nằm ở
    training/checkpoint (chưa hội tụ, point cloud init sai, checkpoint load
    nhầm...), không liên quan gì tới CSV.

Chạy:
    python scripts/debug_eval_render.py --config configs/default.yaml --idx 0
"""
import argparse
import os
import torch

from models.gaussian_model import GaussianModel
from dataloader.bts_dataset import BTSDataset
from renderer.gaussian_renderer import render
from utils.config_loader import load_config, cfg_get
from utils.general_utils import psnr as psnr_fn
from visualization.viz_utils import side_by_side_comparison


def main(cfg_path, idx=0, out_path="debug_eval_compare.png"):
    cfg = load_config(cfg_path)
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    gaussians = GaussianModel(cfg_get(cfg, "model.sh_degree", 3), device=device)

    checkpoint_path = cfg_get(cfg, "inference.checkpoint") or os.path.join(
        cfg_get(cfg, "training.checkpoint_dir",
                cfg_get(cfg, "training.checkpoint.directory", "./outputs/bts_scene_01/checkpoints")),
        "last.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại '{checkpoint_path}'. Sửa lại "
            f"'inference.checkpoint' trong config hoặc kiểm tra checkpoint_dir.")

    ckpt = torch.load(checkpoint_path, map_location=device)
    gaussians.restore(ckpt["gaussians"])
    print(f"[debug] Checkpoint @ iteration {ckpt['iteration']}, "
          f"n_gaussians = {gaussians.xyz.shape[0]}")
    if gaussians.xyz.shape[0] < 1000:
        print("[debug] CẢNH BÁO: số Gaussian quá ít (<1000) - dấu hiệu training "
              "thất bại/chưa hội tụ hoặc densify không chạy (kiểm tra "
              "densify.enabled và log 'param-group xyz bị tắt' nếu có).")

    dataset = BTSDataset(cfg, device=device)
    print(f"[debug] Train views: {len(dataset.train_cameras)}, "
          f"Eval views: {len(dataset.eval_cameras)}, "
          f"Target(test) views: {len(dataset.target_cameras)}")

    if not dataset.eval_cameras:
        print("[debug] Không có eval camera nào (dataset.split.ratio quá nhỏ?) "
              "- không so sánh được, thử tăng dataset.split.ratio và train lại, "
              "hoặc so sánh trực tiếp 1 train_camera thay thế.")
        return

    idx = min(idx, len(dataset.eval_cameras) - 1)
    cam = dataset.eval_cameras[idx]

    bg_color = torch.tensor(cfg_get(cfg, "renderer.background.color", [0, 0, 0]),
                            dtype=torch.float32, device=device)
    backend = cfg_get(cfg, "renderer.backend", "gsplat")

    with torch.no_grad():
        out = render(cam, gaussians, bg_color, backend=backend)

    rendered = out["render"]
    gt = cam.image.to(rendered.device)
    p = psnr_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).mean().item()
    print(f"[debug] Eval camera '{cam.image_name}' (idx={idx}) - PSNR = {p:.2f} dB")
    if p < 15:
        print("[debug] PSNR < 15dB rất thấp -> model gần như chưa học được gì "
              "trên chính tập train/eval (không liên quan tới CSV). Kiểm tra "
              "lại quá trình training (log loss, n_gaussians theo thời gian, "
              "point cloud khởi tạo).")
    else:
        print("[debug] PSNR có vẻ hợp lý -> nếu target views (từ test_poses.csv) "
              "vẫn đen/lấm tấm, bug nhiều khả năng nằm ở convert pose CSV, không "
              "phải ở training.")

    side_by_side_comparison(rendered, gt, out_path)
    print(f"[debug] Đã lưu ảnh so sánh (Ground Truth | Rendered): {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--idx", type=int, default=0, help="Chỉ số eval camera muốn kiểm tra")
    parser.add_argument("--out", type=str, default="debug_eval_compare.png")
    args = parser.parse_args()
    main(args.config, args.idx, args.out)
