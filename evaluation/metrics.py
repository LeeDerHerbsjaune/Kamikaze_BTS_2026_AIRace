"""
Evaluation: đo chất lượng ảnh render trên tập validation (có ground truth,
lấy từ 100-300 ảnh gốc, tách ra 1 phần làm eval) bằng PSNR, SSIM, LPIPS.
Dùng để theo dõi hội tụ trong lúc train, KHÔNG dùng cho 20-50 novel views
mục tiêu của đề bài vì các view đó không có ground truth.
"""
import torch
from losses.loss import ssim as ssim_fn
from utils.general_utils import psnr as psnr_fn

_lpips_model = None


def get_lpips_model(device="cuda"):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    return _lpips_model


@torch.no_grad()
def evaluate_view(rendered: torch.Tensor, gt: torch.Tensor, use_lpips=True):
    """rendered, gt: (3,H,W) trong [0,1]."""
    p = psnr_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).mean().item()
    s = ssim_fn(rendered, gt).item()
    result = {"psnr": p, "ssim": s}
    if use_lpips:
        model = get_lpips_model(rendered.device)
        l = model(rendered.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item()
        result["lpips"] = l
    return result


@torch.no_grad()
def evaluate_dataset(cameras, gaussians, render_fn, bg_color, use_lpips=True):
    """Chạy evaluate trên toàn bộ tập eval_cameras, trả về giá trị trung bình.
    Trả về {} nếu không có camera eval nào (vd split ratio quá nhỏ)."""
    if not cameras:
        return {}
    totals = {}
    n = len(cameras)
    for cam in cameras:
        out = render_fn(cam, gaussians, bg_color)
        metrics = evaluate_view(out["render"], cam.image.to(out["render"].device), use_lpips)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
    return {k: v / n for k, v in totals.items()}
