"""
Visualization: hỗ trợ kiểm tra định tính chất lượng reconstruction -
xuất point cloud Gaussian ra .ply để mở bằng viewer (viser, SuperSplat,
CloudCompare), và vẽ quỹ đạo camera (train/eval/target) để kiểm tra coverage
góc nhìn quanh trạm BTS.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_camera_trajectory(train_cams, eval_cams=None, target_cams=None, save_path=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def _plot(cams, color, label):
        if not cams:
            return
        centers = np.stack([c.camera_center.cpu().numpy() for c in cams])
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=color, label=label, s=15)

    _plot(train_cams, "tab:blue", "Train views")
    _plot(eval_cams, "tab:orange", "Eval views")
    _plot(target_cams, "tab:red", "Target novel views")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Phân bố góc nhìn camera quanh trạm BTS")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ quỹ đạo camera: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def side_by_side_comparison(rendered, gt, save_path):
    """So sánh trực quan ảnh render vs ground truth (dùng cho tập eval)."""
    import torch
    r = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    g = gt.clamp(0, 1).permute(1, 2, 0).cpu().numpy() if isinstance(gt, torch.Tensor) else gt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(g); axes[0].set_title("Ground Truth"); axes[0].axis("off")
    axes[1].imshow(r); axes[1].set_title("Rendered"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
