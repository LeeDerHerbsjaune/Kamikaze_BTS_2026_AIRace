"""
Camera utilities: chuyển đổi pose (R, T) và intrinsics (fx, fy, cx, cy)
sang các ma trận view/projection dùng cho renderer, và class Camera đại diện
cho 1 view (train/eval/novel).
"""
import numpy as np
import torch


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def getWorld2View(R, t):
    """R: (3,3) world->cam rotation, t: (3,) world->cam translation."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class Camera:
    """Đại diện cho 1 camera view (ảnh + pose + intrinsics)."""

    def __init__(self, uid, R, T, FoVx, FoVy, image, image_name,
                 width=None, height=None, znear=0.01, zfar=100.0, device="cuda"):
        self.uid = uid
        self.R = R              # (3,3) world->cam
        self.T = T              # (3,)  world->cam
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image = image      # (3,H,W) tensor float [0,1] hoặc None cho novel view
        self.width = width if width is not None else (image.shape[2] if image is not None else None)
        self.height = height if height is not None else (image.shape[1] if image is not None else None)
        if self.width is None or self.height is None:
            raise ValueError(
                f"Camera '{image_name}': cần width/height tường minh khi image=None "
                f"(novel view không có ground truth để suy ra kích thước ảnh).")
        self.znear = znear
        self.zfar = zfar
        self.device = device

        self.world_view_transform = torch.tensor(
            getWorld2View(R, T), dtype=torch.float32).transpose(0, 1).to(device)
        self.projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def cameras_extent(cam_list):
    """Ước lượng bán kính scene (dùng làm spatial_lr_scale và chuẩn hoá densify)."""
    centers = np.stack([c.camera_center.cpu().numpy() for c in cam_list])
    center = centers.mean(axis=0)
    dist = np.linalg.norm(centers - center, axis=1)
    return float(dist.max()) * 1.1
