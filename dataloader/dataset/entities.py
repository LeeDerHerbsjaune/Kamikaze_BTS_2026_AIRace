"""
Entity dùng chung cho pipeline load dữ liệu scene BTS (dataloader/colmap_loader.py
+ dataloader/bts_dataset.py).

Phân tầng:
  Camera   - intrinsics thuần (1 Camera COLMAP có thể được nhiều Image dùng chung).
  Image    - metadata + pose THÔ (world->cam) đọc thẳng từ COLMAP, CHƯA normalize,
             CHƯA có pixel data.
  Point3D  - 1 điểm trong sparse point cloud.
  Frame    - 1 camera view HOÀN CHỈNH: pose + intrinsics (qua `camera`) + ảnh thật
             (`image=None` cho phép dùng lại Frame cho target/novel-view chưa có
             ground-truth, dù ColmapLoader hiện luôn gán ảnh thật).
  Scene    - gói cameras + frames + point_cloud của 1 scene, trả về từ
             ColmapLoader.load_scene().

Cố tình KHÔNG đưa FoVx/FoVy vào đây: FoV là khái niệm riêng của renderer (tính
từ fx/fy + width/height qua utils.camera_utils.focal2fov), không phải thuộc
tính nội tại của intrinsics COLMAP - việc tính FoV thuộc về lớp tiêu thụ
(dataloader/bts_dataset.py), giữ entities.py độc lập với training convention.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import torch


@dataclass
class Camera:
    """Intrinsics của 1 camera COLMAP."""
    id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Image:
    """Metadata + pose thô (world->cam) của 1 ảnh, đọc thẳng từ COLMAP."""
    id: int
    name: str
    camera_id: int
    R: np.ndarray    # (3,3) world->cam rotation
    t: np.ndarray    # (3,)  world->cam translation

    def camera_center(self) -> np.ndarray:
        """Vị trí camera trong world coords: C = -R^T @ t."""
        return -self.R.T @ self.t


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray     # (3,)
    color: np.ndarray   # (3,) uint8


@dataclass
class Frame:
    """1 camera view hoàn chỉnh: pose (R, t) + camera intrinsics + ảnh thật.
    `image=None` được cho phép để tái dùng Frame cho target/novel-view (không có
    ground-truth), dù luồng ColmapLoader hiện tại luôn điền ảnh thật."""
    camera: Camera
    R: np.ndarray
    t: np.ndarray
    image_name: str
    image: Optional[torch.Tensor] = None   # (3, H, W) float [0,1]

    def camera_center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Scene:
    cameras: Dict[int, Camera]
    frames: List[Frame]
    point_cloud: Dict[int, Point3D]