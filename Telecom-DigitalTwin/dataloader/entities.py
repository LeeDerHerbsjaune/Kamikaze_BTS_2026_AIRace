"""
Shared entities for the BTS scene data-loading pipeline (dataloader/colmap.py
+ dataloader/bts_dataset.py).

Layers:
  Camera   - pure intrinsics (one COLMAP Camera can be shared by many Images).
  Image    - metadata + RAW pose (world->cam) read straight from COLMAP, NOT
             yet normalized, with NO pixel data.
  Point3D  - one point in the sparse point cloud.
  Frame    - one COMPLETE camera view: pose + intrinsics (via `camera`) +
             the actual image (`image=None` allows reusing Frame for a
             target/novel view with no ground truth, even though ColmapLoader
             currently always fills in a real image).
  Scene    - bundles cameras + frames + point_cloud for one scene, returned
             by ColmapLoader.load_scene().

FoVx/FoVy are deliberately NOT included here: FoV is a renderer-specific
concept (computed from fx/fy + width/height via
utils.camera_utils.focal2fov), not an intrinsic property of COLMAP
intrinsics - computing FoV is the consumer's responsibility
(dataloader/bts_dataset.py), keeping entities.py independent of any
training convention.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import torch


@dataclass
class Camera:
    """Intrinsics of one COLMAP camera."""
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
    """Metadata + raw pose (world->cam) of one image, read straight from COLMAP."""
    id: int
    name: str
    camera_id: int
    R: np.ndarray    # (3,3) world->cam rotation
    t: np.ndarray    # (3,)  world->cam translation

    def camera_center(self) -> np.ndarray:
        """Camera position in world coords: C = -R^T @ t."""
        return -self.R.T @ self.t


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray     # (3,)
    color: np.ndarray   # (3,) uint8


@dataclass
class Frame:
    """One complete camera view: pose (R, t) + camera intrinsics + the real
    image. `image=None` is allowed so Frame can be reused for a target/novel
    view (no ground truth), even though the current ColmapLoader flow always
    fills in a real image."""
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
