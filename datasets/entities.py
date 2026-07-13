from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class Camera:
    """syntax: camera = <cameras>(id = image.camera_id, model = image.camera.model, width = image.camera.width,
      height = image.camera.height, fx=..., fy=..., cx=..., cy=...)"""
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

    id: int
    name: str
    camera_id: int
    #cam_from_world: np.ndarray # a matrix of shape (4,4) representing the camera pose in global coordinates
    R: np.ndarray # a matrix 3x3 represeting the rotation f the camera in global coordinates
    t: np.ndarray # a vector 3x1 representing the translation of the camera in global coordiantes
    @property
    def cam_from_world(self) -> np.ndarray:
        """infer the camera pose matrix from R and t, compatible with both old and new pycolmap API."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.R
        mat[:3, 3] = self.t
        return mat

@dataclass
class Point3D:
    id: int
    xyz: np.ndarray # a vector 3x1 representing the 3d coordinates of the point in global cooordiantes
    color: np.ndarray # a vector 3x1 representing the color of the point in RGB format

@dataclass
class Frame:
    camera: Camera
    image: torch.Tensor
    image_name: str
    R: np.ndarray # a matrix 3x3 representing the rotation of the camera in global coordinates
    t: np.ndarray # a vector 3x1 representing the translation of the camera in global coordinates

    @property
    def cam_from_world(self) -> np.ndarray:
        """infer the camera pose matrix from R and t, compatible with both old and new pycolmap API."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.R
        mat[:3, 3] = self.t
        return mat

@dataclass
class Scene:
    cameras: dict[int, Camera] = field(default_factory=dict)

    frames: list[Frame] = field(default_factory=list)

    point_cloud: dict[int, Point3D] = field(default_factory=dict)

    def get_camera(self, camera_id: int) -> Camera:
        return self.cameras[camera_id]

    def num_cameras(self) -> int:
        return len(self.cameras)

    def num_frames(self) -> int:
        return len(self.frames)

    def __len__(self):
        return len(self.frames)