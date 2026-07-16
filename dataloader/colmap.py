import os
from typing import Optional

import numpy as np
import cv2
import torch

import pycolmap
from dataloader.entities import Image, Camera, Point3D, Frame, Scene

_CAMERA_PARAM_LAYOUT = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
}


class ColmapLoader:
    """Load sparse reconstruction (cameras, images, points3D) từ COLMAP qua pycolmap,
    và ghép lại thành 1 Scene hoàn chỉnh sẵn sàng đưa cho Trainer."""

    def __init__(self, sparse_path: str, images_dir: str):
        if not os.path.isdir(sparse_path):
            raise FileNotFoundError(f"Không tìm thấy thư mục sparse reconstruction: {sparse_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

        self.images_dir = images_dir
        self.reconstruction = pycolmap.Reconstruction(sparse_path) # hàm đọc các tệp .bin hoặc .txt 

        if len(self.reconstruction.cameras) == 0:
            raise ValueError(f"Reconstruction tại {sparse_path} không có camera nào.")
        if len(self.reconstruction.images) == 0:
            raise ValueError(f"Reconstruction tại {sparse_path} không có image nào.")

    # ------------------------------------------------------------------
    def load_cameras(self) -> dict[int, Camera]:
        cameras = {}
        for camera_id, camera in self.reconstruction.cameras.items():
            model_name = camera.model.name if hasattr(camera.model, "name") else str(camera.model)
            fx, fy, cx, cy = self._extract_intrinsics(camera, model_name)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name,
                width=camera.width, height=camera.height,
                fx=fx, fy=fy, cx=cx, cy=cy,
            )
        return cameras

    @staticmethod
    def _extract_intrinsics(camera, model_name: str) -> tuple[float, float, float, float]:
        params = camera.params
        layout = _CAMERA_PARAM_LAYOUT.get(model_name)
        if layout is None:
            raise NotImplementedError(
                f"Camera model '{model_name}' chưa được hỗ trợ. Bổ sung vào _CAMERA_PARAM_LAYOUT.")
        if "fx" in layout and "fy" in layout:
            fx, fy = params[layout.index("fx")], params[layout.index("fy")]
        else:
            fx = fy = params[layout.index("f")]
        cx = params[layout.index("cx")]
        cy = params[layout.index("cy")]
        return float(fx), float(fy), float(cx), float(cy)

    # ------------------------------------------------------------------
    def load_images(self) -> dict[int, Image]:
        """Chỉ đọc metadata + pose, KHÔNG đụng tới file ảnh trên đĩa."""
        images = {}
        for image_id, colmap_image in self.reconstruction.images.items():
            R, t = self._extract_pose(colmap_image)
            images[image_id] = Image(
                id=image_id,
                name=colmap_image.name,
                camera_id=colmap_image.camera_id,
                R=R, t=t,
            )
        return images

    @staticmethod
    def _extract_pose(colmap_image):
        """Tương thích cả API pycolmap cũ và mới, chỉ trả về (R, t)."""
        if hasattr(colmap_image, "cam_from_world"):
            pose = colmap_image.cam_from_world
            return pose.rotation.matrix(), pose.translation
        if hasattr(colmap_image, "R") and hasattr(colmap_image, "t"):
            return colmap_image.R, colmap_image.t
        raise AttributeError(
            "Không tìm thấy pose (cam_from_world hoặc R/t) - kiểm tra version pycolmap.")

    # ------------------------------------------------------------------
    def load_points3D(self, min_track_length: int = 3,
                       max_error: Optional[float] = 2.0) -> dict[int, Point3D]:
        points3D = {}
        n_skipped = 0
        for point3D_id, point3D in self.reconstruction.points3D.items():
            track_length = len(point3D.track.elements) if hasattr(point3D, "track") else None
            if min_track_length and track_length is not None and track_length < min_track_length:
                n_skipped += 1
                continue
            if max_error is not None and getattr(point3D, "error", 0.0) > max_error:
                n_skipped += 1
                continue
            points3D[point3D_id] = Point3D(id=point3D_id, xyz=point3D.xyz, color=point3D.color)

        if n_skipped:
            print(f"[ColmapLoader] Đã lọc bỏ {n_skipped}/{len(self.reconstruction.points3D)} điểm nhiễu.")
        return points3D

    # ------------------------------------------------------------------
    def _read_image_tensor(self, image_name: str) -> torch.Tensor:
        """Đọc 1 file ảnh từ đĩa -> tensor (3,H,W) float [0,1]. Tách riêng để
        BTSDataset có thể override (vd lazy-load / cache) mà không đụng loader."""
        image_path = os.path.join(self.images_dir, image_name)
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        return tensor.permute(2, 0, 1)

    def load_scene(self, min_track_length: int = 3, max_error: Optional[float] = 2.0) -> Scene:
        """Bước ghép cuối cùng: Camera + Image (pose) + pixel thật trên đĩa -> Frame,
        rồi gói tất cả (cameras, frames, point_cloud) thành 1 Scene trả về Trainer.

        Đây là entry point chính nên gọi từ bên ngoài, thay vì gọi rời từng
        load_cameras()/load_images()/load_points3D() rồi tự ghép tay.
        """
        cameras = self.load_cameras()
        images = self.load_images()
        point_cloud = self.load_points3D(min_track_length=min_track_length, max_error=max_error)

        frames = []
        n_missing = 0
        for image in images.values():
            camera = cameras.get(image.camera_id)
            if camera is None:
                n_missing += 1
                continue
            image_tensor = self._read_image_tensor(image.name)
            frames.append(Frame(
                camera=camera,
                image=image_tensor,
                image_name=image.name,
                R=image.R,
                t=image.t,
            ))

        if n_missing:
            print(f"[ColmapLoader] Bỏ qua {n_missing} ảnh do không tìm thấy camera_id tương ứng.")

        return Scene(cameras=cameras, frames=frames, point_cloud=point_cloud)