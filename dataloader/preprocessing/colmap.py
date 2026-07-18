import os
import shutil
import subprocess
from typing import Optional

import numpy as np
import cv2
import torch
import pycolmap

# Giả sử bạn đã định nghĩa các entity này trong dự án
from dataloader.dataset.entities import Image, Camera, Point3D, Frame, Scene

_CAMERA_PARAM_LAYOUT = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")
}

class ColmapLoader:
    """
    Dataloader kết hợp: Tự động chạy SfM nếu cần, sau đó load sparse reconstruction 
    bằng pycolmap và đóng gói thành Scene hoàn chỉnh.
    """

    def __init__(self, workspace_dir: str, images_dir: str, 
                 auto_run_colmap: bool = False,
                 colmap_exe: str = "colmap", 
                 camera_model: str = "PINHOLE"):
        
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

        self.workspace_dir = workspace_dir
        self.images_dir = images_dir
        self.sparse_path = os.path.join(workspace_dir, "sparse", "0")

        # 1. Kiểm tra và tự động chạy COLMAP SfM nếu chưa có dữ liệu
        if not os.path.exists(self.sparse_path):
            if auto_run_colmap:
                print("[ColmapLoader] Chưa có sparse reconstruction, bắt đầu tự động chạy COLMAP SfM...")
                self._run_colmap_pipeline(colmap_exe, camera_model)
            else:
                raise FileNotFoundError(
                    f"Không tìm thấy thư mục sparse tại {self.sparse_path}. "
                    f"Vui lòng set auto_run_colmap=True hoặc chạy thủ công."
                )

        # 2. Sử dụng pycolmap để load dữ liệu
        self.reconstruction = pycolmap.Reconstruction(self.sparse_path)

        if len(self.reconstruction.cameras) == 0:
            raise ValueError(f"Reconstruction tại {self.sparse_path} không có camera nào.")
        if len(self.reconstruction.images) == 0:
            raise ValueError(f"Reconstruction tại {self.sparse_path} không có image nào.")

    def _run_colmap_pipeline(self, colmap_exe: str, camera_model: str):
        """Chạy pipeline SfM đầy đủ qua subprocess (tích hợp từ colmap_utils.py)."""
        if shutil.which(colmap_exe) is None and not os.path.isfile(colmap_exe):
            raise FileNotFoundError(f"Không tìm thấy executable COLMAP: '{colmap_exe}'. Hãy thêm vào PATH.")

        os.makedirs(self.workspace_dir, exist_ok=True)
        db_path = os.path.join(self.workspace_dir, "database.db")
        sparse_dir = os.path.join(self.workspace_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)

        print("--> Chạy feature_extractor...")
        subprocess.run([
            colmap_exe, "feature_extractor", "--database_path", db_path,
            "--image_path", self.images_dir, "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1"
        ], check=True)

        print("--> Chạy exhaustive_matcher...")
        subprocess.run([colmap_exe, "exhaustive_matcher", "--database_path", db_path], check=True)

        print("--> Chạy mapper...")
        subprocess.run([
            colmap_exe, "mapper", "--database_path", db_path,
            "--image_path", self.images_dir, "--output_path", sparse_dir
        ], check=True)
        print(f"[ColmapLoader] Hoàn thành SfM. Dữ liệu lưu tại: {self.sparse_path}")

    def load_cameras(self) -> dict[int, Camera]:
        cameras = {}
        for camera_id, camera in self.reconstruction.cameras.items():
            model_name = camera.model.name if hasattr(camera.model, "name") else str(camera.model)
            
            params = camera.params
            layout = _CAMERA_PARAM_LAYOUT.get(model_name)
            if layout is None:
                raise NotImplementedError(f"Model '{model_name}' chưa được hỗ trợ.")
            
            fx = float(params[layout.index("fx")]) if "fx" in layout else float(params[layout.index("f")])
            fy = float(params[layout.index("fy")]) if "fy" in layout else float(params[layout.index("f")])
            cx, cy = float(params[layout.index("cx")]), float(params[layout.index("cy")])

            cameras[camera_id] = Camera(
                id=camera_id, model=model_name,
                width=camera.width, height=camera.height,
                fx=fx, fy=fy, cx=cx, cy=cy,
            )
        return cameras

    def load_images(self) -> dict[int, Image]:
        images = {}
        for image_id, colmap_image in self.reconstruction.images.items():
            if hasattr(colmap_image, "cam_from_world"):
                pose = colmap_image.cam_from_world()
                R, t = pose.rotation.matrix(), pose.translation
            else:
                R, t = colmap_image.R, colmap_image.t

            images[image_id] = Image(
                id=image_id, name=colmap_image.name,
                camera_id=colmap_image.camera_id, R=R, t=t
            )
        return images

    def load_points3D(self, min_track_length: int = 3, max_error: Optional[float] = 2.0) -> dict[int, Point3D]:
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
        
        if n_skipped > 0:
            print(f"[ColmapLoader] Đã lọc bỏ {n_skipped} điểm nhiễu.")
        return points3D

    def _read_image_tensor(self, image_name: str) -> torch.Tensor:
        image_path = os.path.join(self.images_dir, image_name)
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        return tensor.permute(2, 0, 1)

    def load_scene(self, min_track_length: int = 3, max_error: Optional[float] = 2.0, normalize: bool = True) -> Scene:
        cameras = self.load_cameras()
        images = self.load_images()
        point_cloud = self.load_points3D(min_track_length, max_error)

        frames = []
        cam_centers = []

        # Ghép Frame
        for image in images.values():
            if image.camera_id not in cameras:
                continue
            # Tính camera center: C = -R^T * t
            center = -np.dot(image.R.T, image.t)
            cam_centers.append(center)

            frames.append(Frame(
                camera=cameras[image.camera_id],
                image=self._read_image_tensor(image.name),
                image_name=image.name,
                R=image.R, t=image.t
            ))
        
        # Chuẩn hóa Scene (Tích hợp từ colmap_utils.py)
        if normalize and len(cam_centers) > 0:
            cam_centers = np.array(cam_centers)
            scene_center = cam_centers.mean(axis=0)
            dist = np.linalg.norm(cam_centers - scene_center, axis=1).max()
            scale = 1.0 / (dist + 1e-6)

            # Cập nhật lại Point Cloud
            for p in point_cloud.values():
                p.xyz = (p.xyz - scene_center) * scale
            
            # Cập nhật lại Translation (t) cho các camera
            for frame in frames:
                # C_new = (C_old - scene_center) * scale
                # t_new = -R * C_new
                C_old = -np.dot(frame.R.T, frame.t)
                C_new = (C_old - scene_center) * scale
                frame.t = -np.dot(frame.R, C_new)
            
            print(f"[ColmapLoader] Đã chuẩn hóa Scene: scale={scale:.4f}, center={scene_center}")

        return Scene(cameras=cameras, frames=frames, point_cloud=point_cloud)