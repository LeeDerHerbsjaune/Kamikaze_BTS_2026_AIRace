import os
import json
import random
from typing import Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pycolmap

# Giả sử đây là các class cấu trúc dữ liệu hiện tại của bạn trong hệ thống
# Nếu cấu trúc import thực tế của bạn khác, hãy điều chỉnh cho phù hợp.
from dataloader.dataset.entities import Image, Camera, Point3D, Frame, Scene

# ==============================================================================
# 1. MODULE LOADER MỚI (GỘP TỪ COLMAP.PY + COLMAP_UTILS.PY DÙNG PYCOLMAP)
# ==============================================================================

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
    Dataloader chuẩn hóa sử dụng pycolmap để đọc sparse reconstruction.
    Hỗ trợ lọc nhiễu điểm 3D và đọc dữ liệu ảnh trực tiếp thành Tensor.
    """
    def __init__(self, sparse_path: str, images_dir: str):
        if not os.path.isdir(sparse_path):
            raise FileNotFoundError(f"Không tìm thấy thư mục sparse reconstruction: {sparse_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")

        self.sparse_path = sparse_path
        self.images_dir = images_dir
        self.reconstruction = pycolmap.Reconstruction(sparse_path)

        if len(self.reconstruction.cameras) == 0:
            raise ValueError(f"Reconstruction tại {sparse_path} không có camera nào.")
        if len(self.reconstruction.images) == 0:
            raise ValueError(f"Reconstruction tại {sparse_path} không có image nào.")

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

    def _read_image_tensor(self, image_name: str) -> Optional[torch.Tensor]:
        """Đọc ảnh từ đĩa an toàn, hỗ trợ bỏ qua nếu mất file thực tế."""
        image_path = os.path.join(self.images_dir, image_name)
        
        # Xử lý trường hợp sai khác phần mở rộng chữ hoa/chữ thường (Dò tìm an toàn)
        if not os.path.exists(image_path):
            base, ext = os.path.splitext(image_path)
            for alt_ext in [ext.upper(), ext.lower(), '.jpg', '.JPG', '.png', '.PNG']:
                if os.path.exists(base + alt_ext):
                    image_path = base + alt_ext
                    break
                    
        rgb = cv2.imread(image_path)
        if rgb is None:
            return None
            
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        return tensor.permute(2, 0, 1)

    def load_scene(self, min_track_length: int = 3, max_error: Optional[float] = 2.0) -> Scene:
        cameras = self.load_cameras()
        images = self.load_images()
        point_cloud = self.load_points3D(min_track_length, max_error)

        frames = []
        for image in images.values():
            if image.camera_id not in cameras:
                continue
                
            image_tensor = self._read_image_tensor(image.name)
            if image_tensor is None:
                print(f"[Warning] Thiếu tệp ảnh thực tế trên đĩa: {image.name}. Bỏ qua frame.")
                continue

            frames.append(Frame(
                camera=cameras[image.camera_id],
                image=image_tensor,
                image_name=image.name,
                R=image.R, t=image.t
            ))

        return Scene(cameras=cameras, frames=frames, point_cloud=point_cloud)


# ==============================================================================
# 2. HELPER UTILS PHỤC VỤ CHO DATASET (TỰ CHỨA ĐỂ KHÔNG PHỤ THUỘC FILE NGOÀI)
# ==============================================================================

def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Tính toán tọa độ tâm Camera trong không gian thế giới: C = -R^T * t"""
    return -np.dot(R.T, t)


# ==============================================================================
# 3. CLASS BTSDATASET ĐÃ ĐƯỢC TÁI CẤU TRÚC
# ==============================================================================

class BTSDataset(Dataset):
    """
    Dataset Loader quản lý toàn bộ vòng đời nạp dữ liệu trạm BTS.
    Sử dụng ColmapLoader dựa trên pycolmap và thực hiện chuẩn hóa nội bộ.
    """
    def __init__(self, cfg: dict, split: str = "train", device: str = "cuda"):
        self.cfg = cfg
        self.split = split
        self.device = device

        self.data_root = cfg["dataset"]["root"]
        self.images_dir = os.path.join(self.data_root, "images")
        self.sparse_path = os.path.join(self.data_root, "sparse", "0")

        # Khởi tạo Bộ tải dữ liệu hướng đối tượng
        loader = ColmapLoader(sparse_path=self.sparse_path, images_dir=self.images_dir)
        
        # Đọc tham số lọc điểm từ cấu hình config
        min_track = cfg.get("preprocessing", {}).get("colmap", {}).get("min_track_length", 3)
        max_err = cfg.get("preprocessing", {}).get("colmap", {}).get("max_error", 2.0)
        
        # Gọi Loader tạo đối tượng Scene thô
        self.raw_scene = loader.load_scene(min_track_length=min_track, max_error=max_err)

        # Trích xuất dữ liệu thô để chuẩn bị normalization
        self.cameras = self.raw_scene.cameras
        self.frames = self.raw_scene.frames
        self.point_cloud = self.raw_scene.point_cloud

        # Khởi tạo các tham số chuẩn hóa mặc định
        self._norm_center = np.zeros(3, dtype=np.float32)
        self._norm_scale = 1.0

        # Thực hiện chuẩn hóa hệ tọa độ đồng nhất nội bộ (Thay thế hoàn toàn colmap_utils.normalize_scene)
        if cfg.get("dataset", {}).get("normalize", True):
            self._fit_normalization()

        # Chia tập dữ liệu Train / Eval
        self.active_frames = []
        self._split_dataset()

    def _fit_normalization(self):
        """
        Tính toán tâm và tỉ lệ scale dựa trên phân bổ vị trí các Camera,
        sau đó áp dụng đồng nhất lên Point Cloud và các Frame.
        """
        cam_centers = []
        for frame in self.frames:
            center = _camera_center(frame.R, frame.t)
            cam_centers.append(center)

        if len(cam_centers) == 0:
            return

        cam_centers = np.array(cam_centers)
        # Lưu lại thông số để áp dụng tương ứng cho các target poses sau này
        self._norm_center = cam_centers.mean(axis=0)
        dist = np.linalg.norm(cam_centers - self._norm_center, axis=1).max()
        self._norm_scale = 1.0 / (dist + 1e-6)

        print(f"[BTSDataset] Thực hiện chuẩn hóa hệ tọa độ:")
        print(f" -> Tâm Scene (World Center): {self._norm_center}")
        print(f" -> Tỉ lệ Scale không gian: {self._norm_scale:.6f}")

        # 1. Chuẩn hóa Point Cloud
        for p in self.point_cloud.values():
            p.xyz = (p.xyz - self._norm_center) * self._norm_scale

        # 2. Chuẩn hóa Vectơ dịch chuyển (t) của các Frame
        for frame in self.frames:
            C_old = _camera_center(frame.R, frame.t)
            C_new = (C_old - self._norm_center) * self._norm_scale
            frame.t = -np.dot(frame.R, C_new)

    def _split_dataset(self):
        """Chia tập dữ liệu Train/Eval dựa trên tỷ lệ được định nghĩa trong cấu hình."""
        eval_ratio = self.cfg.get("dataset", {}).get("eval_ratio", 0.1)
        seed = self.cfg.get("dataset", {}).get("seed", 42)
        
        # Clone danh sách để xáo trộn không ảnh hưởng cấu trúc gốc
        all_frames = list(self.frames)
        random.seed(seed)
        random.shuffle(all_frames)

        n_eval = int(len(all_frames) * eval_ratio)
        
        if self.split == "train":
            self.active_frames = all_frames[n_eval:]
            print(f"[BTSDataset] Tạo tập TRAIN với {len(self.active_frames)} frames.")
        elif self.split == "eval":
            self.active_frames = all_frames[:n_eval]
            print(f"[BTSDataset] Tạo tập EVAL với {len(self.active_frames)} frames.")
        else:
            # Nếu chạy inference/test, lấy toàn bộ frame
            self.active_frames = all_frames
            print(f"[BTSDataset] Chế độ {self.split.upper()}: Nạp toàn bộ {len(self.active_frames)} frames.")

    def load_target_views(self) -> list[Camera]:
        """
        Đọc các góc chụp ảo (novel views) cần render từ file cấu hình JSON.
        Áp dụng chung một tham số chuẩn hóa tọa độ để đảm bảo tính đồng nhất hệ quy chiếu.
        """
        target_file = self.cfg.get("dataset", {}).get("target_views", {}).get("file", "target_poses.json")
        target_path = os.path.join(self.data_root, target_file)
        if not os.path.exists(target_path):
            print(f"[BTSDataset] Không tìm thấy file novel view pose tại: {target_path}")
            return []
            
        with open(target_path, encoding="utf-8") as f:
            targets = json.load(f)

        target_cameras = []
        for i, t in enumerate(targets):
            R = np.array(t["R"])
            T = np.array(t["T"])
            
            # Áp dụng chuẩn hóa đồng nhất hệ tọa độ nếu có thay đổi
            if self._norm_scale != 1.0 or np.any(self._norm_center != 0):
                C = _camera_center(R, T)
                C_norm = (C - self._norm_center) * self._norm_scale
                T = -np.dot(R, C_norm)
                
            # Đóng gói thành thực thể Camera giả lập phục vụ render
            cam = Camera(
                id=f"target_{i}", R=R, t=T,
                width=t["width"], height=t["height"],
                fx=t.get("fx", 0.0), fy=t.get("fy", 0.0),
                cx=t.get("cx", t["width"]/2.0), cy=t.get("cy", t["height"]/2.0)
            )
            target_cameras.append(cam)
            
        print(f"[BTSDataset] Đã nạp thành công {len(target_cameras)} góc novel-view ảo phục vụ Inference.")
        return target_cameras

    def __len__(self) -> int:
        return len(self.active_frames)

    def __getitem__(self, idx: int) -> dict:
        frame = self.active_frames[idx]
        
        # Chuyển đổi tensor dữ liệu sang thiết bị phần cứng đích (CPU/GPU)
        image = frame.image.to(self.device)
        R = torch.from_numpy(frame.R).float().to(self.device)
        t = torch.from_numpy(frame.t).float().to(self.device)
        
        return {
            "image": image,
            "R": R,
            "t": t,
            "image_name": frame.image_name,
            "camera_id": frame.camera.id
        }