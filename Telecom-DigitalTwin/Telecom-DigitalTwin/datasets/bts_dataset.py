"""
Dataset loader cho scene trạm BTS: đọc ảnh drone RGB (100-300 ảnh/scene) + pose
(dạng COLMAP hoặc transforms.json kiểu NeRF), chia train/eval, và cung cấp
danh sách target novel-view poses (20-50 pose) cần render cho vòng inference.
"""
import os
import json
import numpy as np
from PIL import Image as PILImage
import torch

from dataloader.preprocessing.colmap_utils import (
    load_colmap_scene, get_scene_pointcloud, qvec2rotmat, run_colmap_pipeline
)
from utils.camera_utils import Camera, focal2fov


class BTSDataset:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.data_root = cfg["dataset"]["data_root"]
        self.format = cfg["dataset"]["format"]

        self.train_cameras = []
        self.eval_cameras = []
        self.target_cameras = []   # novel views cần sinh (không có ground truth)
        self.point_cloud_xyz = None
        self.point_cloud_rgb = None

        if self.format == "colmap":
            self._load_colmap()
        elif self.format == "nerf_transforms":
            self._load_nerf_transforms()
        else:
            raise ValueError(f"Định dạng dataset không hỗ trợ: {self.format}")

        self._load_target_views()

    # ------------------------------------------------------------------
    def _load_colmap(self):
        sparse_dir = os.path.join(self.data_root, "sparse", "0")
        if self.cfg["preprocessing"]["run_colmap"] or not os.path.exists(sparse_dir):
            image_dir = os.path.join(self.data_root, self.cfg["dataset"]["images_dir"])
            sparse_dir = run_colmap_pipeline(
                image_dir, self.data_root,
                colmap_exe=self.cfg["preprocessing"]["colmap_executable"],
                camera_model=self.cfg["preprocessing"]["camera_model"],
            )

        cameras_meta, images_meta, points3D = load_colmap_scene(sparse_dir)
        self.point_cloud_xyz, self.point_cloud_rgb = get_scene_pointcloud(points3D)

        images_dir = os.path.join(self.data_root, self.cfg["dataset"]["images_dir"])
        all_cams = []
        for img_id, img_meta in sorted(images_meta.items()):
            cam_meta = cameras_meta[img_meta.camera_id]
            R = qvec2rotmat(img_meta.qvec)
            T = img_meta.tvec

            img_path = os.path.join(images_dir, img_meta.name)
            pil_img = PILImage.open(img_path).convert("RGB")
            w, h = pil_img.size

            if cam_meta.model in ("PINHOLE", "OPENCV"):
                fx, fy = cam_meta.params[0], cam_meta.params[1]
            else:  # SIMPLE_PINHOLE / SIMPLE_RADIAL
                fx = fy = cam_meta.params[0]

            FoVx = focal2fov(fx, w)
            FoVy = focal2fov(fy, h)

            image_tensor = torch.from_numpy(
                np.array(pil_img)).permute(2, 0, 1).float() / 255.0

            cam = Camera(uid=img_id, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                         image=image_tensor, image_name=img_meta.name,
                         device=self.device)
            all_cams.append(cam)

        self._split_train_eval(all_cams)

    # ------------------------------------------------------------------
    def _load_nerf_transforms(self):
        """Đọc format transforms.json kiểu NeRF/Instant-NGP (dùng khi đề bài
        cấp sẵn intrinsics/extrinsics thay vì COLMAP thô)."""
        with open(os.path.join(self.data_root, "transforms.json")) as f:
            meta = json.load(f)

        images_dir = self.data_root
        camera_angle_x = meta.get("camera_angle_x")
        all_cams = []
        for i, frame in enumerate(meta["frames"]):
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1  # NeRF -> COLMAP convention
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            img_path = os.path.join(images_dir, frame["file_path"])
            if not img_path.endswith((".png", ".jpg", ".jpeg")):
                img_path += ".png"
            pil_img = PILImage.open(img_path).convert("RGB")
            w, h = pil_img.size

            FoVx = camera_angle_x if camera_angle_x else focal2fov(frame["fl_x"], w)
            FoVy = focal2fov(fov2focal_from_x(FoVx, w), h) if camera_angle_x else focal2fov(frame["fl_y"], h)

            image_tensor = torch.from_numpy(
                np.array(pil_img)).permute(2, 0, 1).float() / 255.0

            cam = Camera(uid=i, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                         image=image_tensor, image_name=os.path.basename(img_path),
                         device=self.device)
            all_cams.append(cam)

        self._split_train_eval(all_cams)
        # Với format này thường không có point cloud SfM sẵn -> khởi tạo random
        self.point_cloud_xyz = None
        self.point_cloud_rgb = None

    # ------------------------------------------------------------------
    def _split_train_eval(self, all_cams):
        ratio = self.cfg["dataset"]["eval_split_ratio"]
        n_eval = max(1, int(len(all_cams) * ratio))
        step = max(1, len(all_cams) // n_eval)
        eval_idx = set(range(0, len(all_cams), step))

        self.train_cameras = [c for i, c in enumerate(all_cams) if i not in eval_idx]
        self.eval_cameras = [c for i, c in enumerate(all_cams) if i in eval_idx]

    # ------------------------------------------------------------------
    def _load_target_views(self):
        """Đọc 20-50 pose mục tiêu (novel views) mà đề bài yêu cầu sinh ảnh.
        File json dạng: [{"name": "target_001", "R":[[..]], "T":[..], "FoVx":.., "FoVy":..,
                          "width":.., "height":..}, ...]
        Không có ảnh ground truth đi kèm (image=None)."""
        target_path = os.path.join(self.data_root, self.cfg["dataset"]["target_views_file"])
        if not os.path.exists(target_path):
            return
        with open(target_path) as f:
            targets = json.load(f)

        for i, t in enumerate(targets):
            R = np.array(t["R"])
            T = np.array(t["T"])
            cam = Camera(uid=f"target_{i}", R=R, T=T,
                         FoVx=t["FoVx"], FoVy=t["FoVy"],
                         image=None, image_name=t.get("name", f"target_{i:03d}"),
                         width=t["width"], height=t["height"],
                         device=self.device)
            self.target_cameras.append(cam)


def fov2focal_from_x(fov_x, w):
    return w / (2 * np.tan(fov_x / 2))
