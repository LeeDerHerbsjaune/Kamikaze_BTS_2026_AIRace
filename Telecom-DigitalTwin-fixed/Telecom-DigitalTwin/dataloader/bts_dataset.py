"""
Dataset loader cho scene trạm BTS: đọc ảnh drone RGB (100-300 ảnh/scene) + pose
(dạng COLMAP hoặc transforms.json kiểu NeRF), chia train/eval, và cung cấp
danh sách target novel-view poses (20-50 pose) cần render cho vòng inference.

QUAN TRỌNG về normalize_scene: nếu preprocessing.normalize_scene=true, scene
được đưa về tâm (0,0,0) và scale theo bán kính camera. Phép biến đổi này phải
được áp dụng ĐỒNG NHẤT lên: point cloud khởi tạo, camera train/eval, VÀ
camera target (novel views) - nếu không target novel views sẽ bị render sai vị
trí vì lệch hệ toạ độ so với model đã train. Bản sửa này áp dụng normalize cho
cả 3 nhóm, dựa trên thống kê tính từ train+eval cameras.
"""
import os
import json
import numpy as np
from PIL import Image as PILImage
import torch

from preprocessing.colmap_utils import (
    load_colmap_scene, get_scene_pointcloud, qvec2rotmat, run_colmap_pipeline,
    normalize_scene,
)
from utils.camera_utils import Camera, focal2fov
from utils.config_loader import cfg_get


class BTSDataset:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.data_root = cfg_get(cfg, "dataset.root")
        self.format = cfg_get(cfg, "dataset.type")
        if self.data_root is None:
            raise KeyError("Thiếu key bắt buộc trong config: 'dataset.root'")
        if self.format is None:
            raise KeyError("Thiếu key bắt buộc trong config: 'dataset.type'")

        self.train_cameras = []
        self.eval_cameras = []
        self.target_cameras = []   # novel views cần sinh (không có ground truth)
        self.point_cloud_xyz = None
        self.point_cloud_rgb = None

        # Tham số normalize_scene (tính 1 lần từ train+eval cameras, áp dụng
        # đồng nhất cho point cloud + target cameras).
        self._norm_center = np.zeros(3, dtype=np.float32)
        self._norm_scale = 1.0

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
        colmap_enabled = cfg_get(self.cfg, "preprocessing.colmap.enabled", False)
        if colmap_enabled or not os.path.exists(sparse_dir):
            image_dir = os.path.join(self.data_root, cfg_get(self.cfg, "dataset.images.directory", "images"))
            sparse_dir = run_colmap_pipeline(
                image_dir, self.data_root,
                colmap_exe=cfg_get(self.cfg, "preprocessing.colmap.executable", "colmap"),
                camera_model=cfg_get(self.cfg, "preprocessing.colmap.camera_model", "PINHOLE"),
            )

        cameras_meta, images_meta, points3D = load_colmap_scene(sparse_dir)

        images_dir = os.path.join(self.data_root, cfg_get(self.cfg, "dataset.images.directory", "images"))
        all_cams_raw = []  # (uid, R, T, FoVx, FoVy, image_tensor, name) trước khi normalize
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

            all_cams_raw.append((img_id, R, T, FoVx, FoVy, image_tensor, img_meta.name))

        if points3D:
            xyz, rgb = get_scene_pointcloud(points3D)
        else:
            xyz, rgb = None, None

        self._fit_normalization(all_cams_raw)
        all_cams = [self._build_camera(*c) for c in all_cams_raw]

        if xyz is not None:
            xyz = (xyz - self._norm_center) * self._norm_scale
        self.point_cloud_xyz, self.point_cloud_rgb = xyz, rgb

        self._split_train_eval(all_cams)

    # ------------------------------------------------------------------
    def _load_nerf_transforms(self):
        """Đọc format transforms.json kiểu NeRF/Instant-NGP (dùng khi đề bài
        cấp sẵn intrinsics/extrinsics thay vì COLMAP thô)."""
        with open(os.path.join(self.data_root, "transforms.json")) as f:
            meta = json.load(f)

        images_dir = self.data_root
        camera_angle_x = meta.get("camera_angle_x")
        all_cams_raw = []
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

            all_cams_raw.append((i, R, T, FoVx, FoVy, image_tensor, os.path.basename(img_path)))

        self._fit_normalization(all_cams_raw)
        all_cams = [self._build_camera(*c) for c in all_cams_raw]

        self._split_train_eval(all_cams)
        # Với format này thường không có point cloud SfM sẵn -> khởi tạo random
        # (xem train.py, dùng preprocessing.initialize_pointcloud.random_points).
        self.point_cloud_xyz = None
        self.point_cloud_rgb = None

    # ------------------------------------------------------------------
    def _fit_normalization(self, all_cams_raw):
        """Tính center/scale của normalize_scene dựa trên vị trí thế giới
        (camera centers) của toàn bộ camera train+eval. Không dùng target
        cameras để tính (chúng chưa tồn tại lúc này và không nên ảnh hưởng
        thống kê chuẩn hoá, vốn phải cố định theo scene quan sát được)."""
        if not cfg_get(self.cfg, "preprocessing.normalize_scene", True):
            return
        centers = np.stack([_camera_center(R, T) for (_, R, T, *_rest) in all_cams_raw])
        # Tái dùng normalize_scene() với xyz=centers chỉ để lấy center/scale
        # nhất quán (dist tính trên chính camera centers).
        _, center, scale = normalize_scene(centers, centers)
        self._norm_center = center.astype(np.float32)
        self._norm_scale = float(scale)

    def _build_camera(self, uid, R, T, FoVx, FoVy, image_tensor, name):
        """Áp normalize_scene (nếu bật) lên 1 camera: giữ nguyên R (chỉ xoay,
        không đổi theo translate/scale), tính lại T sao cho camera center mới
        khớp với point cloud đã normalize."""
        if self._norm_scale != 1.0 or np.any(self._norm_center != 0):
            C = _camera_center(R, T)
            C_norm = (C - self._norm_center) * self._norm_scale
            T = -R @ C_norm
        return Camera(uid=uid, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                      image=image_tensor, image_name=name, device=self.device)

    # ------------------------------------------------------------------
    def _split_train_eval(self, all_cams):
        ratio = cfg_get(self.cfg, "dataset.split.ratio", 0.1)
        n_eval = max(1, round(len(all_cams) * ratio))
        # Chọn đều n_eval chỉ số trong [0, len) thay vì range(0, len, step)
        # (step-based range có thể sinh nhiều/ít hơn n_eval do làm tròn).
        eval_idx = set(np.linspace(0, len(all_cams) - 1, num=n_eval, dtype=int).tolist())

        self.train_cameras = [c for i, c in enumerate(all_cams) if i not in eval_idx]
        self.eval_cameras = [c for i, c in enumerate(all_cams) if i in eval_idx]

    # ------------------------------------------------------------------
    def _load_target_views(self):
        """Đọc 20-50 pose mục tiêu (novel views) mà đề bài yêu cầu sinh ảnh.
        File json dạng: [{"name": "target_001", "R":[[..]], "T":[..], "FoVx":.., "FoVy":..,
                          "width":.., "height":..}, ...]
        Không có ảnh ground truth đi kèm (image=None).

        Các pose này đến từ đề bài (hệ toạ độ COLMAP gốc), nên phải áp dụng
        CÙNG normalize_scene đã dùng cho train/eval, nếu không model sẽ nhận
        toạ độ camera sai lệch hệ quy chiếu và render ra ảnh sai hoàn toàn.
        """
        target_file = cfg_get(self.cfg, "dataset.target_views.file", "target_poses.json")
        target_path = os.path.join(self.data_root, target_file)
        if not os.path.exists(target_path):
            return
        with open(target_path) as f:
            targets = json.load(f)

        for i, t in enumerate(targets):
            R = np.array(t["R"])
            T = np.array(t["T"])
            if self._norm_scale != 1.0 or np.any(self._norm_center != 0):
                C = _camera_center(R, T)
                C_norm = (C - self._norm_center) * self._norm_scale
                T = -R @ C_norm
            cam = Camera(uid=f"target_{i}", R=R, T=T,
                         FoVx=t["FoVx"], FoVy=t["FoVy"],
                         image=None, image_name=t.get("name", f"target_{i:03d}"),
                         width=t["width"], height=t["height"],
                         device=self.device)
            self.target_cameras.append(cam)


def _camera_center(R, T):
    """Vị trí camera trong world coords: C = -R^T @ T (world->cam convention)."""
    return -R.T @ T


def fov2focal_from_x(fov_x, w):
    return w / (2 * np.tan(fov_x / 2))
