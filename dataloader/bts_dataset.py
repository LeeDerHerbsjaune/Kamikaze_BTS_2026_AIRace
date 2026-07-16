"""
Dataset loader cho scene trạm BTS: đọc ảnh drone RGB (100-300 ảnh/scene) + pose
(dạng COLMAP hoặc transforms.json kiểu NeRF), chia train/eval, và cung cấp
danh sách target novel-view poses (20-50 pose) cần render cho vòng inference.

KIẾN TRÚC: phần đọc sparse reconstruction (cameras/images/points3D) qua pycolmap
được giao cho `dataloader.colmap_loader.ColmapLoader` (đọc thuần, không biết gì
về config/normalize/split). Module này CHỈ bọc thêm những phần ColmapLoader
không đảm nhiệm:
  1. Tự động chạy SfM bằng pycolmap nếu chưa có sparse model sẵn
     (preprocessing.colmap.enabled=true) - ColmapLoader yêu cầu sparse_path đã
     tồn tại, không tự chạy SfM.
  2. Bỏ qua (thay vì raise ngay) ảnh có trong sparse model nhưng thiếu file
     trên đĩa, có dò thêm biến thể đuôi file (.jpg/.JPG/.png...) - qua subclass
     `_TolerantColmapLoader` override `_read_image_tensor` + `load_scene`.
  3. normalize_scene: đưa scene về tâm (0,0,0), scale theo bán kính camera.
     BẮT BUỘC áp dụng ĐỒNG NHẤT lên point cloud khởi tạo, camera train/eval,
     VÀ camera target (novel views) - nếu không target novel views sẽ bị
     render sai vị trí vì lệch hệ toạ độ so với model đã train.
  4. Chia train/eval theo tỉ lệ config.
  5. Đọc target novel-view poses (target_poses.json), áp cùng normalize.
  6. Hỗ trợ song song format `nerf_transforms` (transforms.json), không đi qua
     ColmapLoader/pycolmap.
"""
import os
import json

import numpy as np
import cv2
import torch
from PIL import Image as PILImage

import pycolmap

from dataloader.entities import Frame, Scene
from dataloader.colmap import ColmapLoader
from preprocessing.colmap_utils import normalize_scene
from utils.camera_utils import Camera, focal2fov
from utils.config_loader import cfg_get


# ======================================================================
# Subclass "tolerant": bù 2 hành vi ColmapLoader gốc không có, KHÔNG sửa
# trực tiếp dataloader/colmap_loader.py (giữ ColmapLoader thuần như thiết kế
# gốc, chỉ mở rộng qua kế thừa).
# ======================================================================
class _TolerantColmapLoader(ColmapLoader):
    """Dò thêm biến thể đuôi file ảnh, và bỏ qua (thay vì raise ngay) các ảnh
    có trong sparse model nhưng không tìm thấy file thật trên đĩa - chỉ raise
    nếu TOÀN BỘ ảnh đều thiếu (giống hành vi BTSDataset bản trước)."""

    def _resolve_image_path(self, image_name: str):
        direct = os.path.join(self.images_dir, image_name)
        if os.path.isfile(direct):
            return direct
        stem, ext = os.path.splitext(image_name)
        for candidate_ext in (ext.lower(), ext.upper(), ".jpg", ".JPG",
                               ".jpeg", ".JPEG", ".png", ".PNG"):
            candidate = os.path.join(self.images_dir, stem + candidate_ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _read_image_tensor(self, image_name: str) -> torch.Tensor:
        image_path = self._resolve_image_path(image_name)
        if image_path is None:
            raise FileNotFoundError(image_name)  # bắt lại ở load_scene() bên dưới
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"File tồn tại nhưng không đọc được (hỏng/không đúng định dạng): {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        return tensor.permute(2, 0, 1)

    def load_scene(self, min_track_length: int = 3, max_error=2.0) -> Scene:
        cameras = self.load_cameras()
        images = self.load_images()
        point_cloud = self.load_points3D(min_track_length=min_track_length, max_error=max_error)

        frames = []
        missing_files = []
        n_missing_camera = 0
        # sort theo tên ảnh để thứ tự train/eval split ổn định, tái lập được
        for image in sorted(images.values(), key=lambda im: im.name):
            camera = cameras.get(image.camera_id)
            if camera is None:
                n_missing_camera += 1
                continue
            try:
                image_tensor = self._read_image_tensor(image.name)
            except FileNotFoundError:
                missing_files.append(image.name)
                continue
            frames.append(Frame(camera=camera, R=image.R, t=image.t,
                                 image_name=image.name, image=image_tensor))

        if n_missing_camera:
            print(f"[BTSDataset] Bỏ qua {n_missing_camera} ảnh do không tìm thấy "
                  f"camera_id tương ứng trong sparse reconstruction.")
        if missing_files:
            preview = ", ".join(missing_files[:5]) + (
                f", ... (+{len(missing_files) - 5} nữa)" if len(missing_files) > 5 else "")
            print(f"[BTSDataset] CẢNH BÁO: {len(missing_files)}/{len(images)} ảnh có trong "
                  f"sparse reconstruction nhưng KHÔNG tìm thấy file trong '{self.images_dir}': "
                  f"{preview}\nCác ảnh này bị BỎ QUA khỏi tập train/eval (không phải lỗi code - "
                  f"kiểm tra lại dataset gốc nếu số lượng thiếu quá lớn).")
        if not frames:
            raise FileNotFoundError(
                f"Không load được BẤT KỲ ảnh nào từ '{self.images_dir}' (toàn bộ "
                f"{len(images)} ảnh trong sparse reconstruction đều thiếu file hoặc "
                f"thiếu camera tương ứng). Kiểm tra lại 'dataset.images.directory' "
                f"trong configs/dataset.yaml.")

        return Scene(cameras=cameras, frames=frames, point_cloud=point_cloud)


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
    @staticmethod
    def _has_colmap_files(d):
        names = ("cameras", "images", "points3D")
        return (all(os.path.isfile(os.path.join(d, f"{n}.bin")) for n in names)
                or all(os.path.isfile(os.path.join(d, f"{n}.txt")) for n in names))

    def _resolve_sparse_dir(self):
        """Ưu tiên sparse/0 (chuẩn COLMAP), fallback về sparse/ trực tiếp nếu
        dataset đóng gói không có thư mục con '0' (một số bộ dataset thi đấu
        làm vậy)."""
        candidate_0 = os.path.join(self.data_root, "sparse", "0")
        if self._has_colmap_files(candidate_0):
            return candidate_0
        candidate_flat = os.path.join(self.data_root, "sparse")
        if self._has_colmap_files(candidate_flat):
            return candidate_flat
        return candidate_0  # mặc định trả về đường dẫn chuẩn để báo lỗi rõ ràng phía sau

    # ------------------------------------------------------------------
    def _load_colmap(self):
        sparse_dir = self._resolve_sparse_dir()
        colmap_enabled = cfg_get(self.cfg, "preprocessing.colmap.enabled", False)
        sparse_ready = self._has_colmap_files(sparse_dir)
        images_dir = os.path.join(self.data_root, cfg_get(self.cfg, "dataset.images.directory", "images"))

        if not sparse_ready:
            if not colmap_enabled:
                raise FileNotFoundError(
                    f"Không tìm thấy sparse reconstruction đầy đủ (cameras/images/points3D, "
                    f".bin hoặc .txt) tại '{sparse_dir}' hoặc '{os.path.join(self.data_root, 'sparse')}'.\n"
                    f"Kiểm tra lại 'dataset.root' trong configs/dataset.yaml, hoặc nếu "
                    f"chưa chạy SfM, bật 'preprocessing.colmap.enabled: true' để tự động "
                    f"chạy SfM bằng pycolmap (chỉ cần `pip install pycolmap`, không cần "
                    f"cài COLMAP CLI/PATH).")
            sparse_dir = self._run_pycolmap_pipeline(images_dir, self.data_root)
        elif colmap_enabled:
            # Người dùng bật colmap.enabled tường minh dù sparse_dir đã tồn tại
            # -> tôn trọng lựa chọn, chạy lại SfM từ đầu.
            sparse_dir = self._run_pycolmap_pipeline(images_dir, self.data_root)

        min_track_length = cfg_get(self.cfg, "preprocessing.pointcloud.min_track_length", 3)
        max_reproj_error = cfg_get(self.cfg, "preprocessing.pointcloud.max_reproj_error", 2.0)

        loader = _TolerantColmapLoader(sparse_dir, images_dir)
        scene = loader.load_scene(min_track_length=min_track_length, max_error=max_reproj_error)

        # (uid, R, T, FoVx, FoVy, image_tensor, name) trước khi normalize -
        # cùng shape mà _fit_normalization/_build_camera mong đợi.
        all_cams_raw = []
        for i, frame in enumerate(scene.frames):
            cam = frame.camera
            FoVx = focal2fov(cam.fx, cam.width)
            FoVy = focal2fov(cam.fy, cam.height)
            all_cams_raw.append((i, frame.R, frame.t, FoVx, FoVy, frame.image, frame.image_name))

        if scene.point_cloud:
            xyz = np.array([p.xyz for p in scene.point_cloud.values()], dtype=np.float32)
            rgb = np.array([p.color for p in scene.point_cloud.values()], dtype=np.uint8)
        else:
            xyz, rgb = None, None

        self._fit_normalization(all_cams_raw)
        all_cams = [self._build_camera(*c) for c in all_cams_raw]

        if xyz is not None:
            xyz = (xyz - self._norm_center) * self._norm_scale
        self.point_cloud_xyz, self.point_cloud_rgb = xyz, rgb

        self._split_train_eval(all_cams)

    def _run_pycolmap_pipeline(self, image_dir, output_root):
        """Tự chạy SfM bằng pycolmap (extract_features -> match_exhaustive ->
        incremental_mapping), không cần COLMAP CLI/PATH. Ghi kết quả ra
        '<output_root>/sparse/0' để lần chạy sau tái sử dụng ngay, khỏi phải
        chạy lại SfM (tốn thời gian với 100-300 ảnh drone)."""
        database_path = os.path.join(output_root, "database.db")
        sparse_out = os.path.join(output_root, "sparse")
        os.makedirs(sparse_out, exist_ok=True)

        if os.path.exists(database_path):
            os.remove(database_path)  # tránh lỗi "table already exists" khi chạy lại

        camera_model = cfg_get(self.cfg, "preprocessing.colmap.camera_model", "PINHOLE")

        pycolmap.extract_features(
            database_path, image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model=camera_model,
        )
        pycolmap.match_exhaustive(database_path)

        maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_out)
        if not maps:
            raise RuntimeError(
                f"pycolmap.incremental_mapping không tạo được reconstruction nào từ "
                f"ảnh trong '{image_dir}'. Kiểm tra lại chất lượng/độ chồng lấp (overlap) "
                f"của ảnh drone, hoặc thử camera_model khác trong "
                f"'preprocessing.colmap.camera_model'.")

        # SfM có thể bị chia thành nhiều model rời rạc nếu ảnh không đủ overlap
        # -> lấy model đăng ký được nhiều ảnh nhất.
        best_key = max(maps, key=lambda k: maps[k].num_reg_images())
        reconstruction = maps[best_key]
        out_dir = os.path.join(sparse_out, "0")
        reconstruction.write(out_dir)
        return out_dir

    # ------------------------------------------------------------------
    def _load_nerf_transforms(self):
        """Đọc format transforms.json kiểu NeRF/Instant-NGP (dùng khi đề bài
        cấp sẵn intrinsics/extrinsics thay vì COLMAP thô). Không đi qua
        ColmapLoader/pycolmap - format này không phải sparse reconstruction."""
        with open(os.path.join(self.data_root, "transforms.json"), encoding="utf-8") as f:
            meta = json.load(f)

        images_dir = self.data_root
        camera_angle_x = meta.get("camera_angle_x")
        all_cams_raw = []
        missing = []
        for i, frame in enumerate(meta["frames"]):
            raw_path = os.path.join(images_dir, frame["file_path"])
            img_path = raw_path if os.path.isfile(raw_path) else None
            if img_path is None:
                for ext in (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"):
                    candidate = raw_path if raw_path.lower().endswith((".png", ".jpg", ".jpeg")) else raw_path + ext
                    if os.path.isfile(candidate):
                        img_path = candidate
                        break
            if img_path is None:
                missing.append(frame["file_path"])
                continue

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1  # NeRF -> COLMAP convention
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            pil_img = PILImage.open(img_path).convert("RGB")
            w, h = pil_img.size

            FoVx = camera_angle_x if camera_angle_x else focal2fov(frame["fl_x"], w)
            FoVy = focal2fov(fov2focal_from_x(FoVx, w), h) if camera_angle_x else focal2fov(frame["fl_y"], h)

            image_tensor = torch.from_numpy(
                np.array(pil_img)).permute(2, 0, 1).float() / 255.0

            all_cams_raw.append((i, R, T, FoVx, FoVy, image_tensor, os.path.basename(img_path)))

        if missing:
            preview = ", ".join(missing[:5]) + (f", ... (+{len(missing) - 5} nữa)" if len(missing) > 5 else "")
            print(f"[BTSDataset] CẢNH BÁO: {len(missing)}/{len(meta['frames'])} frame trong transforms.json "
                  f"nhưng KHÔNG tìm thấy file ảnh trong '{images_dir}': {preview}\n"
                  f"Các frame này bị BỎ QUA khỏi tập train/eval.")
        if not all_cams_raw:
            raise FileNotFoundError(
                f"Không load được BẤT KỲ ảnh nào cho transforms.json (toàn bộ "
                f"{len(meta['frames'])} frame đều thiếu file ảnh trong '{images_dir}').")

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
        with open(target_path, encoding="utf-8") as f:
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