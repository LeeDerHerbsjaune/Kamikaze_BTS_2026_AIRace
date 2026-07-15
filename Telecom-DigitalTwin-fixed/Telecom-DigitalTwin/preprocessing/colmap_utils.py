"""
Preprocessing: đọc dữ liệu COLMAP (cameras.bin/txt, images.bin/txt, points3D.bin/txt)
để lấy pose camera + point cloud khởi tạo cho Gaussian Model.

Nếu dataset chưa có sparse reconstruction (chỉ có ảnh drone thô + GPS/EXIF),
có thể set preprocessing.colmap.enabled=true để tự động chạy COLMAP SfM
(feature_extractor -> exhaustive_matcher -> mapper) trước khi vào bước này.
"""
import os
import shutil
import struct
import subprocess
import numpy as np
from collections import namedtuple

CameraModel = namedtuple("CameraModel", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error"])

# id -> (model_name, num_params), theo đúng thứ tự CAMERA_MODEL_ID của COLMAP gốc
# (src/colmap/sensor/models.h). Cần bảng này để đọc cameras.bin đúng.
_CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def run_colmap_pipeline(image_dir: str, workspace_dir: str, colmap_exe: str = "colmap",
                         camera_model: str = "PINHOLE"):
    """Chạy pipeline SfM đầy đủ của COLMAP để tạo sparse reconstruction từ ảnh drone."""
    if shutil.which(colmap_exe) is None and not os.path.isfile(colmap_exe):
        raise FileNotFoundError(
            f"Không tìm thấy chương trình COLMAP ('{colmap_exe}') trong PATH hệ thống.\n"
            f"Nguyên nhân thường gặp: dataset.root ('{workspace_dir}') chưa có sẵn thư mục "
            f"'sparse/0' với pose COLMAP -> code tự động cố chạy COLMAP CLI để tự dựng SfM.\n"
            f"Cách khắc phục, chọn 1 trong 2:\n"
            f"  1) Nếu bạn ĐÃ có sẵn pose (COLMAP hoặc transforms.json): kiểm tra lại "
            f"'dataset.root' trong configs/dataset.yaml có trỏ đúng thư mục chứa "
            f"'sparse/0/cameras.txt, images.txt, points3D.txt' hay chưa.\n"
            f"  2) Nếu bạn CHƯA có pose và muốn tự động chạy SfM: cài COLMAP "
            f"(https://colmap.github.io/install.html), thêm vào PATH, hoặc sửa "
            f"'preprocessing.colmap.executable' trong configs/dataset.yaml trỏ thẳng tới "
            f"file .exe của COLMAP.")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh '{image_dir}' để chạy COLMAP feature_extractor. "
            f"Kiểm tra lại dataset.root + dataset.images.directory trong configs/dataset.yaml.")

    os.makedirs(workspace_dir, exist_ok=True)
    db_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    subprocess.run([
        colmap_exe, "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
    ], check=True)

    subprocess.run([
        colmap_exe, "exhaustive_matcher",
        "--database_path", db_path,
    ], check=True)

    subprocess.run([
        colmap_exe, "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
    ], check=True)

    return os.path.join(sparse_dir, "0")


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2],
    ])


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            props = _read_next_bytes(f, 24, "iiQQ")
            cam_id, model_id, width, height = props[0], props[1], props[2], props[3]
            model_name, num_params = _CAMERA_MODEL_IDS[model_id]
            params = np.array(_read_next_bytes(f, 8 * num_params, "d" * num_params))
            cameras[cam_id] = CameraModel(cam_id, model_name, width, height, params)
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_reg_images = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read_next_bytes(f, 64, "idddddddi")
            img_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            c = _read_next_bytes(f, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = _read_next_bytes(f, 1, "c")[0]
            num_points2d = _read_next_bytes(f, 8, "Q")[0]
            # xys + point3D_id không cần cho pipeline (chỉ dùng pose) -> đọc rồi bỏ
            _read_next_bytes(f, 24 * num_points2d, "ddq" * num_points2d)
            images[img_id] = Image(img_id, qvec, tvec, camera_id, name, None, None)
    return images


def read_points3D_binary(path):
    points = {}
    with open(path, "rb") as f:
        num_points = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(f, 43, "QdddBBBd")
            pid = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = float(props[7])
            track_length = _read_next_bytes(f, 8, "Q")[0]
            _read_next_bytes(f, 8 * track_length, "ii" * track_length)  # track, không cần dùng
            points[pid] = Point3D(pid, xyz, rgb, error)
    return points


def read_cameras_text(path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            elems = line.split()
            cam_id = int(elems[0])
            model = elems[1]
            width, height = int(elems[2]), int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            cameras[cam_id] = CameraModel(cam_id, model, width, height, params)
    return cameras


def read_images_text(path):
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#") and len(l.strip()) > 0]
    for i in range(0, len(lines), 2):
        elems = lines[i].split()
        img_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        camera_id = int(elems[8])
        name = elems[9]
        images[img_id] = Image(img_id, qvec, tvec, camera_id, name, None, None)
    return images


def read_points3D_text(path):
    points = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            elems = line.split()
            pid = int(elems[0])
            xyz = np.array(list(map(float, elems[1:4])))
            rgb = np.array(list(map(int, elems[4:7])))
            error = float(elems[7])
            points[pid] = Point3D(pid, xyz, rgb, error)
    return points


def load_colmap_scene(sparse_dir: str):
    """Load cameras/images/points3D từ thư mục sparse/0 của COLMAP.
    Tự động phát hiện định dạng .bin (mặc định của COLMAP) hoặc .txt
    (xuất ra khi chạy `colmap model_converter --output_type TXT`).
    Ưu tiên .bin nếu cả 2 cùng tồn tại."""
    has_bin = all(os.path.isfile(os.path.join(sparse_dir, f"{n}.bin"))
                  for n in ("cameras", "images", "points3D"))
    has_txt = all(os.path.isfile(os.path.join(sparse_dir, f"{n}.txt"))
                  for n in ("cameras", "images", "points3D"))

    if has_bin:
        cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        points3D = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))
    elif has_txt:
        cameras = read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
        images = read_images_text(os.path.join(sparse_dir, "images.txt"))
        points3D = read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))
    else:
        raise FileNotFoundError(
            f"Không tìm thấy bộ file COLMAP đầy đủ (cameras/images/points3D, "
            f".bin hoặc .txt) trong '{sparse_dir}'. Kiểm tra lại dataset.root trong "
            f"configs/dataset.yaml, hoặc nếu sparse_dir của bạn không có thư mục con "
            f"'0' (vd sparse/cameras.bin thay vì sparse/0/cameras.bin), sửa lại đường "
            f"dẫn cho khớp cấu trúc thật của dataset.")
    return cameras, images, points3D


def get_scene_pointcloud(points3D):
    """Trả về (N,3) xyz và (N,3) rgb [0,1] để khởi tạo Gaussian Model."""
    xyz = np.stack([p.xyz for p in points3D.values()])
    rgb = np.stack([p.rgb for p in points3D.values()]) / 255.0
    return xyz.astype(np.float32), rgb.astype(np.float32)


def normalize_scene(xyz, cam_centers):
    """Đưa scene về tâm (0,0,0), scale sao cho các camera nằm trong bán kính ~1.
    Trả về (xyz_norm, center, scale) - center/scale cần được lưu lại cùng
    checkpoint nếu muốn ánh xạ ngược kết quả về hệ toạ độ gốc (vd để so khớp
    với toạ độ GPS/EXIF thật của trạm BTS)."""
    center = cam_centers.mean(axis=0)
    dist = np.linalg.norm(cam_centers - center, axis=1).max()
    scale = 1.0 / (dist + 1e-6)
    xyz_norm = (xyz - center) * scale
    return xyz_norm, center, scale


def apply_normalization(cam_centers, center, scale):
    """Áp cùng 1 phép biến đổi normalize_scene() lên toạ độ camera (T là
    world->cam translation nên không thể transform y hệt xyz điểm; hàm này
    chỉ dùng cho các đại lượng dạng vị trí thế giới, vd cam_centers thật,
    KHÔNG áp trực tiếp lên tvec COLMAP)."""
    return (cam_centers - center) * scale
