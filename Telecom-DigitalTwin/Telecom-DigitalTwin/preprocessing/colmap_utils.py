"""
Preprocessing: đọc dữ liệu COLMAP (cameras.bin/txt, images.bin/txt, points3D.bin/txt)
để lấy pose camera + point cloud khởi tạo cho Gaussian Model.

Nếu dataset chưa có sparse reconstruction (chỉ có ảnh drone thô + GPS/EXIF),
có thể set preprocessing.run_colmap=true để tự động chạy COLMAP SfM
(feature_extractor -> exhaustive_matcher -> mapper) trước khi vào bước này.
"""
import os
import struct
import subprocess
import numpy as np
from collections import namedtuple

CameraModel = namedtuple("CameraModel", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error"])


def run_colmap_pipeline(image_dir: str, workspace_dir: str, colmap_exe: str = "colmap",
                         camera_model: str = "PINHOLE"):
    """Chạy pipeline SfM đầy đủ của COLMAP để tạo sparse reconstruction từ ảnh drone."""
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


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
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
    with open(path, "r") as f:
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
    with open(path, "r") as f:
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
    """Load cameras/images/points3D dạng .txt từ thư mục sparse/0 của COLMAP."""
    cameras = read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
    images = read_images_text(os.path.join(sparse_dir, "images.txt"))
    points3D = read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))
    return cameras, images, points3D


def get_scene_pointcloud(points3D):
    """Trả về (N,3) xyz và (N,3) rgb [0,1] để khởi tạo Gaussian Model."""
    xyz = np.stack([p.xyz for p in points3D.values()])
    rgb = np.stack([p.rgb for p in points3D.values()]) / 255.0
    return xyz.astype(np.float32), rgb.astype(np.float32)


def normalize_scene(xyz, cam_centers):
    """Đưa scene về tâm (0,0,0), scale sao cho các camera nằm trong bán kính ~1."""
    center = cam_centers.mean(axis=0)
    dist = np.linalg.norm(cam_centers - center, axis=1).max()
    scale = 1.0 / (dist + 1e-6)
    xyz_norm = (xyz - center) * scale
    return xyz_norm, center, scale
