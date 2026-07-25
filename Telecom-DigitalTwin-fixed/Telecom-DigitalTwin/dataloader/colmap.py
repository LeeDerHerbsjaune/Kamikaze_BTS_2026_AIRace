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
    """Loads a sparse reconstruction (cameras, images, points3D) from COLMAP
    via pycolmap, and assembles it into one complete Scene ready to hand off
    to the Trainer."""

    def __init__(self, sparse_path: str, images_dir: str):
        if not os.path.isdir(sparse_path):
            raise FileNotFoundError(f"Sparse reconstruction directory not found: {sparse_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        self.images_dir = images_dir
        self.reconstruction = pycolmap.Reconstruction(sparse_path)  # reads the .bin or .txt files

        if len(self.reconstruction.cameras) == 0:
            raise ValueError(f"Reconstruction at {sparse_path} has no cameras.")
        if len(self.reconstruction.images) == 0:
            raise ValueError(f"Reconstruction at {sparse_path} has no images.")

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
                f"Camera model '{model_name}' is not supported yet. Add it to _CAMERA_PARAM_LAYOUT.")
        if "fx" in layout and "fy" in layout:
            fx, fy = params[layout.index("fx")], params[layout.index("fy")]
        else:
            fx = fy = params[layout.index("f")]
        cx = params[layout.index("cx")]
        cy = params[layout.index("cy")]
        return float(fx), float(fy), float(cx), float(cy)

    # ------------------------------------------------------------------
    def load_images(self) -> dict[int, Image]:
        """Reads metadata + pose only, does NOT touch any image files on disk."""
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
        """Compatible with both the old pycolmap API (cam_from_world is a
        PROPERTY, returns a Rigid3d directly) and the new one
        (pycolmap>=3.x: cam_from_world is a METHOD, must be called `()` to
        get a Rigid3d - verified against pycolmap 4.1.0 in practice).
        hasattr() can't tell these two cases apart (a bound method still
        satisfies hasattr), so we must check callable() and call it if
        needed, to avoid the bug where a
        'builtin_function_or_method' object has no attribute 'rotation'."""
        if hasattr(colmap_image, "cam_from_world"):
            pose = colmap_image.cam_from_world
            if callable(pose):
                pose = pose()
            return pose.rotation.matrix(), pose.translation
        if hasattr(colmap_image, "R") and hasattr(colmap_image, "t"):
            return colmap_image.R, colmap_image.t
        raise AttributeError(
            "Could not find a pose (cam_from_world or R/t) - check your pycolmap version.")

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
            print(f"[ColmapLoader] Filtered out {n_skipped}/{len(self.reconstruction.points3D)} noisy points.")
        return points3D

    # ------------------------------------------------------------------
    def _read_image_tensor(self, image_name: str) -> torch.Tensor:
        """Reads one image file from disk -> (3,H,W) float [0,1] tensor.
        Kept as a separate method so BTSDataset can override it (e.g. for
        lazy-loading / caching) without touching the loader itself."""
        image_path = os.path.join(self.images_dir, image_name)
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        return tensor.permute(2, 0, 1)

    def load_scene(self, min_track_length: int = 3, max_error: Optional[float] = 2.0) -> Scene:
        """Final assembly step: Camera + Image (pose) + real pixels on disk
        -> Frame, then bundles everything (cameras, frames, point_cloud)
        into one Scene returned to the Trainer.

        This is the main entry point to call from outside - prefer it over
        calling load_cameras()/load_images()/load_points3D() separately and
        assembling them by hand.
        """
        cameras = self.load_cameras()
        images = self.load_images()
        point_cloud = self.load_points3D(min_track_length=min_track_length, max_error=max_error)

        frames = []
        n_missing = 0
        # Sort by image name: dict.values() does not guarantee a stable
        # order across runs/machines - without sorting, the train/eval split
        # (index-based) would not be reproducible even with the same seed.
        # BTSDataset (the main caller) uses a tolerant subclass that
        # overrides load_scene() with its own sort, but we sort here too so
        # ColmapLoader still behaves correctly when used standalone.
        for image in sorted(images.values(), key=lambda im: im.name):
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
            print(f"[ColmapLoader] Skipped {n_missing} images with no matching camera_id.")

        return Scene(cameras=cameras, frames=frames, point_cloud=point_cloud)