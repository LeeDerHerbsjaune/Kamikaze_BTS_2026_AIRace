"""
Dataset loader for BTS site scenes: reads drone RGB images (100-300 per
scene) + poses (COLMAP or NeRF-style transforms.json), splits train/eval,
and provides the list of target novel-view poses (20-50) that need to be
rendered for the inference round.

ARCHITECTURE: reading the sparse reconstruction (cameras/images/points3D)
via pycolmap is delegated to `dataloader.colmap.ColmapLoader` (a pure
reader, unaware of config/normalize/split). This module ONLY adds the parts
ColmapLoader doesn't handle:
  1. Automatically running SfM via pycolmap if no sparse model exists yet
     (preprocessing.colmap.enabled=true) - ColmapLoader requires sparse_path
     to already exist and never runs SfM itself.
  2. Skipping (instead of raising immediately) images that are listed in the
     sparse model but missing on disk, with extra detection of filename
     extension variants (.jpg/.JPG/.png...) - via the `_TolerantColmapLoader`
     subclass overriding `_read_image_tensor` + `load_scene`.
  3. normalize_scene: recenters the scene to (0,0,0) and scales it by the
     camera radius. MUST be applied CONSISTENTLY to the initial point cloud,
     train/eval cameras, AND target cameras (novel views) - otherwise target
     novel views would be rendered at the wrong position due to a mismatched
     coordinate frame relative to the trained model.
  4. Splitting train/eval according to the configured ratio.
  5. Reading target novel-view poses (target_poses.json), applying the same
     normalization.
  6. Supporting the alternative `nerf_transforms` format (transforms.json)
     in parallel, which does not go through ColmapLoader/pycolmap.
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


def _resize_rgb_array(rgb: np.ndarray, resolution) -> np.ndarray:
    """Resize an image (HxWx3 numpy array) according to
    dataset.images.resolution.

    IMPORTANT: raw DJI drone images are typically 4000x3000+ - training/eval
    at full resolution is a common cause of CUDA OOM (especially when
    evaluate_dataset() runs LPIPS/VGG, which scales with H*W). Convention
    (same as the reference 3DGS implementation):
      -1    : SAFE default - caps the longest side at 1600px if larger.
       0    : keep the original resolution (needs a lot of VRAM).
      N > 0 : divide the original resolution by N.

    NOT applied to target novel-view cameras (width/height are fixed by
    target_poses.json - submitted images must match the exact size the
    challenge requires).
    """
    if resolution == 0:
        return rgb
    h, w = rgb.shape[:2]
    if resolution == -1:
        max_side = max(w, h)
        if max_side <= 1600:
            return rgb
        scale = 1600.0 / max_side
    elif resolution > 0:
        if resolution == 1:
            return rgb
        scale = 1.0 / resolution
    else:
        raise ValueError(f"Invalid dataset.images.resolution: {resolution} "
                          f"(only -1, 0, or a positive integer are accepted).")
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ======================================================================
# "Tolerant" subclass: fills in 2 behaviors the base ColmapLoader doesn't
# have, WITHOUT modifying dataloader/colmap.py directly (keeping
# ColmapLoader pure as originally designed, only extended via inheritance).
# ======================================================================
class _TolerantColmapLoader(ColmapLoader):
    """Additionally probes for filename extension variants, and skips
    (instead of raising immediately) images listed in the sparse model but
    with no matching file on disk - only raises if ALL images are missing
    (same behavior as the previous BTSDataset version). Also resizes images
    per dataset.images.resolution (see _resize_rgb_array) - the base
    ColmapLoader knows nothing about resolution, only this config-aware
    subclass needs to."""

    def __init__(self, sparse_path: str, images_dir: str, resolution=-1):
        super().__init__(sparse_path, images_dir)
        self.resolution = resolution

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
            raise FileNotFoundError(image_name)  # caught again in load_scene() below
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"File exists but could not be read (corrupt/unsupported format): {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = _resize_rgb_array(rgb, self.resolution)
        tensor = torch.from_numpy(rgb).float() / 255.0
        return tensor.permute(2, 0, 1)

    def load_scene(self, min_track_length: int = 3, max_error=2.0) -> Scene:
        cameras = self.load_cameras()
        images = self.load_images()
        point_cloud = self.load_points3D(min_track_length=min_track_length, max_error=max_error)

        frames = []
        missing_files = []
        n_missing_camera = 0
        # sort by image name so the train/eval split order is stable and reproducible
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
            print(f"[BTSDataset] Skipped {n_missing_camera} images with no matching "
                  f"camera_id in the sparse reconstruction.")
        if missing_files:
            preview = ", ".join(missing_files[:5]) + (
                f", ... (+{len(missing_files) - 5} more)" if len(missing_files) > 5 else "")
            print(f"[BTSDataset] WARNING: {len(missing_files)}/{len(images)} images are listed in the "
                  f"sparse reconstruction but were NOT found in '{self.images_dir}': "
                  f"{preview}\nThese images are SKIPPED from the train/eval split (this is not a "
                  f"code bug - check the source dataset if the number missing is unexpectedly large).")
        if not frames:
            raise FileNotFoundError(
                f"Could not load ANY image from '{self.images_dir}' (all "
                f"{len(images)} images in the sparse reconstruction are either missing a "
                f"file or a matching camera). Check 'dataset.images.directory' "
                f"in configs/dataset.yaml.")

        return Scene(cameras=cameras, frames=frames, point_cloud=point_cloud)


class BTSDataset:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.data_root = cfg_get(cfg, "dataset.root")
        self.format = cfg_get(cfg, "dataset.type")
        if self.data_root is None:
            raise KeyError("Missing required config key: 'dataset.root'")
        if self.format is None:
            raise KeyError("Missing required config key: 'dataset.type'")

        self.train_cameras = []
        self.eval_cameras = []
        self.target_cameras = []   # novel views to generate (no ground truth)
        self.point_cloud_xyz = None
        self.point_cloud_rgb = None

        # normalize_scene parameters (computed once from train+eval cameras,
        # applied consistently to the point cloud + target cameras).
        self._norm_center = np.zeros(3, dtype=np.float32)
        self._norm_scale = 1.0

        if self.format == "colmap":
            self._load_colmap()
        elif self.format == "nerf_transforms":
            self._load_nerf_transforms()
        else:
            raise ValueError(f"Unsupported dataset format: {self.format}")

        self._load_target_views()

    # ------------------------------------------------------------------
    @staticmethod
    def _has_colmap_files(d):
        names = ("cameras", "images", "points3D")
        return (all(os.path.isfile(os.path.join(d, f"{n}.bin")) for n in names)
                or all(os.path.isfile(os.path.join(d, f"{n}.txt")) for n in names))

    def _resolve_sparse_dir(self):
        """Prefers sparse/0 (the COLMAP standard), falls back to sparse/
        directly if the packaged dataset has no '0' subdirectory (some
        competition datasets do this). If preprocessing.colmap.workspace is
        set (e.g. on Kaggle, where dataset.root lives under the read-only
        /kaggle/input), also checks sparse/ inside that workspace - so a
        later run can reuse SfM results from a previous run instead of
        re-running SfM from scratch."""
        candidates = []
        workspace = cfg_get(self.cfg, "preprocessing.colmap.workspace")
        if workspace:
            candidates.append(os.path.join(workspace, "sparse", "0"))
            candidates.append(os.path.join(workspace, "sparse"))
        candidates.append(os.path.join(self.data_root, "sparse", "0"))
        candidates.append(os.path.join(self.data_root, "sparse"))

        for candidate in candidates:
            if self._has_colmap_files(candidate):
                return candidate
        return candidates[0]  # default to the preferred path, for a clear error message later

    # ------------------------------------------------------------------
    def _load_colmap(self):
        sparse_dir = self._resolve_sparse_dir()
        colmap_enabled = cfg_get(self.cfg, "preprocessing.colmap.enabled", False)
        sparse_ready = self._has_colmap_files(sparse_dir)
        images_dir = os.path.join(self.data_root, cfg_get(self.cfg, "dataset.images.directory", "images"))

        # IMPORTANT: if dataset.root is read-only (e.g. /kaggle/input on
        # Kaggle), _run_pycolmap_pipeline() CANNOT write database.db/sparse/
        # into it -> PermissionError. preprocessing.colmap.workspace lets you
        # point to a different writable location (e.g. /kaggle/working/<scene>)
        # when auto-running SfM is needed. If unset, defaults to dataset.root
        # as before (fine for local use, where dataset.root is usually writable).
        workspace_dir = cfg_get(self.cfg, "preprocessing.colmap.workspace") or self.data_root

        if not sparse_ready:
            if not colmap_enabled:
                raise FileNotFoundError(
                    f"Could not find a complete sparse reconstruction (cameras/images/points3D, "
                    f".bin or .txt) at '{sparse_dir}'.\n"
                    f"Check 'dataset.root' in configs/dataset.yaml, or if you haven't "
                    f"run SfM yet, set 'preprocessing.colmap.enabled: true' to automatically "
                    f"run SfM via pycolmap (just needs `pip install pycolmap`, no need "
                    f"to install the COLMAP CLI/PATH). If 'dataset.root' is read-only "
                    f"(e.g. /kaggle/input), also set 'preprocessing.colmap.workspace' to "
                    f"point at a writable directory.")
            sparse_dir = self._run_pycolmap_pipeline(images_dir, workspace_dir)
        elif colmap_enabled:
            # The user explicitly enabled colmap.enabled even though
            # sparse_dir already exists -> honor that choice and re-run SfM
            # from scratch.
            sparse_dir = self._run_pycolmap_pipeline(images_dir, workspace_dir)

        min_track_length = cfg_get(self.cfg, "preprocessing.pointcloud.min_track_length", 3)
        max_reproj_error = cfg_get(self.cfg, "preprocessing.pointcloud.max_reproj_error", 2.0)

        resolution = cfg_get(self.cfg, "dataset.images.resolution", -1)
        loader = _TolerantColmapLoader(sparse_dir, images_dir, resolution=resolution)
        scene = loader.load_scene(min_track_length=min_track_length, max_error=max_reproj_error)

        # (uid, R, T, FoVx, FoVy, image_tensor, name) before normalization -
        # the same shape _fit_normalization/_build_camera expect.
        all_cams_raw = []
        for i, frame in enumerate(scene.frames):
            cam = frame.camera
            FoVx = focal2fov(cam.fx, cam.width)
            FoVy = focal2fov(cam.fy, cam.height)
            all_cams_raw.append((i, frame.R, frame.t, FoVx, FoVy, frame.image, frame.image_name))

        if scene.point_cloud:
            xyz = np.array([p.xyz for p in scene.point_cloud.values()], dtype=np.float32)
            # IMPORTANT: Point3D.color is uint8 [0,255] (matches COLMAP).
            # GaussianModel.create_from_pcd() calls RGB2SH() with the formula
            # (rgb-0.5)/C0, which assumes rgb is already normalized to
            # [0,1] - it MUST be divided by 255 here, otherwise the initial
            # Gaussian colors would be off by a factor of ~255, corrupting
            # the SH coefficients from the very first step.
            rgb = np.array([p.color for p in scene.point_cloud.values()], dtype=np.float32) / 255.0
        else:
            xyz, rgb = None, None

        self._fit_normalization(all_cams_raw)
        all_cams = [self._build_camera(*c) for c in all_cams_raw]

        if xyz is not None:
            xyz = (xyz - self._norm_center) * self._norm_scale
        self.point_cloud_xyz, self.point_cloud_rgb = xyz, rgb

        self._split_train_eval(all_cams)

    def _run_pycolmap_pipeline(self, image_dir, output_root):
        """Automatically runs SfM via pycolmap (extract_features ->
        match_exhaustive -> incremental_mapping), no COLMAP CLI/PATH needed.
        Writes the result to '<output_root>/sparse/0' so a later run can
        reuse it right away instead of re-running SfM (which is slow with
        100-300 drone images)."""
        database_path = os.path.join(output_root, "database.db")
        sparse_out = os.path.join(output_root, "sparse")
        os.makedirs(sparse_out, exist_ok=True)

        if os.path.exists(database_path):
            os.remove(database_path)  # avoid a "table already exists" error on re-run

        camera_model = cfg_get(self.cfg, "preprocessing.colmap.camera_model", "PINHOLE")

        # IMPORTANT: camera_model is NOT a direct kwarg of extract_features()
        # - it must go through reader_options=ImageReaderOptions(
        # camera_model=...). Verified in practice with pycolmap 4.1.0:
        # passing camera_model= directly raises a TypeError (the parameter
        # doesn't exist on that signature - easy to get confused because
        # pycolmap.CameraMode.SINGLE IS a direct kwarg of extract_features,
        # just not part of reader_options).
        pycolmap.extract_features(
            database_path, image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=pycolmap.ImageReaderOptions(camera_model=camera_model),
        )
        pycolmap.match_exhaustive(database_path)

        maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_out)
        if not maps:
            raise RuntimeError(
                f"pycolmap.incremental_mapping produced no reconstruction from the "
                f"images in '{image_dir}'. Check the quality/overlap of the drone "
                f"images, or try a different camera_model via "
                f"'preprocessing.colmap.camera_model'.")

        # SfM can split into several disconnected models if the images don't
        # have enough overlap -> take the model that registered the most images.
        best_key = max(maps, key=lambda k: maps[k].num_reg_images())
        reconstruction = maps[best_key]
        out_dir = os.path.join(sparse_out, "0")
        reconstruction.write(out_dir)
        return out_dir

    # ------------------------------------------------------------------
    def _load_nerf_transforms(self):
        """Reads the NeRF/Instant-NGP-style transforms.json format (used
        when the challenge provides intrinsics/extrinsics directly instead
        of raw COLMAP). Does not go through ColmapLoader/pycolmap - this
        format is not a sparse reconstruction."""
        with open(os.path.join(self.data_root, "transforms.json"), encoding="utf-8") as f:
            meta = json.load(f)

        images_dir = self.data_root
        camera_angle_x = meta.get("camera_angle_x")
        resolution = cfg_get(self.cfg, "dataset.images.resolution", -1)
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

            rgb_arr = _resize_rgb_array(np.array(pil_img), resolution)
            image_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).float() / 255.0

            all_cams_raw.append((i, R, T, FoVx, FoVy, image_tensor, os.path.basename(img_path)))

        if missing:
            preview = ", ".join(missing[:5]) + (f", ... (+{len(missing) - 5} more)" if len(missing) > 5 else "")
            print(f"[BTSDataset] WARNING: {len(missing)}/{len(meta['frames'])} frames in transforms.json "
                  f"have NO matching image file in '{images_dir}': {preview}\n"
                  f"These frames are SKIPPED from the train/eval split.")
        if not all_cams_raw:
            raise FileNotFoundError(
                f"Could not load ANY image for transforms.json (all "
                f"{len(meta['frames'])} frames are missing their image file in '{images_dir}').")

        self._fit_normalization(all_cams_raw)
        all_cams = [self._build_camera(*c) for c in all_cams_raw]

        self._split_train_eval(all_cams)
        # This format usually has no ready-made SfM point cloud -> random
        # init instead (see train.py, uses
        # preprocessing.initialize_pointcloud.random_points).
        self.point_cloud_xyz = None
        self.point_cloud_rgb = None

    # ------------------------------------------------------------------
    def _fit_normalization(self, all_cams_raw):
        """Computes normalize_scene's center/scale from the world position
        (camera centers) of all train+eval cameras. Target cameras are not
        used for this computation (they don't exist yet at this point, and
        shouldn't influence the normalization stats, which must be fixed
        relative to the observed scene)."""
        if not cfg_get(self.cfg, "preprocessing.normalize_scene", True):
            return
        centers = np.stack([_camera_center(R, T) for (_, R, T, *_rest) in all_cams_raw])
        # Reuse normalize_scene() with xyz=centers just to get a consistent
        # center/scale (dist computed on the camera centers themselves).
        _, center, scale = normalize_scene(centers, centers)
        self._norm_center = center.astype(np.float32)
        self._norm_scale = float(scale)

    def _build_camera(self, uid, R, T, FoVx, FoVy, image_tensor, name):
        """Applies normalize_scene (if enabled) to one camera: keeps R
        unchanged (rotation only, unaffected by translate/scale), recomputes
        T so the new camera center matches the normalized point cloud."""
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
        # Evenly pick n_eval indices in [0, len) instead of range(0, len, step)
        # (a step-based range can produce more/fewer than n_eval due to rounding).
        eval_idx = set(np.linspace(0, len(all_cams) - 1, num=n_eval, dtype=int).tolist())

        self.train_cameras = [c for i, c in enumerate(all_cams) if i not in eval_idx]
        self.eval_cameras = [c for i, c in enumerate(all_cams) if i in eval_idx]

    # ------------------------------------------------------------------
    def _load_target_views(self):
        """Reads the 20-50 target (novel-view) poses the challenge requires
        images to be generated for. JSON format:
        [{"name": "target_001", "R":[[..]], "T":[..], "FoVx":.., "FoVy":..,
          "width":.., "height":..}, ...]
        No ground-truth image is attached (image=None).

        These poses come from the challenge (original COLMAP coordinate
        frame), so the SAME normalize_scene applied to train/eval must be
        applied here too - otherwise the model would receive camera
        coordinates in the wrong reference frame and render completely
        wrong images.
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
    """Camera position in world coords: C = -R^T @ T (world->cam convention)."""
    return -R.T @ T


def fov2focal_from_x(fov_x, w):
    return w / (2 * np.tan(fov_x / 2))
