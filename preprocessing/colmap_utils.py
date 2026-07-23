"""
Preprocessing: shared scene-normalization utility used by the data loading
pipeline.

ARCHITECTURE NOTE: reading the sparse reconstruction (cameras/images/points3D)
and auto-running SfM when it doesn't exist yet is now handled by
`dataloader/colmap.py` (ColmapLoader, via the `pycolmap` library) and
`dataloader/bts_dataset.py` (_run_pycolmap_pipeline) - no more hand-written
.bin/.txt parsers or shelling out to the `colmap` CLI like the previous
version. This file only keeps `normalize_scene()`, the one function
`dataloader/bts_dataset.py` still relies on.
"""
import numpy as np


def normalize_scene(xyz, cam_centers):
    """Recenter the scene to (0,0,0) and scale it so the cameras sit within
    roughly unit radius. Returns (xyz_norm, center, scale) - center/scale
    should be saved alongside the checkpoint if you ever need to map results
    back to the original coordinate system (e.g. to match real GPS/EXIF
    coordinates of the BTS site)."""
    center = cam_centers.mean(axis=0)
    dist = np.linalg.norm(cam_centers - center, axis=1).max()
    scale = 1.0 / (dist + 1e-6)
    xyz_norm = (xyz - center) * scale
    return xyz_norm, center, scale
