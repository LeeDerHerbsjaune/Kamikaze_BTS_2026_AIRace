"""
Renderer: wraps a differentiable rasterizer for 3D Gaussian Splatting.
Defaults to `gsplat` (pip install gsplat) - lighter and easier to build than
the original CUDA extension `diff-gaussian-rasterization`. Backend is
selectable via config (renderer.backend: "gsplat" | "diff_gaussian_rasterization").

Cross-checked against the reference render() in
graphdeco-inria/gaussian-splatting/gaussian_renderer/__init__.py:
- SH -> RGB conversion formula `clamp_min(sh2rgb + 0.5, 0.0)` matches exactly
  (reference applies this in its Python `convert_SHs_python` path via eval_sh).
- The screenspace_points / retain_grad() pattern for capturing 2D gradients
  for densification matches the reference's own zero-tensor + retain_grad()
  trick (see _render_diffgs below).

Returns: rendered image (3,H,W), radii (used to prune by screen size), and
viewspace_points (keeps gradient, used for densification stats).
"""
import torch
import math


def render(camera, gaussians, bg_color, backend="gsplat", scaling_modifier=1.0):
    """
    camera: Camera object (utils/camera_utils.py)
    gaussians: GaussianModel
    bg_color: (3,) tensor, background color
    """
    if backend == "gsplat":
        return _render_gsplat(camera, gaussians, bg_color, scaling_modifier)
    elif backend == "diff_gaussian_rasterization":
        return _render_diffgs(camera, gaussians, bg_color, scaling_modifier)
    else:
        raise ValueError(f"Unsupported renderer backend: {backend}")


def _render_gsplat(camera, gaussians, bg_color, scaling_modifier):
    import gsplat

    means3d = gaussians.xyz
    scales = gaussians.scaling * scaling_modifier
    quats = gaussians.rotation
    opacities = gaussians.opacity.squeeze(-1)

    # SH -> view-dependent color, evaluated along the direction from
    # camera_center to each Gaussian (same convention as the reference's
    # `dir_pp_normalized = (get_xyz - camera_center)` in the Python SH path).
    viewdirs = torch.nn.functional.normalize(
        means3d - camera.camera_center, dim=-1)
    colors = gsplat.spherical_harmonics(
        gaussians.active_sh_degree, viewdirs, gaussians.features)
    colors = torch.clamp_min(colors + 0.5, 0.0)

    # camera.world_view_transform (utils/camera_utils.py) is already the
    # world->cam matrix TRANSPOSED once, following the row-vector convention
    # used by the original diff-gaussian-rasterization extension.
    # gsplat.rasterization() instead expects a plain world->cam matrix
    # (column-vector convention, NOT transposed) - so we transpose it back
    # once here to undo the earlier transpose and match gsplat's convention.
    # If Camera in utils/camera_utils.py is ever rewritten, re-check this line.
    viewmat = camera.world_view_transform.transpose(0, 1)
    K = _fov_to_intrinsics(camera)

    render_colors, render_alphas, meta = gsplat.rasterization(
        means=means3d,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=camera.width,
        height=camera.height,
        backgrounds=bg_color.unsqueeze(0),
        packed=False,
    )

    image = render_colors[0].permute(2, 0, 1).clamp(0, 1)  # (3,H,W)
    radii = meta["radii"][0] if "radii" in meta else torch.zeros(means3d.shape[0], device=means3d.device)

    means2d = meta.get("means2d")
    if isinstance(means2d, torch.Tensor):
        viewspace_points = means2d[0]
        # IMPORTANT: means2d is an intermediate tensor in gsplat's computation
        # graph, NOT a leaf tensor - .grad would stay None after
        # loss.backward() unless retain_grad() is called explicitly here.
        # Missing this line silently breaks Trainer.add_densification_stats()
        # (out["viewspace_points"].grad always None), disabling densify/
        # clone/split entirely even with densify.enabled=true. Same pattern
        # as the reference's own `screenspace_points.retain_grad()` in
        # gaussian_renderer/__init__.py - see module docstring above.
        if viewspace_points.requires_grad and not viewspace_points.is_leaf:
            viewspace_points.retain_grad()
    else:
        viewspace_points = means3d

    visibility_filter = radii > 0

    return {
        "render": image,
        "viewspace_points": viewspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }


def _render_diffgs(camera, gaussians, bg_color, scaling_modifier):
    """Alternative backend using the original CUDA extension from 3DGS
    (Kerbl et al. 2023), kept for teams that already have the extension
    built on their training machine."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)

    # Same zero-tensor + retain_grad() trick as the reference implementation
    # (`screenspace_points = torch.zeros_like(pc.get_xyz, ...) + 0`) to make
    # PyTorch return gradients of the 2D (screen-space) means, used later for
    # densification statistics.
    screenspace_points = torch.zeros_like(gaussians.xyz, requires_grad=True, device=gaussians.device)
    screenspace_points.retain_grad()

    raster_settings = GaussianRasterizationSettings(
        image_height=camera.height, image_width=camera.width,
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color, scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
        means3D=gaussians.xyz, means2D=screenspace_points,
        shs=gaussians.features, colors_precomp=None,
        opacities=gaussians.opacity, scales=gaussians.scaling,
        rotations=gaussians.rotation, cov3D_precomp=None,
    )
    return {
        "render": rendered_image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def _fov_to_intrinsics(camera):
    from utils.camera_utils import fov2focal
    fx = fov2focal(camera.FoVx, camera.width)
    fy = fov2focal(camera.FoVy, camera.height)
    cx, cy = camera.width / 2.0, camera.height / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                      device=camera.device, dtype=torch.float32)
    return K
