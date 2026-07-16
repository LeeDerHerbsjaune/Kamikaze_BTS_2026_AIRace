"""
Renderer: wrap thư viện rasterization khả vi cho 3D Gaussian Splatting.
Mặc định dùng `gsplat` (pip install gsplat) - nhẹ và dễ cài hơn CUDA extension
gốc `diff-gaussian-rasterization`. Có thể đổi backend qua config
(renderer.backend: "gsplat" | "diff_gaussian_rasterization").

Trả về: ảnh render (3,H,W), radii (dùng để prune theo screen size), và
viewspace_points (giữ gradient để tính stats densification).
"""
import torch
import math


def render(camera, gaussians, bg_color, backend="gsplat", scaling_modifier=1.0):
    """
    camera: Camera object (utils/camera_utils.py)
    gaussians: GaussianModel
    bg_color: (3,) tensor, màu nền
    """
    if backend == "gsplat":
        return _render_gsplat(camera, gaussians, bg_color, scaling_modifier)
    elif backend == "diff_gaussian_rasterization":
        return _render_diffgs(camera, gaussians, bg_color, scaling_modifier)
    else:
        raise ValueError(f"Renderer backend không hỗ trợ: {backend}")


def _render_gsplat(camera, gaussians, bg_color, scaling_modifier):
    import gsplat

    means3d = gaussians.xyz
    scales = gaussians.scaling * scaling_modifier
    quats = gaussians.rotation
    opacities = gaussians.opacity.squeeze(-1)

    # SH -> màu view-dependent, evaluate theo hướng nhìn từ camera_center
    viewdirs = torch.nn.functional.normalize(
        means3d - camera.camera_center, dim=-1)
    colors = gsplat.spherical_harmonics(
        gaussians.active_sh_degree, viewdirs, gaussians.features)
    colors = torch.clamp_min(colors + 0.5, 0.0)

    # camera.world_view_transform (utils/camera_utils.py) đã là ma trận
    # world->cam CHUYỂN VỊ 1 lần theo convention của diff-gaussian-rasterization
    # gốc (row-vector). gsplat.rasterization() lại cần viewmat dạng world->cam
    # thông thường (column-vector, KHÔNG transpose) -> transpose lại 1 lần nữa
    # ở đây để "huỷ" transpose trước đó và trả về đúng convention gsplat cần.
    # Nếu đổi/viết lại Camera trong utils/camera_utils.py, kiểm tra lại dòng này.
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
        # QUAN TRỌNG: means2d là tensor trung gian trong đồ thị tính toán của
        # gsplat, KHÔNG phải leaf tensor -> .grad sẽ luôn là None sau
        # loss.backward() nếu không gọi retain_grad() tường minh ở đây. Thiếu
        # dòng này khiến Trainer.add_densification_stats() không bao giờ chạy
        # (out["viewspace_points"].grad luôn None), vô hiệu hoá hoàn toàn
        # densify/clone/split dù config có bật densify.enabled=true.
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
    """Backend thay thế dùng CUDA extension gốc của 3DGS (Kerbl et al. 2023),
    giữ nguyên nếu team đã có sẵn extension build được trên máy huấn luyện."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)

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
