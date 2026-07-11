# Telecom Digital Twin — 3D Reconstruction & Novel View Synthesis cho trạm BTS

Hệ thống tái dựng cấu trúc 3D ngầm định (implicit) của trạm BTS từ ảnh drone,
và sinh ảnh RGB tại các góc nhìn mới, dựa trên **3D Gaussian Splatting (3DGS)**.

## Vì sao chọn 3DGS thay vì NeRF thuần?

| Tiêu chí | NeRF (MLP-based) | 3D Gaussian Splatting |
|---|---|---|
| Tốc độ train | Chậm (giờ) | Nhanh (10-30 phút/scene) |
| Tốc độ render | Chậm (giây/ảnh) | Real-time (>100 FPS) |
| Chi tiết hình học nhỏ (anten, dây cáp) | Trung bình, dễ mờ | Tốt hơn nhờ rasterization tường minh |
| Yêu cầu | Cần nhiều view, chính xác pose | Cần point cloud khởi tạo (SfM) tốt |

Với ràng buộc 100-300 ảnh/scene và yêu cầu render 20-50 novel view chất lượng
cao đúng hình học + vị trí thiết bị, 3DGS cân bằng tốt giữa tốc độ và độ chi tiết.

## Cấu trúc thư mục

```
Telecom-DigitalTwin/
├── configs/default.yaml        # toàn bộ hyperparameters
├── datasets/bts_dataset.py     # load ảnh + pose (COLMAP hoặc transforms.json)
├── preprocessing/colmap_utils.py  # chạy SfM, đọc sparse reconstruction
├── models/gaussian_model.py    # 3D Gaussian: xyz, scale, rotation, opacity, SH
├── renderer/gaussian_renderer.py  # rasterizer khả vi (gsplat)
├── losses/loss.py              # L1 + SSIM
├── trainers/trainer.py         # training loop: render->loss->backward->densify->prune
├── evaluation/metrics.py       # PSNR/SSIM/LPIPS trên tập eval (có GT)
├── inference/render_novel_views.py  # render 20-50 pose mục tiêu (không GT)
├── visualization/viz_utils.py  # xuất .ply, vẽ quỹ đạo camera
├── utils/                      # camera, SH, general, logger
├── scripts/                    # train.sh, render.sh
└── train.py                    # entry point nối toàn bộ pipeline
```

## Chuẩn bị dữ liệu

```
data/bts_scene_01/
├── images/                     # 100-300 ảnh RGB gốc
├── sparse/0/                   # cameras.txt, images.txt, points3D.txt (COLMAP)
└── target_poses.json           # 20-50 pose mục tiêu cần sinh ảnh
```

Nếu ban tổ chức chỉ cấp ảnh thô + EXIF/GPS (chưa có pose), bật:
```yaml
preprocessing:
  run_colmap: true
```
để tự động chạy SfM (`feature_extractor -> exhaustive_matcher -> mapper`).

Định dạng `target_poses.json`:
```json
[
  {"name": "target_001", "R": [[...]], "T": [tx, ty, tz],
   "FoVx": 0.9, "FoVy": 0.7, "width": 1920, "height": 1080}
]
```

## Chạy training + inference end-to-end

```bash
pip install -r requirements.txt
bash scripts/train.sh configs/default.yaml
```

`train.py` tự động: build dataset -> preprocessing -> build Gaussian model
-> train (densify/prune) -> validation định kỳ -> save checkpoint -> render
20-50 novel view ảnh cuối cùng vào `outputs/<experiment_name>/novel_views/`.

## Chỉ render lại (đã có checkpoint)

```bash
bash scripts/render.sh configs/default.yaml
```

## Các điểm cần lưu ý khi áp dụng cho trạm BTS

1. **Chi tiết nhỏ, mảnh** (dây cáp, anten, bu-lông): tăng `sh_degree` lên 3,
   giảm `densify_grad_threshold` để sinh nhiều Gaussian nhỏ hơn ở vùng chi tiết cao.
2. **Bề mặt kim loại phản chiếu** trên trụ ăng-ten: cân nhắc tăng `lambda_dssim`
   để ưu tiên giữ cấu trúc hơn là khớp màu pixel-wise tuyệt đối.
3. **Scale ngoài trời lớn**: đảm bảo `normalize_scene: true` và kiểm tra lại
   `spatial_lr_scale` (tính từ `cameras_extent`) để learning rate vị trí không
   quá nhỏ/lớn so với kích thước thật của trạm.
4. **Đánh giá hình học** (không chỉ ảnh): nên xuất thêm point cloud `.ply` để
   so sánh trực quan vị trí thiết bị với ảnh drone gốc, không chỉ dựa vào PSNR/SSIM.
