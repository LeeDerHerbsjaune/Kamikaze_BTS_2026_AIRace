# Telecom Digital Twin — 3D Reconstruction & Novel View Synthesis cho trạm BTS

Hệ thống tái dựng cấu trúc 3D ngầm định (implicit) của trạm BTS từ ảnh drone,
và sinh ảnh RGB tại các góc nhìn mới, dựa trên **3D Gaussian Splatting (3DGS)**.

> **Bản này đã được sửa lỗi mapping config + dọn code chết.**
> Xem chi tiết từng thay đổi trong [`CHANGELOG.md`](./CHANGELOG.md).

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
├── configs/                    # toàn bộ hyperparameters (chia nhỏ theo chủ đề)
│   ├── default.yaml             # gộp tất cả file con qua key `include`
│   ├── dataset.yaml               # dataset + preprocessing
│   ├── gaussian.yaml               # model (sh_degree, init, optimize flags)
│   ├── renderer.yaml                # renderer backend/background
│   ├── loss.yaml                     # loss weights + evaluation metrics
│   ├── train.yaml                     # optimizer + training + densify/pruning
│   └── inference.yaml                  # checkpoint + output + video
├── dataloader/                 # ĐỌC dữ liệu thô -> Camera/Scene sẵn dùng cho Trainer
│   ├── entities.py               # dataclass thuần: Camera/Image/Point3D/Frame/Scene
│   │                              # (chỉ intrinsics/pose/point3D thô, KHÔNG biết config)
│   ├── colmap.py                  # ColmapLoader: đọc sparse reconstruction qua
│   │                              # `pycolmap` (KHÔNG tự viết tay parser .bin/.txt),
│   │                              # thuần - không biết gì về config/resolution/normalize
│   └── bts_dataset.py              # BTSDataset (entry point) + _TolerantColmapLoader:
│                                    # lớp BỌC config-aware, cộng thêm vào ColmapLoader:
│                                    #  - tự dò biến thể đuôi file ảnh, bỏ qua ảnh thiếu
│                                    #  - resize ảnh theo dataset.images.resolution
│                                    #  - tự chạy SfM qua pycolmap nếu chưa có sparse/
│                                    #    (extract_features -> match_exhaustive ->
│                                    #    incremental_mapping - KHÔNG cần cài CLI `colmap`!)
│                                    #  - normalize_scene, train/eval split, target views
├── preprocessing/colmap_utils.py  # CHỈ còn normalize_scene() - tiện ích thuần toán học
│                                    # dùng chung, không liên quan đọc/ghi file. Việc đọc
│                                    # COLMAP đã chuyển hết sang dataloader/colmap.py.
├── models/gaussian_model.py    # 3D Gaussian: xyz, scale, rotation, opacity, SH
├── renderer/gaussian_renderer.py  # rasterizer khả vi (gsplat / diff-gaussian)
├── losses/loss.py              # L1 + SSIM
├── trainers/trainer.py         # training loop: render->loss->backward->densify->prune
├── evaluation/metrics.py       # PSNR/SSIM/LPIPS trên tập eval (có GT)
├── inference/render_novel_views.py  # render 20-50 pose mục tiêu (không GT)
├── visualization/viz_utils.py  # xuất .ply, vẽ quỹ đạo camera
├── utils/                      # camera, config_loader (cfg_get), SH, general, logger
├── scripts/                    # train.sh, render.sh
└── train.py                    # entry point nối toàn bộ pipeline
```

**`dataloader/` vs `preprocessing/` khác nhau ở đâu?**
- `dataloader/` = biết cách **đọc** dữ liệu (ảnh, pose, point cloud) từ đĩa và
  ghép thành đối tượng Python sẵn dùng. `colmap.py` (đọc thuần qua pycolmap)
  tách biệt khỏi `bts_dataset.py` (lớp bọc thêm mọi hành vi phụ thuộc config:
  resize, tolerant-missing-file, auto-SfM, normalize, split) - tách theo
  nguyên tắc 1 lớp thuần (dễ test/tái dùng) + 1 lớp cấu hình (biết `cfg`).
- `preprocessing/` = tiện ích **xử lý số liệu thuần tuý**, không đọc/ghi file,
  không phụ thuộc định dạng dữ liệu gốc. Hiện chỉ còn `normalize_scene()`
  (đưa scene về tâm + scale theo bán kính camera) vì đây là phép biến đổi áp
  dụng SAU khi đã có Camera/point cloud, bất kể chúng đến từ COLMAP hay
  transforms.json.

**Lưu ý:** thư mục `models/gaussian/` (stub rỗng), `datasets/` (kiến trúc cũ
đọc COLMAP bằng parser tự viết tay), và `datasets/entities.py`/
`datasets/colmap_loader.py` (bản nháp trước đó của `dataloader/`, không tương
thích renderer) đã được loại bỏ khỏi bản này — xem lý do và lịch sử đầy đủ
trong `CHANGELOG.md`.

## Ghi log & theo dõi kết quả

Mỗi lần chạy `train.py`/`scripts/render.sh` tạo trong
`outputs/<experiment_name>/`:
- **`train.log`** — log dạng dòng, có timestamp, gồm cả các dòng `[progress]`
  (loss, n_gaussians, it/s, ETA, learning rate ghi mỗi `log_interval`) và
  `[densify @ iter N]` (số Gaussian trước/sau mỗi lần densify).
- **`notes.md`** — bảng markdown liệt kê MỌI file kết quả sinh ra (checkpoint,
  `point_cloud_final.ply`, `camera_trajectory.png`, ảnh + video novel view),
  kèm iteration, kích thước file, và eval metric gần nhất tại thời điểm đó.
  Có thêm khối "Tổng kết" ở cuối training/inference (tổng thời gian, PSNR tốt
  nhất, v.v.) - đọc nhanh file này để biết 1 lần chạy đã tạo ra những gì mà
  không cần lục cả `train.log`.
- **TensorBoard** (nếu cài `tensorboard`) — cùng thư mục, xem bằng
  `tensorboard --logdir outputs/<experiment_name>`.

## Chuẩn bị dữ liệu

```
data/bts_scene_01/
├── images/                     # 100-300 ảnh RGB gốc
├── sparse/0/                   # cameras.txt, images.txt, points3D.txt (COLMAP)
└── target_poses.json           # 20-50 pose mục tiêu cần sinh ảnh
```

Nếu ban tổ chức chỉ cấp ảnh thô + EXIF/GPS (chưa có pose), bật trong
`configs/dataset.yaml`:
```yaml
preprocessing:
  colmap:
    enabled: true
```
để tự động chạy SfM qua thư viện `pycolmap` (`extract_features ->
match_exhaustive -> incremental_mapping`) — **không cần cài đặt COLMAP CLI**
riêng, chỉ cần `pip install pycolmap` (đã có trong `requirements.txt`). Phù
hợp cả trên Kaggle, nơi không có sẵn `colmap.exe`/binary trong PATH.

Định dạng `target_poses.json`:
```json
[
  {"name": "target_001", "R": [[...]], "T": [tx, ty, tz],
   "FoVx": 0.9, "FoVy": 0.7, "width": 1920, "height": 1080}
]
```
Pose này ở hệ toạ độ COLMAP gốc — nếu `preprocessing.normalize_scene: true`,
pipeline sẽ tự áp cùng phép chuẩn hoá scene lên target pose trước khi render,
bạn **không cần tự normalize file này**.

## Chạy training + inference end-to-end

```bash
pip install -r requirements.txt
bash scripts/train.sh configs/default.yaml
```

`train.py` tự động: build dataset -> preprocessing -> build Gaussian model
-> train (densify/prune) -> validation định kỳ -> save checkpoint -> render
20-50 novel view ảnh cuối cùng vào `outputs/<experiment_name>/novel_views/`.

## Chạy trên Kaggle Notebook

Xem hướng dẫn riêng: [`KAGGLE.md`](./KAGGLE.md). Dùng
`configs/default.kaggle.yaml` (include `configs/dataset.kaggle.yaml`, trỏ
sẵn vào `/kaggle/input/...` và ghi output vào `/kaggle/working/outputs/`)
thay vì `configs/default.yaml`.

## Chỉ render lại (đã có checkpoint)

```bash
bash scripts/render.sh configs/default.yaml
```
Đảm bảo `configs/inference.yaml` → `inference.checkpoint` trỏ đúng file
`.pth` (mặc định `./outputs/bts_scene_01/checkpoints/last.pth`).

## Kiểm tra config nhanh (khuyến nghị chạy trước khi train)

```bash
python -c "from utils.config_loader import load_config; print(load_config('configs/default.yaml').keys())"
```

## Các điểm cần lưu ý khi áp dụng cho trạm BTS

1. **Chi tiết nhỏ, mảnh** (dây cáp, anten, bu-lông): tăng `model.sh_degree` lên 3,
   giảm `densify.grad_threshold` để sinh nhiều Gaussian nhỏ hơn ở vùng chi tiết cao.
2. **Bề mặt kim loại phản chiếu** trên trụ ăng-ten: cân nhắc tăng
   `loss.dssim.weight` để ưu tiên giữ cấu trúc hơn là khớp màu pixel-wise tuyệt đối.
3. **Scale ngoài trời lớn**: đảm bảo `preprocessing.normalize_scene: true` —
   phép chuẩn hoá này giờ áp dụng nhất quán cho cả point cloud, train/eval
   camera, và target camera (xem `CHANGELOG.md` mục 5).
4. **Đánh giá hình học** (không chỉ ảnh): nên xuất thêm point cloud `.ply` để
   so sánh trực quan vị trí thiết bị với ảnh drone gốc, không chỉ dựa vào PSNR/SSIM.
5. **Tắt học 1 phần thuộc tính Gaussian** (`model.optimize.*=false`): chỉ nên
   dùng khi `densify.enabled: false`, do densify/prune cần resize đồng bộ cả
   6 thuộc tính (xyz, scale, rotation, opacity, SH DC/rest).
