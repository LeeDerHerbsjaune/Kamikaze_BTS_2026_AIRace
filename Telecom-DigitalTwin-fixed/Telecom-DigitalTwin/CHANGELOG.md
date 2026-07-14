# Changelog — bản sửa lỗi & cải thiện repo

## 🔴 Lỗi nghiêm trọng (repo cũ crash ngay khi chạy `train.py`)

1. **`cfg["train"]` không tồn tại.** `configs/train.yaml` khai báo key gốc là
   `training:`, nhưng `train.py`/`trainer.py` cũ đọc `cfg["train"]`. Đã đồng bộ
   toàn bộ code để đọc đúng `training.*`.

2. **`trainers/trainer.py` đọc sai gần như mọi key config**: `optimizer.iterations`
   thay vì `training.iterations`, `optimizer.position_lr_init` (flat) thay vì
   `optimizer.position.lr_init` (nested), `densify_cfg["min_opacity"]` /
   `["max_screen_size"]` / `["opacity_reset_interval"]` vốn thực ra nằm ở
   section `pruning.*` chứ không phải `densify.*`, `loss.lambda_dssim` thay vì
   `loss.dssim.weight`, `loss.use_lpips_eval` thay vì `evaluation.use_lpips`...
   → Đã viết lại toàn bộ bằng `cfg_get(cfg, "dotted.path", default)` khớp
   chính xác cấu trúc yaml, có giá trị mặc định an toàn thay vì `KeyError`.

3. **`datasets/bts_dataset.py` đọc sai toàn bộ `configs/dataset.yaml`**:
   `dataset.data_root` → đúng phải là `dataset.root`; `dataset.format` →
   `dataset.type`; `dataset.images_dir` → `dataset.images.directory`;
   `preprocessing.run_colmap` → `preprocessing.colmap.enabled`;
   `dataset.eval_split_ratio` → `dataset.split.ratio`;
   `dataset.target_views_file` → `dataset.target_views.file`. Đã sửa hết.

4. **`inference/render_novel_views.py`** đọc `cfg["inference"]["output_dir"]`,
   `["render_video"]`, `["fps"]` — các key này không tồn tại trong
   `configs/inference.yaml` (đúng phải là `inference.output.directory`,
   `inference.render.video.enabled`, `inference.render.video.fps`). Script chỉ
   "chạy được" trước đây vì `train.py` tự set `output_dir` bằng tay, còn chạy
   độc lập qua `scripts/render.sh` thì luôn crash — đã sửa để chạy độc lập được.

## 🟠 Lỗi logic/thiết kế

5. **`normalize_scene()` được định nghĩa nhưng KHÔNG BAO GIỜ được gọi** dù
   config bật `preprocessing.normalize_scene: true`. Đã nối lại: scene được
   chuẩn hoá dựa trên thống kê camera centers, áp dụng nhất quán cho:
   point cloud khởi tạo, camera train/eval, **và cả camera target (novel
   views)** — điểm quan trọng hay bị bỏ sót nhất, vì nếu chỉ normalize
   point cloud/train camera mà không normalize target pose thì model sẽ
   render sai hoàn toàn vị trí ở bước inference (lệch hệ toạ độ).

6. **`model.optimize.{position,rotation,scale,opacity,sh}`** trong
   `configs/gaussian.yaml` trước đây không được đọc ở đâu cả — Trainer luôn
   bật gradient cho cả 6 param-group. Đã sửa để tôn trọng đúng các cờ này,
   kèm cảnh báo rõ ràng nếu người dùng tắt optimize riêng lẻ trong khi
   `densify.enabled=true` (2 tính năng này xung đột về mặt shape, xem code).

7. **Thư mục `models/gaussian/` (initialization.py, gaussian.py, densify.py,
   optimizer.py, prune.py) là các file stub rỗng, không được import ở bất kỳ
   đâu** — sinh ra do bug trong `docs/createfolder.py` (biến `parent_dir` lẽ
   ra phải là `"docs"` nhưng lại là `"models"`). Logic Gaussian thật nằm ở
   `models/gaussian_model.py` (flat). Đã xoá bỏ thư mục stub gây nhầm lẫn này.

8. **`datasets/entities.py` + `datasets/colmap_loader.py` là một pipeline dữ
   liệu song song, hoàn toàn không được `train.py`/`trainer.py`/renderer sử
   dụng**, và không tương thích: `ColmapLoader` trả về `Scene`/`Frame` dùng
   `fx,fy,cx,cy` trong khi `renderer/gaussian_renderer.py` cần
   `camera.world_view_transform`, `FoVx`, `FoVy` từ `utils/camera_utils.Camera`.
   Nếu vô tình dùng nhầm `ColmapLoader` thay `BTSDataset` sẽ lỗi runtime khó
   hiểu. Đã loại bỏ khỏi pipeline chính (không dùng trong `train.py`); nếu bạn
   cần lại pycolmap-based loader, nên viết adapter chuyển `Scene` → `Camera`
   trước khi dùng, thay vì import trực tiếp.

## 🟡 Lỗi nhỏ

9. **`GaussianModel.create_from_pcd`**: dòng `features[:, 3:, 1:] = 0.0` là
   no-op (chiều dim=1 chỉ có size 3, index `3:` luôn rỗng) — không sai kết quả
   (vì đã `torch.zeros` sẵn) nhưng gây hiểu nhầm ý đồ code. Đã sửa lại đúng ý
   định gốc: chỉ set DC term (`features[:, :, 0]`), các bậc SH cao hơn giữ 0.
   Đồng thời `opacity_init`/`scale_init_factor` giờ được truyền từ
   `model.init.opacity` / `model.init.scale_factor` trong config thay vì
   hard-code trong hàm.

10. **`bts_dataset.py._split_train_eval`**: cách chia cũ dùng
    `range(0, len, step)` có thể sinh ra số lượng eval camera lệch khá nhiều
    so với `ratio` khai báo do làm tròn số nguyên của `step`. Đã đổi sang
    `np.linspace` để chọn đúng `n_eval` chỉ số phân bố đều.

11. **`renderer/gaussian_renderer.py`**: comment cũ tự đánh dấu nghi ngờ
    ("gsplat dùng cam->world? kiểm tra convention"). Đã viết rõ lý do
    transpose 2 lần (huỷ transpose của `world_view_transform` gốc rồi áp lại
    đúng convention gsplat cần) để người sau không xoá nhầm dòng này.

12. **`utils/camera_utils.Camera`**: thêm validate `width`/`height` bắt buộc
    khi `image=None` (trường hợp target/novel view) để báo lỗi rõ ràng thay
    vì crash mơ hồ ở bước dựng ma trận.

13. **`utils/logger.Logger`**: thêm `close()` để đóng file log / TensorBoard
    writer đúng cách khi training kết thúc (train.py đã gọi `logger.close()`).

## Không đổi (đã đúng từ trước, chỉ giữ nguyên)

- Toàn bộ công thức toán: `losses/loss.py` (L1+SSIM), `utils/geometry.py`
  (build_rotation/build_scaling_rotation), `utils/sh_utils.py`,
  `preprocessing/colmap_utils.py` (đọc COLMAP text format), thuật toán
  densify/clone/split/prune trong `trainers/trainer.py`.
- Kiến trúc tổng thể (`docs/architecture.md`) và luồng
  Dataset → Preprocessing → GaussianModel → Renderer → Trainer → Inference.

## Cách kiểm tra nhanh sau khi sửa

```bash
pip install -r requirements.txt
python -c "from utils.config_loader import load_config; c = load_config('configs/default.yaml'); print(c.keys())"
```
Lệnh trên phải chạy không lỗi và in ra đầy đủ các section
(`dataset`, `preprocessing`, `model`, `renderer`, `loss`, `evaluation`,
`training`, `optimizer`, `sh`, `densify`, `pruning`, `inference`,
`experiment_name`, `seed`, `device`).
