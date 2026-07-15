# Chạy trên Kaggle Notebook

## 1. Add dataset

Trong notebook Kaggle, bấm **"+ Add Input"** → chọn dataset chứa
`images/` + `sparse/` (+ `target_poses.json` nếu có). Kaggle sẽ mount
read-only tại `/kaggle/input/<tên-slug-dataset>/...`.

Tìm đúng đường dẫn:
```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    if "images" in dirs and "sparse" in dirs:
        print("Tìm thấy scene tại:", root)
```

## 2. Sửa `configs/dataset.kaggle.yaml`

Mở file, sửa `dataset.root` cho khớp path in ra ở bước 1:
```yaml
dataset:
  root: "/kaggle/input/<tên-slug-dataset>/train"   # <-- sửa dòng này
```

Nếu dataset của bạn KHÔNG có sẵn `sparse/0` (chỉ có ảnh thô + cần tự chạy
COLMAP) — Kaggle **không hỗ trợ** tự chạy COLMAP CLI theo cách repo này
thiết kế (vì `/kaggle/input` read-only và COLMAP không có sẵn trên image
Kaggle mặc định). Cách xử lý: chạy COLMAP cục bộ trên máy bạn trước, hoặc
cài COLMAP qua `apt-get` trong notebook rồi tự copy sparse reconstruction
ra `/kaggle/working/` và trỏ `dataset.root` vào đó thay vì `/kaggle/input`.

## 3. Cài dependency còn thiếu

Image Kaggle GPU đã có sẵn torch/numpy/pillow/opencv. Cần cài thêm:
```python
!pip install -q gsplat lpips plyfile imageio imageio-ffmpeg
```

## 4. Chạy training

```python
!cd /kaggle/working/<tên-thư-mục-repo> && python train.py --config configs/default.kaggle.yaml
```

(Hoặc `%cd` vào thư mục repo trước rồi chạy `!python train.py ...`.)

Toàn bộ checkpoint / log / novel views sẽ được ghi vào
`/kaggle/working/outputs/bts_scene_01/` (do `output_root` trong
`configs/default.kaggle.yaml` đã trỏ sẵn về đây) — đây là thư mục duy nhất
Kaggle cho phép ghi và cũng là nơi bạn tải kết quả về sau khi chạy xong
(qua tab "Output" của notebook).

## 5. Giới hạn cần lưu ý trên Kaggle

- **Thời gian chạy**: notebook Kaggle GPU thường giới hạn 9-12 giờ/phiên.
  Với `training.iterations: 30000` (mặc định), 1 scene 3DGS thường mất
  10-30 phút trên GPU T4/P100 — kiểm tra `eval_interval`/`save_interval`
  trong `configs/train.yaml` để có checkpoint trung gian nếu phiên bị ngắt.
- **Dung lượng `/kaggle/working`**: thường giới hạn ~20GB — nếu train nhiều
  scene liên tiếp, dọn bớt checkpoint cũ (`outputs/<exp>/checkpoints/iter_*.pth`)
  giữa các lần chạy, chỉ giữ `last.pth`.
- **GPU**: đảm bảo notebook Settings → Accelerator đã chọn GPU (T4x2/P100),
  nếu không `device: "cuda"` trong config sẽ tự fallback về CPU (train.py đã
  có sẵn cảnh báo khi không có CUDA) và sẽ RẤT chậm với 3DGS.
