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

## 🔵 Bổ sung sau kiểm thử tích hợp (integration test) lần 2

14. **`read_images_text()` lọc bỏ luôn cả dòng trống**, không chỉ dòng
    comment. Trong `images.txt` chuẩn COLMAP, mỗi ảnh chiếm đúng 2 dòng
    (metadata + điểm 2D), và dòng điểm 2D CÓ THỂ RỖNG nếu ảnh không có
    keypoint nào khớp track. Cách lọc cũ (`[l for l in f if not l.startswith("#")
    and len(l.strip()) > 0]` rồi bắt cặp `range(0, len(lines), 2)`) sẽ làm
    lệch toàn bộ cặp dòng ngay khi gặp ảnh đầu tiên có dòng điểm 2D rỗng,
    khiến các ảnh phía sau trong file bị đọc sai id/pose (số ảnh đọc được có
    thể giảm gần một nửa, tuỳ dataset). Đã sửa lại đọc tuần tự `readline()`
    từng dòng, khớp đúng cách COLMAP chính thức đọc file này. **Đây là bug
    có khả năng ảnh hưởng dữ liệu thật cao nhất trong đợt sửa này** - nếu
    dataset thi đấu của bạn từng "chạy được nhưng thiếu ảnh train" thì rất
    có thể là do bug này.

15. **`run_colmap_pipeline()`**: kiểm tra `colmap_exe` có tồn tại trong PATH
    trước khi gọi `subprocess.run(...)`, báo lỗi hướng dẫn cụ thể (cài COLMAP
    / sửa `preprocessing.colmap.executable` / kiểm tra `dataset.root`) thay vì
    để `subprocess` ném `FileNotFoundError: WinError 2` khó hiểu.

16. **`_load_colmap()`**: tự dò `sparse/0/` hoặc `sparse/` (fallback không có
    thư mục con `0`), và chỉ tự động chạy COLMAP khi
    `preprocessing.colmap.enabled: true` được bật tường minh - trước đây nó
    tự chạy bất cứ khi nào không thấy sẵn sparse reconstruction, kể cả khi
    người dùng chỉ gõ sai đường dẫn `dataset.root`.

17. **`load_colmap_scene()` giờ hỗ trợ cả `.bin` (mặc định COLMAP) lẫn `.txt`**,
    tự phát hiện định dạng. Trước đây chỉ đọc được `.txt`, trong khi hầu hết
    dataset SfM thật xuất ra `.bin`. Đã roundtrip-test `read_cameras_binary`,
    `read_images_binary`, `read_points3D_binary` với dữ liệu nhị phân giả lập
    khớp đúng spec COLMAP gốc.

## Đã kiểm thử tích hợp (integration test)

Repo đã được test end-to-end trên dataset COLMAP giả lập (10 ảnh, point cloud
200 điểm, 3 target novel views) và dataset `nerf_transforms` giả lập (6 ảnh),
bao gồm: load dataset → train/eval split đúng tỉ lệ → normalize_scene đúng
(camera centers nằm trong bán kính ~1) → khởi tạo GaussianModel từ point cloud
lẫn từ random init → build Trainer (tôn trọng `model.optimize.*`, báo lỗi rõ
ràng khi cấu hình xung đột với densify) → chạy thử densify/prune/reset_opacity
(shape các thuộc tính Gaussian luôn đồng bộ sau resize) → save/restore
checkpoint → save .ply → evaluate_dataset với render function giả (PSNR/SSIM).
Toàn bộ chạy đúng, không lỗi shape/KeyError nào phát sinh.

## 🟢 Bổ sung sau khi chạy trên dataset thi đấu thật (lần 3)

18. **`images.txt` trong sparse reconstruction tham chiếu ảnh KHÔNG tồn tại
    trong thư mục `images/`** (dataset thi đấu có thể thiếu 1 vài ảnh so với
    reconstruction gốc, hoặc lệch hoa/thường phần mở rộng, vd `.JPG` vs
    `.jpg`). Trước đây 1 ảnh thiếu sẽ làm crash TOÀN BỘ pipeline ngay từ đầu
    (`PILImage.open()` → `FileNotFoundError`), dù 99% ảnh còn lại vẫn dùng
    được. Đã sửa `_load_colmap()`:
    - Tự dò vài biến thể phần mở rộng phổ biến (`.jpg/.JPG/.jpeg/.JPEG/.png/.PNG`)
      trước khi coi là thiếu hẳn.
    - Ảnh nào thực sự không tìm thấy sẽ bị **bỏ qua** (không đưa vào
      train/eval), kèm cảnh báo in ra danh sách ảnh thiếu - không phải lỗi
      code, chỉ là cảnh báo dữ liệu.
    - Chỉ raise lỗi cứng nếu **toàn bộ** ảnh trong reconstruction đều thiếu
      (khi đó gần như chắc chắn `dataset.images.directory` trỏ sai).
    Đã test với dataset giả lập thiếu 2/10 ảnh + 1 ảnh lệch hoa/thường phần mở
    rộng: pipeline load đúng 8 ảnh còn lại, tự dò ra bản `.JPG`, không crash.
    Cùng cơ chế bỏ-qua-ảnh-thiếu này cũng đã được áp dụng cho
    `_load_nerf_transforms()` (trước đó chỉ có ở `_load_colmap()`), test với
    dataset `transforms.json` giả lập thiếu 1/6 frame: load đúng 5 frame còn
    lại.



```bash
pip install -r requirements.txt
python -c "from utils.config_loader import load_config; c = load_config('configs/default.yaml'); print(c.keys())"
```
Lệnh trên phải chạy không lỗi và in ra đầy đủ các section
(`dataset`, `preprocessing`, `model`, `renderer`, `loss`, `evaluation`,
`training`, `optimizer`, `sh`, `densify`, `pruning`, `inference`,
`experiment_name`, `seed`, `device`).

## 🟢 Hỗ trợ target views dạng CSV (lần 6)

26. **Thêm hỗ trợ đọc `target_poses.csv`** bên cạnh `.json` sẵn có -
    `_load_target_views()` tự nhận diện định dạng qua đuôi file
    (`dataset.target_views.file`). Hàm `_read_target_views_csv()` mới tự
    động dò tên cột (không phân biệt hoa/thường, dấu gạch dưới, khoảng
    trắng) qua nhiều alias phổ biến: tên ảnh (`name`/`image_name`/`id`/...),
    rotation (quaternion `qw,qx,qy,qz` HOẶC ma trận phẳng `r00..r22`),
    translation (`tx,ty,tz` hoặc `x,y,z`), field of view (`fovx,fovy` trực
    tiếp HOẶC `fx,fy` + `width/height` để tự tính). Nếu thiếu cột bắt buộc,
    raise lỗi liệt kê rõ nhóm cột còn thiếu + toàn bộ cột thực tế phát hiện
    được trong file, dễ đối chiếu/báo lại để bổ sung alias. Target view đọc
    từ CSV đi qua đúng cùng code path với JSON (cùng áp dụng normalize_scene,
    cùng validate width/height) - không có logic riêng biệt/trùng lặp. Đã
    test: 2 biến thể tên cột (quaternion+FoV trực tiếp, ma trận+fx/fy), báo
    lỗi khi thiếu cột, và tích hợp đầy đủ qua `BTSDataset` với
    `target_views.file` trỏ vào `.csv`.

## 🔴 Sửa lỗi render novel view bị vỡ (camera pose convention) (lần 7)

27. **Render target novel view ra "đám ellipsoid vỡ nát, camera như đang ở
    trong point cloud"** dù eval PSNR trên train/eval bình thường (~18-21) -
    cô lập được lỗi chỉ nằm ở target view (đọc từ CSV thi đấu), không phải
    do train chưa đủ. Nguyên nhân: `(R, T)` trong file CSV rất có thể theo
    convention **camera->world** (R = rotation cam->world, T = vị trí
    camera trong world - phổ biến ở nhiều công cụ log pose drone), trong
    khi toàn bộ codebase (giống COLMAP) luôn giả định **world->cam**. Dùng
    sai convention khiến vị trí/hướng camera bị tính hoàn toàn sai, camera
    kết thúc ở gần/bên trong point cloud thay vì đúng khoảng cách thật.
    Đã thêm:
    - Config mới `dataset.target_views.pose_convention` (`"world_to_cam"`
      mặc định - không đổi hành vi cũ, hoặc `"cam_to_world"`).
    - Hàm `_to_world_to_cam()` tự chuyển đổi đúng công thức
      (`R_w2c = R^T`, `T_w2c = -R^T @ T`) khi chọn `cam_to_world`.
    - `_warn_if_target_views_look_misplaced()`: so sánh khoảng cách trung
      bình từ target camera tới tâm scene (đã normalize) với train camera -
      nếu lệch >5x hoặc <0.2x, in cảnh báo gợi ý đổi `pose_convention` NGAY
      lúc load dataset, tránh phải chờ hết cả training mới phát hiện render
      bị vỡ. Đã test: convention đúng cho vị trí camera hợp lý, convention
      sai với lệch lớn (mô phỏng đúng tình huống thực tế) trigger đúng cảnh
      báo, còn JSON target views mặc định `world_to_cam` không đổi hành vi.

## 🔴 Sửa memory leak nghiêm trọng trong Adam optimizer (lần 8)

28. **VRAM tăng dần đều rồi tràn giữa chừng training (crash ở iter ~9800/30000,
    khi `n_gaussians` vẫn ổn định ~148k, thậm chí đang giảm do prune)** -
    không liên quan `dataset.images.resolution` hay `max_gaussians`. Nguyên
    nhân: `_append_points()`, `_prune_points()`, `reset_opacity()` trong
    `trainers/trainer.py` thay `group["params"][0]` bằng **object
    `nn.Parameter` HOÀN TOÀN MỚI** mỗi lần densify/prune/reset, nhưng
    `torch.optim.Adam` lưu momentum (`exp_avg`, `exp_avg_sq`) trong
    `self.optimizer.state`, một dict **đánh index theo chính object
    Parameter**. Không xoá entry cũ trước khi thay Parameter khiến state
    (2 tensor CUDA cùng shape với Parameter tại thời điểm đó) bị **mồ côi
    vĩnh viễn** - không còn được dùng nhưng không bao giờ được giải phóng,
    vì `self.optimizer.state` vẫn giữ tham chiếu tới nó. `densify_and_prune`
    chạy ~93 lần trước khi crash (mỗi 100 iter từ iter 500) → tích luỹ đủ
    rác để tràn VRAM dù model đang hoạt động bình thường.

    Đã sửa theo đúng pattern repo gốc `graphdeco-inria/gaussian-splatting`
    (`cat_tensors_to_optimizer`/`_prune_optimizer`/`replace_tensor_to_optimizer`):
    trước khi thay Parameter, lấy state cũ qua
    `self.optimizer.state.get(old_param)`, resize đúng theo phép biến đổi
    tương ứng (`torch.cat` với zero cho điểm mới ở `_append_points`, index
    bằng `valid` mask ở `_prune_points`, reset về 0 ở `reset_opacity` vì giá
    trị vừa nhảy bất liên tục), `del` entry cũ, rồi gán lại state đã migrate
    cho Parameter mới.

    **Đã fix thêm 1 lỗi tự gây ra trong lúc sửa**: bản sửa đầu tiên vô tình
    làm mất dòng `setattr(self.gaussians, ..., group["params"][0])` trong
    `_prune_points()` (đồng bộ attribute `_xyz/_opacity/...` của
    `GaussianModel` với Parameter mới) - phát hiện và sửa ngay trong cùng
    lượt trước khi test.

    **Đã verify bằng test đo trực tiếp bộ nhớ**, không chỉ test shape:
    - Đếm tổng `numel()` của mọi tensor `exp_avg`/`exp_avg_sq` trong
      `optimizer.state`, so với giá trị kỳ vọng (= 2× số phần tử mỗi
      Parameter hiện tại) qua 10 chu kỳ `densify_and_prune` +
      `reset_opacity` liên tiếp - khớp chính xác 100%, chênh lệch = 0.
    - Kiểm tra không còn object Parameter mồ côi nào trong
      `optimizer.state.keys()` (so khớp `id()` với params hiện tại).
    - **Verify ngược bằng chính code cũ (có bug)** để xác nhận phương pháp
      test thực sự phát hiện được lỗi: chỉ 1 lần densify+prune khiến state
      tăng từ 23,600 lên 102,306 phần tử (>4 lần), với đúng 6 Parameter mồ
      côi còn sót (khớp 6 param-group) - xác nhận cả bug lẫn cách test đều
      đúng, không phải false positive.

## 🟢 Clamp anisotropy - khắc phục Gaussian dạng "kim" gây vệt sọc (lần 9)

29. **Render novel view bị vệt sọc dài sáng/xám bất thường** (khác hẳn 2 lỗi
    trước đó - vỡ nát do pose sai, hoặc đen do alpha thấp) - dấu hiệu kinh
    điển của Gaussian bị optimizer kéo thành hình "kim" (1 trục rất dài, 2
    trục còn lại gần như bằng 0). Từ góc nhìn khác hẳn train view (như target
    novel view thường gặp), silhouette của 1 quả kim thay đổi cực mạnh theo
    góc nhìn, chiếu lên màn hình thành vệt sọc dài thay vì 1 đốm mờ bình
    thường.

    Đã thêm `Trainer._clamp_anisotropy(max_ratio)`: sau MỖI
    `optimizer.step()`, kéo trục ngắn nhất của mỗi Gaussian lên tối thiểu
    bằng `trục dài nhất / max_ratio` (không đụng tới trục dài nhất - không
    giới hạn kích thước bề mặt phẳng lớn hợp lệ như mái nhà, chỉ ngăn trục
    ngắn co lại quá mức so với trục dài). Áp dụng trực tiếp lên `.data` dưới
    `no_grad`, KHÔNG tạo `nn.Parameter` mới nên không đụng tới
    `optimizer.state` (không lặp lại bug leak ở mục 28) - đã verify bằng
    test so sánh state Adam trước/sau clamp giống hệt nhau.

    Config mới `densify.max_anisotropy` (mặc định `10.0`, đặt `null`/`0` để
    tắt). Đã test: Gaussian tỉ lệ 10000:1 bị kéo đúng về 10:1 (trục lớn nhất
    giữ nguyên), Gaussian đẳng hướng hoặc dưới ngưỡng không bị đổi, và chạy
    full training loop (mock renderer) với clamp bật không lỗi.
