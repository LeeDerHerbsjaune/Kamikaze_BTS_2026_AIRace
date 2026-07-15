"""
Config loader: đọc file config gốc (vd configs/default.yaml), gộp (deep merge)
các file con được liệt kê trong key `include` (đường dẫn tương đối so với
thư mục chứa file gốc), rồi merge đè các key ở file gốc lên trên cùng
(cho phép override nhanh 1 vài giá trị ngay trong default.yaml nếu cần).

Thứ tự merge: include[0] -> include[1] -> ... -> include[-1] -> (key riêng ở file gốc)
File include đứng sau đè key trùng của file đứng trước.
"""
import os
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        root_cfg = yaml.safe_load(f) or {}

    base_dir = os.path.dirname(os.path.abspath(path))
    includes = root_cfg.pop("include", [])

    merged = {}
    for inc_path in includes:
        full_path = inc_path if os.path.isabs(inc_path) else os.path.join(base_dir, inc_path)
        with open(full_path, encoding="utf-8") as f:
            inc_cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, inc_cfg)

    # Các key còn lại trong file gốc (experiment_name, seed, device, hoặc override thủ công)
    merged = _deep_merge(merged, root_cfg)
    return merged


def cfg_get(cfg: dict, dotted_path: str, default=None):
    """Truy cập an toàn 1 key lồng nhau bằng dotted path, vd:
    cfg_get(cfg, "optimizer.position.lr_init")
    Tránh phải viết cfg["optimizer"]["position"]["lr_init"] lặp lại khắp nơi
    và tránh KeyError khi 1 field optional chưa được khai báo trong yaml.
    """
    node = cfg
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
