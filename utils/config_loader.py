"""
Config loader: reads the root config file (e.g. configs/default.yaml), deep-
merges the sub-files listed under the `include` key (paths relative to the
root file's directory), then merges the root file's own keys on top (so a
few values can be quickly overridden right inside default.yaml if needed).

Merge order: include[0] -> include[1] -> ... -> include[-1] -> (root file's own keys)
A later include file overwrites matching keys from an earlier one.
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

    # Any remaining keys in the root file (experiment_name, seed, device, or
    # manual overrides).
    merged = _deep_merge(merged, root_cfg)
    return merged


def cfg_get(cfg: dict, dotted_path: str, default=None):
    """Safely access a nested key via a dotted path, e.g.:
    cfg_get(cfg, "optimizer.position.lr_init")
    Avoids repeating cfg["optimizer"]["position"]["lr_init"] everywhere and
    avoids a KeyError when an optional field isn't declared in the yaml.
    """
    node = cfg
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
