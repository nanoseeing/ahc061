from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _decode_scalar(raw: str) -> Any:
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except Exception:
        pass
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def set_by_dotted_key(obj: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = obj
    parts = dotted_key.split(".")
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got: {ov}")
        k, raw = ov.split("=", 1)
        set_by_dotted_key(out, k.strip(), _decode_scalar(raw.strip()))
    return out


def load_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    suf = p.suffix.lower()
    if suf == ".json":
        obj = json.loads(text)
    elif suf in (".toml", ".tml"):
        import tomllib

        obj = tomllib.loads(text)
    elif suf in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML (`pip install pyyaml`)") from e
        obj = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported config extension: {suf} (supported: .json/.toml/.yaml)")
    if not isinstance(obj, dict):
        raise ValueError("top-level config must be an object")
    return obj


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")

