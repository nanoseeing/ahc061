"""`config_utils` に関するユーティリティ。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`deep_merge` を実行する。

    Args:
        base (dict[str, Any]): base の値。
        override (dict[str, Any]): override の値。

    Returns:
        dict[str, Any]: 計算結果。
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_file(path: str | Path) -> dict[str, Any]:
    """`config_file`を読み込む。

    Args:
        path (str | Path): 対象パス。

    Returns:
        dict[str, Any]: 計算結果。
    """
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
        except ImportError as e:
            raise RuntimeError("YAML config requires PyYAML (`pip install pyyaml`)") from e
        obj = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported config extension: {suf} (supported: .json/.toml/.yaml)")
    if not isinstance(obj, dict):
        raise ValueError("top-level config must be an object")
    return obj


def save_json(path: str | Path, obj: Any) -> None:
    """`json`を保存する。

    Args:
        path (str | Path): 対象パス。
        obj (Any): obj の値。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")

