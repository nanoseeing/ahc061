from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .config_utils import apply_overrides, deep_merge, load_config_file, save_json

_RUN_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RunLayout:
    root: Path
    config_dir: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    reports_dir: Path
    artifacts_dir: Path

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


def _safe_name(name: str) -> str:
    return _RUN_SAFE.sub("_", name).strip("_")


def make_run_name(prefix: str, *, seed: int | None = None, now: float | None = None) -> str:
    now_sec = now if now is not None else time.time()
    ts_ms = int(now_sec * 1000)
    pid = os.getpid()
    out = _safe_name(prefix)
    if seed is not None:
        out += f"__seed{int(seed)}"
    out += f"__{ts_ms}__p{pid}"
    return out


def create_run_layout(run_root: str | Path, run_name: str) -> RunLayout:
    root = Path(run_root) / run_name
    layout = RunLayout(
        root=root,
        config_dir=root / "config",
        data_dir=root / "data",
        models_dir=root / "models",
        logs_dir=root / "logs",
        reports_dir=root / "reports",
        artifacts_dir=root / "artifacts",
    )
    for p in (
        layout.root,
        layout.config_dir,
        layout.data_dir,
        layout.models_dir,
        layout.logs_dir,
        layout.reports_dir,
        layout.artifacts_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)
    return layout


def update_manifest(layout: RunLayout, patch: dict[str, Any]) -> dict[str, Any]:
    path = layout.root / "manifest.json"
    cur: dict[str, Any] = {}
    if path.exists():
        try:
            cur_obj = load_config_file(path)
            if isinstance(cur_obj, dict):
                cur = cur_obj
        except Exception:
            cur = {}
    merged = deep_merge(cur, to_jsonable(patch))
    save_json(path, merged)
    return merged


def resolve_config(
    *,
    defaults: dict[str, Any],
    config_file: str | Path | None,
    config_section: str | None,
    overrides: list[str],
) -> dict[str, Any]:
    cfg = defaults
    if config_file:
        loaded = load_config_file(config_file)
        sectioned = loaded
        if config_section:
            obj = loaded.get(config_section)
            if obj is None:
                raise ValueError(f"config section not found: {config_section}")
            if not isinstance(obj, dict):
                raise ValueError(f"config section must be an object: {config_section}")
            sectioned = obj
        if not isinstance(sectioned, dict):
            raise ValueError("config must be an object")
        cfg = deep_merge(cfg, sectioned)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if is_dataclass(x):
        return to_jsonable(asdict(x))
    return str(x)


def coerce_optional_path(x: Any, *, dot_is_none: bool = False) -> Path | None:
    if x is None:
        return None
    if isinstance(x, Path):
        return None if dot_is_none and x == Path(".") else x
    s = str(x).strip()
    if not s:
        return None
    p = Path(s)
    if dot_is_none and p == Path("."):
        return None
    return p
