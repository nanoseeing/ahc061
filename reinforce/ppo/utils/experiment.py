"""`experiment` に関するユーティリティ。"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .config_utils import deep_merge, load_config_file, save_json

_RUN_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RunLayout:
    """`RunLayout` を表すクラス。"""
    root: Path
    config_dir: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    reports_dir: Path
    artifacts_dir: Path

    def as_dict(self) -> dict[str, str]:
        """`as_dict` を実行する。

        Returns:
            dict[str, str]: 計算結果。
        """
        return {k: str(v) for k, v in asdict(self).items()}


def _safe_name(name: str) -> str:
    """内部ヘルパー: `safe_name` を実行する。

    Args:
        name (str): name の値。

    Returns:
        str: 計算結果。
    """
    return _RUN_SAFE.sub("_", name).strip("_")


def make_run_name(prefix: str, *, seed: int | None = None, now: float | None = None) -> str:
    """`run_name`を作成する。

    Args:
        prefix (str): prefix の値。
        seed (int | None): 乱数シード。
        now (float | None): now の値。

    Returns:
        str: 計算結果。
    """
    now_sec = now if now is not None else time.time()
    ts_ms = int(now_sec * 1000)
    pid = os.getpid()
    out = _safe_name(prefix)
    if seed is not None:
        out += f"__seed{int(seed)}"
    out += f"__{ts_ms}__p{pid}"
    return out


def create_run_layout(run_root: str | Path, run_name: str) -> RunLayout:
    """`run_layout`を作成する。

    Args:
        run_root (str | Path): run_root の値。
        run_name (str): run_name の値。

    Returns:
        RunLayout: 計算結果。
    """
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
    """`update_manifest` を実行する。

    Args:
        layout (RunLayout): layout の値。
        patch (dict[str, Any]): patch の値。

    Returns:
        dict[str, Any]: 計算結果。
    """
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


def to_jsonable(x: Any) -> Any:
    """`jsonable`に変換する。

    Args:
        x (Any): 入力テンソル。

    Returns:
        Any: 計算結果。
    """
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
    """`coerce_optional_path` を実行する。

    Args:
        x (Any): 入力テンソル。
        dot_is_none (bool): 有効化フラグ。

    Returns:
        Path | None: 計算結果。
    """
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
