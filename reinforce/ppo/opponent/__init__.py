"""`__init__` に関する相手モデル処理。"""
from __future__ import annotations

import contextlib
import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType

_MODULE_NAME = "reinforce.ppo.opponent._opponent_bayes_cpp"
_FALLBACK_LOCK = threading.Lock()


@contextlib.contextmanager
def _build_lock():
    # opponent/ -> ppo/ -> reinforce/
    """内部ヘルパー: `build_lock` を実行する。"""
    reinforce_dir = Path(__file__).resolve().parents[2]
    lock_path = reinforce_dir / "build" / "ahc061_cpp_bayes" / "opponent_bayes_cpp.build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("w", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            with _FALLBACK_LOCK:
                yield
    finally:
        fh.close()


def load_cpp_backend(*, build_if_missing: bool = True, force_build: bool = False, verbose: bool = False) -> ModuleType:
    """`cpp_backend`を読み込む。

    Args:
        build_if_missing (bool): 有効化フラグ。
        force_build (bool): 有効化フラグ。
        verbose (bool): 有効化フラグ。

    Returns:
        ModuleType: 計算結果。
    """
    if force_build:
        with _build_lock():
            from .build_ext import build_extension

            build_extension(force=True, verbose=verbose)
            importlib.invalidate_caches()
            sys.modules.pop(_MODULE_NAME, None)
            return importlib.import_module(_MODULE_NAME)

    try:
        return importlib.import_module(_MODULE_NAME)
    except Exception:
        if not build_if_missing:
            raise

    with _build_lock():
        try:
            return importlib.import_module(_MODULE_NAME)
        except Exception:
            pass

        from .build_ext import build_extension

        build_extension(force=force_build, verbose=verbose)
        importlib.invalidate_caches()
        return importlib.import_module(_MODULE_NAME)


def ensure_cpp_backend(*, build_if_missing: bool = True, force_build: bool = False, verbose: bool = False) -> bool:
    """`cpp_backend`を保証する。

    Args:
        build_if_missing (bool): 有効化フラグ。
        force_build (bool): 有効化フラグ。
        verbose (bool): 有効化フラグ。

    Returns:
        bool: 計算結果。
    """
    try:
        load_cpp_backend(build_if_missing=build_if_missing, force_build=force_build, verbose=verbose)
        return True
    except Exception:
        return False
