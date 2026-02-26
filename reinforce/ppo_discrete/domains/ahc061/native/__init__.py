from __future__ import annotations

import contextlib
import importlib
import sys
import threading
from pathlib import Path
from types import ModuleType

_MODULE_NAME = "reinforce.ppo_discrete.domains.ahc061.native._opponent_bayes_cpp"
_FALLBACK_LOCK = threading.Lock()


@contextlib.contextmanager
def _build_lock():
    lock_path = Path(__file__).resolve().parent / ".opponent_bayes_cpp.build.lock"
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
    if force_build:
        with _build_lock():
            from .build_opponent_bayes_cpp import build_extension

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

        from .build_opponent_bayes_cpp import build_extension

        build_extension(force=force_build, verbose=verbose)
        importlib.invalidate_caches()
        return importlib.import_module(_MODULE_NAME)


def ensure_cpp_backend(*, build_if_missing: bool = True, force_build: bool = False, verbose: bool = False) -> bool:
    try:
        load_cpp_backend(build_if_missing=build_if_missing, force_build=force_build, verbose=verbose)
        return True
    except Exception:
        return False
