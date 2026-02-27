from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "RunLayout": ("experiment", "RunLayout"),
    "ScalarSummary": ("metrics", "ScalarSummary"),
    "MetricTracker": ("tracking", "MetricTracker"),
    "save_checkpoint_payload": ("checkpoint", "save_checkpoint_payload"),
    "load_checkpoint_payload": ("checkpoint", "load_checkpoint_payload"),
    "deep_merge": ("config_utils", "deep_merge"),
    "set_by_dotted_key": ("config_utils", "set_by_dotted_key"),
    "apply_overrides": ("config_utils", "apply_overrides"),
    "load_config_file": ("config_utils", "load_config_file"),
    "save_json": ("config_utils", "save_json"),
    "create_run_layout": ("experiment", "create_run_layout"),
    "make_run_name": ("experiment", "make_run_name"),
    "resolve_config": ("experiment", "resolve_config"),
    "to_jsonable": ("experiment", "to_jsonable"),
    "coerce_optional_path": ("experiment", "coerce_optional_path"),
    "update_manifest": ("experiment", "update_manifest"),
    "get_logger": ("log_utils", "get_logger"),
    "summarize": ("metrics", "summarize"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(name)
    module_name, attr_name = spec
    mod = import_module(f"{__name__}.{module_name}")
    return getattr(mod, attr_name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
