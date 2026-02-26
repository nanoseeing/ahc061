from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "PPOConfig": ("config", "PPOConfig"),
    "FlattenedBatch": ("rollout_buffer", "FlattenedBatch"),
    "RolloutBuffer": ("rollout_buffer", "RolloutBuffer"),
    "PPOTrainer": ("trainer", "PPOTrainer"),
    "UpdateStats": ("trainer", "UpdateStats"),
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
