from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "PPOConfig": ("config", "PPOConfig"),
    "FlattenedBatch": ("rollout_buffer", "FlattenedBatch"),
    "RolloutBuffer": ("rollout_buffer", "RolloutBuffer"),
    "PPOTrainer": ("trainer", "PPOTrainer"),
    "UpdateStats": ("trainer", "UpdateStats"),
    "PPOScheduleSet": ("train_utils", "PPOScheduleSet"),
    "ScalarSchedule": ("train_utils", "ScalarSchedule"),
    "parse_schedule_expr": ("train_utils", "parse_schedule_expr"),
    "resolve_vecnorm_gamma": ("train_utils", "resolve_vecnorm_gamma"),
    "schedule_progress": ("train_utils", "schedule_progress"),
    "validate_ppo_config": ("train_utils", "validate_ppo_config"),
    "validate_schedule_args": ("train_utils", "validate_schedule_args"),
    "validate_vecnorm_config": ("train_utils", "validate_vecnorm_config"),
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
