from __future__ import annotations

import importlib
from typing import Any

import torch.nn as nn


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str, cls: type[nn.Module]) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("model name must not be empty")
    prev = _MODEL_REGISTRY.get(key)
    if prev is not None and prev is not cls:
        raise ValueError(f"model '{key}' already registered with different class")
    _MODEL_REGISTRY[key] = cls


def list_registered_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


def _import_class(path: str) -> type[nn.Module]:
    text = str(path).strip()
    if not text:
        raise ValueError("empty model class path")

    if ":" in text:
        mod, attr = text.split(":", 1)
    else:
        if "." not in text:
            raise ValueError(
                f"unknown model type '{text}'. Pass a registered name or full import path like 'pkg.mod:Class'"
            )
        mod, attr = text.rsplit(".", 1)

    module = importlib.import_module(mod)
    cls: Any = getattr(module, attr)
    if not isinstance(cls, type):
        raise TypeError(f"resolved symbol is not a class: {text}")
    if not issubclass(cls, nn.Module):
        raise TypeError(f"resolved class is not nn.Module: {text}")
    return cls


def resolve_model_class(type_name: str) -> type[nn.Module]:
    key = str(type_name).strip()
    if not key:
        raise ValueError("model type must not be empty")
    if key in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[key]
    return _import_class(key)
