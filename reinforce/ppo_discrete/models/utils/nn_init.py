from __future__ import annotations

from math import sqrt
from typing import Any, Iterable

import torch
import torch.nn as nn


def to_int_tuple(v: Any) -> tuple[int, ...]:
    """Convert various sequence types to tuple[int, ...]."""
    if v is None:
        return tuple()
    if isinstance(v, tuple):
        return tuple(int(x) for x in v)
    if isinstance(v, list):
        return tuple(int(x) for x in v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return tuple()
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if isinstance(v, Iterable):
        return tuple(int(x) for x in v)
    raise TypeError(f"unsupported sequence value: {type(v)}")


def layer_init(layer: nn.Module, std: float = sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
