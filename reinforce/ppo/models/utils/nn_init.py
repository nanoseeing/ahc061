"""ニューラルネットワーク初期化に関する補助関数。"""
from __future__ import annotations

from math import sqrt
from typing import Any, Iterable

import torch
import torch.nn as nn


def to_int_tuple(v: Any) -> tuple[int, ...]:
    """多様な入力を `tuple[int, ...]` に正規化する。

    Args:
        v (Any): 変換対象。`None`、シーケンス、カンマ区切り文字列を受け付ける。

    Returns:
        tuple[int, ...]: 整数タプル。

    Raises:
        TypeError: サポート外の型が渡された場合。
    """
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
    """レイヤー重みを直交初期化し、バイアスを定数で初期化する。

    Args:
        layer (nn.Module): 初期化対象レイヤー。
        std (float): 直交初期化時のスケール係数。
        bias_const (float): バイアス初期値。

    Returns:
        nn.Module: 初期化済みレイヤー（入力オブジェクトをそのまま返す）。
    """
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
