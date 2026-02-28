"""`runtime` に関するユーティリティ。"""
from __future__ import annotations

import json
from typing import Any

import torch


def choose_device(name: str) -> torch.device:
    """`device`を選択する。

    Args:
        name (str): name の値。

    Returns:
        torch.device: 計算結果。
    """
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(name))


def parse_json_object(text: str, *, field_name: str) -> dict[str, Any]:
    """`json_object`を解析する。

    Args:
        text (str): text の値。
        field_name (str): field_name の値。

    Returns:
        dict[str, Any]: 計算結果。
    """
    try:
        obj = json.loads(str(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"{field_name} must be valid JSON object: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return obj
