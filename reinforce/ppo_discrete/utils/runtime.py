from __future__ import annotations

import json
from typing import Any

import torch


def choose_device(name: str) -> torch.device:
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(name))


def parse_json_object(text: str, *, field_name: str) -> dict[str, Any]:
    try:
        obj = json.loads(str(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"{field_name} must be valid JSON object: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return obj
