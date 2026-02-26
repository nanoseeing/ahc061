from __future__ import annotations

from typing import Any

import numpy as np


def extract_action_mask(obs: Any, info: dict[str, Any], expected_action_dim: int) -> np.ndarray | None:
    """Extract action-mask from obs/info in a framework-agnostic way.

    Priority:
    1) info["action_mask"]
    2) obs["action_mask"] when observation is dict-like
    """
    if isinstance(info, dict) and "action_mask" in info:
        mask = np.asarray(info["action_mask"], dtype=np.bool_)
        return _normalize_mask(mask, expected_action_dim)

    if isinstance(obs, dict) and "action_mask" in obs:
        mask = np.asarray(obs["action_mask"], dtype=np.bool_)
        return _normalize_mask(mask, expected_action_dim)

    return None


def _normalize_mask(mask: np.ndarray, expected_action_dim: int) -> np.ndarray:
    if mask.ndim == 1:
        if mask.shape[0] != expected_action_dim:
            raise ValueError(f"invalid mask size: {mask.shape[0]} != {expected_action_dim}")
        return mask

    if mask.ndim == 2:
        if mask.shape[1] != expected_action_dim:
            raise ValueError(f"invalid mask width: {mask.shape[1]} != {expected_action_dim}")
        return mask

    raise ValueError(f"action mask must be 1D or 2D, got shape={mask.shape}")
