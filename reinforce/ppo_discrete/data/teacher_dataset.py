from __future__ import annotations

from typing import Any

import numpy as np

from ..game_constants import BAYES_TAIL_SHAPE, OPP_PARAM_DIM, OPP_SLOT_COUNT

DATASET_KEY_OPP_PARAM_TRUE = "opp_param_true"
DATASET_KEY_OPP_VALID = "opp_valid"

BAYES_SOURCE_ENV_INFO = "env_info"
BAYES_SOURCE_ZERO_FALLBACK = "zero_fallback"
BAYES_SOURCE_ZERO_COMPAT = "zero_compat"
BAYES_SOURCE_CPP = "cpp"


def zero_bayes_params(shape: tuple[int, ...] = BAYES_TAIL_SHAPE) -> np.ndarray:
    return np.zeros(tuple(int(x) for x in shape), dtype=np.float32)


def normalize_bayes_params(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def extract_bayes_params_from_info(info: Any) -> tuple[np.ndarray, str]:
    if isinstance(info, dict) and "bayes_params" in info:
        return normalize_bayes_params(info["bayes_params"]), BAYES_SOURCE_ENV_INFO
    return zero_bayes_params(), BAYES_SOURCE_ZERO_FALLBACK


def resolve_bayes_sources_for_meta(sources: set[str]) -> list[str]:
    if not sources:
        return ["none"]
    return sorted(str(s) for s in sources)


def aux_to_teacher_opp_arrays(opp_param: Any, opp_valid: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract enemy-only slices (players 1..7) from aux tensors."""
    p = np.asarray(opp_param, dtype=np.float32)
    v = np.asarray(opp_valid, dtype=np.uint8)
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]
    if p.ndim != 2 or p.shape[1] != 5:
        raise ValueError(f"opp_param shape mismatch: expected [M_MAX,5], got {p.shape}")
    if v.ndim != 1:
        raise ValueError(f"opp_valid shape mismatch: expected [M_MAX], got {v.shape}")
    need = int(OPP_SLOT_COUNT)
    if p.shape[0] < need + 1 or v.shape[0] < need + 1:
        raise ValueError(f"aux player axis too short: opp_param={p.shape}, opp_valid={v.shape}")
    p_enemy = np.asarray(p[1 : 1 + need, :], dtype=np.float32).copy()
    v_enemy = np.asarray(v[1 : 1 + need], dtype=np.uint8).copy()
    return p_enemy, v_enemy
