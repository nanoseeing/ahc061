"""`builder` に関するモデル処理。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn

from ...utils.config_utils import deep_merge, load_config_file
from .registry import resolve_model_class


_RESERVED_KEYS = {
    "type",
    "class_name",
    "target",
    "kwargs",
}


def _deepcopy_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """内部ヘルパー: `deepcopy_cfg` を実行する。

    Args:
        cfg (Mapping[str, Any]): 設定オブジェクト。

    Returns:
        dict[str, Any]: 計算結果。
    """
    return copy.deepcopy(dict(cfg))


def _unwrap_model_key(cfg: dict[str, Any]) -> dict[str, Any]:
    """内部ヘルパー: `unwrap_model_key` を実行する。

    Args:
        cfg (dict[str, Any]): 設定オブジェクト。

    Returns:
        dict[str, Any]: 計算結果。
    """
    if "model" in cfg and isinstance(cfg["model"], dict):
        top_reserved = _RESERVED_KEYS.intersection(cfg.keys())
        if not top_reserved:
            return _deepcopy_cfg(cfg["model"])
    return cfg


def _extract_type(cfg: dict[str, Any], default_type: str) -> str:
    """内部ヘルパー: `extract_type` を実行する。

    Args:
        cfg (dict[str, Any]): 設定オブジェクト。
        default_type (str): default_type の値。

    Returns:
        str: 計算結果。
    """
    for key in ("type", "class_name", "target"):
        v = cfg.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return str(default_type).strip() or "DiscreteBoardAgent"


def normalize_model_config(
    model_config: Mapping[str, Any] | None,
    *,
    default_type: str = "DiscreteBoardAgent",
) -> dict[str, Any]:
    """`model_config`を正規化する。

    Args:
        model_config (Mapping[str, Any] | None): model_config の値。
        default_type (str): default_type の値。

    Returns:
        dict[str, Any]: 計算結果。
    """

    if model_config is None:
        return {
            "type": str(default_type).strip() or "DiscreteBoardAgent",
            "kwargs": {},
        }

    if not isinstance(model_config, Mapping):
        raise TypeError(f"model config must be a dict-like object, got: {type(model_config)}")

    cfg = _unwrap_model_key(_deepcopy_cfg(model_config))
    if not isinstance(cfg, dict):
        raise TypeError("model config must resolve to a dict")

    type_name = _extract_type(cfg, default_type)

    kwargs_raw = cfg.get("kwargs", {})
    if kwargs_raw is None:
        kwargs_raw = {}
    if not isinstance(kwargs_raw, Mapping):
        raise TypeError("model config 'kwargs' must be a dict")

    kwargs = _deepcopy_cfg(kwargs_raw)
    for k, v in cfg.items():
        if k in _RESERVED_KEYS:
            continue
        if k not in kwargs:
            kwargs[k] = copy.deepcopy(v)

    return {
        "type": type_name,
        "kwargs": kwargs,
    }


def build_agent(
    *,
    obs_shape: tuple[int, ...],
    action_dim: int,
    model_config: Mapping[str, Any] | None,
    default_type: str = "DiscreteBoardAgent",
) -> tuple[nn.Module, dict[str, Any]]:
    """`agent`を構築する。

    Args:
        obs_shape (tuple[int, ...]): obs_shape の値。
        action_dim (int): action_dim の値。
        model_config (Mapping[str, Any] | None): model_config の値。
        default_type (str): default_type の値。

    Returns:
        tuple[nn.Module, dict[str, Any]]: 計算結果。
    """

    resolved = normalize_model_config(
        model_config,
        default_type=default_type,
    )
    cls = resolve_model_class(resolved["type"])
    kwargs = dict(resolved.get("kwargs", {}))

    agent = cls(obs_shape=obs_shape, action_dim=int(action_dim), **kwargs)

    # Keep canonical build config on the instance for checkpoint reproducibility.
    if hasattr(agent, "model_config") and isinstance(getattr(agent, "model_config"), Mapping):
        resolved = normalize_model_config(getattr(agent, "model_config"), default_type=resolved["type"])
    else:
        setattr(agent, "model_config", copy.deepcopy(resolved))

    return agent, resolved


def load_model_config_from_sources(
    *,
    model_config_file: str | Path | None,
    model_config_json: str | None,
) -> dict[str, Any] | None:
    """`model_config_from_sources`を読み込む。

    Args:
        model_config_file (str | Path | None): 対象パス。
        model_config_json (str | None): model_config_json の値。

    Returns:
        dict[str, Any] | None: 計算結果。
    """

    cfg: dict[str, Any] | None = None

    if model_config_file is not None and str(model_config_file).strip():
        raw = load_config_file(model_config_file)
        if not isinstance(raw, dict):
            raise TypeError("model config file must load to a dict")
        cfg = _unwrap_model_key(dict(raw))

    if model_config_json is not None and str(model_config_json).strip():
        raw_json = json.loads(str(model_config_json))
        if not isinstance(raw_json, dict):
            raise TypeError("--model-config-json must decode to a JSON object")
        raw_json = _unwrap_model_key(dict(raw_json))
        cfg = raw_json if cfg is None else deep_merge(cfg, raw_json)

    if cfg is None:
        return None
    return cfg
