"""モデルチェックポイントの保存・読込を担うサービス層。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import build_agent, normalize_model_config
from ..models.nets.discrete_board import DiscreteBoardAgent
from ..utils.checkpoint import load_checkpoint_payload, save_checkpoint_payload


def save_agent_checkpoint(
    path: str | Path,
    agent: DiscreteBoardAgent,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """エージェント状態を再現可能なチェックポイントとして保存する。

    Args:
        path (str | Path): 出力先パス。
        agent (DiscreteBoardAgent): 保存対象エージェント。
        optimizer (torch.optim.Optimizer | None): 併せて保存する最適化器状態。
        meta (dict[str, Any] | None): 任意メタ情報。
    """
    model_cfg = getattr(agent, "model_config", None)
    resolved_model_cfg = normalize_model_config(
        model_cfg if isinstance(model_cfg, dict) else None,
        default_type=agent.__class__.__name__,
    )

    payload: dict[str, Any] = {
        "model_state_dict": agent.state_dict(),
        "obs_shape": tuple(agent.obs_shape),
        "action_dim": int(agent.action_dim),
        "model_config": resolved_model_cfg,
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    save_checkpoint_payload(path, payload)


def load_agent_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[DiscreteBoardAgent, dict[str, Any]]:
    """チェックポイントからエージェントを復元する。

    Args:
        path (str | Path): 読み込み元パス。
        device (torch.device | str): 復元後モデルを配置するデバイス。

    Returns:
        tuple[DiscreteBoardAgent, dict[str, Any]]:
            `(agent, meta)`。

    Raises:
        ValueError: 必須キー不足などチェックポイント形式が不正な場合。
    """
    payload = load_checkpoint_payload(path, device=device)
    if "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format (missing model_state_dict): {path}")
    if "obs_shape" not in payload or "action_dim" not in payload:
        raise ValueError(f"invalid checkpoint format (missing obs_shape/action_dim): {path}")
    model_config = payload.get("model_config")
    if model_config is None:
        raise ValueError("checkpoint does not contain model_config; legacy model_kwargs checkpoints are unsupported")

    obs_shape = tuple(payload["obs_shape"])
    action_dim = int(payload["action_dim"])
    agent, _resolved_model_config = build_agent(
        obs_shape=obs_shape,
        action_dim=action_dim,
        model_config=model_config,
        default_type="DiscreteBoardAgent",
    )
    agent.load_state_dict(payload["model_state_dict"])
    agent.to(device)
    meta = payload.get("meta")
    return agent, (dict(meta) if isinstance(meta, dict) else {})
