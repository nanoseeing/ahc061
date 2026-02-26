from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import build_agent, normalize_model_config
from ..models.discrete_board import DiscreteBoardAgent


def save_agent_checkpoint(
    path: str | Path,
    agent: DiscreteBoardAgent,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
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
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_agent_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[DiscreteBoardAgent, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    obs_shape = tuple(payload["obs_shape"])
    action_dim = int(payload["action_dim"])
    model_config = payload.get("model_config")
    if model_config is None:
        raise ValueError("checkpoint does not contain model_config; legacy model_kwargs checkpoints are unsupported")
    default_type = "DiscreteBoardAgent"
    agent, _resolved_model_config = build_agent(
        obs_shape=obs_shape,
        action_dim=action_dim,
        model_config=model_config,
        default_type=default_type,
    )
    agent.load_state_dict(payload["model_state_dict"])
    agent.to(device)
    meta = dict(payload.get("meta", {}))
    return agent, meta
