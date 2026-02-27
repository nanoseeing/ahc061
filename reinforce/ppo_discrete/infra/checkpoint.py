from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint_payload(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), out_path)


def load_checkpoint_payload(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be dict: {path}")
    return dict(payload)
