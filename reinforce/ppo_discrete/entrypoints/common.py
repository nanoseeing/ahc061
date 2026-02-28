from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from ..utils.experiment import coerce_optional_path


def cfg_to_namespace(
    cfg: DictConfig,
    *,
    optional_paths: Mapping[str, bool] | None = None,
    default_paths: Mapping[str, Path] | None = None,
) -> SimpleNamespace:
    """Convert Hydra config to SimpleNamespace with common path coercions.

    optional_paths:
        key -> dot_is_none flag for coerce_optional_path.
    default_paths:
        key -> default path used when the config value is empty.
    """
    d: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[assignment]
    for key, dot_is_none in dict(optional_paths or {}).items():
        d[key] = coerce_optional_path(d.get(key), dot_is_none=bool(dot_is_none))
    for key, default_path in dict(default_paths or {}).items():
        raw = d.get(key)
        d[key] = Path(raw) if raw else Path(default_path)
    return SimpleNamespace(**d)
