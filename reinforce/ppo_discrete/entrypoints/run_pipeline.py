from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils.experiment import coerce_optional_path
from ..pipeline.pipeline_service import run_pipeline

_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


@hydra.main(version_base="1.3", config_path=_CONF_DIR, config_name="run_pipeline/default")
def main(cfg: DictConfig) -> None:
    args = _cfg_to_ns(cfg)
    raise SystemExit(int(run_pipeline(args)))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    d: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[assignment]
    d["model_config_file"] = coerce_optional_path(d.get("model_config_file"), dot_is_none=True)
    d["ppo_init_model"] = coerce_optional_path(d.get("ppo_init_model"), dot_is_none=True)
    run_root = d.get("run_root")
    d["run_root"] = Path(run_root) if run_root else Path("reinforce/outputs/pipeline_runs")
    return SimpleNamespace(**d)


if __name__ == "__main__":
    main()
