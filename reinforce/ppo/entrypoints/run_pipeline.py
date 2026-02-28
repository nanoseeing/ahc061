from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig

from ..pipeline.pipeline_service import run_pipeline
from .common import cfg_to_namespace

_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


@hydra.main(version_base="1.3", config_path=_CONF_DIR, config_name="run_pipeline/default")
def main(cfg: DictConfig) -> None:
    args = _cfg_to_ns(cfg)
    raise SystemExit(int(run_pipeline(args)))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    return cfg_to_namespace(
        cfg,
        optional_paths={
            "model_config_file": True,
            "ppo_init_model": True,
        },
        default_paths={"run_root": Path("reinforce/outputs/pipeline_runs")},
    )


if __name__ == "__main__":
    main()
