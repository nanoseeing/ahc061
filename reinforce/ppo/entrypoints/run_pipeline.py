"""BC→PPO→評価の実験パイプラインを起動する CLI。"""
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
    """Hydra 設定を読み込み実験パイプラインを開始する。

    Args:
        cfg (DictConfig): `run_pipeline/default` から解決された設定。
    """
    args = _cfg_to_ns(cfg)
    raise SystemExit(int(run_pipeline(args)))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    """`run_pipeline` 用に設定を実行引数へ正規化する。

    Args:
        cfg (DictConfig): Hydra が解決した設定。

    Returns:
        SimpleNamespace: サービス層に渡す名前空間引数。
    """
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
