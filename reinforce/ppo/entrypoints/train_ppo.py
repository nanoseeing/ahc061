"""AHC061 の PPO 学習を起動する CLI エントリーポイント。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
from omegaconf import DictConfig

from ..ppo.config import PPOConfig
from ..train.schedule import (
    validate_schedule_args,
    validate_vecnorm_config,
)
from ..train.requests import ppo_config_from_source
from ..utils.runtime import choose_device, parse_json_object
from ..utils.log_utils import get_logger
from .common import cfg_to_namespace

logger = get_logger("train_ppo")
_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


@hydra.main(version_base="1.3", config_path=_CONF_DIR, config_name="train_ppo/default")
def main(cfg: DictConfig) -> None:
    """Hydra 設定を読み込み PPO 学習ジョブを開始する。

    Args:
        cfg (DictConfig): `train_ppo/default` から解決された実行設定。
    """
    args = _cfg_to_ns(cfg)
    raise SystemExit(_run(args))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    """`train_ppo` 用に設定オブジェクトを `SimpleNamespace` 化する。

    Args:
        cfg (DictConfig): Hydra が解決した設定。

    Returns:
        SimpleNamespace: パス正規化済みの引数セット。
    """
    return cfg_to_namespace(
        cfg,
        optional_paths={
            "init_model": True,
            "resume_from": True,
            "model_config_file": True,
        },
        default_paths={"run_dir": Path("reinforce/outputs/ppo_runs")},
    )


def _ns_to_ppo_cfg(args: SimpleNamespace) -> PPOConfig:
    """CLI 引数相当の名前空間から `PPOConfig` を生成する。

    Args:
        args (SimpleNamespace): 学習関連の引数群。

    Returns:
        PPOConfig: 検証済み PPO ハイパーパラメータ。
    """
    return ppo_config_from_source(args)


def _run(args: SimpleNamespace) -> int:
    """前処理検証を行ったうえで PPO 学習本体を実行する。

    Args:
        args (SimpleNamespace): 実行時引数。

    Returns:
        int: プロセス終了コード。
    """
    cfg = _ns_to_ppo_cfg(args)
    validate_schedule_args(cfg)
    validate_vecnorm_config(
        enabled=bool(args.vecnorm),
        clip_obs=float(args.vecnorm_clip_obs),
        clip_reward=float(args.vecnorm_clip_reward),
        epsilon=float(args.vecnorm_epsilon),
        vecnorm_gamma=args.vecnorm_gamma,
        ppo_gamma=float(cfg.gamma),
    )

    env_id = str(args.env_id).strip()
    if env_id != "AHC061Local-v0":
        raise ValueError("train_ppo supports only env_id=AHC061Local-v0")

    device = choose_device(args.device)
    return _run_backend(args=args, cfg=cfg, device=device)


def _run_backend(*, args: SimpleNamespace, cfg: PPOConfig, device: torch.device) -> int:
    """サービス層リクエストを組み立てて PPO 学習を委譲する。

    Args:
        args (SimpleNamespace): 実行時引数。
        cfg (PPOConfig): 学習設定。
        device (torch.device): 学習実行デバイス。

    Returns:
        int: プロセス終了コード。
    """
    from ..train.ppo_service import TrainPPORequest, run_ppo_from_train_request

    env_kwargs = parse_json_object(str(args.env_kwargs_json), field_name="env_kwargs_json")
    train_req = TrainPPORequest(
        env_id=str(args.env_id),
        run_dir=args.run_dir,
        run_name=str(args.run_name),
        train_seed_min=int(args.train_seed_min),
        train_seed_max_exclusive=int(args.train_seed_max_exclusive),
        init_model=args.init_model,
        resume=bool(args.resume),
        resume_from=args.resume_from,
        checkpoint_interval_iterations=int(args.checkpoint_interval_iterations),
        eval_interval_iterations=int(args.eval_interval_iterations),
        eval_episodes=int(args.eval_episodes),
        eval_num_envs=int(args.eval_num_envs),
        eval_seed_start=int(args.eval_seed_start),
        eval_fixed_seeds=bool(args.eval_fixed_seeds),
        eval_at_start=bool(args.eval_at_start),
        vecnorm=bool(args.vecnorm),
        vecnorm_norm_obs=bool(args.vecnorm_norm_obs),
        vecnorm_norm_reward=bool(args.vecnorm_norm_reward),
        vecnorm_eval_norm_reward=bool(args.vecnorm_eval_norm_reward),
        vecnorm_clip_obs=float(args.vecnorm_clip_obs),
        vecnorm_clip_reward=float(args.vecnorm_clip_reward),
        vecnorm_epsilon=float(args.vecnorm_epsilon),
        vecnorm_gamma=args.vecnorm_gamma,
        model_class=str(args.model_class),
        model_config_file=args.model_config_file,
        model_config_json=str(args.model_config_json),
        model_preset=str(args.model_preset),
        feature_id=str(args.feature_id),
        pf_enabled=bool(args.pf_enabled),
        use_action_mask=bool(args.use_action_mask),
        amp=bool(args.amp),
        memory_format=str(args.memory_format),
        pin_memory=bool(args.pin_memory),
        rollout_cache_device=str(args.rollout_cache_device),
        distributed=str(args.distributed),
        compile=bool(args.compile),
        log_interval_iters=int(args.log_interval_iters),
        mlflow_tracking_uri=str(args.mlflow_tracking_uri),
        mlflow_experiment=str(args.mlflow_experiment),
        mlflow_run_name=str(args.mlflow_run_name),
        tensorboard=bool(args.tensorboard),
    )
    return int(
        run_ppo_from_train_request(
            train_req=train_req,
            cfg=cfg,
            device=device,
            env_kwargs=env_kwargs,
        )
    )


if __name__ == "__main__":
    main()
