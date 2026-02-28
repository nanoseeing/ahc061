"""`requests` に関する学習処理。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..ppo.config import PPOConfig
from ..utils.log_utils import get_logger

logger = get_logger("train_ppo")


@dataclass(frozen=True)
class TrainPPORequest:
    """`TrainPPORequest` を表すクラス。"""
    env_id: str
    run_dir: Path
    run_name: str
    train_seed_min: int
    train_seed_max_exclusive: int
    init_model: Path | None
    resume: bool
    resume_from: Path | None
    checkpoint_interval_iterations: int
    eval_interval_iterations: int
    eval_episodes: int
    eval_num_envs: int
    eval_seed_start: int
    eval_fixed_seeds: bool
    eval_at_start: bool
    vecnorm: bool
    vecnorm_norm_obs: bool
    vecnorm_norm_reward: bool
    vecnorm_eval_norm_reward: bool
    vecnorm_clip_obs: float
    vecnorm_clip_reward: float
    vecnorm_epsilon: float
    vecnorm_gamma: float | None
    model_class: str
    model_config_file: Path | None
    model_config_json: str
    model_preset: str
    feature_id: str
    pf_enabled: bool
    use_action_mask: bool
    amp: bool
    memory_format: str
    pin_memory: bool
    rollout_cache_device: str
    distributed: str
    compile: bool
    log_interval_iters: int
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str
    tensorboard: bool


@dataclass(frozen=True)
class PPORequest:
    """`PPORequest` を表すクラス。"""
    run_dir: Path
    run_name: str
    seed: int
    train_seed_min: int
    train_seed_max_exclusive: int
    device: str
    init_model: Path | None
    resume: bool
    resume_from: Path | None
    total_iterations: int
    num_envs: int
    num_steps: int
    learning_rate: float
    learning_rate_schedule: str
    warmup_iters: int
    gamma: float
    gae_lambda: float
    num_minibatches: int
    update_epochs: int
    norm_adv: bool
    clip_coef: float
    clip_coef_schedule: str
    clip_coef_final: float | None
    clip_coef_schedule_expr: str
    clip_range_vf: float | None
    clip_range_vf_schedule: str
    clip_range_vf_final: float | None
    clip_range_vf_schedule_expr: str
    clip_vloss: bool
    ent_coef: float
    ent_coef_schedule: str
    ent_coef_final: float | None
    ent_coef_schedule_expr: str
    vf_coef: float
    aux_opp_param_loss_coef: float
    aux_opp_param_use_valid_mask: bool
    max_grad_norm: float
    target_kl: float | None
    save_interval: int
    checkpoint_interval_iterations: int
    eval_interval_iterations: int
    eval_episodes: int
    eval_num_envs: int
    eval_seed_start: int
    eval_fixed_seeds: bool
    eval_at_start: bool
    vecnorm: bool
    vecnorm_norm_obs: bool
    vecnorm_norm_reward: bool
    vecnorm_eval_norm_reward: bool
    vecnorm_clip_obs: float
    vecnorm_clip_reward: float
    vecnorm_epsilon: float
    vecnorm_gamma: float | None
    model_class: str
    model_config_file: Path | None
    model_config_json: str
    model_preset: str
    feature_id: str
    pf_enabled: bool
    use_action_mask: bool
    amp: bool
    memory_format: str
    pin_memory: bool
    rollout_cache_device: str
    distributed: str
    compile: bool
    log_interval_iters: int
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str
    tensorboard: bool


def args_to_cfg(args: PPORequest) -> PPOConfig:
    """`args_to_cfg` を実行する。

    Args:
        args (PPORequest): 設定引数。

    Returns:
        PPOConfig: 計算結果。
    """
    return ppo_config_from_source(args)


def ppo_config_from_source(src: Any) -> PPOConfig:
    """`ppo_config_from_source` を実行する。

    Args:
        src (Any): src の値。

    Returns:
        PPOConfig: 計算結果。
    """
    return PPOConfig(
        seed=src.seed,
        total_iterations=src.total_iterations,
        num_envs=src.num_envs,
        num_steps=src.num_steps,
        learning_rate=src.learning_rate,
        learning_rate_schedule=str(src.learning_rate_schedule),
        warmup_iters=int(src.warmup_iters),
        gamma=src.gamma,
        gae_lambda=src.gae_lambda,
        num_minibatches=src.num_minibatches,
        update_epochs=src.update_epochs,
        norm_adv=src.norm_adv,
        clip_coef=src.clip_coef,
        clip_coef_schedule=str(src.clip_coef_schedule),
        clip_coef_final=src.clip_coef_final,
        clip_coef_schedule_expr=str(src.clip_coef_schedule_expr),
        clip_range_vf=src.clip_range_vf,
        clip_range_vf_schedule=str(src.clip_range_vf_schedule),
        clip_range_vf_final=src.clip_range_vf_final,
        clip_range_vf_schedule_expr=str(src.clip_range_vf_schedule_expr),
        clip_vloss=src.clip_vloss,
        ent_coef=src.ent_coef,
        ent_coef_schedule=str(src.ent_coef_schedule),
        ent_coef_final=src.ent_coef_final,
        ent_coef_schedule_expr=str(src.ent_coef_schedule_expr),
        vf_coef=src.vf_coef,
        aux_opp_param_loss_coef=float(src.aux_opp_param_loss_coef),
        aux_opp_param_use_valid_mask=bool(src.aux_opp_param_use_valid_mask),
        max_grad_norm=src.max_grad_norm,
        target_kl=src.target_kl,
        save_interval=src.save_interval,
    )


def build_ppo_request(
    *,
    train_req: TrainPPORequest,
    cfg: PPOConfig,
    device: torch.device,
    env_kwargs: Mapping[str, Any],
) -> PPORequest:
    """`ppo_request`を構築する。

    Args:
        train_req (TrainPPORequest): train_req の値。
        cfg (PPOConfig): 設定オブジェクト。
        device (torch.device): 実行デバイス。
        env_kwargs (Mapping[str, Any]): 環境関連の値。

    Returns:
        PPORequest: 計算結果。
    """
    if str(train_req.env_id).strip() not in ("", "AHC061Local-v0"):
        raise ValueError(
            "train_ppo supports only AHC061Local-v0 "
            f"(got env_id={train_req.env_id!r})"
        )

    feature_id = str(train_req.feature_id)
    if "feature_id" in env_kwargs:
        feature_id = str(env_kwargs["feature_id"])
    pf_enabled = bool(train_req.pf_enabled)
    if "pf_enabled" in env_kwargs:
        pf_enabled = bool(env_kwargs["pf_enabled"])

    unsupported_env_keys = sorted(k for k in env_kwargs.keys() if k not in {"feature_id", "pf_enabled"})
    if unsupported_env_keys:
        logger.warning(
            "cpp backend ignores unsupported env_kwargs keys: %s",
            ", ".join(unsupported_env_keys),
        )

    use_action_mask = bool(train_req.use_action_mask)
    if not use_action_mask:
        logger.warning("cpp backend forces use_action_mask=true to avoid illegal actions")
        use_action_mask = True

    return PPORequest(
        run_dir=train_req.run_dir,
        run_name=train_req.run_name,
        seed=int(cfg.seed),
        train_seed_min=int(train_req.train_seed_min),
        train_seed_max_exclusive=int(train_req.train_seed_max_exclusive),
        device=str(device),
        init_model=train_req.init_model,
        resume=bool(train_req.resume),
        resume_from=train_req.resume_from,
        total_iterations=int(cfg.total_iterations),
        num_envs=int(cfg.num_envs),
        num_steps=int(cfg.num_steps),
        learning_rate=float(cfg.learning_rate),
        learning_rate_schedule=str(cfg.learning_rate_schedule),
        warmup_iters=int(cfg.warmup_iters),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        num_minibatches=int(cfg.num_minibatches),
        update_epochs=int(cfg.update_epochs),
        norm_adv=bool(cfg.norm_adv),
        clip_coef=float(cfg.clip_coef),
        clip_coef_schedule=str(cfg.clip_coef_schedule),
        clip_coef_final=(None if cfg.clip_coef_final is None else float(cfg.clip_coef_final)),
        clip_coef_schedule_expr=str(cfg.clip_coef_schedule_expr),
        clip_range_vf=(None if cfg.clip_range_vf is None else float(cfg.clip_range_vf)),
        clip_range_vf_schedule=str(cfg.clip_range_vf_schedule),
        clip_range_vf_final=(None if cfg.clip_range_vf_final is None else float(cfg.clip_range_vf_final)),
        clip_range_vf_schedule_expr=str(cfg.clip_range_vf_schedule_expr),
        clip_vloss=bool(cfg.clip_vloss),
        ent_coef=float(cfg.ent_coef),
        ent_coef_schedule=str(cfg.ent_coef_schedule),
        ent_coef_final=(None if cfg.ent_coef_final is None else float(cfg.ent_coef_final)),
        ent_coef_schedule_expr=str(cfg.ent_coef_schedule_expr),
        vf_coef=float(cfg.vf_coef),
        aux_opp_param_loss_coef=float(cfg.aux_opp_param_loss_coef),
        aux_opp_param_use_valid_mask=bool(cfg.aux_opp_param_use_valid_mask),
        max_grad_norm=float(cfg.max_grad_norm),
        target_kl=(None if cfg.target_kl is None else float(cfg.target_kl)),
        save_interval=int(cfg.save_interval),
        checkpoint_interval_iterations=int(train_req.checkpoint_interval_iterations),
        eval_interval_iterations=int(train_req.eval_interval_iterations),
        eval_episodes=int(train_req.eval_episodes),
        eval_num_envs=int(train_req.eval_num_envs),
        eval_seed_start=int(train_req.eval_seed_start),
        eval_fixed_seeds=bool(train_req.eval_fixed_seeds),
        eval_at_start=bool(train_req.eval_at_start),
        vecnorm=bool(train_req.vecnorm),
        vecnorm_norm_obs=bool(train_req.vecnorm_norm_obs),
        vecnorm_norm_reward=bool(train_req.vecnorm_norm_reward),
        vecnorm_eval_norm_reward=bool(train_req.vecnorm_eval_norm_reward),
        vecnorm_clip_obs=float(train_req.vecnorm_clip_obs),
        vecnorm_clip_reward=float(train_req.vecnorm_clip_reward),
        vecnorm_epsilon=float(train_req.vecnorm_epsilon),
        vecnorm_gamma=train_req.vecnorm_gamma,
        model_class=str(train_req.model_class),
        model_config_file=train_req.model_config_file,
        model_config_json=str(train_req.model_config_json),
        model_preset=str(train_req.model_preset),
        feature_id=feature_id,
        pf_enabled=pf_enabled,
        use_action_mask=bool(use_action_mask),
        amp=bool(train_req.amp),
        memory_format=str(train_req.memory_format),
        pin_memory=bool(train_req.pin_memory),
        rollout_cache_device=str(train_req.rollout_cache_device),
        distributed=str(train_req.distributed),
        compile=bool(train_req.compile),
        log_interval_iters=int(train_req.log_interval_iters),
        mlflow_tracking_uri=str(train_req.mlflow_tracking_uri),
        mlflow_experiment=str(train_req.mlflow_experiment),
        mlflow_run_name=str(train_req.mlflow_run_name),
        tensorboard=bool(train_req.tensorboard),
    )
