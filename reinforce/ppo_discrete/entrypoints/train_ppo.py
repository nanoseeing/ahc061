from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Optional

import torch
import typer

from ..ppo.config import PPOConfig
from ..train.schedule import (
    validate_ppo_config,
    validate_schedule_args,
    validate_vecnorm_config,
)
from ..utils.experiment import coerce_optional_path, resolve_config
from ..utils.log_utils import get_logger

logger = get_logger("train_ppo")
app = typer.Typer(add_completion=False)

_DEFAULTS: dict[str, Any] = {
    # environment
    "env_id": "AHC061Local-v0",
    "env_kwargs_json": "{}",
    # run management
    "run_dir": Path("reinforce/outputs/ppo_runs"),
    "run_name": "",
    "seed": 1,
    "device": "auto",
    "init_model": None,
    "resume": False,
    "resume_from": None,
    # feature / architecture
    "feature_id": "submit_v1",
    "pf_enabled": True,
    "amp": False,
    "memory_format": "auto",
    "pin_memory": True,
    "rollout_cache_device": "auto",
    "distributed": "auto",
    "model_preset": "",
    # PPO core
    "total_timesteps": 500_000,
    "num_envs": 8,
    "num_steps": 100,
    "learning_rate": 2.5e-4,
    "learning_rate_schedule": "linear",
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "clip_range_vf": None,
    "clip_range_vf_schedule": "constant",
    "clip_range_vf_final": None,
    "clip_range_vf_schedule_expr": "",
    "clip_vloss": True,
    "ent_coef": 0.01,
    "ent_coef_schedule": "constant",
    "ent_coef_final": None,
    "ent_coef_schedule_expr": "",
    "vf_coef": 0.5,
    "aux_opp_param_loss_coef": 0.0,
    "aux_opp_param_use_valid_mask": True,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "clip_coef_schedule": "constant",
    "clip_coef_final": None,
    "clip_coef_schedule_expr": "",
    # model
    "model_class": "",
    "model_config_file": None,
    "model_config_json": "",
    "use_action_mask": False,
    # checkpointing / eval
    "save_interval": 10,
    "checkpoint_interval_steps": 0,
    "eval_interval_steps": 0,
    "eval_episodes": 100,
    "eval_seed_start": 2_000_000,
    "eval_fixed_seeds": True,
    "eval_deterministic": True,
    "eval_at_start": True,
    "eval_env_kwargs_json": "",
    # vecnorm
    "vecnorm": False,
    "vecnorm_norm_obs": True,
    "vecnorm_norm_reward": True,
    "vecnorm_eval_norm_reward": False,
    "vecnorm_clip_obs": 10.0,
    "vecnorm_clip_reward": 10.0,
    "vecnorm_epsilon": 1e-8,
    "vecnorm_gamma": None,
    # logging / tracking
    "log_interval_iters": 1,
    "mlflow_tracking_uri": "",
    "mlflow_experiment": "ppo_discrete",
    "mlflow_run_name": "",
}


def _ns(cfg: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**cfg)


@app.command()
def main(
    config_file: Annotated[Optional[Path], typer.Option("--config-file")] = None,
    config_section: Annotated[str, typer.Option("--config-section")] = "train_ppo",
    set_: Annotated[Optional[list[str]], typer.Option("--set")] = None,
    # environment
    env_id: Annotated[Optional[str], typer.Option("--env-id")] = None,
    env_kwargs_json: Annotated[Optional[str], typer.Option("--env-kwargs-json")] = None,
    # run management
    run_dir: Annotated[Optional[Path], typer.Option("--run-dir")] = None,
    run_name: Annotated[Optional[str], typer.Option("--run-name")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed")] = None,
    device: Annotated[Optional[str], typer.Option("--device")] = None,
    init_model: Annotated[Optional[Path], typer.Option("--init-model")] = None,
    resume: Annotated[Optional[bool], typer.Option("--resume/--no-resume")] = None,
    resume_from: Annotated[Optional[Path], typer.Option("--resume-from")] = None,
    # feature / architecture
    feature_id: Annotated[Optional[str], typer.Option("--feature-id")] = None,
    pf_enabled: Annotated[Optional[bool], typer.Option("--pf-enabled/--no-pf-enabled")] = None,
    amp: Annotated[Optional[bool], typer.Option("--amp/--no-amp")] = None,
    memory_format: Annotated[Optional[str], typer.Option("--memory-format")] = None,
    pin_memory: Annotated[Optional[bool], typer.Option("--pin-memory/--no-pin-memory")] = None,
    rollout_cache_device: Annotated[Optional[str], typer.Option("--rollout-cache-device")] = None,
    distributed: Annotated[Optional[str], typer.Option("--distributed")] = None,
    model_preset: Annotated[Optional[str], typer.Option("--model-preset")] = None,
    # PPO core
    total_timesteps: Annotated[Optional[int], typer.Option("--total-timesteps")] = None,
    num_envs: Annotated[Optional[int], typer.Option("--num-envs")] = None,
    num_steps: Annotated[Optional[int], typer.Option("--num-steps")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--learning-rate")] = None,
    learning_rate_schedule: Annotated[Optional[str], typer.Option("--learning-rate-schedule")] = None,
    gamma: Annotated[Optional[float], typer.Option("--gamma")] = None,
    gae_lambda: Annotated[Optional[float], typer.Option("--gae-lambda")] = None,
    num_minibatches: Annotated[Optional[int], typer.Option("--num-minibatches")] = None,
    update_epochs: Annotated[Optional[int], typer.Option("--update-epochs")] = None,
    norm_adv: Annotated[Optional[bool], typer.Option("--norm-adv/--no-norm-adv")] = None,
    clip_coef: Annotated[Optional[float], typer.Option("--clip-coef")] = None,
    clip_range_vf: Annotated[Optional[float], typer.Option("--clip-range-vf")] = None,
    clip_range_vf_schedule: Annotated[Optional[str], typer.Option("--clip-range-vf-schedule")] = None,
    clip_range_vf_final: Annotated[Optional[float], typer.Option("--clip-range-vf-final")] = None,
    clip_range_vf_schedule_expr: Annotated[Optional[str], typer.Option("--clip-range-vf-schedule-expr")] = None,
    clip_vloss: Annotated[Optional[bool], typer.Option("--clip-vloss/--no-clip-vloss")] = None,
    ent_coef: Annotated[Optional[float], typer.Option("--ent-coef")] = None,
    ent_coef_schedule: Annotated[Optional[str], typer.Option("--ent-coef-schedule")] = None,
    ent_coef_final: Annotated[Optional[float], typer.Option("--ent-coef-final")] = None,
    ent_coef_schedule_expr: Annotated[Optional[str], typer.Option("--ent-coef-schedule-expr")] = None,
    vf_coef: Annotated[Optional[float], typer.Option("--vf-coef")] = None,
    aux_opp_param_loss_coef: Annotated[Optional[float], typer.Option("--aux-opp-param-loss-coef")] = None,
    aux_opp_param_use_valid_mask: Annotated[Optional[bool], typer.Option("--aux-opp-param-use-valid-mask/--no-aux-opp-param-use-valid-mask")] = None,
    max_grad_norm: Annotated[Optional[float], typer.Option("--max-grad-norm")] = None,
    target_kl: Annotated[Optional[float], typer.Option("--target-kl")] = None,
    clip_coef_schedule: Annotated[Optional[str], typer.Option("--clip-coef-schedule")] = None,
    clip_coef_final: Annotated[Optional[float], typer.Option("--clip-coef-final")] = None,
    clip_coef_schedule_expr: Annotated[Optional[str], typer.Option("--clip-coef-schedule-expr")] = None,
    # model
    model_class: Annotated[Optional[str], typer.Option("--model-class")] = None,
    model_config_file: Annotated[Optional[Path], typer.Option("--model-config-file")] = None,
    model_config_json: Annotated[Optional[str], typer.Option("--model-config-json")] = None,
    use_action_mask: Annotated[Optional[bool], typer.Option("--use-action-mask/--no-use-action-mask")] = None,
    # checkpointing / eval
    save_interval: Annotated[Optional[int], typer.Option("--save-interval")] = None,
    checkpoint_interval_steps: Annotated[Optional[int], typer.Option("--checkpoint-interval-steps")] = None,
    eval_interval_steps: Annotated[Optional[int], typer.Option("--eval-interval-steps")] = None,
    eval_episodes: Annotated[Optional[int], typer.Option("--eval-episodes")] = None,
    eval_seed_start: Annotated[Optional[int], typer.Option("--eval-seed-start")] = None,
    eval_fixed_seeds: Annotated[Optional[bool], typer.Option("--eval-fixed-seeds/--no-eval-fixed-seeds")] = None,
    eval_deterministic: Annotated[Optional[bool], typer.Option("--eval-deterministic/--no-eval-deterministic")] = None,
    eval_at_start: Annotated[Optional[bool], typer.Option("--eval-at-start/--no-eval-at-start")] = None,
    eval_env_kwargs_json: Annotated[Optional[str], typer.Option("--eval-env-kwargs-json")] = None,
    # vecnorm
    vecnorm: Annotated[Optional[bool], typer.Option("--vecnorm/--no-vecnorm")] = None,
    vecnorm_norm_obs: Annotated[Optional[bool], typer.Option("--vecnorm-norm-obs/--no-vecnorm-norm-obs")] = None,
    vecnorm_norm_reward: Annotated[Optional[bool], typer.Option("--vecnorm-norm-reward/--no-vecnorm-norm-reward")] = None,
    vecnorm_eval_norm_reward: Annotated[Optional[bool], typer.Option("--vecnorm-eval-norm-reward/--no-vecnorm-eval-norm-reward")] = None,
    vecnorm_clip_obs: Annotated[Optional[float], typer.Option("--vecnorm-clip-obs")] = None,
    vecnorm_clip_reward: Annotated[Optional[float], typer.Option("--vecnorm-clip-reward")] = None,
    vecnorm_epsilon: Annotated[Optional[float], typer.Option("--vecnorm-epsilon")] = None,
    vecnorm_gamma: Annotated[Optional[float], typer.Option("--vecnorm-gamma")] = None,
    # logging / tracking
    log_interval_iters: Annotated[Optional[int], typer.Option("--log-interval-iters")] = None,
    mlflow_tracking_uri: Annotated[Optional[str], typer.Option("--mlflow-tracking-uri")] = None,
    mlflow_experiment: Annotated[Optional[str], typer.Option("--mlflow-experiment")] = None,
    mlflow_run_name: Annotated[Optional[str], typer.Option("--mlflow-run-name")] = None,
) -> None:
    cfg = resolve_config(
        defaults=_DEFAULTS,
        config_file=config_file,
        config_section=config_section,
        overrides=list(set_ or []),
    )
    _cli: dict[str, Any] = {
        "env_id": env_id,
        "env_kwargs_json": env_kwargs_json,
        "run_dir": run_dir,
        "run_name": run_name,
        "seed": seed,
        "device": device,
        "init_model": init_model,
        "resume": resume,
        "resume_from": resume_from,
        "feature_id": feature_id,
        "pf_enabled": pf_enabled,
        "amp": amp,
        "memory_format": memory_format,
        "pin_memory": pin_memory,
        "rollout_cache_device": rollout_cache_device,
        "distributed": distributed,
        "model_preset": model_preset,
        "total_timesteps": total_timesteps,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "learning_rate_schedule": learning_rate_schedule,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "num_minibatches": num_minibatches,
        "update_epochs": update_epochs,
        "norm_adv": norm_adv,
        "clip_coef": clip_coef,
        "clip_range_vf": clip_range_vf,
        "clip_range_vf_schedule": clip_range_vf_schedule,
        "clip_range_vf_final": clip_range_vf_final,
        "clip_range_vf_schedule_expr": clip_range_vf_schedule_expr,
        "clip_vloss": clip_vloss,
        "ent_coef": ent_coef,
        "ent_coef_schedule": ent_coef_schedule,
        "ent_coef_final": ent_coef_final,
        "ent_coef_schedule_expr": ent_coef_schedule_expr,
        "vf_coef": vf_coef,
        "aux_opp_param_loss_coef": aux_opp_param_loss_coef,
        "aux_opp_param_use_valid_mask": aux_opp_param_use_valid_mask,
        "max_grad_norm": max_grad_norm,
        "target_kl": target_kl,
        "clip_coef_schedule": clip_coef_schedule,
        "clip_coef_final": clip_coef_final,
        "clip_coef_schedule_expr": clip_coef_schedule_expr,
        "model_class": model_class,
        "model_config_file": model_config_file,
        "model_config_json": model_config_json,
        "use_action_mask": use_action_mask,
        "save_interval": save_interval,
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "eval_interval_steps": eval_interval_steps,
        "eval_episodes": eval_episodes,
        "eval_seed_start": eval_seed_start,
        "eval_fixed_seeds": eval_fixed_seeds,
        "eval_deterministic": eval_deterministic,
        "eval_at_start": eval_at_start,
        "eval_env_kwargs_json": eval_env_kwargs_json,
        "vecnorm": vecnorm,
        "vecnorm_norm_obs": vecnorm_norm_obs,
        "vecnorm_norm_reward": vecnorm_norm_reward,
        "vecnorm_eval_norm_reward": vecnorm_eval_norm_reward,
        "vecnorm_clip_obs": vecnorm_clip_obs,
        "vecnorm_clip_reward": vecnorm_clip_reward,
        "vecnorm_epsilon": vecnorm_epsilon,
        "vecnorm_gamma": vecnorm_gamma,
        "log_interval_iters": log_interval_iters,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_experiment": mlflow_experiment,
        "mlflow_run_name": mlflow_run_name,
    }
    cfg.update({k: v for k, v in _cli.items() if v is not None})
    cfg["init_model"] = coerce_optional_path(cfg.get("init_model"), dot_is_none=True)
    cfg["resume_from"] = coerce_optional_path(cfg.get("resume_from"), dot_is_none=True)
    cfg["model_config_file"] = coerce_optional_path(cfg.get("model_config_file"), dot_is_none=True)
    args = _ns(cfg)
    raise SystemExit(_run(args))


def _ns_to_ppo_cfg(args: SimpleNamespace) -> PPOConfig:
    return PPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        learning_rate_schedule=str(args.learning_rate_schedule),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_range_vf=args.clip_range_vf,
        clip_range_vf_schedule=str(args.clip_range_vf_schedule),
        clip_range_vf_final=args.clip_range_vf_final,
        clip_range_vf_schedule_expr=str(args.clip_range_vf_schedule_expr),
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        ent_coef_schedule=str(args.ent_coef_schedule),
        ent_coef_final=args.ent_coef_final,
        ent_coef_schedule_expr=str(args.ent_coef_schedule_expr),
        vf_coef=args.vf_coef,
        aux_opp_param_loss_coef=float(args.aux_opp_param_loss_coef),
        aux_opp_param_use_valid_mask=bool(args.aux_opp_param_use_valid_mask),
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        clip_coef_schedule=str(args.clip_coef_schedule),
        clip_coef_final=args.clip_coef_final,
        clip_coef_schedule_expr=str(args.clip_coef_schedule_expr),
        save_interval=args.save_interval,
    )


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_env_kwargs(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--env-kwargs-json must be a JSON object")
    return obj


def _run(args: SimpleNamespace) -> int:
    cfg = _ns_to_ppo_cfg(args)
    validate_ppo_config(cfg)
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
        raise ValueError("train_ppo supports only --env-id AHC061Local-v0")

    device = choose_device(args.device)
    return _run_backend(args=args, cfg=cfg, device=device)


def _run_backend(*, args: SimpleNamespace, cfg: PPOConfig, device: torch.device) -> int:
    from ..train.ppo_service import TrainPPORequest, run_ppo_from_train_request

    env_kwargs = parse_env_kwargs(args.env_kwargs_json)
    train_req = TrainPPORequest(
        env_id=str(args.env_id),
        run_dir=args.run_dir,
        run_name=str(args.run_name),
        init_model=args.init_model,
        resume=bool(args.resume),
        resume_from=args.resume_from,
        checkpoint_interval_steps=int(args.checkpoint_interval_steps),
        eval_interval_steps=int(args.eval_interval_steps),
        eval_episodes=int(args.eval_episodes),
        eval_seed_start=int(args.eval_seed_start),
        eval_fixed_seeds=bool(args.eval_fixed_seeds),
        eval_deterministic=bool(args.eval_deterministic),
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
        log_interval_iters=int(args.log_interval_iters),
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
    app()
