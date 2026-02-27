from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..ppo.config import PPOConfig
from ..train.schedule import (
    validate_schedule_args,
    validate_vecnorm_config,
)
from ..utils.experiment import coerce_optional_path
from ..utils.log_utils import get_logger

logger = get_logger("train_ppo")
_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


@hydra.main(version_base="1.3", config_path=_CONF_DIR, config_name="train_ppo/default")
def main(cfg: DictConfig) -> None:
    args = _cfg_to_ns(cfg)
    raise SystemExit(_run(args))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    d: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[assignment]
    d["init_model"] = coerce_optional_path(d.get("init_model"), dot_is_none=True)
    d["resume_from"] = coerce_optional_path(d.get("resume_from"), dot_is_none=True)
    d["model_config_file"] = coerce_optional_path(d.get("model_config_file"), dot_is_none=True)
    run_dir = d.get("run_dir")
    d["run_dir"] = Path(run_dir) if run_dir else Path("reinforce/outputs/ppo_runs")
    return SimpleNamespace(**d)


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
        raise ValueError("env_kwargs_json must be a JSON object")
    return obj


def _run(args: SimpleNamespace) -> int:
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
        compile=bool(args.compile),
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
    main()
