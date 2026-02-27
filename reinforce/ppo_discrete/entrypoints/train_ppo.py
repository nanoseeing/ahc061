from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..core.ppo.config import PPOConfig
from ..core.ppo.train_utils import (
    resolve_vecnorm_gamma,
    validate_ppo_config,
    validate_schedule_args,
    validate_vecnorm_config,
)
from ..infra.experiment import coerce_optional_path, resolve_config
from ..infra.log_utils import get_logger

logger = get_logger("train_ppo")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train discrete-board policy with PPO.")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="train_ppo", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--env-id", type=str, default="AHC061Local-v0")
    p.add_argument("--env-kwargs-json", type=str, default="{}")
    p.add_argument("--run-dir", type=Path, default=Path("reinforce/outputs/ppo_runs"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--init-model", type=Path, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume-from", type=Path, default=None, help="checkpoint path to resume from (defaults to <run>/models/last.pt when --resume)")
    p.add_argument("--feature-id", type=str, default="submit_v1")
    p.add_argument("--pf-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--memory-format", choices=["auto", "nchw", "channels_last"], default="auto")
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rollout-cache-device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--distributed", choices=["auto", "off", "on"], default="auto")
    p.add_argument("--model-preset", type=str, default="")

    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument(
        "--learning-rate-schedule",
        type=str,
        default="linear",
        help=(
            "optional LR schedule expression, e.g. "
            "'constant(3e-4)', 'linear(3e-4,1e-5)', "
            "'cosine(3e-4,1e-5)', 'exp(3e-4,1e-5)', "
            "'piecewise(0:3e-4,0.5:1.5e-4,1:5e-5)'. "
            "If empty, linear(learning_rate, 0.0) is used."
        ),
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-range-vf", type=float, default=None, help="optional value-function clip range; if unset, value clipping is disabled")
    p.add_argument("--clip-range-vf-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--clip-range-vf-final", type=float, default=None, help="final value clip range for schedule")
    p.add_argument(
        "--clip-range-vf-schedule-expr",
        type=str,
        default="",
        help=(
            "optional value-clip schedule expression; overrides clip-range-vf-schedule/final "
            "(same expression format as --learning-rate-schedule)"
        ),
    )
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-coef-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--ent-coef-final", type=float, default=None, help="final entropy coefficient for schedule")
    p.add_argument(
        "--ent-coef-schedule-expr",
        type=str,
        default="",
        help=(
            "optional entropy schedule expression; overrides ent-coef-schedule/ent-coef-final "
            "(same expression format as --learning-rate-schedule)"
        ),
    )
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--aux-opp-param-loss-coef", type=float, default=0.0)
    p.add_argument("--aux-opp-param-use-valid-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None, help="SB3 semantics: early stop when approx_kl > 1.5 * target_kl")
    p.add_argument("--clip-coef-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--clip-coef-final", type=float, default=None, help="final clip coefficient for schedule")
    p.add_argument(
        "--clip-coef-schedule-expr",
        type=str,
        default="",
        help=(
            "optional clip-coef schedule expression; overrides clip-coef-schedule/clip-coef-final "
            "(same expression format as --learning-rate-schedule)"
        ),
    )

    p.add_argument("--model-class", type=str, default="", help="registered model name or import path")
    p.add_argument("--model-config-file", type=Path, default=None, help="optional model config (json/toml/yaml)")
    p.add_argument("--model-config-json", type=str, default="", help="optional model config JSON override")
    p.add_argument("--use-action-mask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--checkpoint-interval-steps", type=int, default=0, help="save step_*.pt snapshots every N global steps (0 disables)")
    p.add_argument(
        "--val-interval-steps",
        "--eval-interval-steps",
        dest="eval_interval_steps",
        type=int,
        default=0,
        help="0 disables periodic val during PPO training",
    )
    p.add_argument("--val-episodes", "--eval-episodes", dest="eval_episodes", type=int, default=100)
    p.add_argument("--val-seed-start", "--eval-seed-start", dest="eval_seed_start", type=int, default=2_000_000)
    p.add_argument("--val-fixed-seeds", "--eval-fixed-seeds", dest="eval_fixed_seeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--val-deterministic",
        "--eval-deterministic",
        dest="eval_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--val-at-start", "--eval-at-start", dest="eval_at_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--val-env-kwargs-json",
        "--eval-env-kwargs-json",
        dest="eval_env_kwargs_json",
        type=str,
        default="",
        help="if empty, training env kwargs are reused",
    )
    p.add_argument("--vecnorm", action=argparse.BooleanOptionalAction, default=False, help="enable VecNormalize-style obs/reward normalization")
    p.add_argument("--vecnorm-norm-obs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vecnorm-norm-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--vecnorm-val-norm-reward",
        "--vecnorm-eval-norm-reward",
        dest="vecnorm_eval_norm_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--vecnorm-clip-reward", type=float, default=10.0)
    p.add_argument("--vecnorm-epsilon", type=float, default=1e-8)
    p.add_argument("--vecnorm-gamma", type=float, default=None, help="if unset, PPO gamma is used")
    p.add_argument(
        "--log-interval-iters",
        type=int,
        default=1,
        help="stdout logging interval in PPO iterations (1 logs every iteration)",
    )

    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True, help="enable TensorBoard logging")
    p.add_argument("--mlflow-tracking-uri", type=str, default="")
    p.add_argument("--mlflow-experiment", type=str, default="ppo_discrete")
    p.add_argument("--mlflow-run-name", type=str, default="")
    return p


def _default_cli_config(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults = vars(parser.parse_args([])).copy()
    for key in ("config_file", "config_section", "set"):
        defaults.pop(key, None)
    return defaults


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    pre, _unknown = parser.parse_known_args()

    cfg = resolve_config(
        defaults=_default_cli_config(parser),
        config_file=pre.config_file,
        config_section=pre.config_section,
        overrides=list(pre.set or []),
    )
    unknown = sorted(k for k in cfg.keys() if k not in _default_cli_config(parser).keys())
    if unknown:
        raise ValueError(f"unknown config keys for train_ppo: {', '.join(unknown)}")

    parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.init_model = coerce_optional_path(args.init_model, dot_is_none=True)
    args.resume_from = coerce_optional_path(args.resume_from, dot_is_none=True)
    args.model_config_file = coerce_optional_path(args.model_config_file, dot_is_none=True)
    return args


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_env_kwargs(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--env-kwargs-json must be a JSON object")
    return obj


def args_to_cfg(args: argparse.Namespace) -> PPOConfig:
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


def _resolve_vecnorm_gamma(args: argparse.Namespace, cfg: PPOConfig) -> float:
    return resolve_vecnorm_gamma(
        vecnorm_gamma=args.vecnorm_gamma,
        ppo_gamma=float(cfg.gamma),
    )


def _validate_vecnorm_args(args: argparse.Namespace, cfg: PPOConfig) -> None:
    validate_vecnorm_config(
        enabled=bool(args.vecnorm),
        clip_obs=float(args.vecnorm_clip_obs),
        clip_reward=float(args.vecnorm_clip_reward),
        epsilon=float(args.vecnorm_epsilon),
        vecnorm_gamma=args.vecnorm_gamma,
        ppo_gamma=float(cfg.gamma),
    )


_validate_ppo_cfg = validate_ppo_config
_validate_schedule_args = validate_schedule_args


def _run_backend_from_train_ppo(*, args: argparse.Namespace, cfg: PPOConfig, device: torch.device) -> int:
    from ..usecases.ppo_service import TrainPPORequest, run_ppo_from_train_request

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


def main() -> int:
    args = parse_args()
    cfg = args_to_cfg(args)
    _validate_ppo_cfg(cfg)
    _validate_schedule_args(cfg)
    _validate_vecnorm_args(args, cfg)

    env_id = str(args.env_id).strip()
    if env_id != "AHC061Local-v0":
        raise ValueError("train_ppo supports only --env-id AHC061Local-v0")

    device = choose_device(args.device)
    return _run_backend_from_train_ppo(args=args, cfg=cfg, device=device)


if __name__ == "__main__":
    raise SystemExit(main())
