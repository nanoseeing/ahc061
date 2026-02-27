from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..infra.experiment import coerce_optional_path, resolve_config
from ..usecases.pipeline_service import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run skeleton PPO pipeline: collect -> BC -> PPO(val) -> eval")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="run_pipeline", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--env-id", type=str, default="AHC061Local-v0")
    p.add_argument("--env-kwargs-json", type=str, default="{}")
    p.add_argument("--run-root", type=Path, default=Path("reinforce/outputs/pipeline_runs"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resume an existing pipeline run (requires --run-name)",
    )
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--collect-episodes", type=int, default=200)
    p.add_argument(
        "--collect-policy",
        choices=["random", "model_stochastic", "model_greedy"],
        default="random",
    )
    p.add_argument("--collect-workers", type=int, default=1)
    p.add_argument("--collect-feature-id", type=str, default="submit_v1")
    p.add_argument("--collect-pf-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--collect-amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--collect-save-aux-targets", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--collect-chunk-episodes", type=int, default=10)
    p.add_argument("--skip-collect", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--bc-aux-opp-param-loss-coef", type=float, default=0.0)
    p.add_argument("--bc-aux-opp-param-use-valid-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--bc-use-collect-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train BC directly from collect worker shards when available to reduce memory usage",
    )
    p.add_argument("--skip-bc", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--ppo-total-timesteps", type=int, default=200_000)
    p.add_argument("--ppo-num-envs", type=int, default=8)
    p.add_argument("--ppo-num-steps", type=int, default=100)
    p.add_argument("--ppo-learning-rate", type=float, default=2.5e-4)
    p.add_argument("--ppo-learning-rate-schedule", type=str, default="linear")
    p.add_argument("--ppo-gamma", type=float, default=0.99)
    p.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-num-minibatches", type=int, default=4)
    p.add_argument("--ppo-update-epochs", type=int, default=4)
    p.add_argument("--ppo-norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-clip-coef", type=float, default=0.2)
    p.add_argument("--ppo-clip-range-vf", type=float, default=None)
    p.add_argument("--ppo-clip-range-vf-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--ppo-clip-range-vf-final", type=float, default=None)
    p.add_argument("--ppo-clip-range-vf-schedule-expr", type=str, default="")
    p.add_argument("--ppo-clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-ent-coef", type=float, default=0.01)
    p.add_argument("--ppo-ent-coef-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--ppo-ent-coef-final", type=float, default=None)
    p.add_argument("--ppo-ent-coef-schedule-expr", type=str, default="")
    p.add_argument("--ppo-vf-coef", type=float, default=0.5)
    p.add_argument("--ppo-aux-opp-param-loss-coef", type=float, default=0.0)
    p.add_argument("--ppo-aux-opp-param-use-valid-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    p.add_argument("--ppo-target-kl", type=float, default=None)
    p.add_argument("--ppo-clip-coef-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--ppo-clip-coef-final", type=float, default=None)
    p.add_argument("--ppo-clip-coef-schedule-expr", type=str, default="")
    p.add_argument("--ppo-checkpoint-interval-steps", type=int, default=0)
    p.add_argument("--ppo-feature-id", type=str, default="submit_v1")
    p.add_argument("--ppo-pf-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ppo-memory-format", choices=["auto", "nchw", "channels_last"], default="auto")
    p.add_argument("--ppo-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-rollout-cache-device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--ppo-distributed", choices=["auto", "off", "on"], default="auto")
    p.add_argument("--ppo-model-preset", type=str, default="")
    p.add_argument(
        "--ppo-val-interval-steps",
        "--ppo-eval-interval-steps",
        dest="ppo_eval_interval_steps",
        type=int,
        default=0,
    )
    p.add_argument("--ppo-val-episodes", "--ppo-eval-episodes", dest="ppo_eval_episodes", type=int, default=100)
    p.add_argument(
        "--ppo-val-seed-start",
        "--ppo-eval-seed-start",
        dest="ppo_eval_seed_start",
        type=int,
        default=2_000_000,
    )
    p.add_argument(
        "--ppo-val-fixed-seeds",
        "--ppo-eval-fixed-seeds",
        dest="ppo_eval_fixed_seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--ppo-val-deterministic",
        "--ppo-eval-deterministic",
        dest="ppo_eval_deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--ppo-val-at-start",
        "--ppo-eval-at-start",
        dest="ppo_eval_at_start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--ppo-val-env-kwargs-json",
        "--ppo-eval-env-kwargs-json",
        dest="ppo_eval_env_kwargs_json",
        type=str,
        default="",
    )
    p.add_argument("--ppo-vecnorm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ppo-vecnorm-norm-obs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-vecnorm-norm-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--ppo-vecnorm-val-norm-reward",
        "--ppo-vecnorm-eval-norm-reward",
        dest="ppo_vecnorm_eval_norm_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--ppo-vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--ppo-vecnorm-clip-reward", type=float, default=10.0)
    p.add_argument("--ppo-vecnorm-epsilon", type=float, default=1e-8)
    p.add_argument("--ppo-vecnorm-gamma", type=float, default=None)
    p.add_argument("--ppo-log-interval-iters", type=int, default=1)
    p.add_argument("--use-action-mask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--model-class", type=str, default="")
    p.add_argument("--model-config-file", type=Path, default=None)
    p.add_argument("--model-config-json", type=str, default="")
    p.add_argument(
        "--ppo-init-model",
        type=Path,
        default=None,
        help="optional PPO init checkpoint path (takes precedence over bc_init.pt)",
    )
    p.add_argument("--skip-ppo", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-env-kwargs-json", type=str, default="")
    p.add_argument("--skip-last-eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
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

    base_defaults = _default_cli_config(parser)
    cfg = resolve_config(
        defaults=base_defaults,
        config_file=pre.config_file,
        config_section=pre.config_section,
        overrides=list(pre.set or []),
    )
    unknown = sorted(k for k in cfg.keys() if k not in base_defaults.keys())
    if unknown:
        raise ValueError(f"unknown config keys for run_pipeline: {', '.join(unknown)}")

    parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.model_config_file = coerce_optional_path(args.model_config_file, dot_is_none=True)
    args.ppo_init_model = coerce_optional_path(args.ppo_init_model, dot_is_none=True)
    return args


def main() -> int:
    return int(run_pipeline(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
