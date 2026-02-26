from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..algorithms.ppo.config import PPOConfig
from ..algorithms.ppo.rollout_buffer import RolloutBuffer
from ..algorithms.ppo.trainer import PPOTrainer
from ..domains.ahc061.opponent_bayes import ensure_cpp_bayes_backend
from ..env.api import extract_action_mask
from ..env.factory import (
    done_to_tensor,
    infer_env_spec,
    make_vector_env,
    obs_to_tensor,
    reset_env,
    reward_to_tensor,
    unwrap_action,
    unwrap_vec_normalize,
)
from ..models import build_agent, load_model_config_from_sources, normalize_model_config
from ..runtime.checkpoint import load_agent_checkpoint, save_agent_checkpoint
from ..runtime.episode_stats import extract_completed_episode_stats
from ..runtime.experiment import coerce_optional_path, create_run_layout, make_run_name, resolve_config, to_jsonable, update_manifest
from ..runtime.log_utils import get_logger
from ..runtime.metrics import summarize
from ..runtime.tracking import MetricTracker

logger = get_logger("train_ppo")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train discrete-board policy with PPO.")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="train_ppo", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--env-kwargs-json", type=str, default="{}")
    p.add_argument("--run-dir", type=Path, default=Path("reinforce/outputs/ppo_runs"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--init-model", type=Path, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume-from", type=Path, default=None, help="checkpoint path to resume from (defaults to <run>/models/last.pt when --resume)")

    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
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
    p.add_argument("--flatten-obs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vector-env", choices=["sync", "async"], default="sync")
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
    p.add_argument("--val-vector-env", "--eval-vector-env", dest="eval_vector_env", choices=["sync", "async"], default="sync")
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


def _normalize_bayes_backend_name(x: Any) -> str:
    b = str(x).strip().lower()
    if not b:
        b = "auto"
    if b not in ("auto", "python", "cpp"):
        raise ValueError(f"unsupported bayes_backend={x!r}; expected auto|python|cpp")
    return b


def _prepare_cpp_bayes_backend_if_needed(*, env_id: str, train_env_kwargs: dict[str, Any], eval_env_kwargs: dict[str, Any]) -> dict[str, Any]:
    if str(env_id) != "AHC061Local-v0":
        return {"enabled": False, "prepared": False, "train_backend": "", "eval_backend": ""}
    train_backend = _normalize_bayes_backend_name(train_env_kwargs.get("bayes_backend", "auto"))
    eval_backend = _normalize_bayes_backend_name(eval_env_kwargs.get("bayes_backend", train_backend))
    needs_cpp = bool(train_backend in ("auto", "cpp") or eval_backend in ("auto", "cpp"))
    out = {
        "enabled": needs_cpp,
        "prepared": False,
        "train_backend": train_backend,
        "eval_backend": eval_backend,
    }
    if not needs_cpp:
        return out

    ok = ensure_cpp_bayes_backend(build_if_missing=True, force_build=False, verbose=False)
    if not ok and (train_backend == "cpp" or eval_backend == "cpp"):
        raise RuntimeError("bayes_backend=cpp was requested but cpp backend build/import failed")
    if not ok:
        logger.warning("cpp bayes backend unavailable; auto backend will fallback to python")
    out["prepared"] = bool(ok)
    return out


def _resolve_model_config(args: argparse.Namespace) -> dict[str, Any]:
    explicit_cfg = load_model_config_from_sources(
        model_config_file=args.model_config_file,
        model_config_json=args.model_config_json,
    )
    model_class = str(args.model_class).strip()
    if explicit_cfg is not None and model_class:
        explicit_cfg["type"] = model_class

    return normalize_model_config(
        explicit_cfg,
        default_type=model_class or "DiscreteBoardAgent",
    )


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
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        clip_coef_schedule=str(args.clip_coef_schedule),
        clip_coef_final=args.clip_coef_final,
        clip_coef_schedule_expr=str(args.clip_coef_schedule_expr),
        save_interval=args.save_interval,
    )


def _validate_ppo_cfg(cfg: PPOConfig) -> None:
    if int(cfg.total_timesteps) <= 0:
        raise ValueError(f"total_timesteps must be positive: {cfg.total_timesteps}")
    if int(cfg.num_envs) <= 0:
        raise ValueError(f"num_envs must be positive: {cfg.num_envs}")
    if int(cfg.num_steps) <= 0:
        raise ValueError(f"num_steps must be positive: {cfg.num_steps}")
    if int(cfg.num_minibatches) <= 0:
        raise ValueError(f"num_minibatches must be positive: {cfg.num_minibatches}")
    if int(cfg.update_epochs) <= 0:
        raise ValueError(f"update_epochs must be positive: {cfg.update_epochs}")
    if float(cfg.learning_rate) <= 0.0:
        raise ValueError(f"learning_rate must be positive: {cfg.learning_rate}")
    if float(cfg.max_grad_norm) <= 0.0:
        raise ValueError(f"max_grad_norm must be positive: {cfg.max_grad_norm}")
    if float(cfg.clip_coef) <= 0.0:
        raise ValueError(f"clip_coef must be positive: {cfg.clip_coef}")
    if str(cfg.clip_coef_schedule).lower() not in ("constant", "linear", "cosine"):
        raise ValueError(f"clip_coef_schedule must be one of constant|linear|cosine: {cfg.clip_coef_schedule}")
    if cfg.clip_coef_final is not None and float(cfg.clip_coef_final) < 0.0:
        raise ValueError(f"clip_coef_final must be >= 0 when set: {cfg.clip_coef_final}")
    if cfg.clip_range_vf is not None and float(cfg.clip_range_vf) <= 0.0:
        raise ValueError(f"clip_range_vf must be positive when set: {cfg.clip_range_vf}")
    if str(cfg.clip_range_vf_schedule).lower() not in ("constant", "linear", "cosine"):
        raise ValueError(
            "clip_range_vf_schedule must be one of constant|linear|cosine: "
            f"{cfg.clip_range_vf_schedule}"
        )
    if cfg.clip_range_vf_final is not None and float(cfg.clip_range_vf_final) < 0.0:
        raise ValueError(f"clip_range_vf_final must be >= 0 when set: {cfg.clip_range_vf_final}")
    if float(cfg.ent_coef) < 0.0:
        raise ValueError(f"ent_coef must be >= 0: {cfg.ent_coef}")
    if str(cfg.ent_coef_schedule).lower() not in ("constant", "linear", "cosine"):
        raise ValueError(f"ent_coef_schedule must be one of constant|linear|cosine: {cfg.ent_coef_schedule}")
    if cfg.ent_coef_final is not None and float(cfg.ent_coef_final) < 0.0:
        raise ValueError(f"ent_coef_final must be >= 0 when set: {cfg.ent_coef_final}")
    if cfg.target_kl is not None and float(cfg.target_kl) <= 0.0:
        raise ValueError(f"target_kl must be positive when set: {cfg.target_kl}")

    batch_size = int(cfg.batch_size)
    if batch_size <= 1 and bool(cfg.norm_adv):
        raise ValueError(
            f"batch_size={batch_size} is too small for advantage normalization; "
            "increase num_envs*num_steps or set --no-norm-adv"
        )
    if batch_size % int(cfg.num_minibatches) != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by num_minibatches={cfg.num_minibatches}"
        )
    minibatch_size = int(cfg.minibatch_size)
    if minibatch_size <= 0:
        raise ValueError(
            f"invalid minibatch_size={minibatch_size}; check num_envs/num_steps/num_minibatches"
        )
    if minibatch_size <= 1 and bool(cfg.norm_adv):
        raise ValueError(
            f"minibatch_size={minibatch_size} is too small for advantage normalization; "
            "reduce num_minibatches or set --no-norm-adv"
        )


def _resolve_vecnorm_gamma(args: argparse.Namespace, cfg: PPOConfig) -> float:
    if args.vecnorm_gamma is None:
        return float(cfg.gamma)
    return float(args.vecnorm_gamma)


def _validate_vecnorm_args(args: argparse.Namespace, cfg: PPOConfig) -> None:
    if not bool(args.vecnorm):
        return
    if float(args.vecnorm_clip_obs) <= 0.0:
        raise ValueError(f"vecnorm_clip_obs must be positive: {args.vecnorm_clip_obs}")
    if float(args.vecnorm_clip_reward) <= 0.0:
        raise ValueError(f"vecnorm_clip_reward must be positive: {args.vecnorm_clip_reward}")
    if float(args.vecnorm_epsilon) <= 0.0:
        raise ValueError(f"vecnorm_epsilon must be positive: {args.vecnorm_epsilon}")
    vgamma = _resolve_vecnorm_gamma(args, cfg)
    if not np.isfinite(vgamma) or vgamma <= 0.0:
        raise ValueError(f"vecnorm_gamma must be positive: {vgamma}")


def _split_csv_args(text: str) -> list[str]:
    raw = str(text).strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_schedule_expr(
    expr: str,
    *,
    default_kind: str,
    default_start: float,
    default_end: float,
) -> tuple[Callable[[float], float], str]:
    text = str(expr).strip()
    if not text:
        k = str(default_kind).strip().lower()
        return (
            lambda p: float(_scheduled_scalar(k, float(default_start), float(default_end), float(p))),
            f"{k}({float(default_start):.12g},{float(default_end):.12g})",
        )

    low = text.lower()
    if low in ("constant", "linear", "cosine"):
        return (
            lambda p: float(_scheduled_scalar(low, float(default_start), float(default_end), float(p))),
            f"{low}({float(default_start):.12g},{float(default_end):.12g})",
        )
    if low == "exp":
        a = float(default_start)
        b = float(default_end)
        if a <= 0.0 or b <= 0.0:
            raise ValueError(f"exp schedule requires positive start/end, got ({a}, {b})")
        return (lambda p: float(a * ((b / a) ** float(np.clip(p, 0.0, 1.0)))), f"exp({a:.12g},{b:.12g})")

    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", text)
    if m is None:
        raise ValueError(
            f"invalid schedule expr={expr!r}; expected e.g. constant(0.1), linear(0.2,0.05), "
            "cosine(0.2,0.01), exp(3e-4,1e-5), piecewise(0:0.2,0.5:0.1,1:0.05)"
        )
    name = str(m.group(1)).strip().lower()
    args = _split_csv_args(m.group(2))

    if name == "constant":
        if len(args) > 1:
            raise ValueError(f"constant(...) expects 0 or 1 args, got: {expr}")
        v = float(default_start if len(args) == 0 else float(args[0]))
        return (lambda _p: float(v), f"constant({v:.12g})")

    if name in ("linear", "cosine", "exp"):
        if len(args) not in (0, 2):
            raise ValueError(f"{name}(...) expects 0 or 2 args, got: {expr}")
        a = float(default_start if len(args) == 0 else float(args[0]))
        b = float(default_end if len(args) == 0 else float(args[1]))
        if name == "exp":
            if a <= 0.0 or b <= 0.0:
                raise ValueError(f"exp schedule requires positive start/end, got ({a}, {b})")
            return (lambda p: float(a * ((b / a) ** float(np.clip(p, 0.0, 1.0)))), f"exp({a:.12g},{b:.12g})")
        return (
            lambda p: float(_scheduled_scalar(name, float(a), float(b), float(p))),
            f"{name}({a:.12g},{b:.12g})",
        )

    if name == "piecewise":
        if len(args) < 2:
            raise ValueError(f"piecewise(...) expects >=2 points, got: {expr}")
        pts: list[tuple[float, float]] = []
        for tok in args:
            if ":" not in tok:
                raise ValueError(f"invalid piecewise point={tok!r}; expected p:v")
            ptxt, vtxt = tok.split(":", 1)
            p = float(ptxt.strip())
            v = float(vtxt.strip())
            if not np.isfinite(p) or not np.isfinite(v):
                raise ValueError(f"invalid piecewise point={tok!r}; non-finite value")
            pts.append((p, v))
        pts.sort(key=lambda x: x[0])
        for i in range(1, len(pts)):
            if abs(pts[i][0] - pts[i - 1][0]) <= 1e-12:
                raise ValueError(f"piecewise has duplicate progress points: {pts[i][0]}")

        def _piecewise(progress: float) -> float:
            p = float(np.clip(progress, 0.0, 1.0))
            if p <= pts[0][0]:
                return float(pts[0][1])
            for j in range(1, len(pts)):
                p0, v0 = pts[j - 1]
                p1, v1 = pts[j]
                if p <= p1:
                    if p1 <= p0 + 1e-12:
                        return float(v1)
                    t = (p - p0) / (p1 - p0)
                    return float(v0 + t * (v1 - v0))
            return float(pts[-1][1])

        desc = "piecewise(" + ",".join(f"{p:.6g}:{v:.6g}" for p, v in pts) + ")"
        return _piecewise, desc

    raise ValueError(
        f"unknown schedule expr kind={name!r}; supported: constant, linear, cosine, exp, piecewise"
    )


def _validate_schedule_samples(name: str, fn: Callable[[float], float], *, nonnegative: bool) -> None:
    for p in (0.0, 0.5, 1.0):
        v = float(fn(float(p)))
        if not np.isfinite(v):
            raise ValueError(f"{name} schedule produced non-finite value at progress={p}: {v}")
        if nonnegative and v < 0.0:
            raise ValueError(f"{name} schedule produced negative value at progress={p}: {v}")


def _build_schedule_fns(cfg: PPOConfig) -> dict[str, tuple[Callable[[float], float] | None, str]]:
    lr_start = float(cfg.learning_rate)
    lr_kind = "linear"
    lr_end = 0.0
    lr_fn, lr_desc = _parse_schedule_expr(
        str(cfg.learning_rate_schedule),
        default_kind=lr_kind,
        default_start=lr_start,
        default_end=lr_end,
    )

    ent_start = float(cfg.ent_coef)
    ent_end = float(ent_start if cfg.ent_coef_final is None else cfg.ent_coef_final)
    ent_fn, ent_desc = _parse_schedule_expr(
        str(cfg.ent_coef_schedule_expr),
        default_kind=str(cfg.ent_coef_schedule),
        default_start=ent_start,
        default_end=ent_end,
    )

    clip_start = float(cfg.clip_coef)
    clip_end = float(clip_start if cfg.clip_coef_final is None else cfg.clip_coef_final)
    clip_fn, clip_desc = _parse_schedule_expr(
        str(cfg.clip_coef_schedule_expr),
        default_kind=str(cfg.clip_coef_schedule),
        default_start=clip_start,
        default_end=clip_end,
    )

    vf_fn: Callable[[float], float] | None = None
    vf_desc = "disabled"
    if cfg.clip_range_vf is not None:
        vf_start = float(cfg.clip_range_vf)
        vf_end = float(vf_start if cfg.clip_range_vf_final is None else cfg.clip_range_vf_final)
        vf_fn, vf_desc = _parse_schedule_expr(
            str(cfg.clip_range_vf_schedule_expr),
            default_kind=str(cfg.clip_range_vf_schedule),
            default_start=vf_start,
            default_end=vf_end,
        )
    elif (
        str(cfg.clip_range_vf_schedule_expr).strip()
        or cfg.clip_range_vf_final is not None
        or str(cfg.clip_range_vf_schedule).strip().lower() != "constant"
    ):
        raise ValueError(
            "clip_range_vf schedule was configured but clip_range_vf is unset; "
            "set clip_range_vf to enable value clipping schedule"
        )

    return {
        "lr": (lr_fn, lr_desc),
        "ent": (ent_fn, ent_desc),
        "clip": (clip_fn, clip_desc),
        "vf_clip": (vf_fn, vf_desc),
    }


def _validate_schedule_args(cfg: PPOConfig) -> None:
    schedules = _build_schedule_fns(cfg)
    _validate_schedule_samples("learning_rate", schedules["lr"][0], nonnegative=True)
    _validate_schedule_samples("ent_coef", schedules["ent"][0], nonnegative=True)
    _validate_schedule_samples("clip_coef", schedules["clip"][0], nonnegative=True)
    vf_fn = schedules["vf_clip"][0]
    if vf_fn is not None:
        _validate_schedule_samples("clip_range_vf", vf_fn, nonnegative=True)


def _schedule_progress(iteration: int, total_iterations: int) -> float:
    if total_iterations <= 1:
        return 1.0
    p = float(iteration - 1) / float(total_iterations - 1)
    return float(np.clip(p, 0.0, 1.0))


def _scheduled_scalar(kind: str, start: float, end: float, progress: float) -> float:
    k = str(kind).strip().lower()
    if k == "constant":
        return float(start)
    if k == "linear":
        return float(start + (end - start) * progress)
    if k == "cosine":
        return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(f"unknown schedule kind: {kind}")


def _mask_to_tensor(mask: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if mask is None:
        return None
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _extract_terminal_obs_from_infos(infos: Any, idx: int) -> np.ndarray | None:
    if not isinstance(infos, dict):
        return None

    def _pick_from_container(container: Any, i: int) -> Any:
        if container is None:
            return None
        if isinstance(container, np.ndarray):
            if container.ndim == 0:
                return None
            if i >= int(container.shape[0]):
                return None
            return container[i]
        if isinstance(container, (list, tuple)):
            if i >= len(container):
                return None
            return container[i]
        return None

    def _mask_allows(mask_obj: Any, i: int) -> bool:
        if mask_obj is None:
            return True
        try:
            m = np.asarray(mask_obj, dtype=np.bool_).reshape(-1)
        except Exception:
            return True
        if i >= m.shape[0]:
            return False
        return bool(m[i])

    # gymnasium vector-info style: final_observation / terminal_observation (+ optional masks)
    for key in ("final_observation", "terminal_observation"):
        if key not in infos:
            continue
        mask_key = f"_{key}"
        if not _mask_allows(infos.get(mask_key), idx):
            continue
        cand = _pick_from_container(infos.get(key), idx)
        if cand is None:
            continue
        arr = np.asarray(cand, dtype=np.float32)
        if arr.size == 0:
            continue
        return arr

    # fallback: final_info list may include terminal observation per env
    finfo = infos.get("final_info")
    item = _pick_from_container(finfo, idx)
    if isinstance(item, dict):
        for key in ("terminal_observation", "final_observation"):
            if key not in item:
                continue
            arr = np.asarray(item.get(key), dtype=np.float32)
            if arr.size == 0:
                continue
            return arr
    return None


def _apply_time_limit_bootstrap(
    *,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    infos: Any,
    agent: torch.nn.Module,
    gamma: float,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    out = np.asarray(rewards, dtype=np.float32).copy()
    term = np.asarray(terminations, dtype=np.bool_).reshape(-1)
    trunc = np.asarray(truncations, dtype=np.bool_).reshape(-1)
    trunc_only = np.logical_and(trunc, np.logical_not(term))
    idxs = np.flatnonzero(trunc_only)
    if idxs.size == 0:
        return out, 0

    applied = 0
    for i in idxs.tolist():
        obs_i = _extract_terminal_obs_from_infos(infos, int(i))
        if obs_i is None:
            # In gymnasium vector env (next-step autoreset), next_obs at done step is terminal obs.
            obs_i = np.asarray(next_obs[i], dtype=np.float32)
        with torch.no_grad():
            v = agent.get_value(obs_to_tensor(np.asarray(obs_i, dtype=np.float32)[None, ...], device)).view(-1)
        out[i] = float(out[i]) + float(gamma * float(v[0].item()))
        applied += 1
    return out, int(applied)


def _append_recent_episode_metrics(
    infos: Any,
    recent_total_returns: list[float],
    recent_illegal_penalties: list[float],
    recent_terminal_scores: list[float],
    recent_terminal_game_scores: list[float],
) -> None:
    for s in extract_completed_episode_stats(infos):
        recent_total_returns.append(float(s["total_return"]))
        recent_illegal_penalties.append(float(s["illegal_penalty_return"]))
        recent_terminal_scores.append(float(s["terminal_score_return"]))
        recent_terminal_game_scores.append(float(s.get("terminal_game_score_return", 0.0)))


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    y = returns.reshape(-1)
    y_pred = values.reshape(-1)
    var_y = torch.var(y, unbiased=False)
    if not torch.isfinite(var_y) or float(var_y.item()) <= 1e-12:
        return float("nan")
    ev = 1.0 - torch.var(y - y_pred, unbiased=False) / (var_y + 1e-12)
    return float(ev.item()) if torch.isfinite(ev) else float("nan")


def _fmt_float(x: Any, digits: int = 6) -> str:
    try:
        v = float(x)
    except Exception:
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _safe_int_or_none(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _vecnorm_state_or_none(envs: Any) -> dict[str, Any] | None:
    vec = unwrap_vec_normalize(envs)
    if vec is None:
        return None
    try:
        return vec.state_dict()
    except Exception:
        return None


def _sync_vecnorm_stats(train_envs: Any, eval_envs: Any) -> None:
    train_vec = unwrap_vec_normalize(train_envs)
    eval_vec = unwrap_vec_normalize(eval_envs)
    if train_vec is None or eval_vec is None:
        return
    eval_vec.load_state_dict(train_vec.state_dict())
    eval_vec.set_training(False)


def _format_train_log_line(row: dict[str, Any]) -> str:
    parts = [
        f"iter={int(row['iteration']):5d}",
        f"step={int(row['global_step']):9d}",
        f"sps={int(row['sps']):5d}",
        (
            "recent(return/game_score)="
            f"{_fmt_float(row['mean_recent_return_100_total'], 5)}"
            f"/{_fmt_float(row.get('mean_recent_return_100_terminal_game_score'), 2)}"
        ),
    ]
    if "periodic_val_mean_return" in row:
        parts.append(
            (
                "val(return/game_score)="
                f"{_fmt_float(row['periodic_val_mean_return'], 5)}"
                f"/{_fmt_float(row.get('periodic_val_mean_terminal_game_score'), 2)}"
            )
        )
    return " | ".join(parts)


def _set_vector_env_case_counter(eval_envs: Any, seed_start: int) -> None:
    env_list = getattr(eval_envs, "envs", None)
    if not isinstance(env_list, (list, tuple)):
        return
    for idx, env in enumerate(env_list):
        base = getattr(env, "unwrapped", env)
        setter = getattr(base, "set_case_counter", None)
        if callable(setter):
            setter(int(seed_start) + int(idx))


def _run_periodic_val(
    *,
    agent: torch.nn.Module,
    eval_envs: Any,
    spec: Any,
    device: torch.device,
    episodes: int,
    seed_start: int,
    deterministic: bool,
    use_action_mask: bool,
) -> dict[str, Any]:
    was_training = bool(agent.training)
    agent.eval()
    _set_vector_env_case_counter(eval_envs, int(seed_start))
    episode_returns: list[float] = []
    episode_illegal_penalties: list[float] = []
    episode_terminal_scores: list[float] = []
    episode_terminal_game_scores: list[float] = []
    episode_game_score_ratio: list[float] = []
    episode_game_score_self: list[float] = []
    episode_game_score_enemy_max: list[float] = []

    try:
        for epi in range(episodes):
            obs, info = eval_envs.reset(seed=seed_start + epi)
            done = np.array([False], dtype=np.bool_)
            ep_ret = 0.0
            while not done[0]:
                mask_np = extract_action_mask(obs, info, spec.action_dim) if use_action_mask else None
                mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device) if mask_np is not None else None
                with torch.no_grad():
                    act_t = agent.act(obs_to_tensor(obs, device), action_mask=mask_t, deterministic=deterministic)
                action = unwrap_action(act_t).astype(np.int64)
                if action.ndim == 0:
                    action = action.reshape(1)
                next_obs, reward, terminations, truncations, info = eval_envs.step(action)
                done = np.logical_or(terminations, truncations)
                ep_ret += float(reward[0])
                obs = next_obs
                if done[0]:
                    parsed = extract_completed_episode_stats(info)
                    if parsed:
                        p = parsed[0]
                        ep_ret = float(p["total_return"])
                        episode_illegal_penalties.append(float(p["illegal_penalty_return"]))
                        episode_terminal_scores.append(float(p["terminal_score_return"]))
                        episode_terminal_game_scores.append(float(p.get("terminal_game_score_return", 0.0)))
                        if np.isfinite(float(p["final_score_ratio"])):
                            episode_game_score_ratio.append(float(p["final_score_ratio"]))
                        if np.isfinite(float(p["final_self_score"])):
                            episode_game_score_self.append(float(p["final_self_score"]))
                        if np.isfinite(float(p["final_enemy_max_score"])):
                            episode_game_score_enemy_max.append(float(p["final_enemy_max_score"]))
                    else:
                        episode_illegal_penalties.append(0.0)
                        episode_terminal_scores.append(0.0)
                        episode_terminal_game_scores.append(0.0)
            episode_returns.append(ep_ret)
    finally:
        if was_training:
            agent.train()

    return {
        "episodes": int(episodes),
        "return": summarize(episode_returns).as_dict(),
        "illegal_penalty": summarize(episode_illegal_penalties).as_dict(),
        "terminal_score": summarize(episode_terminal_scores).as_dict(),
        "terminal_game_score": summarize(episode_terminal_game_scores).as_dict(),
        "game_score_ratio": summarize(episode_game_score_ratio).as_dict(),
        "game_score_self": summarize(episode_game_score_self).as_dict(),
        "game_score_enemy_max": summarize(episode_game_score_enemy_max).as_dict(),
    }


def main() -> int:
    args = parse_args()
    cfg = args_to_cfg(args)
    _validate_ppo_cfg(cfg)
    _validate_schedule_args(cfg)
    _validate_vecnorm_args(args, cfg)
    device = choose_device(args.device)
    env_kwargs = parse_env_kwargs(args.env_kwargs_json)
    eval_env_kwargs = (
        parse_env_kwargs(args.eval_env_kwargs_json) if str(args.eval_env_kwargs_json).strip() else dict(env_kwargs)
    )
    bayes_backend_meta = _prepare_cpp_bayes_backend_if_needed(
        env_id=str(args.env_id),
        train_env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
    )
    if bool(bayes_backend_meta.get("enabled")):
        logger.info(
            "bayes_backend: train=%s eval=%s cpp_prepared=%s",
            bayes_backend_meta.get("train_backend"),
            bayes_backend_meta.get("eval_backend"),
            bayes_backend_meta.get("prepared"),
        )
    model_config = _resolve_model_config(args)

    run_name = args.run_name or make_run_name(args.env_id.replace("/", "_"), seed=args.seed)
    layout = create_run_layout(args.run_dir, run_name)
    resume_enabled = bool(args.resume or args.resume_from is not None)
    resume_from = args.resume_from
    if resume_enabled and resume_from is None:
        resume_from = layout.models_dir / "last.pt"
    if resume_enabled and (resume_from is None or not Path(resume_from).exists()):
        raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")

    started_at = time.time()
    envs = None
    eval_envs = None
    train_vecnorm = None
    eval_vecnorm = None
    tracker: MetricTracker | None = None
    global_step = 0
    best_mean_return = -1e100
    best_metric_name = "mean_recent_return_100_terminal_game_score"
    best_metric_source = ""
    resume_meta: dict[str, Any] = {}
    resume_payload: dict[str, Any] | None = None
    recent_total_returns: list[float] = []
    recent_illegal_penalties: list[float] = []
    recent_terminal_scores: list[float] = []
    recent_terminal_game_scores: list[float] = []
    restored_eval_round: int | None = None
    restored_next_eval_step: int | None = None
    restored_next_checkpoint_step: int | None = None

    try:
        vecnorm_gamma = _resolve_vecnorm_gamma(args, cfg)
        envs = make_vector_env(
            args.env_id,
            num_envs=cfg.num_envs,
            seed=cfg.seed,
            capture_video=args.capture_video,
            run_name=run_name,
            flatten_obs=args.flatten_obs,
            vector_env=args.vector_env,
            env_kwargs=env_kwargs,
            vecnorm=bool(args.vecnorm),
            vecnorm_norm_obs=bool(args.vecnorm_norm_obs),
            vecnorm_norm_reward=bool(args.vecnorm_norm_reward),
            vecnorm_clip_obs=float(args.vecnorm_clip_obs),
            vecnorm_clip_reward=float(args.vecnorm_clip_reward),
            vecnorm_epsilon=float(args.vecnorm_epsilon),
            vecnorm_gamma=float(vecnorm_gamma),
            vecnorm_training=True,
        )
        train_vecnorm = unwrap_vec_normalize(envs)
        spec = infer_env_spec(envs)

        if resume_enabled:
            assert resume_from is not None
            if args.init_model is not None:
                logger.info("resume is enabled; --init-model is ignored: %s", args.init_model)
            agent, init_meta = load_agent_checkpoint(resume_from, device=device)
            resume_payload = torch.load(resume_from, map_location=device, weights_only=False)
            resume_meta = dict(init_meta)
            model_config = normalize_model_config(
                getattr(agent, "model_config", None),
                default_type=agent.__class__.__name__,
            )
            if tuple(agent.obs_shape) != tuple(spec.obs_shape):
                raise ValueError(f"obs_shape mismatch: model={agent.obs_shape}, env={spec.obs_shape}")
            if int(agent.action_dim) != int(spec.action_dim):
                raise ValueError(f"action_dim mismatch: model={agent.action_dim}, env={spec.action_dim}")
            logger.info("loaded resume model: %s", resume_from)
        elif args.init_model is not None:
            agent, init_meta = load_agent_checkpoint(args.init_model, device=device)
            model_config = normalize_model_config(
                getattr(agent, "model_config", None),
                default_type=agent.__class__.__name__,
            )
            if tuple(agent.obs_shape) != tuple(spec.obs_shape):
                raise ValueError(f"obs_shape mismatch: model={agent.obs_shape}, env={spec.obs_shape}")
            if int(agent.action_dim) != int(spec.action_dim):
                raise ValueError(f"action_dim mismatch: model={agent.action_dim}, env={spec.action_dim}")
            logger.info("loaded init model: %s", args.init_model)
        else:
            agent, model_config = build_agent(
                obs_shape=spec.obs_shape,
                action_dim=spec.action_dim,
                model_config=model_config,
            )
            agent = agent.to(device)
            init_meta = {}

        optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        if resume_payload is not None and "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        trainer = PPOTrainer(cfg, agent, optimizer)

        log_jsonl = layout.logs_dir / "train_metrics.jsonl"
        periodic_val_jsonl = layout.logs_dir / "periodic_val_metrics.jsonl"
        best_model = layout.models_dir / "best.pt"
        last_model = layout.models_dir / "last.pt"

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        if resume_enabled:
            global_step = int(resume_meta.get("global_step", 0))
            if global_step < 0:
                global_step = 0
            resume_cfg_raw = resume_meta.get("cfg")
            if isinstance(resume_cfg_raw, dict):
                prev_num_envs = _safe_int_or_none(resume_cfg_raw.get("num_envs"))
                prev_num_steps = _safe_int_or_none(resume_cfg_raw.get("num_steps"))
                prev_batch_size = _safe_int_or_none(resume_cfg_raw.get("batch_size"))
                if prev_batch_size is None and prev_num_envs is not None and prev_num_steps is not None:
                    prev_batch_size = int(prev_num_envs * prev_num_steps)

                mismatch: list[str] = []
                if prev_num_envs is not None and prev_num_envs != int(cfg.num_envs):
                    mismatch.append(f"num_envs(prev={prev_num_envs}, now={cfg.num_envs})")
                if prev_num_steps is not None and prev_num_steps != int(cfg.num_steps):
                    mismatch.append(f"num_steps(prev={prev_num_steps}, now={cfg.num_steps})")
                if prev_batch_size is not None and prev_batch_size != int(cfg.batch_size):
                    mismatch.append(f"batch_size(prev={prev_batch_size}, now={cfg.batch_size})")

                if mismatch:
                    raise ValueError(
                        "resume config mismatch for rollout geometry: "
                        + ", ".join(mismatch)
                        + ". Resume requires the same num_envs/num_steps as the checkpoint."
                    )

            if global_step % cfg.batch_size != 0:
                raise ValueError(
                    "resume global_step is not aligned to current batch_size: "
                    f"step={global_step}, batch_size={cfg.batch_size}. "
                    "Use the same num_envs/num_steps as the checkpoint, or start a new run."
                )
            rr_total = resume_meta.get("recent_total_returns")
            rr_illegal = resume_meta.get("recent_illegal_penalties")
            rr_terminal = resume_meta.get("recent_terminal_scores")
            rr_terminal_game = resume_meta.get("recent_terminal_game_scores")
            if isinstance(rr_total, list):
                recent_total_returns = [float(x) for x in rr_total]
            if isinstance(rr_illegal, list):
                recent_illegal_penalties = [float(x) for x in rr_illegal]
            if isinstance(rr_terminal, list):
                recent_terminal_scores = [float(x) for x in rr_terminal]
            if isinstance(rr_terminal_game, list):
                recent_terminal_game_scores = [float(x) for x in rr_terminal_game]
            if "best_mean_return" in resume_meta:
                try:
                    best_mean_return = float(resume_meta["best_mean_return"])
                except Exception:
                    pass
            if isinstance(resume_meta.get("best_metric_name"), str):
                best_metric_name = str(resume_meta["best_metric_name"])
            if isinstance(resume_meta.get("best_metric_source"), str):
                best_metric_source = str(resume_meta["best_metric_source"])
            if "eval_round" in resume_meta:
                try:
                    restored_eval_round = int(resume_meta["eval_round"])
                except Exception:
                    restored_eval_round = None
            if "next_eval_step" in resume_meta:
                try:
                    restored_next_eval_step = int(resume_meta["next_eval_step"])
                except Exception:
                    restored_next_eval_step = None
            if "next_checkpoint_step" in resume_meta:
                try:
                    restored_next_checkpoint_step = int(resume_meta["next_checkpoint_step"])
                except Exception:
                    restored_next_checkpoint_step = None

        incoming_vec_state = None
        incoming_vec_source = ""
        if resume_enabled:
            cand = resume_meta.get("vecnormalize_state")
            if isinstance(cand, dict):
                incoming_vec_state = cand
                incoming_vec_source = "resume"
        elif isinstance(init_meta, dict):
            cand = init_meta.get("vecnormalize_state")
            if isinstance(cand, dict):
                incoming_vec_state = cand
                incoming_vec_source = "init_model"

        if train_vecnorm is not None:
            if isinstance(incoming_vec_state, dict):
                train_vecnorm.load_state_dict(incoming_vec_state)
                train_vecnorm.set_training(True)
                logger.info("restored vecnormalize state from %s checkpoint", incoming_vec_source)
            elif resume_enabled:
                logger.warning("vecnorm is enabled but resume checkpoint has no vecnormalize_state")
            logger.info(
                "vecnorm: enabled=true norm_obs=%s norm_reward=%s clip_obs=%.4f clip_reward=%.4f eps=%g gamma=%.6f",
                bool(train_vecnorm.norm_obs),
                bool(train_vecnorm.norm_reward),
                float(train_vecnorm.clip_obs),
                float(train_vecnorm.clip_reward),
                float(train_vecnorm.epsilon),
                float(train_vecnorm.gamma),
            )
        else:
            if isinstance(incoming_vec_state, dict):
                logger.warning("checkpoint has vecnormalize_state but vecnorm is disabled; observations/rewards will be unnormalized")
            logger.info("vecnorm: enabled=false")

        run_config = to_jsonable(
            {
                "args": vars(args),
                "ppo_cfg": asdict(cfg),
                "env_id": args.env_id,
                "env_kwargs": env_kwargs,
                "val_env_kwargs": eval_env_kwargs,
                "bayes_backend": bayes_backend_meta,
                "obs_shape": list(spec.obs_shape),
                "action_dim": int(spec.action_dim),
                "model_config": to_jsonable(model_config),
                "vecnormalize": {
                    "enabled": bool(train_vecnorm is not None),
                    "norm_obs": bool(args.vecnorm_norm_obs),
                    "norm_reward_train": bool(args.vecnorm_norm_reward),
                    "norm_reward_val": bool(args.vecnorm_eval_norm_reward),
                    "clip_obs": float(args.vecnorm_clip_obs),
                    "clip_reward": float(args.vecnorm_clip_reward),
                    "epsilon": float(args.vecnorm_epsilon),
                    "gamma": float(vecnorm_gamma),
                },
                "layout": layout.as_dict(),
            }
        )
        (layout.config_dir / "train_ppo.args.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        tracker = MetricTracker(
            layout.root,
            run_name=run_name,
            enable_tensorboard=bool(args.tensorboard),
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
            config=run_config,
        )

        update_manifest(
            layout,
            {
                "job": "train_ppo",
                "status": "running",
                "run_name": run_name,
                "layout": layout.as_dict(),
                "env": {"env_id": args.env_id, "env_kwargs": env_kwargs},
                "config": run_config,
                "artifacts": {
                    "logs": {"train_metrics_jsonl": str(log_jsonl)},
                    "models": {"best": str(best_model), "last": str(last_model)},
                },
                "timestamps": {"started_at": started_at},
            },
        )

        session_start_global_step = int(global_step)
        start_time = time.time()
        periodic_val_enabled = bool(args.eval_interval_steps > 0 and args.eval_episodes > 0)
        eval_at_start_enabled = bool(args.eval_at_start and args.eval_episodes > 0)
        eval_runtime_enabled = bool(periodic_val_enabled or eval_at_start_enabled)
        log_interval_iters = max(1, int(args.log_interval_iters))
        if periodic_val_enabled and best_metric_name in (
            "mean_recent_return_100",
            "mean_recent_return_100_terminal_game_score",
            "periodic_val_mean_return",
        ):
            best_metric_name = "periodic_val_mean_terminal_game_score"
        elif (not periodic_val_enabled) and best_metric_name in ("mean_recent_return_100", "periodic_val_mean_return"):
            best_metric_name = "mean_recent_return_100_terminal_game_score"
        if (
            eval_runtime_enabled
            and str(args.eval_vector_env) != "sync"
            and str(eval_env_kwargs.get("case_select_mode", "")) == "sequential"
        ):
            logger.warning(
                "val_vector_env=async with case_select_mode=sequential may weaken fixed-seed reproducibility; prefer --val-vector-env sync"
            )
        eval_spec = spec
        next_eval_step = int(max(1, args.eval_interval_steps))
        eval_round = 0
        if resume_enabled:
            if restored_eval_round is not None and restored_eval_round >= 0:
                eval_round = int(restored_eval_round)
            elif periodic_val_enabled and int(args.eval_interval_steps) > 0:
                eval_round = int(global_step // int(args.eval_interval_steps))
                if bool(eval_at_start_enabled) and global_step > 0:
                    eval_round += 1
            if restored_next_eval_step is not None and restored_next_eval_step > 0:
                next_eval_step = int(restored_next_eval_step)
            elif periodic_val_enabled and int(args.eval_interval_steps) > 0:
                interval = int(args.eval_interval_steps)
                next_eval_step = int(((global_step // interval) + 1) * interval)

        checkpoint_interval_steps = int(args.checkpoint_interval_steps)
        checkpoint_dir = layout.models_dir / "checkpoints"
        next_checkpoint_step = 0
        if checkpoint_interval_steps > 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if resume_enabled and restored_next_checkpoint_step is not None and restored_next_checkpoint_step > 0:
                next_checkpoint_step = int(restored_next_checkpoint_step)
            elif global_step > 0:
                next_checkpoint_step = int(((global_step // checkpoint_interval_steps) + 1) * checkpoint_interval_steps)
            else:
                next_checkpoint_step = int(checkpoint_interval_steps)

        if eval_runtime_enabled:
            eval_vecnorm_state = _vecnorm_state_or_none(envs)
            eval_envs = make_vector_env(
                args.env_id,
                num_envs=1,
                seed=int(args.eval_seed_start),
                capture_video=False,
                run_name=f"{run_name}_periodic_val",
                flatten_obs=args.flatten_obs,
                vector_env=args.eval_vector_env,
                env_kwargs=eval_env_kwargs,
                vecnorm=bool(args.vecnorm),
                vecnorm_norm_obs=bool(args.vecnorm_norm_obs),
                vecnorm_norm_reward=bool(args.vecnorm_eval_norm_reward),
                vecnorm_clip_obs=float(args.vecnorm_clip_obs),
                vecnorm_clip_reward=float(args.vecnorm_clip_reward),
                vecnorm_epsilon=float(args.vecnorm_epsilon),
                vecnorm_gamma=float(vecnorm_gamma),
                vecnorm_training=False,
                vecnorm_state=eval_vecnorm_state,
            )
            eval_vecnorm = unwrap_vec_normalize(eval_envs)
            if eval_vecnorm is not None:
                eval_vecnorm.set_training(False)
            eval_spec = infer_env_spec(eval_envs)
            if tuple(spec.obs_shape) != tuple(eval_spec.obs_shape):
                raise ValueError(f"val obs_shape mismatch: train={spec.obs_shape}, val={eval_spec.obs_shape}")
            if int(spec.action_dim) != int(eval_spec.action_dim):
                raise ValueError(f"val action_dim mismatch: train={spec.action_dim}, val={eval_spec.action_dim}")
            if eval_at_start_enabled and global_step == 0:
                _sync_vecnorm_stats(envs, eval_envs)
                seed_base = int(args.eval_seed_start)
                if not bool(args.eval_fixed_seeds):
                    seed_base = int(args.eval_seed_start) + eval_round * int(args.eval_episodes)
                eval_summary = _run_periodic_val(
                    agent=agent,
                    eval_envs=eval_envs,
                    spec=eval_spec,
                    device=device,
                    episodes=int(args.eval_episodes),
                    seed_start=seed_base,
                    deterministic=bool(args.eval_deterministic),
                    use_action_mask=bool(args.use_action_mask),
                )
                eval_row = {
                    "global_step": 0,
                    "eval_round": int(eval_round),
                    "seed_base": int(seed_base),
                    "fixed_seeds": bool(args.eval_fixed_seeds),
                    "summary": eval_summary,
                }
                with periodic_val_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_row) + "\n")
                tracker.log_metrics(
                    0,
                    {
                        "val/periodic_mean_return": float(eval_summary["return"]["mean"]),
                        "val/periodic_mean_illegal_penalty": float(eval_summary["illegal_penalty"]["mean"]),
                        "val/periodic_mean_terminal_score": float(eval_summary["terminal_score"]["mean"]),
                        "val/periodic_mean_terminal_game_score": float(eval_summary["terminal_game_score"]["mean"]),
                        "val/periodic_mean_game_score_ratio": float(eval_summary["game_score_ratio"]["mean"]),
                    },
                )
                tracker.log_event("periodic_val", eval_row)
                logger.info(
                    "periodic_val@start step=0 round=%d return=%s game_score=%s",
                    int(eval_round),
                    _fmt_float(eval_summary["return"]["mean"], 5),
                    _fmt_float(eval_summary["terminal_game_score"]["mean"], 2),
                )
                eval_mean_terminal_game = float(eval_summary["terminal_game_score"]["mean"])
                if np.isfinite(eval_mean_terminal_game) and eval_mean_terminal_game > best_mean_return:
                    best_mean_return = eval_mean_terminal_game
                    best_metric_source = "periodic_val_at_start"
                    save_agent_checkpoint(
                        best_model,
                        agent,
                        optimizer=optimizer,
                        meta={
                            "type": "ppo_discrete_best",
                            "cfg": asdict(cfg),
                            "env_id": args.env_id,
                            "env_kwargs": env_kwargs,
                            "global_step": 0,
                            "best_mean_return": best_mean_return,
                            "best_metric_name": best_metric_name,
                            "best_metric_source": best_metric_source,
                            "vecnormalize_state": _vecnorm_state_or_none(envs),
                        },
                    )
                    tracker.log_event(
                        "save_best_model",
                        {
                            "path": str(best_model),
                            "global_step": 0,
                            "value": best_mean_return,
                            "metric": best_metric_name,
                            "source": best_metric_source,
                        },
                    )
                eval_round += 1

        next_obs_np, next_info = reset_env(envs, seed=cfg.seed)
        next_obs_t = obs_to_tensor(next_obs_np, device)
        next_done_t = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
        next_mask_np = extract_action_mask(next_obs_np, next_info, spec.action_dim) if args.use_action_mask else None
        next_mask_t = _mask_to_tensor(next_mask_np, device)

        start_iteration = int(global_step // cfg.batch_size) + 1
        if start_iteration > cfg.num_iterations:
            logger.info(
                "no PPO update needed: global_step=%d already reached total_timesteps=%d",
                global_step,
                cfg.total_timesteps,
            )

        def _build_resume_meta() -> dict[str, Any]:
            return {
                "type": "ppo_discrete_last",
                "cfg": asdict(cfg),
                "env_id": args.env_id,
                "env_kwargs": env_kwargs,
                "global_step": global_step,
                "recent_total_returns": recent_total_returns[-100:],
                "recent_illegal_penalties": recent_illegal_penalties[-100:],
                "recent_terminal_scores": recent_terminal_scores[-100:],
                "recent_terminal_game_scores": recent_terminal_game_scores[-100:],
                "best_mean_return": best_mean_return,
                "best_metric_name": best_metric_name,
                "best_metric_source": best_metric_source,
                "eval_round": int(eval_round),
                "next_eval_step": int(next_eval_step),
                "next_checkpoint_step": int(next_checkpoint_step),
                "init_meta": init_meta,
                "vecnormalize_state": _vecnorm_state_or_none(envs),
            }

        schedules = _build_schedule_fns(cfg)
        lr_schedule_fn, lr_schedule_desc = schedules["lr"]
        ent_schedule_fn, ent_schedule_desc = schedules["ent"]
        clip_schedule_fn, clip_schedule_desc = schedules["clip"]
        vf_clip_schedule_fn, vf_clip_schedule_desc = schedules["vf_clip"]
        logger.info(
            "schedules: lr=%s ent=%s clip=%s vf_clip=%s",
            lr_schedule_desc,
            ent_schedule_desc,
            clip_schedule_desc,
            vf_clip_schedule_desc,
        )
        if bool(cfg.clip_vloss) and vf_clip_schedule_fn is None:
            logger.warning("clip_vloss=true but clip_range_vf is unset; value clipping is disabled")

        for iteration in range(start_iteration, cfg.num_iterations + 1):
            progress = _schedule_progress(int(iteration), int(cfg.num_iterations))
            lr_current = float(lr_schedule_fn(progress))
            if not np.isfinite(lr_current) or lr_current < 0.0:
                raise ValueError(f"invalid scheduled learning_rate={lr_current} at progress={progress}")
            optimizer.param_groups[0]["lr"] = lr_current

            ent_coef_current = float(ent_schedule_fn(progress))
            clip_coef_current = float(clip_schedule_fn(progress))
            if not np.isfinite(ent_coef_current) or ent_coef_current < 0.0:
                raise ValueError(f"invalid scheduled ent_coef={ent_coef_current} at progress={progress}")
            if not np.isfinite(clip_coef_current) or clip_coef_current < 0.0:
                raise ValueError(f"invalid scheduled clip_coef={clip_coef_current} at progress={progress}")
            clip_coef_current = float(max(0.0, clip_coef_current))
            clip_range_vf_current: float | None = None
            if vf_clip_schedule_fn is not None:
                clip_range_vf_current = float(vf_clip_schedule_fn(progress))
                if not np.isfinite(clip_range_vf_current) or clip_range_vf_current < 0.0:
                    raise ValueError(
                        f"invalid scheduled clip_range_vf={clip_range_vf_current} at progress={progress}"
                    )
            trainer.set_runtime_coefficients(
                ent_coef=ent_coef_current,
                clip_coef=clip_coef_current,
                clip_range_vf=clip_range_vf_current,
            )
            trunc_bootstrap_count_iter = 0

            buffer = RolloutBuffer(
                num_steps=cfg.num_steps,
                num_envs=cfg.num_envs,
                obs_shape=spec.obs_shape,
                action_shape=tuple(),
                device=device,
                use_action_mask=args.use_action_mask,
                action_dim=spec.action_dim,
            )

            for step in range(cfg.num_steps):
                with torch.no_grad():
                    action, logprob, _entropy, value = agent.get_action_and_value(
                        next_obs_t,
                        action_mask=next_mask_t if args.use_action_mask else None,
                    )

                action_np = unwrap_action(action)
                current_mask_t = next_mask_t if args.use_action_mask else None
                next_obs_np, reward_np, terminations, truncations, infos = envs.step(action_np)
                done_np = np.logical_or(terminations, truncations)

                reward_np, trunc_bootstrap_applied = _apply_time_limit_bootstrap(
                    rewards=np.asarray(reward_np, dtype=np.float32),
                    next_obs=np.asarray(next_obs_np),
                    terminations=np.asarray(terminations, dtype=np.bool_),
                    truncations=np.asarray(truncations, dtype=np.bool_),
                    infos=infos,
                    agent=agent,
                    gamma=float(cfg.gamma),
                    device=device,
                )
                trunc_bootstrap_count_iter += int(trunc_bootstrap_applied)

                reward_t = reward_to_tensor(reward_np, device)
                done_t = done_to_tensor(done_np, device)

                buffer.add(
                    step,
                    obs=next_obs_t,
                    action=action.long(),
                    logprob=logprob,
                    reward=reward_t.view(-1),
                    done=next_done_t,
                    value=value.view(-1),
                    action_mask=current_mask_t,
                )

                next_obs_t = obs_to_tensor(next_obs_np, device)
                next_done_t = done_t
                next_mask_np = extract_action_mask(next_obs_np, infos, spec.action_dim) if args.use_action_mask else None
                next_mask_t = _mask_to_tensor(next_mask_np, device)

                _append_recent_episode_metrics(
                    infos,
                    recent_total_returns,
                    recent_illegal_penalties,
                    recent_terminal_scores,
                    recent_terminal_game_scores,
                )

            with torch.no_grad():
                next_value = agent.get_value(next_obs_t).view(-1)
            buffer.compute_gae(next_value, next_done_t, cfg.gamma, cfg.gae_lambda)
            explained_variance = _explained_variance(buffer.values, buffer.returns)
            stats = trainer.update(buffer)

            global_step += cfg.batch_size
            elapsed = max(1e-9, time.time() - start_time)
            sps = int(max(0, global_step - session_start_global_step) / elapsed)
            mean_recent_return_total = (
                float(np.mean(recent_total_returns[-100:])) if recent_total_returns else float("nan")
            )
            mean_recent_return_illegal = (
                float(np.mean(recent_illegal_penalties[-100:])) if recent_illegal_penalties else float("nan")
            )
            mean_recent_return_terminal = (
                float(np.mean(recent_terminal_scores[-100:])) if recent_terminal_scores else float("nan")
            )
            mean_recent_return_terminal_game = (
                float(np.mean(recent_terminal_game_scores[-100:])) if recent_terminal_game_scores else float("nan")
            )

            row = {
                "iteration": iteration,
                "global_step": global_step,
                "sps": sps,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "schedule_progress": float(progress),
                "ent_coef_current": float(ent_coef_current),
                "clip_coef_current": float(clip_coef_current),
                "clip_range_vf_current": (
                    float(clip_range_vf_current) if clip_range_vf_current is not None else float("nan")
                ),
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy": stats.entropy,
                "approx_kl": stats.approx_kl,
                "clipfrac": stats.clipfrac,
                "value_clipfrac": stats.value_clipfrac,
                "update_epochs_used": int(stats.update_epochs_used),
                "update_epochs_cfg": int(cfg.update_epochs),
                "early_stop_by_kl": bool(stats.early_stop_by_kl),
                "target_kl_threshold": (
                    float(stats.target_kl_threshold)
                    if stats.target_kl_threshold is not None
                    else float("nan")
                ),
                "explained_variance": explained_variance,
                "trunc_bootstrap_count": int(trunc_bootstrap_count_iter),
                "mean_recent_return_100": mean_recent_return_total,
                "mean_recent_return_100_total": mean_recent_return_total,
                "mean_recent_return_100_illegal_penalty": mean_recent_return_illegal,
                "mean_recent_return_100_terminal_score": mean_recent_return_terminal,
                "mean_recent_return_100_terminal_game_score": mean_recent_return_terminal_game,
            }
            best_candidate = float("nan")
            best_source = ""

            if periodic_val_enabled and global_step >= next_eval_step:
                _sync_vecnorm_stats(envs, eval_envs)
                seed_base = int(args.eval_seed_start)
                if not bool(args.eval_fixed_seeds):
                    seed_base = int(args.eval_seed_start) + eval_round * int(args.eval_episodes)
                eval_summary = _run_periodic_val(
                    agent=agent,
                    eval_envs=eval_envs,
                    spec=eval_spec,
                    device=device,
                    episodes=int(args.eval_episodes),
                    seed_start=seed_base,
                    deterministic=bool(args.eval_deterministic),
                    use_action_mask=bool(args.use_action_mask),
                )
                eval_row = {
                    "global_step": int(global_step),
                    "eval_round": int(eval_round),
                    "seed_base": int(seed_base),
                    "fixed_seeds": bool(args.eval_fixed_seeds),
                    "summary": eval_summary,
                }
                with periodic_val_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_row) + "\n")
                row["periodic_val_mean_return"] = float(eval_summary["return"]["mean"])
                row["periodic_val_mean_illegal_penalty"] = float(eval_summary["illegal_penalty"]["mean"])
                row["periodic_val_mean_terminal_score"] = float(eval_summary["terminal_score"]["mean"])
                row["periodic_val_mean_terminal_game_score"] = float(eval_summary["terminal_game_score"]["mean"])
                row["periodic_val_mean_game_score_ratio"] = float(eval_summary["game_score_ratio"]["mean"])
                best_candidate = float(eval_summary["terminal_game_score"]["mean"])
                best_source = "periodic_val"
                tracker.log_metrics(
                    global_step,
                    {
                        "val/periodic_mean_return": float(eval_summary["return"]["mean"]),
                        "val/periodic_mean_illegal_penalty": float(eval_summary["illegal_penalty"]["mean"]),
                        "val/periodic_mean_terminal_score": float(eval_summary["terminal_score"]["mean"]),
                        "val/periodic_mean_terminal_game_score": float(eval_summary["terminal_game_score"]["mean"]),
                        "val/periodic_mean_game_score_ratio": float(eval_summary["game_score_ratio"]["mean"]),
                    },
                )
                tracker.log_event("periodic_val", eval_row)
                eval_round += 1
                next_eval_step += int(max(1, args.eval_interval_steps))
            elif not periodic_val_enabled:
                best_candidate = mean_recent_return_terminal_game
                best_source = "mean_recent_return_100_terminal_game_score"

            with log_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            should_log_train_line = (
                int(iteration) == 1
                or int(iteration) % int(log_interval_iters) == 0
                or ("periodic_val_mean_terminal_game_score" in row)
                or int(global_step) >= int(cfg.total_timesteps)
            )
            if should_log_train_line:
                logger.info("%s", _format_train_log_line(row))
            tracker.log_metrics(
                global_step,
                {
                    "charts/iteration": iteration,
                    "charts/sps": sps,
                    "charts/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "charts/schedule_progress": float(progress),
                    "charts/ent_coef": float(ent_coef_current),
                    "charts/clip_coef": float(clip_coef_current),
                    "charts/clip_range_vf": (
                        float(clip_range_vf_current) if clip_range_vf_current is not None else float("nan")
                    ),
                    "losses/policy_loss": stats.policy_loss,
                    "losses/value_loss": stats.value_loss,
                    "losses/entropy": stats.entropy,
                    "losses/approx_kl": stats.approx_kl,
                    "losses/clipfrac": stats.clipfrac,
                    "losses/value_clipfrac": stats.value_clipfrac,
                    "charts/update_epochs_used": int(stats.update_epochs_used),
                    "charts/early_stop_by_kl": float(1.0 if stats.early_stop_by_kl else 0.0),
                    "charts/target_kl_threshold": (
                        float(stats.target_kl_threshold)
                        if stats.target_kl_threshold is not None
                        else float("nan")
                    ),
                    "losses/explained_variance": explained_variance,
                    "rollout/trunc_bootstrap_count": int(trunc_bootstrap_count_iter),
                    "val/mean_recent_return_100": mean_recent_return_total,
                    "val/mean_recent_return_100_total": mean_recent_return_total,
                    "val/mean_recent_return_100_illegal_penalty": mean_recent_return_illegal,
                    "val/mean_recent_return_100_terminal_score": mean_recent_return_terminal,
                    "val/mean_recent_return_100_terminal_game_score": mean_recent_return_terminal_game,
                },
            )

            if np.isfinite(best_candidate) and best_candidate > best_mean_return:
                best_mean_return = float(best_candidate)
                best_metric_source = best_source
                save_agent_checkpoint(
                    best_model,
                    agent,
                    optimizer=optimizer,
                    meta={
                        "type": "ppo_discrete_best",
                        "cfg": asdict(cfg),
                        "env_id": args.env_id,
                        "env_kwargs": env_kwargs,
                        "global_step": global_step,
                        "best_mean_return": best_mean_return,
                        "best_metric_name": best_metric_name,
                        "best_metric_source": best_metric_source,
                        "vecnormalize_state": _vecnorm_state_or_none(envs),
                    },
                )
                tracker.log_event(
                    "save_best_model",
                    {
                        "path": str(best_model),
                        "global_step": global_step,
                        "value": best_mean_return,
                        "metric": best_metric_name,
                        "source": best_metric_source,
                    },
                )

            if checkpoint_interval_steps > 0 and global_step >= next_checkpoint_step:
                ckpt_path = checkpoint_dir / f"step_{int(global_step):012d}.pt"
                save_agent_checkpoint(
                    ckpt_path,
                    agent,
                    optimizer=optimizer,
                    meta=_build_resume_meta(),
                )
                tracker.log_event(
                    "save_checkpoint",
                    {
                        "path": str(ckpt_path),
                        "global_step": int(global_step),
                        "next_checkpoint_step": int(next_checkpoint_step),
                    },
                )
                next_checkpoint_step += checkpoint_interval_steps

            if int(cfg.save_interval) > 0 and iteration % int(cfg.save_interval) == 0:
                save_agent_checkpoint(
                    last_model,
                    agent,
                    optimizer=optimizer,
                    meta=_build_resume_meta(),
                )
                tracker.log_event("save_last_model", {"path": str(last_model), "global_step": global_step})

        save_agent_checkpoint(
            last_model,
            agent,
            optimizer=optimizer,
            meta=_build_resume_meta(),
        )
        summary = {
            "run_name": run_name,
            "run_dir": str(layout.root),
            "env_id": args.env_id,
            "env_kwargs": env_kwargs,
            "vecnormalize": {
                "enabled": bool(train_vecnorm is not None),
                "norm_obs": bool(args.vecnorm_norm_obs),
                "norm_reward_train": bool(args.vecnorm_norm_reward),
                "norm_reward_val": bool(args.vecnorm_eval_norm_reward),
                "clip_obs": float(args.vecnorm_clip_obs),
                "clip_reward": float(args.vecnorm_clip_reward),
                "epsilon": float(args.vecnorm_epsilon),
                "gamma": float(vecnorm_gamma),
            },
            "periodic_val": {
                "enabled": bool(periodic_val_enabled),
                "interval_steps": int(args.eval_interval_steps),
                "episodes": int(args.eval_episodes),
                "seed_start": int(args.eval_seed_start),
                "fixed_seeds": bool(args.eval_fixed_seeds),
                "deterministic": bool(args.eval_deterministic),
                "val_at_start": bool(args.eval_at_start),
                "val_at_start_enabled": bool(eval_at_start_enabled),
                "val_env_kwargs": eval_env_kwargs if eval_runtime_enabled else {},
                "metrics_jsonl": str(periodic_val_jsonl) if eval_runtime_enabled else "",
            },
            "global_step": global_step,
            "best_mean_return": best_mean_return,
            "best_metric_name": best_metric_name,
            "best_metric_source": best_metric_source,
            "resume": {
                "enabled": bool(resume_enabled),
                "resume_from": str(resume_from) if resume_from is not None else "",
                "initial_global_step": int(session_start_global_step),
            },
            "models": {
                "best": str(best_model),
                "last": str(last_model),
                "checkpoint_dir": str(checkpoint_dir) if checkpoint_interval_steps > 0 else "",
                "checkpoint_interval_steps": int(checkpoint_interval_steps),
            },
            "logs": {
                "train_metrics_jsonl": str(log_jsonl),
                "metrics_jsonl": str(layout.logs_dir / "metrics.jsonl"),
                "events_jsonl": str(layout.logs_dir / "events.jsonl"),
            },
        }
        (layout.reports_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (layout.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        update_manifest(
            layout,
            {
                "status": "completed",
                "result": summary,
                "timestamps": {"finished_at": time.time()},
            },
        )
        tracker.log_event("train_complete", summary)
        logger.info("completed run_dir=%s", layout.root)
        return 0
    except Exception as e:
        update_manifest(
            layout,
            {
                "status": "failed",
                "error": str(e),
                "progress": {"global_step": global_step, "best_mean_return": best_mean_return},
                "timestamps": {"failed_at": time.time()},
            },
        )
        if tracker is not None:
            tracker.log_event("train_failed", {"error": str(e), "global_step": global_step})
        raise
    finally:
        if envs is not None:
            envs.close()
        if eval_envs is not None:
            eval_envs.close()
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())
