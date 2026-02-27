from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..usecases.eval_service import run_policy_episodes
from ..usecases.model_checkpoint_service import load_agent_checkpoint
from ..infra.experiment import coerce_optional_path, create_run_layout, make_run_name, resolve_config, to_jsonable, update_manifest
from ..infra.log_utils import get_logger
from ..infra.metrics import summarize
from ..infra.tracking import MetricTracker

logger = get_logger("eval_policy")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a saved discrete PPO/BC policy.")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="evaluate_policy", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--env-id", type=str, default="AHC061Local-v0")
    p.add_argument("--feature-id", type=str, default="submit_v1")
    p.add_argument("--pf-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-action-mask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vecnorm-mode", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--vecnorm-norm-obs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vecnorm-norm-reward", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--vecnorm-clip-reward", type=float, default=10.0)
    p.add_argument("--vecnorm-epsilon", type=float, default=1e-8)
    p.add_argument("--vecnorm-gamma", type=float, default=0.99)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--env-kwargs-json", type=str, default="{}")

    p.add_argument("--run-root", type=Path, default=None, help="optional run root for managed outputs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--prefer-run-layout", action=argparse.BooleanOptionalAction, default=True)
    return p


def _parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in parser._actions:
        if a.dest == "help":
            continue
        if a.default is argparse.SUPPRESS:
            continue
        out[a.dest] = a.default
    return out


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    pre, _unknown = parser.parse_known_args()

    base_defaults = _parser_defaults(parser)
    for key in ("config_file", "config_section", "set"):
        base_defaults.pop(key, None)

    cfg = resolve_config(
        defaults=base_defaults,
        config_file=pre.config_file,
        config_section=pre.config_section,
        overrides=list(pre.set or []),
    )
    unknown = sorted(k for k in cfg.keys() if k not in base_defaults.keys())
    if unknown:
        raise ValueError(f"unknown config keys for evaluate_policy: {', '.join(unknown)}")

    parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.model_path = coerce_optional_path(args.model_path, dot_is_none=True)
    args.output_json = coerce_optional_path(args.output_json, dot_is_none=True)
    args.run_root = coerce_optional_path(args.run_root)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.model_path is None:
        raise ValueError("--model-path is required")
    env_id = str(args.env_id).strip()
    if env_id != "AHC061Local-v0":
        raise ValueError("evaluate_policy supports only --env-id AHC061Local-v0")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_env_kwargs(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--env-kwargs-json must be a JSON object")
    return obj


def _prepare_run(args: argparse.Namespace) -> tuple[Any, MetricTracker | None]:
    if args.run_root is None:
        return None, None

    run_name = args.run_name or make_run_name(args.env_id.replace("/", "_") + "_eval", seed=args.seed)
    layout = create_run_layout(args.run_root, run_name)
    if args.prefer_run_layout or args.output_json is None:
        args.output_json = layout.reports_dir / "eval_policy.json"

    config_snapshot = to_jsonable({"args": vars(args), "layout": layout.as_dict()})
    (layout.config_dir / "evaluate_policy.args.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    tracker = MetricTracker(layout.root, run_name=run_name, enable_tensorboard=False, config=config_snapshot)
    update_manifest(
        layout,
        {
            "job": "evaluate_policy",
            "status": "running",
            "run_name": run_name,
            "layout": layout.as_dict(),
            "config": config_snapshot,
            "timestamps": {"started_at": time.time()},
        },
    )
    return layout, tracker


def _build_summary(
    *,
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    model_meta: dict[str, Any] | Any,
    episode_returns: list[float],
    episode_illegal_penalties: list[float],
    episode_terminal_scores: list[float],
    episode_terminal_game_scores: list[float],
    episode_game_score_ratio: list[float],
    episode_game_score_self: list[float],
    episode_game_score_enemy_max: list[float],
) -> dict[str, Any]:
    return {
        "env_id": args.env_id,
        "episodes": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "return": summarize(episode_returns).as_dict(),
        "reward_components": {
            "illegal_penalty": summarize(episode_illegal_penalties).as_dict(),
            "terminal_score": summarize(episode_terminal_scores).as_dict(),
            "terminal_score_ratio": summarize(episode_terminal_scores).as_dict(),
            "terminal_game_score": summarize(episode_terminal_game_scores).as_dict(),
        },
        "game_score": {
            "ratio": summarize(episode_game_score_ratio).as_dict(),
            "self_score": summarize(episode_game_score_self).as_dict(),
            "enemy_max_score": summarize(episode_game_score_enemy_max).as_dict(),
        },
        "model_path": str(args.model_path),
        "env_kwargs": env_kwargs,
        "model_meta": to_jsonable(model_meta),
        "args": to_jsonable(vars(args)),
    }


def _run_eval(
    *,
    args: argparse.Namespace,
    agent: torch.nn.Module,
    model_meta: dict[str, Any] | Any,
    device: torch.device,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    ignored_kwargs = sorted(str(k) for k in env_kwargs.keys())
    if ignored_kwargs:
        logger.warning(
            "cpp batch env ignores --env-kwargs-json keys: %s",
            ", ".join(ignored_kwargs),
        )

    vec_state = model_meta.get("vecnormalize_state") if isinstance(model_meta, dict) else None
    vecnorm_mode = str(args.vecnorm_mode).lower().strip()
    use_vecnorm = False
    if vecnorm_mode == "on":
        use_vecnorm = True
    elif vecnorm_mode == "auto":
        use_vecnorm = isinstance(vec_state, dict)
    if use_vecnorm and not isinstance(vec_state, dict) and vecnorm_mode == "on":
        logger.warning("vecnorm-mode=on but checkpoint has no vecnormalize_state; using fresh statistics")

    stats = run_policy_episodes(
        env_id=str(args.env_id),
        episodes=int(args.episodes),
        seed=int(args.seed),
        feature_id=str(args.feature_id),
        pf_enabled=bool(args.pf_enabled),
        policy=("model_greedy" if bool(args.deterministic) else "model_stochastic"),
        agent=agent,
        device=device,
        use_action_mask=bool(args.use_action_mask),
        amp=bool(args.amp),
        vecnorm_enabled=bool(use_vecnorm),
        vecnorm_state=(vec_state if isinstance(vec_state, dict) else None),
        vecnorm_norm_obs=bool(args.vecnorm_norm_obs),
        vecnorm_norm_reward=bool(args.vecnorm_norm_reward),
        vecnorm_clip_obs=float(args.vecnorm_clip_obs),
        vecnorm_clip_reward=float(args.vecnorm_clip_reward),
        vecnorm_epsilon=float(args.vecnorm_epsilon),
        vecnorm_gamma=float(args.vecnorm_gamma),
    )

    return _build_summary(
        args=args,
        env_kwargs=env_kwargs,
        model_meta=model_meta,
        episode_returns=stats.episode_returns,
        episode_illegal_penalties=stats.episode_illegal_penalties,
        episode_terminal_scores=stats.episode_terminal_scores,
        episode_terminal_game_scores=stats.episode_terminal_game_scores,
        episode_game_score_ratio=stats.episode_game_score_ratio,
        episode_game_score_self=stats.episode_game_score_self,
        episode_game_score_enemy_max=stats.episode_game_score_enemy_max,
    )


def main() -> int:
    args = parse_args()
    _validate_args(args)

    layout = None
    tracker: MetricTracker | None = None
    try:
        layout, tracker = _prepare_run(args)
        device = choose_device(args.device)
        env_kwargs = parse_env_kwargs(args.env_kwargs_json)
        agent, meta = load_agent_checkpoint(args.model_path, device=device)

        summary = _run_eval(
            args=args,
            agent=agent,
            model_meta=meta,
            device=device,
            env_kwargs=env_kwargs,
        )

        logger.info("%s", json.dumps(summary, ensure_ascii=True, indent=2))

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger.info("saved: %s", args.output_json)

        if tracker is not None and layout is not None:
            report_path = layout.reports_dir / "evaluate_policy_summary.json"
            report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            update_manifest(
                layout,
                {
                    "status": "completed",
                    "result": {
                        "output_json": str(args.output_json) if args.output_json is not None else "",
                        "report_json": str(report_path),
                        "mean_return": summary["return"]["mean"],
                        "mean_illegal_penalty": summary["reward_components"]["illegal_penalty"]["mean"],
                        "mean_terminal_score": summary["reward_components"]["terminal_score"]["mean"],
                        "mean_terminal_game_score": summary["reward_components"]["terminal_game_score"]["mean"],
                        "mean_game_score_ratio": summary["game_score"]["ratio"]["mean"],
                    },
                    "timestamps": {"finished_at": time.time()},
                },
            )
            tracker.log_event("evaluate_complete", {"mean_return": summary["return"]["mean"]})

        return 0
    except Exception as e:
        if tracker is not None and layout is not None:
            update_manifest(layout, {"status": "failed", "error": str(e), "timestamps": {"failed_at": time.time()}})
            tracker.log_event("evaluate_failed", {"error": str(e)})
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())
