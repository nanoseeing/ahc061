from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from ..eval.eval_service import run_policy_episodes
from ..pipeline.model_checkpoint_service import load_agent_checkpoint
from ..utils.experiment import (
    create_run_layout,
    make_run_name,
    to_jsonable,
    update_manifest,
)
from ..utils.log_utils import get_logger
from ..utils.metrics import summarize
from ..utils.runtime import choose_device, parse_json_object
from ..utils.tracking import MetricTracker
from .common import cfg_to_namespace

logger = get_logger("eval_policy")
_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


@hydra.main(version_base="1.3", config_path=_CONF_DIR, config_name="eval_policy/default")
def main(cfg: DictConfig) -> None:
    args = _cfg_to_ns(cfg)
    raise SystemExit(_run(args))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    return cfg_to_namespace(
        cfg,
        optional_paths={
            "model_path": True,
            "output_json": True,
            "run_root": False,
        },
    )


def _prepare_run(args: SimpleNamespace) -> tuple[Any, MetricTracker | None]:
    if args.run_root is None:
        return None, None

    run_name = args.run_name or make_run_name(args.env_id.replace("/", "_") + "_eval", seed=args.seed)
    layout = create_run_layout(args.run_root, run_name)
    if args.prefer_run_layout or args.output_json is None:
        args.output_json = layout.reports_dir / "eval_policy.json"

    config_snapshot = to_jsonable({"args": vars(args), "layout": layout.as_dict()})
    (layout.config_dir / "evaluate_policy.args.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    tracker = MetricTracker(layout.root, run_name=run_name, config=config_snapshot)
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
    args: SimpleNamespace,
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
    args: SimpleNamespace,
    agent: torch.nn.Module,
    model_meta: dict[str, Any] | Any,
    device: torch.device,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    ignored_kwargs = sorted(str(k) for k in env_kwargs.keys())
    if ignored_kwargs:
        logger.warning("cpp batch env ignores env_kwargs_json keys: %s", ", ".join(ignored_kwargs))

    vec_state = model_meta.get("vecnormalize_state") if isinstance(model_meta, dict) else None
    vecnorm_mode = str(args.vecnorm_mode).lower().strip()
    use_vecnorm = False
    if vecnorm_mode == "on":
        use_vecnorm = True
    elif vecnorm_mode == "auto":
        use_vecnorm = isinstance(vec_state, dict)
    if use_vecnorm and not isinstance(vec_state, dict) and vecnorm_mode == "on":
        logger.warning("vecnorm_mode=on but checkpoint has no vecnormalize_state; using fresh statistics")

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


def _run(args: SimpleNamespace) -> int:
    if args.model_path is None:
        raise ValueError("model_path is required")
    if str(args.env_id).strip() != "AHC061Local-v0":
        raise ValueError(
            f"evaluate_policy supports only env_id=AHC061Local-v0 (got {args.env_id!r})"
        )

    layout = None
    tracker: MetricTracker | None = None
    try:
        layout, tracker = _prepare_run(args)
        device_obj = choose_device(str(args.device))
        env_kwargs_dict = parse_json_object(str(args.env_kwargs_json), field_name="env_kwargs_json")
        agent, meta = load_agent_checkpoint(args.model_path, device=device_obj)

        summary = _run_eval(args=args, agent=agent, model_meta=meta, device=device_obj, env_kwargs=env_kwargs_dict)
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
    except Exception as e:
        if tracker is not None and layout is not None:
            update_manifest(layout, {"status": "failed", "error": str(e), "timestamps": {"failed_at": time.time()}})
            tracker.log_event("evaluate_failed", {"error": str(e)})
        raise
    finally:
        if tracker is not None:
            tracker.close()
    return 0


if __name__ == "__main__":
    main()
