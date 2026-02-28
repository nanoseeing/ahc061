from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from ..eval.eval_service import run_policy_episodes
from ..pipeline.model_checkpoint_service import load_agent_checkpoint
from ..utils.experiment import (
    create_run_layout,
    make_run_name,
    to_jsonable,
    update_manifest,
)
from ..utils.log_utils import get_logger
from ..utils.metrics import summarize, group_score_mean_variance_by_m_u
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
    if args.run_root is None and (
        bool(getattr(args, "tensorboard", False)) or bool(str(getattr(args, "mlflow_tracking_uri", "")).strip())
    ):
        if args.output_json is not None:
            args.run_root = args.output_json.parent / "_eval_runs"
        else:
            args.run_root = Path("reinforce/outputs/eval_runs")
    if args.run_root is None:
        return None, None

    run_name = args.run_name or make_run_name(args.env_id.replace("/", "_") + "_eval", seed=args.seed)
    layout = create_run_layout(args.run_root, run_name)
    if args.prefer_run_layout or args.output_json is None:
        args.output_json = layout.reports_dir / "eval_policy.json"

    config_snapshot = to_jsonable({"args": vars(args), "layout": layout.as_dict()})
    (layout.config_dir / "evaluate_policy.args.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    tracker = MetricTracker(
        layout.root,
        run_name=run_name,
        mlflow_tracking_uri=str(getattr(args, "mlflow_tracking_uri", "")),
        mlflow_experiment=str(getattr(args, "mlflow_experiment", "ppo_discrete")),
        mlflow_run_name=str(getattr(args, "mlflow_run_name", "")),
        tensorboard=bool(getattr(args, "tensorboard", False)),
        config=config_snapshot,
    )
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
    episode_scores: list[float],
    episode_m: list[int],
    episode_u: list[int],
    episode_self_scores: list[float],
    episode_enemy_max_scores: list[float],
) -> dict[str, Any]:
    if not (
        len(episode_scores)
        == len(episode_m)
        == len(episode_u)
        == len(episode_self_scores)
        == len(episode_enemy_max_scores)
    ):
        raise ValueError(
            "eval episode summary length mismatch: "
            f"scores={len(episode_scores)} m={len(episode_m)} u={len(episode_u)} "
            f"self={len(episode_self_scores)} enemy={len(episode_enemy_max_scores)}"
        )

    per_episode: list[dict[str, float | int]] = []
    for idx in range(len(episode_scores)):
        per_episode.append(
            {
                "episode": int(idx),
                "M": int(episode_m[idx]),
                "U": int(episode_u[idx]),
                "self_score": float(episode_self_scores[idx]),
                "enemy_max_score": float(episode_enemy_max_scores[idx]),
                "score": float(episode_scores[idx]),
            }
        )

    grouped_score = group_score_mean_variance_by_m_u(
        scores=episode_scores,
        m_values=episode_m,
        u_values=episode_u,
        m_key="M",
        u_key="U",
    )

    score_summary = summarize(episode_scores).as_dict()
    return {
        "env_id": args.env_id,
        "episodes": int(args.episodes),
        "return": summarize(episode_returns).as_dict(),
        "score": score_summary,
        "terminal_game_score": score_summary,
        "per_episode": per_episode,
        "grouped_score": grouped_score,
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

    total_episodes = int(args.episodes)
    with tqdm(total=total_episodes, desc="eval", unit="ep", dynamic_ncols=True) as pbar:
        stats = run_policy_episodes(
            env_id=str(args.env_id),
            episodes=total_episodes,
            num_envs=int(args.num_envs),
            seed=int(args.start_seed),
            feature_id=str(args.feature_id),
            pf_enabled=bool(args.pf_enabled),
            policy="model_greedy",
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
            on_episode_end=lambda _epi, _ret, _len: pbar.update(1),
            collect_score_breakdown=True,
        )

    return _build_summary(
        args=args,
        env_kwargs=env_kwargs,
        model_meta=model_meta,
        episode_returns=stats.episode_returns,
        episode_scores=stats.episode_terminal_game_scores,
        episode_m=stats.episode_m,
        episode_u=stats.episode_u,
        episode_self_scores=stats.episode_game_score_self,
        episode_enemy_max_scores=stats.episode_game_score_enemy_max,
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
            tracker.log_metrics(
                0,
                {
                    "eval/episodes": int(summary["episodes"]),
                    "eval/mean_return": float(summary["return"]["mean"]),
                    "eval/mean_terminal_game_score": float(summary["terminal_game_score"]["mean"]),
                    "eval/std_terminal_game_score": float(summary["terminal_game_score"]["std"]),
                },
            )
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
                        "mean_terminal_game_score": summary["terminal_game_score"]["mean"],
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
