from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Optional

import torch
import typer

from ..eval.eval_service import run_policy_episodes
from ..pipeline.model_checkpoint_service import load_agent_checkpoint
from ..utils.experiment import (
    coerce_optional_path,
    create_run_layout,
    make_run_name,
    resolve_config,
    to_jsonable,
    update_manifest,
)
from ..utils.log_utils import get_logger
from ..utils.metrics import summarize
from ..utils.tracking import MetricTracker

logger = get_logger("eval_policy")
app = typer.Typer(add_completion=False)

_DEFAULTS: dict[str, Any] = {
    "env_id": "AHC061Local-v0",
    "feature_id": "submit_v1",
    "pf_enabled": True,
    "amp": False,
    "episodes": 50,
    "seed": 1,
    "device": "auto",
    "deterministic": False,
    "use_action_mask": False,
    "vecnorm_mode": "auto",
    "vecnorm_norm_obs": True,
    "vecnorm_norm_reward": False,
    "vecnorm_clip_obs": 10.0,
    "vecnorm_clip_reward": 10.0,
    "vecnorm_epsilon": 1e-8,
    "vecnorm_gamma": 0.99,
    "output_json": None,
    "env_kwargs_json": "{}",
    "run_root": None,
    "run_name": "",
    "prefer_run_layout": True,
}


def _ns(cfg: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**cfg)


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
        logger.warning("cpp batch env ignores --env-kwargs-json keys: %s", ", ".join(ignored_kwargs))

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


@app.command()
def main(
    config_file: Annotated[Optional[Path], typer.Option("--config-file", help="json/toml/yaml config file")] = None,
    config_section: Annotated[str, typer.Option("--config-section")] = "evaluate_policy",
    set_: Annotated[Optional[list[str]], typer.Option("--set", help="override key=value (repeatable)")] = None,
    model_path: Annotated[Optional[Path], typer.Option("--model-path")] = None,
    env_id: Annotated[Optional[str], typer.Option("--env-id")] = None,
    feature_id: Annotated[Optional[str], typer.Option("--feature-id")] = None,
    pf_enabled: Annotated[Optional[bool], typer.Option("--pf-enabled/--no-pf-enabled")] = None,
    amp: Annotated[Optional[bool], typer.Option("--amp/--no-amp")] = None,
    episodes: Annotated[Optional[int], typer.Option("--episodes")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed")] = None,
    device: Annotated[Optional[str], typer.Option("--device")] = None,
    deterministic: Annotated[Optional[bool], typer.Option("--deterministic/--no-deterministic")] = None,
    use_action_mask: Annotated[Optional[bool], typer.Option("--use-action-mask/--no-use-action-mask")] = None,
    vecnorm_mode: Annotated[Optional[str], typer.Option("--vecnorm-mode")] = None,
    vecnorm_norm_obs: Annotated[Optional[bool], typer.Option("--vecnorm-norm-obs/--no-vecnorm-norm-obs")] = None,
    vecnorm_norm_reward: Annotated[Optional[bool], typer.Option("--vecnorm-norm-reward/--no-vecnorm-norm-reward")] = None,
    vecnorm_clip_obs: Annotated[Optional[float], typer.Option("--vecnorm-clip-obs")] = None,
    vecnorm_clip_reward: Annotated[Optional[float], typer.Option("--vecnorm-clip-reward")] = None,
    vecnorm_epsilon: Annotated[Optional[float], typer.Option("--vecnorm-epsilon")] = None,
    vecnorm_gamma: Annotated[Optional[float], typer.Option("--vecnorm-gamma")] = None,
    output_json: Annotated[Optional[Path], typer.Option("--output-json")] = None,
    env_kwargs_json: Annotated[Optional[str], typer.Option("--env-kwargs-json")] = None,
    run_root: Annotated[Optional[Path], typer.Option("--run-root")] = None,
    run_name: Annotated[Optional[str], typer.Option("--run-name")] = None,
    prefer_run_layout: Annotated[Optional[bool], typer.Option("--prefer-run-layout/--no-prefer-run-layout")] = None,
) -> None:
    """Evaluate a saved discrete PPO/BC policy."""
    cfg = resolve_config(
        defaults=_DEFAULTS,
        config_file=config_file,
        config_section=config_section,
        overrides=list(set_ or []),
    )
    # CLI args (non-None) override config file values
    _cli: dict[str, Any] = {
        "model_path": model_path,
        "env_id": env_id,
        "feature_id": feature_id,
        "pf_enabled": pf_enabled,
        "amp": amp,
        "episodes": episodes,
        "seed": seed,
        "device": device,
        "deterministic": deterministic,
        "use_action_mask": use_action_mask,
        "vecnorm_mode": vecnorm_mode,
        "vecnorm_norm_obs": vecnorm_norm_obs,
        "vecnorm_norm_reward": vecnorm_norm_reward,
        "vecnorm_clip_obs": vecnorm_clip_obs,
        "vecnorm_clip_reward": vecnorm_clip_reward,
        "vecnorm_epsilon": vecnorm_epsilon,
        "vecnorm_gamma": vecnorm_gamma,
        "output_json": output_json,
        "env_kwargs_json": env_kwargs_json,
        "run_root": run_root,
        "run_name": run_name,
        "prefer_run_layout": prefer_run_layout,
    }
    cfg.update({k: v for k, v in _cli.items() if v is not None})
    cfg["model_path"] = coerce_optional_path(cfg.get("model_path"), dot_is_none=True)
    cfg["output_json"] = coerce_optional_path(cfg.get("output_json"), dot_is_none=True)
    cfg["run_root"] = coerce_optional_path(cfg.get("run_root"))

    if cfg["model_path"] is None:
        typer.echo("Error: --model-path is required", err=True)
        raise typer.Exit(1)
    if str(cfg["env_id"]).strip() != "AHC061Local-v0":
        typer.echo(f"Error: evaluate_policy supports only --env-id AHC061Local-v0 (got {cfg['env_id']!r})", err=True)
        raise typer.Exit(1)

    args = _ns(cfg)
    layout = None
    tracker: MetricTracker | None = None
    try:
        layout, tracker = _prepare_run(args)
        device_obj = _choose_device(str(args.device))
        env_kwargs_dict = json.loads(str(args.env_kwargs_json))
        if not isinstance(env_kwargs_dict, dict):
            raise ValueError("--env-kwargs-json must be a JSON object")
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


if __name__ == "__main__":
    app()
