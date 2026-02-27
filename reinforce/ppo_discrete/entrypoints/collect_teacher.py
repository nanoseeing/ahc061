from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..eval.eval_service import AuxTransition, Transition, run_policy_episodes
from ..pipeline.model_checkpoint_service import load_agent_checkpoint
from ..utils.experiment import coerce_optional_path, create_run_layout, make_run_name, resolve_config, to_jsonable, update_manifest
from ..utils.log_utils import get_logger
from ..utils.metrics import summarize
from ..data.teacher_dataset_merge import merge_teacher_shards
from ..data.teacher_dataset import (
    AHC061_BAYES_TAIL_SHAPE,
    AHC061_OPP_PARAM_TRUE_TAIL_SHAPE,
    AHC061_OPP_VALID_TAIL_SHAPE,
    BAYES_SOURCE_CPP,
    BAYES_SOURCE_ZERO_COMPAT,
    DATASET_KEY_OPP_PARAM_TRUE,
    DATASET_KEY_OPP_VALID,
    aux_to_teacher_opp_arrays,
    resolve_bayes_sources_for_meta,
    zero_bayes_params,
)
from ..utils.tracking import MetricTracker

logger = get_logger("collect_teacher")
_DATA_KEYS = ("obs", "action", "reward", "done", "episode", "step", "bayes_params")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect teacher trajectories for discrete PPO/BC.")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="collect_teacher", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--env-id", type=str, default="AHC061Local-v0")
    p.add_argument("--feature-id", type=str, default="submit_v1")
    p.add_argument("--pf-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--save-aux-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save auxiliary opponent targets into optional NPZ keys",
    )
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps-per-episode", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-npz", type=Path, default=None)
    p.add_argument(
        "--policy",
        choices=["random", "model_stochastic", "model_greedy"],
        default="random",
    )
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--env-kwargs-json", type=str, default="{}")
    p.add_argument("--log-interval-episodes", type=int, default=50, help="progress log interval in episodes (<=0 disables)")
    p.add_argument(
        "--chunk-episodes",
        type=int,
        default=10,
        help="flush temporary shard every N episodes to reduce peak memory (<=0 disables chunking)",
    )

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
        raise ValueError(f"unknown config keys for collect_teacher: {', '.join(unknown)}")

    parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.output_npz = coerce_optional_path(args.output_npz, dot_is_none=True)
    args.model_path = coerce_optional_path(args.model_path, dot_is_none=True)
    args.run_root = coerce_optional_path(args.run_root)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.output_npz is None and args.run_root is None:
        raise ValueError("--output-npz is required (or set --run-root)")
    if args.policy == "random":
        args.model_path = None
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required when policy is model_stochastic/model_greedy")

    env_id = str(args.env_id).strip()
    if env_id != "AHC061Local-v0":
        raise ValueError("collect_teacher supports only --env-id AHC061Local-v0")


def parse_env_kwargs(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--env-kwargs-json must be a JSON object")
    return obj


def _prepare_run(args: argparse.Namespace) -> tuple[Any, MetricTracker | None]:
    if args.run_root is None:
        return None, None

    run_name = args.run_name or make_run_name(args.env_id.replace("/", "_") + "_collect", seed=args.seed)
    layout = create_run_layout(args.run_root, run_name)
    if args.prefer_run_layout or args.output_npz is None:
        args.output_npz = layout.data_dir / "teacher.npz"

    config_snapshot = to_jsonable({"args": vars(args), "layout": layout.as_dict()})
    (layout.config_dir / "collect_teacher.args.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    tracker = MetricTracker(layout.root, run_name=run_name, enable_tensorboard=False, config=config_snapshot)
    update_manifest(
        layout,
        {
            "job": "collect_teacher",
            "status": "running",
            "run_name": run_name,
            "layout": layout.as_dict(),
            "config": config_snapshot,
            "timestamps": {"started_at": time.time()},
        },
    )
    return layout, tracker


def _empty_buffers(*, include_aux_targets: bool) -> dict[str, list[Any]]:
    out = {k: [] for k in _DATA_KEYS}
    if bool(include_aux_targets):
        out[DATASET_KEY_OPP_PARAM_TRUE] = []
        out[DATASET_KEY_OPP_VALID] = []
    return out


def _flush_chunk(
    *,
    buffers: dict[str, list[Any]],
    spec_obs_shape: tuple[int, ...],
    spec_action_dim: int,
    output_npz: Path,
    chunk_idx: int,
    chunk_paths: list[Path],
    include_aux_targets: bool,
) -> int:
    n = len(buffers["obs"])
    if n <= 0:
        return 0

    obs_arr = np.asarray(buffers["obs"], dtype=np.float32)
    action_arr = np.asarray(buffers["action"], dtype=np.int64)
    reward_arr = np.asarray(buffers["reward"], dtype=np.float32)
    done_arr = np.asarray(buffers["done"], dtype=np.uint8)
    episode_arr = np.asarray(buffers["episode"], dtype=np.int32)
    step_arr = np.asarray(buffers["step"], dtype=np.int32)
    bayes_arr = np.asarray(buffers["bayes_params"], dtype=np.float32)

    chunk_path = output_npz.parent / f".{output_npz.stem}.chunk_{chunk_idx:06d}.npz"
    payload: dict[str, Any] = {
        "obs": obs_arr,
        "action": action_arr,
        "reward": reward_arr,
        "done": done_arr,
        "episode": episode_arr,
        "step": step_arr,
        "bayes_params": bayes_arr,
        "obs_shape": np.asarray(spec_obs_shape, dtype=np.int32),
        "action_dim": np.asarray([spec_action_dim], dtype=np.int32),
    }
    if bool(include_aux_targets):
        payload[DATASET_KEY_OPP_PARAM_TRUE] = np.asarray(buffers[DATASET_KEY_OPP_PARAM_TRUE], dtype=np.float32)
        payload[DATASET_KEY_OPP_VALID] = np.asarray(buffers[DATASET_KEY_OPP_VALID], dtype=np.uint8)
    np.savez_compressed(chunk_path, **payload)
    chunk_paths.append(chunk_path)

    for v in buffers.values():
        v.clear()
    return int(n)


def main() -> int:
    args = parse_args()
    _validate_args(args)

    layout = None
    tracker: MetricTracker | None = None
    try:
        layout, tracker = _prepare_run(args)
        env_kwargs = parse_env_kwargs(args.env_kwargs_json)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = None
        if args.policy in ("model_stochastic", "model_greedy"):
            agent, _meta = load_agent_checkpoint(args.model_path, device=device)
            agent.eval()

        save_aux_targets = bool(args.save_aux_targets)
        buffers = _empty_buffers(include_aux_targets=save_aux_targets)
        chunk_paths: list[Path] = []
        chunk_index = 0
        transitions_total = 0
        chunk_episodes = max(0, int(args.chunk_episodes))
        first_bayes_shape: tuple[int, ...] | None = None
        first_opp_param_shape: tuple[int, ...] | None = None
        first_opp_valid_shape: tuple[int, ...] | None = None
        bayes_sources: set[str] = set()
        spec_obs_shape: tuple[int, ...] | None = None
        spec_action_dim: int | None = None
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        collect_started = time.time()
        interval = int(args.log_interval_episodes)

        def _on_episode_done(epi: int, ep_ret: float, ep_len: int) -> None:
            nonlocal transitions_total, chunk_index
            episode_returns.append(float(ep_ret))
            episode_lengths.append(int(ep_len))
            done_episodes = int(epi) + 1
            if chunk_episodes > 0 and (done_episodes % chunk_episodes == 0):
                if spec_obs_shape is None or spec_action_dim is None:
                    raise RuntimeError("internal error: spec is not initialized before flush")
                transitions_total += _flush_chunk(
                    buffers=buffers,
                    spec_obs_shape=tuple(spec_obs_shape),
                    spec_action_dim=int(spec_action_dim),
                    output_npz=args.output_npz,
                    chunk_idx=chunk_index,
                    chunk_paths=chunk_paths,
                    include_aux_targets=save_aux_targets,
                )
                chunk_index += 1
            if interval > 0 and (done_episodes % interval == 0 or done_episodes == int(args.episodes)):
                i0 = max(0, done_episodes - interval)
                recent_ret = episode_returns[i0:done_episodes]
                recent_len = episode_lengths[i0:done_episodes]
                elapsed = max(1e-9, time.time() - collect_started)
                logger.info(
                    "progress episodes=%d/%d transitions=%d chunks=%d recent_return_mean=%.6f recent_length_mean=%.2f eps_per_sec=%.2f elapsed_sec=%.1f",
                    done_episodes,
                    int(args.episodes),
                    int(transitions_total + len(buffers["obs"])),
                    int(len(chunk_paths)),
                    float(np.mean(recent_ret)) if recent_ret else float("nan"),
                    float(np.mean(recent_len)) if recent_len else float("nan"),
                    float(done_episodes / elapsed),
                    float(elapsed),
                )

        if env_kwargs:
            logger.warning(
                "cpp batch env ignores --env-kwargs-json keys: %s",
                ", ".join(sorted(str(k) for k in env_kwargs.keys())),
            )
        def _on_env_ready(obs_shape: tuple[int, ...], action_dim: int) -> None:
            nonlocal spec_obs_shape, spec_action_dim
            spec_obs_shape = tuple(obs_shape)
            spec_action_dim = int(action_dim)

        def _on_transition(tr: Transition) -> None:
            nonlocal first_bayes_shape
            buffers["obs"].append(np.asarray(tr.obs, dtype=np.float32))
            buffers["action"].append(int(tr.action))
            buffers["reward"].append(float(tr.reward))
            buffers["done"].append(int(bool(tr.done)))
            buffers["episode"].append(int(tr.episode))
            buffers["step"].append(int(tr.step))
            b = np.asarray(tr.bayes_params, dtype=np.float32).reshape(-1)
            if b.shape != AHC061_BAYES_TAIL_SHAPE:
                b = zero_bayes_params()
                bayes_sources.add(BAYES_SOURCE_ZERO_COMPAT)
            else:
                bayes_sources.add(BAYES_SOURCE_CPP)
            if first_bayes_shape is None:
                first_bayes_shape = tuple(int(x) for x in b.shape)
            buffers["bayes_params"].append(b)

        def _on_aux_transition(tr: AuxTransition) -> None:
            nonlocal first_opp_param_shape, first_opp_valid_shape
            p_enemy, v_enemy = aux_to_teacher_opp_arrays(tr.opp_param, tr.opp_valid)
            if first_opp_param_shape is None:
                first_opp_param_shape = tuple(int(x) for x in p_enemy.shape)
            if first_opp_valid_shape is None:
                first_opp_valid_shape = tuple(int(x) for x in v_enemy.shape)
            buffers[DATASET_KEY_OPP_PARAM_TRUE].append(p_enemy)
            buffers[DATASET_KEY_OPP_VALID].append(v_enemy)

        run_policy_episodes(
            env_id=str(args.env_id),
            episodes=int(args.episodes),
            seed=int(args.seed),
            feature_id=str(args.feature_id),
            pf_enabled=bool(args.pf_enabled),
            policy=str(args.policy),
            agent=agent,
            device=device,
            use_action_mask=True,
            amp=bool(args.amp),
            max_steps_per_episode=int(args.max_steps_per_episode),
            on_env_ready=_on_env_ready,
            on_transition=_on_transition,
            on_aux_transition=(_on_aux_transition if save_aux_targets else None),
            on_episode_end=_on_episode_done,
        )

        if spec_obs_shape is None or spec_action_dim is None:
            raise RuntimeError("internal error: spec was not initialized")
        transitions_total += _flush_chunk(
            buffers=buffers,
            spec_obs_shape=tuple(spec_obs_shape),
            spec_action_dim=int(spec_action_dim),
            output_npz=args.output_npz,
            chunk_idx=chunk_index,
            chunk_paths=chunk_paths,
            include_aux_targets=save_aux_targets,
        )

        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        if not chunk_paths:
            empty_bayes_shape = first_bayes_shape if first_bayes_shape is not None else AHC061_BAYES_TAIL_SHAPE
            np.savez_compressed(
                args.output_npz,
                obs=np.zeros((0, *tuple(spec_obs_shape)), dtype=np.float32),
                action=np.zeros((0,), dtype=np.int64),
                reward=np.zeros((0,), dtype=np.float32),
                done=np.zeros((0,), dtype=np.uint8),
                episode=np.zeros((0,), dtype=np.int32),
                step=np.zeros((0,), dtype=np.int32),
                bayes_params=np.zeros((0, *empty_bayes_shape), dtype=np.float32),
                obs_shape=np.asarray(spec_obs_shape, dtype=np.int32),
                action_dim=np.asarray([spec_action_dim], dtype=np.int32),
                **(
                    {
                        DATASET_KEY_OPP_PARAM_TRUE: np.zeros(
                            (0, *(first_opp_param_shape or AHC061_OPP_PARAM_TRUE_TAIL_SHAPE)),
                            dtype=np.float32,
                        ),
                        DATASET_KEY_OPP_VALID: np.zeros(
                            (0, *(first_opp_valid_shape or AHC061_OPP_VALID_TAIL_SHAPE)),
                            dtype=np.uint8,
                        ),
                    }
                    if save_aux_targets
                    else {}
                ),
            )
        elif len(chunk_paths) == 1:
            chunk_paths[0].replace(args.output_npz)
            chunk_paths = []
        else:
            merge_teacher_shards(chunk_paths, args.output_npz, offset_episode_ids=False)
            for p in chunk_paths:
                if p.exists():
                    p.unlink()
            chunk_paths = []

        ret_summary = summarize(episode_returns)
        len_summary = summarize(episode_lengths)
        meta = {
            "env_id": args.env_id,
            "env_kwargs": env_kwargs,
            "args": to_jsonable(vars(args)),
            "episodes": int(args.episodes),
            "transitions": int(transitions_total),
            "policy": args.policy,
            "seed": int(args.seed),
            "obs_shape": list(spec_obs_shape),
            "action_dim": int(spec_action_dim),
            "bayes_param_shape": list(first_bayes_shape if first_bayes_shape is not None else AHC061_BAYES_TAIL_SHAPE),
            "bayes_param_sources": resolve_bayes_sources_for_meta(bayes_sources),
            "bayes_param_policy": "store_observation_time_bayes_params;cpp_posterior",
            "aux_saved": bool(save_aux_targets),
            "aux_keys": ([DATASET_KEY_OPP_PARAM_TRUE, DATASET_KEY_OPP_VALID] if save_aux_targets else []),
            "opp_param_true_shape": (
                list(first_opp_param_shape if first_opp_param_shape is not None else AHC061_OPP_PARAM_TRUE_TAIL_SHAPE)
                if save_aux_targets
                else []
            ),
            "opp_valid_shape": (
                list(first_opp_valid_shape if first_opp_valid_shape is not None else AHC061_OPP_VALID_TAIL_SHAPE)
                if save_aux_targets
                else []
            ),
            "aux_policy": ("optional_keys_store_true_opponent_params" if save_aux_targets else ""),
            "episode_return": ret_summary.as_dict(),
            "episode_length": len_summary.as_dict(),
        }
        meta_path = args.output_npz.with_suffix(args.output_npz.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if tracker is not None and layout is not None:
            summary_path = layout.reports_dir / "collect_teacher_summary.json"
            summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            update_manifest(
                layout,
                {
                    "status": "completed",
                    "result": {
                        "output_npz": str(args.output_npz),
                        "meta_json": str(meta_path),
                        "report_json": str(summary_path),
                        "transitions": int(transitions_total),
                    },
                    "timestamps": {"finished_at": time.time()},
                },
            )
            tracker.log_event(
                "collect_complete",
                {
                    "output_npz": str(args.output_npz),
                    "transitions": int(transitions_total),
                },
            )

        logger.info("%s", json.dumps(meta, ensure_ascii=True))
        logger.info("saved: %s", args.output_npz)
        logger.info("meta: %s", meta_path)
        return 0
    except Exception as e:
        if tracker is not None and layout is not None:
            update_manifest(layout, {"status": "failed", "error": str(e), "timestamps": {"failed_at": time.time()}})
            tracker.log_event("collect_failed", {"error": str(e)})
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())
