from __future__ import annotations

import argparse
import json
import time
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..env.api import extract_action_mask
from ..env.factory import infer_env_spec, make_vector_env, obs_to_tensor, unwrap_action
from ..runtime.checkpoint import load_agent_checkpoint
from ..runtime.experiment import coerce_optional_path, create_run_layout, make_run_name, resolve_config, to_jsonable, update_manifest
from ..runtime.log_utils import get_logger
from ..runtime.metrics import summarize
from ..runtime.tracking import MetricTracker

logger = get_logger("collect_teacher")
_DATA_KEYS = ("obs", "action", "reward", "done", "episode", "step", "bayes_params")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect teacher trajectories for discrete PPO/BC.")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="collect_teacher", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps-per-episode", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-npz", type=Path, default=None)
    p.add_argument(
        "--policy",
        choices=["random", "model_stochastic", "model_greedy", "ahc061_main_greedy"],
        default="random",
    )
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--flatten-obs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--vector-env", choices=["sync", "async"], default="sync")
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
    if args.policy in ("random", "ahc061_main_greedy"):
        args.model_path = None
        if args.policy == "ahc061_main_greedy" and args.env_id != "AHC061Local-v0":
            raise ValueError("--policy ahc061_main_greedy requires --env-id AHC061Local-v0")
        if args.policy == "ahc061_main_greedy" and args.vector_env != "sync":
            raise ValueError("--policy ahc061_main_greedy requires --vector-env sync")
        return
    if args.model_path is None:
        raise ValueError("--model-path is required when policy is model_stochastic/model_greedy")


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


def _empty_buffers() -> dict[str, list[Any]]:
    return {k: [] for k in _DATA_KEYS}


def _flush_chunk(
    *,
    buffers: dict[str, list[Any]],
    spec_obs_shape: tuple[int, ...],
    spec_action_dim: int,
    output_npz: Path,
    chunk_idx: int,
    chunk_paths: list[Path],
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
    np.savez_compressed(
        chunk_path,
        obs=obs_arr,
        action=action_arr,
        reward=reward_arr,
        done=done_arr,
        episode=episode_arr,
        step=step_arr,
        bayes_params=bayes_arr,
        obs_shape=np.asarray(spec_obs_shape, dtype=np.int32),
        action_dim=np.asarray([spec_action_dim], dtype=np.int32),
    )
    chunk_paths.append(chunk_path)

    for v in buffers.values():
        v.clear()
    return int(n)


def _merge_npz_shards_streaming(shard_paths: list[Path], output_npz: Path) -> None:
    obs_shape_ref: np.ndarray | None = None
    action_dim_ref: np.ndarray | None = None
    obs_tail_shape: tuple[int, ...] | None = None
    bayes_tail_shape: tuple[int, ...] | None = None
    total = 0

    for shard in shard_paths:
        with np.load(shard) as d:
            obs_shape = np.asarray(d["obs_shape"], dtype=np.int32)
            action_dim = np.asarray(d["action_dim"], dtype=np.int32)
            if obs_shape_ref is None:
                obs_shape_ref = obs_shape
                action_dim_ref = action_dim
                obs_tail_shape = tuple(np.asarray(d["obs"]).shape[1:])
                bayes_tail_shape = tuple(np.asarray(d["bayes_params"]).shape[1:])
            else:
                if not np.array_equal(obs_shape_ref, obs_shape):
                    raise ValueError(f"obs_shape mismatch in shard: {shard}")
                if not np.array_equal(action_dim_ref, action_dim):
                    raise ValueError(f"action_dim mismatch in shard: {shard}")
            total += int(np.asarray(d["action"]).shape[0])

    if obs_shape_ref is None or action_dim_ref is None or obs_tail_shape is None or bayes_tail_shape is None:
        raise ValueError("no shard data to merge")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_npz.stem}.merge_", dir=str(output_npz.parent)) as tmp_dir:
        tmp = Path(tmp_dir)
        obs_mm = np.lib.format.open_memmap(tmp / "obs.npy", mode="w+", dtype=np.float32, shape=(total, *obs_tail_shape))
        action_mm = np.lib.format.open_memmap(tmp / "action.npy", mode="w+", dtype=np.int64, shape=(total,))
        reward_mm = np.lib.format.open_memmap(tmp / "reward.npy", mode="w+", dtype=np.float32, shape=(total,))
        done_mm = np.lib.format.open_memmap(tmp / "done.npy", mode="w+", dtype=np.uint8, shape=(total,))
        episode_mm = np.lib.format.open_memmap(tmp / "episode.npy", mode="w+", dtype=np.int32, shape=(total,))
        step_mm = np.lib.format.open_memmap(tmp / "step.npy", mode="w+", dtype=np.int32, shape=(total,))
        bayes_mm = np.lib.format.open_memmap(
            tmp / "bayes_params.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total, *bayes_tail_shape),
        )

        cursor = 0
        for shard in shard_paths:
            with np.load(shard) as d:
                n = int(np.asarray(d["action"]).shape[0])
                if n <= 0:
                    continue
                sl = slice(cursor, cursor + n)
                obs_mm[sl] = np.asarray(d["obs"], dtype=np.float32)
                action_mm[sl] = np.asarray(d["action"], dtype=np.int64)
                reward_mm[sl] = np.asarray(d["reward"], dtype=np.float32)
                done_mm[sl] = np.asarray(d["done"], dtype=np.uint8)
                step_mm[sl] = np.asarray(d["step"], dtype=np.int32)
                bayes_mm[sl] = np.asarray(d["bayes_params"], dtype=np.float32)

                episode_mm[sl] = np.asarray(d["episode"], dtype=np.int32)
                cursor += n

        np.savez_compressed(
            output_npz,
            obs=obs_mm,
            action=action_mm,
            reward=reward_mm,
            done=done_mm,
            episode=episode_mm,
            step=step_mm,
            bayes_params=bayes_mm,
            obs_shape=obs_shape_ref,
            action_dim=action_dim_ref,
        )


def main() -> int:
    args = parse_args()
    _validate_args(args)

    layout = None
    tracker: MetricTracker | None = None
    envs = None
    try:
        layout, tracker = _prepare_run(args)
        env_kwargs = parse_env_kwargs(args.env_kwargs_json)

        envs = make_vector_env(
            args.env_id,
            num_envs=1,
            seed=args.seed,
            capture_video=args.capture_video,
            run_name=f"collect_teacher_{args.env_id}",
            flatten_obs=args.flatten_obs,
            vector_env=args.vector_env,
            env_kwargs=env_kwargs,
        )
        spec = infer_env_spec(envs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = None
        if args.policy in ("model_stochastic", "model_greedy"):
            agent, _meta = load_agent_checkpoint(args.model_path, device=device)
            if tuple(agent.obs_shape) != tuple(spec.obs_shape):
                raise ValueError(f"obs_shape mismatch: model={agent.obs_shape}, env={spec.obs_shape}")
            if int(agent.action_dim) != int(spec.action_dim):
                raise ValueError(f"action_dim mismatch: model={agent.action_dim}, env={spec.action_dim}")
            agent.eval()

        buffers = _empty_buffers()
        chunk_paths: list[Path] = []
        chunk_index = 0
        transitions_total = 0
        chunk_episodes = max(0, int(args.chunk_episodes))
        first_bayes_shape: tuple[int, ...] | None = None
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        collect_started = time.time()
        interval = int(args.log_interval_episodes)

        for epi in range(args.episodes):
            obs, _info = envs.reset(seed=args.seed + epi)
            done = np.array([False], dtype=np.bool_)
            ep_ret = 0.0
            ep_len = 0
            for step in range(args.max_steps_per_episode):
                if args.policy == "random":
                    action = np.array([envs.single_action_space.sample()], dtype=np.int64)
                elif args.policy == "ahc061_main_greedy":
                    base_env = getattr(envs.envs[0], "unwrapped", envs.envs[0])
                    if not hasattr(base_env, "expert_action_main_greedy"):
                        raise AttributeError("env does not expose expert_action_main_greedy()")
                    action = np.array([int(base_env.expert_action_main_greedy())], dtype=np.int64)
                else:
                    with torch.no_grad():
                        obs_t = obs_to_tensor(obs, device)
                        mask_np = extract_action_mask(obs, _info, spec.action_dim)
                        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device) if mask_np is not None else None
                        action_t = agent.act(
                            obs_t,
                            action_mask=mask_t,
                            deterministic=(args.policy == "model_greedy"),
                        )
                        action = unwrap_action(action_t).astype(np.int64)
                        if action.ndim == 0:
                            action = action.reshape(1)

                buffers["obs"].append(np.asarray(obs[0], dtype=np.float32))
                buffers["action"].append(int(action[0]))
                buffers["episode"].append(epi)
                buffers["step"].append(step)
                if isinstance(_info, dict) and "bayes_params" in _info:
                    b = np.asarray(_info["bayes_params"], dtype=np.float32)
                    if b.ndim == 2 and b.shape[0] == 1:
                        b = b[0]
                    if b.ndim != 1:
                        b = b.reshape(-1)
                else:
                    b = np.zeros((4 * 7,), dtype=np.float32)
                if first_bayes_shape is None:
                    first_bayes_shape = tuple(int(x) for x in b.shape)
                buffers["bayes_params"].append(b)

                next_obs, reward, terminations, truncations, infos = envs.step(action)
                done = np.logical_or(terminations, truncations)

                r = float(reward[0])
                buffers["reward"].append(r)
                buffers["done"].append(int(done[0]))
                ep_ret += r
                ep_len += 1

                obs = next_obs
                _info = infos
                if done[0]:
                    if "final_info" in infos:
                        for fi in infos["final_info"]:
                            if fi and "episode" in fi:
                                ep_ret = float(fi["episode"]["r"])
                                ep_len = int(fi["episode"]["l"])
                    break

            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            done_episodes = epi + 1
            if chunk_episodes > 0 and (done_episodes % chunk_episodes == 0):
                transitions_total += _flush_chunk(
                    buffers=buffers,
                    spec_obs_shape=tuple(spec.obs_shape),
                    spec_action_dim=int(spec.action_dim),
                    output_npz=args.output_npz,
                    chunk_idx=chunk_index,
                    chunk_paths=chunk_paths,
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
        transitions_total += _flush_chunk(
            buffers=buffers,
            spec_obs_shape=tuple(spec.obs_shape),
            spec_action_dim=int(spec.action_dim),
            output_npz=args.output_npz,
            chunk_idx=chunk_index,
            chunk_paths=chunk_paths,
        )

        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        if not chunk_paths:
            empty_bayes_shape = first_bayes_shape if first_bayes_shape is not None else (4 * 7,)
            np.savez_compressed(
                args.output_npz,
                obs=np.zeros((0, *tuple(spec.obs_shape)), dtype=np.float32),
                action=np.zeros((0,), dtype=np.int64),
                reward=np.zeros((0,), dtype=np.float32),
                done=np.zeros((0,), dtype=np.uint8),
                episode=np.zeros((0,), dtype=np.int32),
                step=np.zeros((0,), dtype=np.int32),
                bayes_params=np.zeros((0, *empty_bayes_shape), dtype=np.float32),
                obs_shape=np.asarray(spec.obs_shape, dtype=np.int32),
                action_dim=np.asarray([spec.action_dim], dtype=np.int32),
            )
        elif len(chunk_paths) == 1:
            chunk_paths[0].replace(args.output_npz)
            chunk_paths = []
        else:
            _merge_npz_shards_streaming(chunk_paths, args.output_npz)
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
            "obs_shape": list(spec.obs_shape),
            "action_dim": int(spec.action_dim),
            "bayes_param_shape": list(first_bayes_shape if first_bayes_shape is not None else (4 * 7,)),
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
        if envs is not None:
            envs.close()
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())
