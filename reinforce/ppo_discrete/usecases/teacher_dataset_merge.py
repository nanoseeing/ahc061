from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from ..domain.ahc061.teacher_dataset import (
    AHC061_BAYES_TAIL_SHAPE,
    AHC061_OPP_PARAM_TRUE_TAIL_SHAPE,
    AHC061_OPP_VALID_TAIL_SHAPE,
    DATASET_KEY_OPP_PARAM_TRUE,
    DATASET_KEY_OPP_VALID,
)
from ..infra.log_utils import get_logger

logger = get_logger("teacher_dataset_merge")


def split_counts(total: int, workers: int) -> list[int]:
    workers = max(1, int(workers))
    base = int(total) // workers
    rem = int(total) % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def merge_teacher_shards(
    shard_paths: list[Path],
    output_npz: Path,
    *,
    offset_episode_ids: bool,
) -> None:
    obs_shape_ref: np.ndarray | None = None
    action_dim_ref: np.ndarray | None = None
    obs_tail_shape: tuple[int, ...] | None = None
    bayes_tail_shape: tuple[int, ...] | None = None
    has_aux_targets: bool | None = None
    opp_param_tail_shape: tuple[int, ...] | None = None
    opp_valid_tail_shape: tuple[int, ...] | None = None
    total = 0

    for shard in shard_paths:
        with np.load(shard) as d:
            obs_shape = np.asarray(d["obs_shape"], dtype=np.int32)
            action_dim = np.asarray(d["action_dim"], dtype=np.int32)
            shard_has_aux_targets = (DATASET_KEY_OPP_PARAM_TRUE in d.files) and (DATASET_KEY_OPP_VALID in d.files)
            if has_aux_targets is None:
                has_aux_targets = bool(shard_has_aux_targets)
            elif bool(has_aux_targets) != bool(shard_has_aux_targets):
                raise ValueError(f"aux key presence mismatch in shard: {shard}")
            if obs_shape_ref is None:
                obs_shape_ref = obs_shape
                action_dim_ref = action_dim
                obs_tail_shape = tuple(np.asarray(d["obs"]).shape[1:])
                bayes_tail_shape = tuple(np.asarray(d["bayes_params"]).shape[1:])
                if bool(shard_has_aux_targets):
                    opp_param_tail_shape = tuple(np.asarray(d[DATASET_KEY_OPP_PARAM_TRUE]).shape[1:])
                    opp_valid_tail_shape = tuple(np.asarray(d[DATASET_KEY_OPP_VALID]).shape[1:])
            else:
                if not np.array_equal(obs_shape_ref, obs_shape):
                    raise ValueError(f"obs_shape mismatch in shard: {shard}")
                if not np.array_equal(action_dim_ref, action_dim):
                    raise ValueError(f"action_dim mismatch in shard: {shard}")
                if bool(shard_has_aux_targets):
                    p_shape = tuple(np.asarray(d[DATASET_KEY_OPP_PARAM_TRUE]).shape[1:])
                    v_shape = tuple(np.asarray(d[DATASET_KEY_OPP_VALID]).shape[1:])
                    if opp_param_tail_shape != p_shape:
                        raise ValueError(f"{DATASET_KEY_OPP_PARAM_TRUE} shape mismatch in shard: {shard}")
                    if opp_valid_tail_shape != v_shape:
                        raise ValueError(f"{DATASET_KEY_OPP_VALID} shape mismatch in shard: {shard}")
            total += int(np.asarray(d["action"]).shape[0])

    if obs_shape_ref is None or action_dim_ref is None:
        raise ValueError("no valid shards to merge")
    if has_aux_targets is None:
        has_aux_targets = False

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if obs_tail_shape is None:
        obs_tail_shape = tuple(int(x) for x in obs_shape_ref.tolist())
    if bayes_tail_shape is None:
        bayes_tail_shape = AHC061_BAYES_TAIL_SHAPE

    if total <= 0:
        payload: dict[str, Any] = {
            "obs": np.zeros((0, *obs_tail_shape), dtype=np.float32),
            "action": np.zeros((0,), dtype=np.int64),
            "reward": np.zeros((0,), dtype=np.float32),
            "done": np.zeros((0,), dtype=np.uint8),
            "episode": np.zeros((0,), dtype=np.int32),
            "step": np.zeros((0,), dtype=np.int32),
            "bayes_params": np.zeros((0, *bayes_tail_shape), dtype=np.float32),
            "obs_shape": obs_shape_ref,
            "action_dim": action_dim_ref,
        }
        if bool(has_aux_targets):
            payload[DATASET_KEY_OPP_PARAM_TRUE] = np.zeros(
                (0, *(opp_param_tail_shape or AHC061_OPP_PARAM_TRUE_TAIL_SHAPE)),
                dtype=np.float32,
            )
            payload[DATASET_KEY_OPP_VALID] = np.zeros(
                (0, *(opp_valid_tail_shape or AHC061_OPP_VALID_TAIL_SHAPE)),
                dtype=np.uint8,
            )
        np.savez_compressed(output_npz, **payload)
        return

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
        opp_param_mm = None
        opp_valid_mm = None
        if bool(has_aux_targets):
            opp_param_mm = np.lib.format.open_memmap(
                tmp / f"{DATASET_KEY_OPP_PARAM_TRUE}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total, *(opp_param_tail_shape or AHC061_OPP_PARAM_TRUE_TAIL_SHAPE)),
            )
            opp_valid_mm = np.lib.format.open_memmap(
                tmp / f"{DATASET_KEY_OPP_VALID}.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(total, *(opp_valid_tail_shape or AHC061_OPP_VALID_TAIL_SHAPE)),
            )

        cursor = 0
        episode_offset = 0
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
                if bool(has_aux_targets):
                    assert opp_param_mm is not None
                    assert opp_valid_mm is not None
                    opp_param_mm[sl] = np.asarray(d[DATASET_KEY_OPP_PARAM_TRUE], dtype=np.float32)
                    opp_valid_mm[sl] = np.asarray(d[DATASET_KEY_OPP_VALID], dtype=np.uint8)

                ep = np.asarray(d["episode"], dtype=np.int32)
                if bool(offset_episode_ids) and ep.size > 0:
                    ep = ep + int(episode_offset)
                    episode_offset = int(ep.max()) + 1
                episode_mm[sl] = ep
                cursor += n

        payload: dict[str, Any] = {
            "obs": obs_mm,
            "action": action_mm,
            "reward": reward_mm,
            "done": done_mm,
            "episode": episode_mm,
            "step": step_mm,
            "bayes_params": bayes_mm,
            "obs_shape": obs_shape_ref,
            "action_dim": action_dim_ref,
        }
        if bool(has_aux_targets):
            assert opp_param_mm is not None
            assert opp_valid_mm is not None
            payload[DATASET_KEY_OPP_PARAM_TRUE] = opp_param_mm
            payload[DATASET_KEY_OPP_VALID] = opp_valid_mm
        np.savez_compressed(output_npz, **payload)


def cleanup_teacher_shards(shard_paths: list[Path], *, shards_dir: Path | None = None) -> None:
    for shard in shard_paths:
        try:
            if shard.exists():
                shard.unlink()
        except Exception as e:
            logger.warning("failed to remove shard file: %s (%s)", shard, e)
        meta = shard.with_suffix(shard.suffix + ".meta.json")
        try:
            if meta.exists():
                meta.unlink()
        except Exception as e:
            logger.warning("failed to remove shard meta: %s (%s)", meta, e)

    if shards_dir is not None:
        try:
            if shards_dir.exists() and not any(shards_dir.iterdir()):
                shards_dir.rmdir()
        except Exception as e:
            logger.warning("failed to clean shard directory: %s (%s)", shards_dir, e)
