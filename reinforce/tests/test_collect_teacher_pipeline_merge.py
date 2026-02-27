from __future__ import annotations

from pathlib import Path

import numpy as np

from reinforce.ppo_discrete.data.teacher_dataset_merge import merge_teacher_shards


def _write_teacher_shard(
    path: Path,
    *,
    episodes: np.ndarray,
    obs_dim: int = 4,
    action_dim: int = 7,
) -> None:
    n = int(episodes.shape[0])
    obs = np.arange(n * obs_dim, dtype=np.float32).reshape(n, obs_dim)
    action = np.arange(n, dtype=np.int64) % action_dim
    reward = np.linspace(0.0, 1.0, n, dtype=np.float32)
    done = np.zeros((n,), dtype=np.uint8)
    step = np.arange(n, dtype=np.int32)
    bayes = np.zeros((n, 28), dtype=np.float32)
    np.savez_compressed(
        path,
        obs=obs,
        action=action,
        reward=reward,
        done=done,
        episode=episodes.astype(np.int32),
        step=step,
        bayes_params=bayes,
        obs_shape=np.asarray([obs_dim], dtype=np.int32),
        action_dim=np.asarray([action_dim], dtype=np.int32),
    )


def test_collect_teacher_internal_merge_keeps_episode_ids(tmp_path: Path) -> None:
    s1 = tmp_path / "chunk_001.npz"
    s2 = tmp_path / "chunk_002.npz"
    out = tmp_path / "merged_collect.npz"
    _write_teacher_shard(s1, episodes=np.asarray([0, 0, 1, 1], dtype=np.int32))
    _write_teacher_shard(s2, episodes=np.asarray([2, 2, 3], dtype=np.int32))

    merge_teacher_shards([s1, s2], out, offset_episode_ids=False)

    with np.load(out) as d:
        ep = np.asarray(d["episode"], dtype=np.int32)
        assert ep.tolist() == [0, 0, 1, 1, 2, 2, 3]
        assert tuple(np.asarray(d["obs"]).shape) == (7, 4)
        assert tuple(np.asarray(d["bayes_params"]).shape) == (7, 28)


def test_run_pipeline_merge_offsets_episode_ids_per_shard(tmp_path: Path) -> None:
    s1 = tmp_path / "worker_001.npz"
    s2 = tmp_path / "worker_002.npz"
    out = tmp_path / "merged_pipeline.npz"
    _write_teacher_shard(s1, episodes=np.asarray([0, 0, 1], dtype=np.int32))
    _write_teacher_shard(s2, episodes=np.asarray([0, 1, 1], dtype=np.int32))

    merge_teacher_shards([s1, s2], out, offset_episode_ids=True)

    with np.load(out) as d:
        ep = np.asarray(d["episode"], dtype=np.int32)
        # second shard should be offset by +2 (max episode in first shard + 1)
        assert ep.tolist() == [0, 0, 1, 2, 3, 3]
        assert tuple(np.asarray(d["obs"]).shape) == (6, 4)
        assert tuple(np.asarray(d["bayes_params"]).shape) == (6, 28)
