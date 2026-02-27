"""Tests for train_bc.py dataset utilities and BCTrainer setup."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from reinforce.ppo_discrete.entrypoints.train_bc import (
    BCTrainer,
    _count_dataset_rows,
    _inspect_aux_keys,
    _validate_args,
)
from reinforce.ppo_discrete.data.teacher_dataset import (
    DATASET_KEY_OPP_PARAM_TRUE,
    DATASET_KEY_OPP_VALID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_shard(
    path: Path,
    *,
    n: int = 8,
    obs_shape: tuple[int, ...] = (3,),
    action_dim: int = 4,
    with_aux: bool = False,
    opp_slot_count: int = 7,
    opp_param_dim: int = 5,
) -> None:
    obs = np.zeros((n,) + obs_shape, dtype=np.float32)
    action = np.zeros((n,), dtype=np.int64)
    payload: dict[str, np.ndarray] = {
        "obs": obs,
        "action": action,
        "obs_shape": np.asarray(list(obs_shape), dtype=np.int32),
        "action_dim": np.asarray([action_dim], dtype=np.int32),
    }
    if with_aux:
        payload[DATASET_KEY_OPP_PARAM_TRUE] = np.zeros(
            (n, opp_slot_count, opp_param_dim), dtype=np.float32
        )
        payload[DATASET_KEY_OPP_VALID] = np.ones((n, opp_slot_count), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def _make_args(tmp_path: Path, **kwargs) -> SimpleNamespace:
    defaults = dict(
        dataset_npz=None,
        dataset_shards_glob="",
        output_model=tmp_path / "model.pt",
        metrics_jsonl=None,
        seed=1,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        valid_ratio=0.2,
        model_class="",
        model_config_file=None,
        model_config_json="",
        device="cpu",
        aux_opp_param_loss_coef=0.0,
        aux_opp_param_use_valid_mask=True,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------


def test_validate_args_missing_dataset_raises(tmp_path: Path) -> None:
    args = _make_args(tmp_path)
    with pytest.raises(ValueError, match="--dataset-npz or --dataset-shards-glob is required"):
        _validate_args(args)


def test_validate_args_missing_output_model_raises(tmp_path: Path) -> None:
    args = _make_args(tmp_path, dataset_npz=tmp_path / "d.npz", output_model=None, run_root=None)
    with pytest.raises(ValueError, match="--output-model is required"):
        _validate_args(args)


def test_validate_args_valid_npz_passes(tmp_path: Path) -> None:
    args = _make_args(tmp_path, dataset_npz=tmp_path / "d.npz")
    _validate_args(args)  # should not raise


def test_validate_args_valid_shards_glob_passes(tmp_path: Path) -> None:
    args = _make_args(tmp_path, dataset_shards_glob="/tmp/shards/*.npz")
    _validate_args(args)  # should not raise


# ---------------------------------------------------------------------------
# _count_dataset_rows
# ---------------------------------------------------------------------------


def test_count_dataset_rows_single_shard(tmp_path: Path) -> None:
    p = tmp_path / "shard.npz"
    _write_shard(p, n=10, obs_shape=(3,), action_dim=4)
    total, sizes, obs_shape, action_dim = _count_dataset_rows([p])
    assert total == 10
    assert sizes == [10]
    assert obs_shape == (3,)
    assert action_dim == 4


def test_count_dataset_rows_multiple_shards(tmp_path: Path) -> None:
    p1 = tmp_path / "shard1.npz"
    p2 = tmp_path / "shard2.npz"
    _write_shard(p1, n=5, obs_shape=(2,), action_dim=3)
    _write_shard(p2, n=7, obs_shape=(2,), action_dim=3)
    total, sizes, obs_shape, action_dim = _count_dataset_rows([p1, p2])
    assert total == 12
    assert sizes == [5, 7]
    assert obs_shape == (2,)
    assert action_dim == 3


def test_count_dataset_rows_obs_shape_mismatch_raises(tmp_path: Path) -> None:
    p1 = tmp_path / "s1.npz"
    p2 = tmp_path / "s2.npz"
    _write_shard(p1, n=4, obs_shape=(3,), action_dim=4)
    _write_shard(p2, n=4, obs_shape=(5,), action_dim=4)
    with pytest.raises(ValueError, match="obs_shape mismatch"):
        _count_dataset_rows([p1, p2])


def test_count_dataset_rows_action_dim_mismatch_raises(tmp_path: Path) -> None:
    p1 = tmp_path / "s1.npz"
    p2 = tmp_path / "s2.npz"
    _write_shard(p1, n=4, obs_shape=(3,), action_dim=4)
    _write_shard(p2, n=4, obs_shape=(3,), action_dim=6)
    with pytest.raises(ValueError, match="action_dim mismatch"):
        _count_dataset_rows([p1, p2])


# ---------------------------------------------------------------------------
# _inspect_aux_keys
# ---------------------------------------------------------------------------


def test_inspect_aux_keys_no_aux(tmp_path: Path) -> None:
    p = tmp_path / "shard.npz"
    _write_shard(p, n=5, with_aux=False)
    info = _inspect_aux_keys([p])
    assert info["available"] is False
    assert info["keys"] == []


def test_inspect_aux_keys_with_aux(tmp_path: Path) -> None:
    p = tmp_path / "shard.npz"
    _write_shard(p, n=5, with_aux=True, opp_slot_count=7, opp_param_dim=5)
    info = _inspect_aux_keys([p])
    assert info["available"] is True
    assert DATASET_KEY_OPP_PARAM_TRUE in info["keys"]
    assert DATASET_KEY_OPP_VALID in info["keys"]
    assert info["opp_param_true_shape"] == [7, 5]
    assert info["opp_valid_shape"] == [7]


def test_inspect_aux_keys_mismatch_raises(tmp_path: Path) -> None:
    p1 = tmp_path / "s1.npz"
    p2 = tmp_path / "s2.npz"
    _write_shard(p1, n=4, with_aux=True)
    _write_shard(p2, n=4, with_aux=False)
    with pytest.raises(ValueError, match="aux key presence mismatch"):
        _inspect_aux_keys([p1, p2])


# ---------------------------------------------------------------------------
# BCTrainer._setup_dataset
# ---------------------------------------------------------------------------


def test_bc_trainer_setup_dataset_single_shard(tmp_path: Path) -> None:
    p = tmp_path / "data.npz"
    _write_shard(p, n=20, obs_shape=(4,), action_dim=3)
    args = _make_args(tmp_path, dataset_npz=p)
    trainer = BCTrainer(args)
    (
        dataset_paths,
        total_n,
        shard_sizes,
        obs_shape,
        action_dim,
        aux_info,
        valid_mask,
        offsets,
        train_count,
        n_valid,
    ) = trainer._setup_dataset()

    assert dataset_paths == [p]
    assert total_n == 20
    assert shard_sizes == [20]
    assert obs_shape == (4,)
    assert action_dim == 3
    assert aux_info["available"] is False
    assert valid_mask.shape == (20,)
    assert offsets == [0]
    assert n_valid >= 1
    assert train_count + n_valid == 20


def test_bc_trainer_setup_dataset_raises_on_too_small(tmp_path: Path) -> None:
    p = tmp_path / "tiny.npz"
    _write_shard(p, n=1, obs_shape=(2,), action_dim=2)
    args = _make_args(tmp_path, dataset_npz=p)
    trainer = BCTrainer(args)
    with pytest.raises(ValueError, match="dataset is too small"):
        trainer._setup_dataset()


# ---------------------------------------------------------------------------
# BCTrainer._train_epoch / _validate_epoch with a mock agent
# ---------------------------------------------------------------------------


class _FlatMockAgent(nn.Module):
    """Minimal agent for a flat obs_shape=(D,), action_dim=A."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(obs_dim, action_dim)

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs)


def test_bc_trainer_train_and_validate_epoch(tmp_path: Path) -> None:
    p = tmp_path / "data.npz"
    n = 20
    obs_dim = 4
    action_dim = 3
    np.random.seed(0)
    obs = np.random.randn(n, obs_dim).astype(np.float32)
    actions = np.random.randint(0, action_dim, size=(n,), dtype=np.int64)
    np.savez_compressed(
        p,
        obs=obs,
        action=actions,
        obs_shape=np.asarray([obs_dim], dtype=np.int32),
        action_dim=np.asarray([action_dim], dtype=np.int32),
    )

    args = _make_args(tmp_path, dataset_npz=p, batch_size=8)
    trainer = BCTrainer(args)
    (
        dataset_paths,
        total_n,
        shard_sizes,
        obs_shape,
        action_dim_out,
        aux_info,
        valid_mask,
        offsets,
        train_count,
        n_valid,
    ) = trainer._setup_dataset()

    agent = _FlatMockAgent(obs_dim=obs_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_loss, train_aux_loss, train_seen = trainer._train_epoch(
        agent=agent,
        optimizer=optimizer,
        dataset_paths=dataset_paths,
        shard_sizes=shard_sizes,
        offsets=offsets,
        valid_mask=valid_mask,
        aux_active=False,
        aux_coef=0.0,
        device=device,
    )
    assert train_seen == train_count
    assert train_loss >= 0.0

    valid_loss, valid_aux_loss, valid_correct, valid_seen = trainer._validate_epoch(
        agent=agent,
        dataset_paths=dataset_paths,
        shard_sizes=shard_sizes,
        offsets=offsets,
        valid_mask=valid_mask,
        aux_active=False,
        device=device,
    )
    assert valid_seen == n_valid
    assert valid_correct <= n_valid
    assert valid_loss >= 0.0
