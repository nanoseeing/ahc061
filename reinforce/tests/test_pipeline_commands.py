"""Tests for reinforce.ppo_discrete.pipeline.pipeline_commands."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reinforce.ppo_discrete.pipeline.pipeline_commands import (
    append_bool_flag,
    build_collect_teacher_cmd,
    build_eval_policy_cmd,
    build_train_bc_cmd,
    build_train_ppo_cmd,
)


# ---------------------------------------------------------------------------
# append_bool_flag
# ---------------------------------------------------------------------------


def test_append_bool_flag_true() -> None:
    cmd: list[str] = []
    append_bool_flag(cmd, "pf_enabled", True)
    assert cmd == ["pf_enabled=true"]


def test_append_bool_flag_false() -> None:
    cmd: list[str] = []
    append_bool_flag(cmd, "pf_enabled", False)
    assert cmd == ["pf_enabled=false"]


def test_append_bool_flag_hyphen_converted() -> None:
    """Hyphens in name are converted to underscores."""
    cmd: list[str] = []
    append_bool_flag(cmd, "pf-enabled", True)
    assert cmd == ["pf_enabled=true"]


# ---------------------------------------------------------------------------
# build_collect_teacher_cmd
# ---------------------------------------------------------------------------


def test_build_collect_teacher_cmd_basic(tmp_path: Path) -> None:
    out = tmp_path / "data.npz"
    cmd = build_collect_teacher_cmd(
        py="python",
        env_id="AHC061Local-v0",
        episodes=10,
        seed=42,
        collect_policy="greedy",
        chunk_episodes=5,
        output_npz=out,
        env_kwargs={"x": 1},
        feature_id="v1",
        pf_enabled=False,
        amp=False,
        save_aux_targets=True,
    )
    assert cmd[0] == "python"
    assert "env_id=AHC061Local-v0" in cmd
    assert "episodes=10" in cmd
    assert "seed=42" in cmd
    assert "policy=greedy" in cmd
    assert "chunk_episodes=5" in cmd
    assert f"output_npz={out}" in cmd
    assert "feature_id=v1" in cmd
    assert "pf_enabled=false" in cmd
    assert "amp=false" in cmd
    assert "save_aux_targets=true" in cmd
    # env_kwargs should be JSON-encoded inside OmegaConf single-quote wrapper
    ek_elem = next(e for e in cmd if e.startswith("env_kwargs_json="))
    inner = ek_elem[len("env_kwargs_json="):]
    assert inner.startswith("'") and inner.endswith("'")
    assert json.loads(inner[1:-1]) == {"x": 1}


# ---------------------------------------------------------------------------
# build_train_bc_cmd with dataset_npz
# ---------------------------------------------------------------------------


def _make_train_bc_args(**kwargs) -> SimpleNamespace:
    defaults = dict(
        seed=7,
        bc_epochs=5,
        bc_aux_opp_param_loss_coef=0.0,
        bc_aux_opp_param_use_valid_mask=True,
        model_class="",
        model_config_file=None,
        model_config_json="",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_train_bc_cmd_with_npz(tmp_path: Path) -> None:
    args = _make_train_bc_args()
    out = tmp_path / "model.pt"
    dataset = tmp_path / "data.npz"
    cmd = build_train_bc_cmd(py="python", args=args, output_model=out, dataset_npz=dataset)
    assert cmd[0] == "python"
    assert f"output_model={out}" in cmd
    assert f"dataset_npz={dataset}" in cmd
    assert "seed=7" in cmd
    assert "epochs=5" in cmd
    # no shards glob when npz is given
    assert not any(e.startswith("dataset_shards_glob=") for e in cmd)


def test_build_train_bc_cmd_with_shards_glob(tmp_path: Path) -> None:
    args = _make_train_bc_args()
    out = tmp_path / "model.pt"
    cmd = build_train_bc_cmd(
        py="python",
        args=args,
        output_model=out,
        dataset_shards_glob="/tmp/shards/*.npz",
    )
    assert any(e.startswith("dataset_shards_glob=") for e in cmd)
    shards_elem = next(e for e in cmd if e.startswith("dataset_shards_glob="))
    assert "/tmp/shards/*.npz" in shards_elem
    assert not any(e.startswith("dataset_npz=") for e in cmd)


def test_build_train_bc_cmd_aux_mask_flag() -> None:
    args = _make_train_bc_args(bc_aux_opp_param_use_valid_mask=False)
    cmd = build_train_bc_cmd(
        py="python",
        args=args,
        output_model=Path("/tmp/out.pt"),
        dataset_npz=Path("/tmp/data.npz"),
    )
    assert "aux_opp_param_use_valid_mask=false" in cmd


# ---------------------------------------------------------------------------
# build_train_ppo_cmd
# ---------------------------------------------------------------------------


def _make_train_ppo_args(**kwargs) -> SimpleNamespace:
    defaults = dict(
        env_id="AHC061Local-v0",
        seed=1,
        ppo_total_timesteps=1000,
        ppo_num_envs=4,
        ppo_num_steps=16,
        ppo_learning_rate=3e-4,
        ppo_gamma=0.99,
        ppo_gae_lambda=0.95,
        ppo_num_minibatches=4,
        ppo_update_epochs=4,
        ppo_clip_coef=0.2,
        ppo_clip_coef_schedule="",
        ppo_clip_coef_final=None,
        ppo_clip_coef_schedule_expr="",
        ppo_ent_coef=0.01,
        ppo_ent_coef_schedule="",
        ppo_ent_coef_final=None,
        ppo_ent_coef_schedule_expr="",
        ppo_vf_coef=0.5,
        ppo_aux_opp_param_loss_coef=0.0,
        ppo_aux_opp_param_use_valid_mask=True,
        ppo_max_grad_norm=0.5,
        ppo_checkpoint_interval_steps=0,
        ppo_eval_interval_steps=0,
        ppo_eval_episodes=0,
        ppo_eval_seed_start=0,
        ppo_log_interval_iters=10,
        ppo_vecnorm_clip_obs=10.0,
        ppo_vecnorm_clip_reward=10.0,
        ppo_vecnorm_epsilon=1e-8,
        ppo_feature_id="v1",
        ppo_norm_adv=True,
        ppo_clip_vloss=True,
        ppo_eval_at_start=False,
        ppo_eval_fixed_seeds=False,
        ppo_eval_deterministic=True,
        ppo_vecnorm=False,
        ppo_vecnorm_norm_obs=True,
        ppo_vecnorm_norm_reward=False,
        ppo_vecnorm_eval_norm_reward=False,
        ppo_amp=False,
        ppo_pin_memory=False,
        ppo_pf_enabled=False,
        ppo_memory_format="nchw",
        ppo_rollout_cache_device="auto",
        ppo_distributed="auto",
        ppo_model_preset="",
        ppo_learning_rate_schedule="",
        ppo_clip_range_vf=None,
        ppo_clip_range_vf_schedule="",
        ppo_clip_range_vf_final=None,
        ppo_clip_range_vf_schedule_expr="",
        ppo_target_kl=None,
        ppo_vecnorm_gamma=None,
        ppo_eval_env_kwargs_json="",
        use_action_mask=True,
        mlflow_tracking_uri="",
        mlflow_experiment="",
        mlflow_run_name="",
        tensorboard=False,
        model_class="",
        model_config_file=None,
        model_config_json="",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_train_ppo_cmd_basic(tmp_path: Path) -> None:
    args = _make_train_ppo_args()
    run_dir = tmp_path / "runs"
    cmd = build_train_ppo_cmd(
        py="python",
        args=args,
        run_dir=run_dir,
        env_kwargs={"a": 2},
        ppo_val_env_kwargs_json="",
    )
    assert cmd[0] == "python"
    assert "env_id=AHC061Local-v0" in cmd
    assert "total_timesteps=1000" in cmd
    assert "num_envs=4" in cmd
    assert "feature_id=v1" in cmd
    assert "use_action_mask=true" in cmd
    assert "tensorboard=false" in cmd
    ek_elem = next(e for e in cmd if e.startswith("env_kwargs_json="))
    inner = ek_elem[len("env_kwargs_json="):]
    assert json.loads(inner[1:-1]) == {"a": 2}


def test_build_train_ppo_cmd_with_target_kl(tmp_path: Path) -> None:
    args = _make_train_ppo_args(ppo_target_kl=0.02)
    cmd = build_train_ppo_cmd(
        py="python",
        args=args,
        run_dir=tmp_path,
        env_kwargs={},
        ppo_val_env_kwargs_json="",
    )
    assert "target_kl=0.02" in cmd


def test_build_train_ppo_cmd_with_init_model(tmp_path: Path) -> None:
    args = _make_train_ppo_args()
    init = tmp_path / "init.pt"
    cmd = build_train_ppo_cmd(
        py="python",
        args=args,
        run_dir=tmp_path,
        env_kwargs={},
        ppo_val_env_kwargs_json="",
        init_model=init,
    )
    assert f"init_model={init}" in cmd
    assert "resume=true" not in cmd


def test_build_train_ppo_cmd_with_resume(tmp_path: Path) -> None:
    args = _make_train_ppo_args()
    resume_path = tmp_path / "last.pt"
    cmd = build_train_ppo_cmd(
        py="python",
        args=args,
        run_dir=tmp_path,
        env_kwargs={},
        ppo_val_env_kwargs_json="",
        resume_run_name="my_run",
        resume_from=resume_path,
    )
    assert "resume=true" in cmd
    assert f"resume_from={resume_path}" in cmd
    assert "run_name=my_run" in cmd


def test_build_train_ppo_cmd_no_resume_without_run_name(tmp_path: Path) -> None:
    args = _make_train_ppo_args()
    resume_path = tmp_path / "last.pt"
    # resume_from provided but no resume_run_name → no resume flags
    cmd = build_train_ppo_cmd(
        py="python",
        args=args,
        run_dir=tmp_path,
        env_kwargs={},
        ppo_val_env_kwargs_json="",
        resume_from=resume_path,
        resume_run_name="",
    )
    assert "resume=true" not in cmd


# ---------------------------------------------------------------------------
# build_eval_policy_cmd
# ---------------------------------------------------------------------------


def _make_eval_args(**kwargs) -> SimpleNamespace:
    defaults = dict(
        env_id="AHC061Local-v0",
        eval_episodes=20,
        seed=0,
        ppo_feature_id="v1",
        ppo_pf_enabled=False,
        use_action_mask=True,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_eval_policy_cmd_basic(tmp_path: Path) -> None:
    args = _make_eval_args()
    model = tmp_path / "best.pt"
    out_json = tmp_path / "eval.json"
    cmd = build_eval_policy_cmd(
        py="python",
        args=args,
        model_path=model,
        output_json=out_json,
        env_kwargs={"k": "v"},
    )
    assert cmd[0] == "python"
    assert "env_id=AHC061Local-v0" in cmd
    assert f"model_path={model}" in cmd
    assert "episodes=20" in cmd
    assert f"output_json={out_json}" in cmd
    assert "deterministic=true" in cmd
    assert "feature_id=v1" in cmd
    assert "use_action_mask=true" in cmd
    assert "pf_enabled=false" in cmd
