from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from reinforce.ppo.entrypoints.common import cfg_to_namespace
from reinforce.ppo.train.requests import ppo_config_from_source
from reinforce.ppo.utils.runtime import parse_json_object


def test_cfg_to_namespace_applies_path_coercion_and_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "run_root": "",
            "model_path": ".",
            "resume_from": "models/last.pt",
        }
    )
    ns = cfg_to_namespace(
        cfg,
        optional_paths={
            "run_root": False,
            "model_path": True,
            "resume_from": True,
        },
        default_paths={"run_root": Path("reinforce/outputs/pipeline_runs")},
    )
    assert ns.run_root == Path("reinforce/outputs/pipeline_runs")
    assert ns.model_path is None
    assert ns.resume_from == Path("models/last.pt")


def test_parse_json_object_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        parse_json_object("[]", field_name="env_kwargs_json")


def test_ppo_config_from_source_builds_expected_fields() -> None:
    src = SimpleNamespace(
        seed=3,
        total_iterations=400,
        num_envs=4,
        num_steps=20,
        learning_rate=1e-3,
        learning_rate_schedule="linear",
        warmup_iters=25,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=3,
        norm_adv=True,
        clip_coef=0.2,
        clip_coef_schedule="linear",
        clip_coef_final=0.1,
        clip_coef_schedule_expr="",
        clip_range_vf=0.3,
        clip_range_vf_schedule="cosine",
        clip_range_vf_final=0.15,
        clip_range_vf_schedule_expr="",
        clip_vloss=True,
        ent_coef=0.01,
        ent_coef_schedule="constant",
        ent_coef_final=None,
        ent_coef_schedule_expr="",
        vf_coef=0.5,
        aux_opp_param_loss_coef=0.0,
        aux_opp_param_use_valid_mask=True,
        max_grad_norm=0.5,
        target_kl=None,
        save_interval=7,
    )
    cfg = ppo_config_from_source(src)
    assert cfg.total_iterations == 400
    assert cfg.learning_rate == pytest.approx(1e-3)
    assert cfg.warmup_iters == 25
    assert cfg.clip_coef_schedule == "linear"
    assert cfg.clip_coef_final == pytest.approx(0.1)
    assert cfg.clip_range_vf_schedule == "cosine"
    assert cfg.save_interval == 7


def test_entrypoints_default_force_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    import reinforce.ppo.entrypoints as entrypoints_pkg

    importlib.reload(entrypoints_pkg)
    assert os.environ.get("FORCE_COLOR") == "1"
