from __future__ import annotations

import pytest

from reinforce.ppo.ppo.config import PPOConfig
from reinforce.ppo.train.schedule import (
    PPOScheduleSet,
    RuntimeScheduleResolver,
    parse_schedule_expr,
    schedule_progress,
    validate_schedule_args,
)


class TestTrainPPOSchedule:
    def test_schedule_progress_bounds(self) -> None:
        assert schedule_progress(1, 1) == 1.0
        assert schedule_progress(1, 2) == 0.0
        assert schedule_progress(2, 2) == 1.0
        assert schedule_progress(3, 5) == pytest.approx(0.5, abs=1e-6)

    def test_parse_piecewise_expr_interpolates(self) -> None:
        fn, desc = parse_schedule_expr(
            "piecewise(0:1.0,0.5:0.0,1.0:2.0)",
            default_kind="constant",
            default_start=0.0,
            default_end=0.0,
        )
        assert "piecewise(" in desc
        assert fn(0.0) == pytest.approx(1.0, abs=1e-6)
        assert fn(0.5) == pytest.approx(0.0, abs=1e-6)
        assert fn(0.75) == pytest.approx(1.0, abs=1e-6)
        assert fn(1.0) == pytest.approx(2.0, abs=1e-6)

    def test_schedule_set_requires_clip_range_vf(self) -> None:
        cfg = PPOConfig(
            clip_range_vf=None,
            clip_range_vf_schedule_expr="linear(0.2,0.1)",
        )
        with pytest.raises(ValueError):
            PPOScheduleSet.from_config(cfg)

    def test_validate_schedule_rejects_negative_entropy_coef(self) -> None:
        cfg = PPOConfig(
            ent_coef=0.01,
            ent_coef_schedule_expr="constant(-0.1)",
        )
        with pytest.raises(ValueError):
            validate_schedule_args(cfg)

    def test_build_schedule_with_vf_clip_schedule(self) -> None:
        cfg = PPOConfig(
            clip_range_vf=0.2,
            clip_range_vf_schedule="linear",
            clip_range_vf_final=0.05,
        )
        schedules = PPOScheduleSet.from_config(cfg)
        vf_fn = schedules.clip_range_vf
        assert vf_fn is not None
        assert vf_fn is not None
        assert vf_fn(0.0) == pytest.approx(0.2, abs=1e-6)
        assert vf_fn(1.0) == pytest.approx(0.05, abs=1e-6)

    def test_runtime_schedule_resolver_applies_warmup(self) -> None:
        cfg = PPOConfig(
            total_timesteps=500,
            num_envs=2,
            num_steps=50,
            num_minibatches=2,
            learning_rate=1.0,
            learning_rate_schedule="constant",
            ent_coef=0.02,
            ent_coef_schedule="constant",
            clip_coef=0.3,
            clip_coef_schedule="constant",
        )
        schedules = PPOScheduleSet.from_config(cfg)
        resolver = RuntimeScheduleResolver(
            schedules=schedules,
            total_iterations=5,
            warmup_steps=200,
            global_batch_size=100,
        )

        c1 = resolver.resolve(iteration=1, global_step=0)
        assert c1.progress == pytest.approx(0.0, abs=1e-6)
        assert c1.learning_rate == pytest.approx(0.5, abs=1e-6)
        assert c1.ent_coef == pytest.approx(0.02, abs=1e-6)
        assert c1.clip_coef == pytest.approx(0.3, abs=1e-6)
        assert c1.clip_range_vf is None

        c2 = resolver.resolve(iteration=2, global_step=100)
        assert c2.progress == pytest.approx(0.25, abs=1e-6)
        assert c2.learning_rate == pytest.approx(1.0, abs=1e-6)

    def test_runtime_schedule_resolver_validates_inputs(self) -> None:
        cfg = PPOConfig(
            total_timesteps=100,
            num_envs=2,
            num_steps=10,
            num_minibatches=2,
        )
        schedules = PPOScheduleSet.from_config(cfg)
        with pytest.raises(ValueError):
            RuntimeScheduleResolver(
                schedules=schedules,
                total_iterations=0,
                warmup_steps=0,
                global_batch_size=20,
            )
        with pytest.raises(ValueError):
            RuntimeScheduleResolver(
                schedules=schedules,
                total_iterations=10,
                warmup_steps=-1,
                global_batch_size=20,
            )
        with pytest.raises(ValueError):
            RuntimeScheduleResolver(
                schedules=schedules,
                total_iterations=10,
                warmup_steps=0,
                global_batch_size=0,
            )

    def test_runtime_schedule_resolver_rejects_negative_global_step(self) -> None:
        cfg = PPOConfig(
            total_timesteps=100,
            num_envs=2,
            num_steps=10,
            num_minibatches=2,
        )
        schedules = PPOScheduleSet.from_config(cfg)
        resolver = RuntimeScheduleResolver(
            schedules=schedules,
            total_iterations=10,
            warmup_steps=100,
            global_batch_size=20,
        )
        with pytest.raises(ValueError):
            resolver.resolve(iteration=1, global_step=-1)
