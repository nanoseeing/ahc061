from __future__ import annotations

import pytest

from reinforce.ppo_discrete.core.ppo.config import PPOConfig
from reinforce.ppo_discrete.core.ppo.train_utils import (
    PPOScheduleSet,
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
