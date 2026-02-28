"""`test_experiment_management` のテストモジュール。"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from reinforce.ppo.utils.experiment import (
    coerce_optional_path,
    create_run_layout,
    make_run_name,
    update_manifest,
)


class TestExperimentManagement:
    """`TestExperimentManagement` のテストケース。"""
    def test_make_run_name_contains_seed_ms_and_pid(self) -> None:
        """`make_run_name_contains_seed_ms_and_pid` の振る舞いを検証する。"""
        name = make_run_name("CartPole/v1", seed=7, now=1700000000.123)
        assert re.search(r"^CartPole_v1__seed7__1700000000123__p\d+$", name) is not None

    def test_manifest_state_transition(self) -> None:
        """`manifest_state_transition` の振る舞いを検証する。"""
        with tempfile.TemporaryDirectory() as td:
            layout = create_run_layout(td, "run")
            m1 = update_manifest(layout, {"status": "running", "config": {"x": 1}, "path": Path("a/b")})
            assert m1["status"] == "running"
            assert m1["config"]["x"] == 1
            assert m1["path"] == "a/b"

            m2 = update_manifest(layout, {"status": "completed", "result": {"ok": True}})
            assert m2["status"] == "completed"
            assert m2["result"]["ok"] is True
            assert m2["config"]["x"] == 1

    def test_coerce_optional_path_for_run_root_resolution(self) -> None:
        """`coerce_optional_path_for_run_root_resolution` の振る舞いを検証する。"""
        assert coerce_optional_path("") is None
        assert coerce_optional_path(".", dot_is_none=True) is None
        assert coerce_optional_path("logs/run") == Path("logs/run")
