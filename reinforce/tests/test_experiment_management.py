from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from reinforce.ppo_discrete.utils.experiment import (
    coerce_optional_path,
    create_run_layout,
    make_run_name,
    resolve_config,
    update_manifest,
)


class TestExperimentManagement:
    def test_make_run_name_contains_seed_ms_and_pid(self) -> None:
        name = make_run_name("CartPole/v1", seed=7, now=1700000000.123)
        assert re.search(r"^CartPole_v1__seed7__1700000000123__p\d+$", name) is not None

    def test_manifest_state_transition(self) -> None:
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

    def test_resolve_config_merge_and_overrides(self) -> None:
        defaults = {"epochs": 5, "optim": {"lr": 0.01, "wd": 0.1}}
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.toml"
            cfg_path.write_text(
                """
[train_bc]
epochs = 10

[train_bc.optim]
lr = 0.001
""".strip()
                + "\n",
                encoding="utf-8",
            )
            cfg = resolve_config(
                defaults=defaults,
                config_file=cfg_path,
                config_section="train_bc",
                overrides=["optim.lr=0.005", "optim.wd=0.0"],
            )
        assert cfg["epochs"] == 10
        assert cfg["optim"]["lr"] == pytest.approx(0.005, abs=1e-12)
        assert cfg["optim"]["wd"] == pytest.approx(0.0, abs=1e-12)

    def test_coerce_optional_path_for_run_root_resolution(self) -> None:
        assert coerce_optional_path("") is None
        assert coerce_optional_path(".", dot_is_none=True) is None
        assert coerce_optional_path("logs/run") == Path("logs/run")
