from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from reinforce.ppo_discrete.pipeline import pipeline_service as ps


def _make_pipeline_args(tmp_path: Path, **overrides) -> SimpleNamespace:
    data = dict(
        env_id="AHC061Local-v0",
        env_kwargs_json="{}",
        eval_env_kwargs_json="",
        run_root=tmp_path,
        run_name="pipeline_unit",
        resume=False,
        seed=1,
        ppo_eval_env_kwargs_json="",
        mlflow_tracking_uri="",
        mlflow_experiment="ppo_discrete",
        mlflow_run_name="",
        bc_teacher_model_path=None,
        ppo_init_model=None,
        skip_bc=True,
        skip_ppo=True,
        skip_last_eval=True,
        ppo_distributed="auto",
        ppo_total_timesteps=1000,
        ppo_aux_opp_param_loss_coef=0.0,
        ppo_aux_opp_param_use_valid_mask=True,
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def test_run_pipeline_skip_all_stages(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ps,
        "_maybe_prepare_cpp_bayes_backend",
        lambda **_kwargs: {"enabled": False, "prepared": False, "train_backend": "", "eval_backend": ""},
    )

    args = _make_pipeline_args(tmp_path, skip_bc=True, skip_ppo=True, skip_last_eval=True)
    rc = ps.run_pipeline(args)
    assert rc == 0

    summary_path = tmp_path / "pipeline_unit" / "reports" / "pipeline_summary.json"
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert obj["stages"]["bc"]["skipped"] is True
    assert obj["stages"]["ppo"]["skipped"] is True
    assert obj["stages"]["eval"]["skipped"] is True


def test_run_pipeline_bc_stage_isolated(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        ps,
        "_maybe_prepare_cpp_bayes_backend",
        lambda **_kwargs: {"enabled": False, "prepared": False, "train_backend": "", "eval_backend": ""},
    )
    monkeypatch.setattr(ps, "build_train_bc_cmd", lambda **_kwargs: ["python", "-m", "dummy_train_bc"])

    def _fake_run(cmd: list[str], *, tracker, stage: str | None, tee_fp) -> None:
        calls.append((str(stage), list(cmd)))

    monkeypatch.setattr(ps, "run", _fake_run)

    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"x")
    args = _make_pipeline_args(
        tmp_path,
        skip_bc=False,
        skip_ppo=True,
        skip_last_eval=True,
        bc_teacher_model_path=teacher,
    )
    rc = ps.run_pipeline(args)
    assert rc == 0
    assert calls == [("train_bc", ["python", "-m", "dummy_train_bc"])]

    summary_path = tmp_path / "pipeline_unit" / "reports" / "pipeline_summary.json"
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert obj["stages"]["bc"]["skipped"] is False
    assert obj["stages"]["bc"]["teacher_model"] == str(teacher)


def test_run_pipeline_resume_skips_ppo_when_target_already_reached(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        ps,
        "_maybe_prepare_cpp_bayes_backend",
        lambda **_kwargs: {"enabled": False, "prepared": False, "train_backend": "", "eval_backend": ""},
    )

    def _fake_run(cmd: list[str], *, tracker, stage: str | None, tee_fp) -> None:
        calls.append((str(stage), list(cmd)))

    monkeypatch.setattr(ps, "run", _fake_run)

    run_name = "resume_ppo"
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "ppo_final.pt").write_bytes(b"x")
    latest_ppo = run_dir / "artifacts" / "ppo_runs" / "r0"
    (latest_ppo / "models").mkdir(parents=True, exist_ok=True)
    (latest_ppo / "reports").mkdir(parents=True, exist_ok=True)
    (latest_ppo / "models" / "last.pt").write_bytes(b"x")
    (latest_ppo / "reports" / "train_summary.json").write_text(
        json.dumps({"global_step": 1000}),
        encoding="utf-8",
    )

    args = _make_pipeline_args(
        tmp_path,
        run_name=run_name,
        resume=True,
        skip_bc=True,
        skip_ppo=False,
        skip_last_eval=True,
        ppo_total_timesteps=800,
    )
    rc = ps.run_pipeline(args)
    assert rc == 0
    assert calls == []

    summary_path = run_dir / "reports" / "pipeline_summary.json"
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert obj["stages"]["ppo"]["resumed"] is True
    assert obj["stages"]["ppo"]["resumed_step"] == 1000
    assert obj["trained_model"] == str(run_dir / "models" / "ppo_final.pt")


def test_run_pipeline_resume_skips_eval_when_report_exists(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        ps,
        "_maybe_prepare_cpp_bayes_backend",
        lambda **_kwargs: {"enabled": False, "prepared": False, "train_backend": "", "eval_backend": ""},
    )

    def _fake_run(cmd: list[str], *, tracker, stage: str | None, tee_fp) -> None:
        calls.append((str(stage), list(cmd)))

    monkeypatch.setattr(ps, "run", _fake_run)

    run_name = "resume_eval"
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "eval_policy.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    args = _make_pipeline_args(
        tmp_path,
        run_name=run_name,
        resume=True,
        skip_bc=True,
        skip_ppo=True,
        skip_last_eval=False,
    )
    rc = ps.run_pipeline(args)
    assert rc == 0
    assert calls == []

    summary_path = run_dir / "reports" / "pipeline_summary.json"
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert obj["stages"]["eval"]["resumed"] is True
    assert obj["stages"]["eval"]["report"] == str(run_dir / "reports" / "eval_policy.json")
