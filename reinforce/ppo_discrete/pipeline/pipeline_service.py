from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from ..opponent import ensure_cpp_backend
from ..utils.experiment import coerce_optional_path, create_run_layout, make_run_name, to_jsonable, update_manifest
from ..utils.log_utils import get_logger
from ..utils.runtime import parse_json_object
from ..utils.tracking import MetricTracker
from .pipeline_commands import (
    PipelineArgs,
    build_eval_policy_cmd,
    build_train_bc_cmd,
    build_train_ppo_cmd,
)

logger = get_logger("run_pipeline")
_STREAM_EMIT_LOCK = threading.Lock()


def _attach_file_handler(log: logging.Logger, path: Path) -> logging.FileHandler | None:
    abs_path = str(path.resolve())
    for h in log.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == abs_path:
            return None
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    log.addHandler(h)
    return h


def _module_version(name: str) -> str:
    try:
        return str(importlib.metadata.version(name))
    except Exception:
        pass
    try:
        mod = importlib.import_module(name)
    except Exception:
        return ""
    return str(getattr(mod, "__version__", "") or "")


def _runtime_env_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    versions: dict[str, str] = {}
    for name in ("torch", "torchvision", "torchaudio", "numpy", "pybind11", "mlflow", "tensorboard", "nncv", "mmcv", "mmengine"):
        v = _module_version(name)
        if v:
            versions[name] = v
    if versions:
        info["versions"] = versions

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_info: dict[str, Any] = {
            "available": cuda_available,
            "version": str(getattr(torch.version, "cuda", None) or ""),
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
        cudnn_ver = torch.backends.cudnn.version()
        if cudnn_ver is not None:
            cuda_info["cudnn"] = int(cudnn_ver)
        if cuda_available and int(cuda_info["device_count"]) > 0:
            cuda_info["device0_name"] = str(torch.cuda.get_device_name(0))
        info["cuda"] = cuda_info
    except Exception as e:
        info["cuda"] = {"error": str(e)}

    return info


def _stream_subprocess(cmd: list[str], *, tee_fp: TextIO | None) -> None:
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if tee_fp is not None:
                tee_fp.write(line)
                tee_fp.flush()
        rc = int(proc.wait())
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _emit_child_output(text: str, *, tee_fp: TextIO | None) -> None:
    if not text:
        return
    if not text.endswith("\n"):
        text += "\n"
    with _STREAM_EMIT_LOCK:
        sys.stdout.write(text)
        sys.stdout.flush()
        if tee_fp is not None:
            tee_fp.write(text)
            tee_fp.flush()



def run(
    cmd: list[str],
    *,
    tracker: MetricTracker | None = None,
    stage: str | None = None,
    tee_fp: TextIO | None = None,
) -> None:
    cmd_text = " ".join(cmd)
    logger.info("run: %s", cmd_text)
    t0 = time.time()
    if tracker is not None and stage is not None:
        tracker.log_event("stage_start", {"stage": stage, "cmd": cmd_text})
    try:
        _stream_subprocess(cmd, tee_fp=tee_fp)
        elapsed = time.time() - t0
        if tracker is not None and stage is not None:
            tracker.log_event("stage_done", {"stage": stage, "elapsed_sec": elapsed})
    except Exception as e:
        elapsed = time.time() - t0
        if tracker is not None and stage is not None:
            tracker.log_event("stage_failed", {"stage": stage, "elapsed_sec": elapsed, "error": str(e)})
        raise


def latest_dir(root: Path) -> Path:
    cands = [p for p in root.iterdir() if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"no subdirs under: {root}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def maybe_latest_dir(root: Path) -> Path | None:
    cands = [p for p in root.iterdir() if p.is_dir()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def resolve_ppo_model(ppo_run_dir: Path) -> Path:
    candidates = [
        ppo_run_dir / "models" / "best.pt",
        ppo_run_dir / "models" / "last.pt",
        ppo_run_dir / "best.pt",
        ppo_run_dir / "last.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"no trained model found under: {ppo_run_dir}")


def read_ppo_global_step(ppo_run_dir: Path) -> int | None:
    candidates = [
        ppo_run_dir / "reports" / "train_summary.json",
        ppo_run_dir / "summary.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "global_step" in obj:
                return int(obj["global_step"])
        except Exception:
            continue
    return None


def _validate_backend_combinations(args: PipelineArgs) -> None:
    env_id = str(args.env_id).strip()
    if env_id != "AHC061Local-v0":
        raise ValueError("run_pipeline supports only env_id=AHC061Local-v0")

    if not bool(args.skip_ppo) and str(args.ppo_distributed).strip().lower() == "on":
        raise ValueError(
            "run_pipeline does not launch torchrun. "
            "Use ppo_distributed=auto|off in run_pipeline, or launch train_ppo with torchrun directly."
        )


def _normalize_bayes_backend_name(x: Any) -> str:
    b = str(x).strip().lower()
    if not b:
        b = "auto"
    if b == "python":
        raise ValueError(
            "bayes_backend='python' is no longer supported; "
            "use bayes_backend='cpp' (or 'auto' which resolves to cpp)"
        )
    if b not in ("auto", "cpp"):
        raise ValueError(f"unsupported bayes_backend={x!r}; expected auto|cpp")
    return b


def _maybe_prepare_cpp_bayes_backend(*, env_id: str, env_kwargs: dict[str, Any], eval_env_kwargs: dict[str, Any]) -> dict[str, Any]:
    if str(env_id) != "AHC061Local-v0":
        return {"enabled": False, "prepared": False, "train_backend": "", "eval_backend": ""}

    train_backend = _normalize_bayes_backend_name(env_kwargs.get("bayes_backend", "auto"))
    eval_backend = _normalize_bayes_backend_name(eval_env_kwargs.get("bayes_backend", train_backend))

    meta = {
        "enabled": True,
        "prepared": False,
        "train_backend": train_backend,
        "eval_backend": eval_backend,
    }

    ok = ensure_cpp_backend(build_if_missing=True, force_build=False, verbose=False)
    if not ok:
        raise RuntimeError(
            "AHC061 bayes backend requires cpp implementation, but cpp backend build/import failed"
        )
    meta["prepared"] = bool(ok)
    return meta



def run_pipeline(args: PipelineArgs) -> int:
    _validate_backend_combinations(args)
    base_env_kwargs = parse_json_object(str(args.env_kwargs_json), field_name="--env-kwargs-json")
    eval_env_kwargs = (
        parse_json_object(str(args.eval_env_kwargs_json), field_name="--eval-env-kwargs-json")
        if str(args.eval_env_kwargs_json).strip()
        else dict(base_env_kwargs)
    )
    env_kwargs = dict(base_env_kwargs)
    resume_enabled = bool(args.resume)

    if resume_enabled and not str(args.run_name).strip():
        raise ValueError("--resume requires --run-name for an existing pipeline directory")
    run_name = args.run_name or make_run_name(args.env_id.replace("/", "_") + "_pipeline", seed=args.seed)
    layout = create_run_layout(args.run_root, run_name)
    if resume_enabled and not (layout.root / "manifest.json").exists():
        raise FileNotFoundError(f"resume requested but manifest was not found: {layout.root / 'manifest.json'}")

    bayes_backend_meta = _maybe_prepare_cpp_bayes_backend(
        env_id=str(args.env_id),
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
    )
    if bool(bayes_backend_meta.get("enabled")):
        logger.info(
            "bayes_backend: train=%s eval=%s cpp_prepared=%s",
            bayes_backend_meta.get("train_backend"),
            bayes_backend_meta.get("eval_backend"),
            bayes_backend_meta.get("prepared"),
        )

    ppo_val_env_kwargs_json_resolved = str(args.ppo_eval_env_kwargs_json).strip()
    if not ppo_val_env_kwargs_json_resolved:
        ppo_val_env_kwargs_json_resolved = json.dumps(eval_env_kwargs)

    config_snapshot = to_jsonable(
        {
            "args": vars(args),
            "env_kwargs": env_kwargs,
            "eval_env_kwargs": eval_env_kwargs,
            "ppo_val_env_kwargs_json_resolved": ppo_val_env_kwargs_json_resolved,
            "bayes_backend": bayes_backend_meta,
            "layout": layout.as_dict(),
        }
    )
    (layout.config_dir / "run_pipeline.args.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    stdout_log_path = layout.logs_dir / "pipeline.stdout.log"
    stdout_fp = stdout_log_path.open("a", encoding="utf-8", buffering=1)
    file_handler = _attach_file_handler(logger, stdout_log_path)
    runtime_env = _runtime_env_snapshot()
    logger.info("runtime_env=%s", json.dumps(runtime_env, ensure_ascii=False, sort_keys=True))

    tracker = MetricTracker(
        layout.root,
        run_name=run_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        config=config_snapshot,
    )
    tracker.log_event("runtime_env", runtime_env)

    update_manifest(
        layout,
        {
            "job": "run_pipeline",
            "status": "running",
            "run_name": run_name,
            "layout": layout.as_dict(),
            "config": config_snapshot,
            "logs": {"stdout_log": str(stdout_log_path)},
            "timestamps": {"started_at": time.time()},
        },
    )

    py = sys.executable
    bc_model = layout.models_dir / "bc_init.pt"
    bc_teacher_model_path = coerce_optional_path(args.bc_teacher_model_path, dot_is_none=True)
    ppo_init_model = coerce_optional_path(args.ppo_init_model, dot_is_none=True)
    if ppo_init_model is not None and not ppo_init_model.exists():
        raise FileNotFoundError(f"--ppo-init-model was set but file does not exist: {ppo_init_model}")
    ppo_root = layout.artifacts_dir / "ppo_runs"
    ppo_root.mkdir(parents=True, exist_ok=True)

    stage_result: dict[str, Any] = {
        "resume": {"enabled": bool(resume_enabled)},
        "bayes_backend": bayes_backend_meta,
        "bc": {"skipped": bool(args.skip_bc) or bc_teacher_model_path is None},
        "ppo": {"skipped": bool(args.skip_ppo)},
        "eval": {"skipped": bool(args.skip_last_eval)},
        "logs": {"stdout_log": str(stdout_log_path)},
    }

    try:
        if not args.skip_bc and bc_teacher_model_path is not None:
            if resume_enabled and bc_model.exists():
                logger.info("resume: skip train_bc (found %s)", bc_model)
                stage_result["bc"]["resumed"] = True
                stage_result["bc"]["model"] = str(bc_model)
            else:
                if not bc_teacher_model_path.exists():
                    raise FileNotFoundError(f"bc_teacher_model_path does not exist: {bc_teacher_model_path}")
                bc_cmd = build_train_bc_cmd(
                    py=py,
                    args=args,
                    output_model=bc_model,
                    teacher_model_path=bc_teacher_model_path,
                )
                run(bc_cmd, tracker=tracker, stage="train_bc", tee_fp=stdout_fp)
                stage_result["bc"]["model"] = str(bc_model)
                stage_result["bc"]["teacher_model"] = str(bc_teacher_model_path)

        trained_model = bc_model
        consolidated_model = layout.models_dir / "ppo_final.pt"
        if not args.skip_ppo:
            resume_ppo_dir = maybe_latest_dir(ppo_root) if resume_enabled else None
            resume_ppo_ckpt = None
            if resume_ppo_dir is not None:
                cand = resume_ppo_dir / "models" / "last.pt"
                if cand.exists():
                    resume_ppo_ckpt = cand
            resume_ppo_step = read_ppo_global_step(resume_ppo_dir) if resume_ppo_dir is not None else None
            if (
                resume_enabled
                and consolidated_model.exists()
                and resume_ppo_step is not None
                and int(resume_ppo_step) >= int(args.ppo_total_timesteps)
            ):
                logger.info(
                    "resume: skip train_ppo (already reached step %d >= target %d)",
                    int(resume_ppo_step),
                    int(args.ppo_total_timesteps),
                )
                trained_model = consolidated_model
                stage_result["ppo"]["resumed"] = True
                stage_result["ppo"]["resumed_step"] = int(resume_ppo_step)
                stage_result["ppo"]["consolidated_model"] = str(consolidated_model)
            else:

                cmd_init_model: Path | None = None
                cmd_resume_name = ""
                cmd_resume_from: Path | None = None
                if resume_ppo_ckpt is not None and resume_ppo_dir is not None:
                    cmd_resume_name = str(resume_ppo_dir.name)
                    cmd_resume_from = resume_ppo_ckpt
                    stage_result["ppo"]["resume_from"] = str(resume_ppo_ckpt)
                elif ppo_init_model is not None:
                    cmd_init_model = ppo_init_model
                    stage_result["ppo"]["init_model"] = str(ppo_init_model)
                elif bc_model.exists():
                    cmd_init_model = bc_model
                    stage_result["ppo"]["init_model"] = str(bc_model)
                cmd = build_train_ppo_cmd(
                    py=py,
                    args=args,
                    run_dir=ppo_root,
                    env_kwargs=env_kwargs,
                    ppo_val_env_kwargs_json=ppo_val_env_kwargs_json_resolved,
                    init_model=cmd_init_model,
                    resume_run_name=cmd_resume_name,
                    resume_from=cmd_resume_from,
                )
                stage_result["ppo"]["aux_opp_param_loss_coef"] = float(args.ppo_aux_opp_param_loss_coef)
                stage_result["ppo"]["aux_opp_param_use_valid_mask"] = bool(args.ppo_aux_opp_param_use_valid_mask)
                run(cmd, tracker=tracker, stage="train_ppo", tee_fp=stdout_fp)

                latest = latest_dir(ppo_root)
                trained_model = resolve_ppo_model(latest)
                stage_result["ppo"]["run_dir"] = str(latest)
                stage_result["ppo"]["model"] = str(trained_model)

                if consolidated_model.exists() or consolidated_model.is_symlink():
                    consolidated_model.unlink()
                shutil.copy2(trained_model, consolidated_model)
                trained_model = consolidated_model
                stage_result["ppo"]["consolidated_model"] = str(consolidated_model)

        if not args.skip_last_eval:
            eval_json = layout.reports_dir / "eval_policy.json"
            if resume_enabled and eval_json.exists():
                logger.info("resume: skip eval_policy (found %s)", eval_json)
                stage_result["eval"]["resumed"] = True
            else:
                eval_cmd = build_eval_policy_cmd(
                    py=py,
                    args=args,
                    model_path=trained_model,
                    output_json=eval_json,
                    env_kwargs=eval_env_kwargs,
                )
                run(
                    eval_cmd,
                    tracker=tracker,
                    stage="evaluate_policy",
                    tee_fp=stdout_fp,
                )
            stage_result["eval"]["report"] = str(eval_json)
            stage_result["eval"]["env_kwargs"] = to_jsonable(eval_env_kwargs)

        summary = {
            "run_name": run_name,
            "run_dir": str(layout.root),
            "trained_model": str(trained_model),
            "stages": stage_result,
            "timestamps": {"finished_at": time.time()},
        }
        (layout.reports_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        update_manifest(layout, {"status": "completed", "result": summary})
        tracker.log_event("pipeline_complete", summary)
        logger.info("done: %s", layout.root)
        return 0
    except Exception as e:
        update_manifest(layout, {"status": "failed", "error": str(e), "timestamps": {"failed_at": time.time()}})
        tracker.log_event("pipeline_failed", {"error": str(e)})
        raise
    finally:
        stdout_fp.close()
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()
        tracker.close()
