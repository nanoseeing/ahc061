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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TextIO

from ..opponent import ensure_cpp_backend
from ..utils.experiment import coerce_optional_path, create_run_layout, make_run_name, to_jsonable, update_manifest
from ..utils.log_utils import get_logger
from ..utils.tracking import MetricTracker
from .pipeline_commands import (
    PipelineArgs,
    build_collect_teacher_cmd,
    build_eval_policy_cmd,
    build_train_bc_cmd,
    build_train_ppo_cmd,
)
from ..data.teacher_dataset_merge import cleanup_teacher_shards, merge_teacher_shards, split_counts

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


def _run_collect_worker_stream(
    *,
    wi: int,
    shard: Path,
    cmd: list[str],
    tee_fp: TextIO | None,
) -> int:
    _emit_child_output(f"[collect_worker={wi}] cmd={' '.join(cmd)}", tee_fp=tee_fp)
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
            line = line.rstrip("\n")
            if line:
                _emit_child_output(f"[collect_worker={wi}] {line}", tee_fp=tee_fp)
        rc = int(proc.wait())
    logger.info("collect worker=%d rc=%d shard=%s", wi, rc, shard)
    return rc


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


def _parse_env_kwargs(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--env-kwargs-json must be a JSON object")
    return obj


def _validate_backend_combinations(args: PipelineArgs) -> None:
    env_id = str(args.env_id).strip()
    if env_id != "AHC061Local-v0":
        raise ValueError("run_pipeline supports only env_id=AHC061Local-v0")

    collect_policy = str(args.collect_policy).strip()

    if not bool(args.skip_collect):
        if collect_policy in ("model_stochastic", "model_greedy"):
            raise ValueError(
                "run_pipeline collect stage does not accept model_* policy yet "
                "(model path is not configurable in pipeline collect). "
                "Use collect_policy=random, or run collect_teacher directly."
            )
        if float(args.bc_aux_opp_param_loss_coef) > 0.0:
            if not bool(args.collect_save_aux_targets):
                raise ValueError(
                    "bc_aux_opp_param_loss_coef > 0 requires collect_save_aux_targets=true "
                    "when collect stage is active"
                )

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


def _run_collect_workers(
    *,
    py: str,
    args: PipelineArgs,
    collect_policy: str,
    env_kwargs: dict[str, Any],
    output_npz: Path,
    tracker: MetricTracker | None,
    merge_output: bool = True,
    cleanup_shards: bool = True,
    tee_fp: TextIO | None = None,
) -> dict[str, Any]:
    workers = int(args.collect_workers)
    env_kwargs_cmd: dict[str, Any] = dict(env_kwargs)
    if workers <= 1:
        cmd = build_collect_teacher_cmd(
            py=py,
            env_id=str(args.env_id),
            episodes=int(args.collect_episodes),
            seed=int(args.seed),
            collect_policy=str(collect_policy),
            chunk_episodes=int(args.collect_chunk_episodes),
            output_npz=output_npz,
            env_kwargs=dict(env_kwargs_cmd),
            feature_id=str(args.collect_feature_id),
            pf_enabled=bool(args.collect_pf_enabled),
            amp=bool(args.collect_amp),
            save_aux_targets=bool(args.collect_save_aux_targets),
        )
        run(cmd, tracker=tracker, stage="collect_teacher", tee_fp=tee_fp)
        return {"workers": 1, "shards": [str(output_npz)], "merged": True, "output_npz": str(output_npz)}

    shards_dir = output_npz.parent / "collect_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_eps = [x for x in _split_counts(int(args.collect_episodes), workers) if x > 0]
    if not shard_eps:
        raise ValueError("collect_episodes must be > 0")

    cmds: list[tuple[int, Path, list[str]]] = []
    for wi, episodes in enumerate(shard_eps):
        shard_path = shards_dir / f"teacher_shard_{wi:03d}.npz"
        seed_i = int(args.seed) + wi * 100003
        env_kwargs_i = dict(env_kwargs_cmd)
        cmd = build_collect_teacher_cmd(
            py=py,
            env_id=str(args.env_id),
            episodes=int(episodes),
            seed=int(seed_i),
            collect_policy=str(collect_policy),
            chunk_episodes=int(args.collect_chunk_episodes),
            output_npz=shard_path,
            env_kwargs=env_kwargs_i,
            feature_id=str(args.collect_feature_id),
            pf_enabled=bool(args.collect_pf_enabled),
            amp=bool(args.collect_amp),
            save_aux_targets=bool(args.collect_save_aux_targets),
        )
        cmds.append((wi, shard_path, cmd))

    t0 = time.time()
    if tracker is not None:
        tracker.log_event(
            "stage_start",
            {
                "stage": "collect_teacher_parallel",
                "workers": len(cmds),
                "episodes": int(args.collect_episodes),
            },
        )

    with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
        futs = {
            ex.submit(
                _run_collect_worker_stream,
                wi=wi,
                shard=shard,
                cmd=cmd,
                tee_fp=tee_fp,
            ): (wi, shard, cmd)
            for wi, shard, cmd in cmds
        }
        for fut in as_completed(futs):
            wi, shard, cmd = futs[fut]
            rc = int(fut.result())
            if rc != 0:
                raise RuntimeError(
                    "collect worker failed: "
                    f"idx={wi} rc={rc}\ncmd={' '.join(cmd)}"
                )

    shard_paths = [shard for _, shard, _ in cmds]
    if merge_output:
        merge_teacher_shards(shard_paths, output_npz, offset_episode_ids=True)
        if cleanup_shards:
            cleanup_teacher_shards(shard_paths, shards_dir=shards_dir)
    elapsed = time.time() - t0
    if tracker is not None:
        tracker.log_event(
            "stage_done",
            {
                "stage": "collect_teacher_parallel",
                "elapsed_sec": elapsed,
                "workers": len(cmds),
                "output_npz": str(output_npz),
            },
        )
    return {
        "workers": len(cmds),
        "shards": [str(s) for _, s, _ in cmds],
        "merged": bool(merge_output),
        "output_npz": (str(output_npz) if merge_output else ""),
        "shards_dir": str(shards_dir),
    }


def run_pipeline(args: PipelineArgs) -> int:
    _validate_backend_combinations(args)
    base_env_kwargs = _parse_env_kwargs(args.env_kwargs_json)
    eval_env_kwargs = (
        _parse_env_kwargs(args.eval_env_kwargs_json) if str(args.eval_env_kwargs_json).strip() else dict(base_env_kwargs)
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
    teacher_npz = layout.data_dir / "teacher.npz"
    collect_shards_dir = teacher_npz.parent / "collect_shards"
    collect_shards_glob = str(collect_shards_dir / "teacher_shard_*.npz")
    bc_model = layout.models_dir / "bc_init.pt"
    ppo_init_model = coerce_optional_path(args.ppo_init_model, dot_is_none=True)
    if ppo_init_model is not None and not ppo_init_model.exists():
        raise FileNotFoundError(f"--ppo-init-model was set but file does not exist: {ppo_init_model}")
    ppo_root = layout.artifacts_dir / "ppo_runs"
    ppo_root.mkdir(parents=True, exist_ok=True)

    stage_result: dict[str, Any] = {
        "resume": {"enabled": bool(resume_enabled)},
        "bayes_backend": bayes_backend_meta,
        "collect": {"skipped": bool(args.skip_collect)},
        "bc": {"skipped": bool(args.skip_bc)},
        "ppo": {"skipped": bool(args.skip_ppo)},
        "eval": {"skipped": bool(args.skip_last_eval)},
        "logs": {"stdout_log": str(stdout_log_path)},
    }

    try:
        bc_prefers_collect_shards = bool(
            (not args.skip_bc)
            and bool(args.bc_use_collect_shards)
            and int(args.collect_workers) > 1
        )
        if not args.skip_collect:
            resume_has_teacher_npz = bool(teacher_npz.exists())
            resume_has_collect_shards = bool(
                bc_prefers_collect_shards
                and collect_shards_dir.exists()
                and any(collect_shards_dir.glob("teacher_shard_*.npz"))
            )
            if resume_enabled and (resume_has_teacher_npz or resume_has_collect_shards):
                if resume_has_teacher_npz:
                    logger.info("resume: skip collect_teacher (found %s)", teacher_npz)
                else:
                    logger.info("resume: skip collect_teacher (found shards %s)", collect_shards_glob)
                stage_result["collect"]["resumed"] = True
                stage_result["collect"]["teacher_npz"] = (str(teacher_npz) if resume_has_teacher_npz else "")
                if resume_has_collect_shards:
                    stage_result["collect"]["teacher_shards_glob"] = collect_shards_glob
            else:
                collect_policy = args.collect_policy
                collect_info = _run_collect_workers(
                    py=py,
                    args=args,
                    collect_policy=collect_policy,
                    env_kwargs=env_kwargs,
                    output_npz=teacher_npz,
                    tracker=tracker,
                    merge_output=not bc_prefers_collect_shards,
                    cleanup_shards=not bc_prefers_collect_shards,
                    tee_fp=stdout_fp,
                )
                if bool(collect_info.get("merged")):
                    stage_result["collect"]["teacher_npz"] = str(teacher_npz)
                else:
                    stage_result["collect"]["teacher_npz"] = ""
                stage_result["collect"]["teacher_shards_glob"] = collect_shards_glob
                stage_result["collect"]["policy"] = collect_policy
                stage_result["collect"]["feature_id"] = str(args.collect_feature_id)
                stage_result["collect"]["pf_enabled"] = bool(args.collect_pf_enabled)
                stage_result["collect"]["save_aux_targets"] = bool(args.collect_save_aux_targets)
                stage_result["collect"]["workers"] = int(collect_info["workers"])
                stage_result["collect"]["shards"] = list(collect_info["shards"])
                stage_result["collect"]["merged"] = bool(collect_info.get("merged"))

        if not args.skip_bc:
            if resume_enabled and bc_model.exists():
                logger.info("resume: skip train_bc (found %s)", bc_model)
                stage_result["bc"]["resumed"] = True
                stage_result["bc"]["model"] = str(bc_model)
            else:
                use_shards_for_bc = False
                if bool(args.bc_use_collect_shards) and int(args.collect_workers) > 1:
                    has_shards = collect_shards_dir.exists() and any(collect_shards_dir.glob("teacher_shard_*.npz"))
                    if has_shards:
                        use_shards_for_bc = True

                bc_dataset_npz: Path | None = None
                bc_dataset_shards_glob = ""
                if use_shards_for_bc:
                    bc_dataset_shards_glob = collect_shards_glob
                    stage_result["bc"]["dataset"] = {"mode": "shards", "glob": collect_shards_glob}
                else:
                    if not teacher_npz.exists():
                        raise FileNotFoundError(
                            f"teacher dataset was not found: {teacher_npz} (and no collect shards matched {collect_shards_glob})"
                        )
                    bc_dataset_npz = teacher_npz
                    stage_result["bc"]["dataset"] = {"mode": "npz", "path": str(teacher_npz)}
                bc_cmd = build_train_bc_cmd(
                    py=py,
                    args=args,
                    output_model=bc_model,
                    dataset_npz=bc_dataset_npz,
                    dataset_shards_glob=bc_dataset_shards_glob,
                )
                run(bc_cmd, tracker=tracker, stage="train_bc", tee_fp=stdout_fp)
                stage_result["bc"]["model"] = str(bc_model)
                stage_result["bc"]["aux_opp_param_loss_coef"] = float(args.bc_aux_opp_param_loss_coef)
                stage_result["bc"]["aux_opp_param_use_valid_mask"] = bool(args.bc_aux_opp_param_use_valid_mask)

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
