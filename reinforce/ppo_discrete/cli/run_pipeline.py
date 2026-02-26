from __future__ import annotations

import argparse
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
from tempfile import TemporaryDirectory
from typing import Any, TextIO

import numpy as np

from ..domains.ahc061.opponent_bayes import ensure_cpp_bayes_backend
from ..runtime.experiment import coerce_optional_path, create_run_layout, make_run_name, resolve_config, to_jsonable, update_manifest
from ..runtime.log_utils import get_logger
from ..runtime.teacher_dataset import (
    AHC061_BAYES_TAIL_SHAPE,
    AHC061_OPP_PARAM_TRUE_TAIL_SHAPE,
    AHC061_OPP_VALID_TAIL_SHAPE,
    DATASET_KEY_OPP_PARAM_TRUE,
    DATASET_KEY_OPP_VALID,
)
from ..runtime.tracking import MetricTracker

logger = get_logger("run_pipeline")
_STREAM_EMIT_LOCK = threading.Lock()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run skeleton PPO pipeline: collect -> BC -> PPO(val) -> eval")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="run_pipeline", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--env-kwargs-json", type=str, default="{}")
    p.add_argument("--run-root", type=Path, default=Path("reinforce/outputs/pipeline_runs"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="resume an existing pipeline run (requires --run-name)")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--collect-episodes", type=int, default=200)
    p.add_argument(
        "--collect-policy",
        choices=["random", "model_stochastic", "model_greedy", "ahc061_main_greedy"],
        default="random",
    )
    p.add_argument("--collect-workers", type=int, default=1)
    p.add_argument("--collect-vector-env", choices=["sync", "async"], default="sync")
    p.add_argument("--collect-rollout-backend", choices=["gym", "native"], default="gym")
    p.add_argument("--collect-native-feature-id", type=str, default="submit_v1")
    p.add_argument("--collect-native-pf-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--collect-native-amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--collect-native-save-aux-targets", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--collect-chunk-episodes", type=int, default=10)
    p.add_argument("--skip-collect", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--bc-aux-opp-param-loss-coef", type=float, default=0.0)
    p.add_argument("--bc-aux-opp-param-use-valid-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--bc-use-collect-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train BC directly from collect worker shards when available to reduce memory usage",
    )
    p.add_argument("--skip-bc", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--ppo-total-timesteps", type=int, default=200_000)
    p.add_argument("--ppo-num-envs", type=int, default=8)
    p.add_argument("--ppo-num-steps", type=int, default=128)
    p.add_argument("--ppo-learning-rate", type=float, default=2.5e-4)
    p.add_argument("--ppo-learning-rate-schedule", type=str, default="linear")
    p.add_argument("--ppo-gamma", type=float, default=0.99)
    p.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-num-minibatches", type=int, default=4)
    p.add_argument("--ppo-update-epochs", type=int, default=4)
    p.add_argument("--ppo-norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-clip-coef", type=float, default=0.2)
    p.add_argument("--ppo-clip-range-vf", type=float, default=None)
    p.add_argument("--ppo-clip-range-vf-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--ppo-clip-range-vf-final", type=float, default=None)
    p.add_argument("--ppo-clip-range-vf-schedule-expr", type=str, default="")
    p.add_argument("--ppo-clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-ent-coef", type=float, default=0.01)
    p.add_argument("--ppo-ent-coef-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--ppo-ent-coef-final", type=float, default=None)
    p.add_argument("--ppo-ent-coef-schedule-expr", type=str, default="")
    p.add_argument("--ppo-vf-coef", type=float, default=0.5)
    p.add_argument("--ppo-aux-opp-param-loss-coef", type=float, default=0.0)
    p.add_argument("--ppo-aux-opp-param-use-valid-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    p.add_argument("--ppo-target-kl", type=float, default=None, help="SB3 semantics: early stop when approx_kl > 1.5 * target_kl")
    p.add_argument("--ppo-clip-coef-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--ppo-clip-coef-final", type=float, default=None)
    p.add_argument("--ppo-clip-coef-schedule-expr", type=str, default="")
    p.add_argument("--ppo-checkpoint-interval-steps", type=int, default=0)
    p.add_argument("--ppo-vector-env", choices=["sync", "async"], default="sync")
    p.add_argument("--ppo-rollout-backend", choices=["gym", "native"], default="gym")
    p.add_argument("--ppo-native-feature-id", type=str, default="submit_v1")
    p.add_argument("--ppo-native-pf-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-native-amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ppo-native-memory-format", choices=["auto", "nchw", "channels_last"], default="auto")
    p.add_argument("--ppo-native-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-native-rollout-cache-device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--ppo-native-distributed", choices=["auto", "off", "on"], default="auto")
    p.add_argument("--ppo-native-model-preset", type=str, default="")
    p.add_argument("--ppo-val-interval-steps", "--ppo-eval-interval-steps", dest="ppo_eval_interval_steps", type=int, default=0)
    p.add_argument("--ppo-val-episodes", "--ppo-eval-episodes", dest="ppo_eval_episodes", type=int, default=100)
    p.add_argument("--ppo-val-seed-start", "--ppo-eval-seed-start", dest="ppo_eval_seed_start", type=int, default=2_000_000)
    p.add_argument("--ppo-val-fixed-seeds", "--ppo-eval-fixed-seeds", dest="ppo_eval_fixed_seeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-val-deterministic", "--ppo-eval-deterministic", dest="ppo_eval_deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-val-at-start", "--ppo-eval-at-start", dest="ppo_eval_at_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-val-vector-env", "--ppo-eval-vector-env", dest="ppo_eval_vector_env", choices=["sync", "async"], default="sync")
    p.add_argument("--ppo-val-env-kwargs-json", "--ppo-eval-env-kwargs-json", dest="ppo_eval_env_kwargs_json", type=str, default="")
    p.add_argument("--ppo-vecnorm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ppo-vecnorm-norm-obs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-vecnorm-norm-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--ppo-vecnorm-val-norm-reward",
        "--ppo-vecnorm-eval-norm-reward",
        dest="ppo_vecnorm_eval_norm_reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--ppo-vecnorm-clip-obs", type=float, default=10.0)
    p.add_argument("--ppo-vecnorm-clip-reward", type=float, default=10.0)
    p.add_argument("--ppo-vecnorm-epsilon", type=float, default=1e-8)
    p.add_argument("--ppo-vecnorm-gamma", type=float, default=None)
    p.add_argument("--ppo-log-interval-iters", type=int, default=1)
    p.add_argument("--use-action-mask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--model-class", type=str, default="")
    p.add_argument("--model-config-file", type=Path, default=None)
    p.add_argument("--model-config-json", type=str, default="")
    p.add_argument(
        "--ppo-init-model",
        type=Path,
        default=None,
        help="optional PPO init checkpoint path (takes precedence over bc_init.pt)",
    )
    p.add_argument("--skip-ppo", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-env-kwargs-json", type=str, default="")
    p.add_argument("--skip-last-eval", action=argparse.BooleanOptionalAction, default=False)
    # eval seed band (AHC061Local-v0): PPO val + final eval
    p.add_argument("--eval-casegen-num-cases", type=int, default=0)
    p.add_argument("--eval-casegen-seed-mode", choices=["sequential", "random"], default="sequential")
    p.add_argument("--eval-casegen-seed-start", type=int, default=0)
    p.add_argument("--eval-casegen-rng-seed", type=int, default=1)
    p.add_argument("--eval-casegen-unique-random", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-casegen-fixed-m", type=int, default=0)
    p.add_argument("--eval-casegen-fixed-u", type=int, default=0)
    p.add_argument("--eval-casegen-tools-dir", type=Path, default=Path("tools"))

    # training seed band (AHC061Local-v0): collect/bc/ppo environments
    p.add_argument("--casegen-enable", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--casegen-num-cases", type=int, default=0)
    p.add_argument("--casegen-seed-mode", choices=["sequential", "random"], default="sequential")
    p.add_argument("--casegen-seed-start", type=int, default=0)
    p.add_argument("--casegen-rng-seed", type=int, default=1)
    p.add_argument("--casegen-unique-random", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--casegen-fixed-m", type=int, default=0)
    p.add_argument("--casegen-fixed-u", type=int, default=0)
    p.add_argument("--casegen-tools-dir", type=Path, default=Path("tools"))
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True, help="enable TensorBoard logging on PPO stage")
    p.add_argument("--mlflow-tracking-uri", type=str, default="")
    p.add_argument("--mlflow-experiment", type=str, default="ppo_discrete")
    p.add_argument("--mlflow-run-name", type=str, default="")
    return p


def _default_cli_config(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults = vars(parser.parse_args([])).copy()
    for key in ("config_file", "config_section", "set"):
        defaults.pop(key, None)
    return defaults


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    pre, _unknown = parser.parse_known_args()

    base_defaults = _default_cli_config(parser)
    cfg = resolve_config(
        defaults=base_defaults,
        config_file=pre.config_file,
        config_section=pre.config_section,
        overrides=list(pre.set or []),
    )
    unknown = sorted(k for k in cfg.keys() if k not in base_defaults.keys())
    if unknown:
        raise ValueError(f"unknown config keys for run_pipeline: {', '.join(unknown)}")

    parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.model_config_file = coerce_optional_path(args.model_config_file, dot_is_none=True)
    args.ppo_init_model = coerce_optional_path(args.ppo_init_model, dot_is_none=True)
    args.casegen_tools_dir = coerce_optional_path(args.casegen_tools_dir) or Path("tools")
    args.eval_casegen_tools_dir = coerce_optional_path(args.eval_casegen_tools_dir) or Path("tools")
    return args


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
    for name in ("torch", "torchvision", "torchaudio", "numpy", "gymnasium", "pybind11", "mlflow", "tensorboard", "nncv", "mmcv", "mmengine"):
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


def _append_model_args(cmd: list[str], args: argparse.Namespace) -> None:
    model_class = str(args.model_class).strip()
    model_config_file = str(args.model_config_file).strip() if args.model_config_file is not None else ""
    model_config_json = str(args.model_config_json).strip()

    if model_class:
        cmd += ["--model-class", model_class]
    if model_config_file:
        cmd += ["--model-config-file", model_config_file]
    if model_config_json:
        cmd += ["--model-config-json", model_config_json]


def _append_bool_flag(cmd: list[str], name: str, value: bool) -> None:
    cmd.append(f"--{name}" if value else f"--no-{name}")


def _parse_env_kwargs(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--env-kwargs-json must be a JSON object")
    return obj


def _validate_backend_combinations(args: argparse.Namespace) -> None:
    env_id = str(args.env_id).strip()
    is_ahc = env_id == "AHC061Local-v0"
    collect_backend = str(args.collect_rollout_backend).strip().lower()
    ppo_backend = str(args.ppo_rollout_backend).strip().lower()
    collect_policy = str(args.collect_policy).strip()

    if not bool(args.skip_collect):
        if collect_backend == "native":
            if not is_ahc:
                raise ValueError(
                    "collect_rollout_backend=native supports only env_id=AHC061Local-v0 "
                    f"(got env_id={env_id!r})"
                )
            if str(args.collect_vector_env) != "sync":
                raise ValueError("collect_rollout_backend=native requires collect_vector_env=sync")
            if collect_policy == "ahc061_main_greedy":
                raise ValueError(
                    "collect_rollout_backend=native does not support collect_policy=ahc061_main_greedy"
                )
            if collect_policy in ("model_stochastic", "model_greedy"):
                raise ValueError(
                    "run_pipeline collect stage does not accept model_* policy yet "
                    "(model path is not configurable in pipeline collect). "
                    "Use collect_policy=random with native backend, or run collect_teacher directly."
                )
            if bool(args.collect_native_save_aux_targets) and str(args.collect_vector_env).strip().lower() != "sync":
                raise ValueError("collect_native_save_aux_targets requires collect_vector_env=sync")
        else:
            if bool(args.collect_native_save_aux_targets):
                raise ValueError("collect_native_save_aux_targets is supported only when collect_rollout_backend=native")
            if collect_policy == "ahc061_main_greedy":
                if not is_ahc:
                    raise ValueError("collect_policy=ahc061_main_greedy requires env_id=AHC061Local-v0")
                if str(args.collect_vector_env) != "sync":
                    raise ValueError("collect_policy=ahc061_main_greedy requires collect_vector_env=sync")
        if float(args.bc_aux_opp_param_loss_coef) > 0.0:
            if collect_backend != "native" or not bool(args.collect_native_save_aux_targets):
                raise ValueError(
                    "bc_aux_opp_param_loss_coef > 0 requires collect_rollout_backend=native "
                    "and collect_native_save_aux_targets=true when collect stage is active"
                )

    if ppo_backend == "native":
        if not is_ahc and (not bool(args.skip_ppo) or not bool(args.skip_last_eval)):
            raise ValueError(
                "ppo_rollout_backend=native supports only env_id=AHC061Local-v0 "
                f"for active ppo/eval stages (got env_id={env_id!r})"
            )
        if not bool(args.skip_ppo) and str(args.ppo_native_distributed).strip().lower() == "on":
            raise ValueError(
                "run_pipeline does not launch torchrun. "
                "Use ppo_native_distributed=auto|off in run_pipeline, or launch train_ppo with torchrun directly."
            )
    if not bool(args.skip_ppo) and float(args.ppo_aux_opp_param_loss_coef) > 0.0 and ppo_backend != "native":
        raise ValueError("ppo_aux_opp_param_loss_coef > 0 requires ppo_rollout_backend=native")

    eval_casegen_requested = int(args.eval_casegen_num_cases) > 0
    if eval_casegen_requested:
        gym_final_eval = (not bool(args.skip_last_eval)) and ppo_backend == "gym"
        gym_periodic_val = (
            (not bool(args.skip_ppo))
            and int(args.ppo_eval_interval_steps) > 0
            and ppo_backend == "gym"
        )
        if not (gym_final_eval or gym_periodic_val):
            raise ValueError(
                "eval_casegen_* is configured but there is no gym-based eval consumer "
                "(final eval / periodic val are native or skipped). "
                "Set ppo_rollout_backend=gym, or disable eval_casegen_num_cases."
            )


def _ensure_gen_one_binary(tools_dir: Path) -> str:
    tools_dir = Path(tools_dir)
    cargo_toml = tools_dir / "Cargo.toml"
    if not cargo_toml.exists():
        raise FileNotFoundError(f"not found: {cargo_toml}")
    gen_one = tools_dir / "target" / "release" / "gen_one"
    if gen_one.exists():
        return str(gen_one)
    cargo_bin = shutil.which("cargo")
    if cargo_bin is None:
        raise FileNotFoundError("cargo was not found and tools/target/release/gen_one is missing")
    subprocess.run(
        [cargo_bin, "build", "-r", "--manifest-path", str(cargo_toml), "--bin", "gen_one"],
        check=True,
    )
    if not gen_one.exists():
        raise FileNotFoundError(f"failed to build gen_one: {gen_one}")
    return str(gen_one)


def _normalize_bayes_backend_name(x: Any) -> str:
    b = str(x).strip().lower()
    if not b:
        b = "auto"
    if b not in ("auto", "python", "cpp"):
        raise ValueError(f"unsupported bayes_backend={x!r}; expected auto|python|cpp")
    return b


def _maybe_prepare_cpp_bayes_backend(*, env_id: str, env_kwargs: dict[str, Any], eval_env_kwargs: dict[str, Any]) -> dict[str, Any]:
    if str(env_id) != "AHC061Local-v0":
        return {"enabled": False, "prepared": False, "train_backend": "", "eval_backend": ""}

    train_backend = _normalize_bayes_backend_name(env_kwargs.get("bayes_backend", "auto"))
    eval_backend = _normalize_bayes_backend_name(eval_env_kwargs.get("bayes_backend", train_backend))
    needs_cpp = bool(train_backend in ("auto", "cpp") or eval_backend in ("auto", "cpp"))

    meta = {
        "enabled": needs_cpp,
        "prepared": False,
        "train_backend": train_backend,
        "eval_backend": eval_backend,
    }
    if not needs_cpp:
        return meta

    ok = ensure_cpp_bayes_backend(build_if_missing=True, force_build=False, verbose=False)
    if not ok and (train_backend == "cpp" or eval_backend == "cpp"):
        raise RuntimeError("bayes_backend=cpp was requested but cpp backend build/import failed")
    if not ok:
        logger.warning("cpp bayes backend was unavailable; auto backend will fallback to python")
    meta["prepared"] = bool(ok)
    return meta


def _case_seed_meta(
    *,
    enabled: bool,
    num_cases: int,
    seed_mode: str,
    seed_start: int,
    rng_seed: int,
    unique_random: bool,
    fixed_m: int,
    fixed_u: int,
    tools_dir: Path,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "num_cases": int(num_cases),
        "seed_mode": str(seed_mode),
        "seed_start": int(seed_start),
        "rng_seed": int(rng_seed),
        "unique_random": bool(unique_random),
        "fixed_m": int(fixed_m),
        "fixed_u": int(fixed_u),
        "tools_dir": str(tools_dir),
    }


def _apply_case_seed_kwargs(
    dst: dict[str, Any],
    *,
    num_cases: int,
    seed_mode: str,
    seed_start: int,
    rng_seed: int,
    unique_random: bool,
    fixed_m: int,
    fixed_u: int,
    tools_dir: Path,
) -> None:
    if int(num_cases) <= 0:
        raise ValueError("case seed band requires num_cases > 0")
    dst["case_num_cases"] = int(num_cases)
    dst["case_seed_mode"] = str(seed_mode)
    dst["case_seed_start"] = int(seed_start)
    dst["case_rng_seed"] = int(rng_seed)
    dst["case_unique_random"] = bool(unique_random)
    dst["case_tools_dir"] = str(tools_dir)
    if int(fixed_m) > 0:
        dst["case_fixed_m"] = int(fixed_m)
    else:
        dst.pop("case_fixed_m", None)
    if int(fixed_u) > 0:
        dst["case_fixed_u"] = int(fixed_u)
    else:
        dst.pop("case_fixed_u", None)


def _split_counts(total: int, workers: int) -> list[int]:
    workers = max(1, int(workers))
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _run_collect_workers(
    *,
    py: str,
    args: argparse.Namespace,
    collect_policy: str,
    env_kwargs: dict[str, Any],
    output_npz: Path,
    tracker: MetricTracker | None,
    merge_output: bool = True,
    cleanup_shards: bool = True,
    tee_fp: TextIO | None = None,
) -> dict[str, Any]:
    workers = int(args.collect_workers)
    collect_backend = str(args.collect_rollout_backend).strip().lower()
    env_kwargs_cmd = dict(env_kwargs) if collect_backend == "gym" else {}
    if workers <= 1:
        cmd = [
            py,
            "-m",
            "reinforce.ppo_discrete.cli.collect_teacher",
            "--env-id",
            args.env_id,
            "--episodes",
            str(args.collect_episodes),
            "--seed",
            str(args.seed),
            "--policy",
            collect_policy,
            "--vector-env",
            args.collect_vector_env,
            "--rollout-backend",
            collect_backend,
            "--chunk-episodes",
            str(args.collect_chunk_episodes),
            "--output-npz",
            str(output_npz),
            "--env-kwargs-json",
            json.dumps(env_kwargs_cmd),
        ]
        if collect_backend == "native":
            cmd += [
                "--native-feature-id",
                str(args.collect_native_feature_id),
            ]
            _append_bool_flag(cmd, "native-pf-enabled", bool(args.collect_native_pf_enabled))
            _append_bool_flag(cmd, "native-amp", bool(args.collect_native_amp))
            _append_bool_flag(cmd, "native-save-aux-targets", bool(args.collect_native_save_aux_targets))
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
        if str(args.env_id) == "AHC061Local-v0" and collect_backend == "gym":
            env_kwargs_i["vector_env_rank"] = int(wi)
            env_kwargs_i["vector_env_size"] = int(len(shard_eps))
        cmd = [
            py,
            "-m",
            "reinforce.ppo_discrete.cli.collect_teacher",
            "--env-id",
            args.env_id,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed_i),
            "--policy",
            collect_policy,
            "--vector-env",
            args.collect_vector_env,
            "--rollout-backend",
            collect_backend,
            "--chunk-episodes",
            str(args.collect_chunk_episodes),
            "--output-npz",
            str(shard_path),
            "--env-kwargs-json",
            json.dumps(env_kwargs_i),
        ]
        if collect_backend == "native":
            cmd += [
                "--native-feature-id",
                str(args.collect_native_feature_id),
            ]
            _append_bool_flag(cmd, "native-pf-enabled", bool(args.collect_native_pf_enabled))
            _append_bool_flag(cmd, "native-amp", bool(args.collect_native_amp))
            _append_bool_flag(cmd, "native-save-aux-targets", bool(args.collect_native_save_aux_targets))
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
        _merge_teacher_shards(shard_paths, output_npz)
        if cleanup_shards:
            _cleanup_collect_shards(shard_paths, shards_dir=shards_dir)
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


def _merge_teacher_shards(shard_paths: list[Path], output_npz: Path) -> None:
    obs_shape_ref: np.ndarray | None = None
    action_dim_ref: np.ndarray | None = None
    obs_tail_shape: tuple[int, ...] | None = None
    bayes_tail_shape: tuple[int, ...] | None = None
    has_native_aux: bool | None = None
    opp_param_tail_shape: tuple[int, ...] | None = None
    opp_valid_tail_shape: tuple[int, ...] | None = None
    total = 0

    for shard in shard_paths:
        with np.load(shard) as d:
            obs_shape = np.asarray(d["obs_shape"], dtype=np.int32)
            action_dim = np.asarray(d["action_dim"], dtype=np.int32)
            shard_has_native_aux = (DATASET_KEY_OPP_PARAM_TRUE in d.files) and (DATASET_KEY_OPP_VALID in d.files)
            if has_native_aux is None:
                has_native_aux = bool(shard_has_native_aux)
            elif bool(has_native_aux) != bool(shard_has_native_aux):
                raise ValueError(f"native aux key presence mismatch in shard: {shard}")
            if obs_shape_ref is None:
                obs_shape_ref = obs_shape
                action_dim_ref = action_dim
                obs_tail_shape = tuple(np.asarray(d["obs"]).shape[1:])
                bayes_tail_shape = tuple(np.asarray(d["bayes_params"]).shape[1:])
                if bool(shard_has_native_aux):
                    opp_param_tail_shape = tuple(np.asarray(d[DATASET_KEY_OPP_PARAM_TRUE]).shape[1:])
                    opp_valid_tail_shape = tuple(np.asarray(d[DATASET_KEY_OPP_VALID]).shape[1:])
            else:
                if not np.array_equal(obs_shape_ref, obs_shape):
                    raise ValueError(f"obs_shape mismatch in shard: {shard}")
                if not np.array_equal(action_dim_ref, action_dim):
                    raise ValueError(f"action_dim mismatch in shard: {shard}")
                if bool(shard_has_native_aux):
                    p_shape = tuple(np.asarray(d[DATASET_KEY_OPP_PARAM_TRUE]).shape[1:])
                    v_shape = tuple(np.asarray(d[DATASET_KEY_OPP_VALID]).shape[1:])
                    if opp_param_tail_shape != p_shape:
                        raise ValueError(f"{DATASET_KEY_OPP_PARAM_TRUE} shape mismatch in shard: {shard}")
                    if opp_valid_tail_shape != v_shape:
                        raise ValueError(f"{DATASET_KEY_OPP_VALID} shape mismatch in shard: {shard}")
            total += int(np.asarray(d["action"]).shape[0])

    if obs_shape_ref is None or action_dim_ref is None:
        raise ValueError("no valid shards to merge")
    if has_native_aux is None:
        has_native_aux = False

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if obs_tail_shape is None:
        obs_tail_shape = tuple(int(x) for x in obs_shape_ref.tolist())
    if bayes_tail_shape is None:
        bayes_tail_shape = AHC061_BAYES_TAIL_SHAPE

    if total <= 0:
        payload: dict[str, Any] = {
            "obs": np.zeros((0, *obs_tail_shape), dtype=np.float32),
            "action": np.zeros((0,), dtype=np.int64),
            "reward": np.zeros((0,), dtype=np.float32),
            "done": np.zeros((0,), dtype=np.uint8),
            "episode": np.zeros((0,), dtype=np.int32),
            "step": np.zeros((0,), dtype=np.int32),
            "bayes_params": np.zeros((0, *bayes_tail_shape), dtype=np.float32),
            "obs_shape": obs_shape_ref,
            "action_dim": action_dim_ref,
        }
        if bool(has_native_aux):
            payload[DATASET_KEY_OPP_PARAM_TRUE] = np.zeros(
                (0, *(opp_param_tail_shape or AHC061_OPP_PARAM_TRUE_TAIL_SHAPE)),
                dtype=np.float32,
            )
            payload[DATASET_KEY_OPP_VALID] = np.zeros(
                (0, *(opp_valid_tail_shape or AHC061_OPP_VALID_TAIL_SHAPE)),
                dtype=np.uint8,
            )
        np.savez_compressed(output_npz, **payload)
        return

    with TemporaryDirectory(prefix=f".{output_npz.stem}.merge_", dir=str(output_npz.parent)) as tmp_dir:
        tmp = Path(tmp_dir)
        obs_mm = np.lib.format.open_memmap(tmp / "obs.npy", mode="w+", dtype=np.float32, shape=(total, *obs_tail_shape))
        action_mm = np.lib.format.open_memmap(tmp / "action.npy", mode="w+", dtype=np.int64, shape=(total,))
        reward_mm = np.lib.format.open_memmap(tmp / "reward.npy", mode="w+", dtype=np.float32, shape=(total,))
        done_mm = np.lib.format.open_memmap(tmp / "done.npy", mode="w+", dtype=np.uint8, shape=(total,))
        episode_mm = np.lib.format.open_memmap(tmp / "episode.npy", mode="w+", dtype=np.int32, shape=(total,))
        step_mm = np.lib.format.open_memmap(tmp / "step.npy", mode="w+", dtype=np.int32, shape=(total,))
        bayes_mm = np.lib.format.open_memmap(
            tmp / "bayes_params.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total, *bayes_tail_shape),
        )
        opp_param_mm = None
        opp_valid_mm = None
        if bool(has_native_aux):
            opp_param_mm = np.lib.format.open_memmap(
                tmp / f"{DATASET_KEY_OPP_PARAM_TRUE}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total, *(opp_param_tail_shape or AHC061_OPP_PARAM_TRUE_TAIL_SHAPE)),
            )
            opp_valid_mm = np.lib.format.open_memmap(
                tmp / f"{DATASET_KEY_OPP_VALID}.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(total, *(opp_valid_tail_shape or AHC061_OPP_VALID_TAIL_SHAPE)),
            )

        cursor = 0
        episode_offset = 0
        for shard in shard_paths:
            with np.load(shard) as d:
                n = int(np.asarray(d["action"]).shape[0])
                if n <= 0:
                    continue
                sl = slice(cursor, cursor + n)
                obs_mm[sl] = np.asarray(d["obs"], dtype=np.float32)
                action_mm[sl] = np.asarray(d["action"], dtype=np.int64)
                reward_mm[sl] = np.asarray(d["reward"], dtype=np.float32)
                done_mm[sl] = np.asarray(d["done"], dtype=np.uint8)
                step_mm[sl] = np.asarray(d["step"], dtype=np.int32)
                bayes_mm[sl] = np.asarray(d["bayes_params"], dtype=np.float32)
                if bool(has_native_aux):
                    assert opp_param_mm is not None
                    assert opp_valid_mm is not None
                    opp_param_mm[sl] = np.asarray(d[DATASET_KEY_OPP_PARAM_TRUE], dtype=np.float32)
                    opp_valid_mm[sl] = np.asarray(d[DATASET_KEY_OPP_VALID], dtype=np.uint8)

                ep = np.asarray(d["episode"], dtype=np.int32)
                if ep.size > 0:
                    ep = ep + episode_offset
                    episode_offset = int(ep.max()) + 1
                episode_mm[sl] = ep
                cursor += n

        payload: dict[str, Any] = {
            "obs": obs_mm,
            "action": action_mm,
            "reward": reward_mm,
            "done": done_mm,
            "episode": episode_mm,
            "step": step_mm,
            "bayes_params": bayes_mm,
            "obs_shape": obs_shape_ref,
            "action_dim": action_dim_ref,
        }
        if bool(has_native_aux):
            assert opp_param_mm is not None
            assert opp_valid_mm is not None
            payload[DATASET_KEY_OPP_PARAM_TRUE] = opp_param_mm
            payload[DATASET_KEY_OPP_VALID] = opp_valid_mm
        np.savez_compressed(output_npz, **payload)


def _cleanup_collect_shards(shard_paths: list[Path], *, shards_dir: Path | None = None) -> None:
    for shard in shard_paths:
        try:
            if shard.exists():
                shard.unlink()
        except Exception as e:
            logger.warning("failed to remove shard file: %s (%s)", shard, e)
        meta = shard.with_suffix(shard.suffix + ".meta.json")
        try:
            if meta.exists():
                meta.unlink()
        except Exception as e:
            logger.warning("failed to remove shard meta: %s (%s)", meta, e)

    if shards_dir is not None:
        try:
            if shards_dir.exists() and not any(shards_dir.iterdir()):
                shards_dir.rmdir()
        except Exception as e:
            logger.warning("failed to clean shard directory: %s (%s)", shards_dir, e)


def main() -> int:
    args = parse_args()
    _validate_backend_combinations(args)
    eval_casegen_requested = int(args.eval_casegen_num_cases) > 0
    eval_casegen_active = bool(eval_casegen_requested and (not args.skip_last_eval or int(args.ppo_eval_interval_steps) > 0))
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

    casegen_train_meta = _case_seed_meta(
        enabled=bool(args.casegen_enable),
        num_cases=int(args.casegen_num_cases),
        seed_mode=args.casegen_seed_mode,
        seed_start=int(args.casegen_seed_start),
        rng_seed=int(args.casegen_rng_seed),
        unique_random=bool(args.casegen_unique_random),
        fixed_m=int(args.casegen_fixed_m),
        fixed_u=int(args.casegen_fixed_u),
        tools_dir=Path(args.casegen_tools_dir),
    )
    casegen_eval_meta = _case_seed_meta(
        enabled=bool(eval_casegen_active),
        num_cases=int(args.eval_casegen_num_cases),
        seed_mode=args.eval_casegen_seed_mode,
        seed_start=int(args.eval_casegen_seed_start),
        rng_seed=int(args.eval_casegen_rng_seed),
        unique_random=bool(args.eval_casegen_unique_random),
        fixed_m=int(args.eval_casegen_fixed_m),
        fixed_u=int(args.eval_casegen_fixed_u),
        tools_dir=Path(args.eval_casegen_tools_dir),
    )

    train_gen_cmd: str | None = None
    if str(args.env_id) == "AHC061Local-v0":
        collect_uses_gym = (not bool(args.skip_collect)) and str(args.collect_rollout_backend).strip().lower() == "gym"
        ppo_uses_gym = (not bool(args.skip_ppo)) and str(args.ppo_rollout_backend).strip().lower() == "gym"
        final_eval_uses_gym = (not bool(args.skip_last_eval)) and str(args.ppo_rollout_backend).strip().lower() == "gym"
        needs_train_casegen = bool(collect_uses_gym or ppo_uses_gym or final_eval_uses_gym)

        if needs_train_casegen and not bool(args.casegen_enable):
            raise ValueError(
                "AHC061Local-v0 with gym-based stages requires casegen_enable=true with casegen_num_cases > 0"
            )
        if bool(args.casegen_enable):
            train_gen_cmd = _ensure_gen_one_binary(Path(args.casegen_tools_dir))
            _apply_case_seed_kwargs(
                env_kwargs,
                num_cases=int(args.casegen_num_cases),
                seed_mode=args.casegen_seed_mode,
                seed_start=int(args.casegen_seed_start),
                rng_seed=int(args.casegen_rng_seed),
                unique_random=bool(args.casegen_unique_random),
                fixed_m=int(args.casegen_fixed_m),
                fixed_u=int(args.casegen_fixed_u),
                tools_dir=Path(args.casegen_tools_dir),
            )
            env_kwargs["case_gen_cmd"] = str(train_gen_cmd)
        if not str(args.eval_env_kwargs_json).strip() and bool(args.casegen_enable):
            _apply_case_seed_kwargs(
                eval_env_kwargs,
                num_cases=int(args.casegen_num_cases),
                seed_mode=args.casegen_seed_mode,
                seed_start=int(args.casegen_seed_start),
                rng_seed=int(args.casegen_rng_seed),
                unique_random=bool(args.casegen_unique_random),
                fixed_m=int(args.casegen_fixed_m),
                fixed_u=int(args.casegen_fixed_u),
                tools_dir=Path(args.casegen_tools_dir),
            )
            if train_gen_cmd is not None:
                eval_env_kwargs["case_gen_cmd"] = str(train_gen_cmd)
        if eval_casegen_active:
            eval_tools_dir = Path(args.eval_casegen_tools_dir)
            eval_gen_cmd = (
                train_gen_cmd
                if (train_gen_cmd is not None and eval_tools_dir.resolve() == Path(args.casegen_tools_dir).resolve())
                else _ensure_gen_one_binary(eval_tools_dir)
            )
            _apply_case_seed_kwargs(
                eval_env_kwargs,
                num_cases=int(args.eval_casegen_num_cases),
                seed_mode=args.eval_casegen_seed_mode,
                seed_start=int(args.eval_casegen_seed_start),
                rng_seed=int(args.eval_casegen_rng_seed),
                unique_random=bool(args.eval_casegen_unique_random),
                fixed_m=int(args.eval_casegen_fixed_m),
                fixed_u=int(args.eval_casegen_fixed_u),
                tools_dir=eval_tools_dir,
            )
            eval_env_kwargs["case_gen_cmd"] = str(eval_gen_cmd)
    elif bool(args.casegen_enable or eval_casegen_active):
        logger.info("case seed band config is ignored for env_id=%s", args.env_id)

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
            "casegen": {
                "train": casegen_train_meta,
                "eval": casegen_eval_meta,
            },
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
        enable_tensorboard=False,
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
        "casegen": {
            "train": {"enabled": bool(casegen_train_meta["enabled"]), "meta": casegen_train_meta},
            "eval": {
                "enabled": bool(casegen_eval_meta["enabled"]),
                "meta": casegen_eval_meta,
            },
        },
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
                if (
                    args.env_id == "AHC061Local-v0"
                    and collect_policy == "random"
                    and str(args.collect_rollout_backend).strip().lower() == "gym"
                ):
                    collect_policy = "ahc061_main_greedy"
                    logger.info("collect_policy auto-set to ahc061_main_greedy for AHC061Local-v0")
                if collect_policy == "ahc061_main_greedy" and args.collect_vector_env != "sync":
                    raise ValueError("collect_policy=ahc061_main_greedy requires --collect-vector-env sync")
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
                stage_result["collect"]["rollout_backend"] = str(args.collect_rollout_backend)
                if str(args.collect_rollout_backend).strip().lower() == "native":
                    stage_result["collect"]["native_feature_id"] = str(args.collect_native_feature_id)
                    stage_result["collect"]["native_pf_enabled"] = bool(args.collect_native_pf_enabled)
                    stage_result["collect"]["native_save_aux_targets"] = bool(args.collect_native_save_aux_targets)
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

                bc_cmd = [
                    py,
                    "-m",
                    "reinforce.ppo_discrete.cli.train_bc",
                    "--output-model",
                    str(bc_model),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.bc_epochs),
                    "--aux-opp-param-loss-coef",
                    str(args.bc_aux_opp_param_loss_coef),
                ]
                _append_bool_flag(bc_cmd, "aux-opp-param-use-valid-mask", bool(args.bc_aux_opp_param_use_valid_mask))
                if use_shards_for_bc:
                    bc_cmd += ["--dataset-shards-glob", collect_shards_glob]
                    stage_result["bc"]["dataset"] = {"mode": "shards", "glob": collect_shards_glob}
                else:
                    if not teacher_npz.exists():
                        raise FileNotFoundError(
                            f"teacher dataset was not found: {teacher_npz} (and no collect shards matched {collect_shards_glob})"
                        )
                    bc_cmd += ["--dataset-npz", str(teacher_npz)]
                    stage_result["bc"]["dataset"] = {"mode": "npz", "path": str(teacher_npz)}
                _append_model_args(bc_cmd, args)
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

                cmd = [
                    py,
                    "-m",
                    "reinforce.ppo_discrete.cli.train_ppo",
                    "--env-id",
                    args.env_id,
                    "--run-dir",
                    str(ppo_root),
                    "--seed",
                    str(args.seed),
                    "--total-timesteps",
                    str(args.ppo_total_timesteps),
                    "--num-envs",
                    str(args.ppo_num_envs),
                    "--num-steps",
                    str(args.ppo_num_steps),
                    "--learning-rate",
                    str(args.ppo_learning_rate),
                    "--gamma",
                    str(args.ppo_gamma),
                    "--gae-lambda",
                    str(args.ppo_gae_lambda),
                    "--num-minibatches",
                    str(args.ppo_num_minibatches),
                    "--update-epochs",
                    str(args.ppo_update_epochs),
                    "--clip-coef",
                    str(args.ppo_clip_coef),
                    "--clip-coef-schedule",
                    str(args.ppo_clip_coef_schedule),
                    "--ent-coef",
                    str(args.ppo_ent_coef),
                    "--ent-coef-schedule",
                    str(args.ppo_ent_coef_schedule),
                    "--vf-coef",
                    str(args.ppo_vf_coef),
                    "--aux-opp-param-loss-coef",
                    str(args.ppo_aux_opp_param_loss_coef),
                    "--max-grad-norm",
                    str(args.ppo_max_grad_norm),
                    "--checkpoint-interval-steps",
                    str(args.ppo_checkpoint_interval_steps),
                    "--vector-env",
                    args.ppo_vector_env,
                    "--rollout-backend",
                    str(args.ppo_rollout_backend),
                    "--val-interval-steps",
                    str(args.ppo_eval_interval_steps),
                    "--val-episodes",
                    str(args.ppo_eval_episodes),
                    "--val-seed-start",
                    str(args.ppo_eval_seed_start),
                    "--val-vector-env",
                    args.ppo_eval_vector_env,
                    "--log-interval-iters",
                    str(args.ppo_log_interval_iters),
                    "--vecnorm-clip-obs",
                    str(args.ppo_vecnorm_clip_obs),
                    "--vecnorm-clip-reward",
                    str(args.ppo_vecnorm_clip_reward),
                    "--vecnorm-epsilon",
                    str(args.ppo_vecnorm_epsilon),
                    "--env-kwargs-json",
                    json.dumps(env_kwargs),
                ]
                _append_model_args(cmd, args)
                _append_bool_flag(cmd, "norm-adv", bool(args.ppo_norm_adv))
                _append_bool_flag(cmd, "clip-vloss", bool(args.ppo_clip_vloss))
                _append_bool_flag(cmd, "val-at-start", bool(args.ppo_eval_at_start))
                _append_bool_flag(cmd, "use-action-mask", bool(args.use_action_mask))
                _append_bool_flag(cmd, "val-fixed-seeds", bool(args.ppo_eval_fixed_seeds))
                _append_bool_flag(cmd, "val-deterministic", bool(args.ppo_eval_deterministic))
                _append_bool_flag(cmd, "vecnorm", bool(args.ppo_vecnorm))
                _append_bool_flag(cmd, "vecnorm-norm-obs", bool(args.ppo_vecnorm_norm_obs))
                _append_bool_flag(cmd, "vecnorm-norm-reward", bool(args.ppo_vecnorm_norm_reward))
                _append_bool_flag(cmd, "vecnorm-val-norm-reward", bool(args.ppo_vecnorm_eval_norm_reward))
                _append_bool_flag(cmd, "aux-opp-param-use-valid-mask", bool(args.ppo_aux_opp_param_use_valid_mask))
                if str(args.ppo_rollout_backend) == "native":
                    cmd += ["--native-feature-id", str(args.ppo_native_feature_id)]
                    _append_bool_flag(cmd, "native-pf-enabled", bool(args.ppo_native_pf_enabled))
                    _append_bool_flag(cmd, "native-amp", bool(args.ppo_native_amp))
                    cmd += ["--native-memory-format", str(args.ppo_native_memory_format)]
                    _append_bool_flag(cmd, "native-pin-memory", bool(args.ppo_native_pin_memory))
                    cmd += ["--native-rollout-cache-device", str(args.ppo_native_rollout_cache_device)]
                    cmd += ["--native-distributed", str(args.ppo_native_distributed)]
                    if str(args.ppo_native_model_preset).strip():
                        cmd += ["--native-model-preset", str(args.ppo_native_model_preset).strip()]
                if str(args.ppo_learning_rate_schedule).strip():
                    cmd += ["--learning-rate-schedule", str(args.ppo_learning_rate_schedule)]
                if args.ppo_clip_range_vf is not None:
                    cmd += ["--clip-range-vf", str(args.ppo_clip_range_vf)]
                if str(args.ppo_clip_range_vf_schedule).strip():
                    cmd += ["--clip-range-vf-schedule", str(args.ppo_clip_range_vf_schedule)]
                if args.ppo_clip_range_vf_final is not None:
                    cmd += ["--clip-range-vf-final", str(args.ppo_clip_range_vf_final)]
                if str(args.ppo_clip_range_vf_schedule_expr).strip():
                    cmd += ["--clip-range-vf-schedule-expr", str(args.ppo_clip_range_vf_schedule_expr)]
                if args.ppo_clip_coef_final is not None:
                    cmd += ["--clip-coef-final", str(args.ppo_clip_coef_final)]
                if str(args.ppo_clip_coef_schedule_expr).strip():
                    cmd += ["--clip-coef-schedule-expr", str(args.ppo_clip_coef_schedule_expr)]
                if args.ppo_ent_coef_final is not None:
                    cmd += ["--ent-coef-final", str(args.ppo_ent_coef_final)]
                if str(args.ppo_ent_coef_schedule_expr).strip():
                    cmd += ["--ent-coef-schedule-expr", str(args.ppo_ent_coef_schedule_expr)]
                if args.ppo_target_kl is not None:
                    cmd += ["--target-kl", str(args.ppo_target_kl)]
                if args.ppo_vecnorm_gamma is not None:
                    cmd += ["--vecnorm-gamma", str(args.ppo_vecnorm_gamma)]
                if ppo_val_env_kwargs_json_resolved:
                    cmd += ["--val-env-kwargs-json", ppo_val_env_kwargs_json_resolved]
                if not args.tensorboard:
                    cmd.append("--no-tensorboard")
                if args.mlflow_tracking_uri:
                    cmd += ["--mlflow-tracking-uri", args.mlflow_tracking_uri]
                    if args.mlflow_experiment:
                        cmd += ["--mlflow-experiment", args.mlflow_experiment]
                    if args.mlflow_run_name:
                        cmd += ["--mlflow-run-name", args.mlflow_run_name]

                if resume_ppo_ckpt is not None and resume_ppo_dir is not None:
                    cmd += [
                        "--run-name",
                        resume_ppo_dir.name,
                        "--resume",
                        "--resume-from",
                        str(resume_ppo_ckpt),
                    ]
                    stage_result["ppo"]["resume_from"] = str(resume_ppo_ckpt)
                elif ppo_init_model is not None:
                    cmd += ["--init-model", str(ppo_init_model)]
                    stage_result["ppo"]["init_model"] = str(ppo_init_model)
                elif bc_model.exists():
                    cmd += ["--init-model", str(bc_model)]
                    stage_result["ppo"]["init_model"] = str(bc_model)
                stage_result["ppo"]["rollout_backend"] = str(args.ppo_rollout_backend)
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
                eval_cmd = [
                    py,
                    "-m",
                    "reinforce.ppo_discrete.cli.eval_policy",
                    "--env-id",
                    args.env_id,
                    "--model-path",
                    str(trained_model),
                    "--episodes",
                    str(args.eval_episodes),
                    "--seed",
                    str(args.seed),
                    "--output-json",
                    str(eval_json),
                    "--deterministic",
                    "--env-kwargs-json",
                    json.dumps(eval_env_kwargs),
                ]
                if str(args.ppo_rollout_backend).strip().lower() == "native":
                    eval_cmd += [
                        "--rollout-backend",
                        "native",
                        "--native-feature-id",
                        str(args.ppo_native_feature_id),
                    ]
                    _append_bool_flag(eval_cmd, "native-pf-enabled", bool(args.ppo_native_pf_enabled))
                _append_bool_flag(eval_cmd, "use-action-mask", bool(args.use_action_mask))
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


if __name__ == "__main__":
    raise SystemExit(main())
