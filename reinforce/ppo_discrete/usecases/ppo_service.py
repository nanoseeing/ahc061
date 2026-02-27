from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..core.ppo.config import PPOConfig
from ..core.ppo.rollout import collect_rollout, create_rollout_workspace
from ..core.ppo.vecnorm import VecNormalize
from ..core.ppo.rollout_buffer import RolloutBuffer
from ..core.ppo.trainer import PPOTrainer
from ..core.ppo.train_utils import (
    PPOScheduleSet,
    resolve_vecnorm_gamma,
    schedule_progress,
    validate_ppo_config,
    validate_vecnorm_config,
)
from ..domain.ahc061.batch_env import BatchEnv, BatchEnvProtocol, ensure_batch_env
from ..domain.ahc061.batch_env.feature_catalog import get_feature_spec
from ..agents import (
    build_agent,
    get_model_config_from_preset,
    get_model_preset,
    load_model_config_from_sources,
    normalize_model_config,
)
from ..usecases.model_checkpoint_service import save_agent_checkpoint
from ..infra.experiment import (
    coerce_optional_path,
    create_run_layout,
    make_run_name,
    to_jsonable,
    update_manifest,
)
from ..infra.log_utils import get_logger
from ..infra.metrics import summarize
from .ppo_requests import PPORequest, TrainPPORequest, args_to_cfg, build_ppo_request
from .eval_service import run_policy_episodes

logger = get_logger("train_ppo")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_model_config(args: PPORequest) -> dict[str, Any]:
    preset_cfg: dict[str, Any] | None = None
    preset_id = str(getattr(args, "model_preset", "")).strip()
    if preset_id:
        preset_cfg = get_model_config_from_preset(preset_id)

    explicit_cfg = load_model_config_from_sources(
        model_config_file=args.model_config_file,
        model_config_json=args.model_config_json,
    )
    base_cfg = explicit_cfg if explicit_cfg is not None else preset_cfg
    model_class = str(args.model_class).strip()
    if base_cfg is not None and model_class:
        base_cfg["type"] = model_class

    return normalize_model_config(
        base_cfg,
        default_type=model_class or "DiscreteBoardAgent",
    )


def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_true_np = y_true.detach().reshape(-1).cpu().numpy()
    y_pred_np = y_pred.detach().reshape(-1).cpu().numpy()
    var_y = float(np.var(y_true_np))
    if var_y <= 1e-12:
        return float("nan")
    return 1.0 - float(np.var(y_true_np - y_pred_np) / var_y)


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum_tensor(x: torch.Tensor) -> torch.Tensor:
    if _dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _all_reduce_max_tensor(x: torch.Tensor) -> torch.Tensor:
    if _dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x


def _resolve_distributed_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in ("auto", "off", "on"):
        raise ValueError(f"unsupported distributed mode={mode!r}; expected auto|off|on")
    return m


def _seed_everything(seed: int, *, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        return getattr(model, "_orig_mod")
    return model


def _resolve_use_channels_last(*, device: torch.device, mode: str) -> bool:
    m = str(mode).strip().lower()
    if m not in ("auto", "nchw", "channels_last"):
        raise ValueError(f"unsupported memory_format={mode!r}; expected auto|nchw|channels_last")
    if device.type != "cuda":
        return False
    if m == "channels_last":
        return True
    if m == "nchw":
        return False
    # auto: keep fast-path enabled on CUDA; caller can force nchw explicitly.
    return True


def _estimate_rollout_cache_nbytes(
    *,
    num_steps: int,
    num_envs: int,
    obs_channels: int,
    board_size: int,
    action_dim: int,
    use_action_mask: bool,
    use_aux_opp_param_targets: bool,
) -> int:
    t = int(num_steps)
    b = int(num_envs)
    n = int(board_size)
    n_obs = t * b * int(obs_channels) * n * n
    n_tb = t * b
    # obs float32; actions int64; logprobs/rewards/values/advantages/returns/dones float32
    total = 0
    total += n_obs * 4
    total += n_tb * 8
    total += n_tb * 4  # logprobs
    total += n_tb * 4  # rewards
    total += n_tb * 4  # dones
    total += n_tb * 4  # values
    total += n_tb * 4  # advantages
    total += n_tb * 4  # returns
    if bool(use_action_mask):
        total += t * b * int(action_dim)  # bool/uint8 action masks
    if bool(use_aux_opp_param_targets):
        total += t * b * (7 * 5) * 4  # aux opp_param_true float32
        total += t * b * 7  # aux opp_valid uint8/bool
    return int(total)


def _choose_rollout_cache_device(*, mode: str, train_device: torch.device, total_bytes: int) -> torch.device:
    m = str(mode).strip().lower()
    if m not in ("auto", "cpu", "gpu"):
        raise ValueError(f"unsupported rollout_cache_device={mode!r}; expected auto|cpu|gpu")
    if train_device.type != "cuda":
        return torch.device("cpu")
    if m == "cpu":
        return torch.device("cpu")
    if m == "gpu":
        return train_device
    # auto
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(train_device)
        # Keep some headroom for model/optimizer/temporary tensors.
        allow_bytes = int(max(0, free_bytes - (2 * (1024**3))))
        if int(total_bytes) <= int(allow_bytes):
            return train_device
        return torch.device("cpu")
    except Exception:
        return train_device


def _load_initial_weights(path: Path, agent: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format (missing model_state_dict): {path}")
    agent.load_state_dict(payload["model_state_dict"], strict=True)
    meta = payload.get("meta")
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=True) + "\n")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _vecnorm_state_or_none(vecnorm: VecNormalize | None) -> dict[str, Any] | None:
    if vecnorm is None:
        return None
    try:
        return vecnorm.state_dict()
    except Exception:
        return None


def _sync_rms_ddp_(
    rms_mean: np.ndarray,
    rms_var: np.ndarray,
    rms_count: float,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not _dist_ready():
        return rms_mean, rms_var, float(rms_count)

    mean_arr = np.asarray(rms_mean, dtype=np.float64)
    var_arr = np.maximum(np.asarray(rms_var, dtype=np.float64), 1e-12)
    cnt = float(max(0.0, float(rms_count)))

    m1 = mean_arr * cnt
    m2 = (var_arr + np.square(mean_arr)) * cnt

    t_count = torch.tensor([cnt], dtype=torch.float64, device=device)
    t_m1 = torch.as_tensor(m1.reshape(-1), dtype=torch.float64, device=device)
    t_m2 = torch.as_tensor(m2.reshape(-1), dtype=torch.float64, device=device)
    dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_m1, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_m2, op=dist.ReduceOp.SUM)

    total = float(t_count.item())
    if total <= 0.0:
        return mean_arr, var_arr, float(max(1e-4, cnt))

    mean_global = (t_m1 / total).cpu().numpy().reshape(mean_arr.shape)
    ex2_global = (t_m2 / total).cpu().numpy().reshape(mean_arr.shape)
    var_global = np.maximum(ex2_global - np.square(mean_global), 1e-12)
    return mean_global, var_global, float(total)


def _sync_vecnorm_ddp_(vecnorm: VecNormalize | None, *, device: torch.device) -> None:
    if vecnorm is None or not _dist_ready():
        return
    m, v, c = _sync_rms_ddp_(
        vecnorm.obs_rms.mean,
        vecnorm.obs_rms.var,
        vecnorm.obs_rms.count,
        device=device,
    )
    vecnorm.obs_rms.mean = np.asarray(m, dtype=np.float64)
    vecnorm.obs_rms.var = np.maximum(np.asarray(v, dtype=np.float64), 1e-12)
    vecnorm.obs_rms.count = float(c)

    m, v, c = _sync_rms_ddp_(
        vecnorm.ret_rms.mean,
        vecnorm.ret_rms.var,
        vecnorm.ret_rms.count,
        device=device,
    )
    vecnorm.ret_rms.mean = np.asarray(m, dtype=np.float64)
    vecnorm.ret_rms.var = np.maximum(np.asarray(v, dtype=np.float64), 1e-12)
    vecnorm.ret_rms.count = float(c)


def _run_periodic_val(
    *,
    env_id: str,
    feature_id: str,
    pf_enabled: bool,
    agent: torch.nn.Module,
    device: torch.device,
    episodes: int,
    seed_start: int,
    deterministic: bool,
    use_action_mask: bool,
    amp: bool,
    vecnorm_state: dict[str, Any] | None = None,
    vecnorm_norm_obs: bool = True,
    vecnorm_norm_reward: bool = False,
    vecnorm_clip_obs: float = 10.0,
    vecnorm_clip_reward: float = 10.0,
    vecnorm_epsilon: float = 1e-8,
    vecnorm_gamma: float = 0.99,
) -> dict[str, Any]:
    stats = run_policy_episodes(
        env_id=str(env_id),
        episodes=int(episodes),
        seed=int(seed_start),
        feature_id=str(feature_id),
        pf_enabled=bool(pf_enabled),
        policy=("model_greedy" if bool(deterministic) else "model_stochastic"),
        agent=agent,
        device=device,
        use_action_mask=bool(use_action_mask),
        amp=bool(amp),
        vecnorm_enabled=bool(vecnorm_state is not None),
        vecnorm_state=vecnorm_state,
        vecnorm_norm_obs=bool(vecnorm_norm_obs),
        vecnorm_norm_reward=bool(vecnorm_norm_reward),
        vecnorm_clip_obs=float(vecnorm_clip_obs),
        vecnorm_clip_reward=float(vecnorm_clip_reward),
        vecnorm_epsilon=float(vecnorm_epsilon),
        vecnorm_gamma=float(vecnorm_gamma),
    )
    return {
        "episodes": int(episodes),
        "return": summarize(stats.episode_returns).as_dict(),
        "illegal_penalty": summarize(stats.episode_illegal_penalties).as_dict(),
        "terminal_score": summarize(stats.episode_terminal_scores).as_dict(),
        "terminal_game_score": summarize(stats.episode_terminal_game_scores).as_dict(),
        "game_score_ratio": summarize(stats.episode_game_score_ratio).as_dict(),
        "game_score_self": summarize(stats.episode_game_score_self).as_dict(),
        "game_score_enemy_max": summarize(stats.episode_game_score_enemy_max).as_dict(),
    }


@dataclass
class _RestoredTrainState:
    init_meta: dict[str, Any]
    resume_meta: dict[str, Any]
    global_step: int
    best_metric_value: float
    best_metric_name: str
    best_metric_source: str
    restored_eval_round: int | None
    restored_next_eval_step: int | None
    restored_next_checkpoint_step: int | None


def _restore_training_state(
    *,
    args: PPORequest,
    resume_enabled: bool,
    resume_from: Path | None,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_shape: tuple[int, ...],
    action_dim: int,
    cfg_global: PPOConfig,
    device: torch.device,
    is_main: bool,
    best_metric_value: float,
    best_metric_name: str,
    best_metric_source: str,
) -> _RestoredTrainState:
    init_meta: dict[str, Any] = {}
    resume_meta: dict[str, Any] = {}
    global_step = 0
    restored_eval_round: int | None = None
    restored_next_eval_step: int | None = None
    restored_next_checkpoint_step: int | None = None

    if resume_enabled:
        assert resume_from is not None
        if args.init_model is not None and is_main:
            logger.info("resume is enabled; --init-model is ignored: %s", args.init_model)
        payload = torch.load(resume_from, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"invalid checkpoint format (missing model_state_dict): {resume_from}")
        ckpt_obs_shape = tuple(payload.get("obs_shape", ()))
        ckpt_action_dim = _safe_int(payload.get("action_dim", -1), -1)
        if ckpt_obs_shape and tuple(ckpt_obs_shape) != tuple(obs_shape):
            raise ValueError(f"obs_shape mismatch: model={ckpt_obs_shape}, env={obs_shape}")
        if ckpt_action_dim > 0 and int(ckpt_action_dim) != int(action_dim):
            raise ValueError(f"action_dim mismatch: model={ckpt_action_dim}, env={action_dim}")
        _unwrap_ddp(agent).load_state_dict(payload["model_state_dict"], strict=True)
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        resume_meta = dict(payload.get("meta", {}))
        global_step = max(0, _safe_int(resume_meta.get("global_step", 0), 0))
        prev_cfg_raw = resume_meta.get("cfg_global")
        if not isinstance(prev_cfg_raw, dict):
            cand = resume_meta.get("cfg")
            prev_cfg_raw = cand if isinstance(cand, dict) else None
        if isinstance(prev_cfg_raw, dict):
            prev_num_envs = _safe_int(prev_cfg_raw.get("num_envs", 0), 0)
            prev_num_steps = _safe_int(prev_cfg_raw.get("num_steps", 0), 0)
            prev_batch_size = _safe_int(prev_cfg_raw.get("batch_size", prev_num_envs * prev_num_steps), 0)
            mismatch: list[str] = []
            if prev_num_envs > 0 and prev_num_envs != int(cfg_global.num_envs):
                mismatch.append(f"num_envs(prev={prev_num_envs}, now={cfg_global.num_envs})")
            if prev_num_steps > 0 and prev_num_steps != int(cfg_global.num_steps):
                mismatch.append(f"num_steps(prev={prev_num_steps}, now={cfg_global.num_steps})")
            if prev_batch_size > 0 and prev_batch_size != int(cfg_global.batch_size):
                mismatch.append(f"batch_size(prev={prev_batch_size}, now={cfg_global.batch_size})")
            if mismatch:
                raise ValueError(
                    "resume config mismatch for rollout geometry: "
                    + ", ".join(mismatch)
                    + ". Resume requires the same num_envs/num_steps as the checkpoint."
                )
        if int(global_step) % int(cfg_global.batch_size) != 0:
            raise ValueError(
                "resume global_step is not aligned to current batch_size: "
                f"step={global_step}, batch_size={cfg_global.batch_size}. "
                "Use the same num_envs/num_steps as the checkpoint, or start a new run."
            )
        try:
            best_metric_value = float(resume_meta.get("best_metric_value", best_metric_value))
        except Exception:
            pass
        if isinstance(resume_meta.get("best_metric_name"), str):
            best_metric_name = str(resume_meta.get("best_metric_name"))
        if isinstance(resume_meta.get("best_metric_source"), str):
            best_metric_source = str(resume_meta.get("best_metric_source"))
        rv = resume_meta.get("eval_round")
        restored_eval_round = _safe_int(rv) if rv is not None else None
        rv = resume_meta.get("next_eval_step")
        restored_next_eval_step = _safe_int(rv) if rv is not None else None
        rv = resume_meta.get("next_checkpoint_step")
        restored_next_checkpoint_step = _safe_int(rv) if rv is not None else None
        if is_main:
            logger.info("loaded resume model: %s", resume_from)
    elif args.init_model is not None:
        init_meta = _load_initial_weights(Path(args.init_model), _unwrap_ddp(agent), device)
        if is_main:
            logger.info("loaded init model: %s", args.init_model)

    return _RestoredTrainState(
        init_meta=init_meta,
        resume_meta=resume_meta,
        global_step=int(global_step),
        best_metric_value=float(best_metric_value),
        best_metric_name=str(best_metric_name),
        best_metric_source=str(best_metric_source),
        restored_eval_round=restored_eval_round,
        restored_next_eval_step=restored_next_eval_step,
        restored_next_checkpoint_step=restored_next_checkpoint_step,
    )


def _resolve_eval_checkpoint_state(
    *,
    args: PPORequest,
    resume_enabled: bool,
    global_step: int,
    best_metric_name: str,
    restored_eval_round: int | None,
    restored_next_eval_step: int | None,
    restored_next_checkpoint_step: int | None,
) -> tuple[bool, bool, str, int, int, int, int]:
    checkpoint_interval_steps = int(max(0, _safe_int(args.checkpoint_interval_steps, 0)))
    periodic_val_enabled = bool(
        _safe_int(getattr(args, "eval_interval_steps", 0), 0) > 0
        and _safe_int(getattr(args, "eval_episodes", 0), 0) > 0
    )
    eval_at_start_enabled = bool(
        getattr(args, "eval_at_start", False) and _safe_int(getattr(args, "eval_episodes", 0), 0) > 0
    )
    if periodic_val_enabled:
        best_metric_name = "periodic_val_mean_terminal_game_score"
    elif best_metric_name == "periodic_val_mean_terminal_game_score":
        best_metric_name = "mean_official_score"

    eval_round = 0
    next_eval_step = int(max(1, _safe_int(getattr(args, "eval_interval_steps", 0), 0)))
    next_checkpoint_step = 0
    if resume_enabled:
        if restored_eval_round is not None and int(restored_eval_round) >= 0:
            eval_round = int(restored_eval_round)
        elif periodic_val_enabled and int(args.eval_interval_steps) > 0:
            eval_round = int(global_step // int(args.eval_interval_steps))
            if bool(eval_at_start_enabled) and global_step > 0:
                eval_round += 1
        if restored_next_eval_step is not None and int(restored_next_eval_step) > 0:
            next_eval_step = int(restored_next_eval_step)
        elif periodic_val_enabled and int(args.eval_interval_steps) > 0:
            interval = int(args.eval_interval_steps)
            next_eval_step = int(((global_step // interval) + 1) * interval)
        if checkpoint_interval_steps > 0:
            if restored_next_checkpoint_step is not None and int(restored_next_checkpoint_step) > 0:
                next_checkpoint_step = int(restored_next_checkpoint_step)
            elif global_step > 0:
                next_checkpoint_step = int(((global_step // checkpoint_interval_steps) + 1) * checkpoint_interval_steps)
            else:
                next_checkpoint_step = int(checkpoint_interval_steps)
    elif checkpoint_interval_steps > 0:
        next_checkpoint_step = int(checkpoint_interval_steps)

    return (
        periodic_val_enabled,
        eval_at_start_enabled,
        str(best_metric_name),
        int(eval_round),
        int(next_eval_step),
        int(checkpoint_interval_steps),
        int(next_checkpoint_step),
    )


def _build_resume_meta_payload(
    *,
    iteration: int,
    run_name: str,
    global_step: int,
    world_size: int,
    feature_id: str,
    pf_enabled: bool,
    obs_shape: tuple[int, ...],
    action_dim: int,
    use_channels_last: bool,
    use_pin_memory: bool,
    cache_device: torch.device,
    cfg_global: PPOConfig,
    cfg_local: PPOConfig,
    best_metric_name: str,
    best_metric_value: float,
    best_metric_source: str,
    eval_round: int,
    next_eval_step: int,
    next_checkpoint_step: int,
    rng: np.random.Generator,
    train_vecnorm: VecNormalize | None,
) -> dict[str, Any]:
    return {
        "kind": "train_ppo",
        "run_name": str(run_name),
        "iteration": int(iteration),
        "global_step": int(global_step),
        "world_size": int(world_size),
        "feature_id": str(feature_id),
        "pf_enabled": bool(pf_enabled),
        "obs_shape": list(obs_shape),
        "action_dim": int(action_dim),
        "memory_format": ("channels_last" if bool(use_channels_last) else "nchw"),
        "pin_memory": bool(use_pin_memory),
        "rollout_cache_device": str(cache_device),
        "cfg_global": asdict(cfg_global),
        "cfg_local": asdict(cfg_local),
        "best_metric_name": str(best_metric_name),
        "best_metric_value": float(best_metric_value),
        "best_metric_source": str(best_metric_source),
        "eval_round": int(eval_round),
        "next_eval_step": int(next_eval_step),
        "next_checkpoint_step": int(next_checkpoint_step),
        "rng_state": to_jsonable(rng.bit_generator.state),
        "vecnormalize_state": _vecnorm_state_or_none(train_vecnorm),
    }


def run_ppo_from_train_request(
    *,
    train_req: TrainPPORequest,
    cfg: PPOConfig,
    device: torch.device,
    env_kwargs: Mapping[str, Any],
) -> int:
    ppo_req = build_ppo_request(
        train_req=train_req,
        cfg=cfg,
        device=device,
        env_kwargs=env_kwargs,
    )
    logger.info(
        "dispatch to cpp backend: feature_id=%s pf_enabled=%s num_envs=%d num_steps=%d",
        ppo_req.feature_id,
        bool(ppo_req.pf_enabled),
        int(cfg.num_envs),
        int(cfg.num_steps),
    )
    return int(run_ppo(ppo_req))


def run_ppo(args: PPORequest) -> int:
    cfg_global = args_to_cfg(args)
    validate_ppo_config(cfg_global)
    schedules = PPOScheduleSet.from_config(cfg_global)
    validate_vecnorm_config(
        enabled=bool(args.vecnorm),
        clip_obs=float(args.vecnorm_clip_obs),
        clip_reward=float(args.vecnorm_clip_reward),
        epsilon=float(args.vecnorm_epsilon),
        vecnorm_gamma=args.vecnorm_gamma,
        ppo_gamma=float(cfg_global.gamma),
    )
    vecnorm_gamma = resolve_vecnorm_gamma(
        vecnorm_gamma=args.vecnorm_gamma,
        ppo_gamma=float(cfg_global.gamma),
    )

    dist_mode = _resolve_distributed_mode(str(args.distributed))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = int(world_size) > 1
    if is_distributed and dist_mode == "off":
        raise RuntimeError("--distributed=off cannot be used with torchrun (WORLD_SIZE>1)")
    if (not is_distributed) and dist_mode == "on":
        raise RuntimeError("--distributed=on requires torchrun (WORLD_SIZE>1)")
    if int(cfg_global.num_envs) % int(world_size) != 0:
        raise RuntimeError(f"--num-envs must be divisible by WORLD_SIZE ({cfg_global.num_envs} vs {world_size})")
    if int(cfg_global.num_minibatches) % int(world_size) != 0:
        raise RuntimeError(
            f"--num-minibatches must be divisible by WORLD_SIZE ({cfg_global.num_minibatches} vs {world_size})"
        )
    local_num_envs = int(cfg_global.num_envs) // int(world_size)
    local_num_minibatches = int(cfg_global.num_minibatches) // int(world_size)
    if local_num_envs <= 0 or local_num_minibatches <= 0:
        raise RuntimeError("local num_envs/num_minibatches must be >= 1")

    local_req = replace(
        args,
        num_envs=int(local_num_envs),
        num_minibatches=int(local_num_minibatches),
    )
    local_cfg = args_to_cfg(local_req)
    if local_cfg.batch_size % local_cfg.num_minibatches != 0:
        raise RuntimeError(
            "local batch_size must be divisible by local num_minibatches: "
            f"batch_size={local_cfg.batch_size}, num_minibatches={local_cfg.num_minibatches}"
        )
    aux_opp_param_loss_coef = float(max(0.0, float(getattr(cfg_global, "aux_opp_param_loss_coef", 0.0))))
    aux_opp_param_use_valid_mask = bool(getattr(cfg_global, "aux_opp_param_use_valid_mask", True))
    aux_opp_param_active = bool(aux_opp_param_loss_coef > 0.0)

    device: torch.device | None = None
    global_step = 0
    is_main = int(rank) == 0
    resume_enabled = bool(args.resume or args.resume_from is not None)
    layout = None
    best_metric_value = float("-inf")
    best_metric_name = "mean_official_score"
    best_metric_source = ""
    eval_round = 0
    next_eval_step = int(max(1, _safe_int(getattr(args, "eval_interval_steps", 0), 0)))
    next_checkpoint_step = 0
    run_name = str(args.run_name).strip()
    resume_from = coerce_optional_path(getattr(args, "resume_from", None), dot_is_none=True)
    summary: dict[str, Any] | None = None
    restored_next_checkpoint_step: int | None = None
    restored_next_eval_step: int | None = None
    restored_eval_round: int | None = None
    train_vecnorm: VecNormalize | None = None
    try:
        if is_distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("distributed training requires CUDA")
            if str(args.device) not in ("auto", "cuda", f"cuda:{local_rank}"):
                raise RuntimeError(
                    f"--device={args.device!r} conflicts with LOCAL_RANK={local_rank}. "
                    "Use --device auto/cuda with torchrun."
                )
            dist.init_process_group(backend="nccl")

        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = choose_device(str(args.device))
        if device.type == "cuda":
            if device.index is None:
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
            else:
                torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        if not run_name and resume_enabled and resume_from is None:
            raise ValueError("--resume requires --run-name or --resume-from for train_ppo")
        if not run_name and resume_from is not None:
            rp = Path(resume_from).resolve()
            if rp.parent.name == "models" and rp.parent.parent.name:
                run_name = rp.parent.parent.name
            else:
                run_name = rp.parent.name
        if not run_name:
            run_name = make_run_name("train_ppo", seed=cfg_global.seed)

        if is_distributed:
            run_name_box = [run_name if is_main else ""]
            dist.broadcast_object_list(run_name_box, src=0)
            run_name = str(run_name_box[0])

        layout = create_run_layout(args.run_dir, run_name)
        if resume_enabled and resume_from is None:
            resume_from = layout.models_dir / "last.pt"
        if resume_enabled and (resume_from is None or not Path(resume_from).exists()):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")

        use_channels_last = _resolve_use_channels_last(device=device, mode=str(args.memory_format))
        use_pin_memory = bool(args.pin_memory and device.type == "cuda")

        seed_base = int(cfg_global.seed) + int(rank) * 1_000_003
        _seed_everything(seed_base, device=device)
        if is_main:
            update_manifest(
                layout,
                {
                    "kind": "train_ppo",
                    "status": "running",
                    "run_name": run_name,
                    "resume": {
                        "enabled": bool(resume_enabled),
                        "resume_from": (str(resume_from) if resume_from is not None else ""),
                    },
                    "args": to_jsonable(vars(args)),
                    "ppo_config": to_jsonable(cfg_global),
                    "ppo_config_local": to_jsonable(local_cfg),
                    "distributed": {
                        "mode": str(dist_mode),
                        "world_size": int(world_size),
                    },
                    "paths": layout.as_dict(),
                },
            )
        if is_distributed:
            dist.barrier()
        if is_main:
            logger.info("[RUN] %s", run_name)
            logger.info("[DEVICE] %s", device)
            logger.info(
                "[DDP] mode=%s world_size=%d local_num_envs=%d local_num_minibatches=%d",
                str(dist_mode),
                int(world_size),
                int(local_num_envs),
                int(local_num_minibatches),
            )

        env_impl = BatchEnv(
            batch_size=int(local_cfg.num_envs),
            feature_id=str(args.feature_id),
            pf_enabled=bool(args.pf_enabled),
            verbose_build=False,
        )
        env: BatchEnvProtocol = ensure_batch_env(env_impl)
        feature_spec = get_feature_spec(str(args.feature_id), verbose_build=False)
        if local_cfg.num_steps > int(env.spec.t_max):
            raise ValueError(
                f"num_steps must be <= env.spec.t_max ({env.spec.t_max}) for cpp batch env; got {local_cfg.num_steps}"
            )

        board_size = int(env.board_size)
        action_dim = int(env.action_dim)
        obs_shape = (int(env.feature_channels), board_size, board_size)
        train_vecnorm = (
            VecNormalize(
                num_envs=int(local_cfg.num_envs),
                obs_shape=tuple(obs_shape),
                norm_obs=bool(args.vecnorm_norm_obs),
                norm_reward=bool(args.vecnorm_norm_reward),
                clip_obs=float(args.vecnorm_clip_obs),
                clip_reward=float(args.vecnorm_clip_reward),
                epsilon=float(args.vecnorm_epsilon),
                gamma=float(vecnorm_gamma),
                training=True,
            )
            if bool(args.vecnorm)
            else None
        )
        model_config = _resolve_model_config(args)
        if aux_opp_param_active:
            kwargs = model_config.get("kwargs")
            if kwargs is None:
                kwargs = {}
                model_config["kwargs"] = kwargs
            if not isinstance(kwargs, dict):
                raise ValueError(f"model_config.kwargs must be dict, got {type(kwargs)!r}")
            kwargs.setdefault("aux_opp_param_head", True)
        preset_id = str(getattr(args, "model_preset", "")).strip()
        if preset_id:
            preset = get_model_preset(preset_id)
            if preset.default_feature_id and str(args.feature_id) != str(preset.default_feature_id):
                logger.warning(
                    "model_preset=%s was tuned for feature_id=%s, but current feature_id=%s",
                    preset_id,
                    str(preset.default_feature_id),
                    str(args.feature_id),
                )

        agent, resolved_model_config = build_agent(
            obs_shape=obs_shape,
            action_dim=action_dim,
            model_config=model_config,
            default_type=str(model_config.get("type", "DiscreteBoardAgent")),
        )
        agent = agent.to(device)
        if use_channels_last and device.type == "cuda":
            agent = agent.to(memory_format=torch.channels_last)
        board_channels = getattr(agent, "board_channels", None)
        if is_main and board_channels is not None and int(board_channels) != int(env.feature_channels):
            logger.warning(
                "board_channels(%d) != env.feature_channels(%d); observation will be split into board/global by flatten order",
                int(board_channels),
                int(env.feature_channels),
            )
        if is_main and int(feature_spec.channels) != int(env.feature_channels):
            logger.warning(
                "feature catalog channels(%d) != env feature_channels(%d) for feature_id=%s",
                int(feature_spec.channels),
                int(env.feature_channels),
                str(args.feature_id),
            )
        if aux_opp_param_active and not callable(getattr(agent, "get_aux_opp_param", None)):
            raise ValueError(
                "aux_opp_param_loss_coef > 0 requires model to implement get_aux_opp_param(obs) -> [B,7,5]"
            )

        if bool(args.compile):
            try:
                agent = torch.compile(agent)
            except Exception as e:  # pragma: no cover
                if is_main:
                    logger.warning("torch.compile failed; continue without compile: %s", e)

        if is_distributed:
            ddp_device_id = int(device.index if device.index is not None else local_rank)
            agent = DDP(
                agent,
                device_ids=[ddp_device_id],
                output_device=ddp_device_id,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        optimizer = torch.optim.Adam(agent.parameters(), lr=float(local_cfg.learning_rate), eps=1e-5)
        restored_state = _restore_training_state(
            args=args,
            resume_enabled=bool(resume_enabled),
            resume_from=resume_from,
            agent=agent,
            optimizer=optimizer,
            obs_shape=tuple(obs_shape),
            action_dim=int(action_dim),
            cfg_global=cfg_global,
            device=device,
            is_main=bool(is_main),
            best_metric_value=float(best_metric_value),
            best_metric_name=str(best_metric_name),
            best_metric_source=str(best_metric_source),
        )
        init_meta = dict(restored_state.init_meta)
        resume_meta = dict(restored_state.resume_meta)
        global_step = int(restored_state.global_step)
        best_metric_value = float(restored_state.best_metric_value)
        best_metric_name = str(restored_state.best_metric_name)
        best_metric_source = str(restored_state.best_metric_source)
        restored_eval_round = restored_state.restored_eval_round
        restored_next_eval_step = restored_state.restored_next_eval_step
        restored_next_checkpoint_step = restored_state.restored_next_checkpoint_step

        incoming_vec_state = None
        incoming_vec_source = ""
        if resume_enabled:
            cand = resume_meta.get("vecnormalize_state")
            if isinstance(cand, dict):
                incoming_vec_state = cand
                incoming_vec_source = "resume"
        else:
            cand = init_meta.get("vecnormalize_state")
            if isinstance(cand, dict):
                incoming_vec_state = cand
                incoming_vec_source = "init_model"

        if train_vecnorm is not None:
            if isinstance(incoming_vec_state, dict):
                train_vecnorm.load_state_dict(incoming_vec_state)
                train_vecnorm.set_training(True)
                if is_main:
                    logger.info("restored vecnormalize state from %s checkpoint", incoming_vec_source)
            elif resume_enabled and is_main:
                logger.warning("vecnorm is enabled but resume checkpoint has no vecnormalize_state")
            if is_main:
                logger.info(
                    "vecnorm: enabled=true norm_obs=%s norm_reward=%s clip_obs=%.4f clip_reward=%.4f eps=%g gamma=%.6f",
                    bool(train_vecnorm.norm_obs),
                    bool(train_vecnorm.norm_reward),
                    float(train_vecnorm.clip_obs),
                    float(train_vecnorm.clip_reward),
                    float(train_vecnorm.epsilon),
                    float(train_vecnorm.gamma),
                )
        else:
            if isinstance(incoming_vec_state, dict) and is_main:
                logger.warning(
                    "checkpoint has vecnormalize_state but vecnorm is disabled; observations/rewards will be unnormalized"
                )
            if is_main:
                logger.info("vecnorm: enabled=false")

        trainer = PPOTrainer(cfg=local_cfg, agent=agent, optimizer=optimizer, use_channels_last=bool(use_channels_last))

        estimated_cache_nbytes = _estimate_rollout_cache_nbytes(
            num_steps=int(local_cfg.num_steps),
            num_envs=int(local_cfg.num_envs),
            obs_channels=int(env.feature_channels),
            board_size=int(env.board_size),
            action_dim=int(action_dim),
            use_action_mask=bool(args.use_action_mask),
            use_aux_opp_param_targets=bool(aux_opp_param_active),
        )
        cache_device = _choose_rollout_cache_device(
            mode=str(args.rollout_cache_device),
            train_device=device,
            total_bytes=int(estimated_cache_nbytes),
        )
        cache_gib = float(estimated_cache_nbytes) / float(1024**3)

        if is_main:
            logger.info(
                "[ENV] feature_id=%s channels=%d t_max=%d pf_enabled=%s",
                args.feature_id,
                int(env.feature_channels),
                int(env.spec.t_max),
                bool(args.pf_enabled),
            )
            logger.info(
                "[FEATURE] id=%s channels=%d submit_supported=%s",
                feature_spec.feature_id,
                int(feature_spec.channels),
                bool(feature_spec.submit_supported),
            )
            logger.info(
                "[PERF] memory_format=%s pin_memory=%s rollout_cache_device=%s estimate=%.2fGiB",
                ("channels_last" if use_channels_last else "nchw"),
                bool(use_pin_memory),
                str(cache_device),
                cache_gib,
            )
            logger.info(
                "[AUX] opp_param_loss_coef=%.6g use_valid_mask=%s active=%s",
                float(aux_opp_param_loss_coef),
                bool(aux_opp_param_use_valid_mask),
                bool(aux_opp_param_active),
            )
            if device.type == "cuda" and cache_device.type == "cpu":
                logger.warning("[PERF] rollout cache is on CPU; this can reduce GPU utilization during PPO updates")
            logger.info("[MODEL] %s", resolved_model_config)
            if resume_enabled:
                logger.info(
                    "[RESUME] enabled=%s from=%s step=%d",
                    bool(resume_enabled),
                    str(resume_from),
                    int(global_step),
                )

        rollout_workspace = create_rollout_workspace(
            env,
            num_steps=int(local_cfg.num_steps),
            device=device,
            channels_last=bool(use_channels_last),
            pin_memory=bool(use_pin_memory),
            collect_aux_targets=bool(aux_opp_param_active),
        )
        buffer = RolloutBuffer(
            num_steps=local_cfg.num_steps,
            num_envs=local_cfg.num_envs,
            obs_shape=obs_shape,
            action_shape=tuple(),
            device=cache_device,
            use_action_mask=bool(args.use_action_mask),
            action_dim=action_dim,
            use_aux_opp_param_targets=bool(aux_opp_param_active),
        )

        rng = np.random.default_rng(int(seed_base))
        if resume_enabled and isinstance(resume_meta.get("rng_state"), dict):
            try:
                rng.bit_generator.state = dict(resume_meta["rng_state"])
            except Exception:
                if is_main:
                    logger.warning("failed to restore rng_state from resume; continuing with seeded RNG")

        train_metrics_jsonl = layout.logs_dir / "train_metrics.jsonl"
        periodic_val_jsonl = layout.logs_dir / "periodic_val_metrics.jsonl"
        best_model = layout.models_dir / "best.pt"
        last_model = layout.models_dir / "last.pt"
        checkpoint_dir = layout.models_dir / "checkpoints"
        (
            periodic_val_enabled,
            eval_at_start_enabled,
            best_metric_name,
            eval_round,
            next_eval_step,
            checkpoint_interval_steps,
            next_checkpoint_step,
        ) = _resolve_eval_checkpoint_state(
            args=args,
            resume_enabled=bool(resume_enabled),
            global_step=int(global_step),
            best_metric_name=str(best_metric_name),
            restored_eval_round=restored_eval_round,
            restored_next_eval_step=restored_next_eval_step,
            restored_next_checkpoint_step=restored_next_checkpoint_step,
        )
        if checkpoint_interval_steps > 0 and is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def _build_resume_meta(*, iteration: int) -> dict[str, Any]:
            return _build_resume_meta_payload(
                iteration=int(iteration),
                run_name=str(run_name),
                global_step=int(global_step),
                world_size=int(world_size),
                feature_id=str(args.feature_id),
                pf_enabled=bool(args.pf_enabled),
                obs_shape=tuple(obs_shape),
                action_dim=int(action_dim),
                use_channels_last=bool(use_channels_last),
                use_pin_memory=bool(use_pin_memory),
                cache_device=cache_device,
                cfg_global=cfg_global,
                cfg_local=local_cfg,
                best_metric_name=str(best_metric_name),
                best_metric_value=float(best_metric_value),
                best_metric_source=str(best_metric_source),
                eval_round=int(eval_round),
                next_eval_step=int(next_eval_step),
                next_checkpoint_step=int(next_checkpoint_step),
                rng=rng,
                train_vecnorm=train_vecnorm,
            )

        lr_schedule_fn = schedules.learning_rate
        ent_schedule_fn = schedules.ent_coef
        clip_schedule_fn = schedules.clip_coef
        vf_clip_schedule_fn = schedules.clip_range_vf
        lr_schedule_desc = schedules.learning_rate.description
        ent_schedule_desc = schedules.ent_coef.description
        clip_schedule_desc = schedules.clip_coef.description
        vf_clip_schedule_desc = (
            "disabled" if schedules.clip_range_vf is None else schedules.clip_range_vf.description
        )
        if is_main:
            logger.info(
                "schedules: lr=%s ent=%s clip=%s vf_clip=%s",
                lr_schedule_desc,
                ent_schedule_desc,
                clip_schedule_desc,
                vf_clip_schedule_desc,
            )
            if bool(cfg_global.clip_vloss) and vf_clip_schedule_fn is None:
                logger.warning("clip_vloss=true but clip_range_vf is unset; value clipping is disabled")

        start_time = time.time()
        global_batch_size = int(cfg_global.batch_size)
        num_iterations = int(cfg_global.num_iterations)
        rollout_agent = _unwrap_ddp(agent)
        start_iteration = int(global_step // global_batch_size) + 1
        if start_iteration > num_iterations and is_main:
            logger.info(
                "no PPO update needed: global_step=%d already reached total_timesteps=%d",
                int(global_step),
                int(cfg_global.total_timesteps),
            )

        if eval_at_start_enabled and global_step == 0:
            if is_distributed:
                dist.barrier()
            if is_main:
                eval_seed_base = int(args.eval_seed_start)
                if not bool(args.eval_fixed_seeds):
                    eval_seed_base = int(args.eval_seed_start) + int(eval_round) * int(args.eval_episodes)
                eval_summary = _run_periodic_val(
                    env_id="AHC061Local-v0",
                    feature_id=str(args.feature_id),
                    pf_enabled=bool(args.pf_enabled),
                    agent=rollout_agent,
                    device=device,
                    episodes=int(args.eval_episodes),
                    seed_start=int(eval_seed_base),
                    deterministic=bool(args.eval_deterministic),
                    use_action_mask=bool(args.use_action_mask),
                    amp=bool(args.amp),
                    vecnorm_state=_vecnorm_state_or_none(train_vecnorm),
                    vecnorm_norm_obs=bool(args.vecnorm_norm_obs),
                    vecnorm_norm_reward=bool(args.vecnorm_eval_norm_reward),
                    vecnorm_clip_obs=float(args.vecnorm_clip_obs),
                    vecnorm_clip_reward=float(args.vecnorm_clip_reward),
                    vecnorm_epsilon=float(args.vecnorm_epsilon),
                    vecnorm_gamma=float(vecnorm_gamma),
                )
                eval_row = {
                    "global_step": 0,
                    "eval_round": int(eval_round),
                    "seed_base": int(eval_seed_base),
                    "fixed_seeds": bool(args.eval_fixed_seeds),
                    "summary": eval_summary,
                }
                _append_jsonl(periodic_val_jsonl, eval_row)
                logger.info(
                    "periodic_val@start step=0 round=%d return=%.5f game_score=%.2f",
                    int(eval_round),
                    float(eval_summary["return"]["mean"]),
                    float(eval_summary["terminal_game_score"]["mean"]),
                )
                cand = float(eval_summary["terminal_game_score"]["mean"])
                if np.isfinite(cand) and cand > best_metric_value:
                    best_metric_value = float(cand)
                    best_metric_source = "periodic_val_at_start"
                    save_agent_checkpoint(
                        best_model,
                        _unwrap_model(agent),
                        optimizer=optimizer,
                        meta=_build_resume_meta(iteration=0),
                    )
            if is_distributed:
                dist.barrier()
            eval_round += 1

        for iteration in range(start_iteration, num_iterations + 1):
            progress = schedule_progress(int(iteration), int(num_iterations))
            lr_current = float(lr_schedule_fn(progress))
            if not np.isfinite(lr_current) or lr_current < 0.0:
                raise ValueError(f"invalid scheduled learning_rate={lr_current} at progress={progress}")
            optimizer.param_groups[0]["lr"] = lr_current

            ent_coef_current = float(ent_schedule_fn(progress))
            clip_coef_current = float(clip_schedule_fn(progress))
            if not np.isfinite(ent_coef_current) or ent_coef_current < 0.0:
                raise ValueError(f"invalid scheduled ent_coef={ent_coef_current} at progress={progress}")
            if not np.isfinite(clip_coef_current) or clip_coef_current < 0.0:
                raise ValueError(f"invalid scheduled clip_coef={clip_coef_current} at progress={progress}")
            clip_range_vf_current: float | None = None
            if vf_clip_schedule_fn is not None:
                clip_range_vf_current = float(vf_clip_schedule_fn(progress))
                if not np.isfinite(clip_range_vf_current) or clip_range_vf_current < 0.0:
                    raise ValueError(
                        f"invalid scheduled clip_range_vf={clip_range_vf_current} at progress={progress}"
                    )
            trainer.set_runtime_coefficients(
                ent_coef=float(ent_coef_current),
                clip_coef=float(clip_coef_current),
                clip_range_vf=(None if clip_range_vf_current is None else float(clip_range_vf_current)),
                aux_opp_param_loss_coef=float(aux_opp_param_loss_coef),
                aux_opp_param_use_valid_mask=bool(aux_opp_param_use_valid_mask),
            )

            seeds = torch.as_tensor(
                rng.integers(0, np.iinfo(np.int64).max, size=(local_cfg.num_envs,), dtype=np.int64),
                dtype=torch.int64,
                device="cpu",
            )
            env.reset_random(seeds)

            rollout = collect_rollout(
                env,
                rollout_agent,
                device,
                local_cfg.num_steps,
                use_action_mask=bool(args.use_action_mask),
                sample=True,
                amp=bool(args.amp),
                channels_last=bool(use_channels_last),
                workspace=rollout_workspace,
                pin_memory=bool(use_pin_memory),
                vecnorm=train_vecnorm,
                collect_aux_targets=bool(aux_opp_param_active),
            )

            copy_non_blocking = bool(cache_device.type == "cuda")
            buffer.obs.copy_(rollout.obs.to(device=cache_device, dtype=torch.float32, non_blocking=copy_non_blocking))
            buffer.actions.copy_(rollout.actions.to(device=cache_device, dtype=torch.long, non_blocking=copy_non_blocking))
            buffer.logprobs.copy_(rollout.logprobs.to(device=cache_device, dtype=torch.float32, non_blocking=copy_non_blocking))
            buffer.rewards.copy_(rollout.rewards.to(device=cache_device, dtype=torch.float32, non_blocking=copy_non_blocking))
            buffer.dones.copy_(rollout.dones.to(device=cache_device, dtype=torch.float32, non_blocking=copy_non_blocking))
            buffer.values.copy_(rollout.values.to(device=cache_device, dtype=torch.float32, non_blocking=copy_non_blocking))
            if buffer.action_masks is not None:
                buffer.action_masks.copy_(rollout.masks.to(device=cache_device, dtype=torch.bool, non_blocking=copy_non_blocking))
            if bool(aux_opp_param_active):
                if rollout.aux_opp_param_true is None or rollout.aux_opp_valid is None:
                    raise RuntimeError("rollout did not provide aux_opp_param targets")
                if buffer.aux_opp_param_true is None or buffer.aux_opp_valid is None:
                    raise RuntimeError("rollout buffer is missing aux_opp_param storage")
                buffer.aux_opp_param_true.copy_(
                    rollout.aux_opp_param_true.to(
                        device=cache_device,
                        dtype=torch.float32,
                        non_blocking=copy_non_blocking,
                    )
                )
                buffer.aux_opp_valid.copy_(
                    rollout.aux_opp_valid.to(
                        device=cache_device,
                        dtype=torch.bool,
                        non_blocking=copy_non_blocking,
                    )
                )

            next_value = rollout.last_value.to(device=cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
            next_done = rollout.last_done.to(device=cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
            buffer.compute_gae(next_value, next_done, local_cfg.gamma, local_cfg.gae_lambda)
            explained_variance_local = _explained_variance(buffer.values, buffer.returns)
            stats = trainer.update(buffer)

            stats_vec = torch.tensor(
                [
                    float(stats.policy_loss),
                    float(stats.value_loss),
                    float(stats.entropy),
                    float(stats.approx_kl),
                    float(stats.clipfrac),
                    (0.0 if not np.isfinite(float(stats.value_clipfrac)) else float(stats.value_clipfrac)),
                    (1.0 if not np.isfinite(float(stats.value_clipfrac)) else 0.0),
                    (0.0 if not np.isfinite(float(explained_variance_local)) else float(explained_variance_local)),
                    (0.0 if not np.isfinite(float(explained_variance_local)) else 1.0),
                    (0.0 if not np.isfinite(float(stats.aux_opp_param_loss)) else float(stats.aux_opp_param_loss)),
                    (0.0 if not np.isfinite(float(stats.aux_opp_param_loss)) else 1.0),
                ],
                device=device,
                dtype=torch.float64,
            )
            _all_reduce_sum_tensor(stats_vec)
            ws = float(max(1, int(world_size)))
            policy_loss_g = float((stats_vec[0] / ws).item())
            value_loss_g = float((stats_vec[1] / ws).item())
            entropy_g = float((stats_vec[2] / ws).item())
            approx_kl_g = float((stats_vec[3] / ws).item())
            clipfrac_g = float((stats_vec[4] / ws).item())
            if float(stats_vec[6].item()) >= ws:
                value_clipfrac_g = float("nan")
            else:
                value_clipfrac_g = float((stats_vec[5] / ws).item())
            if float(stats_vec[8].item()) <= 0.0:
                explained_variance_g = float("nan")
            else:
                explained_variance_g = float((stats_vec[7] / stats_vec[8]).item())
            if float(stats_vec[10].item()) <= 0.0:
                aux_opp_param_loss_g = float("nan")
            else:
                aux_opp_param_loss_g = float((stats_vec[9] / stats_vec[10]).item())

            score_now = env.official_score().to(dtype=torch.float32)
            metric_vec = torch.tensor(
                [
                    float(score_now.sum().item()),
                    float(score_now.numel()),
                ],
                device=device,
                dtype=torch.float64,
            )
            _all_reduce_sum_tensor(metric_vec)
            mean_official = float((metric_vec[0] / metric_vec[1]).item()) if float(metric_vec[1].item()) > 0.0 else 0.0

            global_step += int(global_batch_size)
            _sync_vecnorm_ddp_(train_vecnorm, device=device)
            elapsed_vec = torch.tensor([max(1e-9, time.time() - start_time)], device=device, dtype=torch.float64)
            _all_reduce_max_tensor(elapsed_vec)
            elapsed = float(elapsed_vec[0].item())
            sps = int(float(global_step) / max(1e-9, elapsed))

            row: dict[str, Any] = {
                "iteration": int(iteration),
                "global_step": int(global_step),
                "sps": int(sps),
                "learning_rate": float(lr_current),
                "schedule_progress": float(progress),
                "ent_coef_current": float(ent_coef_current),
                "clip_coef_current": float(clip_coef_current),
                "clip_range_vf_current": (
                    float(clip_range_vf_current) if clip_range_vf_current is not None else float("nan")
                ),
                "policy_loss": float(policy_loss_g),
                "value_loss": float(value_loss_g),
                "entropy": float(entropy_g),
                "approx_kl": float(approx_kl_g),
                "clipfrac": float(clipfrac_g),
                "value_clipfrac": float(value_clipfrac_g),
                "explained_variance": float(explained_variance_g),
                "aux_opp_param_loss": float(aux_opp_param_loss_g),
                "mean_official_score": float(mean_official),
            }

            run_periodic_eval = bool(periodic_val_enabled and global_step >= next_eval_step)
            if run_periodic_eval and is_distributed:
                dist.barrier()
            if run_periodic_eval and is_main:
                eval_seed_base = int(args.eval_seed_start)
                if not bool(args.eval_fixed_seeds):
                    eval_seed_base = int(args.eval_seed_start) + int(eval_round) * int(args.eval_episodes)
                eval_summary = _run_periodic_val(
                    env_id="AHC061Local-v0",
                    feature_id=str(args.feature_id),
                    pf_enabled=bool(args.pf_enabled),
                    agent=rollout_agent,
                    device=device,
                    episodes=int(args.eval_episodes),
                    seed_start=int(eval_seed_base),
                    deterministic=bool(args.eval_deterministic),
                    use_action_mask=bool(args.use_action_mask),
                    amp=bool(args.amp),
                    vecnorm_state=_vecnorm_state_or_none(train_vecnorm),
                    vecnorm_norm_obs=bool(args.vecnorm_norm_obs),
                    vecnorm_norm_reward=bool(args.vecnorm_eval_norm_reward),
                    vecnorm_clip_obs=float(args.vecnorm_clip_obs),
                    vecnorm_clip_reward=float(args.vecnorm_clip_reward),
                    vecnorm_epsilon=float(args.vecnorm_epsilon),
                    vecnorm_gamma=float(vecnorm_gamma),
                )
                eval_row = {
                    "global_step": int(global_step),
                    "eval_round": int(eval_round),
                    "seed_base": int(eval_seed_base),
                    "fixed_seeds": bool(args.eval_fixed_seeds),
                    "summary": eval_summary,
                }
                _append_jsonl(periodic_val_jsonl, eval_row)
                row["periodic_val_mean_return"] = float(eval_summary["return"]["mean"])
                row["periodic_val_mean_illegal_penalty"] = float(eval_summary["illegal_penalty"]["mean"])
                row["periodic_val_mean_terminal_score"] = float(eval_summary["terminal_score"]["mean"])
                row["periodic_val_mean_terminal_game_score"] = float(eval_summary["terminal_game_score"]["mean"])
                row["periodic_val_mean_game_score_ratio"] = float(eval_summary["game_score_ratio"]["mean"])
                cand = float(eval_summary["terminal_game_score"]["mean"])
                if np.isfinite(cand) and cand > best_metric_value:
                    best_metric_value = float(cand)
                    best_metric_source = "periodic_val"
                    save_agent_checkpoint(
                        best_model,
                        _unwrap_model(agent),
                        optimizer=optimizer,
                        meta=_build_resume_meta(iteration=int(iteration)),
                    )
            if run_periodic_eval and is_distributed:
                dist.barrier()
            if run_periodic_eval:
                eval_round += 1
                next_eval_step += int(max(1, args.eval_interval_steps))
            elif is_main and not periodic_val_enabled:
                cand = float(mean_official)
                if np.isfinite(cand) and cand > best_metric_value:
                    best_metric_value = float(cand)
                    best_metric_source = "mean_official_score"
                    save_agent_checkpoint(
                        best_model,
                        _unwrap_model(agent),
                        optimizer=optimizer,
                        meta=_build_resume_meta(iteration=int(iteration)),
                    )

            if is_main:
                _append_jsonl(train_metrics_jsonl, row)

            if is_main and args.log_interval_iters > 0 and (iteration % int(args.log_interval_iters) == 0):
                logger.info(
                    "iter=%d/%d step=%d sps=%d lr=%.6g ploss=%.5f vloss=%.5f aux=%.5f ent=%.5f kl=%.5f ev=%.5f vclip=%.5f score=%.1f",
                    iteration,
                    num_iterations,
                    global_step,
                    sps,
                    lr_current,
                    policy_loss_g,
                    value_loss_g,
                    aux_opp_param_loss_g,
                    entropy_g,
                    approx_kl_g,
                    explained_variance_g,
                    value_clipfrac_g,
                    mean_official,
                )

            if is_main and (iteration % int(max(1, local_cfg.save_interval)) == 0 or iteration == num_iterations):
                save_agent_checkpoint(
                    last_model,
                    _unwrap_model(agent),
                    optimizer=optimizer,
                    meta=_build_resume_meta(iteration=int(iteration)),
                )

            if is_main and checkpoint_interval_steps > 0 and global_step >= int(next_checkpoint_step):
                save_agent_checkpoint(
                    checkpoint_dir / f"step_{int(global_step):012d}.pt",
                    _unwrap_model(agent),
                    optimizer=optimizer,
                    meta=_build_resume_meta(iteration=int(iteration)),
                )
                next_checkpoint_step += int(checkpoint_interval_steps)

        if is_main:
            final_iteration = int(max(0, min(num_iterations, int(global_step // max(1, global_batch_size)))))
            save_agent_checkpoint(
                last_model,
                _unwrap_model(agent),
                optimizer=optimizer,
                meta=_build_resume_meta(iteration=final_iteration),
            )
            summary = {
                "run_name": run_name,
                "run_dir": str(layout.root),
                "global_step": int(global_step),
                "final_iteration": int(final_iteration),
                "env_id": "AHC061Local-v0",
                "feature_id": str(args.feature_id),
                "pf_enabled": bool(args.pf_enabled),
                "aux_opp_param": {
                    "loss_coef": float(aux_opp_param_loss_coef),
                    "use_valid_mask": bool(aux_opp_param_use_valid_mask),
                    "active": bool(aux_opp_param_active),
                },
                "vecnormalize": {
                    "enabled": bool(train_vecnorm is not None),
                    "norm_obs": bool(args.vecnorm_norm_obs),
                    "norm_reward_train": bool(args.vecnorm_norm_reward),
                    "norm_reward_val": bool(args.vecnorm_eval_norm_reward),
                    "clip_obs": float(args.vecnorm_clip_obs),
                    "clip_reward": float(args.vecnorm_clip_reward),
                    "epsilon": float(args.vecnorm_epsilon),
                    "gamma": float(vecnorm_gamma),
                },
                "periodic_val": {
                    "enabled": bool(periodic_val_enabled),
                    "interval_steps": int(args.eval_interval_steps),
                    "episodes": int(args.eval_episodes),
                    "seed_start": int(args.eval_seed_start),
                    "fixed_seeds": bool(args.eval_fixed_seeds),
                    "deterministic": bool(args.eval_deterministic),
                    "val_at_start": bool(args.eval_at_start),
                    "metrics_jsonl": str(periodic_val_jsonl),
                },
                "resume": {
                    "enabled": bool(resume_enabled),
                    "resume_from": (str(resume_from) if resume_from is not None else ""),
                },
                "best_metric": {
                    "name": str(best_metric_name),
                    "value": float(best_metric_value),
                    "source": str(best_metric_source),
                },
                "models": {
                    "best": str(best_model),
                    "last": str(last_model),
                    "checkpoint_dir": (str(checkpoint_dir) if checkpoint_interval_steps > 0 else ""),
                    "checkpoint_interval_steps": int(checkpoint_interval_steps),
                },
                "logs": {
                    "train_metrics_jsonl": str(train_metrics_jsonl),
                    "periodic_val_metrics_jsonl": str(periodic_val_jsonl),
                },
                "elapsed_sec": float(time.time() - start_time),
            }
            (layout.reports_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (layout.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            update_manifest(
                layout,
                {
                    "status": "completed",
                    "result": summary,
                },
            )
        if is_distributed:
            dist.barrier()
        return 0
    except Exception as e:
        if layout is not None and is_main:
            update_manifest(
                layout,
                {
                    "status": "failed",
                    "error": str(e),
                    "progress": {"global_step": int(global_step)},
                },
            )
        raise
    finally:
        if _dist_ready():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
