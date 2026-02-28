"""PPO 学習のセットアップと反復実行を統括するサービス層。"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..env import BatchEnv, BatchEnvProtocol, ensure_batch_env
from ..env.feature_catalog import get_feature_spec
from ..eval.eval_service import run_policy_episodes
from ..game_constants import AUX_OPP_PARAM_TOTAL, OPP_SLOT_COUNT
from ..models import (
    build_agent,
    get_model_config_from_preset,
    get_model_preset,
    load_model_config_from_sources,
    normalize_model_config,
)
from ..pipeline.model_checkpoint_service import save_agent_checkpoint
from ..ppo.config import PPOConfig
from ..ppo.rollout import collect_rollout, create_rollout_workspace
from ..ppo.rollout_buffer import RolloutBuffer
from ..ppo.trainer import PPOTrainer, UpdateStats
from ..ppo.vecnorm import VecNormalize
from ..train.schedule import (
    PPOScheduleSet,
    RuntimeScheduleCoefficients,
    RuntimeScheduleResolver,
    resolve_vecnorm_gamma,
    validate_vecnorm_config,
)
from ..utils.experiment import coerce_optional_path, create_run_layout, make_run_name, to_jsonable, update_manifest
from ..utils.log_utils import get_logger
from ..utils.metrics import group_score_mean_variance_by_m_u, summarize
from ..utils.runtime import choose_device as choose_runtime_device
from ..utils.tracking import MetricTracker
from .requests import PPORequest, TrainPPORequest, args_to_cfg, build_ppo_request

logger = get_logger("train_ppo")


def choose_device(name: str) -> torch.device:
    """実行デバイス指定文字列を `torch.device` に解決する。

    Args:
        name (str): `auto` / `cpu` / `cuda` などの指定。

    Returns:
        torch.device: 解決済みデバイス。
    """
    return choose_runtime_device(name)


def _resolve_model_config(args: PPORequest) -> dict[str, Any]:
    """`PPORequest` から最終的なモデル設定辞書を解決する。"""
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
    """価値予測の explained variance（決定係数相当）を計算する。"""
    y_true_np = y_true.detach().reshape(-1).cpu().numpy()
    y_pred_np = y_pred.detach().reshape(-1).cpu().numpy()
    var_y = float(np.var(y_true_np))
    if var_y <= 1e-12:
        return float("nan")
    return 1.0 - float(np.var(y_true_np - y_pred_np) / var_y)


def _dist_ready() -> bool:
    """`torch.distributed` が利用可能かつ初期化済みかを返す。"""
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum_tensor(x: torch.Tensor) -> torch.Tensor:
    """分散実行時にテンソルを全 rank で加算集約する。"""
    if _dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _all_reduce_max_tensor(x: torch.Tensor) -> torch.Tensor:
    """分散実行時にテンソルを全 rank で最大値集約する。"""
    if _dist_ready():
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x


def _resolve_distributed_mode(mode: str) -> str:
    """分散実行モード文字列を正規化し妥当性を検証する。"""
    m = str(mode).strip().lower()
    if m not in ("auto", "off", "on"):
        raise ValueError(f"unsupported distributed mode={mode!r}; expected auto|off|on")
    return m


def _seed_everything(seed: int, *, device: torch.device) -> None:
    """Python・NumPy・PyTorch の乱数シードをまとめて設定する。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """`DDP` でラップされていれば元モデルを返す。"""
    if isinstance(model, DDP):
        return model.module
    return model


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """`DDP` / `torch.compile` ラッパを外して実体モデルを返す。"""
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        return getattr(model, "_orig_mod")
    return model


def _resolve_use_channels_last(*, device: torch.device, mode: str) -> bool:
    """`memory_format` 指定から channels-last 適用可否を解決する。"""
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
    """ロールアウトバッファに必要な概算メモリ量（bytes）を見積もる。"""
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
        total += t * b * AUX_OPP_PARAM_TOTAL * 4  # aux opp_param_true float32
        total += t * b * OPP_SLOT_COUNT  # aux opp_valid uint8/bool
    return int(total)


def _choose_rollout_cache_device(*, mode: str, train_device: torch.device, total_bytes: int) -> torch.device:
    """ロールアウトキャッシュ配置先（CPU/GPU）を決定する。"""
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
    """初期重みチェックポイントを読み込み、モデルへ反映する。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format (missing model_state_dict): {path}")
    _unwrap_model(agent).load_state_dict(payload["model_state_dict"], strict=True)
    meta = payload.get("meta")
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """JSONL ファイル末尾に 1 レコード追記する。"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=True) + "\n")


def _safe_int(x: Any, default: int = 0) -> int:
    """`int()` 変換に失敗した場合は既定値を返す。"""
    try:
        return int(x)
    except Exception:
        return int(default)


def _resolve_train_seed_range(args: PPORequest) -> tuple[int, int]:
    """学習時に使うランダム seed 範囲を検証して返す。"""
    i64_max = int(np.iinfo(np.int64).max)
    seed_min = int(_safe_int(getattr(args, "train_seed_min", 0), 0))
    seed_max_exclusive = int(_safe_int(getattr(args, "train_seed_max_exclusive", i64_max), i64_max))
    if seed_min < 0:
        raise ValueError(f"train_seed_min must be >= 0, got {seed_min}")
    if seed_max_exclusive <= seed_min:
        raise ValueError(
            f"train_seed_max_exclusive must be greater than train_seed_min: {seed_max_exclusive} <= {seed_min}"
        )
    if seed_max_exclusive > i64_max:
        raise ValueError(f"train_seed_max_exclusive must be <= int64 max ({i64_max}), got {seed_max_exclusive}")
    return int(seed_min), int(seed_max_exclusive)


def _vecnorm_state_or_none(vecnorm: VecNormalize | None) -> dict[str, Any] | None:
    """`VecNormalize` の状態を安全に辞書化して返す。"""
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
    """RunningMeanStd 統計を DDP rank 間で合成する。"""
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
    """`VecNormalize` の統計量を DDP rank 間で同期する。"""
    if vecnorm is None or not _dist_ready():
        return
    m, v, c = _sync_rms_ddp_(
        vecnorm.obs_rms.mean.numpy(),
        vecnorm.obs_rms.var.numpy(),
        vecnorm.obs_rms.count,
        device=device,
    )
    vecnorm.obs_rms.mean = torch.as_tensor(m, dtype=torch.float64).clone()
    vecnorm.obs_rms.var = torch.clamp(torch.as_tensor(v, dtype=torch.float64), min=1e-12).clone()
    vecnorm.obs_rms.count = float(c)

    m, v, c = _sync_rms_ddp_(
        vecnorm.ret_rms.mean.numpy(),
        vecnorm.ret_rms.var.numpy(),
        vecnorm.ret_rms.count,
        device=device,
    )
    vecnorm.ret_rms.mean = torch.as_tensor(m, dtype=torch.float64).clone()
    vecnorm.ret_rms.var = torch.clamp(torch.as_tensor(v, dtype=torch.float64), min=1e-12).clone()
    vecnorm.ret_rms.count = float(c)


def _run_periodic_val(
    *,
    env_id: str,
    feature_id: str,
    pf_enabled: bool,
    agent: torch.nn.Module,
    device: torch.device,
    episodes: int,
    num_envs: int,
    seed_start: int,
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
    """定期評価を 1 回実行し、集計済みメトリクス辞書を返す。"""
    stats = run_policy_episodes(
        env_id=str(env_id),
        episodes=int(episodes),
        num_envs=int(num_envs),
        seed=int(seed_start),
        feature_id=str(feature_id),
        pf_enabled=bool(pf_enabled),
        policy="model_greedy",
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
        collect_score_breakdown=False,
    )
    return {
        "episodes": int(episodes),
        "return": summarize(stats.episode_returns).as_dict(),
        "terminal_game_score": summarize(stats.episode_terminal_game_scores).as_dict(),
        "terminal_game_score_grouped": group_score_mean_variance_by_m_u(
            scores=stats.episode_terminal_game_scores,
            m_values=stats.episode_m,
            u_values=stats.episode_u,
            m_key="m",
            u_key="u",
        ),
    }


@dataclass
class _RestoredTrainState:
    """チェックポイント復元で得られた学習状態をまとめるデータクラス。

    Attributes:
        init_meta (dict[str, Any]): `init_model` から復元したメタ情報。
        resume_meta (dict[str, Any]): `resume` チェックポイントのメタ情報。
        global_step (int): 復元後の学習ステップ。
        best_metric_value (float): ベストモデル判定に使う最大指標値。
        best_metric_name (str): ベスト判定の指標名。
        best_metric_source (str): ベスト値の更新元（評価/学習）。
        restored_eval_round (int | None): 復元された評価ラウンド数。
        restored_next_eval_iteration (int | None): 次回評価イテレーション。
        restored_next_checkpoint_iteration (int | None): 次回チェックポイント保存イテレーション。
    """
    init_meta: dict[str, Any]
    resume_meta: dict[str, Any]
    global_step: int
    best_metric_value: float
    best_metric_name: str
    best_metric_source: str
    restored_eval_round: int | None
    restored_next_eval_iteration: int | None
    restored_next_checkpoint_iteration: int | None


@dataclass(frozen=True)
class _IterationAggregateStats:
    """1 イテレーション分の集約済み学習統計。"""
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    value_clipfrac: float
    explained_variance: float
    aux_opp_param_loss: float
    mean_official_score: float


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
    """初期重み/再開チェックポイントを読み込み学習状態を復元する。"""
    init_meta: dict[str, Any] = {}
    resume_meta: dict[str, Any] = {}
    global_step = 0
    restored_eval_round: int | None = None
    restored_next_eval_iteration: int | None = None
    restored_next_checkpoint_iteration: int | None = None

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
        _unwrap_model(agent).load_state_dict(payload["model_state_dict"], strict=True)
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
        rv = resume_meta.get("next_eval_iteration")
        restored_next_eval_iteration = _safe_int(rv) if rv is not None else None
        rv = resume_meta.get("next_checkpoint_iteration")
        restored_next_checkpoint_iteration = _safe_int(rv) if rv is not None else None
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
        restored_next_eval_iteration=restored_next_eval_iteration,
        restored_next_checkpoint_iteration=restored_next_checkpoint_iteration,
    )


def _resolve_eval_checkpoint_state(
    *,
    args: PPORequest,
    resume_enabled: bool,
    completed_iterations: int,
    best_metric_name: str,
    restored_eval_round: int | None,
    restored_next_eval_iteration: int | None,
    restored_next_checkpoint_iteration: int | None,
) -> tuple[bool, bool, str, int, int, int, int]:
    """評価・チェックポイントの次回実行タイミングを復元/初期化する。"""
    checkpoint_interval_iterations = int(max(0, _safe_int(args.checkpoint_interval_iterations, 0)))
    periodic_val_enabled = bool(
        _safe_int(getattr(args, "eval_interval_iterations", 0), 0) > 0
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
    next_eval_iteration = int(max(1, _safe_int(getattr(args, "eval_interval_iterations", 0), 0)))
    next_checkpoint_iteration = 0
    if resume_enabled:
        if restored_eval_round is not None and int(restored_eval_round) >= 0:
            eval_round = int(restored_eval_round)
        elif periodic_val_enabled and int(args.eval_interval_iterations) > 0:
            eval_round = int(completed_iterations // int(args.eval_interval_iterations))
            if bool(eval_at_start_enabled) and completed_iterations > 0:
                eval_round += 1
        if restored_next_eval_iteration is not None and int(restored_next_eval_iteration) > 0:
            next_eval_iteration = int(restored_next_eval_iteration)
        elif periodic_val_enabled and int(args.eval_interval_iterations) > 0:
            interval = int(args.eval_interval_iterations)
            next_eval_iteration = int(((completed_iterations // interval) + 1) * interval)
        if checkpoint_interval_iterations > 0:
            if restored_next_checkpoint_iteration is not None and int(restored_next_checkpoint_iteration) > 0:
                next_checkpoint_iteration = int(restored_next_checkpoint_iteration)
            elif completed_iterations > 0:
                next_checkpoint_iteration = int(
                    ((completed_iterations // checkpoint_interval_iterations) + 1) * checkpoint_interval_iterations
                )
            else:
                next_checkpoint_iteration = int(checkpoint_interval_iterations)
    elif checkpoint_interval_iterations > 0:
        next_checkpoint_iteration = int(checkpoint_interval_iterations)

    return (
        periodic_val_enabled,
        eval_at_start_enabled,
        str(best_metric_name),
        int(eval_round),
        int(next_eval_iteration),
        int(checkpoint_interval_iterations),
        int(next_checkpoint_iteration),
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
    next_eval_iteration: int,
    next_checkpoint_iteration: int,
    rng: np.random.Generator,
    train_vecnorm: VecNormalize | None,
) -> dict[str, Any]:
    """再開実行に必要なメタ情報ペイロードを組み立てる。"""
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
        "cfg_global": cfg_global.model_dump(),
        "cfg_local": cfg_local.model_dump(),
        "best_metric_name": str(best_metric_name),
        "best_metric_value": float(best_metric_value),
        "best_metric_source": str(best_metric_source),
        "eval_round": int(eval_round),
        "next_eval_iteration": int(next_eval_iteration),
        "next_checkpoint_iteration": int(next_checkpoint_iteration),
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
    """`TrainPPORequest` を内部実行形式へ変換して PPO 学習を起動する。

    Args:
        train_req (TrainPPORequest): エントリーポイント/パイプラインから渡される学習要求。
        cfg (PPOConfig): 検証済み PPO 設定。
        device (torch.device): 学習実行デバイス。
        env_kwargs (Mapping[str, Any]): 環境初期化オプション。

    Returns:
        int: プロセス終了コード。
    """
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


class PPORunner:
    """PPO 学習を段階実行する状態保持オーケストレータ。

    デバイス初期化、環境/モデル構築、復元処理、学習ループ、
    評価・チェックポイント保存、最終レポート出力を一貫して管理する。
    """

    def __init__(
        self,
        args: PPORequest,
        *,
        cfg_global: PPOConfig,
        local_cfg: PPOConfig,
        schedules: PPOScheduleSet,
        vecnorm_gamma: float,
        dist_mode: str,
        world_size: int,
        rank: int,
        local_rank: int,
        local_num_envs: int,
        local_num_minibatches: int,
        train_seed_min: int,
        train_seed_max_exclusive: int,
        aux_opp_param_loss_coef: float,
        aux_opp_param_use_valid_mask: bool,
        aux_opp_param_active: bool,
    ) -> None:
        """学習実行に必要な設定と可変状態を初期化する。"""
        self.args = args
        self.cfg_global = cfg_global
        self.local_cfg = local_cfg
        self.schedules = schedules
        self.vecnorm_gamma = vecnorm_gamma
        self.dist_mode = dist_mode
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.local_num_envs = local_num_envs
        self.local_num_minibatches = local_num_minibatches
        self.train_seed_min = int(train_seed_min)
        self.train_seed_max_exclusive = int(train_seed_max_exclusive)
        self.aux_opp_param_loss_coef = aux_opp_param_loss_coef
        self.aux_opp_param_use_valid_mask = aux_opp_param_use_valid_mask
        self.aux_opp_param_active = aux_opp_param_active
        self.is_distributed = world_size > 1
        self.is_main = rank == 0
        self.resume_enabled = bool(args.resume or args.resume_from is not None)
        self.run_name = str(args.run_name).strip()
        self.resume_from: Path | None = coerce_optional_path(getattr(args, "resume_from", None), dot_is_none=True)
        # Mutable state set by phase methods
        self.device: torch.device = torch.device("cpu")
        self.layout: Any = None
        self.use_channels_last = False
        self.use_pin_memory = False
        self.seed_base = 0
        self.env: BatchEnvProtocol | None = None
        self.feature_spec: Any = None
        self.obs_shape: tuple[int, ...] = ()
        self.action_dim = 0
        self.agent: torch.nn.Module | None = None
        self.rollout_agent: torch.nn.Module | None = None
        self.resolved_model_config: dict[str, Any] = {}
        self.train_vecnorm: VecNormalize | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.trainer: PPOTrainer | None = None
        self.cache_device: torch.device = torch.device("cpu")
        self.rollout_workspace: Any = None
        self.buffer: RolloutBuffer | None = None
        self.rng: np.random.Generator = np.random.default_rng(0)
        self.init_meta: dict[str, Any] = {}
        self.resume_meta: dict[str, Any] = {}
        self.global_step = 0
        self.best_metric_value = float("-inf")
        self.best_metric_name = "mean_official_score"
        self.best_metric_source = ""
        self.eval_round = 0
        self.next_eval_iteration = int(max(1, _safe_int(getattr(args, "eval_interval_iterations", 0), 0)))
        self.next_checkpoint_iteration = 0
        self.checkpoint_interval_iterations = 0
        self.periodic_val_enabled = False
        self.eval_at_start_enabled = False
        self.train_metrics_jsonl: Path = Path(".")
        self.periodic_val_jsonl: Path = Path(".")
        self.tracker: MetricTracker | None = None
        self.best_model: Path = Path(".")
        self.last_model: Path = Path(".")
        self.checkpoint_dir: Path = Path(".")
        self.start_time = 0.0
        self.num_iterations = int(cfg_global.num_iterations)
        self.global_batch_size = int(cfg_global.batch_size)
        self.start_iteration = 1
        self.summary: dict[str, Any] | None = None
        self.runtime_schedules = RuntimeScheduleResolver(
            schedules=self.schedules,
            total_iterations=int(self.num_iterations),
            warmup_iterations=int(self.args.warmup_iters),
        )

    # ------------------------------------------------------------------
    # Stage 1: device and distributed setup
    # ------------------------------------------------------------------

    def _setup_device_and_dist(self) -> None:
        """分散設定を初期化し、実行デバイスを確定する。"""
        if self.is_distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("distributed training requires CUDA")
            if str(self.args.device) not in ("auto", "cuda", f"cuda:{self.local_rank}"):
                raise RuntimeError(
                    f"--device={self.args.device!r} conflicts with LOCAL_RANK={self.local_rank}. "
                    "Use --device auto/cuda with torchrun."
                )
            dist.init_process_group(backend="nccl")
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = choose_device(str(self.args.device))
        if self.device.type == "cuda":
            if self.device.index is None:
                torch.cuda.set_device(0)
                self.device = torch.device("cuda:0")
            else:
                torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # Stage 2: run directory, seeding, and initial manifest
    # ------------------------------------------------------------------

    def _setup_run_dir(self) -> None:
        """実験ディレクトリとトラッキング初期状態を準備する。"""
        args = self.args
        if not self.run_name and self.resume_enabled and self.resume_from is None:
            raise ValueError("--resume requires --run-name or --resume-from for train_ppo")
        if not self.run_name and self.resume_from is not None:
            rp = Path(self.resume_from).resolve()
            if rp.parent.name == "models" and rp.parent.parent.name:
                self.run_name = rp.parent.parent.name
            else:
                self.run_name = rp.parent.name
        if not self.run_name:
            self.run_name = make_run_name("train_ppo", seed=self.cfg_global.seed)
        if self.is_distributed:
            run_name_box = [self.run_name if self.is_main else ""]
            dist.broadcast_object_list(run_name_box, src=0)
            self.run_name = str(run_name_box[0])

        self.layout = create_run_layout(args.run_dir, self.run_name)
        if self.resume_enabled and self.resume_from is None:
            self.resume_from = self.layout.models_dir / "last.pt"
        if self.resume_enabled and (self.resume_from is None or not Path(self.resume_from).exists()):
            raise FileNotFoundError(f"resume checkpoint not found: {self.resume_from}")

        self.use_channels_last = _resolve_use_channels_last(device=self.device, mode=str(args.memory_format))
        self.use_pin_memory = bool(args.pin_memory and self.device.type == "cuda")
        self.seed_base = int(self.cfg_global.seed) + int(self.rank) * 1_000_003
        _seed_everything(self.seed_base, device=self.device)

        if self.is_main:
            args_snapshot = to_jsonable(vars(args))
            cfg_snapshot = to_jsonable(self.cfg_global)
            local_cfg_snapshot = to_jsonable(self.local_cfg)
            distributed_snapshot = {
                "mode": str(self.dist_mode),
                "world_size": int(self.world_size),
            }
            self.tracker = MetricTracker(
                self.layout.root,
                run_name=self.run_name,
                mlflow_tracking_uri=str(getattr(args, "mlflow_tracking_uri", "")),
                mlflow_experiment=str(getattr(args, "mlflow_experiment", "ahc061")),
                mlflow_run_name=str(getattr(args, "mlflow_run_name", "")),
                tensorboard=bool(getattr(args, "tensorboard", False)),
                config={
                    "args": args_snapshot,
                    "ppo_config": cfg_snapshot,
                    "ppo_config_local": local_cfg_snapshot,
                    "distributed": distributed_snapshot,
                    "paths": self.layout.as_dict(),
                },
            )
            update_manifest(
                self.layout,
                {
                    "kind": "train_ppo",
                    "status": "running",
                    "run_name": self.run_name,
                    "resume": {
                        "enabled": bool(self.resume_enabled),
                        "resume_from": (str(self.resume_from) if self.resume_from is not None else ""),
                    },
                    "args": args_snapshot,
                    "ppo_config": cfg_snapshot,
                    "ppo_config_local": local_cfg_snapshot,
                    "distributed": distributed_snapshot,
                    "paths": self.layout.as_dict(),
                },
            )
            self.tracker.log_event(
                "train_start",
                {
                    "device": str(self.device),
                    "distributed": distributed_snapshot,
                },
            )
        if self.is_distributed:
            dist.barrier()
        if self.is_main:
            logger.info("[RUN] %s", self.run_name)
            logger.info("[DEVICE] %s", self.device)
            logger.info(
                "[DDP] mode=%s world_size=%d local_num_envs=%d local_num_minibatches=%d",
                str(self.dist_mode),
                int(self.world_size),
                int(self.local_num_envs),
                int(self.local_num_minibatches),
            )

    # ------------------------------------------------------------------
    # Stage 3: environment and model construction
    # ------------------------------------------------------------------

    def _build_env_and_model(self) -> None:
        """環境・モデル・最適化器を構築して学習準備を整える。"""
        args = self.args
        local_cfg = self.local_cfg

        env_impl = BatchEnv(
            batch_size=int(local_cfg.num_envs),
            feature_id=str(args.feature_id),
            pf_enabled=bool(args.pf_enabled),
            verbose_build=False,
        )
        self.env = ensure_batch_env(env_impl)
        self.feature_spec = get_feature_spec(str(args.feature_id), verbose_build=False)
        if local_cfg.num_steps > int(self.env.spec.t_max):
            raise ValueError(
                f"num_steps must be <= env.spec.t_max ({self.env.spec.t_max}) for cpp batch env; got {local_cfg.num_steps}"
            )

        board_size = int(self.env.board_size)
        self.action_dim = int(self.env.action_dim)
        self.obs_shape = (int(self.env.feature_channels), board_size, board_size)
        self.train_vecnorm = (
            VecNormalize(
                num_envs=int(local_cfg.num_envs),
                obs_shape=tuple(self.obs_shape),
                norm_obs=bool(args.vecnorm_norm_obs),
                norm_reward=bool(args.vecnorm_norm_reward),
                clip_obs=float(args.vecnorm_clip_obs),
                clip_reward=float(args.vecnorm_clip_reward),
                epsilon=float(args.vecnorm_epsilon),
                gamma=float(self.vecnorm_gamma),
                training=True,
            )
            if bool(args.vecnorm)
            else None
        )

        model_config = _resolve_model_config(args)
        if self.aux_opp_param_active:
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

        agent, self.resolved_model_config = build_agent(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            model_config=model_config,
            default_type=str(model_config.get("type", "DiscreteBoardAgent")),
        )
        agent = agent.to(self.device)
        if self.use_channels_last and self.device.type == "cuda":
            agent = agent.to(memory_format=torch.channels_last)
        board_channels = getattr(agent, "board_channels", None)
        if self.is_main and board_channels is not None and int(board_channels) != int(self.env.feature_channels):
            logger.warning(
                "board_channels(%d) != env.feature_channels(%d); observation will be split into board/global by flatten order",
                int(board_channels),
                int(self.env.feature_channels),
            )
        if self.is_main and int(self.feature_spec.channels) != int(self.env.feature_channels):
            logger.warning(
                "feature catalog channels(%d) != env feature_channels(%d) for feature_id=%s",
                int(self.feature_spec.channels),
                int(self.env.feature_channels),
                str(args.feature_id),
            )
        if self.aux_opp_param_active and not callable(getattr(agent, "get_aux_opp_param", None)):
            raise ValueError(
                "aux_opp_param_loss_coef > 0 requires model to implement get_aux_opp_param(obs) -> [B,7,5]"
            )

        # Check before compile: aux head present but coef=0 → DDP must track unused params.
        _has_aux_head = bool(getattr(agent, "use_aux_opp_param_head", False))

        if bool(args.compile):
            try:
                agent = torch.compile(agent)
            except Exception as e:  # pragma: no cover
                if self.is_main:
                    logger.warning("torch.compile failed; continue without compile: %s", e)

        if self.is_distributed:
            ddp_device_id = int(self.device.index if self.device.index is not None else self.local_rank)
            _find_unused = _has_aux_head and not self.aux_opp_param_active
            agent = DDP(
                agent,
                device_ids=[ddp_device_id],
                output_device=ddp_device_id,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=_find_unused,
            )

        self.agent = agent
        self.rollout_agent = _unwrap_ddp(agent)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=float(self.local_cfg.learning_rate), eps=1e-5)

    # ------------------------------------------------------------------
    # Stage 4: weight loading and rollout infrastructure
    # ------------------------------------------------------------------

    def _restore_weights_and_infra(self) -> None:
        """重み復元、VecNorm 復元、バッファ類の確保を行う。"""
        args = self.args
        local_cfg = self.local_cfg

        restored_state = _restore_training_state(
            args=args,
            resume_enabled=bool(self.resume_enabled),
            resume_from=self.resume_from,
            agent=self.agent,
            optimizer=self.optimizer,
            obs_shape=tuple(self.obs_shape),
            action_dim=int(self.action_dim),
            cfg_global=self.cfg_global,
            device=self.device,
            is_main=bool(self.is_main),
            best_metric_value=float(self.best_metric_value),
            best_metric_name=str(self.best_metric_name),
            best_metric_source=str(self.best_metric_source),
        )
        self.init_meta = dict(restored_state.init_meta)
        self.resume_meta = dict(restored_state.resume_meta)
        self.global_step = int(restored_state.global_step)
        self.best_metric_value = float(restored_state.best_metric_value)
        self.best_metric_name = str(restored_state.best_metric_name)
        self.best_metric_source = str(restored_state.best_metric_source)
        restored_eval_round = restored_state.restored_eval_round
        restored_next_eval_iteration = restored_state.restored_next_eval_iteration
        restored_next_checkpoint_iteration = restored_state.restored_next_checkpoint_iteration

        incoming_vec_state = None
        incoming_vec_source = ""
        if self.resume_enabled:
            cand = self.resume_meta.get("vecnormalize_state")
            if isinstance(cand, dict):
                incoming_vec_state = cand
                incoming_vec_source = "resume"
        else:
            cand = self.init_meta.get("vecnormalize_state")
            if isinstance(cand, dict):
                incoming_vec_state = cand
                incoming_vec_source = "init_model"

        if self.train_vecnorm is not None:
            if isinstance(incoming_vec_state, dict):
                self.train_vecnorm.load_state_dict(incoming_vec_state)
                self.train_vecnorm.set_training(True)
                if self.is_main:
                    logger.info("restored vecnormalize state from %s checkpoint", incoming_vec_source)
            elif self.resume_enabled and self.is_main:
                logger.warning("vecnorm is enabled but resume checkpoint has no vecnormalize_state")
            elif (not self.resume_enabled) and self.args.init_model is not None and self.is_main:
                logger.warning(
                    "vecnorm is enabled but init_model checkpoint has no vecnormalize_state; "
                    "using fresh statistics"
                )
            if self.is_main:
                logger.info(
                    "vecnorm: enabled=true norm_obs=%s norm_reward=%s clip_obs=%.4f clip_reward=%.4f eps=%g gamma=%.6f",
                    bool(self.train_vecnorm.norm_obs),
                    bool(self.train_vecnorm.norm_reward),
                    float(self.train_vecnorm.clip_obs),
                    float(self.train_vecnorm.clip_reward),
                    float(self.train_vecnorm.epsilon),
                    float(self.train_vecnorm.gamma),
                )
        else:
            if isinstance(incoming_vec_state, dict) and self.is_main:
                logger.warning(
                    "checkpoint has vecnormalize_state but vecnorm is disabled; observations/rewards will be unnormalized"
                )
            if self.is_main:
                logger.info("vecnorm: enabled=false")

        self.trainer = PPOTrainer(
            cfg=local_cfg, agent=self.agent, optimizer=self.optimizer, use_channels_last=bool(self.use_channels_last)
        )

        estimated_cache_nbytes = _estimate_rollout_cache_nbytes(
            num_steps=int(local_cfg.num_steps),
            num_envs=int(local_cfg.num_envs),
            obs_channels=int(self.env.feature_channels),
            board_size=int(self.env.board_size),
            action_dim=int(self.action_dim),
            use_action_mask=bool(args.use_action_mask),
            use_aux_opp_param_targets=bool(self.aux_opp_param_active),
        )
        self.cache_device = _choose_rollout_cache_device(
            mode=str(args.rollout_cache_device),
            train_device=self.device,
            total_bytes=int(estimated_cache_nbytes),
        )
        cache_gib = float(estimated_cache_nbytes) / float(1024**3)

        if self.is_main:
            logger.info(
                "[ENV] feature_id=%s channels=%d t_max=%d pf_enabled=%s",
                args.feature_id,
                int(self.env.feature_channels),
                int(self.env.spec.t_max),
                bool(args.pf_enabled),
            )
            logger.info(
                "[FEATURE] id=%s channels=%d submit_supported=%s",
                self.feature_spec.feature_id,
                int(self.feature_spec.channels),
                bool(self.feature_spec.submit_supported),
            )
            logger.info(
                "[PERF] memory_format=%s pin_memory=%s rollout_cache_device=%s estimate=%.2fGiB",
                ("channels_last" if self.use_channels_last else "nchw"),
                bool(self.use_pin_memory),
                str(self.cache_device),
                cache_gib,
            )
            logger.info(
                "[AUX] opp_param_loss_coef=%.6g use_valid_mask=%s active=%s",
                float(self.aux_opp_param_loss_coef),
                bool(self.aux_opp_param_use_valid_mask),
                bool(self.aux_opp_param_active),
            )
            if self.device.type == "cuda" and self.cache_device.type == "cpu":
                logger.warning("[PERF] rollout cache is on CPU; this can reduce GPU utilization during PPO updates")
            logger.info("[MODEL] %s", self.resolved_model_config)
            if self.resume_enabled:
                logger.info(
                    "[RESUME] enabled=%s from=%s step=%d",
                    bool(self.resume_enabled),
                    str(self.resume_from),
                    int(self.global_step),
                )

        self.rollout_workspace = create_rollout_workspace(
            self.env,
            num_steps=int(local_cfg.num_steps),
            device=self.device,
            channels_last=bool(self.use_channels_last),
            pin_memory=bool(self.use_pin_memory),
            collect_aux_targets=bool(self.aux_opp_param_active),
        )
        self.buffer = RolloutBuffer(
            num_steps=local_cfg.num_steps,
            num_envs=local_cfg.num_envs,
            obs_shape=self.obs_shape,
            action_shape=tuple(),
            device=self.cache_device,
            use_action_mask=bool(args.use_action_mask),
            action_dim=self.action_dim,
            use_aux_opp_param_targets=bool(self.aux_opp_param_active),
        )

        # Episode-seed sampler RNG is intentionally decoupled from global seed.
        self.rng = np.random.default_rng()
        if self.resume_enabled and isinstance(self.resume_meta.get("rng_state"), dict):
            try:
                self.rng.bit_generator.state = dict(self.resume_meta["rng_state"])
            except Exception:
                if self.is_main:
                    logger.warning("failed to restore rng_state from resume; continuing with fresh RNG state")

        self.train_metrics_jsonl = self.layout.logs_dir / "train_metrics.jsonl"
        self.periodic_val_jsonl = self.layout.logs_dir / "periodic_val_metrics.jsonl"
        self.best_model = self.layout.models_dir / "best.pt"
        self.last_model = self.layout.models_dir / "last.pt"
        self.checkpoint_dir = self.layout.models_dir / "checkpoints"
        completed_iterations = int(self.global_step // max(1, self.global_batch_size))
        (
            self.periodic_val_enabled,
            self.eval_at_start_enabled,
            self.best_metric_name,
            self.eval_round,
            self.next_eval_iteration,
            self.checkpoint_interval_iterations,
            self.next_checkpoint_iteration,
        ) = _resolve_eval_checkpoint_state(
            args=args,
            resume_enabled=bool(self.resume_enabled),
            completed_iterations=int(completed_iterations),
            best_metric_name=str(self.best_metric_name),
            restored_eval_round=restored_eval_round,
            restored_next_eval_iteration=restored_next_eval_iteration,
            restored_next_checkpoint_iteration=restored_next_checkpoint_iteration,
        )
        if self.checkpoint_interval_iterations > 0 and self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        vf_clip_schedule_desc = (
            "disabled" if self.schedules.clip_range_vf is None else self.schedules.clip_range_vf.description
        )
        if self.is_main:
            logger.info(
                "schedules: lr=%s ent=%s clip=%s vf_clip=%s",
                self.schedules.learning_rate.description,
                self.schedules.ent_coef.description,
                self.schedules.clip_coef.description,
                vf_clip_schedule_desc,
            )
            if bool(self.cfg_global.clip_vloss) and self.schedules.clip_range_vf is None:
                logger.warning("clip_vloss=true but clip_range_vf is unset; value clipping is disabled")

        self.start_time = time.time()
        self.start_iteration = int(completed_iterations) + 1
        if self.start_iteration > self.num_iterations and self.is_main:
            logger.info(
                "no PPO update needed: completed_iterations=%d already reached total_iterations=%d",
                int(completed_iterations),
                int(self.cfg_global.total_iterations),
            )
            logger.info(
                "resume progress: global_step=%d global_batch_size=%d",
                int(self.global_step),
                int(self.global_batch_size),
            )

    # ------------------------------------------------------------------
    # Helpers shared across stages
    # ------------------------------------------------------------------

    def _build_resume_meta(self, *, iteration: int) -> dict[str, Any]:
        """現在状態をチェックポイント保存用メタ情報へ変換する。"""
        return _build_resume_meta_payload(
            iteration=int(iteration),
            run_name=str(self.run_name),
            global_step=int(self.global_step),
            world_size=int(self.world_size),
            feature_id=str(self.args.feature_id),
            pf_enabled=bool(self.args.pf_enabled),
            obs_shape=tuple(self.obs_shape),
            action_dim=int(self.action_dim),
            use_channels_last=bool(self.use_channels_last),
            use_pin_memory=bool(self.use_pin_memory),
            cache_device=self.cache_device,
            cfg_global=self.cfg_global,
            cfg_local=self.local_cfg,
            best_metric_name=str(self.best_metric_name),
            best_metric_value=float(self.best_metric_value),
            best_metric_source=str(self.best_metric_source),
            eval_round=int(self.eval_round),
            next_eval_iteration=int(self.next_eval_iteration),
            next_checkpoint_iteration=int(self.next_checkpoint_iteration),
            rng=self.rng,
            train_vecnorm=self.train_vecnorm,
        )

    def _make_val_kwargs(self) -> dict[str, Any]:
        """定期評価呼び出しに渡す共通引数辞書を作成する。"""
        return dict(
            env_id="AHC061Local-v0",
            feature_id=str(self.args.feature_id),
            pf_enabled=bool(self.args.pf_enabled),
            agent=self.rollout_agent,
            device=self.device,
            episodes=int(self.args.eval_episodes),
            num_envs=int(getattr(self.args, "eval_num_envs", 0)),
            use_action_mask=bool(self.args.use_action_mask),
            amp=bool(self.args.amp),
            vecnorm_state=_vecnorm_state_or_none(self.train_vecnorm),
            vecnorm_norm_obs=bool(self.args.vecnorm_norm_obs),
            vecnorm_norm_reward=bool(self.args.vecnorm_eval_norm_reward),
            vecnorm_clip_obs=float(self.args.vecnorm_clip_obs),
            vecnorm_clip_reward=float(self.args.vecnorm_clip_reward),
            vecnorm_epsilon=float(self.args.vecnorm_epsilon),
            vecnorm_gamma=float(self.vecnorm_gamma),
        )

    def _eval_seed_base(self) -> int:
        """現在ラウンドで使う評価 seed の基準値を返す。"""
        if not bool(self.args.eval_fixed_seeds):
            return int(self.args.eval_seed_start) + int(self.eval_round) * int(self.args.eval_episodes)
        return int(self.args.eval_seed_start)

    # ------------------------------------------------------------------
    # Stage 5: optional evaluation before first training iteration
    # ------------------------------------------------------------------

    def _run_eval_at_start(self) -> None:
        """学習開始前評価が有効な場合に 1 回だけ評価を実行する。"""
        if not (self.eval_at_start_enabled and self.global_step == 0):
            return
        if self.is_distributed:
            dist.barrier()
        if self.is_main:
            seed_base = self._eval_seed_base()
            eval_summary = _run_periodic_val(seed_start=seed_base, **self._make_val_kwargs())
            eval_row = {
                "global_step": 0,
                "eval_round": int(self.eval_round),
                "seed_base": int(seed_base),
                "fixed_seeds": bool(self.args.eval_fixed_seeds),
                "summary": eval_summary,
            }
            _append_jsonl(self.periodic_val_jsonl, eval_row)
            logger.info(
                "periodic_val@start step=0 round=%d return=%.5f game_score=%.2f",
                int(self.eval_round),
                float(eval_summary["return"]["mean"]),
                float(eval_summary["terminal_game_score"]["mean"]),
            )
            if self.tracker is not None:
                self.tracker.log_metrics(
                    0,
                    {
                        "val/episodes": int(eval_summary["episodes"]),
                        "val/mean_return": float(eval_summary["return"]["mean"]),
                        "val/mean_terminal_game_score": float(eval_summary["terminal_game_score"]["mean"]),
                    },
                )
            cand = float(eval_summary["terminal_game_score"]["mean"])
            if np.isfinite(cand) and cand > self.best_metric_value:
                self.best_metric_value = float(cand)
                self.best_metric_source = "periodic_val_at_start"
                save_agent_checkpoint(
                    self.best_model,
                    _unwrap_model(self.agent),
                    optimizer=self.optimizer,
                    meta=self._build_resume_meta(iteration=0),
                )
        if self.is_distributed:
            dist.barrier()
        self.eval_round += 1

    # ------------------------------------------------------------------
    # Stage 6: main training loop
    # ------------------------------------------------------------------

    def _apply_runtime_schedule(self, *, iteration: int) -> RuntimeScheduleCoefficients:
        """現在イテレーションのスケジュール係数を optimizer/trainer に反映する。"""
        runtime = self.runtime_schedules.resolve(iteration=int(iteration))
        self.optimizer.param_groups[0]["lr"] = float(runtime.learning_rate)
        self.trainer.set_runtime_coefficients(
            ent_coef=float(runtime.ent_coef),
            clip_coef=float(runtime.clip_coef),
            clip_range_vf=(None if runtime.clip_range_vf is None else float(runtime.clip_range_vf)),
            aux_opp_param_loss_coef=float(self.aux_opp_param_loss_coef),
            aux_opp_param_use_valid_mask=bool(self.aux_opp_param_use_valid_mask),
        )
        return runtime

    def _collect_rollout_and_update(self) -> tuple[UpdateStats, float]:
        """ロールアウト収集から PPO 更新までを 1 回実行する。"""
        args = self.args
        local_cfg = self.local_cfg
        seeds = torch.as_tensor(
            self.rng.integers(
                int(self.train_seed_min),
                int(self.train_seed_max_exclusive),
                size=(local_cfg.num_envs,),
                dtype=np.int64,
            ),
            dtype=torch.int64,
            device="cpu",
        )
        self.env.reset_random(seeds)

        rollout = collect_rollout(
            self.env,
            self.rollout_agent,
            self.device,
            local_cfg.num_steps,
            use_action_mask=bool(args.use_action_mask),
            sample=True,
            amp=bool(args.amp),
            channels_last=bool(self.use_channels_last),
            workspace=self.rollout_workspace,
            pin_memory=bool(self.use_pin_memory),
            vecnorm=self.train_vecnorm,
            collect_aux_targets=bool(self.aux_opp_param_active),
        )

        copy_non_blocking = bool(self.cache_device.type == "cuda")
        self.buffer.obs.copy_(
            rollout.obs.to(device=self.cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
        )
        self.buffer.actions.copy_(
            rollout.actions.to(device=self.cache_device, dtype=torch.long, non_blocking=copy_non_blocking)
        )
        self.buffer.logprobs.copy_(
            rollout.logprobs.to(device=self.cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
        )
        self.buffer.rewards.copy_(
            rollout.rewards.to(device=self.cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
        )
        self.buffer.dones.copy_(
            rollout.dones.to(device=self.cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
        )
        self.buffer.values.copy_(
            rollout.values.to(device=self.cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
        )
        if self.buffer.action_masks is not None:
            self.buffer.action_masks.copy_(
                rollout.masks.to(device=self.cache_device, dtype=torch.bool, non_blocking=copy_non_blocking)
            )
        if bool(self.aux_opp_param_active):
            if rollout.aux_opp_param_true is None or rollout.aux_opp_valid is None:
                raise RuntimeError("rollout did not provide aux_opp_param targets")
            if self.buffer.aux_opp_param_true is None or self.buffer.aux_opp_valid is None:
                raise RuntimeError("rollout buffer is missing aux_opp_param storage")
            self.buffer.aux_opp_param_true.copy_(
                rollout.aux_opp_param_true.to(
                    device=self.cache_device,
                    dtype=torch.float32,
                    non_blocking=copy_non_blocking,
                )
            )
            self.buffer.aux_opp_valid.copy_(
                rollout.aux_opp_valid.to(
                    device=self.cache_device,
                    dtype=torch.bool,
                    non_blocking=copy_non_blocking,
                )
            )

        next_value = rollout.last_value.to(
            device=self.cache_device, dtype=torch.float32, non_blocking=copy_non_blocking
        )
        next_done = rollout.last_done.to(device=self.cache_device, dtype=torch.float32, non_blocking=copy_non_blocking)
        self.buffer.compute_gae(next_value, next_done, local_cfg.gamma, local_cfg.gae_lambda)
        explained_variance_local = _explained_variance(self.buffer.values, self.buffer.returns)
        stats = self.trainer.update(self.buffer)
        return stats, float(explained_variance_local)

    def _aggregate_iteration_stats(
        self, *, stats: UpdateStats, explained_variance_local: float
    ) -> _IterationAggregateStats:
        """ローカル更新統計を rank 間集約して記録用指標へ整形する。"""
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
            device=self.device,
            dtype=torch.float64,
        )
        _all_reduce_sum_tensor(stats_vec)
        ws = float(max(1, int(self.world_size)))
        policy_loss = float((stats_vec[0] / ws).item())
        value_loss = float((stats_vec[1] / ws).item())
        entropy = float((stats_vec[2] / ws).item())
        approx_kl = float((stats_vec[3] / ws).item())
        clipfrac = float((stats_vec[4] / ws).item())
        value_clipfrac = float("nan") if float(stats_vec[6].item()) >= ws else float((stats_vec[5] / ws).item())
        explained_variance = (
            float("nan") if float(stats_vec[8].item()) <= 0.0 else float((stats_vec[7] / stats_vec[8]).item())
        )
        aux_opp_param_loss = (
            float("nan") if float(stats_vec[10].item()) <= 0.0 else float((stats_vec[9] / stats_vec[10]).item())
        )

        score_now = self.env.official_score().to(dtype=torch.float32)
        metric_vec = torch.tensor(
            [float(score_now.sum().item()), float(score_now.numel())],
            device=self.device,
            dtype=torch.float64,
        )
        _all_reduce_sum_tensor(metric_vec)
        mean_official_score = (
            float((metric_vec[0] / metric_vec[1]).item()) if float(metric_vec[1].item()) > 0.0 else 0.0
        )
        return _IterationAggregateStats(
            policy_loss=float(policy_loss),
            value_loss=float(value_loss),
            entropy=float(entropy),
            approx_kl=float(approx_kl),
            clipfrac=float(clipfrac),
            value_clipfrac=float(value_clipfrac),
            explained_variance=float(explained_variance),
            aux_opp_param_loss=float(aux_opp_param_loss),
            mean_official_score=float(mean_official_score),
        )

    def _advance_global_step_and_measure_sps(self) -> int:
        """`global_step` を進め、経過時間から SPS を算出する。"""
        self.global_step += int(self.global_batch_size)
        _sync_vecnorm_ddp_(self.train_vecnorm, device=self.device)
        elapsed_vec = torch.tensor([max(1e-9, time.time() - self.start_time)], device=self.device, dtype=torch.float64)
        _all_reduce_max_tensor(elapsed_vec)
        elapsed = float(elapsed_vec[0].item())
        return int(float(self.global_step) / max(1e-9, elapsed))

    def _build_train_row(
        self,
        *,
        iteration: int,
        sps: int,
        runtime: RuntimeScheduleCoefficients,
        agg: _IterationAggregateStats,
    ) -> dict[str, Any]:
        """学習ログ 1 行分のメトリクス辞書を組み立てる。"""
        return {
            "iteration": int(iteration),
            "global_step": int(self.global_step),
            "sps": int(sps),
            "learning_rate": float(runtime.learning_rate),
            "schedule_progress": float(runtime.progress),
            "ent_coef_current": float(runtime.ent_coef),
            "clip_coef_current": float(runtime.clip_coef),
            "clip_range_vf_current": (
                float(runtime.clip_range_vf) if runtime.clip_range_vf is not None else float("nan")
            ),
            "policy_loss": float(agg.policy_loss),
            "value_loss": float(agg.value_loss),
            "entropy": float(agg.entropy),
            "approx_kl": float(agg.approx_kl),
            "clipfrac": float(agg.clipfrac),
            "value_clipfrac": float(agg.value_clipfrac),
            "explained_variance": float(agg.explained_variance),
            "aux_opp_param_loss": float(agg.aux_opp_param_loss),
            "mean_official_score": float(agg.mean_official_score),
        }

    def _maybe_run_periodic_eval(self, *, iteration: int, row: dict[str, Any], mean_official: float) -> None:
        """必要なタイミングで定期評価とベストモデル更新を実施する。"""
        args = self.args
        run_periodic_eval = bool(self.periodic_val_enabled and int(iteration) >= int(self.next_eval_iteration))
        if run_periodic_eval and self.is_distributed:
            dist.barrier()
        if run_periodic_eval and self.is_main:
            seed_base = self._eval_seed_base()
            eval_summary = _run_periodic_val(seed_start=seed_base, **self._make_val_kwargs())
            eval_row = {
                "iteration": int(iteration),
                "global_step": int(self.global_step),
                "eval_round": int(self.eval_round),
                "seed_base": int(seed_base),
                "fixed_seeds": bool(args.eval_fixed_seeds),
                "summary": eval_summary,
            }
            _append_jsonl(self.periodic_val_jsonl, eval_row)
            row["periodic_val_mean_return"] = float(eval_summary["return"]["mean"])
            row["periodic_val_episodes"] = int(eval_summary["episodes"])
            row["periodic_val_mean_terminal_game_score"] = float(eval_summary["terminal_game_score"]["mean"])
            logger.info(
                "periodic_val step=%d round=%d episodes=%d return=%.5f game_score=%.2f",
                int(self.global_step),
                int(self.eval_round),
                int(eval_summary["episodes"]),
                float(eval_summary["return"]["mean"]),
                float(eval_summary["terminal_game_score"]["mean"]),
            )
            cand = float(eval_summary["terminal_game_score"]["mean"])
            if np.isfinite(cand) and cand > self.best_metric_value:
                self.best_metric_value = float(cand)
                self.best_metric_source = "periodic_val"
                save_agent_checkpoint(
                    self.best_model,
                    _unwrap_model(self.agent),
                    optimizer=self.optimizer,
                    meta=self._build_resume_meta(iteration=int(iteration)),
                )
        if run_periodic_eval and self.is_distributed:
            dist.barrier()
        if run_periodic_eval:
            self.eval_round += 1
            self.next_eval_iteration += int(max(1, args.eval_interval_iterations))
        elif self.is_main and not self.periodic_val_enabled:
            cand = float(mean_official)
            if np.isfinite(cand) and cand > self.best_metric_value:
                self.best_metric_value = float(cand)
                self.best_metric_source = "mean_official_score"
                save_agent_checkpoint(
                    self.best_model,
                    _unwrap_model(self.agent),
                    optimizer=self.optimizer,
                    meta=self._build_resume_meta(iteration=int(iteration)),
                )

    def _record_and_log_iteration(
        self,
        *,
        iteration: int,
        sps: int,
        runtime: RuntimeScheduleCoefficients,
        agg: _IterationAggregateStats,
        row: dict[str, Any],
    ) -> None:
        """イテレーション結果を JSONL / tracker / ログへ出力する。"""
        args = self.args
        if self.is_main:
            _append_jsonl(self.train_metrics_jsonl, row)
            if self.tracker is not None:
                tracker_metrics: dict[str, float | int] = {}
                for k, v in row.items():
                    if k == "global_step":
                        continue
                    metric_key = f"val/{k[len('periodic_val_') :]}" if k.startswith("periodic_val_") else f"train/{k}"
                    tracker_metrics[metric_key] = v
                self.tracker.log_metrics(step=int(self.global_step), metrics=tracker_metrics)

        if self.is_main and args.log_interval_iters > 0 and (iteration % int(args.log_interval_iters) == 0):
            logger.info(
                "iter=%d/%d step=%d sps=%d lr=%.6g ploss=%.5f vloss=%.5f aux=%.5f ent=%.5f kl=%.5f clip=%.4f ev=%.5f vclip=%.5f score=%.1f",
                iteration,
                self.num_iterations,
                self.global_step,
                sps,
                float(runtime.learning_rate),
                float(agg.policy_loss),
                float(agg.value_loss),
                float(agg.aux_opp_param_loss),
                float(agg.entropy),
                float(agg.approx_kl),
                float(agg.clipfrac),
                float(agg.explained_variance),
                float(agg.value_clipfrac),
                float(agg.mean_official_score),
            )

    def _maybe_save_iteration_checkpoints(self, *, iteration: int) -> None:
        """`last.pt` と定期間隔チェックポイントを必要に応じて保存する。"""
        local_cfg = self.local_cfg
        if self.is_main and (iteration % int(max(1, local_cfg.save_interval)) == 0 or iteration == self.num_iterations):
            save_agent_checkpoint(
                self.last_model,
                _unwrap_model(self.agent),
                optimizer=self.optimizer,
                meta=self._build_resume_meta(iteration=int(iteration)),
            )

        if (
            self.is_main
            and self.checkpoint_interval_iterations > 0
            and int(iteration) >= int(self.next_checkpoint_iteration)
        ):
            save_agent_checkpoint(
                self.checkpoint_dir / f"iter_{int(iteration):08d}.pt",
                _unwrap_model(self.agent),
                optimizer=self.optimizer,
                meta=self._build_resume_meta(iteration=int(iteration)),
            )
            self.next_checkpoint_iteration += int(self.checkpoint_interval_iterations)

    def _main_training_loop(self) -> None:
        """PPO 更新ループ本体を実行する。"""
        for iteration in range(self.start_iteration, self.num_iterations + 1):
            runtime = self._apply_runtime_schedule(iteration=int(iteration))
            update_stats, explained_variance_local = self._collect_rollout_and_update()
            agg = self._aggregate_iteration_stats(
                stats=update_stats,
                explained_variance_local=float(explained_variance_local),
            )
            sps = self._advance_global_step_and_measure_sps()
            row = self._build_train_row(
                iteration=int(iteration),
                sps=int(sps),
                runtime=runtime,
                agg=agg,
            )
            self._maybe_run_periodic_eval(
                iteration=int(iteration), row=row, mean_official=float(agg.mean_official_score)
            )
            self._record_and_log_iteration(
                iteration=int(iteration),
                sps=int(sps),
                runtime=runtime,
                agg=agg,
                row=row,
            )
            self._maybe_save_iteration_checkpoints(iteration=int(iteration))

    # ------------------------------------------------------------------
    # Stage 7: finalization
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        """最終チェックポイント・サマリ・manifest を書き出して終了する。"""
        args = self.args
        if self.is_main:
            final_iteration = int(
                max(0, min(self.num_iterations, int(self.global_step // max(1, self.global_batch_size))))
            )
            save_agent_checkpoint(
                self.last_model,
                _unwrap_model(self.agent),
                optimizer=self.optimizer,
                meta=self._build_resume_meta(iteration=final_iteration),
            )
            self.summary = {
                "run_name": self.run_name,
                "run_dir": str(self.layout.root),
                "global_step": int(self.global_step),
                "final_iteration": int(final_iteration),
                "env_id": "AHC061Local-v0",
                "feature_id": str(args.feature_id),
                "pf_enabled": bool(args.pf_enabled),
                "train_seed_range": {
                    "min": int(self.train_seed_min),
                    "max_exclusive": int(self.train_seed_max_exclusive),
                },
                "aux_opp_param": {
                    "loss_coef": float(self.aux_opp_param_loss_coef),
                    "use_valid_mask": bool(self.aux_opp_param_use_valid_mask),
                    "active": bool(self.aux_opp_param_active),
                },
                "vecnormalize": {
                    "enabled": bool(self.train_vecnorm is not None),
                    "norm_obs": bool(args.vecnorm_norm_obs),
                    "norm_reward_train": bool(args.vecnorm_norm_reward),
                    "norm_reward_val": bool(args.vecnorm_eval_norm_reward),
                    "clip_obs": float(args.vecnorm_clip_obs),
                    "clip_reward": float(args.vecnorm_clip_reward),
                    "epsilon": float(args.vecnorm_epsilon),
                    "gamma": float(self.vecnorm_gamma),
                },
                "periodic_val": {
                    "enabled": bool(self.periodic_val_enabled),
                    "interval_iterations": int(args.eval_interval_iterations),
                    "episodes": int(args.eval_episodes),
                    "num_envs": int(getattr(args, "eval_num_envs", 0)),
                    "seed_start": int(args.eval_seed_start),
                    "fixed_seeds": bool(args.eval_fixed_seeds),
                    "val_at_start": bool(args.eval_at_start),
                    "metrics_jsonl": str(self.periodic_val_jsonl),
                },
                "resume": {
                    "enabled": bool(self.resume_enabled),
                    "resume_from": (str(self.resume_from) if self.resume_from is not None else ""),
                },
                "best_metric": {
                    "name": str(self.best_metric_name),
                    "value": float(self.best_metric_value),
                    "source": str(self.best_metric_source),
                },
                "models": {
                    "best": str(self.best_model),
                    "last": str(self.last_model),
                    "checkpoint_dir": (str(self.checkpoint_dir) if self.checkpoint_interval_iterations > 0 else ""),
                    "checkpoint_interval_iterations": int(self.checkpoint_interval_iterations),
                },
                "logs": {
                    "train_metrics_jsonl": str(self.train_metrics_jsonl),
                    "periodic_val_metrics_jsonl": str(self.periodic_val_jsonl),
                },
                "elapsed_sec": float(time.time() - self.start_time),
            }
            (self.layout.reports_dir / "train_summary.json").write_text(
                json.dumps(self.summary, indent=2), encoding="utf-8"
            )
            (self.layout.root / "summary.json").write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
            update_manifest(
                self.layout,
                {
                    "status": "completed",
                    "result": self.summary,
                },
            )
            if self.tracker is not None:
                self.tracker.log_event(
                    "train_complete",
                    {
                        "global_step": int(self.global_step),
                        "best_metric_name": str(self.best_metric_name),
                        "best_metric_value": float(self.best_metric_value),
                    },
                )
        if self.is_distributed:
            dist.barrier()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> int:
        """PPO 学習ステージを順番に実行して終了コードを返す。

        Returns:
            int: プロセス終了コード。
        """
        self._setup_device_and_dist()
        self._setup_run_dir()
        self._build_env_and_model()
        self._restore_weights_and_infra()
        self._run_eval_at_start()
        self._main_training_loop()
        self._finalize()
        return 0


def run_ppo(args: PPORequest) -> int:
    """PPO 学習のトップレベル実行関数。

    分散設定・バッチ分割・スケジュールを解決したうえで `PPORunner` を起動する。

    Args:
        args (PPORequest): 学習実行要求。

    Returns:
        int: プロセス終了コード。
    """
    cfg_global = args_to_cfg(args)
    train_seed_min, train_seed_max_exclusive = _resolve_train_seed_range(args)
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

    runner = PPORunner(
        args,
        cfg_global=cfg_global,
        local_cfg=local_cfg,
        schedules=schedules,
        vecnorm_gamma=float(vecnorm_gamma),
        dist_mode=str(dist_mode),
        world_size=int(world_size),
        rank=int(rank),
        local_rank=int(local_rank),
        local_num_envs=int(local_num_envs),
        local_num_minibatches=int(local_num_minibatches),
        train_seed_min=int(train_seed_min),
        train_seed_max_exclusive=int(train_seed_max_exclusive),
        aux_opp_param_loss_coef=float(aux_opp_param_loss_coef),
        aux_opp_param_use_valid_mask=bool(aux_opp_param_use_valid_mask),
        aux_opp_param_active=bool(aux_opp_param_active),
    )
    try:
        return runner.run()
    except Exception as e:
        if runner.layout is not None and runner.is_main:
            update_manifest(
                runner.layout,
                {
                    "status": "failed",
                    "error": str(e),
                    "progress": {"global_step": int(runner.global_step)},
                },
            )
        raise
    finally:
        if runner.is_main and runner.tracker is not None:
            runner.tracker.close()
        if _dist_ready():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
