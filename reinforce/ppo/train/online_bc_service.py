"""オンライン BC（教師分布蒸留）を実行する学習サービス。"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from ..env import BatchEnv
from ..models import (
    build_agent,
    get_model_config_from_preset,
    load_model_config_from_sources,
    normalize_model_config,
)
from ..pipeline.model_checkpoint_service import load_agent_checkpoint, save_agent_checkpoint
from ..utils.experiment import coerce_optional_path, create_run_layout, make_run_name, to_jsonable, update_manifest
from ..utils.log_utils import get_logger
from ..utils.tracking import MetricTracker

logger = get_logger("online_bc")


@dataclass
class OnlineBCConfig:
    """online BC 実行に必要な設定を束ねるデータクラス。

    Attributes:
        env_id (str): 実行環境 ID。
        feature_id (str): 観測特徴量 ID。
        pf_enabled (bool): PF ベイズ推定の有効化フラグ。
        use_action_mask (bool): 合法手マスク利用フラグ。
        amp (bool): AMP 利用フラグ。
        teacher_model_path (Path): 教師モデルチェックポイント。
        model_class (str): 学生モデル種別。
        model_config_file (Path | None): 学生モデル設定ファイル。
        model_config_json (str): 学生モデル設定 JSON。
        model_preset (str): 学生モデル preset 名。
        init_model (Path | None): 学生初期重み。
        output_model (Path): 最終保存先。
        num_envs (int): 同時環境数。
        num_steps (int): 1 反復あたりロールアウト長。
        total_iterations (int): 学習反復回数。
        learning_rate (float): 学習率。
        weight_decay (float): Weight decay。
        num_minibatches (int): ミニバッチ分割数。
        max_grad_norm (float): 勾配クリップ上限。
        temperature (float): 教師分布温度。
        log_interval_iters (int): ログ出力間隔。
        seed (int): 学習乱数シード。
        seed_min (int): 生成シード最小値。
        seed_max_exclusive (int): 生成シード上限（排他）。
        run_root (Path | None): 実験ラン保存ルート。
        run_name (str): ラン名。
    """

    # Environment
    env_id: str
    feature_id: str
    pf_enabled: bool
    use_action_mask: bool
    amp: bool
    # Teacher (frozen)
    teacher_model_path: Path
    # Student
    model_class: str
    model_config_file: Path | None
    model_config_json: str
    model_preset: str
    init_model: Path | None
    output_model: Path
    # Rollout
    num_envs: int
    num_steps: int
    # Training
    total_iterations: int
    learning_rate: float
    weight_decay: float
    num_minibatches: int
    max_grad_norm: float
    temperature: float
    log_interval_iters: int
    seed: int
    seed_min: int
    seed_max_exclusive: int
    # Run management
    run_root: Path | None
    run_name: str

    @property
    def batch_size(self) -> int:
        """`batch_size` を実行する。

        Returns:
            int: 計算結果。
        """
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """`minibatch_size` を実行する。

        Returns:
            int: 計算結果。
        """
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        """`num_iterations` を実行する。

        Returns:
            int: 計算結果。
        """
        return int(self.total_iterations)


def _resolve_student_model_config(cfg: OnlineBCConfig) -> dict[str, Any]:
    """内部ヘルパー: `resolve_student_model_config` を実行する。

    Args:
        cfg (OnlineBCConfig): 設定オブジェクト。

    Returns:
        dict[str, Any]: 計算結果。
    """
    preset_id = str(cfg.model_preset).strip()
    preset_cfg: dict[str, Any] | None = None
    if preset_id:
        preset_cfg = get_model_config_from_preset(preset_id)

    explicit_cfg = load_model_config_from_sources(
        model_config_file=cfg.model_config_file,
        model_config_json=cfg.model_config_json,
    )
    base_cfg = explicit_cfg if explicit_cfg is not None else preset_cfg
    model_class = str(cfg.model_class).strip()
    if base_cfg is not None and model_class:
        base_cfg["type"] = model_class

    return normalize_model_config(
        base_cfg,
        default_type=model_class or "DiscreteBoardAgent",
    )


def _resolve_seed_range(cfg: OnlineBCConfig) -> tuple[int, int]:
    """内部ヘルパー: `resolve_seed_range` を実行する。

    Args:
        cfg (OnlineBCConfig): 設定オブジェクト。

    Returns:
        tuple[int, int]: 計算結果。
    """
    i64_max = int(np.iinfo(np.int64).max)
    seed_min = int(cfg.seed_min)
    seed_max_exclusive = int(cfg.seed_max_exclusive)
    if seed_min < 0:
        raise ValueError(f"seed_min must be >= 0, got {seed_min}")
    if seed_max_exclusive <= seed_min:
        raise ValueError(
            "seed_max_exclusive must be greater than seed_min: "
            f"{seed_max_exclusive} <= {seed_min}"
        )
    if seed_max_exclusive > i64_max:
        raise ValueError(f"seed_max_exclusive must be <= int64 max ({i64_max}), got {seed_max_exclusive}")
    return seed_min, seed_max_exclusive


def _prepare_run(cfg: OnlineBCConfig) -> tuple[Any, MetricTracker | None]:
    """内部ヘルパー: `prepare_run` を実行する。

    Args:
        cfg (OnlineBCConfig): 設定オブジェクト。

    Returns:
        tuple[Any, MetricTracker | None]: 計算結果。
    """
    if cfg.run_root is None:
        return None, None

    run_name = cfg.run_name or make_run_name("online_bc", seed=cfg.seed)
    layout = create_run_layout(cfg.run_root, run_name)

    config_snapshot = to_jsonable({"config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}})
    (layout.config_dir / "online_bc.args.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    tracker = MetricTracker(layout.root, run_name=run_name, config=config_snapshot)
    update_manifest(
        layout,
        {
            "job": "online_bc",
            "status": "running",
            "run_name": run_name,
            "layout": layout.as_dict(),
            "config": config_snapshot,
            "timestamps": {"started_at": time.time()},
        },
    )
    return layout, tracker


def run_online_bc(cfg: OnlineBCConfig) -> None:
    """online BC 学習フローを実行する。

    準備（ラン管理）・本体学習・完了/失敗時後処理までを担当する。

    Args:
        cfg (OnlineBCConfig): online BC 実行設定。
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    layout = None
    tracker: MetricTracker | None = None
    try:
        layout, tracker = _prepare_run(cfg)
        _run_online_bc(cfg, layout=layout, tracker=tracker)
        if tracker is not None and layout is not None:
            update_manifest(layout, {"status": "completed", "timestamps": {"finished_at": time.time()}})
    except Exception as e:
        if tracker is not None and layout is not None:
            update_manifest(layout, {"status": "failed", "error": str(e), "timestamps": {"failed_at": time.time()}})
            tracker.log_event("online_bc_failed", {"error": str(e)})
        raise
    finally:
        if tracker is not None:
            tracker.close()


def _run_online_bc(cfg: OnlineBCConfig, *, layout: Any, tracker: MetricTracker | None) -> None:
    """teacher/student を用いたオンライン蒸留ループを実行する。

    Args:
        cfg (OnlineBCConfig): 学習設定。
        layout (Any): ランディレクトリ情報。
        tracker (MetricTracker | None): 実験トラッカー。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.amp and device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    logger.info("device=%s amp=%s", device, use_amp)

    # Load teacher (frozen)
    logger.info("loading teacher from %s", cfg.teacher_model_path)
    teacher, _teacher_meta = load_agent_checkpoint(cfg.teacher_model_path, device=device)
    teacher.eval()
    teacher.requires_grad_(False)

    obs_shape = tuple(teacher.obs_shape)
    action_dim = int(teacher.action_dim)
    logger.info("teacher obs_shape=%s action_dim=%d", obs_shape, action_dim)

    # Build student
    model_cfg = _resolve_student_model_config(cfg)
    if cfg.init_model is not None:
        logger.info("loading student init from %s", cfg.init_model)
        student, _meta = load_agent_checkpoint(cfg.init_model, device=device)
        if tuple(student.obs_shape) != obs_shape:
            raise ValueError(f"init_model obs_shape mismatch: student={student.obs_shape} teacher={obs_shape}")
        if int(student.action_dim) != action_dim:
            raise ValueError(f"init_model action_dim mismatch: student={student.action_dim} teacher={action_dim}")
    else:
        student, model_cfg = build_agent(obs_shape=obs_shape, action_dim=action_dim, model_config=model_cfg)
        student.to(device)

    student.train()

    # Validate config
    if int(cfg.total_iterations) <= 0:
        raise ValueError(f"total_iterations must be > 0, got {cfg.total_iterations}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {cfg.batch_size}")
    if cfg.batch_size % cfg.num_minibatches != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by num_minibatches ({cfg.num_minibatches})"
        )
    if cfg.minibatch_size < 1:
        raise ValueError(f"minibatch_size must be >= 1, got {cfg.minibatch_size}")

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Environment
    logger.info(
        "creating BatchEnv num_envs=%d feature_id=%s pf_enabled=%s",
        cfg.num_envs, cfg.feature_id, cfg.pf_enabled,
    )
    env = BatchEnv(
        batch_size=cfg.num_envs,
        feature_id=str(cfg.feature_id),
        pf_enabled=bool(cfg.pf_enabled),
    )
    c = int(env.feature_channels)
    n = int(env.board_size)
    a = int(env.action_dim)
    env_obs_shape = (c, n, n)
    if env_obs_shape != obs_shape:
        raise ValueError(f"env obs_shape {env_obs_shape} does not match teacher obs_shape {obs_shape}")
    if a != action_dim:
        raise ValueError(f"env action_dim {a} does not match teacher action_dim {action_dim}")

    B = cfg.num_envs
    T = cfg.num_steps
    A = action_dim
    if T > int(env.spec.t_max):
        raise ValueError(f"num_steps must be <= env.spec.t_max ({env.spec.t_max}), got {T}")
    seed_min, seed_max_exclusive = _resolve_seed_range(cfg)
    use_cuda = device.type == "cuda"
    pin = use_cuda

    # Pre-allocate rollout buffers on CPU (pinned for fast H2D transfer)
    obs_buf = torch.zeros(T, B, *obs_shape, dtype=torch.float32, pin_memory=pin)
    mask_buf = torch.zeros(T, B, A, dtype=torch.uint8, pin_memory=pin)
    tlogit_buf = torch.zeros(T, B, A, dtype=torch.float32, pin_memory=pin)

    # Working tensors (CPU env output)
    board = torch.empty(B, *obs_shape, dtype=torch.float32)
    mask = torch.empty(B, A, dtype=torch.uint8)
    reward = torch.empty(B, dtype=torch.float32)
    done = torch.empty(B, dtype=torch.uint8)

    # GPU mirrors of board and mask (used when CUDA is available)
    board_dev = torch.empty(B, *obs_shape, dtype=torch.float32, device=device) if use_cuda else board
    mask_dev = torch.empty(B, A, dtype=torch.uint8, device=device) if use_cuda else mask

    num_iterations = int(cfg.num_iterations)
    T_temp = float(cfg.temperature)
    total_transitions_done = 0
    iter_kl_losses: list[float] = []
    t_start = time.time()
    shuffle_rng = np.random.default_rng(int(cfg.seed))
    # Episode-seed sampler RNG is intentionally decoupled from global seed.
    seed_sampler = np.random.default_rng()

    logger.info(
        "starting online BC: total_iterations=%d num_envs=%d num_steps=%d "
        "num_minibatches=%d minibatch_size=%d num_iterations=%d temperature=%.3f",
        int(cfg.total_iterations),
        B,
        T,
        cfg.num_minibatches,
        cfg.minibatch_size,
        num_iterations,
        cfg.temperature,
    )

    for iteration in range(num_iterations):
        rollout_seeds = torch.as_tensor(
            seed_sampler.integers(
                int(seed_min),
                int(seed_max_exclusive),
                size=(B,),
                dtype=np.int64,
            ),
            dtype=torch.int64,
            device="cpu",
        )
        env.reset_random(rollout_seeds)
        env.observe_into(board, mask)
        if use_cuda:
            board_dev.copy_(board)
            mask_dev.copy_(mask)

        # -------------------------------------------------------
        # Phase 1: Rollout with frozen teacher (no grad)
        # -------------------------------------------------------
        with torch.inference_mode():
            for t in range(T):
                obs_in = board_dev if use_cuda else board
                mask_in = mask_dev if use_cuda else mask

                with autocast_ctx:
                    teacher_logits = teacher.get_logits(obs_in)  # [B, A]

                # Mask logits for sampling
                if bool(cfg.use_action_mask):
                    sample_logits = teacher_logits.masked_fill(~mask_in.bool(), -1e9)
                else:
                    sample_logits = teacher_logits

                action_dev = Categorical(logits=sample_logits).sample()  # [B]
                action_cpu = action_dev.to("cpu")

                # Store pre-step obs, mask, and teacher logits
                obs_buf[t].copy_(board)
                mask_buf[t].copy_(mask)
                tlogit_buf[t].copy_(teacher_logits.to("cpu", non_blocking=True))

                # Step environment (board and mask are updated with next obs in-place)
                env.step_observe_into(action_cpu, board, mask, reward, done)

                if use_cuda:
                    board_dev.copy_(board, non_blocking=True)
                    mask_dev.copy_(mask, non_blocking=True)

            total_transitions_done += B * T

        # -------------------------------------------------------
        # Phase 2: BC update (student learns teacher's distribution)
        # -------------------------------------------------------
        flat_obs = obs_buf.view(T * B, *obs_shape)
        flat_mask = mask_buf.view(T * B, A)
        flat_tlogit = tlogit_buf.view(T * B, A)

        indices = shuffle_rng.permutation(T * B)
        mb_size = cfg.minibatch_size
        iter_kl = 0.0
        n_mb = 0

        student.train()
        for start in range(0, T * B, mb_size):
            mb_idx = indices[start : start + mb_size]
            mb_obs = flat_obs[mb_idx].to(device=device, non_blocking=True)
            mb_mask = flat_mask[mb_idx].to(device=device, non_blocking=True)
            mb_tlogit = flat_tlogit[mb_idx].to(device=device, non_blocking=True)

            with autocast_ctx:
                student_logits = student.get_logits(mb_obs)  # [mb, A]

            # Apply action mask to both distributions; apply temperature to teacher
            if bool(cfg.use_action_mask):
                mb_mask_bool = mb_mask.bool()
                teacher_logits_m = mb_tlogit.masked_fill(~mb_mask_bool, -1e8)
                student_logits_m = student_logits.masked_fill(~mb_mask_bool, -1e8)
            else:
                teacher_logits_m = mb_tlogit
                student_logits_m = student_logits

            # KL(teacher || student) = sum(p_t * log(p_t / p_s))
            kl_loss = F.kl_div(
                F.log_softmax(student_logits_m, dim=-1),
                F.softmax(teacher_logits_m / T_temp, dim=-1).detach(),
                reduction="batchmean",
                log_target=False,
            )

            optimizer.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
            optimizer.step()

            iter_kl += float(kl_loss.item())
            n_mb += 1

        iter_kl_mean = iter_kl / max(1, n_mb)
        iter_kl_losses.append(iter_kl_mean)

        if tracker is not None:
            tracker.log_metrics(iteration + 1, {"losses/kl": iter_kl_mean})

        if cfg.log_interval_iters > 0 and (iteration + 1) % cfg.log_interval_iters == 0:
            recent = iter_kl_losses[-cfg.log_interval_iters :]
            elapsed = max(1e-9, time.time() - t_start)
            logger.info(
                "iter=%d/%d transitions=%d kl_loss=%.6f elapsed_sec=%.1f",
                iteration + 1,
                num_iterations,
                total_transitions_done,
                float(np.mean(recent)),
                elapsed,
            )

    # Save final student model
    cfg.output_model.parent.mkdir(parents=True, exist_ok=True)
    final_kl = float(np.mean(iter_kl_losses[-10:])) if iter_kl_losses else float("nan")
    meta = {
        "bc_type": "online_kl_distillation",
        "teacher_model_path": str(cfg.teacher_model_path),
        "total_iterations": int(cfg.total_iterations),
        "total_transitions": total_transitions_done,
        "num_iterations": num_iterations,
        "final_kl_loss": final_kl,
        "temperature": float(cfg.temperature),
        "seed": int(cfg.seed),
        "seed_min": int(seed_min),
        "seed_max_exclusive": int(seed_max_exclusive),
    }
    save_agent_checkpoint(cfg.output_model, student, meta=meta)
    logger.info("saved student model: %s (kl_loss=%.6f)", cfg.output_model, final_kl)

    if tracker is not None and layout is not None:
        summary_path = layout.reports_dir / "online_bc_summary.json"
        summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tracker.log_event("online_bc_complete", {"output_model": str(cfg.output_model), "final_kl_loss": final_kl})
