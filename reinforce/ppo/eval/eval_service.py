"""バッチ環境でのポリシー評価処理を提供するサービス層。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ..env import BatchEnv
from ..ppo.vecnorm import VecNormalize


@dataclass
class EpisodeStats:
    """評価エピソードを集計した結果コンテナ。

    Attributes:
        obs_shape (tuple[int, ...]): 観測テンソル形状 `(C, N, N)`。
        action_dim (int): 行動数。
        episode_returns (list[float]): 各エピソードの累積報酬。
        episode_lengths (list[int]): 各エピソードの長さ。
        episode_terminal_game_scores (list[float]): 各エピソードの最終公式スコア。
        episode_m (list[int]): 問題インスタンスの M 値。
        episode_u (list[int]): 問題インスタンスの U 値。
        episode_game_score_self (list[float]): 自チーム側スコア。
        episode_game_score_enemy_max (list[float]): 相手最大スコア。
        aux_move_dist (torch.Tensor | None): 収集した行動分布教師信号。
        aux_opp_param (torch.Tensor | None): 収集した相手推定教師信号。
        aux_opp_valid (torch.Tensor | None): 補助教師信号の有効マスク。
    """
    obs_shape: tuple[int, ...]
    action_dim: int
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_terminal_game_scores: list[float]
    episode_m: list[int]
    episode_u: list[int]
    episode_game_score_self: list[float]
    episode_game_score_enemy_max: list[float]
    aux_move_dist: torch.Tensor | None = None  # [E, T, M_MAX, A]
    aux_opp_param: torch.Tensor | None = None  # [E, T, M_MAX, 5]
    aux_opp_valid: torch.Tensor | None = None  # [E, T, M_MAX]


def _resolve_batch_size(*, episodes: int, num_envs: int) -> int:
    """評価エピソード数と `num_envs` から実行バッチサイズを決める。"""
    if int(episodes) <= 0:
        return 0
    requested = int(num_envs)
    if requested <= 0:
        requested = int(episodes)
    return int(max(1, min(int(episodes), requested)))


def _build_eval_vecnorm(
    *,
    batch_size: int,
    obs_shape: tuple[int, ...],
    enabled: bool,
    state: dict[str, Any] | None,
    norm_obs: bool,
    norm_reward: bool,
    clip_obs: float,
    clip_reward: float,
    epsilon: float,
    gamma: float,
) -> VecNormalize | None:
    """評価専用 `VecNormalize` を構築し、必要なら状態を復元する。"""
    if not bool(enabled):
        return None
    vecnorm = VecNormalize(
        num_envs=int(batch_size),
        obs_shape=obs_shape,
        norm_obs=bool(norm_obs),
        norm_reward=bool(norm_reward),
        clip_obs=float(clip_obs),
        clip_reward=float(clip_reward),
        epsilon=float(epsilon),
        gamma=float(gamma),
        training=False,
    )
    if isinstance(state, dict):
        vecnorm.load_state_dict(state)
    vecnorm.set_training(False)
    return vecnorm


def _sample_random_actions(mask: torch.Tensor, action_dim: int, rng: np.random.Generator, *, use_action_mask: bool) -> torch.Tensor:
    """ランダム方策の行動を生成する。"""
    bsz = int(mask.size(0))
    if not bool(use_action_mask):
        return torch.as_tensor(
            rng.integers(0, int(action_dim), size=(bsz,), dtype=np.int64),
            dtype=torch.int64,
            device="cpu",
        )
    out = torch.empty((bsz,), dtype=torch.int64, device="cpu")
    for i in range(bsz):
        legal = torch.nonzero(mask[i] > 0, as_tuple=False).view(-1)
        if int(legal.numel()) <= 0:
            out[i] = 0
        else:
            ri = int(rng.integers(0, int(legal.numel())))
            out[i] = int(legal[ri].item())
    return out


def _policy_actions(
    *,
    policy_name: str,
    agent: torch.nn.Module | None,
    board: torch.Tensor,
    mask: torch.Tensor,
    board_dev: torch.Tensor | None,
    mask_dev: torch.Tensor | None,
    device: torch.device,
    use_action_mask: bool,
    use_amp: bool,
    action_dim: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """指定方策（random/model）に応じて 1 ステップ分の行動を返す。"""
    if policy_name == "random":
        return _sample_random_actions(mask, int(action_dim), rng, use_action_mask=bool(use_action_mask)).contiguous()

    if agent is None:
        raise RuntimeError("agent is unexpectedly None")

    if device.type == "cuda":
        if board_dev is None:
            raise RuntimeError("board_dev is required on CUDA")
        board_dev.copy_(board, non_blocking=True)
        obs_model = board_dev
        mask_model = None
        if bool(use_action_mask):
            if mask_dev is None:
                raise RuntimeError("mask_dev is required on CUDA when use_action_mask=true")
            mask_dev.copy_(mask, non_blocking=True)
            mask_model = mask_dev
    else:
        obs_model = board
        mask_model = mask if bool(use_action_mask) else None

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(use_amp))
    with torch.no_grad(), autocast:
        action_t = agent.act(
            obs_model,
            action_mask=mask_model,
            deterministic=(policy_name == "model_greedy"),
        )
    return action_t.to(dtype=torch.int64, device="cpu").contiguous()


def _collect_chunk(
    *,
    env: BatchEnv,
    seed_begin: int,
    policy_name: str,
    agent: torch.nn.Module | None,
    device: torch.device,
    use_action_mask: bool,
    use_amp: bool,
    t_limit: int,
    vecnorm: VecNormalize | None,
    collect_score_breakdown: bool,
    collect_aux_targets: bool,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """`seed_begin` から連続 seed を使って評価チャンクを収集する。"""
    bsz = int(env.batch_size)
    c = int(env.feature_channels)
    n = int(env.board_size)
    a = int(env.action_dim)
    m_max = int(env.spec.m_max)

    board = torch.empty((bsz, c, n, n), dtype=torch.float32, device="cpu")
    mask = torch.empty((bsz, a), dtype=torch.uint8, device="cpu")
    reward = torch.empty((bsz,), dtype=torch.float32, device="cpu")
    done = torch.empty((bsz,), dtype=torch.uint8, device="cpu")

    seed_t = torch.arange(int(seed_begin), int(seed_begin) + int(bsz), dtype=torch.int64, device="cpu")
    env.reset_random(seed_t)

    mu = env.m_u().to(dtype=torch.int64, device="cpu")
    m_vals = mu[:, 0].detach().cpu().numpy().astype(np.int64)
    u_vals = mu[:, 1].detach().cpu().numpy().astype(np.int64)

    env.observe_into(board, mask)
    if vecnorm is not None:
        vecnorm.normalize_obs_inplace(board)

    ep_ret = torch.zeros((bsz,), dtype=torch.float64, device="cpu")
    ep_len = torch.zeros((bsz,), dtype=torch.int64, device="cpu")
    active = torch.ones((bsz,), dtype=torch.bool, device="cpu")

    use_cuda = device.type == "cuda"
    board_dev = torch.empty((bsz, c, n, n), dtype=torch.float32, device=device) if use_cuda else None
    mask_dev = torch.empty((bsz, a), dtype=torch.uint8, device=device) if (use_cuda and bool(use_action_mask)) else None

    aux_move_all: torch.Tensor | None = None
    aux_param_all: torch.Tensor | None = None
    aux_valid_all: torch.Tensor | None = None
    aux_move_tmp: torch.Tensor | None = None
    aux_param_tmp: torch.Tensor | None = None
    aux_valid_tmp: torch.Tensor | None = None

    if bool(collect_aux_targets):
        aux_move_all = torch.empty((bsz, int(t_limit), m_max, a), dtype=torch.float32, device="cpu")
        aux_param_all = torch.empty((bsz, int(t_limit), m_max, 5), dtype=torch.float32, device="cpu")
        aux_valid_all = torch.empty((bsz, int(t_limit), m_max), dtype=torch.uint8, device="cpu")
        aux_move_tmp = torch.empty((bsz, m_max, a), dtype=torch.float32, device="cpu")
        aux_param_tmp = torch.empty((bsz, m_max, 5), dtype=torch.float32, device="cpu")
        aux_valid_tmp = torch.empty((bsz, m_max), dtype=torch.uint8, device="cpu")

    for step in range(int(t_limit)):
        action_cpu = _policy_actions(
            policy_name=policy_name,
            agent=agent,
            board=board,
            mask=mask,
            board_dev=board_dev,
            mask_dev=mask_dev,
            device=device,
            use_action_mask=bool(use_action_mask),
            use_amp=bool(use_amp),
            action_dim=int(a),
            rng=rng,
        )

        try:
            if bool(collect_aux_targets):
                assert aux_move_tmp is not None
                assert aux_param_tmp is not None
                assert aux_valid_tmp is not None
                env.step_observe_aux_into(action_cpu, board, mask, reward, done, aux_move_tmp, aux_param_tmp, aux_valid_tmp)
                assert aux_move_all is not None
                assert aux_param_all is not None
                assert aux_valid_all is not None
                aux_move_all[:, step].copy_(aux_move_tmp)
                aux_param_all[:, step].copy_(aux_param_tmp)
                aux_valid_all[:, step].copy_(aux_valid_tmp)
            else:
                env.step_observe_into(action_cpu, board, mask, reward, done)
        except Exception as exc:
            raise RuntimeError(
                "cpp step_observe_into failed (likely illegal action). enable/use action mask for rollout"
            ) from exc

        if vecnorm is not None:
            vecnorm.normalize_reward_inplace(reward, done)
            vecnorm.normalize_obs_inplace(board)

        active_f = active.to(dtype=torch.float64)
        ep_ret += reward.to(dtype=torch.float64) * active_f
        ep_len += active.to(dtype=torch.int64)
        done_b = done.to(dtype=torch.bool)
        active = active & (~done_b)
        if not bool(active.any().item()):
            break

    game_score = env.official_score().to(dtype=torch.float64, device="cpu").detach().cpu().numpy()

    self_score_np: np.ndarray | None = None
    enemy_score_np: np.ndarray | None = None
    if bool(collect_score_breakdown):
        score_pair = env.score_s0_sa().to(dtype=torch.float64, device="cpu")
        self_score_np = score_pair[:, 0].detach().cpu().numpy()
        enemy_score_np = score_pair[:, 1].detach().cpu().numpy()

    return (
        ep_ret.detach().cpu().numpy(),
        ep_len.detach().cpu().numpy().astype(np.int64),
        game_score,
        m_vals,
        u_vals,
        self_score_np,
        enemy_score_np,
        aux_move_all,
        aux_param_all,
        aux_valid_all,
    )


def run_policy_episodes(
    *,
    env_id: str,
    episodes: int,
    num_envs: int = 1,
    seed: int,
    feature_id: str,
    pf_enabled: bool,
    policy: str,
    agent: torch.nn.Module | None,
    device: torch.device,
    use_action_mask: bool,
    amp: bool = False,
    max_steps_per_episode: int = 0,
    vecnorm_enabled: bool = False,
    vecnorm_state: dict[str, Any] | None = None,
    vecnorm_norm_obs: bool = True,
    vecnorm_norm_reward: bool = False,
    vecnorm_clip_obs: float = 10.0,
    vecnorm_clip_reward: float = 10.0,
    vecnorm_epsilon: float = 1e-8,
    vecnorm_gamma: float = 0.99,
    on_env_ready: Callable[[tuple[int, ...], int], None] | None = None,
    on_episode_end: Callable[[int, float, int], None] | None = None,
    collect_score_breakdown: bool = False,
    collect_aux_targets: bool = False,
) -> EpisodeStats:
    """指定ポリシーで複数エピソードを実行し統計量を返す。

    Args:
        env_id (str): 実行環境 ID（現在は `AHC061Local-v0` のみ）。
        episodes (int): 評価エピソード数。
        num_envs (int): 同時実行する環境数。
        seed (int): 評価開始シード。
        feature_id (str): 観測特徴量 ID。
        pf_enabled (bool): PF ベイズ推定を有効化するか。
        policy (str): `random` / `model_stochastic` / `model_greedy`。
        agent (torch.nn.Module | None): モデルポリシー利用時のエージェント。
        device (torch.device): 推論デバイス。
        use_action_mask (bool): 合法手マスクを使うか。
        amp (bool): CUDA 上で AMP 推論を使うか。
        max_steps_per_episode (int): 1 エピソードあたりの最大ステップ。
        vecnorm_enabled (bool): VecNormalize を有効化するか。
        vecnorm_state (dict[str, Any] | None): 復元する正規化状態。
        vecnorm_norm_obs (bool): 観測正規化を有効化するか。
        vecnorm_norm_reward (bool): 報酬正規化を有効化するか。
        vecnorm_clip_obs (float): 観測クリップ上限。
        vecnorm_clip_reward (float): 報酬クリップ上限。
        vecnorm_epsilon (float): 正規化の数値安定化係数。
        vecnorm_gamma (float): 走行平均更新用割引率。
        on_env_ready (Callable[[tuple[int, ...], int], None] | None):
            環境準備完了時のコールバック。
        on_episode_end (Callable[[int, float, int], None] | None):
            各エピソード終了時のコールバック。
        collect_score_breakdown (bool): self/enemy スコア内訳を収集するか。
        collect_aux_targets (bool): 補助教師信号を収集するか。

    Returns:
        EpisodeStats: 評価結果の集計。

    Raises:
        ValueError: 引数整合性が崩れている場合。
        RuntimeError: 実行中に不正アクション等で環境側が失敗した場合。
    """
    if str(env_id) != "AHC061Local-v0":
        raise ValueError(
            "cpp runner supports only env_id='AHC061Local-v0'; "
            f"got {env_id!r}"
        )

    total_episodes = int(episodes)
    if total_episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes}")

    policy_name = str(policy).strip().lower()
    if policy_name not in ("random", "model_stochastic", "model_greedy"):
        raise ValueError(f"unsupported policy={policy!r}")
    if policy_name in ("model_stochastic", "model_greedy") and agent is None:
        raise ValueError("agent is required for policy model_stochastic/model_greedy")

    if total_episodes == 0:
        env0 = BatchEnv(batch_size=1, feature_id=str(feature_id), pf_enabled=bool(pf_enabled))
        c0 = int(env0.feature_channels)
        n0 = int(env0.board_size)
        a0 = int(env0.action_dim)
        obs_shape0 = (c0, n0, n0)
        if on_env_ready is not None:
            on_env_ready(obs_shape0, a0)
        return EpisodeStats(
            obs_shape=obs_shape0,
            action_dim=a0,
            episode_returns=[],
            episode_lengths=[],
            episode_terminal_game_scores=[],
            episode_m=[],
            episode_u=[],
            episode_game_score_self=[],
            episode_game_score_enemy_max=[],
        )

    batch_size = _resolve_batch_size(episodes=total_episodes, num_envs=int(num_envs))
    env_probe = BatchEnv(batch_size=1, feature_id=str(feature_id), pf_enabled=bool(pf_enabled))
    c = int(env_probe.feature_channels)
    n = int(env_probe.board_size)
    a = int(env_probe.action_dim)
    obs_shape = (c, n, n)
    if on_env_ready is not None:
        on_env_ready(obs_shape, a)

    if agent is not None:
        if tuple(agent.obs_shape) != tuple(obs_shape):
            raise ValueError(f"obs_shape mismatch: model={agent.obs_shape}, env={obs_shape}")
        if int(agent.action_dim) != int(a):
            raise ValueError(f"action_dim mismatch: model={agent.action_dim}, env={a}")

    t_limit = int(env_probe.spec.t_max)
    if int(max_steps_per_episode) > 0:
        t_limit = int(min(t_limit, int(max_steps_per_episode)))

    ep_returns = np.empty((total_episodes,), dtype=np.float64)
    ep_lengths = np.empty((total_episodes,), dtype=np.int64)
    ep_scores = np.empty((total_episodes,), dtype=np.float64)
    ep_m = np.empty((total_episodes,), dtype=np.int64)
    ep_u = np.empty((total_episodes,), dtype=np.int64)
    ep_self = np.empty((total_episodes,), dtype=np.float64) if bool(collect_score_breakdown) else None
    ep_enemy = np.empty((total_episodes,), dtype=np.float64) if bool(collect_score_breakdown) else None

    aux_move_dist: torch.Tensor | None = None
    aux_opp_param: torch.Tensor | None = None
    aux_opp_valid: torch.Tensor | None = None

    was_training = bool(agent.training) if agent is not None else False
    if agent is not None:
        agent.eval()

    rng = np.random.default_rng(int(seed))

    try:
        epi_start = 0
        while epi_start < total_episodes:
            cur_bsz = int(min(batch_size, total_episodes - epi_start))
            env = BatchEnv(
                batch_size=cur_bsz,
                feature_id=str(feature_id),
                pf_enabled=bool(pf_enabled),
            )
            vecnorm = _build_eval_vecnorm(
                batch_size=cur_bsz,
                obs_shape=obs_shape,
                enabled=bool(vecnorm_enabled),
                state=vecnorm_state,
                norm_obs=bool(vecnorm_norm_obs),
                norm_reward=bool(vecnorm_norm_reward),
                clip_obs=float(vecnorm_clip_obs),
                clip_reward=float(vecnorm_clip_reward),
                epsilon=float(vecnorm_epsilon),
                gamma=float(vecnorm_gamma),
            )

            (
                ret_chunk,
                len_chunk,
                score_chunk,
                m_chunk,
                u_chunk,
                self_chunk,
                enemy_chunk,
                aux_move_chunk,
                aux_param_chunk,
                aux_valid_chunk,
            ) = _collect_chunk(
                env=env,
                seed_begin=int(seed) + int(epi_start),
                policy_name=policy_name,
                agent=agent,
                device=device,
                use_action_mask=bool(use_action_mask),
                use_amp=bool(amp and device.type == "cuda"),
                t_limit=int(t_limit),
                vecnorm=vecnorm,
                collect_score_breakdown=bool(collect_score_breakdown),
                collect_aux_targets=bool(collect_aux_targets),
                rng=rng,
            )

            sl = slice(int(epi_start), int(epi_start + cur_bsz))
            ep_returns[sl] = ret_chunk
            ep_lengths[sl] = len_chunk
            ep_scores[sl] = score_chunk
            ep_m[sl] = m_chunk
            ep_u[sl] = u_chunk
            if bool(collect_score_breakdown):
                if ep_self is None or ep_enemy is None:
                    raise RuntimeError("internal error: score breakdown buffers are not initialized")
                if self_chunk is None or enemy_chunk is None:
                    raise RuntimeError("internal error: score breakdown chunk is missing")
                ep_self[sl] = self_chunk
                ep_enemy[sl] = enemy_chunk

            if bool(collect_aux_targets):
                if aux_move_chunk is None or aux_param_chunk is None or aux_valid_chunk is None:
                    raise RuntimeError("internal error: aux chunk tensors are missing")
                if aux_move_dist is None:
                    aux_move_dist = torch.empty(
                        (total_episodes, int(t_limit), int(env.spec.m_max), int(a)),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    aux_opp_param = torch.empty(
                        (total_episodes, int(t_limit), int(env.spec.m_max), 5),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    aux_opp_valid = torch.empty(
                        (total_episodes, int(t_limit), int(env.spec.m_max)),
                        dtype=torch.uint8,
                        device="cpu",
                    )
                assert aux_opp_param is not None
                assert aux_opp_valid is not None
                aux_move_dist[sl].copy_(aux_move_chunk)
                aux_opp_param[sl].copy_(aux_param_chunk)
                aux_opp_valid[sl].copy_(aux_valid_chunk)

            if on_episode_end is not None:
                for i in range(cur_bsz):
                    gi = int(epi_start + i)
                    on_episode_end(gi, float(ep_returns[gi]), int(ep_lengths[gi]))

            epi_start += cur_bsz
    finally:
        if agent is not None and bool(was_training):
            agent.train()

    return EpisodeStats(
        obs_shape=tuple(obs_shape),
        action_dim=int(a),
        episode_returns=[float(x) for x in ep_returns.tolist()],
        episode_lengths=[int(x) for x in ep_lengths.tolist()],
        episode_terminal_game_scores=[float(x) for x in ep_scores.tolist()],
        episode_m=[int(x) for x in ep_m.tolist()],
        episode_u=[int(x) for x in ep_u.tolist()],
        episode_game_score_self=([] if ep_self is None else [float(x) for x in ep_self.tolist()]),
        episode_game_score_enemy_max=([] if ep_enemy is None else [float(x) for x in ep_enemy.tolist()]),
        aux_move_dist=aux_move_dist,
        aux_opp_param=aux_opp_param,
        aux_opp_valid=aux_opp_valid,
    )
