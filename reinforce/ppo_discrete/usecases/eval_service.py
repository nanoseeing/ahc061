from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ..core.ppo.vecnorm import VecNormalize
from ..domain.ahc061.batch_env import BatchEnv


@dataclass
class Transition:
    episode: int
    step: int
    obs: np.ndarray
    action: int
    reward: float
    done: bool
    bayes_params: np.ndarray


@dataclass
class EpisodeStats:
    obs_shape: tuple[int, ...]
    action_dim: int
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_illegal_penalties: list[float]
    episode_terminal_scores: list[float]
    episode_terminal_game_scores: list[float]
    episode_game_score_ratio: list[float]
    episode_game_score_self: list[float]
    episode_game_score_enemy_max: list[float]


@dataclass
class AuxTransition:
    episode: int
    step: int
    move_dist: np.ndarray
    opp_param: np.ndarray
    opp_valid: np.ndarray


def _unwrap_action(action: torch.Tensor) -> np.ndarray:
    return action.detach().cpu().numpy()


def run_policy_episodes(
    *,
    env_id: str,
    episodes: int,
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
    on_transition: Callable[[Transition], None] | None = None,
    on_aux_transition: Callable[[AuxTransition], None] | None = None,
    on_episode_end: Callable[[int, float, int], None] | None = None,
) -> EpisodeStats:
    if str(env_id) != "AHC061Local-v0":
        raise ValueError(
            "cpp runner supports only env_id='AHC061Local-v0'; "
            f"got {env_id!r}"
        )
    policy_name = str(policy).strip().lower()
    if policy_name not in ("random", "model_stochastic", "model_greedy"):
        raise ValueError(f"unsupported policy={policy!r}")
    if policy_name in ("model_stochastic", "model_greedy") and agent is None:
        raise ValueError("agent is required for policy model_stochastic/model_greedy")

    env = BatchEnv(
        batch_size=1,
        feature_id=str(feature_id),
        pf_enabled=bool(pf_enabled),
    )
    c = int(env.feature_channels)
    n = int(env.board_size)
    a = int(env.action_dim)
    obs_shape = (c, n, n)
    if on_env_ready is not None:
        on_env_ready(obs_shape, a)

    if agent is not None:
        if tuple(agent.obs_shape) != tuple(obs_shape):
            raise ValueError(f"obs_shape mismatch: model={agent.obs_shape}, env={obs_shape}")
        if int(agent.action_dim) != int(a):
            raise ValueError(f"action_dim mismatch: model={agent.action_dim}, env={a}")
    was_training = bool(agent.training) if agent is not None else False
    if agent is not None:
        agent.eval()

    vecnorm: VecNormalize | None = None
    if bool(vecnorm_enabled):
        vecnorm = VecNormalize(
            num_envs=1,
            obs_shape=obs_shape,
            norm_obs=bool(vecnorm_norm_obs),
            norm_reward=bool(vecnorm_norm_reward),
            clip_obs=float(vecnorm_clip_obs),
            clip_reward=float(vecnorm_clip_reward),
            epsilon=float(vecnorm_epsilon),
            gamma=float(vecnorm_gamma),
            training=False,
        )
        if isinstance(vecnorm_state, dict):
            vecnorm.load_state_dict(vecnorm_state)
        vecnorm.set_training(False)

    board = torch.empty((1, c, n, n), dtype=torch.float32, device="cpu")
    mask = torch.empty((1, a), dtype=torch.uint8, device="cpu")
    reward = torch.empty((1,), dtype=torch.float32, device="cpu")
    done = torch.empty((1,), dtype=torch.uint8, device="cpu")
    seed_t = torch.empty((1,), dtype=torch.int64, device="cpu")
    need_aux = on_aux_transition is not None
    need_bayes = on_transition is not None
    move_dist = torch.empty((1, 8, int(a)), dtype=torch.float32, device="cpu") if need_aux else None
    opp_param = torch.empty((1, 8, 5), dtype=torch.float32, device="cpu") if need_aux else None
    opp_valid = torch.empty((1, 8), dtype=torch.uint8, device="cpu") if need_aux else None
    bayes_params = torch.empty((1, 7, 4), dtype=torch.float32, device="cpu") if need_bayes else None

    use_cuda = device.type == "cuda"
    use_amp = bool(use_cuda and amp)
    board_dev = torch.empty((1, c, n, n), dtype=torch.float32, device=device) if use_cuda else None
    mask_dev = torch.empty((1, a), dtype=torch.uint8, device=device) if use_cuda else None
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_illegal_penalties: list[float] = []
    ep_terminal_scores: list[float] = []
    ep_terminal_game_scores: list[float] = []
    ep_game_score_ratio: list[float] = []
    ep_game_score_self: list[float] = []
    ep_game_score_enemy_max: list[float] = []
    rng = np.random.default_rng(int(seed))

    t_limit = int(env.spec.t_max)
    if int(max_steps_per_episode) > 0:
        t_limit = int(min(t_limit, int(max_steps_per_episode)))

    try:
        for epi in range(int(episodes)):
            seed_t[0] = int(seed) + int(epi)
            env.reset_random(seed_t)
            env.observe_into(board, mask)
            if vecnorm is not None:
                vecnorm.normalize_obs_inplace(board)

            ep_ret = 0.0
            ep_len = 0
            for step in range(t_limit):
                obs_before = np.asarray(board[0], dtype=np.float32).copy() if on_transition is not None else None
                bayes_before = None
                if need_bayes:
                    assert bayes_params is not None
                    env.bayes_params_into(bayes_params)
                    bayes_before = np.asarray(bayes_params[0], dtype=np.float32).reshape(-1).copy()

                if policy_name == "random":
                    if bool(use_action_mask):
                        legal = torch.nonzero(mask[0] > 0, as_tuple=False).view(-1)
                        if int(legal.numel()) <= 0:
                            action_idx = 0
                        else:
                            ri = int(rng.integers(0, int(legal.numel())))
                            action_idx = int(legal[ri].item())
                    else:
                        action_idx = int(rng.integers(0, int(a)))
                    action_cpu = torch.tensor([action_idx], dtype=torch.int64, device="cpu")
                else:
                    if agent is None:
                        raise RuntimeError("agent is unexpectedly None")
                    if use_cuda:
                        assert board_dev is not None
                        board_dev.copy_(board, non_blocking=True)
                        obs_model = board_dev
                        mask_model = None
                        if bool(use_action_mask):
                            assert mask_dev is not None
                            mask_dev.copy_(mask, non_blocking=True)
                            mask_model = mask_dev
                    else:
                        obs_model = board
                        mask_model = mask if bool(use_action_mask) else None
                    with torch.no_grad(), autocast:
                        action_t = agent.act(
                            obs_model,
                            action_mask=mask_model,
                            deterministic=(policy_name == "model_greedy"),
                        )
                    action_np = _unwrap_action(action_t).astype(np.int64)
                    if action_np.ndim == 0:
                        action_np = action_np.reshape(1)
                    action_cpu = torch.as_tensor(action_np, dtype=torch.int64, device="cpu").contiguous()
                    action_idx = int(action_cpu[0].item())

                try:
                    if need_aux:
                        assert move_dist is not None
                        assert opp_param is not None
                        assert opp_valid is not None
                        env.step_observe_aux_into(action_cpu, board, mask, reward, done, move_dist, opp_param, opp_valid)
                    else:
                        env.step_observe_into(action_cpu, board, mask, reward, done)
                except Exception as exc:
                    raise RuntimeError(
                        "cpp step_observe_into failed (likely illegal action). "
                        "enable/use action mask for rollout"
                    ) from exc

                if vecnorm is not None:
                    vecnorm.normalize_reward_inplace(reward, done)
                    vecnorm.normalize_obs_inplace(board)

                r = float(reward[0].item())
                d = bool(int(done[0].item()) != 0)
                if on_transition is not None:
                    assert obs_before is not None
                    on_transition(
                        Transition(
                            episode=int(epi),
                            step=int(step),
                            obs=obs_before,
                            action=int(action_idx),
                            reward=float(r),
                            done=bool(d),
                            bayes_params=(
                                bayes_before
                                if bayes_before is not None
                                else np.zeros((28,), dtype=np.float32)
                            ),
                        )
                    )
                if on_aux_transition is not None:
                    assert move_dist is not None
                    assert opp_param is not None
                    assert opp_valid is not None
                    on_aux_transition(
                        AuxTransition(
                            episode=int(epi),
                            step=int(step),
                            move_dist=np.asarray(move_dist[0], dtype=np.float32).copy(),
                            opp_param=np.asarray(opp_param[0], dtype=np.float32).copy(),
                            opp_valid=np.asarray(opp_valid[0], dtype=np.uint8).copy(),
                        )
                    )
                ep_ret += r
                ep_len += 1
                if d:
                    break

            score_pair = env.score_s0_sa().to(dtype=torch.float64, device="cpu")
            s0 = float(score_pair[0, 0].item())
            sa = float(score_pair[0, 1].item())
            ratio = float("inf") if sa <= 0.0 else float(s0 / sa)
            terminal_score = float(math.log2(max(1e-9, s0) / max(1e-9, sa)))
            terminal_game_score = float(env.official_score()[0].item())

            ep_returns.append(float(ep_ret))
            ep_lengths.append(int(ep_len))
            # cpp env raises on illegal action; penalty component remains zero.
            ep_illegal_penalties.append(0.0)
            ep_terminal_scores.append(terminal_score)
            ep_terminal_game_scores.append(terminal_game_score)
            if np.isfinite(ratio):
                ep_game_score_ratio.append(float(ratio))
            if np.isfinite(s0):
                ep_game_score_self.append(float(s0))
            if np.isfinite(sa):
                ep_game_score_enemy_max.append(float(sa))

            if on_episode_end is not None:
                on_episode_end(int(epi), float(ep_ret), int(ep_len))
    finally:
        if agent is not None and bool(was_training):
            agent.train()

    return EpisodeStats(
        obs_shape=tuple(obs_shape),
        action_dim=int(a),
        episode_returns=ep_returns,
        episode_lengths=ep_lengths,
        episode_illegal_penalties=ep_illegal_penalties,
        episode_terminal_scores=ep_terminal_scores,
        episode_terminal_game_scores=ep_terminal_game_scores,
        episode_game_score_ratio=ep_game_score_ratio,
        episode_game_score_self=ep_game_score_self,
        episode_game_score_enemy_max=ep_game_score_enemy_max,
    )
