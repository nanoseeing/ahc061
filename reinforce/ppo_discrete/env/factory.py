from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

from ..domains.ahc061.env import AHC061LocalEnv
from .vec_normalize import VecNormalize


@dataclass(frozen=True)
class EnvSpec:
    obs_shape: tuple[int, ...]
    action_dim: int


def _maybe_flatten_obs(env: gym.Env, flatten_obs: bool) -> gym.Env:
    if not flatten_obs:
        return env
    if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 1:
        return env
    return gym.wrappers.FlattenObservation(env)


def make_single_env(
    env_id: str,
    *,
    idx: int = 0,
    num_envs: int = 1,
    capture_video: bool = False,
    run_name: str = "",
    flatten_obs: bool = True,
    env_kwargs: dict[str, Any] | None = None,
) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        kwargs = dict(env_kwargs or {})
        if env_id == "AHC061Local-v0":
            kwargs.setdefault("vector_env_rank", int(idx))
            kwargs.setdefault("vector_env_size", int(max(1, num_envs)))
            env = AHC061LocalEnv(**kwargs)
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array", **kwargs)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, **kwargs)
        env = _maybe_flatten_obs(env, flatten_obs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_vector_env(
    env_id: str,
    *,
    num_envs: int,
    seed: int,
    capture_video: bool = False,
    run_name: str = "",
    flatten_obs: bool = True,
    vector_env: str = "sync",
    env_kwargs: dict[str, Any] | None = None,
    vecnorm: bool = False,
    vecnorm_norm_obs: bool = True,
    vecnorm_norm_reward: bool = True,
    vecnorm_clip_obs: float = 10.0,
    vecnorm_clip_reward: float = 10.0,
    vecnorm_epsilon: float = 1e-8,
    vecnorm_gamma: float = 0.99,
    vecnorm_training: bool = True,
    vecnorm_state: dict[str, Any] | None = None,
) -> gym.vector.VectorEnv:
    if vector_env not in ("sync", "async"):
        raise ValueError(f"unsupported vector_env={vector_env}; expected 'sync' or 'async'")
    env_ctor = gym.vector.SyncVectorEnv if vector_env == "sync" else gym.vector.AsyncVectorEnv
    envs = env_ctor(
        [
            make_single_env(
                env_id,
                idx=i,
                num_envs=num_envs,
                capture_video=capture_video,
                run_name=run_name,
                flatten_obs=flatten_obs,
                env_kwargs=env_kwargs,
            )
            for i in range(num_envs)
        ]
    )
    envs.reset(seed=seed)
    if not vecnorm:
        return envs
    wrapped = VecNormalize(
        envs,
        norm_obs=bool(vecnorm_norm_obs),
        norm_reward=bool(vecnorm_norm_reward),
        clip_obs=float(vecnorm_clip_obs),
        clip_reward=float(vecnorm_clip_reward),
        epsilon=float(vecnorm_epsilon),
        gamma=float(vecnorm_gamma),
        training=bool(vecnorm_training),
    )
    if isinstance(vecnorm_state, dict):
        wrapped.load_state_dict(vecnorm_state)
        wrapped.set_training(bool(vecnorm_training))
    return wrapped


def infer_env_spec(envs: gym.vector.VectorEnv) -> EnvSpec:
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("only discrete action space is supported by ppo_discrete scaffold")
    if not isinstance(envs.single_observation_space, gym.spaces.Box):
        raise ValueError("observation must be gym.spaces.Box; apply flatten wrapper if needed")
    obs_shape = tuple(int(x) for x in envs.single_observation_space.shape)
    action_dim = int(envs.single_action_space.n)
    return EnvSpec(obs_shape=obs_shape, action_dim=action_dim)


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def done_to_tensor(done: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(done, dtype=torch.float32, device=device)


def reward_to_tensor(reward: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(reward, dtype=torch.float32, device=device)


def unwrap_action(action: torch.Tensor) -> np.ndarray:
    return action.detach().cpu().numpy()


def reset_env(envs: gym.vector.VectorEnv, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    obs, info = envs.reset(seed=seed)
    return np.asarray(obs), dict(info) if isinstance(info, dict) else {}


def unwrap_vec_normalize(envs: Any) -> VecNormalize | None:
    cur = envs
    seen: set[int] = set()
    while cur is not None:
        cid = id(cur)
        if cid in seen:
            break
        seen.add(cid)
        if isinstance(cur, VecNormalize):
            return cur
        nxt = getattr(cur, "env", None)
        if nxt is None:
            nxt = getattr(cur, "venv", None)
        if nxt is None:
            break
        cur = nxt
    return None
