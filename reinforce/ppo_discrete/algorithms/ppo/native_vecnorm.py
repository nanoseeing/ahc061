from __future__ import annotations

from typing import Any

import numpy as np
import torch


class RunningMeanStd:
    """Online running mean/variance using stable parallel update."""

    def __init__(self, *, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = float(arr.shape[0])
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        if batch_count <= 0:
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        new_var = m2 / total

        self.mean = np.asarray(new_mean, dtype=np.float64)
        self.var = np.maximum(np.asarray(new_var, dtype=np.float64), 1e-12)
        self.count = float(total)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64).copy(),
            "var": np.asarray(self.var, dtype=np.float64).copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state.get("mean", self.mean), dtype=np.float64).copy()
        self.var = np.maximum(np.asarray(state.get("var", self.var), dtype=np.float64), 1e-12).copy()
        self.count = float(state.get("count", self.count))


class NativeVecNormalize:
    """Gym-free VecNormalize equivalent for native BatchEnv rollouts."""

    def __init__(
        self,
        *,
        num_envs: int,
        obs_shape: tuple[int, ...],
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        epsilon: float = 1e-8,
        gamma: float = 0.99,
        training: bool = True,
    ) -> None:
        self.norm_obs = bool(norm_obs)
        self.norm_reward = bool(norm_reward)
        self.clip_obs = float(clip_obs)
        self.clip_reward = float(clip_reward)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.training = bool(training)

        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.obs_rms = RunningMeanStd(shape=self.obs_shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros((int(num_envs),), dtype=np.float64)

    def set_training(self, mode: bool) -> None:
        self.training = bool(mode)

    def _obs_view(self, obs: torch.Tensor) -> np.ndarray:
        if obs.device.type != "cpu":
            raise ValueError("NativeVecNormalize expects CPU observation tensor")
        arr = obs.numpy()
        if arr.shape == self.obs_shape:
            return arr.reshape((1,) + self.obs_shape)
        if len(arr.shape) < len(self.obs_shape):
            raise ValueError(f"obs shape mismatch: got={arr.shape}, expected suffix={self.obs_shape}")
        if tuple(arr.shape[-len(self.obs_shape) :]) != self.obs_shape:
            raise ValueError(f"obs shape mismatch: got={arr.shape}, expected suffix={self.obs_shape}")
        return arr.reshape((-1,) + self.obs_shape)

    def _reward_done_view(self, reward: torch.Tensor, done: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if reward.device.type != "cpu" or done.device.type != "cpu":
            raise ValueError("NativeVecNormalize expects CPU reward/done tensors")
        r = reward.numpy().reshape(-1).astype(np.float32, copy=False)
        d = np.asarray(done.numpy().reshape(-1), dtype=np.bool_)
        if self.returns.shape[0] != r.shape[0]:
            self.returns = np.zeros((int(r.shape[0]),), dtype=np.float64)
        return r, d

    def normalize_obs_inplace(self, obs: torch.Tensor) -> None:
        view = self._obs_view(obs)
        if self.training and self.norm_obs:
            self.obs_rms.update(view)
        if not self.norm_obs:
            return
        denom = np.sqrt(self.obs_rms.var + self.epsilon)
        view -= self.obs_rms.mean
        view /= denom
        np.clip(view, -self.clip_obs, self.clip_obs, out=view)

    def normalize_reward_inplace(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        r, d = self._reward_done_view(reward, done)
        if self.training and self.norm_reward:
            self.returns = self.returns * self.gamma + np.asarray(r, dtype=np.float64)
            self.ret_rms.update(self.returns)
        if self.norm_reward:
            scale = np.sqrt(float(self.ret_rms.var) + self.epsilon)
            r /= float(scale)
            np.clip(r, -self.clip_reward, self.clip_reward, out=r)
        self.returns[d] = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "norm_obs": bool(self.norm_obs),
            "norm_reward": bool(self.norm_reward),
            "clip_obs": float(self.clip_obs),
            "clip_reward": float(self.clip_reward),
            "epsilon": float(self.epsilon),
            "gamma": float(self.gamma),
            "training": bool(self.training),
            "obs_rms": self.obs_rms.state_dict(),
            "ret_rms": self.ret_rms.state_dict(),
            "returns": np.asarray(self.returns, dtype=np.float64).copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.norm_obs = bool(state.get("norm_obs", self.norm_obs))
        self.norm_reward = bool(state.get("norm_reward", self.norm_reward))
        self.clip_obs = float(state.get("clip_obs", self.clip_obs))
        self.clip_reward = float(state.get("clip_reward", self.clip_reward))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.gamma = float(state.get("gamma", self.gamma))
        self.training = bool(state.get("training", self.training))

        obs_rms = state.get("obs_rms")
        if isinstance(obs_rms, dict):
            self.obs_rms.load_state_dict(obs_rms)
        ret_rms = state.get("ret_rms")
        if isinstance(ret_rms, dict):
            self.ret_rms.load_state_dict(ret_rms)

        returns = np.asarray(state.get("returns", self.returns), dtype=np.float64).reshape(-1)
        if returns.size == self.returns.size:
            self.returns = returns.copy()
        elif returns.size == 1:
            self.returns.fill(float(returns[0]))
        else:
            self.returns.fill(0.0)
