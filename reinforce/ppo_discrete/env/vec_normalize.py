from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


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


def normalize_obs_with_state(obs: np.ndarray, vecnorm_state: dict[str, Any]) -> np.ndarray:
    """Normalize observation array using saved VecNormalize state."""
    if not isinstance(vecnorm_state, dict):
        return np.asarray(obs, dtype=np.float32)
    obs_rms = vecnorm_state.get("obs_rms")
    if not isinstance(obs_rms, dict):
        return np.asarray(obs, dtype=np.float32)
    mean = np.asarray(obs_rms.get("mean"), dtype=np.float64)
    var = np.asarray(obs_rms.get("var"), dtype=np.float64)
    if mean.size == 0 or var.size == 0:
        return np.asarray(obs, dtype=np.float32)

    eps = float(vecnorm_state.get("epsilon", 1e-8))
    clip_obs = float(vecnorm_state.get("clip_obs", 10.0))
    arr = np.asarray(obs, dtype=np.float32)
    obs_shape = tuple(int(x) for x in mean.shape)
    if arr.shape == obs_shape:
        norm = (arr - mean) / np.sqrt(var + eps)
        return np.clip(norm, -clip_obs, clip_obs).astype(np.float32)
    if len(arr.shape) >= len(obs_shape) and tuple(arr.shape[-len(obs_shape) :]) == obs_shape:
        view = arr.reshape((-1,) + obs_shape)
        norm = (view - mean) / np.sqrt(var + eps)
        norm = np.clip(norm, -clip_obs, clip_obs).astype(np.float32)
        return norm.reshape(arr.shape)
    return arr


class VecNormalize:
    """Lightweight VecNormalize wrapper for gymnasium vector envs."""

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        *,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        epsilon: float = 1e-8,
        gamma: float = 0.99,
        training: bool = True,
    ) -> None:
        self.env = env
        self.norm_obs = bool(norm_obs)
        self.norm_reward = bool(norm_reward)
        self.clip_obs = float(clip_obs)
        self.clip_reward = float(clip_reward)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.training = bool(training)

        obs_shape = tuple(int(x) for x in env.single_observation_space.shape)
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros((int(env.num_envs),), dtype=np.float64)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def set_training(self, mode: bool) -> None:
        self.training = bool(mode)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if not self.norm_obs:
            return arr
        denom = np.sqrt(self.obs_rms.var + self.epsilon)
        norm = (arr - self.obs_rms.mean) / denom
        return np.clip(norm, -self.clip_obs, self.clip_obs).astype(np.float32)

    def _normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        arr = np.asarray(reward, dtype=np.float32)
        if not self.norm_reward:
            return arr
        scale = np.sqrt(float(self.ret_rms.var) + self.epsilon)
        norm = arr / scale
        return np.clip(norm, -self.clip_reward, self.clip_reward).astype(np.float32)

    def _normalize_obs_like(self, obs_like: Any) -> Any:
        arr = np.asarray(obs_like, dtype=np.float32)
        if arr.size == 0:
            return obs_like
        obs_shape = tuple(int(x) for x in self.obs_rms.mean.shape)
        if arr.shape == obs_shape:
            return self._normalize_obs(arr[None, ...])[0]
        if len(arr.shape) >= len(obs_shape) and tuple(arr.shape[-len(obs_shape) :]) == obs_shape:
            view = arr.reshape((-1,) + obs_shape)
            norm = self._normalize_obs(view)
            return norm.reshape(arr.shape)
        return obs_like

    def _normalize_info_obs_container(self, container: Any, mask: Any) -> Any:
        if container is None:
            return container
        mask_arr = None
        if mask is not None:
            try:
                mask_arr = np.asarray(mask, dtype=np.bool_).reshape(-1)
            except Exception:
                mask_arr = None

        def _allowed(i: int) -> bool:
            if mask_arr is None:
                return True
            return i < mask_arr.shape[0] and bool(mask_arr[i])

        if isinstance(container, np.ndarray):
            if container.ndim >= 1 and container.shape[0] == int(self.env.num_envs) and container.dtype != object:
                out = np.asarray(container, dtype=np.float32).copy()
                if mask_arr is None:
                    return self._normalize_obs(out)
                idxs = np.flatnonzero(mask_arr)
                if idxs.size > 0:
                    out[idxs] = self._normalize_obs(out[idxs])
                return out
            if container.dtype == object:
                out = container.copy()
                n = int(out.shape[0]) if out.ndim > 0 else 0
                for i in range(n):
                    if _allowed(i):
                        out[i] = self._normalize_obs_like(out[i])
                return out
            return container

        if isinstance(container, (list, tuple)):
            out = list(container)
            for i in range(len(out)):
                if _allowed(i):
                    out[i] = self._normalize_obs_like(out[i])
            if isinstance(container, tuple):
                return tuple(out)
            return out
        return container

    def _normalize_infos(self, infos: Any) -> Any:
        if not self.norm_obs or not isinstance(infos, dict):
            return infos
        out = dict(infos)
        for key in ("final_observation", "terminal_observation"):
            if key not in out:
                continue
            mask_key = f"_{key}"
            out[key] = self._normalize_info_obs_container(out.get(key), out.get(mask_key))

        finfo = out.get("final_info")
        if isinstance(finfo, (list, tuple, np.ndarray)):
            rows = list(finfo)
            changed = False
            for i, item in enumerate(rows):
                if not isinstance(item, dict):
                    continue
                d = dict(item)
                local_changed = False
                for key in ("terminal_observation", "final_observation"):
                    if key in d:
                        d[key] = self._normalize_obs_like(d[key])
                        local_changed = True
                if local_changed:
                    rows[i] = d
                    changed = True
            if changed:
                if isinstance(finfo, tuple):
                    out["final_info"] = tuple(rows)
                elif isinstance(finfo, np.ndarray):
                    out["final_info"] = np.asarray(rows, dtype=object)
                else:
                    out["final_info"] = rows
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs_arr = np.asarray(obs, dtype=np.float32)
        if self.training and self.norm_obs:
            self.obs_rms.update(obs_arr)
        self.returns.fill(0.0)
        return self._normalize_obs(obs_arr), info

    def step(self, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        obs, reward, terminations, truncations, infos = self.env.step(action)
        obs_arr = np.asarray(obs, dtype=np.float32)
        reward_arr = np.asarray(reward, dtype=np.float32)
        done = np.logical_or(terminations, truncations)

        if self.training and self.norm_obs:
            self.obs_rms.update(obs_arr)
        obs_out = self._normalize_obs(obs_arr)

        if self.training and self.norm_reward:
            self.returns = self.returns * self.gamma + reward_arr
            self.ret_rms.update(self.returns)
        reward_out = self._normalize_reward(reward_arr)

        self.returns[np.asarray(done, dtype=np.bool_)] = 0.0
        infos_out = self._normalize_infos(infos)
        return obs_out, reward_out, terminations, truncations, infos_out

    def close(self) -> None:
        self.env.close()

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
