"""`vecnorm` に関するモジュール。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class RunningMeanStd:
    """`RunningMeanStd` を表すクラス。"""

    def __init__(self, *, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
        """インスタンスを初期化する。

        Args:
            epsilon (float): epsilon の値。
            shape (tuple[int, ...]): shape の値。
        """
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var = torch.ones(shape, dtype=torch.float64)
        self.count = float(epsilon)

    def update(self, x: torch.Tensor) -> None:
        """`update` を実行する。

        Args:
            x (torch.Tensor): 入力テンソル。
        """
        if x.numel() == 0:
            return
        xd = x.to(torch.float64)
        batch_mean = xd.mean(dim=0)
        batch_var = xd.var(dim=0, unbiased=False) if xd.shape[0] > 1 else torch.zeros_like(batch_mean)
        batch_count = float(xd.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        """内部ヘルパー: `update_from_moments` を実行する。

        Args:
            batch_mean (torch.Tensor): batch_mean の値。
            batch_var (torch.Tensor): batch_var の値。
            batch_count (float): batch_count の値。
        """
        if batch_count <= 0:
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / total)
        self.var = torch.clamp(m2 / total, min=1e-12)
        self.count = float(total)

    def state_dict(self) -> dict[str, Any]:
        """`state_dict` を実行する。

        Returns:
            dict[str, Any]: 計算結果。
        """
        return {
            "mean": self.mean.numpy().copy(),
            "var": self.var.numpy().copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """`state_dict`を読み込む。

        Args:
            state (dict[str, Any]): state の値。
        """
        self.mean = torch.as_tensor(np.asarray(state.get("mean", self.mean), dtype=np.float64)).clone()
        self.var = torch.clamp(
            torch.as_tensor(np.asarray(state.get("var", self.var), dtype=np.float64)), min=1e-12
        ).clone()
        self.count = float(state.get("count", self.count))


class VecNormalize:
    """`VecNormalize` を表すクラス。"""

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
        """インスタンスを初期化する。

        Args:
            num_envs (int): 環境関連の値。
            obs_shape (tuple[int, ...]): obs_shape の値。
            norm_obs (bool): 有効化フラグ。
            norm_reward (bool): 有効化フラグ。
            clip_obs (float): clip_obs の値。
            clip_reward (float): clip_reward の値。
            epsilon (float): epsilon の値。
            gamma (float): gamma の値。
            training (bool): 有効化フラグ。
        """
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
        self.returns = torch.zeros(int(num_envs), dtype=torch.float64)

    def set_training(self, mode: bool) -> None:
        """`training`を設定する。

        Args:
            mode (bool): 有効化フラグ。
        """
        self.training = bool(mode)

    def normalize_obs_inplace(self, obs: torch.Tensor) -> None:
        """`obs_inplace`を正規化する。

        Args:
            obs (torch.Tensor): 入力テンソル。
        """
        if obs.device.type != "cpu":
            raise ValueError("VecNormalize expects CPU observation tensor")

        view = obs.reshape(-1, *self.obs_shape).to(torch.float64)
        if self.training and self.norm_obs:
            self.obs_rms.update(view)
        if not self.norm_obs:
            return

        denom = torch.sqrt(self.obs_rms.var + self.epsilon)
        normalized = ((view - self.obs_rms.mean) / denom).clamp(-self.clip_obs, self.clip_obs)
        obs.copy_(normalized.to(obs.dtype).reshape_as(obs))

    def normalize_reward_inplace(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        """`reward_inplace`を正規化する。

        Args:
            reward (torch.Tensor): reward の値。
            done (torch.Tensor): done の値。
        """
        if reward.device.type != "cpu" or done.device.type != "cpu":
            raise ValueError("VecNormalize expects CPU reward/done tensors")

        r = reward.reshape(-1).to(torch.float64)
        d = done.reshape(-1).bool()
        n = r.shape[0]

        if self.returns.shape[0] != n:
            self.returns = torch.zeros(n, dtype=torch.float64)

        if self.training and self.norm_reward:
            self.returns = self.returns * self.gamma + r
            self.ret_rms.update(self.returns)

        if self.norm_reward:
            scale = torch.sqrt(self.ret_rms.var + self.epsilon)
            normalized = (r / scale).clamp(-self.clip_reward, self.clip_reward)
            reward.copy_(normalized.to(reward.dtype).reshape_as(reward))

        self.returns[d] = 0.0

    def state_dict(self) -> dict[str, Any]:
        """`state_dict` を実行する。

        Returns:
            dict[str, Any]: 計算結果。
        """
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
            "returns": self.returns.numpy().copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """`state_dict`を読み込む。

        Args:
            state (dict[str, Any]): state の値。
        """
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

        stored = state.get("returns")
        if stored is not None:
            returns = torch.as_tensor(np.asarray(stored, dtype=np.float64)).reshape(-1)
            if returns.numel() == self.returns.numel():
                self.returns = returns.clone()
            elif returns.numel() == 1:
                self.returns.fill_(float(returns[0]))
            else:
                self.returns.zero_()
