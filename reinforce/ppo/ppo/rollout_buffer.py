"""PPO 学習用のロールアウトバッファを管理するモジュール。

ロールアウト中に収集した観測・行動・報酬を保持し、GAE 計算と
ミニバッチ学習向けの平坦化を提供する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FlattenedBatch:
    """`RolloutBuffer` の平坦化後バッチを表すデータクラス。

    Attributes:
        obs (torch.Tensor): 観測テンソル `[T*B, *obs_shape]`。
        actions (torch.Tensor): 行動テンソル `[T*B, *action_shape]`。
        logprobs (torch.Tensor): 収集時の行動対数確率 `[T*B]`。
        advantages (torch.Tensor): GAE で計算したアドバンテージ `[T*B]`。
        returns (torch.Tensor): 目標価値（リターン）`[T*B]`。
        values (torch.Tensor): 収集時の価値推定 `[T*B]`。
        action_masks (torch.Tensor | None): 行動マスク `[T*B, action_dim]`。
        aux_opp_param_true (torch.Tensor | None): 補助教師信号 `[T*B, slots, param_dim]`。
        aux_opp_valid (torch.Tensor | None): 補助教師信号の有効フラグ `[T*B, slots]`。
    """
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    action_masks: torch.Tensor | None
    aux_opp_param_true: torch.Tensor | None
    aux_opp_valid: torch.Tensor | None


class RolloutBuffer:
    """固定長ロールアウトを格納し、学習入力へ変換するバッファ。"""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: torch.device,
        use_action_mask: bool = False,
        action_dim: int = 0,
        use_aux_opp_param_targets: bool = False,
        aux_opp_slot_count: int = 7,
        aux_opp_param_dim: int = 5,
    ):
        """ロールアウトバッファを初期化する。

        Args:
            num_steps (int): 1 ロールアウトあたりの時間ステップ数 `T`。
            num_envs (int): 並列環境数 `B`。
            obs_shape (tuple[int, ...]): 観測テンソル 1 サンプル分の形状。
            action_shape (tuple[int, ...]): 行動テンソル 1 サンプル分の形状。
            device (torch.device): バッファ確保先デバイス。
            use_action_mask (bool): 行動マスクを保持する場合は `True`。
            action_dim (int): 行動空間次元。`use_action_mask=True` 時に使用。
            use_aux_opp_param_targets (bool): 補助教師信号バッファを確保するか。
            aux_opp_slot_count (int): 補助教師信号のスロット数。
            aux_opp_param_dim (int): 補助教師信号 1 スロットあたりの次元。

        Raises:
            ValueError: `use_action_mask=True` で `action_dim <= 0` の場合。
            ValueError: 補助教師信号を有効化したのにスロット数や次元が不正な場合。
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, device=device, dtype=torch.long)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)

        if use_action_mask:
            if action_dim <= 0:
                raise ValueError("action_dim must be positive when use_action_mask=True")
            self.action_masks: torch.Tensor | None = torch.zeros((num_steps, num_envs, action_dim), device=device, dtype=torch.bool)
        else:
            self.action_masks = None
        if use_aux_opp_param_targets:
            slots = int(aux_opp_slot_count)
            param_dim = int(aux_opp_param_dim)
            if slots <= 0 or param_dim <= 0:
                raise ValueError(
                    "aux_opp_slot_count and aux_opp_param_dim must be positive when use_aux_opp_param_targets=True"
                )
            self.aux_opp_param_true: torch.Tensor | None = torch.zeros(
                (num_steps, num_envs, slots, param_dim),
                device=device,
                dtype=torch.float32,
            )
            self.aux_opp_valid: torch.Tensor | None = torch.zeros(
                (num_steps, num_envs, slots),
                device=device,
                dtype=torch.bool,
            )
        else:
            self.aux_opp_param_true = None
            self.aux_opp_valid = None

    def add(
        self,
        step: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> None:
        """1 ステップ分の遷移データをバッファへ書き込む。

        Args:
            step (int): 書き込み対象ステップ index（`0 <= step < num_steps`）。
            obs (torch.Tensor): 観測 `[B, *obs_shape]`。
            action (torch.Tensor): 行動 `[B, *action_shape]`。
            logprob (torch.Tensor): 行動の対数確率 `[B]`。
            reward (torch.Tensor): 即時報酬 `[B]`。
            done (torch.Tensor): 終端フラグ `[B]`。
            value (torch.Tensor): 価値推定 `[B]`。
            action_mask (torch.Tensor | None): 行動マスク `[B, action_dim]`。
        """
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value.view(-1)
        if self.action_masks is not None and action_mask is not None:
            self.action_masks[step] = action_mask

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """GAE-Lambda により `advantages` と `returns` を計算する。

        Args:
            next_value (torch.Tensor): ロールアウト末尾直後の価値推定 `[B]`。
            next_done (torch.Tensor): ロールアウト末尾直後の終端フラグ `[B]`。
            gamma (float): 割引率。
            gae_lambda (float): GAE の平滑化係数。
        """
        lastgaelam = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            self.advantages[t] = lastgaelam
        self.returns = self.advantages + self.values

    def flatten(self) -> FlattenedBatch:
        """`[T, B, ...]` 形式のバッファを `[T*B, ...]` に平坦化する。

        Returns:
            FlattenedBatch: ミニバッチ分割しやすい平坦化済みテンソル群。
        """
        mask = self.action_masks.reshape((-1, self.action_masks.shape[-1])) if self.action_masks is not None else None
        aux_opp_param_true = (
            self.aux_opp_param_true.reshape((-1, self.aux_opp_param_true.shape[-2], self.aux_opp_param_true.shape[-1]))
            if self.aux_opp_param_true is not None
            else None
        )
        aux_opp_valid = (
            self.aux_opp_valid.reshape((-1, self.aux_opp_valid.shape[-1])) if self.aux_opp_valid is not None else None
        )
        return FlattenedBatch(
            obs=self.obs.reshape((-1,) + self.obs.shape[2:]),
            actions=self.actions.reshape((-1,) + self.actions.shape[2:]),
            logprobs=self.logprobs.reshape(-1),
            advantages=self.advantages.reshape(-1),
            returns=self.returns.reshape(-1),
            values=self.values.reshape(-1),
            action_masks=mask,
            aux_opp_param_true=aux_opp_param_true,
            aux_opp_valid=aux_opp_valid,
        )

    def shuffled_indices(self) -> np.ndarray:
        """平坦化バッチ（`T*B`）のシャッフル index を返す。

        Returns:
            np.ndarray: `0..T*B-1` のランダム順列。
        """
        return np.random.permutation(self.num_steps * self.num_envs)
