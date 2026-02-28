"""`exp002_resnet` に関するモデル処理。"""
from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from ...game_constants import AUX_OPP_PARAM_TOTAL, OPP_PARAM_DIM, OPP_SLOT_COUNT


def _pick_gn_groups(channels: int) -> int:
    """内部ヘルパー: `pick_gn_groups` を実行する。

    Args:
        channels (int): channels の値。

    Returns:
        int: 計算結果。
    """
    for g in (8, 4, 2, 1):
        if int(channels) % g == 0:
            return g
    return 1


class _ResidualBlock(nn.Module):
    """`_ResidualBlock` を表すクラス。"""
    def __init__(self, channels: int):
        """インスタンスを初期化する。

        Args:
            channels (int): channels の値。
        """
        super().__init__()
        g = _pick_gn_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`forward` を実行する。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.silu(y)
        y = self.conv2(y)
        y = self.gn2(y)
        return F.silu(x + y)


class Exp002ResNetBoardAgent(nn.Module):
    """`Exp002ResNetBoardAgent` を表すクラス。"""

    model_type: str = "exp002_resnet_v1"

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        *,
        board_channels: int = 46,
        board_size: int = 10,
        global_dim: int = 0,
        hidden_channels: int = 64,
        blocks: int = 6,
        aux_opp_param_head: bool = False,
        aux_opp_param_hidden_dim: int | None = None,
    ):
        """インスタンスを初期化する。

        Args:
            obs_shape (tuple[int, ...]): obs_shape の値。
            action_dim (int): action_dim の値。
            board_channels (int): board_channels の値。
            board_size (int): board_size の値。
            global_dim (int): global_dim の値。
            hidden_channels (int): hidden_channels の値。
            blocks (int): blocks の値。
            aux_opp_param_head (bool): 有効化フラグ。
            aux_opp_param_hidden_dim (int | None): aux_opp_param_hidden_dim の値。
        """
        super().__init__()
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if int(board_channels) <= 0:
            raise ValueError("board_channels must be positive")
        if int(board_size) <= 0:
            raise ValueError("board_size must be positive")
        if int(hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if int(blocks) < 0:
            raise ValueError("blocks must be >= 0")

        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)
        self.board_size = int(board_size)
        self.hidden_channels = int(hidden_channels)
        self.blocks = int(blocks)
        self.use_aux_opp_param_head = bool(aux_opp_param_head)

        obs_dim = int(np.prod(self.obs_shape))
        board_area = int(self.board_size * self.board_size)
        requested_board_channels = int(board_channels)
        resolved_board_channels = int(requested_board_channels)
        if int(global_dim) <= 0:
            requested_board_dim = int(requested_board_channels * board_area)
            if obs_dim < requested_board_dim:
                if obs_dim % board_area != 0:
                    raise ValueError(
                        f"obs_dim too small: obs_dim={obs_dim}, requested_board_dim={requested_board_dim}, "
                        f"and obs_dim is not divisible by board_area={board_area}"
                    )
                inferred_channels = int(obs_dim // board_area)
                if inferred_channels <= 0:
                    raise ValueError(f"inferred board_channels must be positive, got: {inferred_channels}")
                resolved_board_channels = inferred_channels
                warnings.warn(
                    "Exp002ResNetBoardAgent: auto-adjusted board_channels "
                    f"from {requested_board_channels} to {resolved_board_channels} "
                    f"to match obs_shape={self.obs_shape}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.global_dim = int(obs_dim - resolved_board_channels * board_area)
        else:
            self.global_dim = int(global_dim)

        self.board_channels = int(resolved_board_channels)
        self.board_dim = int(self.board_channels * board_area)
        if self.global_dim < 0 or self.board_dim + self.global_dim != obs_dim:
            raise ValueError(
                f"invalid global_dim={self.global_dim} for obs_dim={obs_dim}, board_dim={self.board_dim}"
            )
        if self.global_dim != 0:
            raise ValueError(
                "Exp002ResNetBoardAgent is board-only and requires global_dim=0 "
                f"(got global_dim={self.global_dim})"
            )
        if self.action_dim != self.board_size * self.board_size:
            raise ValueError(
                "Exp002ResNetBoardAgent expects action_dim == board_size * board_size "
                f"(got action_dim={self.action_dim}, board={self.board_size}x{self.board_size})"
            )

        g = _pick_gn_groups(self.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.board_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, self.hidden_channels),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(*[_ResidualBlock(self.hidden_channels) for _ in range(self.blocks)])
        self.policy_head = nn.Conv2d(self.hidden_channels, 1, kernel_size=1, padding=0, bias=True)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1),
        )
        self.aux_opp_param_head: nn.Sequential | None = None
        aux_hidden = int(aux_opp_param_hidden_dim) if aux_opp_param_hidden_dim is not None else self.hidden_channels
        if self.use_aux_opp_param_head:
            if aux_hidden <= 0:
                raise ValueError(f"aux_opp_param_hidden_dim must be positive, got: {aux_hidden}")
            self.aux_opp_param_head = nn.Sequential(
                nn.Linear(self.hidden_channels, aux_hidden),
                nn.SiLU(),
                nn.Linear(aux_hidden, AUX_OPP_PARAM_TOTAL),
            )

        kwargs: dict[str, Any] = {
            "board_channels": int(self.board_channels),
            "board_size": int(self.board_size),
            "global_dim": int(self.global_dim),
            "hidden_channels": int(self.hidden_channels),
            "blocks": int(self.blocks),
            "aux_opp_param_head": bool(self.use_aux_opp_param_head),
        }
        if aux_opp_param_hidden_dim is not None:
            kwargs["aux_opp_param_hidden_dim"] = int(aux_hidden)
        self.model_config: dict[str, Any] = {
            "type": "Exp002ResNetBoardAgent",
            "kwargs": kwargs,
        }

    def _flatten_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """内部ヘルパー: `flatten_obs` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        return obs.reshape(obs.shape[0], -1)

    def _split_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """内部ヘルパー: `split_obs` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        x = self._flatten_obs(obs)
        return x[:, : self.board_dim].view(x.shape[0], self.board_channels, self.board_size, self.board_size)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """内部ヘルパー: `encode` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        h = self.stem(self._split_obs(obs))
        return self.res_blocks(h)

    @staticmethod
    def _pool(h: torch.Tensor) -> torch.Tensor:
        """内部ヘルパー: `pool` を実行する。

        Args:
            h (torch.Tensor): h の値。

        Returns:
            torch.Tensor: 計算結果。
        """
        return h.mean(dim=(2, 3))

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """`logits`を取得する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        h = self._encode(obs)
        return self.policy_head(h).flatten(1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """`value`を取得する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        h = self._encode(obs)
        return self.value_head(self._pool(h))

    def get_aux_opp_param(self, obs: torch.Tensor) -> torch.Tensor:
        """`aux_opp_param`を取得する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        if self.aux_opp_param_head is None:
            raise RuntimeError("aux_opp_param_head is disabled for this model")
        h = self._encode(obs)
        out = self.aux_opp_param_head(self._pool(h))
        return out.view(out.shape[0], OPP_SLOT_COUNT, OPP_PARAM_DIM)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        return_aux_opp_param: bool = False,
    ):
        """`action_and_value`を取得する。

        Args:
            obs (torch.Tensor): 入力テンソル。
            action (torch.Tensor | None): action の値。
            action_mask (torch.Tensor | None): action_mask の値。
            return_aux_opp_param (bool): 有効化フラグ。

        Returns:
            Any: 計算結果。
        """
        h = self._encode(obs)
        logits = self.policy_head(h).flatten(1)

        if action_mask is not None:
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape != logits.shape:
                raise ValueError(f"action_mask shape mismatch: mask={action_mask.shape}, logits={logits.shape}")
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        pooled = self._pool(h)
        value = self.value_head(pooled)
        if bool(return_aux_opp_param):
            if self.aux_opp_param_head is None:
                raise RuntimeError("aux_opp_param_head is disabled for this model")
            aux = self.aux_opp_param_head(pooled).view(pooled.shape[0], OPP_SLOT_COUNT, OPP_PARAM_DIM)
            return action, dist.log_prob(action), dist.entropy(), value, aux
        return action, dist.log_prob(action), dist.entropy(), value

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        return_aux_opp_param: bool = False,
    ):
        """`forward` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。
            action (torch.Tensor | None): action の値。
            action_mask (torch.Tensor | None): action_mask の値。
            return_aux_opp_param (bool): 有効化フラグ。

        Returns:
            Any: 計算結果。
        """
        return self.get_action_and_value(
            obs,
            action=action,
            action_mask=action_mask,
            return_aux_opp_param=bool(return_aux_opp_param),
        )

    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """`act` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。
            action_mask (torch.Tensor | None): action_mask の値。
            deterministic (bool): 有効化フラグ。

        Returns:
            torch.Tensor: 計算結果。
        """
        logits = self.get_logits(obs)
        if action_mask is not None:
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()
