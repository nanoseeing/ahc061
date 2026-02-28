"""`student_m` に関するモデル処理。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ...game_constants import AUX_OPP_PARAM_TOTAL, OPP_PARAM_DIM, OPP_SLOT_COUNT
from ..utils.nn_init import layer_init, to_int_tuple


def _build_activation(name: str) -> nn.Module:
    """内部ヘルパー: `build_activation` を実行する。

    Args:
        name (str): name の値。

    Returns:
        nn.Module: 計算結果。
    """
    key = str(name).strip().lower()
    if key in ("tanh", ""):
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}; expected tanh|relu|silu")


class _ResidualBlock(nn.Module):
    """`_ResidualBlock` を表すクラス。"""
    def __init__(self, width: int, *, activation: str = "tanh"):
        """インスタンスを初期化する。

        Args:
            width (int): width の値。
            activation (str): activation の値。
        """
        super().__init__()
        act = _build_activation(activation)
        self.conv1 = layer_init(nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1))
        self.act1 = act
        self.act2 = _build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`forward` を実行する。

        Args:
            x (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        h = self.act1(self.conv1(x))
        h = self.conv2(h)
        return self.act2(x + h)


class StudentMBoardAgent(nn.Module):
    """`StudentMBoardAgent` を表すクラス。"""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        *,
        board_channels: int = 46,
        board_size: int = 10,
        global_dim: int = 0,
        width: int = 48,
        num_blocks: int = 2,
        global_hidden_dim: int = 64,
        value_hidden_dims: tuple[int, ...] | list[int] | str = (64,),
        activation: str = "tanh",
        use_global_film: bool = True,
        use_global_policy_bias: bool = True,
        aux_opp_param_head: bool = False,
        aux_opp_param_hidden_dim: int = 64,
    ):
        """インスタンスを初期化する。

        Args:
            obs_shape (tuple[int, ...]): obs_shape の値。
            action_dim (int): action_dim の値。
            board_channels (int): board_channels の値。
            board_size (int): board_size の値。
            global_dim (int): global_dim の値。
            width (int): width の値。
            num_blocks (int): num_blocks の値。
            global_hidden_dim (int): global_hidden_dim の値。
            value_hidden_dims (tuple[int, ...] | list[int] | str): value_hidden_dims の値。
            activation (str): activation の値。
            use_global_film (bool): 有効化フラグ。
            use_global_policy_bias (bool): 有効化フラグ。
            aux_opp_param_head (bool): 有効化フラグ。
            aux_opp_param_hidden_dim (int): aux_opp_param_hidden_dim の値。
        """
        super().__init__()
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be positive")
        if int(width) <= 0:
            raise ValueError("width must be positive")
        if int(num_blocks) < 0:
            raise ValueError("num_blocks must be >= 0")

        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)
        self.board_channels = int(board_channels)
        self.board_size = int(board_size)
        self.board_dim = int(self.board_channels * self.board_size * self.board_size)

        obs_dim = int(np.prod(self.obs_shape))
        if obs_dim < self.board_dim:
            raise ValueError(f"obs_dim too small: obs_dim={obs_dim}, board_dim={self.board_dim}")

        if int(global_dim) <= 0:
            self.global_dim = int(obs_dim - self.board_dim)
        else:
            self.global_dim = int(global_dim)
        if self.global_dim < 0 or self.board_dim + self.global_dim != obs_dim:
            raise ValueError(
                f"invalid global_dim={self.global_dim} for obs_dim={obs_dim}, board_dim={self.board_dim}"
            )

        if self.action_dim != self.board_size * self.board_size:
            raise ValueError(
                "StudentMBoardAgent currently expects action_dim == board_size * board_size "
                f"(got action_dim={self.action_dim}, board={self.board_size}x{self.board_size})"
            )

        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.global_hidden_dim = int(max(1, global_hidden_dim))
        self.use_global_film = bool(use_global_film and self.global_dim > 0)
        self.use_global_policy_bias = bool(use_global_policy_bias and self.global_dim > 0)
        self.use_aux_opp_param_head = bool(aux_opp_param_head)
        self.aux_opp_param_hidden_dim = int(max(1, aux_opp_param_hidden_dim))
        self.activation = str(activation)
        value_hidden_dims_t = to_int_tuple(value_hidden_dims)

        self.stem = nn.Sequential(
            layer_init(nn.Conv2d(self.board_channels, self.width, kernel_size=3, stride=1, padding=1)),
            _build_activation(self.activation),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(self.width, activation=self.activation) for _ in range(self.num_blocks)]
        )

        self.global_mlp: nn.Module | None = None
        self.film: nn.Linear | None = None
        self.policy_global_bias: nn.Linear | None = None
        if self.global_dim > 0:
            self.global_mlp = nn.Sequential(
                layer_init(nn.Linear(self.global_dim, self.global_hidden_dim)),
                _build_activation(self.activation),
                layer_init(nn.Linear(self.global_hidden_dim, self.global_hidden_dim)),
                _build_activation(self.activation),
            )
            if self.use_global_film:
                self.film = layer_init(nn.Linear(self.global_hidden_dim, 2 * self.width), std=0.01)
            if self.use_global_policy_bias:
                self.policy_global_bias = layer_init(nn.Linear(self.global_hidden_dim, self.action_dim), std=0.01)

        self.policy_conv = layer_init(nn.Conv2d(self.width, 1, kernel_size=1, stride=1, padding=0), std=0.01)

        value_in_dim = self.width + (self.global_hidden_dim if self.global_dim > 0 else 0)
        v_layers: list[nn.Module] = []
        cur = int(value_in_dim)
        for h in value_hidden_dims_t:
            if h <= 0:
                raise ValueError(f"value hidden dim must be positive, got: {h}")
            v_layers.append(layer_init(nn.Linear(cur, h)))
            v_layers.append(_build_activation(self.activation))
            cur = h
        v_layers.append(layer_init(nn.Linear(cur, 1), std=1.0))
        self.value_head = nn.Sequential(*v_layers)
        self.aux_opp_param_head: nn.Sequential | None = None
        if self.use_aux_opp_param_head:
            self.aux_opp_param_head = nn.Sequential(
                layer_init(nn.Linear(value_in_dim, self.aux_opp_param_hidden_dim)),
                _build_activation(self.activation),
                layer_init(nn.Linear(self.aux_opp_param_hidden_dim, AUX_OPP_PARAM_TOTAL), std=0.01),
            )

        self.model_config: dict[str, Any] = {
            "type": "StudentMBoardAgent",
            "kwargs": {
                "board_channels": int(self.board_channels),
                "board_size": int(self.board_size),
                "global_dim": int(self.global_dim),
                "width": int(self.width),
                "num_blocks": int(self.num_blocks),
                "global_hidden_dim": int(self.global_hidden_dim),
                "value_hidden_dims": list(value_hidden_dims_t),
                "activation": str(self.activation),
                "use_global_film": bool(self.use_global_film),
                "use_global_policy_bias": bool(self.use_global_policy_bias),
                "aux_opp_param_head": bool(self.use_aux_opp_param_head),
                "aux_opp_param_hidden_dim": int(self.aux_opp_param_hidden_dim),
            },
        }

    def _flatten_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # reshape handles non-standard strides (e.g., channels_last) safely.
        """内部ヘルパー: `flatten_obs` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        return obs.reshape(obs.shape[0], -1)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """内部ヘルパー: `split_obs` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: 計算結果。
        """
        x = self._flatten_obs(obs)
        board = x[:, : self.board_dim].view(x.shape[0], self.board_channels, self.board_size, self.board_size)
        g = x[:, self.board_dim : self.board_dim + self.global_dim] if self.global_dim > 0 else None
        return board, g

    def _encode_board(self, board: torch.Tensor, global_emb: torch.Tensor | None) -> torch.Tensor:
        """内部ヘルパー: `encode_board` を実行する。

        Args:
            board (torch.Tensor): 入力テンソル。
            global_emb (torch.Tensor | None): global_emb の値。

        Returns:
            torch.Tensor: 計算結果。
        """
        h = self.stem(board)
        h = self.blocks(h)
        if self.film is not None and global_emb is not None:
            film = self.film(global_emb)
            gamma = film[:, : self.width].view(film.shape[0], self.width, 1, 1)
            beta = film[:, self.width :].view(film.shape[0], self.width, 1, 1)
            h = h * (1.0 + gamma) + beta
        return h

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """内部ヘルパー: `encode` を実行する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: 計算結果。
        """
        board, g = self._split_obs(obs)
        g_emb = self.global_mlp(g) if (self.global_mlp is not None and g is not None) else None
        h = self._encode_board(board, g_emb)
        return h, g_emb

    @staticmethod
    def _merge_value_input(h: torch.Tensor, g_emb: torch.Tensor | None) -> torch.Tensor:
        """内部ヘルパー: `merge_value_input` を実行する。

        Args:
            h (torch.Tensor): h の値。
            g_emb (torch.Tensor | None): g_emb の値。

        Returns:
            torch.Tensor: 計算結果。
        """
        pooled = h.mean(dim=(2, 3))
        if g_emb is not None:
            pooled = torch.cat([pooled, g_emb], dim=1)
        return pooled

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """`logits`を取得する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        h, g_emb = self._encode(obs)
        logits = self.policy_conv(h).reshape(h.shape[0], -1)
        if self.policy_global_bias is not None and g_emb is not None:
            logits = logits + self.policy_global_bias(g_emb)
        return logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """`value`を取得する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        h, g_emb = self._encode(obs)
        pooled = self._merge_value_input(h, g_emb)
        return self.value_head(pooled)

    def get_aux_opp_param(self, obs: torch.Tensor) -> torch.Tensor:
        """`aux_opp_param`を取得する。

        Args:
            obs (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 計算結果。
        """
        if self.aux_opp_param_head is None:
            raise RuntimeError("aux_opp_param_head is disabled for this model")
        h, g_emb = self._encode(obs)
        pooled = self._merge_value_input(h, g_emb)
        out = self.aux_opp_param_head(pooled)
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
        h, g_emb = self._encode(obs)
        logits = self.policy_conv(h).reshape(h.shape[0], -1)
        if self.policy_global_bias is not None and g_emb is not None:
            logits = logits + self.policy_global_bias(g_emb)

        if action_mask is not None:
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape != logits.shape:
                raise ValueError(f"action_mask shape mismatch: mask={action_mask.shape}, logits={logits.shape}")
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        pooled = self._merge_value_input(h, g_emb)
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
        # Keep PPO update path on module forward so DDP hooks are triggered.
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


class TeacherP0V1BoardAgent(StudentMBoardAgent):
    """`TeacherP0V1BoardAgent` を表すクラス。"""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        *,
        board_channels: int = 88,
        board_size: int = 10,
        global_dim: int = 0,
        width: int = 64,
        num_blocks: int = 4,
        global_hidden_dim: int = 64,
        value_hidden_dims: tuple[int, ...] | list[int] | str = (128, 64),
        activation: str = "tanh",
        use_global_film: bool = False,
        use_global_policy_bias: bool = False,
    ):
        """インスタンスを初期化する。

        Args:
            obs_shape (tuple[int, ...]): obs_shape の値。
            action_dim (int): action_dim の値。
            board_channels (int): board_channels の値。
            board_size (int): board_size の値。
            global_dim (int): global_dim の値。
            width (int): width の値。
            num_blocks (int): num_blocks の値。
            global_hidden_dim (int): global_hidden_dim の値。
            value_hidden_dims (tuple[int, ...] | list[int] | str): value_hidden_dims の値。
            activation (str): activation の値。
            use_global_film (bool): 有効化フラグ。
            use_global_policy_bias (bool): 有効化フラグ。
        """
        super().__init__(
            obs_shape=obs_shape,
            action_dim=action_dim,
            board_channels=board_channels,
            board_size=board_size,
            global_dim=global_dim,
            width=width,
            num_blocks=num_blocks,
            global_hidden_dim=global_hidden_dim,
            value_hidden_dims=value_hidden_dims,
            activation=activation,
            use_global_film=use_global_film,
            use_global_policy_bias=use_global_policy_bias,
        )
        self.model_config = {
            "type": "TeacherP0V1BoardAgent",
            "kwargs": {
                "board_channels": int(self.board_channels),
                "board_size": int(self.board_size),
                "global_dim": int(self.global_dim),
                "width": int(self.width),
                "num_blocks": int(self.num_blocks),
                "global_hidden_dim": int(self.global_hidden_dim),
                "value_hidden_dims": list(to_int_tuple(value_hidden_dims)),
                "activation": str(self.activation),
                "use_global_film": bool(self.use_global_film),
                "use_global_policy_bias": bool(self.use_global_policy_bias),
            },
        }
