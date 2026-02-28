from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from ...game_constants import AUX_OPP_PARAM_TOTAL, OPP_PARAM_DIM, OPP_SLOT_COUNT
from ..utils.nn_init import layer_init, to_int_tuple


def _broadcast_tuple(v: tuple[int, ...], n: int, name: str) -> tuple[int, ...]:
    if len(v) == n:
        return v
    if len(v) == 1 and n > 1:
        return tuple(v[0] for _ in range(n))
    raise ValueError(f"{name} length mismatch: expected 1 or {n}, got {len(v)}")


def _build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, *, out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = in_dim
    for h in hidden_dims:
        if h <= 0:
            raise ValueError(f"hidden dim must be positive, got: {h}")
        layers.append(layer_init(nn.Linear(cur, h)))
        layers.append(nn.Tanh())
        cur = h
    layers.append(layer_init(nn.Linear(cur, out_dim), std=out_std))
    return nn.Sequential(*layers)


class DiscreteBoardAgentBase(nn.Module, ABC):
    """Shared forward interface for all discrete-board actor-critic agents.

    Concrete subclasses must implement `_encode(obs)` and expose:
      - `action_dim: int`
      - `actor: nn.Module`
      - `critic: nn.Module`
      - `aux_opp_param_head: nn.Module | None`
      - `use_aux_opp_param_head: bool`
    """

    action_dim: int
    actor: nn.Module
    critic: nn.Module
    aux_opp_param_head: nn.Module | None
    use_aux_opp_param_head: bool

    def _flatten_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # reshape handles non-standard strides (e.g., channels_last) safely.
        return obs.reshape(obs.shape[0], -1)

    @abstractmethod
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        ...

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self._encode(obs)
        return self.actor(feat)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self._encode(obs)
        return self.critic(feat)

    def get_aux_opp_param(self, obs: torch.Tensor) -> torch.Tensor:
        if self.aux_opp_param_head is None:
            raise RuntimeError("aux_opp_param_head is disabled for this model")
        feat = self._encode(obs)
        out = self.aux_opp_param_head(feat)
        return out.view(out.shape[0], OPP_SLOT_COUNT, OPP_PARAM_DIM)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        return_aux_opp_param: bool = False,
    ):
        feat = self._encode(obs)
        logits = self.actor(feat)

        if action_mask is not None:
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape != logits.shape:
                raise ValueError(f"action_mask shape mismatch: mask={action_mask.shape}, logits={logits.shape}")
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        value = self.critic(feat)
        if bool(return_aux_opp_param):
            if self.aux_opp_param_head is None:
                raise RuntimeError("aux_opp_param_head is disabled for this model")
            aux = self.aux_opp_param_head(feat).view(feat.shape[0], OPP_SLOT_COUNT, OPP_PARAM_DIM)
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
        logits = self.get_logits(obs)
        if action_mask is not None:
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        if deterministic:
            return torch.argmax(logits, dim=-1)
        dist = Categorical(logits=logits)
        return dist.sample()


class DiscreteBoardMLPAgent(DiscreteBoardAgentBase):
    """Flat MLP actor-critic for discrete board-game actions."""

    model_type: str = "mlp"

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        *,
        mlp_hidden_dims: tuple[int, ...] | list[int] | str | None = None,
        aux_opp_param_head: bool = False,
        aux_opp_param_hidden_dim: int = 64,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        obs_dim = int(np.prod(obs_shape))
        self.obs_shape = obs_shape
        self.action_dim = int(action_dim)
        self.use_aux_opp_param_head = bool(aux_opp_param_head)
        self.aux_opp_param_hidden_dim = int(max(1, aux_opp_param_hidden_dim))

        if mlp_hidden_dims is None:
            mlp_hidden_dims_t = tuple(int(hidden_dim) for _ in range(int(hidden_layers)))
        else:
            mlp_hidden_dims_t = to_int_tuple(mlp_hidden_dims)

        self.actor = _build_mlp(obs_dim, mlp_hidden_dims_t, self.action_dim, out_std=0.01)
        self.critic = _build_mlp(obs_dim, mlp_hidden_dims_t, 1, out_std=1.0)
        self._encoded_dim = int(obs_dim)
        self.aux_opp_param_head: nn.Module | None = None

        if self.use_aux_opp_param_head:
            self.aux_opp_param_head = nn.Sequential(
                layer_init(nn.Linear(self._encoded_dim, self.aux_opp_param_hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(self.aux_opp_param_hidden_dim, AUX_OPP_PARAM_TOTAL), std=0.01),
            )

        self.model_config: dict[str, Any] = {
            "type": "DiscreteBoardMLPAgent",
            "kwargs": {
                "model_type": "mlp",
                "mlp_hidden_dims": list(mlp_hidden_dims_t),
                "aux_opp_param_head": bool(self.use_aux_opp_param_head),
                "aux_opp_param_hidden_dim": int(self.aux_opp_param_hidden_dim),
            },
        }

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self._flatten_obs(obs)


class DiscreteBoardCNNFCAgent(DiscreteBoardAgentBase):
    """Board CNN + global FC fusion actor-critic for discrete board-game actions."""

    model_type: str = "cnn_fc"

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        *,
        board_channels: int = 46,
        board_size: int = 10,
        global_dim: int = 0,
        cnn_channels: tuple[int, ...] | list[int] | str = (32, 64),
        cnn_kernels: tuple[int, ...] | list[int] | str = (3, 3),
        cnn_strides: tuple[int, ...] | list[int] | str = (1, 1),
        cnn_paddings: tuple[int, ...] | list[int] | str = (1, 1),
        trunk_hidden_dims: tuple[int, ...] | list[int] | str = (256, 128),
        actor_head_hidden_dims: tuple[int, ...] | list[int] | str = (),
        critic_head_hidden_dims: tuple[int, ...] | list[int] | str = (),
        aux_opp_param_head: bool = False,
        aux_opp_param_hidden_dim: int = 64,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        obs_dim = int(np.prod(obs_shape))
        self.obs_shape = obs_shape
        self.action_dim = int(action_dim)
        self.use_aux_opp_param_head = bool(aux_opp_param_head)
        self.aux_opp_param_hidden_dim = int(max(1, aux_opp_param_hidden_dim))

        self.board_channels = int(board_channels)
        self.board_size = int(board_size)
        self.board_dim = int(self.board_channels * self.board_size * self.board_size)

        cnn_channels_t = to_int_tuple(cnn_channels)
        cnn_kernels_t = to_int_tuple(cnn_kernels)
        cnn_strides_t = to_int_tuple(cnn_strides)
        cnn_paddings_t = to_int_tuple(cnn_paddings)
        trunk_hidden_dims_t = to_int_tuple(trunk_hidden_dims)
        actor_head_hidden_dims_t = to_int_tuple(actor_head_hidden_dims)
        critic_head_hidden_dims_t = to_int_tuple(critic_head_hidden_dims)

        if obs_dim < self.board_dim:
            raise ValueError(f"obs_dim too small for cnn_fc: obs_dim={obs_dim}, board_dim={self.board_dim}")

        if int(global_dim) <= 0:
            self.global_dim = int(obs_dim - self.board_dim)
        else:
            self.global_dim = int(global_dim)
        if self.global_dim < 0 or self.board_dim + self.global_dim != obs_dim:
            raise ValueError(
                f"invalid global_dim={self.global_dim} for obs_dim={obs_dim}, board_dim={self.board_dim}"
            )

        if len(cnn_channels_t) == 0:
            raise ValueError("cnn_channels must not be empty for model_type=cnn_fc")
        n_conv = len(cnn_channels_t)
        kernels = _broadcast_tuple(cnn_kernels_t, n_conv, "cnn_kernels")
        strides = _broadcast_tuple(cnn_strides_t, n_conv, "cnn_strides")
        paddings = _broadcast_tuple(cnn_paddings_t, n_conv, "cnn_paddings")

        conv_layers: list[nn.Module] = []
        in_ch = self.board_channels
        for out_ch, k, s, p in zip(cnn_channels_t, kernels, strides, paddings):
            conv_layers.append(layer_init(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)))
            conv_layers.append(nn.Tanh())
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros((1, self.board_channels, self.board_size, self.board_size), dtype=torch.float32)
            conv_out = self.conv(dummy)
            conv_flat_dim = int(conv_out.reshape(1, -1).shape[1])

        trunk_in = conv_flat_dim + self.global_dim
        if len(trunk_hidden_dims_t) > 0:
            self.trunk: nn.Module = _build_mlp(trunk_in, trunk_hidden_dims_t[:-1], trunk_hidden_dims_t[-1], out_std=1.0)
            trunk_out = int(trunk_hidden_dims_t[-1])
        else:
            self.trunk = nn.Identity()
            trunk_out = trunk_in
        self._encoded_dim = int(trunk_out)

        self.actor = _build_mlp(trunk_out, actor_head_hidden_dims_t, self.action_dim, out_std=0.01)
        self.critic = _build_mlp(trunk_out, critic_head_hidden_dims_t, 1, out_std=1.0)
        self.aux_opp_param_head: nn.Module | None = None

        if self.use_aux_opp_param_head:
            self.aux_opp_param_head = nn.Sequential(
                layer_init(nn.Linear(self._encoded_dim, self.aux_opp_param_hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(self.aux_opp_param_hidden_dim, AUX_OPP_PARAM_TOTAL), std=0.01),
            )

        self.model_config: dict[str, Any] = {
            "type": "DiscreteBoardCNNFCAgent",
            "kwargs": {
                "model_type": "cnn_fc",
                "board_channels": int(self.board_channels),
                "board_size": int(self.board_size),
                "global_dim": int(self.global_dim),
                "cnn_channels": list(cnn_channels_t),
                "cnn_kernels": list(kernels),
                "cnn_strides": list(strides),
                "cnn_paddings": list(paddings),
                "trunk_hidden_dims": list(trunk_hidden_dims_t),
                "actor_head_hidden_dims": list(actor_head_hidden_dims_t),
                "critic_head_hidden_dims": list(critic_head_hidden_dims_t),
                "aux_opp_param_head": bool(self.use_aux_opp_param_head),
                "aux_opp_param_hidden_dim": int(self.aux_opp_param_hidden_dim),
            },
        }

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._flatten_obs(obs)
        if x.shape[1] < self.board_dim:
            raise ValueError(f"obs width too small: {x.shape[1]} < board_dim={self.board_dim}")

        board = x[:, : self.board_dim].view(x.shape[0], self.board_channels, self.board_size, self.board_size)
        board_feat = self.conv(board).reshape(x.shape[0], -1)

        if self.global_dim > 0:
            global_feat = x[:, self.board_dim : self.board_dim + self.global_dim]
            feat = torch.cat([board_feat, global_feat], dim=1)
        else:
            feat = board_feat

        return self.trunk(feat)


class DiscreteBoardAgent(DiscreteBoardAgentBase):
    """Auto-selecting actor-critic for discrete board-game actions.

    Dispatches to MLP or CNN+FC based on `model_type`. Use
    `DiscreteBoardMLPAgent` or `DiscreteBoardCNNFCAgent` directly for new code.

    Supported model_type values:
    - "auto": selects "cnn_fc" when obs_dim > board_dim, else "mlp"
    - "mlp": flat MLP actor / critic
    - "cnn_fc": board CNN + global FC fusion, actor / critic heads
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        *,
        model_type: str = "auto",
        mlp_hidden_dims: tuple[int, ...] | list[int] | str | None = None,
        board_channels: int = 46,
        board_size: int = 10,
        global_dim: int = 0,
        cnn_channels: tuple[int, ...] | list[int] | str = (32, 64),
        cnn_kernels: tuple[int, ...] | list[int] | str = (3, 3),
        cnn_strides: tuple[int, ...] | list[int] | str = (1, 1),
        cnn_paddings: tuple[int, ...] | list[int] | str = (1, 1),
        trunk_hidden_dims: tuple[int, ...] | list[int] | str = (256, 128),
        actor_head_hidden_dims: tuple[int, ...] | list[int] | str = (),
        critic_head_hidden_dims: tuple[int, ...] | list[int] | str = (),
        aux_opp_param_head: bool = False,
        aux_opp_param_hidden_dim: int = 64,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        obs_dim = int(np.prod(obs_shape))
        self.obs_shape = obs_shape
        self.action_dim = int(action_dim)

        self.board_channels = int(board_channels)
        self.board_size = int(board_size)
        self.board_dim = int(self.board_channels * self.board_size * self.board_size)

        if mlp_hidden_dims is None:
            mlp_hidden_dims_t = tuple(int(hidden_dim) for _ in range(int(hidden_layers)))
        else:
            mlp_hidden_dims_t = to_int_tuple(mlp_hidden_dims)

        cnn_channels_t = to_int_tuple(cnn_channels)
        cnn_kernels_t = to_int_tuple(cnn_kernels)
        cnn_strides_t = to_int_tuple(cnn_strides)
        cnn_paddings_t = to_int_tuple(cnn_paddings)
        trunk_hidden_dims_t = to_int_tuple(trunk_hidden_dims)
        actor_head_hidden_dims_t = to_int_tuple(actor_head_hidden_dims)
        critic_head_hidden_dims_t = to_int_tuple(critic_head_hidden_dims)

        resolved_model_type = model_type
        if resolved_model_type == "auto":
            resolved_model_type = "cnn_fc" if obs_dim > self.board_dim else "mlp"
        if resolved_model_type not in ("mlp", "cnn_fc"):
            raise ValueError(f"unknown model_type: {resolved_model_type}")
        self.model_type = resolved_model_type

        self.use_aux_opp_param_head = bool(aux_opp_param_head)
        self.aux_opp_param_hidden_dim = int(max(1, aux_opp_param_hidden_dim))
        self.global_dim = 0
        self._encoded_dim = 0

        self.actor: nn.Module
        self.critic: nn.Module
        self.conv: nn.Module | None = None
        self.trunk: nn.Module | None = None
        self.aux_opp_param_head: nn.Module | None = None

        if self.model_type == "mlp":
            self.actor = _build_mlp(obs_dim, mlp_hidden_dims_t, self.action_dim, out_std=0.01)
            self.critic = _build_mlp(obs_dim, mlp_hidden_dims_t, 1, out_std=1.0)
            self._encoded_dim = int(obs_dim)
            self.model_config: dict[str, Any] = {
                "type": "DiscreteBoardAgent",
                "kwargs": {
                    "model_type": "mlp",
                    "mlp_hidden_dims": list(mlp_hidden_dims_t),
                    "aux_opp_param_head": bool(self.use_aux_opp_param_head),
                    "aux_opp_param_hidden_dim": int(self.aux_opp_param_hidden_dim),
                },
            }
        else:
            if obs_dim < self.board_dim:
                raise ValueError(
                    f"obs_dim too small for cnn_fc: obs_dim={obs_dim}, board_dim={self.board_dim}"
                )

            if int(global_dim) <= 0:
                self.global_dim = int(obs_dim - self.board_dim)
            else:
                self.global_dim = int(global_dim)
            if self.global_dim < 0 or self.board_dim + self.global_dim != obs_dim:
                raise ValueError(
                    f"invalid global_dim={self.global_dim} for obs_dim={obs_dim}, board_dim={self.board_dim}"
                )

            if len(cnn_channels_t) == 0:
                raise ValueError("cnn_channels must not be empty for model_type=cnn_fc")
            n_conv = len(cnn_channels_t)
            kernels = _broadcast_tuple(cnn_kernels_t, n_conv, "cnn_kernels")
            strides = _broadcast_tuple(cnn_strides_t, n_conv, "cnn_strides")
            paddings = _broadcast_tuple(cnn_paddings_t, n_conv, "cnn_paddings")

            conv_layers: list[nn.Module] = []
            in_ch = self.board_channels
            for out_ch, k, s, p in zip(cnn_channels_t, kernels, strides, paddings):
                conv_layers.append(layer_init(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)))
                conv_layers.append(nn.Tanh())
                in_ch = out_ch
            self.conv = nn.Sequential(*conv_layers)

            with torch.no_grad():
                dummy = torch.zeros((1, self.board_channels, self.board_size, self.board_size), dtype=torch.float32)
                conv_out = self.conv(dummy)
                conv_flat_dim = int(conv_out.reshape(1, -1).shape[1])

            trunk_in = conv_flat_dim + self.global_dim
            if len(trunk_hidden_dims_t) > 0:
                self.trunk = _build_mlp(trunk_in, trunk_hidden_dims_t[:-1], trunk_hidden_dims_t[-1], out_std=1.0)
                trunk_out = int(trunk_hidden_dims_t[-1])
            else:
                self.trunk = nn.Identity()
                trunk_out = trunk_in
            self._encoded_dim = int(trunk_out)

            self.actor = _build_mlp(trunk_out, actor_head_hidden_dims_t, self.action_dim, out_std=0.01)
            self.critic = _build_mlp(trunk_out, critic_head_hidden_dims_t, 1, out_std=1.0)

            self.model_config = {
                "type": "DiscreteBoardAgent",
                "kwargs": {
                    "model_type": "cnn_fc",
                    "board_channels": int(self.board_channels),
                    "board_size": int(self.board_size),
                    "global_dim": int(self.global_dim),
                    "cnn_channels": list(cnn_channels_t),
                    "cnn_kernels": list(kernels),
                    "cnn_strides": list(strides),
                    "cnn_paddings": list(paddings),
                    "trunk_hidden_dims": list(trunk_hidden_dims_t),
                    "actor_head_hidden_dims": list(actor_head_hidden_dims_t),
                    "critic_head_hidden_dims": list(critic_head_hidden_dims_t),
                    "aux_opp_param_head": bool(self.use_aux_opp_param_head),
                    "aux_opp_param_hidden_dim": int(self.aux_opp_param_hidden_dim),
                },
            }

        if self.use_aux_opp_param_head:
            self.aux_opp_param_head = nn.Sequential(
                layer_init(nn.Linear(self._encoded_dim, self.aux_opp_param_hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(self.aux_opp_param_hidden_dim, AUX_OPP_PARAM_TOTAL), std=0.01),
            )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._flatten_obs(obs)
        if self.model_type == "mlp":
            return x

        assert self.conv is not None
        if x.shape[1] < self.board_dim:
            raise ValueError(f"obs width too small: {x.shape[1]} < board_dim={self.board_dim}")

        board = x[:, : self.board_dim].view(x.shape[0], self.board_channels, self.board_size, self.board_size)
        board_feat = self.conv(board).reshape(x.shape[0], -1)

        if self.global_dim > 0:
            global_feat = x[:, self.board_dim : self.board_dim + self.global_dim]
            feat = torch.cat([board_feat, global_feat], dim=1)
        else:
            feat = board_feat

        if self.trunk is not None:
            feat = self.trunk(feat)
        return feat
