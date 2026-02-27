from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FlattenedBatch:
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
    """On-policy rollout storage compatible with PPO updates."""

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
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value.view(-1)
        if self.action_masks is not None and action_mask is not None:
            self.action_masks[step] = action_mask

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor, gamma: float, gae_lambda: float) -> None:
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
        return np.random.permutation(self.num_steps * self.num_envs)
