from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from reinforce.ppo_discrete.core.ppo.config import PPOConfig
from reinforce.ppo_discrete.core.ppo.rollout_buffer import RolloutBuffer
from reinforce.ppo_discrete.core.ppo.trainer import PPOTrainer


class _DummyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        return_aux_opp_param: bool = False,
    ):
        base = obs[:, 0] * self.theta
        logits = torch.stack([base, torch.zeros_like(base)], dim=1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = torch.zeros((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)
        return action, logprob, entropy, value


def _build_buffer(
    *,
    cfg: PPOConfig,
    obs_rows: list[list[float]],
    actions: list[int],
    logprobs: list[float],
    advantages: list[float],
    returns: list[float],
    values: list[float],
) -> RolloutBuffer:
    device = torch.device("cpu")
    buf = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        obs_shape=(2,),
        action_shape=tuple(),
        device=device,
    )
    obs = torch.tensor(obs_rows, dtype=torch.float32).view(cfg.num_steps, cfg.num_envs, 2)
    buf.obs.copy_(obs)
    buf.actions.copy_(torch.tensor(actions, dtype=torch.long).view(cfg.num_steps, cfg.num_envs))
    buf.logprobs.copy_(torch.tensor(logprobs, dtype=torch.float32).view(cfg.num_steps, cfg.num_envs))
    buf.advantages.copy_(torch.tensor(advantages, dtype=torch.float32).view(cfg.num_steps, cfg.num_envs))
    buf.returns.copy_(torch.tensor(returns, dtype=torch.float32).view(cfg.num_steps, cfg.num_envs))
    buf.values.copy_(torch.tensor(values, dtype=torch.float32).view(cfg.num_steps, cfg.num_envs))
    buf.shuffled_indices = lambda: np.arange(cfg.batch_size, dtype=np.int64)  # type: ignore[method-assign]
    return buf


class TestPPOTrainer:
    def test_target_kl_uses_epoch_mean(self) -> None:
        cfg = PPOConfig(
            num_envs=2,
            num_steps=2,
            num_minibatches=2,
            update_epochs=3,
            target_kl=0.05,
            clip_coef=0.2,
        )
        agent = _DummyAgent()
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.0)
        trainer = PPOTrainer(cfg, agent, optimizer)
        buffer = _build_buffer(
            cfg=cfg,
            obs_rows=[
                [2.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            actions=[0, 0, 0, 0],
            logprobs=[-4.0, -4.0, -0.69314718, -0.69314718],
            advantages=[1.0, 1.0, 1.0, 1.0],
            returns=[0.0, 0.0, 0.0, 0.0],
            values=[0.0, 0.0, 0.0, 0.0],
        )

        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step_mock:
            with mock.patch("numpy.random.shuffle", lambda x: None):
                stats = trainer.update(buffer)

        # 2 minibatches processed in the first epoch, then early-stop by epoch-mean KL.
        assert step_mock.call_count == 2
        assert cfg.target_kl is not None
        assert stats.approx_kl > 1.5 * cfg.target_kl
        assert stats.early_stop_by_kl
        assert stats.update_epochs_used == 1
        assert (stats.target_kl_threshold or 0.0) == pytest.approx(1.5 * cfg.target_kl, abs=1e-12)

    def test_runtime_clip_coef_overrides_cfg(self) -> None:
        cfg = PPOConfig(
            num_envs=2,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
            clip_coef=0.5,
            target_kl=None,
        )
        agent = _DummyAgent()
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.0)
        trainer = PPOTrainer(cfg, agent, optimizer)
        buffer = _build_buffer(
            cfg=cfg,
            obs_rows=[
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            actions=[0, 0, 0, 0],
            logprobs=[-0.79314718, -0.79314718, -0.79314718, -0.79314718],
            advantages=[1.0, 1.0, 1.0, 1.0],
            returns=[0.0, 0.0, 0.0, 0.0],
            values=[0.0, 0.0, 0.0, 0.0],
        )

        with mock.patch("numpy.random.shuffle", lambda x: None):
            stats_default = trainer.update(buffer)

        trainer.set_runtime_coefficients(clip_coef=0.01)
        with mock.patch("numpy.random.shuffle", lambda x: None):
            stats_runtime = trainer.update(buffer)

        assert stats_default.clipfrac < 0.01
        assert stats_runtime.clipfrac > 0.99
        assert not stats_default.early_stop_by_kl
        assert not stats_runtime.early_stop_by_kl
        assert stats_default.update_epochs_used == cfg.update_epochs
        assert stats_runtime.update_epochs_used == cfg.update_epochs
        assert np.isnan(stats_default.value_clipfrac)
        assert np.isnan(stats_runtime.value_clipfrac)

    def test_runtime_clip_range_vf_overrides_cfg(self) -> None:
        cfg = PPOConfig(
            num_envs=2,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
            clip_vloss=True,
            clip_range_vf=1.0,
            target_kl=None,
        )
        agent = _DummyAgent()
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.0)
        trainer = PPOTrainer(cfg, agent, optimizer)
        buffer = _build_buffer(
            cfg=cfg,
            obs_rows=[
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            actions=[0, 0, 0, 0],
            logprobs=[-0.69314718, -0.69314718, -0.69314718, -0.69314718],
            advantages=[1.0, 1.0, 1.0, 1.0],
            returns=[2.0, 2.0, 2.0, 2.0],
            values=[-1.0, -1.0, -1.0, -1.0],
        )

        with mock.patch("numpy.random.shuffle", lambda x: None):
            stats_default = trainer.update(buffer)

        trainer.set_runtime_coefficients(clip_range_vf=0.1)
        with mock.patch("numpy.random.shuffle", lambda x: None):
            stats_runtime = trainer.update(buffer)

        assert stats_runtime.value_loss > stats_default.value_loss
        assert stats_default.value_clipfrac < 0.01
        assert stats_runtime.value_clipfrac > 0.99
