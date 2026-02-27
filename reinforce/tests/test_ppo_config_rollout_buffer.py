from __future__ import annotations

import numpy as np
import torch

from reinforce.ppo_discrete.ppo.config import PPOConfig
from reinforce.ppo_discrete.ppo.rollout_buffer import RolloutBuffer


class TestPPOConfigAndRolloutBuffer:
    def test_num_iterations_uses_ceil(self) -> None:
        cfg = PPOConfig(total_timesteps=250, num_envs=2, num_steps=64)
        assert cfg.batch_size == 128
        assert cfg.num_iterations == 2

    def test_rollout_buffer_compute_gae_matches_manual(self) -> None:
        device = torch.device("cpu")
        buf = RolloutBuffer(
            num_steps=3,
            num_envs=1,
            obs_shape=(2,),
            action_shape=tuple(),
            device=device,
        )
        buf.rewards[:, 0] = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float32)
        buf.values[:, 0] = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32)
        buf.dones[:, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        next_value = torch.tensor([0.3], dtype=torch.float32)
        next_done = torch.tensor([1.0], dtype=torch.float32)
        gamma = 0.99
        gae_lambda = 0.95
        buf.compute_gae(next_value, next_done, gamma, gae_lambda)

        rewards = [1.0, 0.5, -0.25]
        values = [0.2, -0.1, 0.4]
        manual_adv = [0.0, 0.0, 0.0]
        last = 0.0
        for t in reversed(range(3)):
            if t == 2:
                next_non_terminal = 1.0 - float(next_done[0].item())
                next_v = float(next_value[0].item())
            else:
                next_non_terminal = 1.0 - 0.0
                next_v = values[t + 1]
            delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
            last = delta + gamma * gae_lambda * next_non_terminal * last
            manual_adv[t] = last
        manual_ret = [manual_adv[i] + values[i] for i in range(3)]

        np.testing.assert_allclose(
            buf.advantages[:, 0].detach().cpu().numpy(),
            np.asarray(manual_adv, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            buf.returns[:, 0].detach().cpu().numpy(),
            np.asarray(manual_ret, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rollout_buffer_action_mask_and_flatten(self) -> None:
        device = torch.device("cpu")
        buf = RolloutBuffer(
            num_steps=2,
            num_envs=3,
            obs_shape=(4,),
            action_shape=tuple(),
            device=device,
            use_action_mask=True,
            action_dim=5,
        )

        for step in range(2):
            obs = torch.full((3, 4), float(step), dtype=torch.float32)
            action = torch.tensor([0, 1, 2], dtype=torch.long)
            logprob = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
            reward = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            done = torch.zeros(3, dtype=torch.float32)
            value = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            mask = torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, False, True, False, False],
                    [False, True, True, False, True],
                ],
                dtype=torch.bool,
            )
            buf.add(
                step,
                obs=obs,
                action=action,
                logprob=logprob,
                reward=reward,
                done=done,
                value=value,
                action_mask=mask,
            )

        flat = buf.flatten()
        assert tuple(flat.obs.shape) == (6, 4)
        assert tuple(flat.actions.shape) == (6,)
        assert flat.action_masks is not None
        assert flat.action_masks is not None
        assert tuple(flat.action_masks.shape) == (6, 5)

        inds = buf.shuffled_indices()
        assert inds.shape[0] == 6
        assert len(np.unique(inds)) == 6
