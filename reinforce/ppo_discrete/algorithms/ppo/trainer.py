from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import PPOConfig
from .rollout_buffer import RolloutBuffer
from ...models.discrete_board import DiscreteBoardAgent


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    value_clipfrac: float
    update_epochs_used: int
    early_stop_by_kl: bool
    target_kl_threshold: float | None


class PPOTrainer:
    """PPO optimizer step implementation for discrete board agent.

    Rollout collection is intentionally left environment-specific; this class
    focuses on update logic and reuses CleanRL-compatible equations.
    """

    def __init__(self, cfg: PPOConfig, agent: DiscreteBoardAgent, optimizer: torch.optim.Optimizer):
        self.cfg = cfg
        self.agent = agent
        self.optimizer = optimizer
        self.runtime_ent_coef: float | None = None
        self.runtime_clip_coef: float | None = None
        self.runtime_clip_range_vf: float | None = None

    def _target_kl_threshold(self) -> float | None:
        if self.cfg.target_kl is None:
            return None
        # SB3-compatible semantics: early-stop when approx_kl > 1.5 * target_kl.
        return float(self.cfg.target_kl) * 1.5

    def set_runtime_coefficients(
        self,
        *,
        ent_coef: float | None = None,
        clip_coef: float | None = None,
        clip_range_vf: float | None = None,
    ) -> None:
        self.runtime_ent_coef = None if ent_coef is None else float(ent_coef)
        self.runtime_clip_coef = None if clip_coef is None else float(clip_coef)
        self.runtime_clip_range_vf = None if clip_range_vf is None else float(clip_range_vf)

    def update(self, buffer: RolloutBuffer) -> UpdateStats:
        batch = buffer.flatten()
        b_inds = buffer.shuffled_indices()
        clipfracs: list[float] = []
        value_clipfracs: list[float] = []
        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []
        clip_coef = float(self.runtime_clip_coef if self.runtime_clip_coef is not None else self.cfg.clip_coef)
        ent_coef = float(self.runtime_ent_coef if self.runtime_ent_coef is not None else self.cfg.ent_coef)
        target_kl_threshold = self._target_kl_threshold()
        update_epochs_used = 0
        early_stop_by_kl = False

        last_pg_loss = torch.tensor(0.0, device=batch.obs.device)
        last_v_loss = torch.tensor(0.0, device=batch.obs.device)
        last_entropy = torch.tensor(0.0, device=batch.obs.device)
        last_approx_kl = torch.tensor(0.0, device=batch.obs.device)

        for _ in range(self.cfg.update_epochs):
            update_epochs_used += 1
            epoch_kl_values: list[float] = []
            np.random.shuffle(b_inds)
            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = b_inds[start:end]

                action_mask = None
                if batch.action_masks is not None:
                    action_mask = batch.action_masks[mb_inds]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    batch.obs[mb_inds],
                    batch.actions.long()[mb_inds].view(-1),
                    action_mask=action_mask,
                )
                logratio = newlogprob - batch.logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    last_approx_kl = ((ratio - 1.0) - logratio).mean()
                    epoch_kl_values.append(float(last_approx_kl.item()))
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_advantages = batch.advantages[mb_inds]
                if self.cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                last_pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                pg_losses.append(float(last_pg_loss.item()))

                newvalue = newvalue.view(-1)
                vf_clip_cfg = self.runtime_clip_range_vf
                if vf_clip_cfg is None:
                    vf_clip_cfg = self.cfg.clip_range_vf
                if self.cfg.clip_vloss and vf_clip_cfg is not None:
                    vf_clip = float(vf_clip_cfg)
                    value_clipfracs.append(((newvalue - batch.values[mb_inds]).abs() > vf_clip).float().mean().item())
                    v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
                    v_clipped = batch.values[mb_inds] + torch.clamp(
                        newvalue - batch.values[mb_inds],
                        -vf_clip,
                        vf_clip,
                    )
                    v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
                    last_v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    last_v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()
                v_losses.append(float(last_v_loss.item()))

                last_entropy = entropy.mean()
                entropies.append(float(last_entropy.item()))
                loss = last_pg_loss - ent_coef * last_entropy + self.cfg.vf_coef * last_v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

            epoch_approx_kl = float(np.mean(epoch_kl_values)) if epoch_kl_values else float(last_approx_kl.item())
            last_approx_kl = torch.tensor(epoch_approx_kl, device=batch.obs.device)
            if target_kl_threshold is not None and epoch_approx_kl > target_kl_threshold:
                early_stop_by_kl = True
                break

        return UpdateStats(
            policy_loss=float(np.mean(pg_losses) if pg_losses else last_pg_loss.item()),
            value_loss=float(np.mean(v_losses) if v_losses else last_v_loss.item()),
            entropy=float(np.mean(entropies) if entropies else last_entropy.item()),
            approx_kl=float(last_approx_kl.item()),
            clipfrac=float(np.mean(clipfracs) if clipfracs else 0.0),
            value_clipfrac=float(np.mean(value_clipfracs) if value_clipfracs else float("nan")),
            update_epochs_used=int(update_epochs_used),
            early_stop_by_kl=bool(early_stop_by_kl),
            target_kl_threshold=(float(target_kl_threshold) if target_kl_threshold is not None else None),
        )
