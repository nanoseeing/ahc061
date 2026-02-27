from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Default PPO hyperparameters for discrete board games.

    The defaults are intentionally close to CleanRL's ppo.py and can be tuned
    per contest environment later.
    """

    exp_name: str = "ppo_discrete_board"
    seed: int = 1
    cuda: bool = True

    total_timesteps: int = 500_000
    num_envs: int = 8
    num_steps: int = 128

    learning_rate: float = 2.5e-4
    learning_rate_schedule: str = ""
    gamma: float = 0.99
    gae_lambda: float = 0.95

    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_range_vf: float | None = None
    clip_range_vf_schedule: str = "constant"
    clip_range_vf_final: float | None = None
    clip_range_vf_schedule_expr: str = ""
    clip_vloss: bool = True
    ent_coef: float = 0.01
    ent_coef_schedule: str = "constant"
    ent_coef_final: float | None = None
    ent_coef_schedule_expr: str = ""
    vf_coef: float = 0.5
    aux_opp_param_loss_coef: float = 0.0
    aux_opp_param_use_valid_mask: bool = True
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    clip_coef_schedule: str = "constant"
    clip_coef_final: float | None = None
    clip_coef_schedule_expr: str = ""

    save_interval: int = 10

    @property
    def batch_size(self) -> int:
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self) -> int:
        return int(self.batch_size // self.num_minibatches)

    @property
    def num_iterations(self) -> int:
        if self.batch_size <= 0:
            return 0
        return int(math.ceil(float(self.total_timesteps) / float(self.batch_size)))
