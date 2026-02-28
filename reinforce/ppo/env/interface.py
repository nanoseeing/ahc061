from __future__ import annotations

from typing import Protocol, Sequence, cast, runtime_checkable

import torch

from .env import EnvSpec


@runtime_checkable
class BatchEnvProtocol(Protocol):
    @property
    def spec(self) -> EnvSpec: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def pf_enabled(self) -> bool: ...

    @property
    def feature_id(self) -> str: ...

    @property
    def board_size(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    @property
    def feature_channels(self) -> int: ...

    def feature_channels_of(self, feature_id: str) -> int: ...

    def set_pf_enabled(self, v: bool) -> None: ...

    def set_feature_id(self, feature_id: str) -> None: ...

    def reset_random(self, seeds: torch.Tensor) -> None: ...

    def reset_from_tools(self, paths: Sequence[str], pf_seeds_extra: torch.Tensor | None = None) -> None: ...

    def observe(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    def observe_into(self, board: torch.Tensor, mask: torch.Tensor) -> None: ...

    def observe_into_feature(self, board: torch.Tensor, mask: torch.Tensor, feature_id: str) -> None: ...

    def observe_pair_into(
        self,
        board_a: torch.Tensor,
        board_b: torch.Tensor,
        mask: torch.Tensor,
        *,
        feature_id_a: str,
        feature_id_b: str,
    ) -> None: ...

    def aux_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def aux_targets_into(self, move_dist: torch.Tensor, opp_param: torch.Tensor, opp_valid: torch.Tensor) -> None: ...

    def bayes_params(self) -> torch.Tensor: ...

    def bayes_params_into(self, bayes_params: torch.Tensor) -> None: ...

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def step_into(self, actions: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None: ...

    def step_observe_into(
        self,
        actions: torch.Tensor,
        board: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None: ...

    def step_observe_aux_into(
        self,
        actions: torch.Tensor,
        board: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        move_dist: torch.Tensor,
        opp_param: torch.Tensor,
        opp_valid: torch.Tensor,
    ) -> None: ...

    def step_observe_pair_into(
        self,
        actions: torch.Tensor,
        board_a: torch.Tensor,
        board_b: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        *,
        feature_id_a: str,
        feature_id_b: str,
    ) -> None: ...

    def pos0(self) -> torch.Tensor: ...

    def official_score(self) -> torch.Tensor: ...

    def score_s0_sa(self) -> torch.Tensor: ...

    def m_u(self) -> torch.Tensor: ...


def ensure_batch_env(env: object) -> BatchEnvProtocol:
    if isinstance(env, BatchEnvProtocol):
        return cast(BatchEnvProtocol, env)
    # Better error than AttributeError deeper in rollout loop.
    raise TypeError(f"env does not satisfy BatchEnvProtocol: {type(env)!r}")
