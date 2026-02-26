from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from reinforce.ppo_discrete.cli.train_ppo import _apply_time_limit_bootstrap, _extract_terminal_obs_from_infos


class _ConstValueAgent(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.v = nn.Parameter(torch.tensor(float(value), dtype=torch.float32))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 1), float(self.v.item()), dtype=torch.float32, device=obs.device)


class TestTrainPPOTimeLimit:
    def test_extract_terminal_obs_from_infos(self) -> None:
        infos = {
            "final_observation": np.asarray([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32),
            "_final_observation": np.asarray([True, False], dtype=np.bool_),
        }
        got0 = _extract_terminal_obs_from_infos(infos, 0)
        got1 = _extract_terminal_obs_from_infos(infos, 1)
        np.testing.assert_allclose(got0, np.asarray([9.0, 8.0], dtype=np.float32))
        assert got1 is None

    def test_extract_terminal_obs_from_final_info_fallback(self) -> None:
        infos = {
            "final_info": [
                {"terminal_observation": np.asarray([1.0, 2.0], dtype=np.float32)},
                None,
            ]
        }
        got = _extract_terminal_obs_from_infos(infos, 0)
        np.testing.assert_allclose(got, np.asarray([1.0, 2.0], dtype=np.float32))

    def test_apply_time_limit_bootstrap(self) -> None:
        rewards = np.asarray([1.0, 2.0], dtype=np.float32)
        next_obs = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        terminations = np.asarray([False, False], dtype=np.bool_)
        truncations = np.asarray([True, False], dtype=np.bool_)
        infos = {
            "final_observation": np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            "_final_observation": np.asarray([True, False], dtype=np.bool_),
        }
        agent = _ConstValueAgent(2.0)
        out, count = _apply_time_limit_bootstrap(
            rewards=rewards,
            next_obs=next_obs,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            agent=agent,
            gamma=0.9,
            device=torch.device("cpu"),
        )

        assert count == 1
        assert float(out[0]) == pytest.approx(1.0 + 0.9 * 2.0, abs=1e-6)
        assert float(out[1]) == pytest.approx(2.0, abs=1e-6)

    def test_truncation_and_termination_does_not_bootstrap(self) -> None:
        rewards = np.asarray([1.0], dtype=np.float32)
        next_obs = np.asarray([[0.0, 0.0]], dtype=np.float32)
        terminations = np.asarray([True], dtype=np.bool_)
        truncations = np.asarray([True], dtype=np.bool_)
        infos = {}
        agent = _ConstValueAgent(3.0)
        out, count = _apply_time_limit_bootstrap(
            rewards=rewards,
            next_obs=next_obs,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            agent=agent,
            gamma=0.99,
            device=torch.device("cpu"),
        )
        assert count == 0
        assert float(out[0]) == pytest.approx(1.0, abs=1e-6)
