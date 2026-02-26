from __future__ import annotations

import math

import numpy as np
import pytest

from reinforce.ppo_discrete.runtime.episode_stats import extract_completed_episode_stats


class TestEpisodeStats:
    def test_extract_from_episode_arrays(self) -> None:
        infos = {
            "episode": {
                "r": np.asarray([1.5, 2.5], dtype=np.float64),
                "l": np.asarray([100, 100], dtype=np.float64),
            },
            "_episode": np.asarray([True, False], dtype=np.bool_),
            "episode_illegal_penalty": np.asarray([-0.2, -0.1], dtype=np.float64),
            "episode_terminal_score": np.asarray([0.3, 0.4], dtype=np.float64),
            "episode_terminal_game_score": np.asarray([123456, 654321], dtype=np.float64),
            "final_scores": np.asarray([[1500.0, 1000.0, 1200.0], [500.0, 700.0, 600.0]], dtype=np.float64),
            "_final_scores": np.asarray([True, False], dtype=np.bool_),
        }
        out = extract_completed_episode_stats(infos)
        assert len(out) == 1
        s = out[0]
        assert s["total_return"] == pytest.approx(1.5, abs=1e-6)
        assert s["illegal_penalty_return"] == pytest.approx(-0.2, abs=1e-6)
        assert s["terminal_score_return"] == pytest.approx(0.3, abs=1e-6)
        assert s["terminal_game_score_return"] == pytest.approx(123456.0, abs=1e-6)
        assert s["final_self_score"] == pytest.approx(1500.0, abs=1e-6)
        assert s["final_enemy_max_score"] == pytest.approx(1200.0, abs=1e-6)
        ratio = 1500.0 / 1200.0
        assert s["final_score_ratio"] == pytest.approx(ratio, abs=1e-6)
        assert s["final_score_log2"] == pytest.approx(math.log2(ratio), abs=1e-6)
        assert s["final_game_score"] == pytest.approx(float(np.round(1e5 * np.log2(1.0 + ratio))), abs=1e-6)

    def test_extract_from_final_info_fallback(self) -> None:
        infos = {
            "final_info": [
                {
                    "episode": {"r": 2.0, "l": 100},
                    "episode_illegal_penalty": -0.5,
                    "episode_terminal_score": 0.2,
                    "final_scores": [900.0, 1000.0],
                }
            ]
        }
        out = extract_completed_episode_stats(infos)
        assert len(out) == 1
        s = out[0]
        assert s["total_return"] == pytest.approx(2.0, abs=1e-6)
        assert s["illegal_penalty_return"] == pytest.approx(-0.5, abs=1e-6)
        assert s["terminal_score_return"] == pytest.approx(0.2, abs=1e-6)
        assert s["final_self_score"] == pytest.approx(900.0, abs=1e-6)
        assert s["final_enemy_max_score"] == pytest.approx(1000.0, abs=1e-6)
        assert s["final_score_ratio"] == pytest.approx(0.9, abs=1e-6)
