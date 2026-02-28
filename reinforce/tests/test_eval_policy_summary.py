from __future__ import annotations

from types import SimpleNamespace

from reinforce.ppo.entrypoints.eval_policy import _build_summary


def test_build_summary_emits_terminal_game_score_summary() -> None:
    args = SimpleNamespace(
        env_id="AHC061Local-v0",
        episodes=2,
        model_path="m.pt",
    )
    summary = _build_summary(
        args=args,
        env_kwargs={},
        model_meta={},
        episode_returns=[1.0, 2.0],
        episode_scores=[100.0, 200.0],
        episode_m=[4, 8],
        episode_u=[4, 9],
        episode_self_scores=[100.0, 200.0],
        episode_enemy_max_scores=[400.0, 266.0],
    )
    assert summary["return"]["mean"] == 1.5
    assert summary["terminal_game_score"]["mean"] == 150.0
