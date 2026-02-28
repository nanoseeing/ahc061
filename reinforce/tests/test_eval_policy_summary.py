from __future__ import annotations

from types import SimpleNamespace

from reinforce.ppo_discrete.entrypoints.eval_policy import _build_summary


def test_build_summary_uses_game_score_ratio_for_terminal_score_ratio() -> None:
    args = SimpleNamespace(
        env_id="AHC061Local-v0",
        episodes=2,
        deterministic=True,
        model_path="m.pt",
    )
    summary = _build_summary(
        args=args,
        env_kwargs={},
        model_meta={},
        episode_returns=[1.0, 2.0],
        episode_illegal_penalties=[0.0, 0.0],
        episode_terminal_scores=[10.0, 20.0],
        episode_terminal_game_scores=[100.0, 200.0],
        episode_game_score_ratio=[0.25, 0.75],
        episode_game_score_self=[100.0, 200.0],
        episode_game_score_enemy_max=[400.0, 266.0],
    )
    assert summary["reward_components"]["terminal_score"]["mean"] == 15.0
    assert summary["reward_components"]["terminal_score_ratio"]["mean"] == 0.5
