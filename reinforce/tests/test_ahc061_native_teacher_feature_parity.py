from __future__ import annotations

from pathlib import Path

import numpy as np

from reinforce.ppo_discrete.domains.ahc061.env import AHC061LocalEnv
from reinforce.ppo_discrete.domains.ahc061.native_batch import BatchEnv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_python_env_from_tools_case(case_path: Path) -> AHC061LocalEnv:
    env = AHC061LocalEnv(
        case_num_cases=1,
        case_seed_mode="sequential",
        case_seed_start=0,
        case_gen_cmd="true",
        case_select_mode="sequential",
        reward_mode="delta_log_ratio",
        illegal_action_penalty=-0.05,
        bayes_num_particles=128,
        bayes_backend="cpp",
        obs_mode="teacher_p0_v1_88ch",
    )

    case = AHC061LocalEnv._parse_case_text(case_path.read_text(encoding="utf-8"))
    st = AHC061LocalEnv._init_state(case)
    env.case = case
    env.state = st
    env.turn = 0
    env.done = False
    env._last_scores = env._score_all(st, case)
    env._episode_return = 0.0
    env._episode_length = 0
    env._episode_illegal_penalty = 0.0
    env._episode_terminal_score = 0.0
    env._episode_terminal_game_score = 0.0
    env._episode_objective_delta = 0.0
    # Parity check target: feature extraction/layout itself.
    # Keep bayes vector fixed-zero to match native pf_enabled=False path.
    env._bayes = None
    return env


def test_teacher_feature_parity_vs_python_env_tools_case_no_bayes() -> None:
    case_ids = [0, 1, 2, 3, 7, 13]
    native_env = BatchEnv(batch_size=1, feature_id="teacher_p0_v1_88ch", pf_enabled=False)

    for cid in case_ids:
        case_path = _repo_root() / "tools" / "in" / f"{cid:04d}.txt"
        assert case_path.exists(), f"missing tools case file: {case_path}"

        py_env = _build_python_env_from_tools_case(case_path)
        native_env.reset_from_tools([str(case_path)])

        board_t, mask_t = native_env.observe()
        native_board = board_t[0].detach().cpu().numpy()
        native_mask = mask_t[0].detach().cpu().numpy().astype(np.bool_)

        py_board = py_env._encode_obs_teacher_p0_v1_88ch().reshape(88, 10, 10)
        py_mask = py_env._build_action_mask()

        np.testing.assert_allclose(native_board, py_board, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(native_mask, py_mask)
        py_env.close()
