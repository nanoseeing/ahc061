"""`test_export_submission_main_cpp` のテストモジュール。"""
from __future__ import annotations

from reinforce.ppo.entrypoints.export_submission_main_cpp import build_submission_source


class TestExportSubmissionMainCpp:
    """`TestExportSubmissionMainCpp` のテストケース。"""
    def test_build_submission_source_replaces_placeholders(self) -> None:
        """`build_submission_source_replaces_placeholders` の振る舞いを検証する。"""
        src = build_submission_source(
            board_channels=7,
            board_size=10,
            global_dim=49,
            width=48,
            num_blocks=2,
            global_hidden_dim=64,
            action_dim=100,
            activation_id=0,
            use_global_film=True,
            use_global_policy_bias=True,
            deterministic=True,
            use_action_mask=True,
            bayes_num_particles=128,
            bayes_seed=1,
            bayes_resample_ess_frac=0.55,
            tensor_half_counts=[2, 3, 4],
            model_blob_b64="AQIDBA==",
        )
        assert "constexpr int kBoardChannels = 7;" in src
        assert "constexpr int kTensorCount = 3;" in src
        assert "static const int kTensorHalfCounts[kTensorCount] = {2, 3, 4};" in src
        assert '"AQIDBA=="' in src
        assert "__BOARD_CHANNELS__" not in src
        assert "__MODEL_BLOB_B64_LINES__" not in src
