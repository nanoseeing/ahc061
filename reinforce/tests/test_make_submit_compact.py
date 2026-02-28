"""`test_make_submit_compact` のテストモジュール。"""
from __future__ import annotations

import argparse

import torch

from reinforce.ppo.entrypoints import export_submission_main_cpp as student_export
from reinforce.ppo.entrypoints.make_submit_compact import (
    _build_student_source,
    _collect_teacher_p0_policy_tensors,
    _compact_layout_safe,
    _rewrite_student_source_with_payload,
    build_parser,
)
from reinforce.ppo.submission.payload_codec import PAYLOAD_CODEC_BASE91


def test_rewrite_student_source_with_payload_replaces_base64_path() -> None:
    """`rewrite_student_source_with_payload_replaces_base64_path` の振る舞いを検証する。"""
    src = student_export.build_submission_source(
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
    out = _rewrite_student_source_with_payload(
        src,
        payload_text="abcDEF",
        codec_id=PAYLOAD_CODEC_BASE91,
        model_blob_bytes=6,
    )
    assert "kModelBlobBase64" not in out
    assert "base64_decode" not in out
    assert "kModelBlobEncoded" in out
    assert "decode_payload(std::string_view encoded)" in out
    assert "constexpr int kPayloadCodec = 91;" in out
    assert 'throw std::runtime_error("invalid decoded payload size");' in out


def test_compact_layout_safe_applies_alias_macros() -> None:
    """`compact_layout_safe_applies_alias_macros` の振る舞いを検証する。"""
    src = "#include <array>\n#include <vector>\nstd::array<int, 2> a; std::vector<int> b;\n"
    out = _compact_layout_safe(src)
    assert "#define A_ std::array" in out
    assert "#define V_ std::vector" in out
    assert "A_<int, 2> a; V_<int> b;" in out


def test_student_tta_mode_nonzero_raises() -> None:
    """`student_tta_mode_nonzero_raises` の振る舞いを検証する。"""
    args = argparse.Namespace(payload_encoding="huff91", tta_mode=1)
    payload: dict[str, object] = {}
    try:
        _build_student_source(payload, args)
        raised = False
    except ValueError as exc:
        raised = True
        assert "TTA is currently supported only for Exp002ResNetBoardAgent export" in str(exc)
    assert raised


def test_collect_teacher_p0_policy_tensors_order_and_count() -> None:
    """`collect_teacher_p0_policy_tensors_order_and_count` の振る舞いを検証する。"""
    state_dict = {
        "stem.0.weight": torch.zeros((64, 88, 3, 3), dtype=torch.float32),
        "stem.0.bias": torch.zeros((64,), dtype=torch.float32),
        "blocks.0.conv1.weight": torch.zeros((64, 64, 3, 3), dtype=torch.float32),
        "blocks.0.conv1.bias": torch.zeros((64,), dtype=torch.float32),
        "blocks.0.conv2.weight": torch.zeros((64, 64, 3, 3), dtype=torch.float32),
        "blocks.0.conv2.bias": torch.zeros((64,), dtype=torch.float32),
        "policy_conv.weight": torch.zeros((1, 64, 1, 1), dtype=torch.float32),
        "policy_conv.bias": torch.zeros((1,), dtype=torch.float32),
    }
    tensors = _collect_teacher_p0_policy_tensors(state_dict, num_blocks=1)
    assert len(tensors) == 8
    assert tensors[0].shape == (64, 88, 3, 3)
    assert tensors[1].shape == (64,)
    assert tensors[-2].shape == (1, 64, 1, 1)
    assert tensors[-1].shape == (1,)


def test_build_parser_has_no_legacy_exp002_flags() -> None:
    """`build_parser_has_no_legacy_exp002_flags` の振る舞いを検証する。"""
    parser = build_parser()
    dests = {a.dest for a in parser._actions}  # noqa: SLF001
    assert "legacy_exp002" not in dests
    assert "legacy_exp002_script" not in dests
