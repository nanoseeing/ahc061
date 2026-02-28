"""`test_payload_codec` のテストモジュール。"""
from __future__ import annotations

import os

from reinforce.ppo.submission.payload_codec import (
    PAYLOAD_CODEC_BASE91,
    PAYLOAD_CODEC_BASE122,
    PAYLOAD_CODEC_HUFF91,
    PAYLOAD_CODEC_HUFF122,
    _decode_base91,
    _decode_base122,
    _huff122_decompress,
    encode_model_payload,
)


def test_payload_codec_roundtrip() -> None:
    """`payload_codec_roundtrip` の振る舞いを検証する。"""
    blob = os.urandom(2048)

    text, codec = encode_model_payload(blob, encoding="base91")
    assert codec == PAYLOAD_CODEC_BASE91
    assert _decode_base91(text) == blob

    text, codec = encode_model_payload(blob, encoding="base122")
    assert codec == PAYLOAD_CODEC_BASE122
    assert _decode_base122(text) == blob

    text, codec = encode_model_payload(blob, encoding="huff122")
    assert codec == PAYLOAD_CODEC_HUFF122
    assert _huff122_decompress(_decode_base122(text), expected_size=len(blob)) == blob

    text, codec = encode_model_payload(blob, encoding="huff91")
    assert codec == PAYLOAD_CODEC_HUFF91
    assert _huff122_decompress(_decode_base91(text), expected_size=len(blob)) == blob
