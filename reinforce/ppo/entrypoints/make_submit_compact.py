"""圧縮埋め込み形式の提出用 `main.cpp` を生成する CLI。"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..submission.payload_codec import (
    PAYLOAD_CODEC_BASE91,
    PAYLOAD_CODEC_BASE122,
    PAYLOAD_CODEC_HUFF91,
    PAYLOAD_CODEC_HUFF122,
    encode_model_payload,
)
from . import export_submission_main_cpp as student_export

DEFAULT_MAX_SOURCE_BYTES = 512 * 1024


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """内部ヘルパー: `normalize_state_dict_keys` を実行する。

    Args:
        state_dict (dict[str, Any]): state_dict の値。

    Returns:
        dict[str, Any]: 計算結果。
    """
    prefixes = ("_orig_mod.", "module.")
    out: dict[str, Any] = {}
    changed = False
    for key, value in state_dict.items():
        nk = str(key)
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
                changed = True
        if nk in out:
            raise ValueError(f"duplicate state_dict key after normalization: {nk!r}")
        out[nk] = value
    return out if changed else state_dict


def _as_f16_array(state_dict: dict[str, Any], key: str) -> np.ndarray:
    """内部ヘルパー: `as_f16_array` を実行する。

    Args:
        state_dict (dict[str, Any]): state_dict の値。
        key (str): key の値。

    Returns:
        np.ndarray: 計算結果。
    """
    v = state_dict.get(key)
    if v is None or not torch.is_tensor(v):
        raise KeyError(f"missing tensor in checkpoint state_dict: {key}")
    arr = v.detach().cpu().numpy()
    if arr.dtype.kind != "f":
        raise ValueError(f"tensor must be float for C++ export: {key} dtype={arr.dtype}")
    return np.asarray(arr, dtype=np.float16, order="C")


def _pack_f16_blob(tensors: list[np.ndarray]) -> tuple[bytes, list[int]]:
    """内部ヘルパー: `pack_f16_blob` を実行する。

    Args:
        tensors (list[np.ndarray]): tensors の値。

    Returns:
        tuple[bytes, list[int]]: 計算結果。
    """
    counts: list[int] = []
    chunks: list[np.ndarray] = []
    for t in tensors:
        h = t.view(np.uint16).reshape(-1)
        counts.append(int(h.size))
        chunks.append(h)
    merged = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.uint16)
    return merged.tobytes(order="C"), counts


def _collect_exp002_resnet_policy_tensors(state_dict: dict[str, Any], *, blocks: int) -> list[np.ndarray]:
    """内部ヘルパー: `collect_exp002_resnet_policy_tensors` を実行する。

    Args:
        state_dict (dict[str, Any]): state_dict の値。
        blocks (int): blocks の値。

    Returns:
        list[np.ndarray]: 計算結果。
    """
    out: list[np.ndarray] = []
    out.append(_as_f16_array(state_dict, "stem.0.weight"))
    out.append(_as_f16_array(state_dict, "stem.1.weight"))
    out.append(_as_f16_array(state_dict, "stem.1.bias"))
    for bi in range(int(blocks)):
        out.append(_as_f16_array(state_dict, f"res_blocks.{bi}.conv1.weight"))
        out.append(_as_f16_array(state_dict, f"res_blocks.{bi}.gn1.weight"))
        out.append(_as_f16_array(state_dict, f"res_blocks.{bi}.gn1.bias"))
        out.append(_as_f16_array(state_dict, f"res_blocks.{bi}.conv2.weight"))
        out.append(_as_f16_array(state_dict, f"res_blocks.{bi}.gn2.weight"))
        out.append(_as_f16_array(state_dict, f"res_blocks.{bi}.gn2.bias"))
    out.append(_as_f16_array(state_dict, "policy_head.weight"))
    out.append(_as_f16_array(state_dict, "policy_head.bias"))
    return out


def _collect_teacher_p0_policy_tensors(state_dict: dict[str, Any], *, num_blocks: int) -> list[np.ndarray]:
    """内部ヘルパー: `collect_teacher_p0_policy_tensors` を実行する。

    Args:
        state_dict (dict[str, Any]): state_dict の値。
        num_blocks (int): num_blocks の値。

    Returns:
        list[np.ndarray]: 計算結果。
    """
    out: list[np.ndarray] = []
    out.append(_as_f16_array(state_dict, "stem.0.weight"))
    out.append(_as_f16_array(state_dict, "stem.0.bias"))
    for bi in range(int(num_blocks)):
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv1.weight"))
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv1.bias"))
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv2.weight"))
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv2.bias"))
    out.append(_as_f16_array(state_dict, "policy_conv.weight"))
    out.append(_as_f16_array(state_dict, "policy_conv.bias"))
    return out


def _cpp_raw_lit(s: str, *, prefix: str = "") -> str:
    """内部ヘルパー: `cpp_raw_lit` を実行する。

    Args:
        s (str): s の値。
        prefix (str): prefix の値。

    Returns:
        str: 計算結果。
    """
    for delim in ("", "_", "x", "X", "q", "Q", "r", "R", "z", "Z"):
        marker = ")" + delim + '"'
        if marker not in s:
            return f'{prefix}R"{delim}({s}){delim}"'
    for k in range(4096):
        delim = f"Q{k}X"
        marker = ")" + delim + '"'
        if marker not in s:
            return f'{prefix}R"{delim}({s}){delim}"'
    raise RuntimeError("failed to choose C++ raw string delimiter")


def _emit_payload_literal(encoded: str, *, chunk: int = 16384) -> str:
    """内部ヘルパー: `emit_payload_literal` を実行する。

    Args:
        encoded (str): encoded の値。
        chunk (int): chunk の値。

    Returns:
        str: 計算結果。
    """
    if not encoded:
        return '""'
    lines: list[str] = []
    for i in range(0, len(encoded), chunk):
        lines.append(_cpp_raw_lit(encoded[i : i + chunk]))
    return "\n".join(lines)


TEMPLATE_EXP002_CPP = r"""// GENERATED FILE. DO NOT EDIT.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ahc061/base/state.hpp"
#include "ahc061/features/feature_common.hpp"
#include "ahc061/features/feature_registry.hpp"
#include "ahc061/game/generator.hpp"
#include "ahc061/opponent/a_softmax_laplace.hpp"
#include "ahc061/opponent/adf_beta_estimator.hpp"
#include "ahc061/opponent/move_summary.hpp"
#include "ahc061/opponent/pf.hpp"

namespace {

constexpr int kBoardChannels = __BOARD_CHANNELS__;
constexpr int kBoardSize = __BOARD_SIZE__;
constexpr int kActionDim = __ACTION_DIM__;
constexpr int kHiddenChannels = __HIDDEN_CHANNELS__;
constexpr int kNumBlocks = __NUM_BLOCKS__;
constexpr bool kPfEnabled = __PF_ENABLED__;
constexpr int kTensorCount = __TENSOR_COUNT__;
constexpr int kPayloadCodec = __PAYLOAD_CODEC__;
constexpr std::size_t kModelBlobBytes = static_cast<std::size_t>(__MODEL_BLOB_BYTES__);

static const int kTensorHalfCounts[kTensorCount] = {__TENSOR_HALF_COUNTS__};
static constexpr const char* kFeatureId = "__FEATURE_ID__";

static const char kModelBlobEncoded[] =
__MODEL_BLOB_ENCODED_LINES__
;

constexpr int kPayloadCodecBase91 = __PAYLOAD_CODEC_BASE91__;
constexpr int kPayloadCodecBase122 = __PAYLOAD_CODEC_BASE122__;
constexpr int kPayloadCodecHuff122 = __PAYLOAD_CODEC_HUFF122__;
constexpr int kPayloadCodecHuff91 = __PAYLOAD_CODEC_HUFF91__;
constexpr int kTtaMode = __TTA_MODE__;
constexpr int kTtaK = __TTA_K__;
constexpr int kTtaAutoOffMs = __TTA_AUTO_OFF_MS__;

inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = static_cast<std::uint32_t>((h >> 10) & 0x1Fu);
    std::uint32_t mant = static_cast<std::uint32_t>(h & 0x03FFu);
    std::uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

std::vector<std::uint8_t> base91_decode(std::string_view s) {
    static const std::array<std::int16_t, 256> kDec = [] {
        std::array<std::int16_t, 256> t{};
        t.fill(-1);
        int idx = 0;
        for (int c = 1; c <= 31; c++) {
            if (c == 10 || c == 13)
                continue;
            t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(idx++);
        }
        for (int c = 32; c <= 127; c++)
            t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(idx++);
        return t;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(s.size() * 13 / 16 + 8);

    int v = -1;
    std::uint32_t b = 0;
    int n = 0;
    for (unsigned char uc : s) {
        if (uc == '\n')
            continue;
        const int d = static_cast<int>(kDec[uc]);
        if (d < 0)
            continue;
        if (v < 0) {
            v = d;
            continue;
        }
        const int val = v + d * 125;
        b |= static_cast<std::uint32_t>(val) << n;
        n += ((val & 8191) < 7433) ? 14 : 13;
        while (n > 7) {
            out.push_back(static_cast<std::uint8_t>(b & 0xFFU));
            b >>= 8;
            n -= 8;
        }
        v = -1;
    }
    if (v >= 0)
        out.push_back(static_cast<std::uint8_t>((b | (static_cast<std::uint32_t>(v) << n)) & 0xFFU));
    return out;
}

std::vector<std::uint8_t> base122_decode(std::string_view s) {
    static constexpr std::array<std::uint8_t, 7> kIllegals = {0, 10, 13, 34, 38, 63, 92};
    static constexpr std::uint8_t kShortened = 7;
    std::vector<std::uint8_t> out;
    out.reserve(s.size());

    std::uint8_t cur_byte = 0;
    int bit_of_byte = 0;
    const auto push7 = [&](std::uint8_t seven) {
        const std::uint16_t byte = static_cast<std::uint16_t>(seven & 0x7FU) << 1;
        cur_byte |= static_cast<std::uint8_t>(byte >> bit_of_byte);
        bit_of_byte += 7;
        if (bit_of_byte >= 8) {
            out.push_back(cur_byte);
            bit_of_byte -= 8;
            cur_byte = static_cast<std::uint8_t>((byte << (7 - bit_of_byte)) & 0xFFU);
        }
    };

    for (std::size_t i = 0; i < s.size(); i++) {
        const std::uint8_t b0 = static_cast<std::uint8_t>(s[i]);
        std::uint16_t cp = 0;
        if ((b0 & 0x80U) == 0U) {
            cp = b0;
        } else {
            if ((b0 & 0xE0U) != 0xC0U || (i + 1) >= s.size())
                continue;
            const std::uint8_t b1 = static_cast<std::uint8_t>(s[++i]);
            if ((b1 & 0xC0U) != 0x80U)
                continue;
            cp = static_cast<std::uint16_t>(((b0 & 0x1FU) << 6) | (b1 & 0x3FU));
        }

        if (cp > 127U) {
            const std::uint8_t illegal_index = static_cast<std::uint8_t>((cp >> 8) & 0x07U);
            if (illegal_index != kShortened) {
                if (illegal_index >= kIllegals.size())
                    continue;
                push7(kIllegals[illegal_index]);
            }
            push7(static_cast<std::uint8_t>(cp & 0x7FU));
        } else {
            push7(static_cast<std::uint8_t>(cp));
        }
    }
    return out;
}

std::vector<std::uint8_t> huff122_decompress(const std::vector<std::uint8_t>& blob, std::size_t expected_size) {
    std::vector<std::uint8_t> out;
    out.reserve(expected_size);
    if (expected_size == 0)
        return out;
    if (blob.size() < 256ULL)
        return {};

    struct SymLen {
        std::uint16_t sym;
        std::uint8_t len;
    };
    std::vector<SymLen> syms;
    syms.reserve(256);
    for (int s = 0; s < 256; s++) {
        const std::uint8_t ln = blob[static_cast<std::size_t>(s)];
        if (ln > 0)
            syms.push_back(SymLen{static_cast<std::uint16_t>(s), ln});
    }
    if (syms.empty())
        return {};

    std::sort(syms.begin(), syms.end(), [](const SymLen& a, const SymLen& b) {
        if (a.len != b.len)
            return a.len < b.len;
        return a.sym < b.sym;
    });

    std::vector<int> left(1, -1), right(1, -1), value(1, -1);
    std::uint64_t code = 0;
    int prev_len = 0;
    for (const SymLen sl : syms) {
        const int ln = static_cast<int>(sl.len);
        if (ln < prev_len || ln > 63)
            return {};
        code <<= static_cast<std::uint64_t>(ln - prev_len);
        int node = 0;
        for (int k = ln - 1; k >= 0; k--) {
            const int bit = static_cast<int>((code >> static_cast<unsigned>(k)) & 1ULL);
            int nxt = bit ? right[node] : left[node];
            if (nxt < 0) {
                nxt = static_cast<int>(value.size());
                left.push_back(-1);
                right.push_back(-1);
                value.push_back(-1);
                if (bit)
                    right[node] = nxt;
                else
                    left[node] = nxt;
            }
            node = nxt;
        }
        value[node] = static_cast<int>(sl.sym);
        code += 1ULL;
        prev_len = ln;
    }

    int node = 0;
    for (std::size_t i = 256ULL; i < blob.size(); i++) {
        const std::uint8_t by = blob[i];
        for (int k = 7; k >= 0; k--) {
            const int bit = static_cast<int>((by >> static_cast<unsigned>(k)) & 1U);
            node = bit ? right[node] : left[node];
            if (node < 0)
                return {};
            const int sym = value[node];
            if (sym >= 0) {
                out.push_back(static_cast<std::uint8_t>(sym));
                if (out.size() == expected_size)
                    return out;
                node = 0;
            }
        }
    }
    return {};
}

std::vector<std::uint8_t> decode_payload(std::string_view encoded) {
    if (kPayloadCodec == kPayloadCodecBase91) {
        return base91_decode(encoded);
    }
    if (kPayloadCodec == kPayloadCodecBase122) {
        return base122_decode(encoded);
    }
    if (kPayloadCodec == kPayloadCodecHuff122) {
        return huff122_decompress(base122_decode(encoded), kModelBlobBytes);
    }
    if (kPayloadCodec == kPayloadCodecHuff91) {
        return huff122_decompress(base91_decode(encoded), kModelBlobBytes);
    }
    return {};
}

inline int pick_gn_groups(int channels) {
    for (int g : {8, 4, 2, 1}) {
        if (channels % g == 0)
            return g;
    }
    return 1;
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

struct ModelWeights {
    std::vector<float> stem_w;
    std::vector<float> stem_gn_w;
    std::vector<float> stem_gn_b;

    struct Block {
        std::vector<float> conv1_w;
        std::vector<float> gn1_w;
        std::vector<float> gn1_b;
        std::vector<float> conv2_w;
        std::vector<float> gn2_w;
        std::vector<float> gn2_b;
    };
    std::vector<Block> blocks;

    std::vector<float> policy_w;
    std::vector<float> policy_b;
};

ModelWeights load_model_weights() {
    const std::string_view encoded(kModelBlobEncoded, sizeof(kModelBlobEncoded) - 1ULL);
    const std::vector<std::uint8_t> bytes = decode_payload(encoded);
    if (bytes.size() != kModelBlobBytes)
        throw std::runtime_error("invalid decoded payload size");
    if ((bytes.size() & 1ULL) != 0ULL)
        throw std::runtime_error("invalid model blob bytes: odd length");

    std::vector<std::uint16_t> halves(bytes.size() / 2ULL);
    for (std::size_t i = 0; i < halves.size(); i++) {
        halves[i] = static_cast<std::uint16_t>(bytes[2ULL * i]) |
                    static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[2ULL * i + 1ULL]) << 8);
    }

    int expect_total = 0;
    for (int i = 0; i < kTensorCount; i++)
        expect_total += kTensorHalfCounts[i];
    if (expect_total != static_cast<int>(halves.size()))
        throw std::runtime_error("model blob size mismatch");

    std::size_t off = 0;
    int idx = 0;
    auto take = [&](int n) -> std::vector<float> {
        std::vector<float> v(static_cast<std::size_t>(n));
        for (int i = 0; i < n; i++)
            v[static_cast<std::size_t>(i)] = half_to_float(halves[off + static_cast<std::size_t>(i)]);
        off += static_cast<std::size_t>(n);
        return v;
    };

    ModelWeights w;
    w.stem_w = take(kTensorHalfCounts[idx++]);
    w.stem_gn_w = take(kTensorHalfCounts[idx++]);
    w.stem_gn_b = take(kTensorHalfCounts[idx++]);

    w.blocks.resize(static_cast<std::size_t>(kNumBlocks));
    for (int bi = 0; bi < kNumBlocks; bi++) {
        auto& b = w.blocks[static_cast<std::size_t>(bi)];
        b.conv1_w = take(kTensorHalfCounts[idx++]);
        b.gn1_w = take(kTensorHalfCounts[idx++]);
        b.gn1_b = take(kTensorHalfCounts[idx++]);
        b.conv2_w = take(kTensorHalfCounts[idx++]);
        b.gn2_w = take(kTensorHalfCounts[idx++]);
        b.gn2_b = take(kTensorHalfCounts[idx++]);
    }

    w.policy_w = take(kTensorHalfCounts[idx++]);
    w.policy_b = take(kTensorHalfCounts[idx++]);

    if (idx != kTensorCount || off != halves.size())
        throw std::runtime_error("tensor decode index mismatch");
    return w;
}

void conv3x3_bias_false(
    const float* in,
    int in_ch,
    int out_ch,
    const std::vector<float>& w,
    std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(out_ch * kActionDim), 0.0f);
    for (int oc = 0; oc < out_ch; oc++) {
        for (int x = 0; x < kBoardSize; x++) {
            for (int y = 0; y < kBoardSize; y++) {
                float s = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kx = 0; kx < 3; kx++) {
                        const int xx = x + kx - 1;
                        if (xx < 0 || xx >= kBoardSize)
                            continue;
                        for (int ky = 0; ky < 3; ky++) {
                            const int yy = y + ky - 1;
                            if (yy < 0 || yy >= kBoardSize)
                                continue;
                            const std::size_t wi =
                                ((((static_cast<std::size_t>(oc) * static_cast<std::size_t>(in_ch) + static_cast<std::size_t>(ic)) * 3ULL +
                                   static_cast<std::size_t>(kx)) *
                                      3ULL) +
                                 static_cast<std::size_t>(ky));
                            const std::size_t ii =
                                (static_cast<std::size_t>(ic) * static_cast<std::size_t>(kActionDim)) +
                                static_cast<std::size_t>(xx * kBoardSize + yy);
                            s += w[wi] * in[ii];
                        }
                    }
                }
                const std::size_t oi =
                    (static_cast<std::size_t>(oc) * static_cast<std::size_t>(kActionDim)) +
                    static_cast<std::size_t>(x * kBoardSize + y);
                out[oi] = s;
            }
        }
    }
}

void group_norm_inplace(
    std::vector<float>& x,
    int channels,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    bool with_silu) {
    const int groups = pick_gn_groups(channels);
    const int cpg = channels / groups;
    constexpr float eps = 1.0e-5f;

    for (int g = 0; g < groups; g++) {
        const int c0 = g * cpg;
        const int c1 = c0 + cpg;
        double sum = 0.0;
        double sq = 0.0;
        int cnt = 0;
        for (int c = c0; c < c1; c++) {
            const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(kActionDim);
            for (int i = 0; i < kActionDim; i++) {
                const float v = x[base + static_cast<std::size_t>(i)];
                sum += static_cast<double>(v);
                sq += static_cast<double>(v) * static_cast<double>(v);
                cnt++;
            }
        }
        const float mean = static_cast<float>(sum / static_cast<double>(cnt));
        float var = static_cast<float>(sq / static_cast<double>(cnt) - static_cast<double>(mean) * static_cast<double>(mean));
        if (var < 0.0f)
            var = 0.0f;
        const float inv_std = 1.0f / std::sqrt(var + eps);

        for (int c = c0; c < c1; c++) {
            const float gw = gamma[static_cast<std::size_t>(c)];
            const float gb = beta[static_cast<std::size_t>(c)];
            const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(kActionDim);
            for (int i = 0; i < kActionDim; i++) {
                float v = (x[base + static_cast<std::size_t>(i)] - mean) * inv_std;
                v = v * gw + gb;
                if (with_silu)
                    v = silu(v);
                x[base + static_cast<std::size_t>(i)] = v;
            }
        }
    }
}

struct Exp002ResNetPolicy {
    explicit Exp002ResNetPolicy(ModelWeights&& w) : w_(std::move(w)) {}

    void forward_logits(const float* board, float* logits) {
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;

        conv3x3_bias_false(board, kBoardChannels, kHiddenChannels, w_.stem_w, x);
        group_norm_inplace(x, kHiddenChannels, w_.stem_gn_w, w_.stem_gn_b, true);

        for (int bi = 0; bi < kNumBlocks; bi++) {
            const auto& b = w_.blocks[static_cast<std::size_t>(bi)];
            conv3x3_bias_false(x.data(), kHiddenChannels, kHiddenChannels, b.conv1_w, y);
            group_norm_inplace(y, kHiddenChannels, b.gn1_w, b.gn1_b, true);
            conv3x3_bias_false(y.data(), kHiddenChannels, kHiddenChannels, b.conv2_w, z);
            group_norm_inplace(z, kHiddenChannels, b.gn2_w, b.gn2_b, false);
            for (std::size_t i = 0; i < x.size(); i++)
                x[i] = silu(x[i] + z[i]);
        }

        const float bias = w_.policy_b.empty() ? 0.0f : w_.policy_b[0];
        for (int i = 0; i < kActionDim; i++) {
            float s = bias;
            for (int c = 0; c < kHiddenChannels; c++) {
                const std::size_t wi = static_cast<std::size_t>(c);
                const std::size_t xi = static_cast<std::size_t>(c * kActionDim + i);
                s += w_.policy_w[wi] * x[xi];
            }
            logits[i] = s;
        }
    }

  private:
    ModelWeights w_;
};

void recompute_scores(const ahc061::State& st, std::array<std::int64_t, ahc061::M_MAX>& score) {
    score.fill(0);
    for (int i = 0; i < ahc061::CELL_MAX; i++) {
        const int o = static_cast<int>(st.owner[i]);
        if (0 <= o && o < st.m) {
            score[static_cast<std::size_t>(o)] +=
                static_cast<std::int64_t>(st.value[i]) * static_cast<std::int64_t>(st.level[i]);
        }
    }
}

int select_action_argmax_masked(const float* logits, const std::uint8_t* mask) {
    int best_i = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < kActionDim; i++) {
        if (mask[i] == 0)
            continue;
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    return best_i;
}

inline float logaddexp_pair(float a, float b) {
    if (!std::isfinite(a))
        return b;
    if (!std::isfinite(b))
        return a;
    const float m = (a > b) ? a : b;
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

inline int resolve_tta_k_runtime() {
    int k = kTtaK;
    if (!(k == 2 || k == 4 || k == 8))
        k = 8;
    if (kTtaMode == 0)
        return 1;
    if (kTtaAutoOffMs <= 0)
        return k;
    using clock = std::chrono::steady_clock;
    static const auto start_tp = clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_tp).count();
    if (ms > static_cast<long long>(kTtaAutoOffMs))
        return 1;
    return k;
}

const std::array<std::array<std::uint8_t, kActionDim>, 8>& tta_perm() {
    static const auto perms = [] {
        std::array<std::array<std::uint8_t, kActionDim>, 8> out{};
        for (int x = 0; x < kBoardSize; x++) {
            for (int y = 0; y < kBoardSize; y++) {
                const int i = x * kBoardSize + y;
                const auto set = [&](int t, int tx, int ty) {
                    out[static_cast<std::size_t>(t)][static_cast<std::size_t>(i)] =
                        static_cast<std::uint8_t>(tx * kBoardSize + ty);
                };
                set(0, x, y);                                       // identity
                set(1, y, kBoardSize - 1 - x);                     // rot90
                set(2, kBoardSize - 1 - x, kBoardSize - 1 - y);    // rot180
                set(3, kBoardSize - 1 - y, x);                     // rot270
                set(4, x, kBoardSize - 1 - y);                     // mirror
                set(5, kBoardSize - 1 - y, kBoardSize - 1 - x);    // mirror+rot90
                set(6, kBoardSize - 1 - x, y);                     // mirror+rot180
                set(7, y, x);                                      // mirror+rot270
            }
        }
        return out;
    }();
    return perms;
}

void transform_board_by_perm(
    const float* board_src,
    const std::array<std::uint8_t, kActionDim>& perm,
    std::array<float, static_cast<std::size_t>(kBoardChannels * kActionDim)>& board_dst) {
    for (int c = 0; c < kBoardChannels; c++) {
        const std::size_t b = static_cast<std::size_t>(c) * static_cast<std::size_t>(kActionDim);
        for (int i = 0; i < kActionDim; i++) {
            const int j = static_cast<int>(perm[static_cast<std::size_t>(i)]);
            board_dst[b + static_cast<std::size_t>(j)] = board_src[b + static_cast<std::size_t>(i)];
        }
    }
}

void transform_mask_by_perm(
    const std::uint8_t* mask_src,
    const std::array<std::uint8_t, kActionDim>& perm,
    std::array<std::uint8_t, kActionDim>& mask_dst) {
    mask_dst.fill(0);
    for (int i = 0; i < kActionDim; i++) {
        const int j = static_cast<int>(perm[static_cast<std::size_t>(i)]);
        mask_dst[static_cast<std::size_t>(j)] = mask_src[static_cast<std::size_t>(i)];
    }
}

int select_action_with_tta(
    Exp002ResNetPolicy& policy,
    const float* board,
    const std::uint8_t* mask,
    float* logits_out) {
    const int tta_k_runtime = resolve_tta_k_runtime();
    if (kTtaMode == 0 || tta_k_runtime <= 1) {
        policy.forward_logits(board, logits_out);
        return select_action_argmax_masked(logits_out, mask);
    }

    std::array<float, kActionDim> acc{};
    if (kTtaMode == 1) {
        acc.fill(-std::numeric_limits<float>::infinity());
    } else {
        acc.fill(0.0f);
    }

    const auto& perm_all = tta_perm();
    std::array<float, static_cast<std::size_t>(kBoardChannels * kActionDim)> board_t{};
    std::array<float, kActionDim> logits_t{};
    std::array<std::uint8_t, kActionDim> mask_t{};

    for (int tk = 0; tk < tta_k_runtime; tk++) {
        const auto& perm = perm_all[static_cast<std::size_t>(tk)];
        transform_board_by_perm(board, perm, board_t);
        transform_mask_by_perm(mask, perm, mask_t);
        policy.forward_logits(board_t.data(), logits_t.data());

        float max_v = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (int j = 0; j < kActionDim; j++) {
            if (mask_t[static_cast<std::size_t>(j)] == 0)
                continue;
            const float v = logits_t[static_cast<std::size_t>(j)];
            if (!any || v > max_v)
                max_v = v;
            any = true;
        }
        if (!any)
            continue;
        double sum_exp = 0.0;
        for (int j = 0; j < kActionDim; j++) {
            if (mask_t[static_cast<std::size_t>(j)] == 0)
                continue;
            sum_exp += std::exp(static_cast<double>(logits_t[static_cast<std::size_t>(j)] - max_v));
        }
        const float log_z = max_v + static_cast<float>(std::log(sum_exp));

        for (int i = 0; i < kActionDim; i++) {
            if (mask[static_cast<std::size_t>(i)] == 0)
                continue;
            const int j = static_cast<int>(perm[static_cast<std::size_t>(i)]);
            if (mask_t[static_cast<std::size_t>(j)] == 0)
                continue;
            const float logp = logits_t[static_cast<std::size_t>(j)] - log_z;
            if (kTtaMode == 1) {
                acc[static_cast<std::size_t>(i)] = logaddexp_pair(acc[static_cast<std::size_t>(i)], logp);
            } else {
                acc[static_cast<std::size_t>(i)] += logp;
            }
        }
    }

    for (int i = 0; i < kActionDim; i++) {
        logits_out[i] = acc[static_cast<std::size_t>(i)];
    }
    return select_action_argmax_masked(logits_out, mask);
}

}  // namespace

int main() {
    using namespace ahc061;

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (feature_channels(kFeatureId) != kBoardChannels)
        throw std::runtime_error("feature_channels mismatch with model board_channels");

    int n = 0, m = 0, t_max = 0, u_max = 0;
    if (!(std::cin >> n >> m >> t_max >> u_max))
        return 0;
    if (n != kBoardSize || kActionDim != n * n)
        throw std::runtime_error("unsupported board size");

    State st{};
    st.m = m;
    st.t_max = t_max;
    st.u_max = u_max;

    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            int v = 0;
            std::cin >> v;
            st.value[cell_index(x, y)] = v;
        }
    }

    st.owner.fill(-1);
    st.level.fill(0);
    for (int p = 0; p < m; p++) {
        int x = 0, y = 0;
        std::cin >> x >> y;
        st.ex[p] = static_cast<std::uint8_t>(x);
        st.ey[p] = static_cast<std::uint8_t>(y);
        const int id = cell_index(x, y);
        st.owner[id] = static_cast<std::int8_t>(p);
        st.level[id] = 1;
    }

    ModelWeights mw = load_model_weights();
    Exp002ResNetPolicy policy(std::move(mw));

    std::array<ParticleFilterSMC, M_MAX> pf{};
    std::array<ASoftmaxLaplaceEstimator, M_MAX> a_softmax{};
    std::array<AdfBetaEstimator, M_MAX> adf_beta{};
    std::array<std::int64_t, M_MAX> score{};
    recompute_scores(st, score);

    const std::uint64_t base_seed = compute_case_seed_for_pf(st);
    for (int p = 0; p < M_MAX; p++) {
        const std::uint64_t s = base_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x9e3779b97f4a7c15ULL) ^
                                0x243f6a8885a308d3ULL;
        pf[static_cast<std::size_t>(p)].reset(s);
        a_softmax[static_cast<std::size_t>(p)].reset();
        adf_beta[static_cast<std::size_t>(p)].reset();
    }

    const auto& fs = get_feature_set(kFeatureId);
    const bool update_pf = kPfEnabled && (fs.next_mode == NextMode::k_uniform_or_pf);
    const bool update_a_softmax = (fs.next_mode == NextMode::k_a_softmax_ut);
    const bool update_adf_beta = (fs.next_mode == NextMode::k_adf_beta);

    std::array<std::array<int, CELL_MAX>, M_MAX> cache_moves{};
    std::array<int, M_MAX> cache_move_cnt{};
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> cache_comp{};
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> cache_reach{};

    std::array<float, static_cast<std::size_t>(kBoardChannels * kActionDim)> board{};
    std::array<std::uint8_t, CELL_MAX> action_mask{};
    std::array<float, kActionDim> logits{};

    for (int turn = 0; turn < t_max; turn++) {
        const State st_start = st;

        FeatureCommon common{};
        compute_feature_common_into(
            st,
            turn,
            &pf,
            kPfEnabled,
            common,
            action_mask.data(),
            cache_moves,
            cache_move_cnt,
            cache_comp,
            cache_reach,
            fs.next_mode,
            &a_softmax,
            &adf_beta,
            &score);
        fs.write_from_common(common, board.data());

        int action = select_action_with_tta(policy, board.data(), action_mask.data(), logits.data());
        if (action < 0 || action >= kActionDim || action_mask[static_cast<std::size_t>(action)] == 0) {
            for (int i = 0; i < kActionDim; i++) {
                if (action_mask[static_cast<std::size_t>(i)] != 0) {
                    action = i;
                    break;
                }
            }
        }

        const int ax = action / kBoardSize;
        const int ay = action % kBoardSize;
        std::cout << ax << ' ' << ay << '\n' << std::flush;

        std::array<int, M_MAX> selected_cell{};
        selected_cell.fill(0);
        for (int p = 0; p < m; p++) {
            int sx = 0, sy = 0;
            std::cin >> sx >> sy;
            selected_cell[static_cast<std::size_t>(p)] = cell_index(sx, sy);
        }

        if (update_pf || update_a_softmax || update_adf_beta) {
            for (int p = 1; p < m; p++) {
                const MoveSummary sum = summarize_ai_observation_from_moves(
                    st_start,
                    p,
                    selected_cell[static_cast<std::size_t>(p)],
                    cache_moves[static_cast<std::size_t>(p)].data(),
                    cache_move_cnt[static_cast<std::size_t>(p)]);
                if (update_a_softmax)
                    a_softmax[static_cast<std::size_t>(p)].update(sum);
                if (update_adf_beta)
                    adf_beta[static_cast<std::size_t>(p)].update(sum);
                if (update_pf)
                    pf[static_cast<std::size_t>(p)].update(sum);
            }
        }

        for (int p = 0; p < m; p++) {
            int ex = 0, ey = 0;
            std::cin >> ex >> ey;
            st.ex[p] = static_cast<std::uint8_t>(ex);
            st.ey[p] = static_cast<std::uint8_t>(ey);
        }
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                int o = 0;
                std::cin >> o;
                st.owner[cell_index(x, y)] = static_cast<std::int8_t>(o);
            }
        }
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                int lv = 0;
                std::cin >> lv;
                st.level[cell_index(x, y)] = static_cast<std::uint8_t>(lv);
            }
        }
        recompute_scores(st, score);
    }

    return 0;
}
"""


TEMPLATE_TEACHER_P0_CPP = r"""// GENERATED FILE. DO NOT EDIT.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ahc061/base/state.hpp"
#include "ahc061/features/feature_common.hpp"
#include "ahc061/features/feature_registry.hpp"
#include "ahc061/game/generator.hpp"
#include "ahc061/opponent/a_softmax_laplace.hpp"
#include "ahc061/opponent/adf_beta_estimator.hpp"
#include "ahc061/opponent/move_summary.hpp"
#include "ahc061/opponent/pf.hpp"

namespace {

constexpr int kBoardChannels = __BOARD_CHANNELS__;
constexpr int kBoardSize = __BOARD_SIZE__;
constexpr int kActionDim = __ACTION_DIM__;
constexpr int kWidth = __WIDTH__;
constexpr int kNumBlocks = __NUM_BLOCKS__;
constexpr int kActivationId = __ACTIVATION_ID__;  // 0:tanh 1:relu 2:silu
constexpr bool kPfEnabled = __PF_ENABLED__;
constexpr int kTensorCount = __TENSOR_COUNT__;
constexpr int kPayloadCodec = __PAYLOAD_CODEC__;
constexpr std::size_t kModelBlobBytes = static_cast<std::size_t>(__MODEL_BLOB_BYTES__);

static const int kTensorHalfCounts[kTensorCount] = {__TENSOR_HALF_COUNTS__};
static constexpr const char* kFeatureId = "__FEATURE_ID__";

static const char kModelBlobEncoded[] =
__MODEL_BLOB_ENCODED_LINES__
;

constexpr int kPayloadCodecBase91 = __PAYLOAD_CODEC_BASE91__;
constexpr int kPayloadCodecBase122 = __PAYLOAD_CODEC_BASE122__;
constexpr int kPayloadCodecHuff122 = __PAYLOAD_CODEC_HUFF122__;
constexpr int kPayloadCodecHuff91 = __PAYLOAD_CODEC_HUFF91__;

inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = static_cast<std::uint32_t>((h >> 10) & 0x1Fu);
    std::uint32_t mant = static_cast<std::uint32_t>(h & 0x03FFu);
    std::uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

std::vector<std::uint8_t> base91_decode(std::string_view s) {
    static const std::array<std::int16_t, 256> kDec = [] {
        std::array<std::int16_t, 256> t{};
        t.fill(-1);
        int idx = 0;
        for (int c = 1; c <= 31; c++) {
            if (c == 10 || c == 13)
                continue;
            t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(idx++);
        }
        for (int c = 32; c <= 127; c++)
            t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(idx++);
        return t;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(s.size() * 13 / 16 + 8);

    int v = -1;
    std::uint32_t b = 0;
    int n = 0;
    for (unsigned char uc : s) {
        if (uc == '\n')
            continue;
        const int d = static_cast<int>(kDec[uc]);
        if (d < 0)
            continue;
        if (v < 0) {
            v = d;
            continue;
        }
        const int val = v + d * 125;
        b |= static_cast<std::uint32_t>(val) << n;
        n += ((val & 8191) < 7433) ? 14 : 13;
        while (n > 7) {
            out.push_back(static_cast<std::uint8_t>(b & 0xFFU));
            b >>= 8;
            n -= 8;
        }
        v = -1;
    }
    if (v >= 0)
        out.push_back(static_cast<std::uint8_t>((b | (static_cast<std::uint32_t>(v) << n)) & 0xFFU));
    return out;
}

std::vector<std::uint8_t> base122_decode(std::string_view s) {
    static constexpr std::array<std::uint8_t, 7> kIllegals = {0, 10, 13, 34, 38, 63, 92};
    static constexpr std::uint8_t kShortened = 7;
    std::vector<std::uint8_t> out;
    out.reserve(s.size());

    std::uint8_t cur_byte = 0;
    int bit_of_byte = 0;
    const auto push7 = [&](std::uint8_t seven) {
        const std::uint16_t byte = static_cast<std::uint16_t>(seven & 0x7FU) << 1;
        cur_byte |= static_cast<std::uint8_t>(byte >> bit_of_byte);
        bit_of_byte += 7;
        if (bit_of_byte >= 8) {
            out.push_back(cur_byte);
            bit_of_byte -= 8;
            cur_byte = static_cast<std::uint8_t>((byte << (7 - bit_of_byte)) & 0xFFU);
        }
    };

    for (std::size_t i = 0; i < s.size(); i++) {
        const std::uint8_t b0 = static_cast<std::uint8_t>(s[i]);
        std::uint16_t cp = 0;
        if ((b0 & 0x80U) == 0U) {
            cp = b0;
        } else {
            if ((b0 & 0xE0U) != 0xC0U || (i + 1) >= s.size())
                continue;
            const std::uint8_t b1 = static_cast<std::uint8_t>(s[++i]);
            if ((b1 & 0xC0U) != 0x80U)
                continue;
            cp = static_cast<std::uint16_t>(((b0 & 0x1FU) << 6) | (b1 & 0x3FU));
        }

        if (cp > 127U) {
            const std::uint8_t illegal_index = static_cast<std::uint8_t>((cp >> 8) & 0x07U);
            if (illegal_index != kShortened) {
                if (illegal_index >= kIllegals.size())
                    continue;
                push7(kIllegals[illegal_index]);
            }
            push7(static_cast<std::uint8_t>(cp & 0x7FU));
        } else {
            push7(static_cast<std::uint8_t>(cp));
        }
    }
    return out;
}

std::vector<std::uint8_t> huff122_decompress(const std::vector<std::uint8_t>& blob, std::size_t expected_size) {
    std::vector<std::uint8_t> out;
    out.reserve(expected_size);
    if (expected_size == 0)
        return out;
    if (blob.size() < 256ULL)
        return {};

    struct SymLen {
        std::uint16_t sym;
        std::uint8_t len;
    };
    std::vector<SymLen> syms;
    syms.reserve(256);
    for (int s = 0; s < 256; s++) {
        const std::uint8_t ln = blob[static_cast<std::size_t>(s)];
        if (ln > 0)
            syms.push_back(SymLen{static_cast<std::uint16_t>(s), ln});
    }
    if (syms.empty())
        return {};

    std::sort(syms.begin(), syms.end(), [](const SymLen& a, const SymLen& b) {
        if (a.len != b.len)
            return a.len < b.len;
        return a.sym < b.sym;
    });

    std::vector<int> left(1, -1), right(1, -1), value(1, -1);
    std::uint64_t code = 0;
    int prev_len = 0;
    for (const SymLen sl : syms) {
        const int ln = static_cast<int>(sl.len);
        if (ln < prev_len || ln > 63)
            return {};
        code <<= static_cast<std::uint64_t>(ln - prev_len);
        int node = 0;
        for (int k = ln - 1; k >= 0; k--) {
            const int bit = static_cast<int>((code >> static_cast<unsigned>(k)) & 1ULL);
            int nxt = bit ? right[node] : left[node];
            if (nxt < 0) {
                nxt = static_cast<int>(value.size());
                left.push_back(-1);
                right.push_back(-1);
                value.push_back(-1);
                if (bit)
                    right[node] = nxt;
                else
                    left[node] = nxt;
            }
            node = nxt;
        }
        value[node] = static_cast<int>(sl.sym);
        code += 1ULL;
        prev_len = ln;
    }

    int node = 0;
    for (std::size_t i = 256ULL; i < blob.size(); i++) {
        const std::uint8_t by = blob[i];
        for (int k = 7; k >= 0; k--) {
            const int bit = static_cast<int>((by >> static_cast<unsigned>(k)) & 1U);
            node = bit ? right[node] : left[node];
            if (node < 0)
                return {};
            const int sym = value[node];
            if (sym >= 0) {
                out.push_back(static_cast<std::uint8_t>(sym));
                if (out.size() == expected_size)
                    return out;
                node = 0;
            }
        }
    }
    return {};
}

std::vector<std::uint8_t> decode_payload(std::string_view encoded) {
    if (kPayloadCodec == kPayloadCodecBase91) {
        return base91_decode(encoded);
    }
    if (kPayloadCodec == kPayloadCodecBase122) {
        return base122_decode(encoded);
    }
    if (kPayloadCodec == kPayloadCodecHuff122) {
        return huff122_decompress(base122_decode(encoded), kModelBlobBytes);
    }
    if (kPayloadCodec == kPayloadCodecHuff91) {
        return huff122_decompress(base91_decode(encoded), kModelBlobBytes);
    }
    return {};
}

struct ModelWeights {
    std::vector<float> stem_w;
    std::vector<float> stem_b;

    struct Block {
        std::vector<float> conv1_w;
        std::vector<float> conv1_b;
        std::vector<float> conv2_w;
        std::vector<float> conv2_b;
    };
    std::vector<Block> blocks;

    std::vector<float> policy_w;
    std::vector<float> policy_b;
};

ModelWeights load_model_weights() {
    const std::string_view encoded(kModelBlobEncoded, sizeof(kModelBlobEncoded) - 1ULL);
    const std::vector<std::uint8_t> bytes = decode_payload(encoded);
    if (bytes.size() != kModelBlobBytes)
        throw std::runtime_error("invalid decoded payload size");
    if ((bytes.size() & 1ULL) != 0ULL)
        throw std::runtime_error("invalid model blob bytes: odd length");

    std::vector<std::uint16_t> halves(bytes.size() / 2ULL);
    for (std::size_t i = 0; i < halves.size(); i++) {
        halves[i] = static_cast<std::uint16_t>(bytes[2ULL * i]) |
                    static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[2ULL * i + 1ULL]) << 8);
    }

    int expect_total = 0;
    for (int i = 0; i < kTensorCount; i++)
        expect_total += kTensorHalfCounts[i];
    if (expect_total != static_cast<int>(halves.size()))
        throw std::runtime_error("model blob size mismatch");

    std::size_t off = 0;
    int idx = 0;
    auto take = [&](int n) -> std::vector<float> {
        std::vector<float> v(static_cast<std::size_t>(n));
        for (int i = 0; i < n; i++)
            v[static_cast<std::size_t>(i)] = half_to_float(halves[off + static_cast<std::size_t>(i)]);
        off += static_cast<std::size_t>(n);
        return v;
    };

    ModelWeights w;
    w.stem_w = take(kTensorHalfCounts[idx++]);
    w.stem_b = take(kTensorHalfCounts[idx++]);

    w.blocks.resize(static_cast<std::size_t>(kNumBlocks));
    for (int bi = 0; bi < kNumBlocks; bi++) {
        auto& b = w.blocks[static_cast<std::size_t>(bi)];
        b.conv1_w = take(kTensorHalfCounts[idx++]);
        b.conv1_b = take(kTensorHalfCounts[idx++]);
        b.conv2_w = take(kTensorHalfCounts[idx++]);
        b.conv2_b = take(kTensorHalfCounts[idx++]);
    }

    w.policy_w = take(kTensorHalfCounts[idx++]);
    w.policy_b = take(kTensorHalfCounts[idx++]);

    if (idx != kTensorCount || off != halves.size())
        throw std::runtime_error("tensor decode index mismatch");
    return w;
}

inline float activation(float x) {
    if (kActivationId == 0) {
        return std::tanh(x);
    }
    if (kActivationId == 1) {
        return (x > 0.0f) ? x : 0.0f;
    }
    return x / (1.0f + std::exp(-x));
}

void conv3x3(
    const float* in,
    int in_ch,
    int out_ch,
    const std::vector<float>& w,
    const std::vector<float>& b,
    std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(out_ch * kActionDim), 0.0f);
    for (int oc = 0; oc < out_ch; oc++) {
        const float bias = b.empty() ? 0.0f : b[static_cast<std::size_t>(oc)];
        for (int x = 0; x < kBoardSize; x++) {
            for (int y = 0; y < kBoardSize; y++) {
                float s = bias;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kx = 0; kx < 3; kx++) {
                        const int xx = x + kx - 1;
                        if (xx < 0 || xx >= kBoardSize)
                            continue;
                        for (int ky = 0; ky < 3; ky++) {
                            const int yy = y + ky - 1;
                            if (yy < 0 || yy >= kBoardSize)
                                continue;
                            const std::size_t wi =
                                ((((static_cast<std::size_t>(oc) * static_cast<std::size_t>(in_ch) + static_cast<std::size_t>(ic)) * 3ULL +
                                   static_cast<std::size_t>(kx)) *
                                      3ULL) +
                                 static_cast<std::size_t>(ky));
                            const std::size_t ii =
                                (static_cast<std::size_t>(ic) * static_cast<std::size_t>(kActionDim)) +
                                static_cast<std::size_t>(xx * kBoardSize + yy);
                            s += w[wi] * in[ii];
                        }
                    }
                }
                const std::size_t oi =
                    (static_cast<std::size_t>(oc) * static_cast<std::size_t>(kActionDim)) +
                    static_cast<std::size_t>(x * kBoardSize + y);
                out[oi] = s;
            }
        }
    }
}

struct TeacherP0Policy {
    explicit TeacherP0Policy(ModelWeights&& w) : w_(std::move(w)) {}

    void forward_logits(const float* board, float* logits) {
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;

        conv3x3(board, kBoardChannels, kWidth, w_.stem_w, w_.stem_b, x);
        for (float& v : x)
            v = activation(v);

        for (int bi = 0; bi < kNumBlocks; bi++) {
            const auto& b = w_.blocks[static_cast<std::size_t>(bi)];
            conv3x3(x.data(), kWidth, kWidth, b.conv1_w, b.conv1_b, y);
            for (float& v : y)
                v = activation(v);
            conv3x3(y.data(), kWidth, kWidth, b.conv2_w, b.conv2_b, z);
            for (std::size_t i = 0; i < x.size(); i++) {
                x[i] = activation(x[i] + z[i]);
            }
        }

        const float bias = w_.policy_b.empty() ? 0.0f : w_.policy_b[0];
        for (int i = 0; i < kActionDim; i++) {
            float s = bias;
            for (int c = 0; c < kWidth; c++) {
                const std::size_t wi = static_cast<std::size_t>(c);
                const std::size_t xi = static_cast<std::size_t>(c * kActionDim + i);
                s += w_.policy_w[wi] * x[xi];
            }
            logits[i] = s;
        }
    }

  private:
    ModelWeights w_;
};

void recompute_scores(const ahc061::State& st, std::array<std::int64_t, ahc061::M_MAX>& score) {
    score.fill(0);
    for (int i = 0; i < ahc061::CELL_MAX; i++) {
        const int o = static_cast<int>(st.owner[i]);
        if (0 <= o && o < st.m) {
            score[static_cast<std::size_t>(o)] +=
                static_cast<std::int64_t>(st.value[i]) * static_cast<std::int64_t>(st.level[i]);
        }
    }
}

int select_action_argmax_masked(const float* logits, const std::uint8_t* mask) {
    int best_i = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < kActionDim; i++) {
        if (mask[i] == 0)
            continue;
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    return best_i;
}

}  // namespace

int main() {
    using namespace ahc061;

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (feature_channels(kFeatureId) != kBoardChannels)
        throw std::runtime_error("feature_channels mismatch with model board_channels");

    int n = 0, m = 0, t_max = 0, u_max = 0;
    if (!(std::cin >> n >> m >> t_max >> u_max))
        return 0;
    if (n != kBoardSize || kActionDim != n * n)
        throw std::runtime_error("unsupported board size");

    State st{};
    st.m = m;
    st.t_max = t_max;
    st.u_max = u_max;

    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            int v = 0;
            std::cin >> v;
            st.value[cell_index(x, y)] = v;
        }
    }

    st.owner.fill(-1);
    st.level.fill(0);
    for (int p = 0; p < m; p++) {
        int x = 0, y = 0;
        std::cin >> x >> y;
        st.ex[p] = static_cast<std::uint8_t>(x);
        st.ey[p] = static_cast<std::uint8_t>(y);
        const int id = cell_index(x, y);
        st.owner[id] = static_cast<std::int8_t>(p);
        st.level[id] = 1;
    }

    ModelWeights mw = load_model_weights();
    TeacherP0Policy policy(std::move(mw));

    std::array<ParticleFilterSMC, M_MAX> pf{};
    std::array<ASoftmaxLaplaceEstimator, M_MAX> a_softmax{};
    std::array<AdfBetaEstimator, M_MAX> adf_beta{};
    std::array<std::int64_t, M_MAX> score{};
    recompute_scores(st, score);

    const std::uint64_t base_seed = compute_case_seed_for_pf(st);
    for (int p = 0; p < M_MAX; p++) {
        const std::uint64_t s = base_seed ^ (static_cast<std::uint64_t>(p + 1) * 0x9e3779b97f4a7c15ULL) ^
                                0x243f6a8885a308d3ULL;
        pf[static_cast<std::size_t>(p)].reset(s);
        a_softmax[static_cast<std::size_t>(p)].reset();
        adf_beta[static_cast<std::size_t>(p)].reset();
    }

    const auto& fs = get_feature_set(kFeatureId);
    const bool update_pf = kPfEnabled && (fs.next_mode == NextMode::k_uniform_or_pf);
    const bool update_a_softmax = (fs.next_mode == NextMode::k_a_softmax_ut);
    const bool update_adf_beta = (fs.next_mode == NextMode::k_adf_beta);

    std::array<std::array<int, CELL_MAX>, M_MAX> cache_moves{};
    std::array<int, M_MAX> cache_move_cnt{};
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> cache_comp{};
    std::array<std::array<std::uint8_t, CELL_MAX>, M_MAX> cache_reach{};

    std::array<float, static_cast<std::size_t>(kBoardChannels * kActionDim)> board{};
    std::array<std::uint8_t, CELL_MAX> action_mask{};
    std::array<float, kActionDim> logits{};

    for (int turn = 0; turn < t_max; turn++) {
        const State st_start = st;

        FeatureCommon common{};
        compute_feature_common_into(
            st,
            turn,
            &pf,
            kPfEnabled,
            common,
            action_mask.data(),
            cache_moves,
            cache_move_cnt,
            cache_comp,
            cache_reach,
            fs.next_mode,
            &a_softmax,
            &adf_beta,
            &score);
        fs.write_from_common(common, board.data());

        policy.forward_logits(board.data(), logits.data());
        int action = select_action_argmax_masked(logits.data(), action_mask.data());
        if (action < 0 || action >= kActionDim || action_mask[static_cast<std::size_t>(action)] == 0) {
            for (int i = 0; i < kActionDim; i++) {
                if (action_mask[static_cast<std::size_t>(i)] != 0) {
                    action = i;
                    break;
                }
            }
        }

        const int ax = action / kBoardSize;
        const int ay = action % kBoardSize;
        std::cout << ax << ' ' << ay << '\n' << std::flush;

        std::array<int, M_MAX> selected_cell{};
        selected_cell.fill(0);
        for (int p = 0; p < m; p++) {
            int sx = 0, sy = 0;
            std::cin >> sx >> sy;
            selected_cell[static_cast<std::size_t>(p)] = cell_index(sx, sy);
        }

        if (update_pf || update_a_softmax || update_adf_beta) {
            for (int p = 1; p < m; p++) {
                const MoveSummary sum = summarize_ai_observation_from_moves(
                    st_start,
                    p,
                    selected_cell[static_cast<std::size_t>(p)],
                    cache_moves[static_cast<std::size_t>(p)].data(),
                    cache_move_cnt[static_cast<std::size_t>(p)]);
                if (update_a_softmax)
                    a_softmax[static_cast<std::size_t>(p)].update(sum);
                if (update_adf_beta)
                    adf_beta[static_cast<std::size_t>(p)].update(sum);
                if (update_pf)
                    pf[static_cast<std::size_t>(p)].update(sum);
            }
        }

        for (int p = 0; p < m; p++) {
            int ex = 0, ey = 0;
            std::cin >> ex >> ey;
            st.ex[p] = static_cast<std::uint8_t>(ex);
            st.ey[p] = static_cast<std::uint8_t>(ey);
        }
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                int o = 0;
                std::cin >> o;
                st.owner[cell_index(x, y)] = static_cast<std::int8_t>(o);
            }
        }
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                int lv = 0;
                std::cin >> lv;
                st.level[cell_index(x, y)] = static_cast<std::uint8_t>(lv);
            }
        }
        recompute_scores(st, score);
    }

    return 0;
}
"""


STUDENT_BLOB_DECL_CPP = r"""
constexpr int kPayloadCodec = __PAYLOAD_CODEC__;
constexpr std::size_t kModelBlobBytes = static_cast<std::size_t>(__MODEL_BLOB_BYTES__);

static const char kModelBlobEncoded[] =
__MODEL_BLOB_ENCODED_LINES__
    ;

constexpr int kPayloadCodecBase91 = __PAYLOAD_CODEC_BASE91__;
constexpr int kPayloadCodecBase122 = __PAYLOAD_CODEC_BASE122__;
constexpr int kPayloadCodecHuff122 = __PAYLOAD_CODEC_HUFF122__;
constexpr int kPayloadCodecHuff91 = __PAYLOAD_CODEC_HUFF91__;
"""

STUDENT_PAYLOAD_DECODE_CPP = r"""
std::vector<std::uint8_t> base91_decode(std::string_view s) {
    static const std::array<std::int16_t, 256> kDec = [] {
        std::array<std::int16_t, 256> t{};
        t.fill(-1);
        int idx = 0;
        for (int c = 1; c <= 31; c++) {
            if (c == 10 || c == 13)
                continue;
            t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(idx++);
        }
        for (int c = 32; c <= 127; c++)
            t[static_cast<std::uint8_t>(c)] = static_cast<std::int16_t>(idx++);
        return t;
    }();

    std::vector<std::uint8_t> out;
    out.reserve(s.size() * 13 / 16 + 8);

    int v = -1;
    std::uint32_t b = 0;
    int n = 0;
    for (unsigned char uc : s) {
        if (uc == '\n')
            continue;
        const int d = static_cast<int>(kDec[uc]);
        if (d < 0)
            continue;
        if (v < 0) {
            v = d;
            continue;
        }
        const int val = v + d * 125;
        b |= static_cast<std::uint32_t>(val) << n;
        n += ((val & 8191) < 7433) ? 14 : 13;
        while (n > 7) {
            out.push_back(static_cast<std::uint8_t>(b & 0xFFU));
            b >>= 8;
            n -= 8;
        }
        v = -1;
    }
    if (v >= 0)
        out.push_back(static_cast<std::uint8_t>((b | (static_cast<std::uint32_t>(v) << n)) & 0xFFU));
    return out;
}

std::vector<std::uint8_t> base122_decode(std::string_view s) {
    static constexpr std::array<std::uint8_t, 7> kIllegals = {0, 10, 13, 34, 38, 63, 92};
    static constexpr std::uint8_t kShortened = 7;
    std::vector<std::uint8_t> out;
    out.reserve(s.size());

    std::uint8_t cur_byte = 0;
    int bit_of_byte = 0;
    const auto push7 = [&](std::uint8_t seven) {
        const std::uint16_t byte = static_cast<std::uint16_t>(seven & 0x7FU) << 1;
        cur_byte |= static_cast<std::uint8_t>(byte >> bit_of_byte);
        bit_of_byte += 7;
        if (bit_of_byte >= 8) {
            out.push_back(cur_byte);
            bit_of_byte -= 8;
            cur_byte = static_cast<std::uint8_t>((byte << (7 - bit_of_byte)) & 0xFFU);
        }
    };

    for (std::size_t i = 0; i < s.size(); i++) {
        const std::uint8_t b0 = static_cast<std::uint8_t>(s[i]);
        std::uint16_t cp = 0;
        if ((b0 & 0x80U) == 0U) {
            cp = b0;
        } else {
            if ((b0 & 0xE0U) != 0xC0U || (i + 1) >= s.size())
                continue;
            const std::uint8_t b1 = static_cast<std::uint8_t>(s[++i]);
            if ((b1 & 0xC0U) != 0x80U)
                continue;
            cp = static_cast<std::uint16_t>(((b0 & 0x1FU) << 6) | (b1 & 0x3FU));
        }

        if (cp > 127U) {
            const std::uint8_t illegal_index = static_cast<std::uint8_t>((cp >> 8) & 0x07U);
            if (illegal_index != kShortened) {
                if (illegal_index >= kIllegals.size())
                    continue;
                push7(kIllegals[illegal_index]);
            }
            push7(static_cast<std::uint8_t>(cp & 0x7FU));
        } else {
            push7(static_cast<std::uint8_t>(cp));
        }
    }
    return out;
}

std::vector<std::uint8_t> huff122_decompress(const std::vector<std::uint8_t>& blob, std::size_t expected_size) {
    std::vector<std::uint8_t> out;
    out.reserve(expected_size);
    if (expected_size == 0)
        return out;
    if (blob.size() < 256ULL)
        return {};

    struct SymLen {
        std::uint16_t sym;
        std::uint8_t len;
    };
    std::vector<SymLen> syms;
    syms.reserve(256);
    for (int s = 0; s < 256; s++) {
        const std::uint8_t ln = blob[static_cast<std::size_t>(s)];
        if (ln > 0)
            syms.push_back(SymLen{static_cast<std::uint16_t>(s), ln});
    }
    if (syms.empty())
        return {};

    std::sort(syms.begin(), syms.end(), [](const SymLen& a, const SymLen& b) {
        if (a.len != b.len)
            return a.len < b.len;
        return a.sym < b.sym;
    });

    std::vector<int> left(1, -1), right(1, -1), value(1, -1);
    std::uint64_t code = 0;
    int prev_len = 0;
    for (const SymLen sl : syms) {
        const int ln = static_cast<int>(sl.len);
        if (ln < prev_len || ln > 63)
            return {};
        code <<= static_cast<std::uint64_t>(ln - prev_len);
        int node = 0;
        for (int k = ln - 1; k >= 0; k--) {
            const int bit = static_cast<int>((code >> static_cast<unsigned>(k)) & 1ULL);
            int nxt = bit ? right[node] : left[node];
            if (nxt < 0) {
                nxt = static_cast<int>(value.size());
                left.push_back(-1);
                right.push_back(-1);
                value.push_back(-1);
                if (bit)
                    right[node] = nxt;
                else
                    left[node] = nxt;
            }
            node = nxt;
        }
        value[node] = static_cast<int>(sl.sym);
        code += 1ULL;
        prev_len = ln;
    }

    int node = 0;
    for (std::size_t i = 256ULL; i < blob.size(); i++) {
        const std::uint8_t by = blob[i];
        for (int k = 7; k >= 0; k--) {
            const int bit = static_cast<int>((by >> static_cast<unsigned>(k)) & 1U);
            node = bit ? right[node] : left[node];
            if (node < 0)
                return {};
            const int sym = value[node];
            if (sym >= 0) {
                out.push_back(static_cast<std::uint8_t>(sym));
                if (out.size() == expected_size)
                    return out;
                node = 0;
            }
        }
    }
    return {};
}

std::vector<std::uint8_t> decode_payload(std::string_view encoded) {
    if (kPayloadCodec == kPayloadCodecBase91) {
        return base91_decode(encoded);
    }
    if (kPayloadCodec == kPayloadCodecBase122) {
        return base122_decode(encoded);
    }
    if (kPayloadCodec == kPayloadCodecHuff122) {
        return huff122_decompress(base122_decode(encoded), kModelBlobBytes);
    }
    if (kPayloadCodec == kPayloadCodecHuff91) {
        return huff122_decompress(base91_decode(encoded), kModelBlobBytes);
    }
    return {};
}
"""


def _replace_once(src: str, pattern: str, repl: str, *, desc: str, flags: int = 0) -> str:
    """内部ヘルパー: `replace_once` を実行する。

    Args:
        src (str): src の値。
        pattern (str): pattern の値。
        repl (str): repl の値。
        desc (str): desc の値。
        flags (int): flags の値。

    Returns:
        str: 計算結果。
    """
    out, n = re.subn(pattern, lambda _m: repl, src, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"failed to patch student source for {desc}: count={n}")
    return out


def _rewrite_student_source_with_payload(
    src: str,
    *,
    payload_text: str,
    codec_id: int,
    model_blob_bytes: int,
) -> str:
    """内部ヘルパー: `rewrite_student_source_with_payload` を実行する。

    Args:
        src (str): src の値。
        payload_text (str): payload_text の値。
        codec_id (int): codec_id の値。
        model_blob_bytes (int): model_blob_bytes の値。

    Returns:
        str: 計算結果。
    """
    if "#include <string_view>" not in src:
        src = _replace_once(
            src,
            r"#include <string>\n",
            "#include <string>\n#include <string_view>\n",
            desc="string_view include",
        )

    blob_decl = STUDENT_BLOB_DECL_CPP
    blob_decl = blob_decl.replace("__PAYLOAD_CODEC__", str(int(codec_id)))
    blob_decl = blob_decl.replace("__MODEL_BLOB_BYTES__", str(int(model_blob_bytes)))
    blob_decl = blob_decl.replace("__MODEL_BLOB_ENCODED_LINES__", _emit_payload_literal(payload_text))
    blob_decl = blob_decl.replace("__PAYLOAD_CODEC_BASE91__", str(PAYLOAD_CODEC_BASE91))
    blob_decl = blob_decl.replace("__PAYLOAD_CODEC_BASE122__", str(PAYLOAD_CODEC_BASE122))
    blob_decl = blob_decl.replace("__PAYLOAD_CODEC_HUFF122__", str(PAYLOAD_CODEC_HUFF122))
    blob_decl = blob_decl.replace("__PAYLOAD_CODEC_HUFF91__", str(PAYLOAD_CODEC_HUFF91))

    src = _replace_once(
        src,
        r"static const char kModelBlobBase64\[\] =\n(?:\s*\".*\"\n)*\s*;\n",
        blob_decl + "\n",
        desc="blob declaration",
        flags=re.MULTILINE,
    )
    src = _replace_once(
        src,
        r"std::vector<std::uint8_t> base64_decode\(const std::string& s\) \{.*?\n\}\n\n",
        STUDENT_PAYLOAD_DECODE_CPP + "\n\n",
        desc="payload decoder function",
        flags=re.DOTALL,
    )
    src = _replace_once(
        src,
        r"const std::string b64\(kModelBlobBase64\);\n\s*const std::vector<std::uint8_t> bytes = base64_decode\(b64\);\n",
        (
            "const std::string_view encoded(kModelBlobEncoded, sizeof(kModelBlobEncoded) - 1ULL);\n"
            "    const std::vector<std::uint8_t> bytes = decode_payload(encoded);\n"
            "    if(bytes.size() != kModelBlobBytes) {\n"
            '        throw std::runtime_error("invalid decoded payload size");\n'
            "    }\n"
        ),
        desc="payload decode callsite",
    )
    return src


def _strip_cpp_comments(src: str) -> str:
    """内部ヘルパー: `strip_cpp_comments` を実行する。

    Args:
        src (str): src の値。

    Returns:
        str: 計算結果。
    """
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]

        # Raw string literal: R"delim(... )delim"
        if c == "R" and (i + 1) < n and src[i + 1] == '"':
            j = i + 2
            while j < n and src[j] != "(":
                j += 1
            if j >= n:
                out.append(src[i:])
                break
            delim = src[i + 2 : j]
            marker = ")" + delim + '"'
            k = src.find(marker, j + 1)
            if k < 0:
                out.append(src[i:])
                break
            k += len(marker)
            out.append(src[i:k])
            i = k
            continue

        # Normal string/char literal
        if c == '"' or c == "'":
            q = c
            out.append(c)
            i += 1
            esc = False
            while i < n:
                ch = src[i]
                out.append(ch)
                i += 1
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == q:
                    break
            continue

        if c == "/" and (i + 1) < n:
            n1 = src[i + 1]
            # Line comment
            if n1 == "/":
                i += 2
                while i < n and src[i] != "\n":
                    i += 1
                if i < n and src[i] == "\n":
                    out.append("\n")
                    i += 1
                continue
            # Block comment
            if n1 == "*":
                i += 2
                while (i + 1) < n and not (src[i] == "*" and src[i + 1] == "/"):
                    i += 1
                i = min(i + 2, n)
                continue

        out.append(c)
        i += 1

    return "".join(out)


def _compact_layout_safe(src: str) -> str:
    """内部ヘルパー: `compact_layout_safe` を実行する。

    Args:
        src (str): src の値。

    Returns:
        str: 計算結果。
    """
    aliases = [
        ("std::array", "A_"),
        ("std::vector", "V_"),
        ("std::size_t", "SZ_"),
        ("std::uint8_t", "U8_"),
        ("std::uint16_t", "U16_"),
        ("std::int16_t", "I16_"),
        ("std::uint32_t", "U32_"),
        ("std::int64_t", "I64_"),
    ]
    defs: list[str] = []
    used: set[str] = set()

    lines: list[str] = []
    prev_blank = False
    for raw in _strip_cpp_comments(src).splitlines():
        line = raw.rstrip()
        if not line:
            if not prev_blank:
                lines.append("")
            prev_blank = True
            continue
        prev_blank = False

        line_stripped = line.strip()
        if 'R"' not in line_stripped and not line_stripped.startswith('"'):
            for old, new in aliases:
                if old in line:
                    line = line.replace(old, new)
                    used.add(new)
            if not line.lstrip().startswith("#"):
                line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    macro_map = {
        "A_": "#define A_ std::array",
        "V_": "#define V_ std::vector",
        "SZ_": "#define SZ_ std::size_t",
        "U8_": "#define U8_ std::uint8_t",
        "U16_": "#define U16_ std::uint16_t",
        "I16_": "#define I16_ std::int16_t",
        "U32_": "#define U32_ std::uint32_t",
        "I64_": "#define I64_ std::int64_t",
    }
    for k in ("A_", "V_", "SZ_", "U8_", "U16_", "I16_", "U32_", "I64_"):
        if k in used:
            defs.append(macro_map[k])

    if defs:
        ins = 0
        while ins < len(lines) and lines[ins].lstrip().startswith("#include"):
            ins += 1
        lines[ins:ins] = defs

    return "\n".join(lines) + "\n"


def _apply_compact_layout(src: str) -> str:
    """内部ヘルパー: `apply_compact_layout` を実行する。

    Args:
        src (str): src の値。

    Returns:
        str: 計算結果。
    """
    return _compact_layout_safe(src)


def _bundle_cpp(entry: Path, *, include_dirs: list[Path]) -> str:
    """内部ヘルパー: `bundle_cpp` を実行する。

    Args:
        entry (Path): entry の値。
        include_dirs (list[Path]): include_dirs の値。

    Returns:
        str: 計算結果。
    """
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
    include_sys_re = re.compile(r"^\s*#\s*include\s+<([^>]+)>\s*$")
    seen_local: set[Path] = set()
    seen_sys: set[str] = set()

    def resolve_include(cur: Path, name: str) -> Path | None:
        """`include`を解決する。

        Args:
            cur (Path): cur の値。
            name (str): name の値。

        Returns:
            Path | None: 計算結果。
        """
        for d in include_dirs:
            cand = (d / name).resolve()
            if cand.is_file():
                return cand
        cand = (cur.parent / name).resolve()
        if cand.is_file():
            return cand
        return None

    def rec(path: Path) -> list[str]:
        """`rec` を実行する。

        Args:
            path (Path): 対象パス。

        Returns:
            list[str]: 計算結果。
        """
        out: list[str] = []
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines(keepends=True):
            if line.strip() == "#pragma once":
                continue
            ms = include_sys_re.match(line)
            if ms is not None:
                hdr = ms.group(1)
                if hdr in seen_sys:
                    continue
                seen_sys.add(hdr)
                out.append(f"#include <{hdr}>\n")
                continue
            m = include_re.match(line)
            if m is None:
                out.append(line)
                continue
            name = m.group(1)
            inc = resolve_include(path, name)
            if inc is None:
                out.append(line)
                continue
            if inc in seen_local:
                continue
            seen_local.add(inc)
            out.extend(rec(inc))
        return out

    return "".join(rec(entry.resolve()))


def _build_exp002_source(
    *,
    state_dict: dict[str, Any],
    board_channels: int,
    board_size: int,
    hidden_channels: int,
    num_blocks: int,
    action_dim: int,
    feature_id: str,
    pf_enabled: bool,
    payload_encoding: str,
    tta_mode: int,
    tta_k: int,
    tta_auto_off_ms: int,
    repo_root: Path,
) -> str:
    """内部ヘルパー: `build_exp002_source` を実行する。

    Args:
        state_dict (dict[str, Any]): state_dict の値。
        board_channels (int): board_channels の値。
        board_size (int): board_size の値。
        hidden_channels (int): hidden_channels の値。
        num_blocks (int): num_blocks の値。
        action_dim (int): action_dim の値。
        feature_id (str): feature_id の値。
        pf_enabled (bool): 有効化フラグ。
        payload_encoding (str): payload_encoding の値。
        tta_mode (int): tta_mode の値。
        tta_k (int): tta_k の値。
        tta_auto_off_ms (int): tta_auto_off_ms の値。
        repo_root (Path): repo_root の値。

    Returns:
        str: 計算結果。
    """
    tensors = _collect_exp002_resnet_policy_tensors(state_dict, blocks=int(num_blocks))
    raw_blob, counts = _pack_f16_blob(tensors)
    payload_text, codec_id = encode_model_payload(raw_blob, encoding=payload_encoding)

    src = TEMPLATE_EXP002_CPP
    repl = {
        "__BOARD_CHANNELS__": str(int(board_channels)),
        "__BOARD_SIZE__": str(int(board_size)),
        "__ACTION_DIM__": str(int(action_dim)),
        "__HIDDEN_CHANNELS__": str(int(hidden_channels)),
        "__NUM_BLOCKS__": str(int(num_blocks)),
        "__PF_ENABLED__": "true" if bool(pf_enabled) else "false",
        "__TENSOR_COUNT__": str(len(counts)),
        "__TENSOR_HALF_COUNTS__": ", ".join(str(int(x)) for x in counts),
        "__PAYLOAD_CODEC__": str(int(codec_id)),
        "__MODEL_BLOB_BYTES__": str(len(raw_blob)),
        "__FEATURE_ID__": str(feature_id),
        "__MODEL_BLOB_ENCODED_LINES__": _emit_payload_literal(payload_text),
        "__PAYLOAD_CODEC_BASE91__": str(PAYLOAD_CODEC_BASE91),
        "__PAYLOAD_CODEC_BASE122__": str(PAYLOAD_CODEC_BASE122),
        "__PAYLOAD_CODEC_HUFF122__": str(PAYLOAD_CODEC_HUFF122),
        "__PAYLOAD_CODEC_HUFF91__": str(PAYLOAD_CODEC_HUFF91),
        "__TTA_MODE__": str(int(tta_mode)),
        "__TTA_K__": str(int(tta_k)),
        "__TTA_AUTO_OFF_MS__": str(int(tta_auto_off_ms)),
    }
    for k, v in repl.items():
        src = src.replace(k, v)

    tmp_dir = repo_root / "reinforce/outputs/submission/.tmp_bundle"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    entry = tmp_dir / "entry_exp002.cpp"
    entry.write_text(src, encoding="utf-8")

    include_dirs = [repo_root / "reinforce/ppo_cpp_ext/include"]
    return _bundle_cpp(entry, include_dirs=include_dirs)


def _build_teacher_source(
    *,
    state_dict: dict[str, Any],
    board_channels: int,
    board_size: int,
    width: int,
    num_blocks: int,
    action_dim: int,
    activation_id: int,
    feature_id: str,
    pf_enabled: bool,
    payload_encoding: str,
    repo_root: Path,
) -> str:
    """内部ヘルパー: `build_teacher_source` を実行する。

    Args:
        state_dict (dict[str, Any]): state_dict の値。
        board_channels (int): board_channels の値。
        board_size (int): board_size の値。
        width (int): width の値。
        num_blocks (int): num_blocks の値。
        action_dim (int): action_dim の値。
        activation_id (int): activation_id の値。
        feature_id (str): feature_id の値。
        pf_enabled (bool): 有効化フラグ。
        payload_encoding (str): payload_encoding の値。
        repo_root (Path): repo_root の値。

    Returns:
        str: 計算結果。
    """
    tensors = _collect_teacher_p0_policy_tensors(state_dict, num_blocks=int(num_blocks))
    raw_blob, counts = _pack_f16_blob(tensors)
    payload_text, codec_id = encode_model_payload(raw_blob, encoding=payload_encoding)

    src = TEMPLATE_TEACHER_P0_CPP
    repl = {
        "__BOARD_CHANNELS__": str(int(board_channels)),
        "__BOARD_SIZE__": str(int(board_size)),
        "__ACTION_DIM__": str(int(action_dim)),
        "__WIDTH__": str(int(width)),
        "__NUM_BLOCKS__": str(int(num_blocks)),
        "__ACTIVATION_ID__": str(int(activation_id)),
        "__PF_ENABLED__": "true" if bool(pf_enabled) else "false",
        "__TENSOR_COUNT__": str(len(counts)),
        "__TENSOR_HALF_COUNTS__": ", ".join(str(int(x)) for x in counts),
        "__PAYLOAD_CODEC__": str(int(codec_id)),
        "__MODEL_BLOB_BYTES__": str(len(raw_blob)),
        "__FEATURE_ID__": str(feature_id),
        "__MODEL_BLOB_ENCODED_LINES__": _emit_payload_literal(payload_text),
        "__PAYLOAD_CODEC_BASE91__": str(PAYLOAD_CODEC_BASE91),
        "__PAYLOAD_CODEC_BASE122__": str(PAYLOAD_CODEC_BASE122),
        "__PAYLOAD_CODEC_HUFF122__": str(PAYLOAD_CODEC_HUFF122),
        "__PAYLOAD_CODEC_HUFF91__": str(PAYLOAD_CODEC_HUFF91),
    }
    for k, v in repl.items():
        src = src.replace(k, v)

    tmp_dir = repo_root / "reinforce/outputs/submission/.tmp_bundle"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    entry = tmp_dir / "entry_teacher_p0.cpp"
    entry.write_text(src, encoding="utf-8")

    include_dirs = [repo_root / "reinforce/ppo_cpp_ext/include"]
    return _bundle_cpp(entry, include_dirs=include_dirs)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """内部ヘルパー: `load_checkpoint` を実行する。

    Args:
        path (Path): 対象パス。

    Returns:
        dict[str, Any]: 計算結果。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be dict")
    return payload


def _detect_feature_id(payload: dict[str, Any], default: str) -> str:
    """内部ヘルパー: `detect_feature_id` を実行する。

    Args:
        payload (dict[str, Any]): payload の値。
        default (str): default の値。

    Returns:
        str: 計算結果。
    """
    meta = payload.get("meta")
    if isinstance(meta, dict):
        fid = meta.get("feature_id")
        if isinstance(fid, str) and fid.strip():
            return fid.strip()
    return default


def _detect_pf_enabled(payload: dict[str, Any], default: bool) -> bool:
    """内部ヘルパー: `detect_pf_enabled` を実行する。

    Args:
        payload (dict[str, Any]): payload の値。
        default (bool): 有効化フラグ。

    Returns:
        bool: 計算結果。
    """
    meta = payload.get("meta")
    if isinstance(meta, dict):
        v = meta.get("pf_enabled")
        if isinstance(v, bool):
            return bool(v)
    return bool(default)


def build_parser() -> argparse.ArgumentParser:
    """`make_submit_compact` 用 CLI 引数パーサーを構築する。

    Returns:
        argparse.ArgumentParser: コマンドライン引数定義済みパーサー。
    """
    p = argparse.ArgumentParser(description="Build compact single-file main.cpp from reinforce checkpoint")
    p.add_argument(
        "--exp002-full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="delegate to vendored exp002 full-feature exporter (ppconcat presets / full quant options)",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--payload-encoding",
        type=str,
        choices=("base91", "base122", "huff122", "huff91"),
        default="huff91",
        help="payload text encoding for embedded weights",
    )
    p.add_argument("--feature-id", type=str, default="", help="override feature id (TeacherP0/Exp002ResNet)")
    p.add_argument("--pf-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tta-mode", type=int, choices=(0, 1, 2), default=0, help="Exp002ResNet TTA mode: 0=off,1=sum-prob,2=prod-prob")
    p.add_argument("--tta-k", type=int, choices=(2, 4, 8), default=8, help="Exp002ResNet TTA transforms")
    p.add_argument("--tta-auto-off-ms", type=int, default=0, help="if >0, force TTA off after elapsed milliseconds")
    p.add_argument(
        "--compact-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply exp002-style C++ token/macro compaction pass",
    )
    p.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    p.add_argument("--strict-size-limit", action=argparse.BooleanOptionalAction, default=False)
    return p


def _build_student_source(payload: dict[str, Any], args: argparse.Namespace) -> str:
    """チェックポイント内容に応じて提出用 C++ ソースを生成する。

    Args:
        payload (dict[str, Any]): 読み込み済みチェックポイント。
        args (argparse.Namespace): CLI 引数。

    Returns:
        str: 生成済み提出用 C++ ソース。
    """
    if int(args.tta_mode) != 0:
        raise ValueError("TTA is currently supported only for Exp002ResNetBoardAgent export")

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint has no valid model_config")
    model_kwargs = model_config.get("kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError("checkpoint model_config.kwargs must be dict")

    state_dict = payload.get("agent_state_dict")
    if not isinstance(state_dict, dict):
        state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no valid model state_dict")
    state_dict = _normalize_state_dict_keys(state_dict)

    board_channels = int(model_kwargs.get("board_channels", 7))
    board_size = int(model_kwargs.get("board_size", 10))
    global_dim = int(model_kwargs.get("global_dim", 49))
    width = int(model_kwargs.get("width", 48))
    num_blocks = int(model_kwargs.get("num_blocks", 2))
    global_hidden_dim = int(model_kwargs.get("global_hidden_dim", 64))
    action_dim = int(payload.get("action_dim", board_size * board_size))
    activation_id = student_export._activation_id(str(model_kwargs.get("activation", "tanh")))
    use_global_film = bool(model_kwargs.get("use_global_film", True)) and global_dim > 0
    use_global_policy_bias = bool(model_kwargs.get("use_global_policy_bias", True)) and global_dim > 0

    tensors = student_export._collect_policy_tensors(
        state_dict=state_dict,
        num_blocks=num_blocks,
        global_dim=global_dim,
        use_global_film=use_global_film,
        use_global_policy_bias=use_global_policy_bias,
    )
    raw_blob, counts = _pack_f16_blob(tensors)
    payload_text, codec_id = encode_model_payload(raw_blob, encoding=str(args.payload_encoding))

    src = student_export.build_submission_source(
        board_channels=board_channels,
        board_size=board_size,
        global_dim=global_dim,
        width=width,
        num_blocks=num_blocks,
        global_hidden_dim=global_hidden_dim,
        action_dim=action_dim,
        activation_id=activation_id,
        use_global_film=use_global_film,
        use_global_policy_bias=use_global_policy_bias,
        deterministic=True,
        use_action_mask=True,
        bayes_num_particles=128,
        bayes_seed=0,
        bayes_resample_ess_frac=0.55,
        tensor_half_counts=counts,
        model_blob_b64="AA==",
    )
    return _rewrite_student_source_with_payload(
        src,
        payload_text=payload_text,
        codec_id=codec_id,
        model_blob_bytes=len(raw_blob),
    )


def main() -> None:
    """CLI 引数を解釈し compact 形式の提出ソースを生成する。"""
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    if bool(args.exp002_full):
        out = Path(args.output).resolve()
        out_dir = out.parent
        cmd: list[str] = [
            sys.executable,
            "-m",
            "reinforce.ppo.entrypoints.make_submit_compact_exp002",
            "--ckpt",
            str(Path(args.checkpoint).resolve()),
            "--out-dir",
            str(out_dir),
        ]
        raw_args = sys.argv[1:]
        has_flag = lambda name: any(a == name or a.startswith(name + "=") for a in raw_args)
        if has_flag("--payload-encoding"):
            cmd += ["--payload-encoding", str(args.payload_encoding)]
        if has_flag("--tta-mode"):
            cmd += ["--tta-mode", str(int(args.tta_mode))]
        if has_flag("--tta-k"):
            cmd += ["--tta-k", str(int(args.tta_k))]
        if has_flag("--tta-auto-off-ms"):
            cmd += ["--tta-auto-off-ms", str(int(args.tta_auto_off_ms))]
        cmd += list(passthrough)
        print(f"[make_submit_compact] delegate exp002_full: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        legacy_main = out_dir / "Main.cpp"
        if not legacy_main.is_file():
            raise FileNotFoundError(f"exp002 full exporter did not produce Main.cpp: {legacy_main}")
        if legacy_main.resolve() != out:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_main, out)
            print(f"[make_submit_compact] copied: {legacy_main} -> {out}")
        return

    if passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")

    ckpt = Path(args.checkpoint)
    payload = _load_checkpoint(ckpt)

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint has no valid model_config")
    model_type = str(model_config.get("type", "")).strip()
    model_kwargs = model_config.get("kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError("checkpoint model_config.kwargs must be dict")

    out_src: str
    if model_type == "StudentMBoardAgent":
        out_src = _build_student_source(payload, args)
    elif model_type == "TeacherP0V1BoardAgent":
        if int(args.tta_mode) != 0:
            raise ValueError("TTA is currently supported only for Exp002ResNetBoardAgent export")

        state_dict = payload.get("agent_state_dict")
        if not isinstance(state_dict, dict):
            state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no valid model_state_dict")
        state_dict = _normalize_state_dict_keys(state_dict)

        board_channels = int(model_kwargs.get("board_channels", 88))
        board_size = int(model_kwargs.get("board_size", 10))
        width = int(model_kwargs.get("width", 64))
        num_blocks = int(model_kwargs.get("num_blocks", 4))
        action_dim = int(payload.get("action_dim", board_size * board_size))
        global_dim = int(model_kwargs.get("global_dim", 0))
        if global_dim != 0:
            raise ValueError("TeacherP0V1BoardAgent compact export currently supports only global_dim=0")
        use_global_film = bool(model_kwargs.get("use_global_film", False))
        if use_global_film:
            raise ValueError("TeacherP0V1BoardAgent compact export currently supports use_global_film=False")
        use_global_policy_bias = bool(model_kwargs.get("use_global_policy_bias", False))
        if use_global_policy_bias:
            raise ValueError(
                "TeacherP0V1BoardAgent compact export currently supports use_global_policy_bias=False"
            )

        activation_id = student_export._activation_id(str(model_kwargs.get("activation", "tanh")))
        feature_id = str(args.feature_id).strip() or _detect_feature_id(payload, default="teacher_p0_v1_88ch")
        pf_enabled = _detect_pf_enabled(payload, default=True) if args.pf_enabled is None else bool(args.pf_enabled)

        repo_root = Path(__file__).resolve().parents[3]
        out_src = _build_teacher_source(
            state_dict=state_dict,
            board_channels=board_channels,
            board_size=board_size,
            width=width,
            num_blocks=num_blocks,
            action_dim=action_dim,
            activation_id=activation_id,
            feature_id=feature_id,
            pf_enabled=pf_enabled,
            payload_encoding=str(args.payload_encoding),
            repo_root=repo_root,
        )
    elif model_type == "Exp002ResNetBoardAgent":
        state_dict = payload.get("agent_state_dict")
        if not isinstance(state_dict, dict):
            state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no valid model_state_dict")
        state_dict = _normalize_state_dict_keys(state_dict)

        board_channels = int(model_kwargs.get("board_channels", 88))
        board_size = int(model_kwargs.get("board_size", 10))
        hidden_channels = int(model_kwargs.get("hidden_channels", 64))
        num_blocks = int(model_kwargs.get("blocks", 6))
        action_dim = int(payload.get("action_dim", board_size * board_size))

        feature_id = str(args.feature_id).strip() or _detect_feature_id(payload, default="teacher_p0_v1_88ch")
        pf_enabled = _detect_pf_enabled(payload, default=True) if args.pf_enabled is None else bool(args.pf_enabled)

        repo_root = Path(__file__).resolve().parents[3]
        out_src = _build_exp002_source(
            state_dict=state_dict,
            board_channels=board_channels,
            board_size=board_size,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            action_dim=action_dim,
            feature_id=feature_id,
            pf_enabled=pf_enabled,
            payload_encoding=str(args.payload_encoding),
            tta_mode=int(args.tta_mode),
            tta_k=int(args.tta_k),
            tta_auto_off_ms=int(args.tta_auto_off_ms),
            repo_root=repo_root,
        )
    else:
        raise ValueError(
            f"unsupported model type for make_submit_compact: {model_type!r}; "
            "expected StudentMBoardAgent, TeacherP0V1BoardAgent, or Exp002ResNetBoardAgent"
        )

    if bool(args.compact_layout):
        try:
            before_bytes = len(out_src.encode("utf-8"))
            out_src = _apply_compact_layout(out_src)
            after_bytes = len(out_src.encode("utf-8"))
            print(f"[make_submit_compact] compact_layout: {before_bytes} -> {after_bytes} bytes")
        except Exception as exc:
            raise RuntimeError(f"failed to apply compact layout: {exc}") from exc

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(out_src, encoding="utf-8")

    source_bytes = out.stat().st_size
    print(f"[make_submit_compact] checkpoint: {ckpt}")
    print(f"[make_submit_compact] model_type: {model_type}")
    print(f"[make_submit_compact] output: {out}")
    print(f"[make_submit_compact] source bytes: {source_bytes}")

    if source_bytes > int(args.max_source_bytes):
        msg = (
            f"[make_submit_compact] WARNING: source exceeds max-source-bytes "
            f"({source_bytes} > {int(args.max_source_bytes)})"
        )
        if bool(args.strict_size_limit):
            raise RuntimeError(msg)
        print(msg)


if __name__ == "__main__":
    main()
