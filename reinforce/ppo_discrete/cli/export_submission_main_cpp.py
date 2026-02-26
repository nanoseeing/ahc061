from __future__ import annotations

import argparse
import base64
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MAX_SOURCE_BYTES = 512 * 1024


TEMPLATE_MAIN_CPP = r"""// GENERATED FILE. DO NOT EDIT.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBoardChannels = __BOARD_CHANNELS__;
constexpr int kBoardSize = __BOARD_SIZE__;
constexpr int kGlobalDim = __GLOBAL_DIM__;
constexpr int kObsDim = kBoardChannels * kBoardSize * kBoardSize + kGlobalDim;
constexpr int kActionDim = __ACTION_DIM__;
constexpr int kWidth = __WIDTH__;
constexpr int kNumBlocks = __NUM_BLOCKS__;
constexpr int kGlobalHiddenDim = __GLOBAL_HIDDEN_DIM__;
constexpr int kActivationId = __ACTIVATION_ID__;  // 0:tanh 1:relu 2:silu
constexpr bool kUseGlobalFilm = __USE_GLOBAL_FILM__;
constexpr bool kUseGlobalPolicyBias = __USE_GLOBAL_POLICY_BIAS__;
constexpr bool kDeterministic = __DETERMINISTIC__;
constexpr bool kUseActionMask = __USE_ACTION_MASK__;
constexpr int kBayesNumParticles = __BAYES_NUM_PARTICLES__;
constexpr std::uint64_t kBayesSeed = static_cast<std::uint64_t>(__BAYES_SEED__);
constexpr double kBayesResampleEssFrac = __BAYES_RESAMPLE_ESS_FRAC__;

constexpr int kTensorCount = __TENSOR_COUNT__;
static const int kTensorHalfCounts[kTensorCount] = {__TENSOR_HALF_COUNTS__};

// W_LO/HI and E_LO/HI must match training-side bayes estimator conventions.
constexpr double W_LO = 0.3;
constexpr double W_HI = 1.0;
constexpr double E_LO = 0.1;
constexpr double E_HI = 0.5;
constexpr double R_LO = W_LO / W_HI;
constexpr double R_HI = W_HI / W_LO;
constexpr double LIKELIHOOD_FLOOR = 1e-12;

constexpr int DX[4] = {-1, 1, 0, 0};
constexpr int DY[4] = {0, 0, -1, 1};

static const char kModelBlobBase64[] =
__MODEL_BLOB_B64_LINES__
    ;

inline double clip(double x, double lo, double hi) {
    if(x < lo) return lo;
    if(x > hi) return hi;
    return x;
}

inline float clipf(float x, float lo, float hi) {
    if(x < lo) return lo;
    if(x > hi) return hi;
    return x;
}

// IEEE754 binary16 -> binary32
inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = static_cast<std::uint32_t>((h >> 10) & 0x1Fu);
    std::uint32_t mant = static_cast<std::uint32_t>(h & 0x03FFu);
    std::uint32_t f = 0;
    if(exp == 0) {
        if(mant == 0) {
            f = sign;
        } else {
            int e = -1;
            do {
                e++;
                mant <<= 1;
            } while((mant & 0x0400u) == 0u);
            mant &= 0x03FFu;
            const std::uint32_t exp32 = static_cast<std::uint32_t>(127 - 15 - e);
            f = sign | (exp32 << 23) | (mant << 13);
        }
    } else if(exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        const std::uint32_t exp32 = static_cast<std::uint32_t>(exp + (127 - 15));
        f = sign | (exp32 << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

std::vector<std::uint8_t> base64_decode(const std::string& s) {
    static std::array<int, 256> T = []() {
        std::array<int, 256> t{};
        t.fill(-1);
        for(int i = 0; i < 26; i++) {
            t[static_cast<std::size_t>('A' + i)] = i;
            t[static_cast<std::size_t>('a' + i)] = 26 + i;
        }
        for(int i = 0; i < 10; i++) {
            t[static_cast<std::size_t>('0' + i)] = 52 + i;
        }
        t[static_cast<std::size_t>('+')] = 62;
        t[static_cast<std::size_t>('/')] = 63;
        return t;
    }();

    std::vector<std::uint8_t> out;
    out.reserve((s.size() * 3) / 4 + 8);

    std::uint32_t val = 0;
    int valb = -8;
    for(char c : s) {
        if(c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int d = T[static_cast<std::uint8_t>(c)];
        if(d < 0) continue;
        val = (val << 6) | static_cast<std::uint32_t>(d);
        valb += 6;
        if(valb >= 0) {
            out.push_back(static_cast<std::uint8_t>((val >> valb) & 0xFFu));
            valb -= 8;
        }
    }
    return out;
}

struct PolicyWeights {
    std::vector<float> stem_w;
    std::vector<float> stem_b;

    struct Block {
        std::vector<float> conv1_w;
        std::vector<float> conv1_b;
        std::vector<float> conv2_w;
        std::vector<float> conv2_b;
    };
    std::vector<Block> blocks;

    std::vector<float> global1_w;
    std::vector<float> global1_b;
    std::vector<float> global2_w;
    std::vector<float> global2_b;

    std::vector<float> film_w;
    std::vector<float> film_b;

    std::vector<float> policy_conv_w;
    std::vector<float> policy_conv_b;

    std::vector<float> policy_global_bias_w;
    std::vector<float> policy_global_bias_b;
};

PolicyWeights load_policy_weights_from_blob() {
    const std::string b64(kModelBlobBase64);
    const std::vector<std::uint8_t> bytes = base64_decode(b64);
    if(bytes.size() % 2 != 0) {
        throw std::runtime_error("invalid model blob bytes: odd length");
    }

    std::vector<std::uint16_t> halves(bytes.size() / 2);
    for(std::size_t i = 0; i < halves.size(); i++) {
        halves[i] = static_cast<std::uint16_t>(bytes[2 * i]) |
                    static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[2 * i + 1]) << 8);
    }

    int expect_total = 0;
    for(int i = 0; i < kTensorCount; i++) expect_total += kTensorHalfCounts[i];
    if(expect_total != static_cast<int>(halves.size())) {
        throw std::runtime_error("model blob size mismatch");
    }

    std::size_t off = 0;
    int idx = 0;
    auto take = [&](int n) -> std::vector<float> {
        std::vector<float> v(static_cast<std::size_t>(n));
        for(int i = 0; i < n; i++) {
            v[static_cast<std::size_t>(i)] = half_to_float(halves[off + static_cast<std::size_t>(i)]);
        }
        off += static_cast<std::size_t>(n);
        return v;
    };

    PolicyWeights w;
    w.stem_w = take(kTensorHalfCounts[idx++]);
    w.stem_b = take(kTensorHalfCounts[idx++]);

    w.blocks.resize(static_cast<std::size_t>(kNumBlocks));
    for(int bi = 0; bi < kNumBlocks; bi++) {
        auto& b = w.blocks[static_cast<std::size_t>(bi)];
        b.conv1_w = take(kTensorHalfCounts[idx++]);
        b.conv1_b = take(kTensorHalfCounts[idx++]);
        b.conv2_w = take(kTensorHalfCounts[idx++]);
        b.conv2_b = take(kTensorHalfCounts[idx++]);
    }

    if(kGlobalDim > 0) {
        w.global1_w = take(kTensorHalfCounts[idx++]);
        w.global1_b = take(kTensorHalfCounts[idx++]);
        w.global2_w = take(kTensorHalfCounts[idx++]);
        w.global2_b = take(kTensorHalfCounts[idx++]);
    }
    if(kUseGlobalFilm) {
        w.film_w = take(kTensorHalfCounts[idx++]);
        w.film_b = take(kTensorHalfCounts[idx++]);
    }

    w.policy_conv_w = take(kTensorHalfCounts[idx++]);
    w.policy_conv_b = take(kTensorHalfCounts[idx++]);

    if(kUseGlobalPolicyBias) {
        w.policy_global_bias_w = take(kTensorHalfCounts[idx++]);
        w.policy_global_bias_b = take(kTensorHalfCounts[idx++]);
    }

    if(idx != kTensorCount || off != halves.size()) {
        throw std::runtime_error("tensor decode index mismatch");
    }
    return w;
}

inline float activation(float x) {
    if(kActivationId == 0) {
        return std::tanh(x);
    }
    if(kActivationId == 1) {
        return (x > 0.0f) ? x : 0.0f;
    }
    // silu
    return x / (1.0f + std::exp(-x));
}

void linear_forward(
    const std::vector<float>& w,
    const std::vector<float>& b,
    const std::vector<float>& x,
    int in_dim,
    int out_dim,
    std::vector<float>& y) {
    y.assign(static_cast<std::size_t>(out_dim), 0.0f);
    for(int o = 0; o < out_dim; o++) {
        float s = b[static_cast<std::size_t>(o)];
        const std::size_t base = static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim);
        for(int i = 0; i < in_dim; i++) {
            s += w[base + static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
        }
        y[static_cast<std::size_t>(o)] = s;
    }
}

void conv3x3_forward(
    const std::vector<float>& w,
    const std::vector<float>& b,
    const std::vector<float>& x,
    int in_ch,
    int out_ch,
    int n,
    std::vector<float>& y) {
    y.assign(static_cast<std::size_t>(out_ch * n * n), 0.0f);
    for(int oc = 0; oc < out_ch; oc++) {
        for(int r = 0; r < n; r++) {
            for(int c = 0; c < n; c++) {
                float s = b[static_cast<std::size_t>(oc)];
                for(int ic = 0; ic < in_ch; ic++) {
                    for(int kr = 0; kr < 3; kr++) {
                        const int rr = r + kr - 1;
                        if(rr < 0 || rr >= n) continue;
                        for(int kc = 0; kc < 3; kc++) {
                            const int cc = c + kc - 1;
                            if(cc < 0 || cc >= n) continue;
                            const std::size_t wi =
                                (((static_cast<std::size_t>(oc) * static_cast<std::size_t>(in_ch) + static_cast<std::size_t>(ic)) * 3u +
                                  static_cast<std::size_t>(kr)) *
                                     3u +
                                 static_cast<std::size_t>(kc));
                            const std::size_t xi = (static_cast<std::size_t>(ic) * static_cast<std::size_t>(n) + static_cast<std::size_t>(rr)) *
                                                       static_cast<std::size_t>(n) +
                                                   static_cast<std::size_t>(cc);
                            s += w[wi] * x[xi];
                        }
                    }
                }
                const std::size_t yi =
                    (static_cast<std::size_t>(oc) * static_cast<std::size_t>(n) + static_cast<std::size_t>(r)) * static_cast<std::size_t>(n) +
                    static_cast<std::size_t>(c);
                y[yi] = s;
            }
        }
    }
}

class StudentMPolicy {
  public:
    explicit StudentMPolicy(PolicyWeights&& w) : w_(std::move(w)) {
        if(kBoardSize != 10 || kBoardChannels != 7 || kGlobalDim != 49 || kActionDim != 100) {
            throw std::runtime_error("unsupported model layout for this exporter");
        }
    }

    void forward_logits(const std::vector<float>& obs, std::vector<float>& logits) {
        if(static_cast<int>(obs.size()) != kObsDim) {
            throw std::runtime_error("obs size mismatch");
        }
        const int n = kBoardSize;
        const int board_dim = kBoardChannels * n * n;

        const float* board_ptr = obs.data();
        const float* global_ptr = obs.data() + board_dim;

        std::vector<float> h;
        conv3x3_forward(w_.stem_w, w_.stem_b, std::vector<float>(board_ptr, board_ptr + board_dim), kBoardChannels, kWidth, n, h);
        for(float& v : h) v = activation(v);

        std::vector<float> z;
        std::vector<float> t;
        for(int bi = 0; bi < kNumBlocks; bi++) {
            const auto& b = w_.blocks[static_cast<std::size_t>(bi)];
            conv3x3_forward(b.conv1_w, b.conv1_b, h, kWidth, kWidth, n, z);
            for(float& v : z) v = activation(v);
            conv3x3_forward(b.conv2_w, b.conv2_b, z, kWidth, kWidth, n, t);
            for(std::size_t i = 0; i < h.size(); i++) {
                h[i] = activation(h[i] + t[i]);
            }
        }

        std::vector<float> g_emb;
        if(kGlobalDim > 0) {
            std::vector<float> g0(global_ptr, global_ptr + kGlobalDim);
            std::vector<float> g1;
            std::vector<float> g2;
            linear_forward(w_.global1_w, w_.global1_b, g0, kGlobalDim, kGlobalHiddenDim, g1);
            for(float& v : g1) v = activation(v);
            linear_forward(w_.global2_w, w_.global2_b, g1, kGlobalHiddenDim, kGlobalHiddenDim, g2);
            for(float& v : g2) v = activation(v);
            g_emb = std::move(g2);
        }

        if(kUseGlobalFilm && !g_emb.empty()) {
            std::vector<float> film;
            linear_forward(w_.film_w, w_.film_b, g_emb, kGlobalHiddenDim, 2 * kWidth, film);
            for(int c = 0; c < kWidth; c++) {
                const float gamma = film[static_cast<std::size_t>(c)];
                const float beta = film[static_cast<std::size_t>(kWidth + c)];
                const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(n * n);
                for(int i = 0; i < n * n; i++) {
                    h[base + static_cast<std::size_t>(i)] = h[base + static_cast<std::size_t>(i)] * (1.0f + gamma) + beta;
                }
            }
        }

        logits.assign(kActionDim, w_.policy_conv_b.empty() ? 0.0f : w_.policy_conv_b[0]);
        for(int i = 0; i < kActionDim; i++) {
            float s = logits[static_cast<std::size_t>(i)];
            for(int c = 0; c < kWidth; c++) {
                s += w_.policy_conv_w[static_cast<std::size_t>(c)] * h[static_cast<std::size_t>(c * kActionDim + i)];
            }
            logits[static_cast<std::size_t>(i)] = s;
        }

        if(kUseGlobalPolicyBias && !g_emb.empty()) {
            std::vector<float> bias;
            linear_forward(w_.policy_global_bias_w, w_.policy_global_bias_b, g_emb, kGlobalHiddenDim, kActionDim, bias);
            for(int i = 0; i < kActionDim; i++) {
                logits[static_cast<std::size_t>(i)] += bias[static_cast<std::size_t>(i)];
            }
        }
    }

  private:
    PolicyWeights w_;
};

struct Particle {
    double wa;
    double wb;
    double wc;
    double wd;
    double eps;
    double w;
};

class OpponentBayesEstimator {
  public:
    OpponentBayesEstimator(int n, int m, int u, int num_particles = 128, double resample_ess_frac = 0.55, std::uint64_t seed = 0)
        : n_(n), m_(m), u_(u), num_particles_(std::max(8, num_particles)), resample_ess_frac_(clip(resample_ess_frac, 0.05, 0.95)), rng_(seed) {
        if(n_ <= 0 || m_ <= 0) {
            throw std::runtime_error("invalid bayes n/m");
        }
        particles_.assign(static_cast<std::size_t>(m_), {});
        for(int p = 1; p < m_; p++) {
            particles_[static_cast<std::size_t>(p)] = sample_prior_particles(num_particles_);
        }
    }

    void update(
        const std::vector<int>& values,
        const std::vector<int>& owner_before,
        const std::vector<int>& level_before,
        const std::vector<std::pair<int, int>>& observed_selected,
        const std::vector<std::vector<std::pair<int, int>>>& observed_candidates) {
        for(int p = 1; p < m_; p++) {
            const std::vector<std::pair<int, int>>& cands = observed_candidates[static_cast<std::size_t>(p)];
            const auto obs = observed_selected[static_cast<std::size_t>(p)];
            auto& ps = particles_[static_cast<std::size_t>(p)];
            for(auto& pt : ps) {
                const double like = likelihood_observed_move(values, owner_before, level_before, p, cands, obs, pt);
                pt.w *= like;
            }
            normalize_weights(ps);
            const double ess = effective_sample_size(ps);
            if(ess < resample_ess_frac_ * static_cast<double>(ps.size())) {
                ps = systematic_resample(ps);
                for(auto& pt : ps) jitter(pt);
                normalize_weights(ps);
            }
        }
    }

    std::array<double, 4> posterior_mean_ratio(int player) const {
        const auto& ps = particles_[static_cast<std::size_t>(player)];
        double wa = 0.0;
        double wb = 0.0;
        double wc = 0.0;
        double wd = 0.0;
        double eps = 0.0;
        for(const auto& p : ps) {
            wa += p.wa * p.w;
            wb += p.wb * p.w;
            wc += p.wc * p.w;
            wd += p.wd * p.w;
            eps += p.eps * p.w;
        }
        if(!std::isfinite(wa) || std::abs(wa) < 1e-12) wa = 1.0;
        const double rb = clip(wb / wa, R_LO, R_HI);
        const double rc = clip(wc / wa, R_LO, R_HI);
        const double rd = clip(wd / wa, R_LO, R_HI);
        const double e = clip(eps, E_LO, E_HI);
        return {rb, rc, rd, e};
    }

    std::vector<float> posterior_feature_vector(int max_enemies = 7, bool normalize = true) const {
        std::vector<float> out(static_cast<std::size_t>(max_enemies * 4), 0.0f);
        auto norm_ratio = [](double v) -> double { return (v - R_LO) / std::max(1e-12, (R_HI - R_LO)); };
        auto norm_eps = [](double v) -> double { return (v - E_LO) / std::max(1e-12, (E_HI - E_LO)); };
        for(int ei = 0; ei < max_enemies; ei++) {
            const int p = ei + 1;
            const int off = ei * 4;
            if(p < m_) {
                const auto r = posterior_mean_ratio(p);
                if(normalize) {
                    out[static_cast<std::size_t>(off + 0)] = static_cast<float>(clip(norm_ratio(r[0]), 0.0, 1.0));
                    out[static_cast<std::size_t>(off + 1)] = static_cast<float>(clip(norm_ratio(r[1]), 0.0, 1.0));
                    out[static_cast<std::size_t>(off + 2)] = static_cast<float>(clip(norm_ratio(r[2]), 0.0, 1.0));
                    out[static_cast<std::size_t>(off + 3)] = static_cast<float>(clip(norm_eps(r[3]), 0.0, 1.0));
                } else {
                    out[static_cast<std::size_t>(off + 0)] = static_cast<float>(r[0]);
                    out[static_cast<std::size_t>(off + 1)] = static_cast<float>(r[1]);
                    out[static_cast<std::size_t>(off + 2)] = static_cast<float>(r[2]);
                    out[static_cast<std::size_t>(off + 3)] = static_cast<float>(r[3]);
                }
            }
        }
        return out;
    }

  private:
    int n_;
    int m_;
    int u_;
    int num_particles_;
    double resample_ess_frac_;
    std::mt19937_64 rng_;
    std::vector<std::vector<Particle>> particles_;

    inline int idx(int x, int y) const { return x * n_ + y; }

    double uniform(double lo, double hi) {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(rng_);
    }
    double uniform01() {
        static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng_);
    }
    double gauss(double sigma) {
        if(sigma <= 0.0) return 0.0;
        std::normal_distribution<double> dist(0.0, sigma);
        return dist(rng_);
    }

    std::vector<Particle> sample_prior_particles(int k) {
        std::vector<Particle> out;
        out.reserve(static_cast<std::size_t>(k));
        for(int i = 0; i < k; i++) {
            out.push_back(
                Particle{
                    uniform(W_LO, W_HI),
                    uniform(W_LO, W_HI),
                    uniform(W_LO, W_HI),
                    uniform(W_LO, W_HI),
                    uniform(E_LO, E_HI),
                    1.0 / static_cast<double>(k),
                });
        }
        return out;
    }

    static void normalize_weights(std::vector<Particle>& ps) {
        double s = 0.0;
        for(const auto& p : ps) s += p.w;
        if(s <= 0.0) {
            const double uni = 1.0 / std::max(1.0, static_cast<double>(ps.size()));
            for(auto& p : ps) p.w = uni;
            return;
        }
        const double inv = 1.0 / s;
        for(auto& p : ps) p.w *= inv;
    }

    static double effective_sample_size(const std::vector<Particle>& ps) {
        double s2 = 0.0;
        for(const auto& p : ps) s2 += p.w * p.w;
        if(s2 <= 1e-18) return 0.0;
        return 1.0 / s2;
    }

    std::vector<Particle> systematic_resample(const std::vector<Particle>& ps) {
        const int n = static_cast<int>(ps.size());
        std::vector<double> cdf(static_cast<std::size_t>(n), 0.0);
        double acc = 0.0;
        for(int i = 0; i < n; i++) {
            acc += ps[static_cast<std::size_t>(i)].w;
            cdf[static_cast<std::size_t>(i)] = acc;
        }
        std::vector<Particle> out;
        out.reserve(static_cast<std::size_t>(n));
        const double u0 = uniform01() / std::max(1, n);
        int j = 0;
        for(int i = 0; i < n; i++) {
            const double u = u0 + static_cast<double>(i) / std::max(1, n);
            while(j < n - 1 && cdf[static_cast<std::size_t>(j)] < u) j++;
            Particle src = ps[static_cast<std::size_t>(j)];
            src.w = 1.0 / std::max(1.0, static_cast<double>(n));
            out.push_back(src);
        }
        return out;
    }

    void jitter(Particle& p) {
        p.wa = clip(p.wa * std::exp(gauss(0.04)), W_LO, W_HI);
        p.wb = clip(p.wb * std::exp(gauss(0.04)), W_LO, W_HI);
        p.wc = clip(p.wc * std::exp(gauss(0.04)), W_LO, W_HI);
        p.wd = clip(p.wd * std::exp(gauss(0.04)), W_LO, W_HI);
        p.eps = clip(p.eps + gauss(0.01), E_LO, E_HI);
    }

    int cell_category(const std::vector<int>& owner, const std::vector<int>& level, int player, int x, int y) const {
        const int o = owner[static_cast<std::size_t>(idx(x, y))];
        if(o == -1) return 0;
        if(o == player) return (level[static_cast<std::size_t>(idx(x, y))] >= u_) ? 2 : 1;
        return (level[static_cast<std::size_t>(idx(x, y))] == 1) ? 3 : 4;
    }

    std::vector<std::pair<int, int>> get_candidates(
        const std::vector<int>& owner,
        const std::vector<int>& px,
        const std::vector<int>& py,
        int player) const {
        const int sx = px[static_cast<std::size_t>(player)];
        const int sy = py[static_cast<std::size_t>(player)];
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(n_ * n_), 0);
        std::vector<std::pair<int, int>> q;
        q.reserve(static_cast<std::size_t>(n_ * n_));
        std::vector<std::pair<int, int>> reachable;
        reachable.reserve(static_cast<std::size_t>(n_ * n_));
        q.push_back({sx, sy});
        seen[static_cast<std::size_t>(idx(sx, sy))] = 1;
        std::size_t head = 0;
        while(head < q.size()) {
            const auto [x, y] = q[head++];
            bool ok = true;
            for(int p = 0; p < m_; p++) {
                if(p == player) continue;
                if(px[static_cast<std::size_t>(p)] == x && py[static_cast<std::size_t>(p)] == y) {
                    ok = false;
                    break;
                }
            }
            if(ok) reachable.push_back({x, y});
            if(owner[static_cast<std::size_t>(idx(x, y))] == player) {
                for(int d = 0; d < 4; d++) {
                    const int nx = x + DX[d];
                    const int ny = y + DY[d];
                    if(nx < 0 || nx >= n_ || ny < 0 || ny >= n_) continue;
                    const int nid = idx(nx, ny);
                    if(seen[static_cast<std::size_t>(nid)] != 0) continue;
                    seen[static_cast<std::size_t>(nid)] = 1;
                    q.push_back({nx, ny});
                }
            }
        }
        return reachable;
    }

    double ai_eval(
        const std::vector<int>& values,
        const std::vector<int>& owner,
        const std::vector<int>& level,
        int player,
        const Particle& theta,
        int x,
        int y) const {
        const int cat = cell_category(owner, level, player, x, y);
        const double val = static_cast<double>(values[static_cast<std::size_t>(idx(x, y))]);
        if(cat == 0) return val * theta.wa;
        if(cat == 1) return val * theta.wb;
        if(cat == 2) return 0.0;
        if(cat == 3) return val * theta.wc;
        return val * theta.wd;
    }

    double likelihood_observed_move(
        const std::vector<int>& values,
        const std::vector<int>& owner,
        const std::vector<int>& level,
        int player,
        const std::vector<std::pair<int, int>>& cands,
        const std::pair<int, int>& observed,
        const Particle& theta) const {
        if(cands.empty()) return LIKELIHOOD_FLOOR;
        int obs_idx = -1;
        std::vector<double> scores;
        scores.reserve(cands.size());
        for(std::size_t i = 0; i < cands.size(); i++) {
            const auto [x, y] = cands[i];
            if(x == observed.first && y == observed.second) obs_idx = static_cast<int>(i);
            scores.push_back(ai_eval(values, owner, level, player, theta, x, y));
        }
        if(obs_idx < 0) return LIKELIHOOD_FLOOR;
        const int b = static_cast<int>(cands.size());
        const double eps = clip(theta.eps, 1e-6, 1.0 - 1e-6);
        const double p_rand = eps / static_cast<double>(b);
        const double mx = *std::max_element(scores.begin(), scores.end());
        const double tol = 1e-9 * std::max(std::abs(mx), 1.0);
        int best_count = 0;
        bool in_best = false;
        for(std::size_t i = 0; i < scores.size(); i++) {
            if(scores[i] >= mx - tol) {
                best_count++;
                if(static_cast<int>(i) == obs_idx) in_best = true;
            }
        }
        if(best_count <= 0) best_count = 1;
        const double p_greedy = in_best ? ((1.0 - eps) / static_cast<double>(best_count)) : 0.0;
        return std::max(LIKELIHOOD_FLOOR, p_rand + p_greedy);
    }
};

struct RuntimeState {
    std::vector<int> owner;  // n*n
    std::vector<int> level;  // n*n
    std::vector<int> px;
    std::vector<int> py;
};

inline int idx2(int n, int x, int y) { return x * n + y; }

std::vector<std::pair<int, int>> get_candidates_for_player(int n, int m, const RuntimeState& st, int player) {
    const int sx = st.px[static_cast<std::size_t>(player)];
    const int sy = st.py[static_cast<std::size_t>(player)];
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(n * n), 0);
    std::vector<std::pair<int, int>> q;
    q.reserve(static_cast<std::size_t>(n * n));
    std::vector<std::pair<int, int>> reachable;
    reachable.reserve(static_cast<std::size_t>(n * n));
    q.push_back({sx, sy});
    seen[static_cast<std::size_t>(idx2(n, sx, sy))] = 1;
    std::size_t head = 0;
    while(head < q.size()) {
        const auto [x, y] = q[head++];
        bool ok = true;
        for(int p = 0; p < m; p++) {
            if(p == player) continue;
            if(st.px[static_cast<std::size_t>(p)] == x && st.py[static_cast<std::size_t>(p)] == y) {
                ok = false;
                break;
            }
        }
        if(ok) reachable.push_back({x, y});
        if(st.owner[static_cast<std::size_t>(idx2(n, x, y))] == player) {
            for(int d = 0; d < 4; d++) {
                const int nx = x + DX[d];
                const int ny = y + DY[d];
                if(nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                const int ni = idx2(n, nx, ny);
                if(seen[static_cast<std::size_t>(ni)] != 0) continue;
                seen[static_cast<std::size_t>(ni)] = 1;
                q.push_back({nx, ny});
            }
        }
    }
    return reachable;
}

std::vector<long long> score_all_players(int n, int m, const std::vector<int>& values, const RuntimeState& st) {
    std::vector<long long> s(static_cast<std::size_t>(m), 0LL);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            const int id = idx2(n, i, j);
            const int o = st.owner[static_cast<std::size_t>(id)];
            if(o >= 0) {
                s[static_cast<std::size_t>(o)] += static_cast<long long>(values[static_cast<std::size_t>(id)]) *
                                                   static_cast<long long>(st.level[static_cast<std::size_t>(id)]);
            }
        }
    }
    return s;
}

void encode_obs(
    int n,
    int m,
    int t,
    int u,
    int turn,
    const std::vector<int>& values,
    const RuntimeState& st,
    const std::vector<long long>& scores,
    const std::vector<float>& bayes_vec,
    long long value_sum,
    std::vector<float>& out_obs) {
    out_obs.assign(static_cast<std::size_t>(kObsDim), 0.0f);
    if(n != 10) {
        throw std::runtime_error("expected n=10 for Student-M exporter");
    }

    const int plane = 100;
    auto at = [&](int ch, int x, int y) -> std::size_t {
        return static_cast<std::size_t>(ch * plane + x * 10 + y);
    };

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            const int id = idx2(n, i, j);
            out_obs[at(0, i, j)] = static_cast<float>(values[static_cast<std::size_t>(id)] / 1000.0);
            const int o = st.owner[static_cast<std::size_t>(id)];
            if(o == 0) {
                out_obs[at(1, i, j)] = 1.0f;
            } else if(o == -1) {
                out_obs[at(3, i, j)] = 1.0f;
            } else {
                out_obs[at(2, i, j)] = 1.0f;
            }
            out_obs[at(4, i, j)] = static_cast<float>(st.level[static_cast<std::size_t>(id)] / static_cast<double>(std::max(1, u)));
        }
    }
    out_obs[at(5, st.px[0], st.py[0])] = 1.0f;
    for(int p = 1; p < m; p++) {
        out_obs[at(6, st.px[static_cast<std::size_t>(p)], st.py[static_cast<std::size_t>(p)])] = 1.0f;
    }

    const int g_off = kBoardChannels * plane;
    auto setg = [&](int k, float v) {
        if(k < 0 || k >= kGlobalDim) return;
        out_obs[static_cast<std::size_t>(g_off + k)] = v;
    };
    setg(0, static_cast<float>(turn / static_cast<double>(std::max(1, t))));
    setg(1, static_cast<float>((m - 2) / 6.0));
    setg(2, static_cast<float>((u - 1) / 4.0));
    setg(3, static_cast<float>(st.px[0] / static_cast<double>(std::max(1, n - 1))));
    setg(4, static_cast<float>(st.py[0] / static_cast<double>(std::max(1, n - 1))));

    int poff = 5;
    for(int ei = 0; ei < 7; ei++) {
        if(ei + 1 < m) {
            const int p = ei + 1;
            setg(poff + 2 * ei, static_cast<float>(st.px[static_cast<std::size_t>(p)] / static_cast<double>(std::max(1, n - 1))));
            setg(poff + 2 * ei + 1, static_cast<float>(st.py[static_cast<std::size_t>(p)] / static_cast<double>(std::max(1, n - 1))));
        }
    }

    const double s_cap = static_cast<double>(std::max(1LL, static_cast<long long>(u) * value_sum));
    const double s0 = static_cast<double>(scores.empty() ? 0LL : scores[0]);
    double sa = 1.0;
    if(m > 1) {
        long long mx = 0;
        for(int p = 1; p < m; p++) mx = std::max(mx, scores[static_cast<std::size_t>(p)]);
        sa = static_cast<double>(mx);
    }
    setg(19, clipf(static_cast<float>(s0 / s_cap), 0.0f, 1.0f));
    setg(20, clipf(static_cast<float>(sa / s_cap), 0.0f, 1.0f));

    // bayes features (28) are appended after global base(21)
    for(int i = 0; i < 28; i++) {
        const float v = (i < static_cast<int>(bayes_vec.size())) ? bayes_vec[static_cast<std::size_t>(i)] : 0.0f;
        setg(21 + i, v);
    }
}

int argmax_masked(const std::vector<float>& logits, const std::array<std::uint8_t, 100>& mask) {
    float best = -std::numeric_limits<float>::infinity();
    int best_i = 0;
    for(int i = 0; i < 100; i++) {
        if(mask[static_cast<std::size_t>(i)] == 0) continue;
        if(logits[static_cast<std::size_t>(i)] > best) {
            best = logits[static_cast<std::size_t>(i)];
            best_i = i;
        }
    }
    return best_i;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int N = 0, M = 0, T = 0, U = 0;
    if(!(std::cin >> N >> M >> T >> U)) {
        return 0;
    }
    if(N != 10) {
        throw std::runtime_error("this exporter currently supports N=10 only");
    }

    std::vector<int> values(static_cast<std::size_t>(N * N), 0);
    long long value_sum = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            int v = 0;
            std::cin >> v;
            values[static_cast<std::size_t>(i * N + j)] = v;
            value_sum += static_cast<long long>(v);
        }
    }

    RuntimeState st;
    st.owner.assign(static_cast<std::size_t>(N * N), -1);
    st.level.assign(static_cast<std::size_t>(N * N), 0);
    st.px.assign(static_cast<std::size_t>(M), 0);
    st.py.assign(static_cast<std::size_t>(M), 0);
    for(int p = 0; p < M; p++) {
        int x = 0, y = 0;
        std::cin >> x >> y;
        st.px[static_cast<std::size_t>(p)] = x;
        st.py[static_cast<std::size_t>(p)] = y;
        st.owner[static_cast<std::size_t>(x * N + y)] = p;
        st.level[static_cast<std::size_t>(x * N + y)] = 1;
    }

    PolicyWeights pw = load_policy_weights_from_blob();
    StudentMPolicy policy(std::move(pw));
    OpponentBayesEstimator bayes(N, M, U, kBayesNumParticles, kBayesResampleEssFrac, kBayesSeed);

    std::vector<float> obs;
    std::vector<float> logits;
    std::array<std::uint8_t, 100> action_mask{};

    for(int turn = 0; turn < T; turn++) {
        const std::vector<long long> scores = score_all_players(N, M, values, st);
        std::vector<std::vector<std::pair<int, int>>> candidates_all(static_cast<std::size_t>(M));
        for(int p = 0; p < M; p++) {
            candidates_all[static_cast<std::size_t>(p)] = get_candidates_for_player(N, M, st, p);
        }
        const std::vector<float> bayes_vec = bayes.posterior_feature_vector(7, true);
        encode_obs(N, M, T, U, turn, values, st, scores, bayes_vec, value_sum, obs);
        policy.forward_logits(obs, logits);

        int action = 0;
        const std::vector<std::pair<int, int>>& legal = candidates_all[0];
        if(kUseActionMask) {
            action_mask.fill(0);
            for(const auto& mv : legal) {
                const int ax = mv.first;
                const int ay = mv.second;
                if(0 <= ax && ax < 10 && 0 <= ay && ay < 10) action_mask[static_cast<std::size_t>(ax * 10 + ay)] = 1;
            }
            bool any = false;
            for(int i = 0; i < 100; i++) any = any || (action_mask[static_cast<std::size_t>(i)] != 0);
            if(any) {
                action = argmax_masked(logits, action_mask);
            } else {
                action = st.px[0] * 10 + st.py[0];
            }
        } else {
            action = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        }

        if(kUseActionMask) {
            bool ok = false;
            for(const auto& mv : legal) {
                if(action == mv.first * 10 + mv.second) {
                    ok = true;
                    break;
                }
            }
            if(!ok) {
                if(!legal.empty()) {
                    action = legal[0].first * 10 + legal[0].second;
                } else {
                    action = st.px[0] * 10 + st.py[0];
                }
            }
        }

        const int tx = action / 10;
        const int ty = action % 10;
        std::cout << tx << " " << ty << "\n" << std::flush;

        std::vector<std::pair<int, int>> observed_selected(static_cast<std::size_t>(M), {0, 0});
        for(int p = 0; p < M; p++) {
            int sx = 0, sy = 0;
            std::cin >> sx >> sy;
            observed_selected[static_cast<std::size_t>(p)] = {sx, sy};
        }

        bayes.update(values, st.owner, st.level, observed_selected, candidates_all);

        for(int p = 0; p < M; p++) {
            int x = 0, y = 0;
            std::cin >> x >> y;
            st.px[static_cast<std::size_t>(p)] = x;
            st.py[static_cast<std::size_t>(p)] = y;
        }
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                int o = 0;
                std::cin >> o;
                st.owner[static_cast<std::size_t>(i * N + j)] = o;
            }
        }
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                int lv = 0;
                std::cin >> lv;
                st.level[static_cast<std::size_t>(i * N + j)] = lv;
            }
        }
    }

    return 0;
}
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export AHC061 Student-M checkpoint to single-file submission main.cpp (policy-only + C++ bayes).")
    p.add_argument("--checkpoint", type=Path, required=True, help="path to checkpoint (*.pt)")
    p.add_argument("--output", type=Path, default=Path("submission_main.cpp"), help="output main.cpp path")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-action-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bayes-num-particles", type=int, default=128)
    p.add_argument("--bayes-seed", type=int, default=0)
    p.add_argument("--bayes-resample-ess-frac", type=float, default=0.55)
    p.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    p.add_argument("--strict-size-limit", action=argparse.BooleanOptionalAction, default=False)
    return p


def _activation_id(name: str) -> int:
    k = str(name).strip().lower()
    if k in ("", "tanh"):
        return 0
    if k == "relu":
        return 1
    if k == "silu":
        return 2
    raise ValueError(f"unsupported activation for C++ exporter: {name!r}")


def _as_f16_array(state_dict: dict[str, Any], key: str) -> np.ndarray:
    v = state_dict.get(key)
    if v is None or not torch.is_tensor(v):
        raise KeyError(f"missing tensor in checkpoint state_dict: {key}")
    arr = v.detach().cpu().numpy()
    if arr.dtype.kind != "f":
        raise ValueError(f"tensor must be float for C++ export: {key} dtype={arr.dtype}")
    return np.asarray(arr, dtype=np.float16, order="C")


def _collect_policy_tensors(*, state_dict: dict[str, Any], num_blocks: int, global_dim: int, use_global_film: bool, use_global_policy_bias: bool) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    out.append(_as_f16_array(state_dict, "stem.0.weight"))
    out.append(_as_f16_array(state_dict, "stem.0.bias"))
    for bi in range(num_blocks):
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv1.weight"))
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv1.bias"))
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv2.weight"))
        out.append(_as_f16_array(state_dict, f"blocks.{bi}.conv2.bias"))
    if global_dim > 0:
        out.append(_as_f16_array(state_dict, "global_mlp.0.weight"))
        out.append(_as_f16_array(state_dict, "global_mlp.0.bias"))
        out.append(_as_f16_array(state_dict, "global_mlp.2.weight"))
        out.append(_as_f16_array(state_dict, "global_mlp.2.bias"))
    if use_global_film:
        out.append(_as_f16_array(state_dict, "film.weight"))
        out.append(_as_f16_array(state_dict, "film.bias"))
    out.append(_as_f16_array(state_dict, "policy_conv.weight"))
    out.append(_as_f16_array(state_dict, "policy_conv.bias"))
    if use_global_policy_bias:
        out.append(_as_f16_array(state_dict, "policy_global_bias.weight"))
        out.append(_as_f16_array(state_dict, "policy_global_bias.bias"))
    return out


def _pack_policy_blob_base64(tensors: list[np.ndarray]) -> tuple[str, list[int]]:
    counts: list[int] = []
    half_chunks: list[np.ndarray] = []
    for t in tensors:
        h = t.view(np.uint16).reshape(-1)
        counts.append(int(h.size))
        half_chunks.append(h)
    if half_chunks:
        merged = np.concatenate(half_chunks, axis=0)
    else:
        merged = np.zeros((0,), dtype=np.uint16)
    blob = merged.tobytes(order="C")
    b64 = base64.b64encode(blob).decode("ascii")
    return b64, counts


def build_submission_source(
    *,
    board_channels: int,
    board_size: int,
    global_dim: int,
    width: int,
    num_blocks: int,
    global_hidden_dim: int,
    action_dim: int,
    activation_id: int,
    use_global_film: bool,
    use_global_policy_bias: bool,
    deterministic: bool,
    use_action_mask: bool,
    bayes_num_particles: int,
    bayes_seed: int,
    bayes_resample_ess_frac: float,
    tensor_half_counts: list[int],
    model_blob_b64: str,
) -> str:
    b64_lines = textwrap.wrap(model_blob_b64, width=120)
    blob_literal = "\n".join(f'    "{ln}"' for ln in b64_lines)
    src = TEMPLATE_MAIN_CPP
    replacements = {
        "__BOARD_CHANNELS__": str(int(board_channels)),
        "__BOARD_SIZE__": str(int(board_size)),
        "__GLOBAL_DIM__": str(int(global_dim)),
        "__ACTION_DIM__": str(int(action_dim)),
        "__WIDTH__": str(int(width)),
        "__NUM_BLOCKS__": str(int(num_blocks)),
        "__GLOBAL_HIDDEN_DIM__": str(int(global_hidden_dim)),
        "__ACTIVATION_ID__": str(int(activation_id)),
        "__USE_GLOBAL_FILM__": "true" if bool(use_global_film) else "false",
        "__USE_GLOBAL_POLICY_BIAS__": "true" if bool(use_global_policy_bias) else "false",
        "__DETERMINISTIC__": "true" if bool(deterministic) else "false",
        "__USE_ACTION_MASK__": "true" if bool(use_action_mask) else "false",
        "__BAYES_NUM_PARTICLES__": str(int(bayes_num_particles)),
        "__BAYES_SEED__": str(int(bayes_seed)),
        "__BAYES_RESAMPLE_ESS_FRAC__": f"{float(bayes_resample_ess_frac):.10g}",
        "__TENSOR_COUNT__": str(len(tensor_half_counts)),
        "__TENSOR_HALF_COUNTS__": ", ".join(str(int(x)) for x in tensor_half_counts),
        "__MODEL_BLOB_B64_LINES__": blob_literal,
    }
    for k, v in replacements.items():
        src = src.replace(k, v)
    return src


def main() -> None:
    args = build_parser().parse_args()
    ckpt = Path(args.checkpoint)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be dict")

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint has no valid model_config")
    model_type = str(model_config.get("type", "")).strip()
    if model_type != "StudentMBoardAgent":
        raise ValueError(f"unsupported model type for C++ export: {model_type!r} (expected StudentMBoardAgent)")
    model_kwargs = model_config.get("kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError("checkpoint model_config.kwargs must be dict")

    state_dict = payload.get("agent_state_dict")
    if not isinstance(state_dict, dict):
        state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no valid model state_dict")

    board_channels = int(model_kwargs.get("board_channels", 7))
    board_size = int(model_kwargs.get("board_size", 10))
    global_dim = int(model_kwargs.get("global_dim", 49))
    width = int(model_kwargs.get("width", 48))
    num_blocks = int(model_kwargs.get("num_blocks", 2))
    global_hidden_dim = int(model_kwargs.get("global_hidden_dim", 64))
    action_dim = int(payload.get("action_dim", board_size * board_size))
    activation_id = _activation_id(str(model_kwargs.get("activation", "tanh")))
    use_global_film = bool(model_kwargs.get("use_global_film", True)) and global_dim > 0
    use_global_policy_bias = bool(model_kwargs.get("use_global_policy_bias", True)) and global_dim > 0

    if board_channels != 7 or board_size != 10 or global_dim != 49 or action_dim != 100:
        raise ValueError(
            f"unsupported Student-M layout for this exporter: "
            f"board_channels={board_channels}, board_size={board_size}, global_dim={global_dim}, action_dim={action_dim}. "
            "expected (7,10,49,100)."
        )

    tensors = _collect_policy_tensors(
        state_dict=state_dict,
        num_blocks=num_blocks,
        global_dim=global_dim,
        use_global_film=use_global_film,
        use_global_policy_bias=use_global_policy_bias,
    )
    blob_b64, counts = _pack_policy_blob_base64(tensors)

    src = build_submission_source(
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
        deterministic=bool(args.deterministic),
        use_action_mask=bool(args.use_action_mask),
        bayes_num_particles=int(args.bayes_num_particles),
        bayes_seed=int(args.bayes_seed),
        bayes_resample_ess_frac=float(args.bayes_resample_ess_frac),
        tensor_half_counts=counts,
        model_blob_b64=blob_b64,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src, encoding="utf-8")

    source_bytes = out.stat().st_size
    print(f"[export-cpp] checkpoint: {ckpt}")
    print(f"[export-cpp] output: {out}")
    print(f"[export-cpp] source bytes: {source_bytes}")
    print(f"[export-cpp] tensors: {len(tensors)}")
    print(f"[export-cpp] tensor half count total: {sum(counts)}")
    if source_bytes > int(args.max_source_bytes):
        msg = f"[export-cpp] WARNING: source exceeds max-source-bytes ({source_bytes} > {int(args.max_source_bytes)})"
        if bool(args.strict_size_limit):
            raise RuntimeError(msg)
        print(msg)


if __name__ == "__main__":
    main()
