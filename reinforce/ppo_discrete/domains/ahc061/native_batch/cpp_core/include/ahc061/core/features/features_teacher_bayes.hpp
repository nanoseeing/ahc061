#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "ahc061/core/base/state.hpp"
#include "ahc061/core/features/feature_common.hpp"
#include "ahc061/core/opponent/move_summary.hpp"

namespace ahc061::exp002 {

inline float teacher_clipf(float v, float lo, float hi) {
    return std::min(hi, std::max(lo, v));
}

inline std::array<float, 4> teacher_bayes_norm_row_from_pf(const FeatureCommon& common, int old_p) {
    std::array<float, 4> out{0.0f, 0.0f, 0.0f, 0.0f};
    if (!common.pf_enabled || common.pf == nullptr)
        return out;
    if (old_p <= 0 || old_p >= common.st->m)
        return out;

    constexpr double R_LO = 0.3;
    constexpr double R_HI = 1.0 / 0.3;
    constexpr double E_LO = 0.1;
    constexpr double E_HI = 0.5;

    const auto clipd = [](double v, double lo, double hi) {
        return std::min(hi, std::max(lo, v));
    };

    const auto& pf_p = (*common.pf)[static_cast<std::size_t>(old_p)];
    std::array<double, ParticleFilterSMC::P> w{};
    pf_p.normalized_weights(w);

    double wa = 0.0;
    double wb = 0.0;
    double wc = 0.0;
    double wd = 0.0;
    double eps = 0.0;
    for (int i = 0; i < ParticleFilterSMC::P; i++) {
        const double wi = w[static_cast<std::size_t>(i)];
        const auto& th = pf_p.particle_theta(i);
        wa += wi * th[0];
        wb += wi * th[1];
        wc += wi * th[2];
        wd += wi * th[3];
        eps += wi * th[4];
    }

    if (!std::isfinite(wa) || std::abs(wa) < 1e-12)
        wa = 1.0;

    const double rb = clipd(wb / wa, R_LO, R_HI);
    const double rc = clipd(wc / wa, R_LO, R_HI);
    const double rd = clipd(wd / wa, R_LO, R_HI);
    const double e = clipd(eps, E_LO, E_HI);

    const double norm_ratio_den = std::max(1e-12, R_HI - R_LO);
    const double norm_eps_den = std::max(1e-12, E_HI - E_LO);
    out[0] = static_cast<float>(clipd((rb - R_LO) / norm_ratio_den, 0.0, 1.0));
    out[1] = static_cast<float>(clipd((rc - R_LO) / norm_ratio_den, 0.0, 1.0));
    out[2] = static_cast<float>(clipd((rd - R_LO) / norm_ratio_den, 0.0, 1.0));
    out[3] = static_cast<float>(clipd((e - E_LO) / norm_eps_den, 0.0, 1.0));
    return out;
}

inline void teacher_decode_bayes_from_normalized_row(
    const std::array<float, 4>& row,
    double& rb,
    double& rc,
    double& rd,
    double& eps) {
    constexpr double R_LO = 0.3;
    constexpr double R_HI = 1.0 / 0.3;
    constexpr double E_LO = 0.1;
    constexpr double E_HI = 0.5;

    const double nb = static_cast<double>(teacher_clipf(row[0], 0.0f, 1.0f));
    const double nc = static_cast<double>(teacher_clipf(row[1], 0.0f, 1.0f));
    const double nd = static_cast<double>(teacher_clipf(row[2], 0.0f, 1.0f));
    const double ne = static_cast<double>(teacher_clipf(row[3], 0.0f, 1.0f));

    rb = R_LO + nb * (R_HI - R_LO);
    rc = R_LO + nc * (R_HI - R_LO);
    rd = R_LO + nd * (R_HI - R_LO);
    eps = E_LO + ne * (E_HI - E_LO);
}

inline void teacher_write_enemy_next_prob_map(
    const State& st,
    int player,
    const int* moves,
    int cnt,
    const std::array<float, 4>& bayes_norm_row,
    float* out_plane) {  // [CELL_MAX]
    std::fill(out_plane, out_plane + CELL_MAX, 0.0f);
    if (cnt <= 0)
        return;

    double rb = 1.0;
    double rc = 1.0;
    double rd = 1.0;
    double eps = 0.3;
    teacher_decode_bayes_from_normalized_row(bayes_norm_row, rb, rc, rd, eps);

    std::array<double, CELL_MAX> scores{};
    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < cnt; i++) {
        const int idx = moves[static_cast<std::size_t>(i)];
        const int cat = categorize_move_for_ai(st, player, idx);
        const double v = static_cast<double>(st.value[idx]);

        double a = 0.0;
        if (cat == 0) {
            a = v;
        } else if (cat == 1) {
            a = v * rb;
        } else if (cat == -1) {
            a = 0.0;
        } else if (cat == 2) {
            a = v * rc;
        } else {
            a = v * rd;
        }

        scores[static_cast<std::size_t>(i)] = a;
        mx = std::max(mx, a);
    }

    const double tol = 1e-9 * std::max(std::abs(mx), 1.0);
    int best_n = 0;
    for (int i = 0; i < cnt; i++) {
        if (scores[static_cast<std::size_t>(i)] >= mx - tol)
            best_n++;
    }
    if (best_n <= 0) {
        const float p = 1.0f / static_cast<float>(cnt);
        for (int i = 0; i < cnt; i++)
            out_plane[moves[static_cast<std::size_t>(i)]] = p;
        return;
    }

    eps = std::min(1.0 - 1e-6, std::max(1e-6, eps));
    const double p_rand = eps / static_cast<double>(cnt);
    const double p_greedy = (1.0 - eps) / static_cast<double>(best_n);

    for (int i = 0; i < cnt; i++) {
        const bool best = (scores[static_cast<std::size_t>(i)] >= mx - tol);
        const double p = p_rand + (best ? p_greedy : 0.0);
        out_plane[moves[static_cast<std::size_t>(i)]] = static_cast<float>(p);
    }
}

}  // namespace ahc061::exp002
