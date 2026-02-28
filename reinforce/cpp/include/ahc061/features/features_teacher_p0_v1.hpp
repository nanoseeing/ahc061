#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ahc061/features/feature_common.hpp"
#include "ahc061/features/feature_grid_utils.hpp"
#include "ahc061/features/features_teacher_bayes.hpp"
#include "ahc061/base/state.hpp"

namespace ahc061 {

static constexpr int FEATURE_TEACHER_P0_V1_88CH_C = 88;

static constexpr int CH_TCH_TURN_FRAC = 0;
static constexpr int CH_TCH_M_BASE = 1;
static constexpr int CH_TCH_U_BASE = 2;
static constexpr int CH_TCH_VALUE_NORM = 3;
static constexpr int CH_TCH_OWNER_NEUTRAL = 4;

static constexpr int CH_TCH_SELF_LEVEL = 5;
static constexpr int CH_TCH_SELF_POS = 6;
static constexpr int CH_TCH_SELF_REACH = 7;
static constexpr int CH_TCH_SELF_FRONTIER = 8;
static constexpr int CH_TCH_SELF_DIST_FROM_PIECE = 9;
static constexpr int CH_TCH_SELF_DIST_TO_REACH = 10;

static constexpr int CH_TCH_ENEMY0 = 11;  // 11 channels x (M_MAX - 1)

static_assert(CH_TCH_ENEMY0 + (M_MAX - 1) * 11 == FEATURE_TEACHER_P0_V1_88CH_C, "FEATURE_TEACHER_P0_V1_88CH_C mismatch");

inline void write_features_teacher_p0_v1_88ch_from_common(const FeatureCommon& common, float* out_board) {
    const State& st = *common.st;
    const int plane_size = CELL_MAX;

    std::memset(out_board, 0, static_cast<std::size_t>(FEATURE_TEACHER_P0_V1_88CH_C * plane_size) * sizeof(float));

    const float turn_frac = (st.t_max > 0) ? (static_cast<float>(common.turn) / static_cast<float>(st.t_max)) : 0.0f;
    const float m_base = static_cast<float>(st.m - 2) / 6.0f;
    const float u_base = static_cast<float>(st.u_max - 1) / 4.0f;

    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_TCH_TURN_FRAC * plane_size + idx] = turn_frac;
        out_board[CH_TCH_M_BASE * plane_size + idx] = m_base;
        out_board[CH_TCH_U_BASE * plane_size + idx] = u_base;
    }

    const float inv_1000 = 1.0f / 1000.0f;
    for (int idx = 0; idx < plane_size; idx++) {
        out_board[CH_TCH_VALUE_NORM * plane_size + idx] = static_cast<float>(st.value[idx]) * inv_1000;
        if (static_cast<int>(st.owner[idx]) == -1)
            out_board[CH_TCH_OWNER_NEUTRAL * plane_size + idx] = 1.0f;
    }

    const auto& reach = *common.reach;
    const auto& moves = *common.moves;
    const auto& move_cnt = *common.move_cnt;

    const float inv_u = (st.u_max > 0) ? (1.0f / static_cast<float>(st.u_max)) : 1.0f;
    for (int idx = 0; idx < plane_size; idx++) {
        if (static_cast<int>(st.owner[idx]) == 0)
            out_board[CH_TCH_SELF_LEVEL * plane_size + idx] = static_cast<float>(st.level[idx]) * inv_u;
        if (reach[0][static_cast<std::size_t>(idx)] != 0)
            out_board[CH_TCH_SELF_REACH * plane_size + idx] = 1.0f;
    }
    const int self_idx = cell_index(static_cast<int>(st.ex[0]), static_cast<int>(st.ey[0]));
    out_board[CH_TCH_SELF_POS * plane_size + self_idx] = 1.0f;

    fill_frontier_from_binary_mask(
        reach[0],
        out_board + static_cast<std::ptrdiff_t>(CH_TCH_SELF_FRONTIER) * plane_size);
    fill_manhattan_dist_from_point_clip(
        static_cast<int>(st.ex[0]),
        static_cast<int>(st.ey[0]),
        18.0f,
        out_board + static_cast<std::ptrdiff_t>(CH_TCH_SELF_DIST_FROM_PIECE) * plane_size);
    fill_manhattan_dist_to_sources_clip(
        reach[0],
        10.0f,
        out_board + static_cast<std::ptrdiff_t>(CH_TCH_SELF_DIST_TO_REACH) * plane_size);

    std::array<int, M_MAX> new_to_old{};
    new_to_old.fill(-1);
    for (int old_p = 0; old_p < st.m; old_p++) {
        const int np = common.old_to_new[static_cast<std::size_t>(old_p)];
        if (0 <= np && np < M_MAX)
            new_to_old[static_cast<std::size_t>(np)] = old_p;
    }

    for (int slot = 0; slot < M_MAX - 1; slot++) {
        const int np = slot + 1;
        const int old_p = new_to_old[static_cast<std::size_t>(np)];
        if (old_p <= 0 || old_p >= st.m)
            continue;

        const int base = CH_TCH_ENEMY0 + slot * 11;

        const int ch_level = base + 0;
        const int ch_pos = base + 1;
        const int ch_reach = base + 2;
        const int ch_frontier = base + 3;
        const int ch_dist_from = base + 4;
        const int ch_dist_to = base + 5;
        const int ch_next_prob = base + 6;
        const int ch_bwb = base + 7;
        const int ch_bwc = base + 8;
        const int ch_bwd = base + 9;
        const int ch_beps = base + 10;

        for (int idx = 0; idx < plane_size; idx++) {
            if (static_cast<int>(st.owner[idx]) == old_p)
                out_board[ch_level * plane_size + idx] = static_cast<float>(st.level[idx]) * inv_u;
            if (reach[static_cast<std::size_t>(old_p)][static_cast<std::size_t>(idx)] != 0)
                out_board[ch_reach * plane_size + idx] = 1.0f;
        }

        const int pidx = cell_index(static_cast<int>(st.ex[old_p]), static_cast<int>(st.ey[old_p]));
        out_board[ch_pos * plane_size + pidx] = 1.0f;

        fill_frontier_from_binary_mask(
            reach[static_cast<std::size_t>(old_p)],
            out_board + static_cast<std::ptrdiff_t>(ch_frontier) * plane_size);
        fill_manhattan_dist_from_point_clip(
            static_cast<int>(st.ex[old_p]),
            static_cast<int>(st.ey[old_p]),
            18.0f,
            out_board + static_cast<std::ptrdiff_t>(ch_dist_from) * plane_size);
        fill_manhattan_dist_to_sources_clip(
            reach[static_cast<std::size_t>(old_p)],
            10.0f,
            out_board + static_cast<std::ptrdiff_t>(ch_dist_to) * plane_size);

        const std::array<float, 4> bayes_row = teacher_bayes_norm_row_from_pf(common, old_p);
        teacher_write_enemy_next_prob_map(
            st,
            old_p,
            moves[static_cast<std::size_t>(old_p)].data(),
            move_cnt[static_cast<std::size_t>(old_p)],
            bayes_row,
            out_board + static_cast<std::ptrdiff_t>(ch_next_prob) * plane_size);

        for (int idx = 0; idx < plane_size; idx++) {
            out_board[ch_bwb * plane_size + idx] = bayes_row[0];
            out_board[ch_bwc * plane_size + idx] = bayes_row[1];
            out_board[ch_bwd * plane_size + idx] = bayes_row[2];
            out_board[ch_beps * plane_size + idx] = bayes_row[3];
        }
    }
}

}  // namespace ahc061
