#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>

#include "ahc061/base/state.hpp"

namespace ahc061::exp002 {

inline void fill_frontier_from_binary_mask(
    const std::array<std::uint8_t, CELL_MAX>& mask,
    float* out_plane) {  // [CELL_MAX]
    for (int idx = 0; idx < CELL_MAX; idx++) {
        if (mask[static_cast<std::size_t>(idx)] != 0) {
            out_plane[idx] = 0.0f;
            continue;
        }
        bool hit = false;
        for (int dir = 0; dir < 4; dir++) {
            const int ni = NEIGH_RDLU[idx][dir];
            if (ni >= 0 && mask[static_cast<std::size_t>(ni)] != 0) {
                hit = true;
                break;
            }
        }
        out_plane[idx] = hit ? 1.0f : 0.0f;
    }
}

inline void fill_manhattan_dist_from_point_clip(
    int px,
    int py,
    float clip_max,
    float* out_plane) {  // [CELL_MAX]
    const float inv_clip = (clip_max > 1e-6f) ? (1.0f / clip_max) : 1.0f;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            const int idx = cell_index(x, y);
            const int d = std::abs(x - px) + std::abs(y - py);
            const float dc = std::min(static_cast<float>(d), clip_max);
            out_plane[idx] = dc * inv_clip;
        }
    }
}

inline void fill_manhattan_dist_to_sources_clip(
    const std::array<std::uint8_t, CELL_MAX>& is_source,
    float clip_max,
    float* out_plane) {  // [CELL_MAX]
    constexpr int INF = 1 << 20;
    std::array<int, CELL_MAX> dist{};
    bool has_source = false;
    for (int i = 0; i < CELL_MAX; i++) {
        if (is_source[static_cast<std::size_t>(i)] != 0) {
            dist[static_cast<std::size_t>(i)] = 0;
            has_source = true;
        } else {
            dist[static_cast<std::size_t>(i)] = INF;
        }
    }

    if (!has_source) {
        for (int i = 0; i < CELL_MAX; i++)
            out_plane[i] = 1.0f;
        return;
    }

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            const int idx = cell_index(x, y);
            int d = dist[static_cast<std::size_t>(idx)];
            if (x > 0)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x - 1, y))] + 1);
            if (y > 0)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x, y - 1))] + 1);
            dist[static_cast<std::size_t>(idx)] = d;
        }
    }
    for (int x = N - 1; x >= 0; x--) {
        for (int y = N - 1; y >= 0; y--) {
            const int idx = cell_index(x, y);
            int d = dist[static_cast<std::size_t>(idx)];
            if (x + 1 < N)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x + 1, y))] + 1);
            if (y + 1 < N)
                d = std::min(d, dist[static_cast<std::size_t>(cell_index(x, y + 1))] + 1);
            dist[static_cast<std::size_t>(idx)] = d;
        }
    }

    const float inv_clip = (clip_max > 1e-6f) ? (1.0f / clip_max) : 1.0f;
    for (int i = 0; i < CELL_MAX; i++) {
        const int d = dist[static_cast<std::size_t>(i)];
        out_plane[i] = (d >= INF) ? 1.0f : (std::min(static_cast<float>(d), clip_max) * inv_clip);
    }
}

}  // namespace ahc061::exp002
