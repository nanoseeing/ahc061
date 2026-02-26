#include "offline.hpp"

#include <cstdlib>
#include <deque>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "utility.hpp"

namespace ahc061 {

OfflineInput read_offline_input(std::istream& is) {
    OfflineInput in;
    is >> in.cfg.N >> in.cfg.M >> in.cfg.T >> in.cfg.U;
    in.cfg.V.assign(in.cfg.N, std::vector<int>(in.cfg.N, 0));
    for(int i = 0; i < in.cfg.N; i++) {
        for(int j = 0; j < in.cfg.N; j++) is >> in.cfg.V[i][j];
    }

    in.xy.resize(in.cfg.M);
    for(int p = 0; p < in.cfg.M; p++) {
        is >> in.xy[p].first >> in.xy[p].second;
    }

    in.wa.resize(in.cfg.M - 1);
    in.wb.resize(in.cfg.M - 1);
    in.wc.resize(in.cfg.M - 1);
    in.wd.resize(in.cfg.M - 1);
    in.eps.resize(in.cfg.M - 1);
    for(int i = 0; i < in.cfg.M - 1; i++) {
        is >> in.wa[i] >> in.wb[i] >> in.wc[i] >> in.wd[i] >> in.eps[i];
    }

    in.r.assign(in.cfg.M - 1, std::vector<double>(2 * in.cfg.T, 0.0));
    for(int t = 0; t < in.cfg.T; t++) {
        for(int i = 0; i < in.cfg.M - 1; i++) {
            is >> in.r[i][2 * t] >> in.r[i][2 * t + 1];
        }
    }

    if(!is) {
        std::cerr << "Failed to read full offline input (hidden params / random table).\n";
        std::cerr << "Use tools/in/*.txt as input for this solver.\n";
        std::exit(1);
    }
    return in;
}

OfflineInput read_offline_input() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    return read_offline_input(std::cin);
}

void validate_offline_input_or_die(const OfflineInput& in) {
    if(in.cfg.M < 2) {
        std::cerr << "Invalid M=" << in.cfg.M << " for contest offline input.\n";
        std::cerr << "Hint: you may be passing tools/out/*.txt. Use tools/in/*.txt instead.\n";
        std::exit(1);
    }
    if(in.cfg.N <= 0 || in.cfg.T <= 0 || in.cfg.U <= 0) {
        std::cerr << "Invalid header values: N=" << in.cfg.N << " T=" << in.cfg.T << " U=" << in.cfg.U << "\n";
        std::exit(1);
    }
    if((int)in.xy.size() != in.cfg.M) {
        std::cerr << "Invalid xy size.\n";
        std::exit(1);
    }
    if((int)in.wa.size() != in.cfg.M - 1 || (int)in.wb.size() != in.cfg.M - 1 || (int)in.wc.size() != in.cfg.M - 1 || (int)in.wd.size() != in.cfg.M - 1 ||
       (int)in.eps.size() != in.cfg.M - 1) {
        std::cerr << "Invalid hidden-parameter vector sizes.\n";
        std::exit(1);
    }
    if((int)in.r.size() != in.cfg.M - 1) {
        std::cerr << "Invalid random table outer size.\n";
        std::exit(1);
    }
    for(int i = 0; i < in.cfg.M - 1; i++) {
        if((int)in.r[i].size() != 2 * in.cfg.T) {
            std::cerr << "Invalid random table inner size at ai=" << i << ".\n";
            std::exit(1);
        }
    }
}

GameState init_state_from_xy(const GameConfig& cfg, const std::vector<std::pair<int, int>>& xy) {
    GameState st;
    st.owner.assign(cfg.N, std::vector<int>(cfg.N, -1));
    st.level.assign(cfg.N, std::vector<int>(cfg.N, 0));
    st.px.resize(cfg.M);
    st.py.resize(cfg.M);
    for(int p = 0; p < cfg.M; p++) {
        st.px[p] = xy[p].first;
        st.py[p] = xy[p].second;
        st.owner[st.px[p]][st.py[p]] = p;
        st.level[st.px[p]][st.py[p]] = 1;
    }
    return st;
}

std::vector<std::pair<int, int>> get_candidates_tools(const GameConfig& cfg, const GameState& st, int player) {
    const int N = cfg.N;
    const int M = cfg.M;
    const int sx = st.px[player];
    const int sy = st.py[player];

    std::vector<std::pair<int, int>> reachable;
    std::vector<std::vector<char>> visited(N, std::vector<char>(N, 0));
    std::deque<std::pair<int, int>> q;
    q.push_back({sx, sy});
    visited[sx][sy] = 1;

    while(!q.empty()) {
        auto [x, y] = q.front();
        q.pop_front();

        bool ok = true;
        for(int i = 0; i < M; i++) {
            if(i != player && st.px[i] == x && st.py[i] == y) {
                ok = false;
                break;
            }
        }
        if(ok) reachable.push_back({x, y});

        if(st.owner[x][y] == player) {
            const int dirs_x[4] = {0, 1, 0, -1}; // right, down, left, up
            const int dirs_y[4] = {1, 0, -1, 0};
            for(int d = 0; d < 4; d++) {
                int nx = x + dirs_x[d];
                int ny = y + dirs_y[d];
                if(0 <= nx && nx < N && 0 <= ny && ny < N && !visited[nx][ny]) {
                    visited[nx][ny] = 1;
                    q.push_back({nx, ny});
                }
            }
        }
    }
    return reachable;
}

bool is_valid_move_tools(const GameConfig& cfg, const GameState& st, int player, std::pair<int, int> target) {
    int x = target.first, y = target.second;
    if(x < 0 || x >= cfg.N || y < 0 || y >= cfg.N) return false;
    for(int i = 0; i < cfg.M; i++) {
        if(i != player && st.px[i] == x && st.py[i] == y) return false;
    }
    auto cands = get_candidates_tools(cfg, st, player);
    for(auto& mv : cands) {
        if(mv == target) return true;
    }
    return false;
}

bool update_state_tools(const GameConfig& cfg, const GameState& st, const std::vector<std::pair<int, int>>& moves, GameState& out_state, std::string& err) {
    if((int)moves.size() != cfg.M) {
        err = "invalid moves size";
        return false;
    }

    for(int i = 0; i < cfg.M; i++) {
        if(!is_valid_move_tools(cfg, st, i, moves[i])) {
            std::ostringstream oss;
            oss << "invalid move for player " << i << ": (" << moves[i].first << "," << moves[i].second << ")";
            err = oss.str();
            return false;
        }
    }

    GameState ns = st.snapshot();
    std::vector<std::pair<int, int>> temp_pos = moves;

    std::unordered_map<long long, int> move_counts;
    move_counts.reserve(cfg.M * 2);
    auto key = [&](int x, int y) -> long long { return (long long)x * 1000LL + y; };
    for(int i = 0; i < cfg.M; i++) {
        move_counts[key(temp_pos[i].first, temp_pos[i].second)]++;
    }

    std::vector<char> collected(cfg.M, 0);
    for(int i = 0; i < cfg.M; i++) {
        auto [x, y] = temp_pos[i];
        if(move_counts[key(x, y)] >= 2) {
            int owner = ns.owner[x][y];
            if(i != owner) collected[i] = 1;
        }
    }

    for(int i = 0; i < cfg.M; i++) {
        if(collected[i]) continue;
        auto [x, y] = temp_pos[i];
        int owner = ns.owner[x][y];
        if(owner == -1) {
            ns.owner[x][y] = i;
            ns.level[x][y] = 1;
        } else if(owner == i) {
            if(ns.level[x][y] < cfg.U) ns.level[x][y] += 1;
        } else {
            ns.level[x][y] -= 1;
            if(ns.level[x][y] == 0) {
                ns.owner[x][y] = i;
                ns.level[x][y] = 1;
            } else {
                collected[i] = 1;
            }
        }
    }

    for(int i = 0; i < cfg.M; i++) {
        if(collected[i]) {
            temp_pos[i] = {st.px[i], st.py[i]};
        }
    }
    for(int i = 0; i < cfg.M; i++) {
        ns.px[i] = temp_pos[i].first;
        ns.py[i] = temp_pos[i].second;
    }

    out_state = std::move(ns);
    return true;
}

} // namespace ahc061
