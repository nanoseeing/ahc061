#include "game.hpp"

#include <deque>
#include <iostream>
#include <unordered_map>

#include "utility.hpp"

namespace ahc061 {

GameState GameState::snapshot() const {
    GameState s;
    s.owner = owner;
    s.level = level;
    s.px = px;
    s.py = py;
    return s;
}

std::pair<GameConfig, GameState> GameIO::read_initial() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    GameConfig cfg;
    std::cin >> cfg.N >> cfg.M >> cfg.T >> cfg.U;
    cfg.V.assign(cfg.N, std::vector<int>(cfg.N, 0));
    for(int i = 0; i < cfg.N; i++) {
        for(int j = 0; j < cfg.N; j++) std::cin >> cfg.V[i][j];
    }
    std::vector<int> sx(cfg.M), sy(cfg.M);
    for(int p = 0; p < cfg.M; p++) std::cin >> sx[p] >> sy[p];

    GameState st;
    st.owner.assign(cfg.N, std::vector<int>(cfg.N, -1));
    st.level.assign(cfg.N, std::vector<int>(cfg.N, 0));
    st.px = sx;
    st.py = sy;

    for(int p = 0; p < cfg.M; p++) {
        st.owner[sx[p]][sy[p]] = p;
        st.level[sx[p]][sy[p]] = 1;
    }
    return {cfg, st};
}

void GameIO::write_move(int x, int y) {
    std::cout << x << " " << y << "\n";
    std::cout.flush();
}

TurnResult GameIO::read_turn_result(const GameConfig& cfg, GameState& st) {
    TurnResult tr;
    tr.tx.assign(cfg.M, 0);
    tr.ty.assign(cfg.M, 0);

    for(int p = 0; p < cfg.M; p++) std::cin >> tr.tx[p] >> tr.ty[p];
    for(int p = 0; p < cfg.M; p++) std::cin >> st.px[p] >> st.py[p];

    for(int i = 0; i < cfg.N; i++) {
        for(int j = 0; j < cfg.N; j++) std::cin >> st.owner[i][j];
    }
    for(int i = 0; i < cfg.N; i++) {
        for(int j = 0; j < cfg.N; j++) std::cin >> st.level[i][j];
    }
    return tr;
}

int cell_category(const std::vector<std::vector<int>>& owner, const std::vector<std::vector<int>>& level, int U, int player, int x, int y) {
    int o = owner[x][y];
    if(o == -1) return 0;
    if(o == player) return (level[x][y] >= U) ? 2 : 1;
    return (level[x][y] == 1) ? 3 : 4;
}

std::vector<std::pair<int, int>> get_candidates_for_player(const GameConfig& cfg, const GameState& st, int player) {
    const int N = cfg.N;
    const int M = cfg.M;
    const int sx = st.px[player];
    const int sy = st.py[player];

    std::vector<std::vector<char>> vis(N, std::vector<char>(N, 0));
    std::deque<std::pair<int, int>> q;
    vis[sx][sy] = 1;
    q.push_back({sx, sy});

    std::vector<std::pair<int, int>> reachable;
    reachable.reserve(N * N);

    while(!q.empty()) {
        auto [x, y] = q.front();
        q.pop_front();
        if(st.owner[x][y] != player) continue;

        reachable.push_back({x, y});
        for(int d = 0; d < 4; d++) {
            int nx = x + DX[d], ny = y + DY[d];
            if(0 <= nx && nx < N && 0 <= ny && ny < N && !vis[nx][ny] && st.owner[nx][ny] == player) {
                vis[nx][ny] = 1;
                q.push_back({nx, ny});
            }
        }
    }

    std::vector<std::vector<char>> inCand(N, std::vector<char>(N, 0));
    std::vector<std::pair<int, int>> cands;
    cands.reserve(N * N);

    auto add = [&](int x, int y) {
        if(!inCand[x][y]) {
            inCand[x][y] = 1;
            cands.push_back({x, y});
        }
    };

    if(reachable.empty()) {
        add(sx, sy);
    } else {
        for(auto [x, y] : reachable)
            add(x, y);
        for(auto [x, y] : reachable) {
            for(int d = 0; d < 4; d++) {
                int nx = x + DX[d], ny = y + DY[d];
                if(0 <= nx && nx < N && 0 <= ny && ny < N) add(nx, ny);
            }
        }
    }

    for(int p = 0; p < M; p++) {
        if(p == player) continue;
        int ox = st.px[p], oy = st.py[p];
        if(0 <= ox && ox < N && 0 <= oy && oy < N) {
            if(inCand[ox][oy]) inCand[ox][oy] = 0;
        }
    }

    std::vector<std::pair<int, int>> filtered;
    filtered.reserve(cands.size());
    for(auto& mv : cands) {
        if(inCand[mv.first][mv.second]) filtered.push_back(mv);
    }
    if(filtered.empty()) filtered.push_back({sx, sy});
    return filtered;
}

std::vector<long long> score_all_players(const GameConfig& cfg, const GameState& st) {
    std::vector<long long> s(cfg.M, 0);
    for(int i = 0; i < cfg.N; i++) {
        for(int j = 0; j < cfg.N; j++) {
            int o = st.owner[i][j];
            if(o >= 0) {
                s[o] += 1LL * cfg.V[i][j] * st.level[i][j];
            }
        }
    }
    return s;
}

GameState simulate_one_turn(const GameConfig& cfg, const GameState& st, const std::vector<std::pair<int, int>>& moves) {
    const int M = cfg.M;
    const int U = cfg.U;

    GameState ns;
    ns.owner = st.owner;
    ns.level = st.level;
    ns.px = st.px;
    ns.py = st.py;

    std::vector<std::pair<int, int>> start_pos(M), dest_pos(M);
    for(int p = 0; p < M; p++) {
        start_pos[p] = {st.px[p], st.py[p]};
        dest_pos[p] = moves[p];
    }

    std::unordered_map<long long, std::vector<int>> cell2ps;
    cell2ps.reserve(M * 2);
    auto key = [&](int x, int y) -> long long { return (long long)x * 1000LL + y; };
    for(int p = 0; p < M; p++) {
        cell2ps[key(dest_pos[p].first, dest_pos[p].second)].push_back(p);
    }

    std::vector<char> collected(M, 0);
    for(auto& kv : cell2ps) {
        auto& ps = kv.second;
        if((int)ps.size() < 2) continue;
        int x = dest_pos[ps[0]].first;
        int y = dest_pos[ps[0]].second;
        int cell_owner = ns.owner[x][y];
        if(cell_owner != -1) {
            bool owner_in = false;
            for(int p : ps)
                if(p == cell_owner) owner_in = true;
            if(owner_in) {
                for(int p : ps)
                    if(p != cell_owner) collected[p] = 1;
            } else {
                for(int p : ps)
                    collected[p] = 1;
            }
        } else {
            for(int p : ps)
                collected[p] = 1;
        }
    }

    for(int p = 0; p < M; p++) {
        if(collected[p]) continue;
        int x = dest_pos[p].first;
        int y = dest_pos[p].second;
        int cell_owner = ns.owner[x][y];
        if(cell_owner == -1) {
            ns.owner[x][y] = p;
            ns.level[x][y] = 1;
        } else if(cell_owner == p) {
            if(ns.level[x][y] < U) ns.level[x][y] += 1;
        } else {
            ns.level[x][y] -= 1;
            if(ns.level[x][y] == 0) {
                ns.owner[x][y] = p;
                ns.level[x][y] = 1;
            } else {
                collected[p] = 1;
            }
        }
    }

    for(int p = 0; p < M; p++) {
        if(collected[p]) dest_pos[p] = start_pos[p];
    }
    for(int p = 0; p < M; p++) {
        ns.px[p] = dest_pos[p].first;
        ns.py[p] = dest_pos[p].second;
    }
    return ns;
}

} // namespace ahc061

