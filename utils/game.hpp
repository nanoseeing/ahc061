#pragma once

#include <utility>
#include <vector>

namespace ahc061 {

struct GameConfig {
    int N = 0;
    int M = 0;
    int T = 0;
    int U = 0;
    std::vector<std::vector<int>> V;
};

struct GameState {
    std::vector<std::vector<int>> owner; // -1..M-1
    std::vector<std::vector<int>> level; // 0..U
    std::vector<int> px, py;

    GameState snapshot() const;
};

struct TurnResult {
    std::vector<int> tx, ty;
};

struct GameIO {
    static std::pair<GameConfig, GameState> read_initial();
    static void write_move(int x, int y);
    static TurnResult read_turn_result(const GameConfig& cfg, GameState& st);
};

int cell_category(const std::vector<std::vector<int>>& owner, const std::vector<std::vector<int>>& level, int U, int player, int x, int y);
std::vector<std::pair<int, int>> get_candidates_for_player(const GameConfig& cfg, const GameState& st, int player);
std::vector<long long> score_all_players(const GameConfig& cfg, const GameState& st);
GameState simulate_one_turn(const GameConfig& cfg, const GameState& st, const std::vector<std::pair<int, int>>& moves);

} // namespace ahc061

