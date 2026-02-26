#pragma once

#include <istream>
#include <string>
#include <utility>
#include <vector>

#include "game.hpp"

namespace ahc061 {

struct OfflineInput {
    GameConfig cfg;
    std::vector<std::pair<int, int>> xy;
    std::vector<double> wa, wb, wc, wd, eps;
    std::vector<std::vector<double>> r;
};

OfflineInput read_offline_input(std::istream& is);
OfflineInput read_offline_input();
void validate_offline_input_or_die(const OfflineInput& in);
GameState init_state_from_xy(const GameConfig& cfg, const std::vector<std::pair<int, int>>& xy);

std::vector<std::pair<int, int>> get_candidates_tools(const GameConfig& cfg, const GameState& st, int player);
bool is_valid_move_tools(const GameConfig& cfg, const GameState& st, int player, std::pair<int, int> target);
bool update_state_tools(const GameConfig& cfg, const GameState& st, const std::vector<std::pair<int, int>>& moves, GameState& out_state, std::string& err);

} // namespace ahc061
