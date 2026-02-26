#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "game.hpp"
#include "utility.hpp"

namespace ahc061 {

struct Particle {
    double wa, wb, wc, wd, eps;
    double w;
};

struct AITrackerParams {
    int num_particles = 1000;
    double resample_ess_frac = 0.55;

    // Ratio-space mutation sigmas for wb/wa, wc/wa, wd/wa (log-space random walk).
    double mutate_sigma_wb = 0.03;
    double mutate_sigma_wc = 0.03;
    double mutate_sigma_wd = 0.03;
    // Epsilon mutation sigma in logit space over normalized [E_LO, E_HI].
    double mutate_sigma_eps_logit = 0.20;
    double mutate_rate = 0.15;

    // Likelihood tempering coefficient lower bound (upper bound is 1.0).
    double temper_beta_min = 0.35;

    // MH rejuvenation after resampling.
    int rejuvenation_steps = 1;
    double rejuvenation_scale = 0.75;
};

struct AITracker {
    const GameConfig* cfg;
    AITrackerParams params;
    RNG* rng;

    std::vector<Particle> particles;
    std::vector<std::pair<int, std::pair<int, int>>> move_history;

    AITracker(const GameConfig& cfg_, const AITrackerParams& params_, RNG& rng_);
    AITracker(const GameConfig& cfg_, int num_particles_, double resample_ess_frac_, RNG& rng_, double mutate_sigma_w_, double mutate_sigma_eps_,
              double mutate_rate_);

    void mutate_some();
    void rejuvenate_with_mh(const GameState& st_start, int player, std::pair<int, int> observed, const std::vector<std::pair<int, int>>& cands, double temper_beta);
    void update(const GameState& st_start, int player, std::pair<int, int> observed, int turn);
    std::array<double, 5> posterior_mean() const;
    std::string debug_string() const;
};

std::vector<std::pair<std::pair<int, int>, double>> predict_ai_move_distribution_topk(
    const GameConfig& cfg, const GameState& st_start, int player, const std::vector<Particle>& particles, int top_k);

std::vector<std::pair<std::pair<int, int>, double>> predict_ai_move_distribution_topk_mean_params(
    const GameConfig& cfg, const GameState& st_start, int player, double wa, double wb, double wc, double wd, double eps, int top_k);

std::pair<int, int> sample_from_dist(const std::vector<std::pair<std::pair<int, int>, double>>& dist, RNG& rng);

} // namespace ahc061
