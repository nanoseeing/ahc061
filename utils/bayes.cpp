#include "bayes.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace ahc061 {

namespace {

constexpr double R_LO = W_LO / W_HI;
constexpr double R_HI = W_HI / W_LO;
const double LOG_R_LO = std::log(R_LO);
const double LOG_R_HI = std::log(R_HI);

struct CanonicalParams {
    double wb_over_wa;
    double wc_over_wa;
    double wd_over_wa;
    double eps;
};

inline double sigmoid(double z) {
    if(z >= 0.0) {
        double e = std::exp(-z);
        return 1.0 / (1.0 + e);
    }
    double e = std::exp(z);
    return e / (1.0 + e);
}

inline double mutate_ratio_with_sigma(double r, double sigma, RNG& rng) {
    if(sigma <= 0.0) return clip(r, R_LO, R_HI);
    double x = std::log(clip(r, R_LO, R_HI));
    x = clip(x + rng.gauss(0.0, sigma), LOG_R_LO, LOG_R_HI);
    return std::exp(x);
}

inline double mutate_eps_logit_with_sigma(double eps, double sigma, RNG& rng) {
    eps = clip(eps, E_LO, E_HI);
    if(sigma <= 0.0) return eps;
    double u = (eps - E_LO) / (E_HI - E_LO);
    u = clip(u, 1e-6, 1.0 - 1e-6);
    double z = std::log(u / (1.0 - u));
    z += rng.gauss(0.0, sigma);
    double u2 = clip(sigmoid(z), 1e-6, 1.0 - 1e-6);
    return E_LO + (E_HI - E_LO) * u2;
}

inline double adaptive_temper_beta(double ess_before, int num_particles, double resample_ess_frac, double beta_min) {
    if(num_particles <= 0) return 1.0;
    double ratio = ess_before / (double)num_particles;
    if(ratio >= resample_ess_frac) return 1.0;
    if(resample_ess_frac <= 1e-9) return 1.0;
    double b = ratio / resample_ess_frac;
    return clip(b, beta_min, 1.0);
}

std::array<double, 4> reconstruct_weights_symmetric_margin(double wb_over_wa, double wc_over_wa, double wd_over_wa) {
    std::array<double, 4> r = {1.0, clip(wb_over_wa, R_LO, R_HI), clip(wc_over_wa, R_LO, R_HI), clip(wd_over_wa, R_LO, R_HI)};
    constexpr double SPREAD_LIMIT = W_HI / W_LO;

    auto get_minmax = [&]() {
        double mn = r[0], mx = r[0];
        for(double v : r) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        return std::pair<double, double>{mn, mx};
    };

    auto [rmin0, rmax0] = get_minmax();
    if(!std::isfinite(rmin0) || !std::isfinite(rmax0) || rmin0 <= 0.0 || rmax0 <= 0.0) {
        double mid = 0.5 * (W_LO + W_HI);
        return {mid, mid, mid, mid};
    }

    // If ratio spread is infeasible for [W_LO, W_HI], compress it toward 1 in log-space.
    double spread = rmax0 / rmin0;
    if(spread > SPREAD_LIMIT + 1e-12) {
        double alpha = std::log(SPREAD_LIMIT) / std::log(spread);
        alpha = clip(alpha, 0.0, 1.0);
        for(double& v : r) {
            v = std::exp(std::log(v) * alpha);
        }
    }

    auto [rmin, rmax] = get_minmax();
    double c_low = W_LO / rmin;
    double c_high = W_HI / rmax;
    if(c_low > c_high) {
        // Numerical safety fallback (should rarely happen after spread compression).
        double c_mid = 0.5 * (c_low + c_high);
        double w = clip(c_mid, W_LO, W_HI);
        return {w, w, w, w};
    }

    // Equalize margin to lower and upper bounds on extrema:
    // c*rmin - W_LO == W_HI - c*rmax  =>  c = (W_LO + W_HI) / (rmin + rmax)
    double c_eq = (W_LO + W_HI) / (rmin + rmax);
    double c = clip(c_eq, c_low, c_high);
    return {
        clip(c * r[0], W_LO, W_HI),
        clip(c * r[1], W_LO, W_HI),
        clip(c * r[2], W_LO, W_HI),
        clip(c * r[3], W_LO, W_HI),
    };
}

CanonicalParams canonicalize(const Particle& p) {
    double wa = p.wa;
    if(!std::isfinite(wa) || std::fabs(wa) < 1e-12) wa = 1.0;
    CanonicalParams cp;
    cp.wb_over_wa = clip(p.wb / wa, R_LO, R_HI);
    cp.wc_over_wa = clip(p.wc / wa, R_LO, R_HI);
    cp.wd_over_wa = clip(p.wd / wa, R_LO, R_HI);
    cp.eps = clip(p.eps, E_LO, E_HI);
    return cp;
}

double score_of_cell_category(int cat, double val, const CanonicalParams& cp) {
    if(cat == 0) return val; // wa is fixed to 1.0 in identifiable parameterization
    if(cat == 1) return val * cp.wb_over_wa;
    if(cat == 2) return 0.0;
    if(cat == 3) return val * cp.wc_over_wa;
    return val * cp.wd_over_wa;
}

Particle sample_prior_particle(RNG& rng) {
    double wa_raw = rng.uniform(W_LO, W_HI);
    double wb_raw = rng.uniform(W_LO, W_HI);
    double wc_raw = rng.uniform(W_LO, W_HI);
    double wd_raw = rng.uniform(W_LO, W_HI);

    Particle p;
    p.wa = 1.0;
    p.wb = clip(wb_raw / wa_raw, R_LO, R_HI);
    p.wc = clip(wc_raw / wa_raw, R_LO, R_HI);
    p.wd = clip(wd_raw / wa_raw, R_LO, R_HI);
    p.eps = rng.uniform(E_LO, E_HI);
    p.w = 1.0;
    return p;
}

void normalize_weights(std::vector<Particle>& ps) {
    double s = 0.0;
    for(auto& p : ps)
        s += p.w;
    if(s <= 0.0) {
        double uni = 1.0 / (double)ps.size();
        for(auto& p : ps)
            p.w = uni;
        return;
    }
    double inv = 1.0 / s;
    for(auto& p : ps)
        p.w *= inv;
}

double effective_sample_size(const std::vector<Particle>& ps) {
    double s2 = 0.0;
    for(auto& p : ps)
        s2 += p.w * p.w;
    if(s2 <= 1e-18) return 0.0;
    return 1.0 / s2;
}

std::vector<Particle> systematic_resample(const std::vector<Particle>& ps, RNG& rng) {
    int n = (int)ps.size();
    std::vector<double> cdf(n);
    double acc = 0.0;
    for(int i = 0; i < n; i++) {
        acc += ps[i].w;
        cdf[i] = acc;
    }

    double u0 = rng.uniform01() / n;
    std::vector<Particle> out;
    out.reserve(n);
    int j = 0;
    for(int i = 0; i < n; i++) {
        double u = u0 + (double)i / n;
        while(j < n - 1 && cdf[j] < u)
            j++;
        Particle src = ps[j];
        src.w = 1.0 / n;
        out.push_back(src);
    }
    return out;
}

double particle_move_likelihood(const GameConfig& cfg, const GameState& st_start, int player, const std::vector<std::pair<int, int>>& candidates,
                                std::pair<int, int> observed, const Particle& part) {
    if(candidates.empty()) return 1e-12;
    const auto cp = canonicalize(part);

    std::vector<double> scores;
    scores.reserve(candidates.size());
    int obs_idx = -1;

    for(int i = 0; i < (int)candidates.size(); i++) {
        auto [x, y] = candidates[i];
        if(x == observed.first && y == observed.second) obs_idx = i;
        int cat = cell_category(st_start.owner, st_start.level, cfg.U, player, x, y);
        double val = (double)cfg.V[x][y];
        scores.push_back(score_of_cell_category(cat, val, cp));
    }

    if(obs_idx < 0) return 1e-12;

    int B = (int)candidates.size();
    double eps = clip(cp.eps, 1e-6, 1.0 - 1e-6);
    double p_rand = eps / (double)B;

    double max_a = *std::max_element(scores.begin(), scores.end());
    double tol = 1e-9 * std::max(std::fabs(max_a), 1.0);

    std::vector<int> best;
    best.reserve(B);
    for(int i = 0; i < B; i++) {
        if(scores[i] >= max_a - tol) best.push_back(i);
    }
    int G = (int)best.size();

    bool in_best = false;
    for(int idx : best) {
        if(idx == obs_idx) {
            in_best = true;
            break;
        }
    }

    double p_greedy = in_best ? (1.0 - eps) * (1.0 / (double)G) : 0.0;
    double p = p_rand + p_greedy;
    if(p < 1e-12) p = 1e-12;
    return p;
}

} // namespace

AITracker::AITracker(const GameConfig& cfg_, const AITrackerParams& params_, RNG& rng_) : cfg(&cfg_), params(params_), rng(&rng_) {
    particles.reserve(params.num_particles);
    for(int i = 0; i < params.num_particles; i++) {
        Particle p = sample_prior_particle(*rng);
        p.w = 1.0 / params.num_particles;
        particles.push_back(p);
    }
}

AITracker::AITracker(const GameConfig& cfg_, int num_particles_, double resample_ess_frac_, RNG& rng_, double mutate_sigma_w_, double mutate_sigma_eps_,
                     double mutate_rate_)
    : AITracker(
          cfg_,
          AITrackerParams{
              num_particles_,
              resample_ess_frac_,
              mutate_sigma_w_,
              mutate_sigma_w_,
              mutate_sigma_w_,
              mutate_sigma_eps_ / std::max(1e-6, 0.25 * (E_HI - E_LO)),
              mutate_rate_,
              0.35,
              1,
              0.75,
          },
          rng_) {
}

void AITracker::mutate_some() {
    double swb = std::max(0.0, params.mutate_sigma_wb);
    double swc = std::max(0.0, params.mutate_sigma_wc);
    double swd = std::max(0.0, params.mutate_sigma_wd);
    double seps = std::max(0.0, params.mutate_sigma_eps_logit);

    for(auto& p : particles) {
        const auto cp = canonicalize(p);
        p.wa = 1.0;
        p.wb = cp.wb_over_wa;
        p.wc = cp.wc_over_wa;
        p.wd = cp.wd_over_wa;
        p.eps = cp.eps;

        if(rng->uniform01() >= params.mutate_rate) continue;
        p.wb = mutate_ratio_with_sigma(p.wb, swb, *rng);
        p.wc = mutate_ratio_with_sigma(p.wc, swc, *rng);
        p.wd = mutate_ratio_with_sigma(p.wd, swd, *rng);
        p.eps = mutate_eps_logit_with_sigma(p.eps, seps, *rng);
    }
    normalize_weights(particles);
}

void AITracker::rejuvenate_with_mh(const GameState& st_start, int player, std::pair<int, int> observed, const std::vector<std::pair<int, int>>& cands, double temper_beta) {
    if(params.rejuvenation_steps <= 0 || cands.empty()) return;

    const double scale = std::max(0.0, params.rejuvenation_scale);
    const double swb = std::max(0.0, params.mutate_sigma_wb * scale);
    const double swc = std::max(0.0, params.mutate_sigma_wc * scale);
    const double swd = std::max(0.0, params.mutate_sigma_wd * scale);
    const double seps = std::max(0.0, params.mutate_sigma_eps_logit * scale);

    for(auto& p : particles) {
        const auto cp = canonicalize(p);
        p.wa = 1.0;
        p.wb = cp.wb_over_wa;
        p.wc = cp.wc_over_wa;
        p.wd = cp.wd_over_wa;
        p.eps = cp.eps;

        double like_cur = particle_move_likelihood(*cfg, st_start, player, cands, observed, p);
        for(int step = 0; step < params.rejuvenation_steps; step++) {
            Particle q = p;
            q.wb = mutate_ratio_with_sigma(q.wb, swb, *rng);
            q.wc = mutate_ratio_with_sigma(q.wc, swc, *rng);
            q.wd = mutate_ratio_with_sigma(q.wd, swd, *rng);
            q.eps = mutate_eps_logit_with_sigma(q.eps, seps, *rng);
            q.wa = 1.0;

            double like_prop = particle_move_likelihood(*cfg, st_start, player, cands, observed, q);
            double log_acc = temper_beta * (std::log(std::max(1e-12, like_prop)) - std::log(std::max(1e-12, like_cur)));
            double u = std::max(1e-12, rng->uniform01());
            if(std::log(u) < std::min(0.0, log_acc)) {
                p = q;
                like_cur = like_prop;
            }
        }
    }
}

void AITracker::update(const GameState& st_start, int player, std::pair<int, int> observed, int turn) {
    move_history.push_back({turn, observed});
    auto cands = get_candidates_for_player(*cfg, st_start, player);

    double ess_before = effective_sample_size(particles);
    double temper_beta = adaptive_temper_beta(ess_before, params.num_particles, params.resample_ess_frac, params.temper_beta_min);

    for(auto& part : particles) {
        double like = particle_move_likelihood(*cfg, st_start, player, cands, observed, part);
        part.w *= std::pow(like, temper_beta);
    }
    normalize_weights(particles);

    double ess = effective_sample_size(particles);
    if(ess < params.resample_ess_frac * params.num_particles) {
        particles = systematic_resample(particles, *rng);
        mutate_some();
        rejuvenate_with_mh(st_start, player, observed, cands, temper_beta);
        normalize_weights(particles);
    }

    (void)turn;
}

std::array<double, 5> AITracker::posterior_mean() const {
    double wb = 0, wc = 0, wd = 0, eps = 0;
    for(auto& p : particles) {
        const auto cp = canonicalize(p);
        wb += cp.wb_over_wa * p.w;
        wc += cp.wc_over_wa * p.w;
        wd += cp.wd_over_wa * p.w;
        eps += cp.eps * p.w;
    }
    auto w = reconstruct_weights_symmetric_margin(wb, wc, wd);
    return {w[0], w[1], w[2], w[3], eps};
}

std::string AITracker::debug_string() const {
    auto m = posterior_mean();
    double ess = effective_sample_size(particles);
    double wa = std::max(1e-12, m[0]);
    double rb = m[1] / wa;
    double rc = m[2] / wa;
    double rd = m[3] / wa;
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(3);
    oss << "mean wa=" << m[0] << " wb=" << m[1] << " wc=" << m[2] << " wd=" << m[3] << " (ratios " << rb << "," << rc << "," << rd << ") eps=" << m[4]
        << " ESS=" << std::setprecision(1) << ess;
    return oss.str();
}

std::vector<std::pair<std::pair<int, int>, double>> predict_ai_move_distribution_topk(
    const GameConfig& cfg, const GameState& st_start, int player, const std::vector<Particle>& particles, int top_k) {
    auto cands = get_candidates_for_player(cfg, st_start, player);
    if(cands.empty()) {
        return {{{st_start.px[player], st_start.py[player]}, 1.0}};
    }

    std::unordered_map<long long, double> mass;
    mass.reserve(cands.size() * 2);
    auto key = [&](int x, int y) -> long long { return (long long)x * 1000LL + y; };

    for(auto& part : particles) {
        const auto cp = canonicalize(part);
        std::vector<double> scores;
        scores.reserve(cands.size());
        for(auto& mv : cands) {
            int x = mv.first, y = mv.second;
            int cat = cell_category(st_start.owner, st_start.level, cfg.U, player, x, y);
            double val = (double)cfg.V[x][y];
            scores.push_back(score_of_cell_category(cat, val, cp));
        }

        int B = (int)cands.size();
        double eps = clip(cp.eps, 1e-6, 1.0 - 1e-6);
        double p_rand_each = eps / (double)B;

        double max_a = *std::max_element(scores.begin(), scores.end());
        double tol = 1e-9 * std::max(std::fabs(max_a), 1.0);

        std::vector<int> best;
        best.reserve(B);
        for(int i = 0; i < B; i++) {
            if(scores[i] >= max_a - tol) best.push_back(i);
        }
        int G = (int)best.size();
        double p_greedy_each = (1.0 - eps) / (double)G;

        std::vector<char> is_best(B, 0);
        for(int idx : best)
            is_best[idx] = 1;

        for(int i = 0; i < B; i++) {
            double p = p_rand_each + (is_best[i] ? p_greedy_each : 0.0);
            auto [x, y] = cands[i];
            mass[key(x, y)] += part.w * p;
        }
    }

    std::vector<std::pair<std::pair<int, int>, double>> items;
    items.reserve(mass.size());
    for(auto& kv : mass) {
        long long k = kv.first;
        int x = (int)(k / 1000LL);
        int y = (int)(k % 1000LL);
        items.push_back({{x, y}, kv.second});
    }
    std::sort(items.begin(), items.end(), [&](auto& a, auto& b) { return a.second > b.second; });
    if((int)items.size() > top_k) items.resize(top_k);

    double s = 0.0;
    for(auto& it : items)
        s += it.second;
    if(s <= 0.0) {
        int k = std::min(top_k, (int)cands.size());
        double uni = 1.0 / (double)k;
        std::vector<std::pair<std::pair<int, int>, double>> uniDist;
        for(int i = 0; i < k; i++)
            uniDist.push_back({cands[i], uni});
        return uniDist;
    }
    for(auto& it : items)
        it.second /= s;
    return items;
}

std::vector<std::pair<std::pair<int, int>, double>> predict_ai_move_distribution_topk_mean_params(
    const GameConfig& cfg, const GameState& st_start, int player, double wa, double wb, double wc, double wd, double eps, int top_k) {
    auto cands = get_candidates_for_player(cfg, st_start, player);
    if(cands.empty()) {
        return {{{st_start.px[player], st_start.py[player]}, 1.0}};
    }
    double scale = wa;
    if(!std::isfinite(scale) || std::fabs(scale) < 1e-12) scale = 1.0;
    CanonicalParams cp = {
        clip(wb / scale, R_LO, R_HI),
        clip(wc / scale, R_LO, R_HI),
        clip(wd / scale, R_LO, R_HI),
        clip(eps, E_LO, E_HI),
    };

    std::vector<double> scores;
    scores.reserve(cands.size());
    for(auto& mv : cands) {
        int x = mv.first, y = mv.second;
        int cat = cell_category(st_start.owner, st_start.level, cfg.U, player, x, y);
        double val = (double)cfg.V[x][y];
        scores.push_back(score_of_cell_category(cat, val, cp));
    }

    int B = (int)cands.size();
    eps = clip(cp.eps, 1e-6, 1.0 - 1e-6);
    double p_rand_each = eps / (double)B;

    double max_a = *std::max_element(scores.begin(), scores.end());
    double tol = 1e-9 * std::max(std::fabs(max_a), 1.0);

    std::vector<int> best;
    best.reserve(B);
    for(int i = 0; i < B; i++) {
        if(scores[i] >= max_a - tol) best.push_back(i);
    }
    int G = (int)best.size();
    double p_greedy_each = (1.0 - eps) / (double)G;

    std::vector<std::pair<std::pair<int, int>, double>> items;
    items.reserve(B);
    std::vector<char> is_best(B, 0);
    for(int idx : best)
        is_best[idx] = 1;
    for(int i = 0; i < B; i++) {
        double p = p_rand_each + (is_best[i] ? p_greedy_each : 0.0);
        items.push_back({cands[i], p});
    }

    std::sort(items.begin(), items.end(), [&](auto& a, auto& b) { return a.second > b.second; });
    if((int)items.size() > top_k) items.resize(top_k);

    double s = 0.0;
    for(auto& it : items)
        s += it.second;
    if(s <= 0.0) {
        int k = std::min(top_k, (int)cands.size());
        double uni = 1.0 / (double)k;
        std::vector<std::pair<std::pair<int, int>, double>> uniDist;
        for(int i = 0; i < k; i++)
            uniDist.push_back({cands[i], uni});
        return uniDist;
    }
    for(auto& it : items)
        it.second /= s;
    return items;
}

std::pair<int, int> sample_from_dist(const std::vector<std::pair<std::pair<int, int>, double>>& dist, RNG& rng) {
    double r = rng.uniform01();
    double acc = 0.0;
    for(auto& it : dist) {
        acc += it.second;
        if(r <= acc) return it.first;
    }
    return dist.back().first;
}

} // namespace ahc061
