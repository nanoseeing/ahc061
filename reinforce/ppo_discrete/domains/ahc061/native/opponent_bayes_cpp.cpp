#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double W_LO = 0.3;
constexpr double W_HI = 1.0;
constexpr double E_LO = 0.1;
constexpr double E_HI = 0.5;
constexpr double R_LO = W_LO / W_HI;
constexpr double R_HI = W_HI / W_LO;
constexpr double LIKELIHOOD_FLOOR = 1e-12;

constexpr int DX[4] = {-1, 1, 0, 0};
constexpr int DY[4] = {0, 0, -1, 1};

struct Particle {
    double wa;
    double wb;
    double wc;
    double wd;
    double eps;
    double w;
};

inline double clip(double x, double lo, double hi) {
    if(x < lo) return lo;
    if(x > hi) return hi;
    return x;
}

class OpponentBayesEstimator {
  public:
    OpponentBayesEstimator(int n, int m, int u, int num_particles = 128, double resample_ess_frac = 0.55, std::uint64_t seed = 0)
        : n_(n), m_(m), u_(u), num_particles_(std::max(8, num_particles)), resample_ess_frac_(clip(resample_ess_frac, 0.05, 0.95)), rng_(seed) {
        if(n_ <= 0) {
            throw py::value_error("n must be > 0");
        }
        if(m_ <= 0) {
            throw py::value_error("m must be > 0");
        }
        particles_.assign(static_cast<size_t>(m_), {});
        for(int p = 1; p < m_; p++) {
            particles_[static_cast<size_t>(p)] = sample_prior_particles(num_particles_);
        }
    }

    void update(
        py::array_t<int, py::array::c_style | py::array::forcecast> values,
        py::array_t<std::int16_t, py::array::c_style | py::array::forcecast> owner_before,
        py::array_t<std::int16_t, py::array::c_style | py::array::forcecast> level_before,
        py::sequence observed_selected,
        py::sequence observed_candidates_seq) {
        if(values.ndim() != 2 || owner_before.ndim() != 2 || level_before.ndim() != 2) {
            throw py::value_error("values/owner_before/level_before must be 2D arrays");
        }
        if(values.shape(0) != n_ || values.shape(1) != n_) {
            throw py::value_error("values shape mismatch");
        }
        if(owner_before.shape(0) != n_ || owner_before.shape(1) != n_) {
            throw py::value_error("owner_before shape mismatch");
        }
        if(level_before.shape(0) != n_ || level_before.shape(1) != n_) {
            throw py::value_error("level_before shape mismatch");
        }
        if(static_cast<int>(py::len(observed_selected)) < m_) {
            throw py::value_error("observed_selected length must be >= m");
        }

        std::vector<int> values_flat(static_cast<size_t>(n_ * n_));
        std::vector<int> owner_flat(static_cast<size_t>(n_ * n_));
        std::vector<int> level_flat(static_cast<size_t>(n_ * n_));

        auto v = values.unchecked<2>();
        auto o = owner_before.unchecked<2>();
        auto l = level_before.unchecked<2>();
        for(int x = 0; x < n_; x++) {
            for(int y = 0; y < n_; y++) {
                const int id = idx(x, y);
                values_flat[static_cast<size_t>(id)] = static_cast<int>(v(x, y));
                owner_flat[static_cast<size_t>(id)] = static_cast<int>(o(x, y));
                level_flat[static_cast<size_t>(id)] = static_cast<int>(l(x, y));
            }
        }

        std::vector<std::pair<int, int>> observed = parse_observed_sequence(observed_selected, m_);
        std::vector<std::vector<std::pair<int, int>>> observed_candidates = parse_candidates_sequence(observed_candidates_seq, m_);

        for(int p = 1; p < m_; p++) {
            const auto& cands = observed_candidates[static_cast<size_t>(p)];
            const auto obs = observed[static_cast<size_t>(p)];
            auto& ps = particles_[static_cast<size_t>(p)];

            for(auto& pt : ps) {
                double like = likelihood_observed_move(values_flat, owner_flat, level_flat, p, cands, obs, pt);
                pt.w *= like;
            }

            normalize_weights(ps);
            const double ess = effective_sample_size(ps);
            if(ess < resample_ess_frac_ * static_cast<double>(ps.size())) {
                ps = systematic_resample(ps);
                for(auto& pt : ps) {
                    jitter(pt);
                }
                normalize_weights(ps);
            }
        }
    }

    std::array<double, 5> posterior_mean_raw(int player) const {
        if(player <= 0 || player >= m_) {
            throw py::value_error("invalid player id");
        }
        const auto& ps = particles_[static_cast<size_t>(player)];
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
        return {wa, wb, wc, wd, eps};
    }

    std::array<double, 4> posterior_mean_ratio(int player) const {
        auto raw = posterior_mean_raw(player);
        double wa = raw[0];
        if(!std::isfinite(wa) || std::abs(wa) < 1e-12) {
            wa = 1.0;
        }
        const double rb = clip(raw[1] / wa, R_LO, R_HI);
        const double rc = clip(raw[2] / wa, R_LO, R_HI);
        const double rd = clip(raw[3] / wa, R_LO, R_HI);
        const double eps = clip(raw[4], E_LO, E_HI);
        return {rb, rc, rd, eps};
    }

    py::array_t<float> posterior_feature_vector(int max_enemies = 7, bool normalize = true) const {
        if(max_enemies < 0) {
            throw py::value_error("max_enemies must be >= 0");
        }
        py::array_t<float> out({max_enemies * 4});
        auto feat = out.mutable_unchecked<1>();

        const auto norm_ratio = [](double v) -> double {
            return (v - R_LO) / std::max(1e-12, (R_HI - R_LO));
        };
        const auto norm_eps = [](double v) -> double {
            return (v - E_LO) / std::max(1e-12, (E_HI - E_LO));
        };

        for(int ei = 0; ei < max_enemies; ei++) {
            const int p = ei + 1;
            const int off = ei * 4;
            if(p < m_) {
                auto r = posterior_mean_ratio(p);
                if(normalize) {
                    feat(off + 0) = static_cast<float>(clip(norm_ratio(r[0]), 0.0, 1.0));
                    feat(off + 1) = static_cast<float>(clip(norm_ratio(r[1]), 0.0, 1.0));
                    feat(off + 2) = static_cast<float>(clip(norm_ratio(r[2]), 0.0, 1.0));
                    feat(off + 3) = static_cast<float>(clip(norm_eps(r[3]), 0.0, 1.0));
                } else {
                    feat(off + 0) = static_cast<float>(r[0]);
                    feat(off + 1) = static_cast<float>(r[1]);
                    feat(off + 2) = static_cast<float>(r[2]);
                    feat(off + 3) = static_cast<float>(r[3]);
                }
            } else {
                feat(off + 0) = 0.0f;
                feat(off + 1) = 0.0f;
                feat(off + 2) = 0.0f;
                feat(off + 3) = 0.0f;
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

    inline int idx(int x, int y) const {
        return x * n_ + y;
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

    std::vector<std::pair<int, int>> parse_observed_sequence(const py::sequence& seq, int expected) const {
        std::vector<std::pair<int, int>> out(static_cast<size_t>(expected), {0, 0});
        if(static_cast<int>(py::len(seq)) < expected) {
            throw py::value_error("observed_selected length must be >= expected");
        }
        for(int i = 0; i < expected; i++) {
            py::handle h = seq[static_cast<size_t>(i)];
            py::sequence mv = py::reinterpret_borrow<py::sequence>(h);
            if(static_cast<int>(py::len(mv)) != 2) {
                throw py::value_error("observed_selected[i] must be length-2 sequence");
            }
            out[static_cast<size_t>(i)] = {py::cast<int>(mv[0]), py::cast<int>(mv[1])};
        }
        return out;
    }

    std::vector<std::vector<std::pair<int, int>>> parse_candidates_sequence(const py::sequence& seq, int expected) const {
        std::vector<std::vector<std::pair<int, int>>> out(static_cast<size_t>(expected));
        if(static_cast<int>(py::len(seq)) < expected) {
            throw py::value_error("observed_candidates length must be >= expected");
        }
        for(int i = 0; i < expected; i++) {
            py::handle h = seq[static_cast<size_t>(i)];
            if(h.is_none()) {
                continue;
            }
            py::sequence cands = py::reinterpret_borrow<py::sequence>(h);
            auto& dst = out[static_cast<size_t>(i)];
            dst.reserve(static_cast<size_t>(py::len(cands)));
            const py::ssize_t cands_len = py::len(cands);
            for(py::ssize_t j = 0; j < cands_len; j++) {
                py::handle mv_h = cands[static_cast<size_t>(j)];
                py::sequence mv = py::reinterpret_borrow<py::sequence>(mv_h);
                if(static_cast<int>(py::len(mv)) != 2) {
                    throw py::value_error("observed_candidates[i][j] must be length-2 sequence");
                }
                dst.push_back({py::cast<int>(mv[0]), py::cast<int>(mv[1])});
            }
        }
        return out;
    }

    std::vector<Particle> sample_prior_particles(int k) {
        std::vector<Particle> out;
        out.reserve(static_cast<size_t>(k));
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

    double uniform(double lo, double hi) {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(rng_);
    }

    static void normalize_weights(std::vector<Particle>& ps) {
        double s = 0.0;
        for(const auto& p : ps) {
            s += p.w;
        }
        if(s <= 0.0) {
            const double uni = 1.0 / std::max(1.0, static_cast<double>(ps.size()));
            for(auto& p : ps) {
                p.w = uni;
            }
            return;
        }
        const double inv = 1.0 / s;
        for(auto& p : ps) {
            p.w *= inv;
        }
    }

    static double effective_sample_size(const std::vector<Particle>& ps) {
        double s2 = 0.0;
        for(const auto& p : ps) {
            s2 += p.w * p.w;
        }
        if(s2 <= 1e-18) return 0.0;
        return 1.0 / s2;
    }

    std::vector<Particle> systematic_resample(const std::vector<Particle>& ps) {
        const int n = static_cast<int>(ps.size());
        std::vector<double> cdf(static_cast<size_t>(n), 0.0);
        double acc = 0.0;
        for(int i = 0; i < n; i++) {
            acc += ps[static_cast<size_t>(i)].w;
            cdf[static_cast<size_t>(i)] = acc;
        }

        std::vector<Particle> out;
        out.reserve(static_cast<size_t>(n));
        const double u0 = uniform01() / std::max(1, n);
        int j = 0;
        for(int i = 0; i < n; i++) {
            const double u = u0 + static_cast<double>(i) / std::max(1, n);
            while(j < n - 1 && cdf[static_cast<size_t>(j)] < u) {
                j++;
            }
            Particle src = ps[static_cast<size_t>(j)];
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
        const int o = owner[static_cast<size_t>(idx(x, y))];
        if(o == -1) return 0;
        if(o == player) return (level[static_cast<size_t>(idx(x, y))] >= u_) ? 2 : 1;
        return (level[static_cast<size_t>(idx(x, y))] == 1) ? 3 : 4;
    }

    double ai_eval(const std::vector<int>& values, const std::vector<int>& owner, const std::vector<int>& level, int player, const Particle& theta, int x, int y) const {
        const int cat = cell_category(owner, level, player, x, y);
        const double val = static_cast<double>(values[static_cast<size_t>(idx(x, y))]);
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

        std::vector<double> scores;
        scores.reserve(cands.size());
        int obs_idx = -1;
        for(size_t i = 0; i < cands.size(); i++) {
            const auto [x, y] = cands[i];
            if(x == observed.first && y == observed.second) {
                obs_idx = static_cast<int>(i);
            }
            scores.push_back(ai_eval(values, owner, level, player, theta, x, y));
        }
        if(obs_idx < 0) return LIKELIHOOD_FLOOR;

        const int b = static_cast<int>(cands.size());
        const double eps = clip(theta.eps, 1e-6, 1.0 - 1e-6);
        const double p_rand = eps / static_cast<double>(b);

        const double max_score = *std::max_element(scores.begin(), scores.end());
        const double tol = 1e-9 * std::max(std::abs(max_score), 1.0);

        int best_count = 0;
        bool in_best = false;
        for(size_t i = 0; i < scores.size(); i++) {
            if(scores[i] >= max_score - tol) {
                best_count++;
                if(static_cast<int>(i) == obs_idx) in_best = true;
            }
        }
        if(best_count <= 0) best_count = 1;
        const double p_greedy = in_best ? ((1.0 - eps) / static_cast<double>(best_count)) : 0.0;
        return std::max(LIKELIHOOD_FLOOR, p_rand + p_greedy);
    }
};

} // namespace

PYBIND11_MODULE(_opponent_bayes_cpp, m) {
    m.doc() = "C++ backend for AHC061 opponent bayes estimator";

    py::class_<OpponentBayesEstimator>(m, "OpponentBayesEstimator")
        .def(
            py::init<int, int, int, int, double, std::uint64_t>(),
            py::arg("n"),
            py::arg("m"),
            py::arg("u"),
            py::arg("num_particles") = 128,
            py::arg("resample_ess_frac") = 0.55,
            py::arg("seed") = static_cast<std::uint64_t>(0))
        .def(
            "update",
            &OpponentBayesEstimator::update,
            py::arg("values"),
            py::arg("owner_before"),
            py::arg("level_before"),
            py::arg("observed_selected"),
            py::arg("observed_candidates"))
        .def("posterior_mean_raw", &OpponentBayesEstimator::posterior_mean_raw, py::arg("player"))
        .def("posterior_mean_ratio", &OpponentBayesEstimator::posterior_mean_ratio, py::arg("player"))
        .def("posterior_feature_vector", &OpponentBayesEstimator::posterior_feature_vector, py::arg("max_enemies") = 7, py::arg("normalize") = true);
}
