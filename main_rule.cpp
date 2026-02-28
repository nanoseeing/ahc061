// Average Score          : 183,462.89
// Average Score (log10)  : 5.21287
// Average Relative Score : 97.649
// Accepted               : 1000 / 1000
// Max Execution Time     : 2,028 ms

#include <bits/stdc++.h>

#include "utils/bayes.hpp"
#include "utils/game.hpp"
#include "utils/utility.hpp"

using namespace std;
using namespace ahc061;

// ============================================================
// Objective / Hyper Parameters (edit here)
// ============================================================

struct SolverHyperParams {
    int num_particles;
    double resample_ess_frac;
    double mutate_sigma_wb;
    double mutate_sigma_wc;
    double mutate_sigma_wd;
    double mutate_sigma_eps_logit;
    double mutate_rate;
    double temper_beta_min;
    int rejuvenation_steps;
    double rejuvenation_scale;
    uint64_t rng_est_seed;
    uint64_t rng_pol_seed;
    int print_every;
};

struct CandidateMeasureStats {
    long long turns = 0;
    long long sum_my = 0;
    long long sum_enemy_total = 0; // sum over all enemy players per turn
    long long sum_all_total = 0;   // sum over all players per turn
    int my_min = INT_MAX;
    int my_max = 0;

    void add(const GameConfig& cfg, const GameState& st) {
        const int M = cfg.M;
        auto my_cands = get_candidates_for_player(cfg, st, 0);
        const int my_cnt = (int)my_cands.size();

        int all_cnt = 0;
        for(int p = 0; p < M; p++) {
            auto cands = get_candidates_for_player(cfg, st, p);
            all_cnt += (int)cands.size();
        }
        const int enemy_total = all_cnt - my_cnt;

        turns += 1;
        sum_my += my_cnt;
        sum_enemy_total += enemy_total;
        sum_all_total += all_cnt;
        my_min = min(my_min, my_cnt);
        my_max = max(my_max, my_cnt);
    }

    double avg_my() const {
        return turns > 0 ? (double)sum_my / (double)turns : 0.0;
    }
    double avg_enemy_total() const {
        return turns > 0 ? (double)sum_enemy_total / (double)turns : 0.0;
    }
    double avg_enemy_per_player(int M) const {
        return turns > 0 ? (double)sum_enemy_total / (double)(turns * max(1, M - 1)) : 0.0;
    }
    double avg_all_per_player(int M) const {
        return turns > 0 ? (double)sum_all_total / (double)(turns * max(1, M)) : 0.0;
    }
};

bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    if(v == nullptr) return false;
    string s(v);
    for(char& ch : s)
        ch = (char)tolower((unsigned char)ch);
    if(s.empty()) return false;
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

struct PolicyConfig {
    int ai_topk;
    vector<int> search_widths;
    int restrict_lv2_reinforce_turns;
    int restrict_enemy_lv2_attack_turns;
    int danger_active_until_turn;
    int danger_horizon;
    int danger_rollouts;
    double danger_decay;
    double danger_lambda;
};

struct PolicyConfigOverride {
    optional<int> ai_topk;
    optional<vector<int>> search_widths;
    optional<int> restrict_lv2_reinforce_turns;
    optional<int> restrict_enemy_lv2_attack_turns;
    optional<int> danger_active_until_turn;
    optional<int> danger_horizon;
    optional<int> danger_rollouts;
    optional<double> danger_decay;
    optional<double> danger_lambda;
};

static const SolverHyperParams kSolverHyper = {
    .num_particles = 1000,
    .resample_ess_frac = 0.55,
    .mutate_sigma_wb = 0.024,
    .mutate_sigma_wc = 0.024,
    .mutate_sigma_wd = 0.024,
    .mutate_sigma_eps_logit = 0.16,
    .mutate_rate = 0.15,
    .temper_beta_min = 0.35,
    .rejuvenation_steps = 1,
    .rejuvenation_scale = 0.75,
    .rng_est_seed = 0,
    .rng_pol_seed = 1,
    .print_every = 10,
};

static const PolicyConfig kPolicyBaseConfig = {
    .ai_topk = 12,
    .search_widths = {30, 10},
    .restrict_lv2_reinforce_turns = 20,
    .restrict_enemy_lv2_attack_turns = 20,
    .danger_active_until_turn = 85,
    .danger_horizon = 20,
    .danger_rollouts = 30,
    .danger_decay = 0.98,
    .danger_lambda = 0.40,
};

static const unordered_map<int, PolicyConfigOverride> kPolicyOverridesByM = {
    {2,
     PolicyConfigOverride{
         .restrict_enemy_lv2_attack_turns = 0,
         .danger_active_until_turn = 0,
     }},
    {3,
     PolicyConfigOverride{
         .danger_lambda = 0.50,
     }},
    {4,
     PolicyConfigOverride{
         .danger_lambda = 0.80,
     }},
    {5,
     PolicyConfigOverride{
         .danger_lambda = 0.55,
     }},
    {6,
     PolicyConfigOverride{
         .danger_lambda = 0.40,
     }},
    {7,
     PolicyConfigOverride{
         .danger_lambda = 0.30,
     }},
    {8,
     PolicyConfigOverride{
         .danger_lambda = 0.70,
     }},
};

static constexpr double kNegInfScore = -1e100;
static constexpr double kWaMinAbs = 1e-12;
static constexpr double kEpsMin = 1e-6;
static constexpr double kEpsMax = 1.0 - 1e-6;
static constexpr double kArgmaxTolScale = 1e-9;
static constexpr double kProbTieTol = 1e-15;
static constexpr double kSearchBackupQuantile = 0.95;
static constexpr uint64_t kNoiseDistDrawXor = 0x13198a2e03707344ULL;

PolicyConfig make_policy_config(int m) {
    PolicyConfig pcfg = kPolicyBaseConfig;
    auto it = kPolicyOverridesByM.find(m);
    if(it == kPolicyOverridesByM.end()) return pcfg;
    const auto& ov = it->second;

    if(ov.ai_topk) pcfg.ai_topk = *ov.ai_topk;
    if(ov.search_widths) pcfg.search_widths = *ov.search_widths;
    if(ov.restrict_lv2_reinforce_turns) pcfg.restrict_lv2_reinforce_turns = *ov.restrict_lv2_reinforce_turns;
    if(ov.restrict_enemy_lv2_attack_turns) pcfg.restrict_enemy_lv2_attack_turns = *ov.restrict_enemy_lv2_attack_turns;
    if(ov.danger_active_until_turn) pcfg.danger_active_until_turn = *ov.danger_active_until_turn;
    if(ov.danger_horizon) pcfg.danger_horizon = *ov.danger_horizon;
    if(ov.danger_rollouts) pcfg.danger_rollouts = *ov.danger_rollouts;
    if(ov.danger_decay) pcfg.danger_decay = *ov.danger_decay;
    if(ov.danger_lambda) pcfg.danger_lambda = *ov.danger_lambda;

    return pcfg;
}

// ============================================================
// Player0 policy
// ============================================================

struct Player0Policy {
    static constexpr int kMaxPlayers = 8;        // official constraint: 2 <= M <= 8
    static constexpr int kMaxChangedCells = 128; // N*N <= 100, so this is safely enough
    static constexpr int kMaxCells = 100;        // N <= 10 -> N*N <= 100
    struct FastRolloutSimulator;

    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    static double unit01_from_u64(uint64_t x) {
        constexpr double kInv2Pow53 = 1.0 / 9007199254740992.0; // 2^53
        return (double)(x >> 11) * kInv2Pow53;
    }

    const GameConfig* cfg;
    PolicyConfig pcfg;
    RNG* rng;

    Player0Policy(const GameConfig& cfg_, PolicyConfig pcfg_, RNG& rng_) : cfg(&cfg_), pcfg(std::move(pcfg_)), rng(&rng_) {
    }

    int width_at_ply(int ply) const {
        if(pcfg.search_widths.empty()) return 1;
        int idx = std::clamp(ply, 0, (int)pcfg.search_widths.size() - 1);
        return max(1, pcfg.search_widths[idx]);
    }

    static const char* mode_name() {
        return "ratio";
    }

    static int to_idx(const GameConfig& cfg, int x, int y) {
        return x * cfg.N + y;
    }

    double evaluate_objective(const vector<long long>& scores) const {
        const int M = cfg->M;
        double s0 = (double)scores[0];
        long long sa = 1;
        for(int i = 1; i < M; i++) {
            sa = max(sa, scores[i]);
        }
        return s0 / max(1.0, (double)sa);
    }

    double evaluate_objective(const vector<double>& scores) const {
        const int M = cfg->M;
        double s0 = scores[0];
        double sa = 1.0;
        for(int i = 1; i < M; i++) {
            sa = max(sa, scores[i]);
        }
        return s0 / max(1.0, sa);
    }

    double evaluate_objective(const array<double, kMaxPlayers>& scores, int M) const {
        double s0 = scores[0];
        double sa = 1.0;
        for(int i = 1; i < M; i++) {
            sa = max(sa, scores[i]);
        }
        return s0 / max(1.0, sa);
    }

    bool is_restricted_lv2_reinforce_move_now(const GameState& st_start, const pair<int, int>& mv, int turn) const {
        if(turn >= pcfg.restrict_lv2_reinforce_turns) return false;
        int x = mv.first;
        int y = mv.second;
        return st_start.owner[x][y] == 0 && st_start.level[x][y] >= 2;
    }

    bool is_restricted_enemy_lv2_attack_cell(int owner, int level, int turn) const {
        if(turn >= pcfg.restrict_enemy_lv2_attack_turns) return false;
        return owner != -1 && owner != 0 && level >= 2;
    }

    bool is_restricted_enemy_lv2_attack_move_now(const GameState& st_start, const pair<int, int>& mv, int turn) const {
        int x = mv.first;
        int y = mv.second;
        return is_restricted_enemy_lv2_attack_cell(st_start.owner[x][y], st_start.level[x][y], turn);
    }

    struct TurnUndo {
        int changed_cnt = 0;
        array<int, kMaxChangedCells> changed_idx{};
        array<int, kMaxChangedCells> old_owner{};
        array<int, kMaxChangedCells> old_level{};
        array<int, kMaxPlayers> old_px{};
        array<int, kMaxPlayers> old_py{};
    };

    struct FastRolloutSimulator {
        using U128 = unsigned __int128;

        const GameConfig* cfg;
        int N, M, U;
        vector<int> cell_value;
        vector<int> owner;
        vector<int> level;
        vector<int> px, py;
        vector<long long> scores;
        U128 valid_mask = 0;
        U128 col0_mask = 0;
        U128 col_last_mask = 0;
        array<U128, kMaxPlayers> owner_masks{};

        vector<int> touched_mark;
        int touched_token;

        FastRolloutSimulator(const GameConfig& cfg_, const GameState& st, const vector<long long>& start_scores)
            : cfg(&cfg_), N(cfg_.N), M(cfg_.M), U(cfg_.U), touched_token(1) {
            if(M > kMaxPlayers) {
                throw runtime_error("M exceeds fixed-size scratch capacity");
            }
            int NN = N * N;
            valid_mask = (NN >= 128) ? ~((U128)0) : ((((U128)1) << NN) - 1);
            col0_mask = 0;
            col_last_mask = 0;
            for(int x = 0; x < N; x++) {
                col0_mask |= (((U128)1) << (x * N));
                col_last_mask |= (((U128)1) << (x * N + (N - 1)));
            }

            cell_value.assign(NN, 0);
            for(int x = 0; x < N; x++) {
                for(int y = 0; y < N; y++) {
                    cell_value[x * N + y] = cfg->V[x][y];
                }
            }

            owner_masks.fill(0);
            owner.assign(NN, -1);
            level.assign(NN, 0);
            for(int x = 0; x < N; x++) {
                for(int y = 0; y < N; y++) {
                    int id = x * N + y;
                    owner[id] = st.owner[x][y];
                    level[id] = st.level[x][y];
                    if(owner[id] >= 0) {
                        owner_masks[owner[id]] |= (((U128)1) << id);
                    }
                }
            }
            px = st.px;
            py = st.py;
            scores = start_scores;

            touched_mark.assign(NN, 0);
        }

        inline int idx(int x, int y) const {
            return x * N + y;
        }

        inline long long contribution(int id, int own, int lv) const {
            if(own < 0) return 0LL;
            return 1LL * cell_value[id] * lv;
        }

        inline U128 bit_of(int id) const {
            return ((U128)1) << id;
        }

        inline U128 neighbor_bits(U128 bits) const {
            U128 left = (bits & ~col0_mask) >> 1;
            U128 right = (bits & ~col_last_mask) << 1;
            U128 up = bits >> N;
            U128 down = (bits << N) & valid_mask;
            return (left | right | up | down) & valid_mask;
        }

        static int popcount_u128(U128 x) {
            uint64_t lo = (uint64_t)x;
            uint64_t hi = (uint64_t)(x >> 64);
            return __builtin_popcountll(lo) + __builtin_popcountll(hi);
        }

        void set_cell(int id, int new_owner, int new_level, TurnUndo& undo) {
            int old_owner = owner[id];
            int old_level = level[id];
            if(old_owner == new_owner && old_level == new_level) return;

            if(touched_mark[id] != touched_token) {
                touched_mark[id] = touched_token;
                if(undo.changed_cnt >= kMaxChangedCells) {
                    throw runtime_error("changed cell buffer overflow");
                }
                int k = undo.changed_cnt++;
                undo.changed_idx[k] = id;
                undo.old_owner[k] = old_owner;
                undo.old_level[k] = old_level;
            }

            if(old_owner >= 0) scores[old_owner] -= contribution(id, old_owner, old_level);
            U128 b = bit_of(id);
            if(old_owner >= 0) owner_masks[old_owner] &= ~b;
            owner[id] = new_owner;
            level[id] = new_level;
            if(new_owner >= 0) owner_masks[new_owner] |= b;
            if(new_owner >= 0) scores[new_owner] += contribution(id, new_owner, new_level);
        }

        TurnUndo apply_turn(const vector<pair<int, int>>& moves) {
            TurnUndo undo;
            for(int p = 0; p < M; p++) {
                undo.old_px[p] = px[p];
                undo.old_py[p] = py[p];
            }

            touched_token++;
            if(touched_token == INT_MAX) {
                fill(touched_mark.begin(), touched_mark.end(), 0);
                touched_token = 1;
            }

            array<int, kMaxPlayers> dest_x{};
            array<int, kMaxPlayers> dest_y{};
            array<int, kMaxPlayers> dest_id{};
            for(int p = 0; p < M; p++) {
                dest_x[p] = moves[p].first;
                dest_y[p] = moves[p].second;
                dest_id[p] = idx(dest_x[p], dest_y[p]);
            }

            array<unsigned char, kMaxPlayers> collected{};
            array<int, kMaxPlayers> order{};
            for(int p = 0; p < M; p++)
                order[p] = p;
            sort(order.begin(), order.begin() + M, [&](int a, int b) {
                if(dest_id[a] != dest_id[b]) return dest_id[a] < dest_id[b];
                return a < b;
            });

            for(int i = 0; i < M;) {
                int j = i + 1;
                while(j < M && dest_id[order[j]] == dest_id[order[i]])
                    j++;
                int group_size = j - i;
                if(group_size >= 2) {
                    int id = dest_id[order[i]];
                    int cell_owner = owner[id];
                    if(cell_owner != -1) {
                        bool owner_in = false;
                        for(int k = i; k < j; k++) {
                            if(order[k] == cell_owner) {
                                owner_in = true;
                                break;
                            }
                        }
                        if(owner_in) {
                            for(int k = i; k < j; k++) {
                                if(order[k] != cell_owner) collected[order[k]] = 1;
                            }
                        } else {
                            for(int k = i; k < j; k++) {
                                collected[order[k]] = 1;
                            }
                        }
                    } else {
                        for(int k = i; k < j; k++) {
                            collected[order[k]] = 1;
                        }
                    }
                }
                i = j;
            }

            for(int p = 0; p < M; p++) {
                if(collected[p]) continue;
                int id = dest_id[p];
                int cell_owner = owner[id];
                if(cell_owner == -1) {
                    set_cell(id, p, 1, undo);
                } else if(cell_owner == p) {
                    if(level[id] < U) set_cell(id, p, level[id] + 1, undo);
                } else {
                    int next_level = level[id] - 1;
                    if(next_level == 0) {
                        set_cell(id, p, 1, undo);
                    } else {
                        set_cell(id, cell_owner, next_level, undo);
                        collected[p] = 1;
                    }
                }
            }

            for(int p = 0; p < M; p++) {
                if(collected[p]) {
                    px[p] = undo.old_px[p];
                    py[p] = undo.old_py[p];
                } else {
                    px[p] = dest_x[p];
                    py[p] = dest_y[p];
                }
            }
            return undo;
        }

        void undo_turn(const TurnUndo& undo) {
            for(int p = 0; p < M; p++) {
                px[p] = undo.old_px[p];
                py[p] = undo.old_py[p];
            }
            for(int i = undo.changed_cnt - 1; i >= 0; i--) {
                int id = undo.changed_idx[i];
                int cur_owner = owner[id];
                int cur_level = level[id];
                if(cur_owner >= 0) scores[cur_owner] -= contribution(id, cur_owner, cur_level);

                int old_owner = undo.old_owner[i];
                int old_level = undo.old_level[i];
                U128 b = bit_of(id);
                if(cur_owner >= 0) owner_masks[cur_owner] &= ~b;
                owner[id] = old_owner;
                level[id] = old_level;
                if(old_owner >= 0) owner_masks[old_owner] |= b;
                if(old_owner >= 0) scores[old_owner] += contribution(id, old_owner, old_level);
            }
        }

        void get_candidates(int player, vector<pair<int, int>>& out) {
            out.clear();
            int sx = px[player], sy = py[player];
            int s = idx(sx, sy);
            U128 start_bit = bit_of(s);

            U128 own = owner_masks[player];
            U128 reachable = start_bit & own;
            U128 cand_bits = 0;
            if(reachable == 0) {
                cand_bits = start_bit;
            } else {
                while(true) {
                    U128 next = reachable | (neighbor_bits(reachable) & own);
                    if(next == reachable) break;
                    reachable = next;
                }
                cand_bits = (reachable | neighbor_bits(reachable)) & valid_mask;
            }

            U128 occupied_other = 0;
            for(int p = 0; p < M; p++) {
                if(p == player) continue;
                occupied_other |= bit_of(idx(px[p], py[p]));
            }
            cand_bits &= ~occupied_other;

            if(cand_bits == 0) cand_bits = start_bit;

            out.reserve(popcount_u128(cand_bits));
            uint64_t lo = (uint64_t)cand_bits;
            while(lo) {
                int b = __builtin_ctzll(lo);
                int id = b;
                out.push_back({id / N, id % N});
                lo &= lo - 1;
            }
            uint64_t hi = (uint64_t)(cand_bits >> 64);
            while(hi) {
                int b = __builtin_ctzll(hi);
                int id = 64 + b;
                if(id < N * N) out.push_back({id / N, id % N});
                hi &= hi - 1;
            }
            if(out.empty()) out.push_back({sx, sy});
        }
    };

    pair<int, int> sample_from_dist_with_noise(const vector<pair<pair<int, int>, double>>& dist, uint64_t noise) const {
        if(dist.empty()) return {0, 0};
        double u = unit01_from_u64(noise);
        double acc = 0.0;
        for(const auto& it : dist) {
            acc += it.second;
            if(u <= acc + kProbTieTol) return it.first;
        }
        return dist.back().first;
    }

    void build_enemy_move_distribution_fast(FastRolloutSimulator& sim, int player, const array<double, 5>& prm, vector<pair<int, int>>& cands_buf,
                                            vector<pair<pair<int, int>, double>>& out_dist) const {
        constexpr double R_LO = W_LO / W_HI;
        constexpr double R_HI = W_HI / W_LO;
        struct WeightedMove {
            pair<int, int> mv;
            double p;
        };

        out_dist.clear();

        sim.get_candidates(player, cands_buf);
        // get_candidates() already emits ascending cell-id order.
        int B = (int)cands_buf.size();
        if(B <= 1) {
            out_dist.push_back({cands_buf[0], 1.0});
            return;
        }
        assert(B <= kMaxChangedCells);

        double wa = prm[0];
        if(!isfinite(wa) || fabs(wa) < kWaMinAbs) wa = 1.0;
        double rb = clip(prm[1] / wa, R_LO, R_HI);
        double rc = clip(prm[2] / wa, R_LO, R_HI);
        double rd = clip(prm[3] / wa, R_LO, R_HI);

        array<double, kMaxChangedCells> scores{};
        double max_a = kNegInfScore;
        for(int i = 0; i < B; i++) {
            int x = cands_buf[i].first, y = cands_buf[i].second;
            int id = sim.idx(x, y);
            int cat;
            if(sim.owner[id] == -1)
                cat = 0;
            else if(sim.owner[id] == player)
                cat = (sim.level[id] >= sim.U ? 2 : 1);
            else
                cat = (sim.level[id] == 1 ? 3 : 4);

            double val = (double)sim.cell_value[id];
            double a = 0.0;
            if(cat == 0)
                a = val;
            else if(cat == 1)
                a = val * rb;
            else if(cat == 2)
                a = 0.0;
            else if(cat == 3)
                a = val * rc;
            else
                a = val * rd;
            scores[i] = a;
            max_a = max(max_a, a);
        }

        double eps = clip(prm[4], kEpsMin, kEpsMax);
        double p_rand_each = eps / (double)B;

        double tol = kArgmaxTolScale * max(fabs(max_a), 1.0);
        int G = 0;
        for(int i = 0; i < B; i++) {
            if(scores[i] >= max_a - tol) G++;
        }
        double p_greedy_each = (1.0 - eps) / (double)G;

        array<WeightedMove, kMaxChangedCells> items{};
        for(int i = 0; i < B; i++) {
            bool is_best = (scores[i] >= max_a - tol);
            double p = p_rand_each + (is_best ? p_greedy_each : 0.0);
            items[i] = {cands_buf[i], p};
        }

        sort(items.begin(), items.begin() + B, [&](const WeightedMove& a, const WeightedMove& b) {
            if(fabs(a.p - b.p) > kProbTieTol) return a.p > b.p;
            return a.mv < b.mv;
        });
        int keep = min(pcfg.ai_topk, B);

        double s = 0.0;
        for(int i = 0; i < keep; i++)
            s += items[i].p;
        if(s <= 0.0) {
            int k = min(pcfg.ai_topk, B);
            if(k <= 0) {
                out_dist.push_back({cands_buf[0], 1.0});
                return;
            }
            double uni_p = 1.0 / (double)k;
            out_dist.reserve(k);
            for(int i = 0; i < k; i++) {
                out_dist.push_back({cands_buf[i], uni_p});
            }
            return;
        }

        out_dist.reserve(keep);
        double inv_s = 1.0 / s;
        for(int i = 0; i < keep; i++) {
            out_dist.push_back({items[i].mv, items[i].p * inv_s});
        }
    }

    pair<int, int> sample_enemy_move_fast(FastRolloutSimulator& sim, int player, const array<double, 5>& prm, vector<pair<int, int>>& cands_buf,
                                          vector<pair<pair<int, int>, double>>& dist_buf, uint64_t base_noise) const {
        build_enemy_move_distribution_fast(sim, player, prm, cands_buf, dist_buf);
        if(dist_buf.empty()) return {sim.px[player], sim.py[player]};
        uint64_t draw_noise = splitmix64(base_noise ^ kNoiseDistDrawXor);
        return sample_from_dist_with_noise(dist_buf, draw_noise);
    }

    pair<int, int> sample_enemy_move_fast(FastRolloutSimulator& sim, int player, const array<double, 5>& prm, vector<pair<int, int>>& cands_buf,
                                          vector<pair<pair<int, int>, double>>& dist_buf) const {
        return sample_enemy_move_fast(sim, player, prm, cands_buf, dist_buf, rng->next_u64());
    }

    static void apply_expected_cell_delta(array<double, kMaxPlayers>& expected_scores, int actor, int cell_owner, int cell_level, int cell_value, int max_level,
                                          double exec_prob) {
        if(exec_prob <= 0.0) return;
        double v = (double)cell_value * exec_prob;

        if(cell_owner == -1) {
            expected_scores[actor] += v;
            return;
        }
        if(cell_owner == actor) {
            if(cell_level < max_level) expected_scores[actor] += v;
            return;
        }
        if(cell_level == 1) {
            expected_scores[cell_owner] -= v;
            expected_scores[actor] += v;
        } else {
            expected_scores[cell_owner] -= v;
        }
    }

    // Same result as sort+linear interpolation, but avoids full sort using nth_element.
    static double quantile_linear(vector<double> vals, double q) {
        if(vals.empty()) return kNegInfScore;
        q = std::clamp(q, 0.0, 1.0);
        const int n = (int)vals.size();
        double pos = q * (double)(n - 1);
        int k0 = (int)floor(pos);
        int k1 = (int)ceil(pos);
        if(k0 < 0) k0 = 0;
        if(k1 >= n) k1 = n - 1;

        auto nth_get = [&](int k) -> double {
            nth_element(vals.begin(), vals.begin() + k, vals.end());
            return vals[k];
        };

        double v0 = nth_get(k0);
        if(k0 == k1) return v0;
        double v1 = nth_get(k1);

        double t = pos - (double)k0;
        return v0 * (1.0 - t) + v1 * t;
    }

    struct EnemyNodeModel {
        vector<vector<double>> prob_by_enemy;

        // Fast products for O(NN*E) expected evaluation:
        // prod_none[c] = Π_i (1 - q_i[c])
        vector<double> prod_none;
        // prod_except[i][c] = Π_{j≠i} (1 - q_j[c])
        vector<vector<double>> prod_except;

        array<pair<int, int>, kMaxPlayers - 1> argmax_enemy_moves{};
    };

    EnemyNodeModel build_enemy_node_model(FastRolloutSimulator& sim, const vector<array<double, 5>>& enemy_post_mean, vector<pair<int, int>>& cands_buf) const {
        EnemyNodeModel model;
        const int enemy_count = sim.M - 1;
        const int NN = sim.N * sim.N;
        assert(NN <= kMaxCells);
        model.prob_by_enemy.assign(enemy_count, vector<double>(NN, 0.0));
        vector<pair<pair<int, int>, double>> dist_buf;

        for(int ei = 0; ei < enemy_count; ei++) {
            build_enemy_move_distribution_fast(sim, ei + 1, enemy_post_mean[ei], cands_buf, dist_buf);
            if(dist_buf.empty()) {
                dist_buf.push_back({{sim.px[ei + 1], sim.py[ei + 1]}, 1.0});
            }
            model.argmax_enemy_moves[ei] = dist_buf[0].first;
            for(const auto& it : dist_buf) {
                int id = sim.idx(it.first.first, it.first.second);
                model.prob_by_enemy[ei][id] += it.second;
            }
        }

        // Precompute per-cell products: prod_none, prod_except (prefix/suffix, robust even if q=1).
        model.prod_none.assign(NN, 1.0);
        model.prod_except.assign(enemy_count, vector<double>(NN, 1.0));

        // For each cell c:
        //   prefix[0]=1
        //   prefix[i+1]=prefix[i]*(1-q_i[c])
        //   suffix[E]=1
        //   suffix[i]=suffix[i+1]*(1-q_i[c])
        //   prod_none = prefix[E]
        //   prod_except[i] = prefix[i]*suffix[i+1]
        vector<double> prefix(enemy_count + 1), suffix(enemy_count + 1);
        for(int c = 0; c < NN; c++) {
            prefix[0] = 1.0;
            for(int i = 0; i < enemy_count; i++) {
                double q = model.prob_by_enemy[i][c];
                double one_minus = max(0.0, 1.0 - q);
                prefix[i + 1] = prefix[i] * one_minus;
            }
            suffix[enemy_count] = 1.0;
            for(int i = enemy_count - 1; i >= 0; i--) {
                double q = model.prob_by_enemy[i][c];
                double one_minus = max(0.0, 1.0 - q);
                suffix[i] = suffix[i + 1] * one_minus;
            }

            model.prod_none[c] = prefix[enemy_count];
            for(int i = 0; i < enemy_count; i++) {
                model.prod_except[i][c] = prefix[i] * suffix[i + 1];
            }
        }

        return model;
    }

    double evaluate_move_expected_marginal_fast(const FastRolloutSimulator& sim, const EnemyNodeModel& em, const pair<int, int>& my_move) const {
        const int M = sim.M;
        const int enemy_count = M - 1;
        const int NN = sim.N * sim.N;
        const int my_id = sim.idx(my_move.first, my_move.second);

        array<double, kMaxPlayers> expected_scores{};
        for(int p = 0; p < M; p++) {
            expected_scores[p] = (double)sim.scores[p];
        }
        for(int p = M; p < kMaxPlayers; p++)
            expected_scores[p] = 0.0;

        // Self action expected contribution.
        {
            int cell_owner = sim.owner[my_id];
            int cell_level = sim.level[my_id];
            double p_exec = (cell_owner == 0) ? 1.0 : em.prod_none[my_id];
            apply_expected_cell_delta(expected_scores, 0, cell_owner, cell_level, sim.cell_value[my_id], sim.U, p_exec);
        }

        // Enemy actions expected contribution with independence approximation, but now O(NN*E):
        // p_exec (non-owner case, and not my_on_cell) = q_i[c] * prod_except[i][c]
        for(int cell_id = 0; cell_id < NN; cell_id++) {
            int cell_owner = sim.owner[cell_id];
            int cell_level = sim.level[cell_id];
            bool my_on_cell = (cell_id == my_id);

            for(int ei = 0; ei < enemy_count; ei++) {
                int actor = ei + 1;
                double q_actor = em.prob_by_enemy[ei][cell_id];
                if(q_actor <= 0.0) continue;

                double p_exec = 0.0;
                if(cell_owner == actor) {
                    // Owner can execute even if others come (per current approximation).
                    p_exec = q_actor;
                } else if(!my_on_cell) {
                    // Approx: executes if no other enemy comes to the same cell.
                    p_exec = q_actor * em.prod_except[ei][cell_id];
                }
                apply_expected_cell_delta(expected_scores, actor, cell_owner, cell_level, sim.cell_value[cell_id], sim.U, p_exec);
            }
        }

        return evaluate_objective(expected_scores, M);
    }

    vector<pair<int, int>> select_top_self_moves_from_marginal(FastRolloutSimulator& sim, const EnemyNodeModel& em, vector<pair<int, int>>& my_cands_buf,
                                                               int branch_width, int abs_turn) const {
        sim.get_candidates(0, my_cands_buf);
        if(my_cands_buf.empty()) return {{sim.px[0], sim.py[0]}};
        branch_width = max(1, branch_width);
        if(abs_turn < pcfg.restrict_lv2_reinforce_turns || abs_turn < pcfg.restrict_enemy_lv2_attack_turns) {
            vector<pair<int, int>> filtered;
            filtered.reserve(my_cands_buf.size());
            for(const auto& mv : my_cands_buf) {
                int id = sim.idx(mv.first, mv.second);
                bool restricted_reinforce = (abs_turn < pcfg.restrict_lv2_reinforce_turns && sim.owner[id] == 0 && sim.level[id] >= 2);
                bool restricted_enemy_lv2_attack = is_restricted_enemy_lv2_attack_cell(sim.owner[id], sim.level[id], abs_turn);
                if(!restricted_reinforce && !restricted_enemy_lv2_attack) filtered.push_back(mv);
            }
            if(!filtered.empty()) my_cands_buf.swap(filtered);
        }

        struct ScoredMove {
            double score;
            pair<int, int> mv;
        };
        vector<ScoredMove> scored;
        scored.reserve(my_cands_buf.size());
        for(const auto& mv : my_cands_buf) {
            double v = evaluate_move_expected_marginal_fast(sim, em, mv);
            scored.push_back({v, mv});
        }
        sort(scored.begin(), scored.end(), [](const ScoredMove& a, const ScoredMove& b) {
            if(a.score != b.score) return a.score > b.score;
            return a.mv < b.mv;
        });
        if((int)scored.size() > branch_width) scored.resize(branch_width);

        vector<pair<int, int>> out;
        out.reserve(scored.size());
        for(const auto& it : scored)
            out.push_back(it.mv);
        return out;
    }

    double evaluate_leaf_value(FastRolloutSimulator& sim, const vector<array<double, 5>>& enemy_post_mean, vector<pair<int, int>>& cands_buf,
                               vector<pair<int, int>>& my_cands_buf, int abs_turn) const {
        const EnemyNodeModel em = build_enemy_node_model(sim, enemy_post_mean, cands_buf);
        auto my_moves = select_top_self_moves_from_marginal(sim, em, my_cands_buf, 1, abs_turn);
        pair<int, int> my_mv = my_moves.empty() ? pair<int, int>{sim.px[0], sim.py[0]} : my_moves[0];
        return evaluate_move_expected_marginal_fast(sim, em, my_mv);
    }

    double search_argmax_enemy(FastRolloutSimulator& sim, int ply_idx, const vector<array<double, 5>>& enemy_post_mean, vector<pair<int, int>>& cands_buf,
                               vector<pair<int, int>>& my_cands_buf, vector<pair<int, int>>& moves_buf, int abs_turn) const {
        if(ply_idx >= (int)pcfg.search_widths.size()) {
            return evaluate_leaf_value(sim, enemy_post_mean, cands_buf, my_cands_buf, abs_turn);
        }

        const EnemyNodeModel em = build_enemy_node_model(sim, enemy_post_mean, cands_buf);
        auto my_moves = select_top_self_moves_from_marginal(sim, em, my_cands_buf, width_at_ply(ply_idx), abs_turn);
        if(my_moves.empty()) {
            return evaluate_objective(sim.scores);
        }

        const int enemy_count = sim.M - 1;
        for(int ei = 0; ei < enemy_count; ei++) {
            moves_buf[ei + 1] = em.argmax_enemy_moves[ei];
        }

        vector<double> branch_vals;
        branch_vals.reserve(my_moves.size());
        for(const auto& my_mv : my_moves) {
            moves_buf[0] = my_mv;
            auto undo = sim.apply_turn(moves_buf);
            double v = search_argmax_enemy(sim, ply_idx + 1, enemy_post_mean, cands_buf, my_cands_buf, moves_buf, abs_turn + 1);
            sim.undo_turn(undo);
            branch_vals.push_back(v);
        }
        return quantile_linear(std::move(branch_vals), kSearchBackupQuantile);
    }

    vector<double> compute_danger_map(const GameState& st_start, const vector<AITracker>& ai_trackers) {
        const int N = cfg->N;
        const int M = cfg->M;
        const int horizon = max(1, pcfg.danger_horizon);
        const int rollouts = max(1, pcfg.danger_rollouts);
        const double decay = clip(pcfg.danger_decay, 0.0, 1.0);

        vector<array<double, 5>> post_mean(M - 1);
        for(int i = 0; i < M - 1; i++) {
            post_mean[i] = ai_trackers[i].posterior_mean();
        }

        vector<double> danger(N * N, 0.0);
        auto start_scores = score_all_players(*cfg, st_start);
        FastRolloutSimulator sim(*cfg, st_start, start_scores);
        vector<pair<int, int>> moves(M);
        vector<pair<int, int>> enemy_moves(M - 1);
        vector<pair<int, int>> cands_buf;
        vector<pair<pair<int, int>, double>> dist_buf;
        vector<TurnUndo> history;
        history.reserve(horizon);

        for(int r = 0; r < rollouts; r++) {
            history.clear();
            double w = 1.0;
            for(int step = 0; step < horizon; step++) {
                moves[0] = {sim.px[0], sim.py[0]};
                for(int p = 1; p < M; p++) {
                    auto prm = post_mean[p - 1];
                    enemy_moves[p - 1] = sample_enemy_move_fast(sim, p, prm, cands_buf, dist_buf);
                    moves[p] = enemy_moves[p - 1];
                }
                history.push_back(sim.apply_turn(moves));
                for(int p = 1; p < M; p++) {
                    danger[to_idx(*cfg, sim.px[p], sim.py[p])] += w;
                }
                w *= decay;
            }
            for(int i = (int)history.size() - 1; i >= 0; i--) {
                sim.undo_turn(history[i]);
            }
        }

        double norm = (double)rollouts * (double)max(1, M - 1);
        for(double& v : danger)
            v /= norm;
        return danger;
    }

    pair<int, int> choose_move(const GameState& st_start, const vector<AITracker>& ai_trackers, int turn) {
        const int M = cfg->M;
        auto my_cands = get_candidates_for_player(*cfg, st_start, 0);
        if(my_cands.empty()) return {st_start.px[0], st_start.py[0]};
        if(turn < pcfg.restrict_lv2_reinforce_turns || turn < pcfg.restrict_enemy_lv2_attack_turns) {
            vector<pair<int, int>> filtered;
            filtered.reserve(my_cands.size());
            for(const auto& mv : my_cands) {
                bool restricted_reinforce = is_restricted_lv2_reinforce_move_now(st_start, mv, turn);
                bool restricted_enemy_lv2_attack = is_restricted_enemy_lv2_attack_move_now(st_start, mv, turn);
                if(!restricted_reinforce && !restricted_enemy_lv2_attack) filtered.push_back(mv);
            }
            if(!filtered.empty()) my_cands.swap(filtered);
        }

        const auto start_scores = score_all_players(*cfg, st_start);

        vector<array<double, 5>> enemy_post_mean;
        enemy_post_mean.reserve(M - 1);
        for(int p = 1; p < M; p++) {
            enemy_post_mean.push_back(ai_trackers[p - 1].posterior_mean());
        }

        vector<double> danger_map;
        const vector<double>* danger_map_ptr = nullptr;
        if(turn < pcfg.danger_active_until_turn) {
            danger_map = compute_danger_map(st_start, ai_trackers);
            danger_map_ptr = &danger_map;
        }

        pair<int, int> best_mv = my_cands[0];
        double best_val = kNegInfScore;

        FastRolloutSimulator sim(*cfg, st_start, start_scores);
        const int enemy_count = M - 1;
        vector<pair<int, int>> cands_buf;
        vector<pair<int, int>> my_cands_buf;
        vector<pair<int, int>> moves_buf(M);
        const EnemyNodeModel root_em = build_enemy_node_model(sim, enemy_post_mean, cands_buf);

        auto root_moves = select_top_self_moves_from_marginal(sim, root_em, my_cands_buf, width_at_ply(0), turn);
        if(root_moves.empty()) root_moves = my_cands;
        if(turn < pcfg.restrict_lv2_reinforce_turns || turn < pcfg.restrict_enemy_lv2_attack_turns) {
            vector<pair<int, int>> filtered_root;
            filtered_root.reserve(root_moves.size());
            for(const auto& mv : root_moves) {
                bool restricted_reinforce = is_restricted_lv2_reinforce_move_now(st_start, mv, turn);
                bool restricted_enemy_lv2_attack = is_restricted_enemy_lv2_attack_move_now(st_start, mv, turn);
                if(!restricted_reinforce && !restricted_enemy_lv2_attack) filtered_root.push_back(mv);
            }
            if(!filtered_root.empty()) root_moves.swap(filtered_root);
        }

        for(const auto& mv0 : root_moves) {
            moves_buf[0] = mv0;
            for(int ei = 0; ei < enemy_count; ei++) {
                moves_buf[ei + 1] = root_em.argmax_enemy_moves[ei];
            }
            auto undo = sim.apply_turn(moves_buf);
            double ev = search_argmax_enemy(sim, 1, enemy_post_mean, cands_buf, my_cands_buf, moves_buf, turn + 1);
            sim.undo_turn(undo);

            double danger_pen = 0.0;
            if(danger_map_ptr) {
                danger_pen = pcfg.danger_lambda * (*danger_map_ptr)[to_idx(*cfg, mv0.first, mv0.second)];
            }
            double score = ev - danger_pen;
            if(score > best_val) {
                best_val = score;
                best_mv = mv0;
            }
        }

        return best_mv;
    }
};

// ============================================================
// Main
// ============================================================

int main() {
    auto [cfg, st] = GameIO::read_initial();
    const bool cand_measure_mode = env_flag_enabled("AHC061_CAND_STATS");
    CandidateMeasureStats cand_stats;

    const PolicyConfig pcfg = make_policy_config(cfg.M);

    RNG rng_est(kSolverHyper.rng_est_seed);
    RNG rng_pol(kSolverHyper.rng_pol_seed);

    AITrackerParams tracker_params;
    tracker_params.num_particles = kSolverHyper.num_particles;
    tracker_params.resample_ess_frac = kSolverHyper.resample_ess_frac;
    tracker_params.mutate_sigma_wb = kSolverHyper.mutate_sigma_wb;
    tracker_params.mutate_sigma_wc = kSolverHyper.mutate_sigma_wc;
    tracker_params.mutate_sigma_wd = kSolverHyper.mutate_sigma_wd;
    tracker_params.mutate_sigma_eps_logit = kSolverHyper.mutate_sigma_eps_logit;
    tracker_params.mutate_rate = kSolverHyper.mutate_rate;
    tracker_params.temper_beta_min = kSolverHyper.temper_beta_min;
    tracker_params.rejuvenation_steps = kSolverHyper.rejuvenation_steps;
    tracker_params.rejuvenation_scale = kSolverHyper.rejuvenation_scale;

    vector<AITracker> ai_trackers;
    ai_trackers.reserve(cfg.M - 1);
    for(int i = 0; i < cfg.M - 1; i++) {
        ai_trackers.emplace_back(cfg, tracker_params, rng_est);
    }

    Player0Policy policy(cfg, pcfg, rng_pol);

    for(int turn = 0; turn < cfg.T; turn++) {
        GameState st_start = st.snapshot();
        if(cand_measure_mode) {
            cand_stats.add(cfg, st_start);
        }

        auto mv0 = policy.choose_move(st_start, ai_trackers, turn);
        GameIO::write_move(mv0.first, mv0.second);

        TurnResult tr = GameIO::read_turn_result(cfg, st);

        for(int p = 1; p < cfg.M; p++) {
            ai_trackers[p - 1].update(st_start, p, {tr.tx[p], tr.ty[p]}, turn);
        }

        if(!cand_measure_mode && kSolverHyper.print_every > 0 && (turn % kSolverHyper.print_every == 0 || turn == cfg.T - 1)) {
            cerr << "[turn=" << setw(3) << setfill('0') << turn << "] mode=" << Player0Policy::mode_name() << "\n";
            for(int p = 1; p < cfg.M; p++) {
                cerr << "  AI" << p << ": " << ai_trackers[p - 1].debug_string() << "\n";
            }
            cerr.flush();
        }
    }

    if(cand_measure_mode) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8);
        oss << "{";
        oss << "\"M\":" << cfg.M << ",";
        oss << "\"turns\":" << cand_stats.turns << ",";
        oss << "\"sum_my\":" << cand_stats.sum_my << ",";
        oss << "\"sum_enemy_total\":" << cand_stats.sum_enemy_total << ",";
        oss << "\"sum_all_total\":" << cand_stats.sum_all_total << ",";
        oss << "\"avg_my\":" << cand_stats.avg_my() << ",";
        oss << "\"avg_enemy_total\":" << cand_stats.avg_enemy_total() << ",";
        oss << "\"avg_enemy_per_player\":" << cand_stats.avg_enemy_per_player(cfg.M) << ",";
        oss << "\"avg_all_per_player\":" << cand_stats.avg_all_per_player(cfg.M) << ",";
        oss << "\"my_min\":" << (cand_stats.turns > 0 ? cand_stats.my_min : 0) << ",";
        oss << "\"my_max\":" << cand_stats.my_max;
        oss << "}";
        cerr << "CAND_STAT_JSON " << oss.str() << "\n";
        cerr.flush();
    }

    return 0;
}
