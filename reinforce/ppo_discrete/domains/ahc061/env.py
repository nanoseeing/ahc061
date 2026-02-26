from __future__ import annotations

import math
import random
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .opponent_bayes import create_opponent_bayes_estimator


MAX_N = 10
MAX_M = 8
R_LO = 0.3 / 1.0
R_HI = 1.0 / 0.3
E_LO = 0.1
E_HI = 0.5


@dataclass
class CaseData:
    n: int
    m: int
    t: int
    u: int
    values: np.ndarray  # [n, n], int32
    start_xy: list[tuple[int, int]]
    wa: np.ndarray  # [m-1]
    wb: np.ndarray
    wc: np.ndarray
    wd: np.ndarray
    eps: np.ndarray
    r: np.ndarray  # [m-1, 2*t]


@dataclass
class RuntimeState:
    owner: np.ndarray  # [n, n], int16
    level: np.ndarray  # [n, n], int16
    px: list[int]
    py: list[int]


class AHC061LocalEnv(gym.Env):
    """AHC061 local simulator wrapped as a Gymnasium environment.

    Action:
      Discrete(100), mapped to (x, y) on a 10x10 board.

    Observation:
      `obs_mode` により次を切り替える:

      - legacy (default):
        7 channels x 100 + 21 global + 28 bayes = 749 dims

      - teacher_p0_v1_88ch:
        88 channels x 100 (globalも盤面へbroadcast) = 8800 dims
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        case_num_cases: int = 1000,
        case_seed_mode: str = "sequential",
        case_seed_start: int = 0,
        case_rng_seed: int = 1,
        case_unique_random: bool = True,
        case_fixed_m: int = 0,
        case_fixed_u: int = 0,
        case_tools_dir: str = "tools",
        case_gen_cmd: str = "",
        case_select_mode: str = "random",
        vector_env_rank: int = 0,
        vector_env_size: int = 1,
        reward_mode: str = "delta_log_ratio",
        illegal_action_penalty: float = -0.05,
        cache_cases: bool = False,
        bayes_num_particles: int = 128,
        bayes_resample_ess_frac: float = 0.55,
        bayes_backend: str = "auto",
        obs_mode: str = "legacy",
    ):
        super().__init__()
        self.case_num_cases = int(case_num_cases)
        self.case_seed_mode = str(case_seed_mode)
        self.case_seed_start = int(case_seed_start)
        self.case_rng_seed = int(case_rng_seed)
        self.case_unique_random = bool(case_unique_random)
        self.case_fixed_m = int(case_fixed_m)
        self.case_fixed_u = int(case_fixed_u)
        self.case_tools_dir = Path(case_tools_dir)
        self.case_gen_cmd = str(case_gen_cmd).strip()
        self.case_select_mode = str(case_select_mode)
        self.vector_env_rank = int(max(0, vector_env_rank))
        self.vector_env_size = int(max(1, vector_env_size))
        self.reward_mode = reward_mode
        self.illegal_action_penalty = float(illegal_action_penalty)
        self.cache_cases = bool(cache_cases)
        self.bayes_num_particles = int(max(8, bayes_num_particles))
        self.bayes_resample_ess_frac = float(bayes_resample_ess_frac)
        self.bayes_backend = str(bayes_backend)
        self.obs_mode = str(obs_mode).strip().lower() or "legacy"
        if self.obs_mode not in ("legacy", "teacher_p0_v1_88ch"):
            raise ValueError(f"unsupported obs_mode={obs_mode!r}; expected legacy|teacher_p0_v1_88ch")

        if self.case_select_mode not in ("random", "sequential"):
            raise ValueError(f"unsupported case_select_mode={self.case_select_mode}; expected random|sequential")

        self._case_seeds = self._build_case_seeds(
            num_cases=self.case_num_cases,
            seed_mode=self.case_seed_mode,
            seed_start=self.case_seed_start,
            rng_seed=self.case_rng_seed,
            unique_random=self.case_unique_random,
        )
        self._case_count = int(len(self._case_seeds))
        if self._case_count <= 0:
            raise RuntimeError("no cases available")
        self._case_cache: dict[str, CaseData] = {}
        self._gen_one_cmd = self._resolve_gen_one_cmd()

        self.action_space = gym.spaces.Discrete(MAX_N * MAX_N)
        self._bayes_feat_dim = 4 * (MAX_M - 1)
        if self.obs_mode == "teacher_p0_v1_88ch":
            self._obs_dim = 88 * MAX_N * MAX_N
        else:
            self._obs_dim = 7 * MAX_N * MAX_N + 21 + self._bayes_feat_dim
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.float32(np.finfo(np.float32).max),
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self.case: CaseData | None = None
        self.state: RuntimeState | None = None
        self.turn: int = 0
        self.done: bool = False
        self._episode_counter: int = 0
        self._last_scores: np.ndarray | None = None
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._episode_illegal_penalty: float = 0.0
        self._episode_terminal_score: float = 0.0
        self._episode_terminal_game_score: float = 0.0
        self._episode_objective_delta: float = 0.0
        self._case_path: str | None = None
        self._bayes: Any | None = None
        self._grid_x, self._grid_y = np.indices((MAX_N, MAX_N), dtype=np.int16)

    @staticmethod
    def _build_case_seeds(
        *,
        num_cases: int,
        seed_mode: str,
        seed_start: int,
        rng_seed: int,
        unique_random: bool,
    ) -> list[int]:
        if int(num_cases) <= 0:
            raise ValueError("case_num_cases must be > 0")
        if seed_mode == "sequential":
            if int(seed_start) < 0:
                raise ValueError("case_seed_start must be >= 0")
            return [int(seed_start) + i for i in range(int(num_cases))]
        if seed_mode == "random":
            rng = random.Random(int(rng_seed))
            if bool(unique_random):
                seen: set[int] = set()
                out: list[int] = []
                while len(out) < int(num_cases):
                    x = int(rng.getrandbits(64))
                    if x in seen:
                        continue
                    seen.add(x)
                    out.append(x)
                return out
            return [int(rng.getrandbits(64)) for _ in range(int(num_cases))]
        raise ValueError(f"unsupported case_seed_mode={seed_mode}; expected sequential|random")

    def _resolve_gen_one_cmd(self) -> list[str]:
        if self.case_gen_cmd:
            return shlex.split(self.case_gen_cmd)

        if not self.case_tools_dir.exists():
            raise FileNotFoundError(f"case_tools_dir not found: {self.case_tools_dir}")
        cargo_toml = self.case_tools_dir / "Cargo.toml"
        if not cargo_toml.exists():
            raise FileNotFoundError(f"not found: {cargo_toml}")

        gen_one = self.case_tools_dir / "target" / "release" / "gen_one"
        cargo_bin = shutil.which("cargo")
        if not gen_one.exists():
            if cargo_bin is None:
                raise FileNotFoundError("cargo was not found and tools/target/release/gen_one is missing")
            subprocess.run(
                [cargo_bin, "build", "-r", "--manifest-path", str(cargo_toml), "--bin", "gen_one"],
                check=True,
            )
        if gen_one.exists():
            return [str(gen_one)]
        if cargo_bin is None:
            raise FileNotFoundError("failed to resolve generator command for AHC case generation")
        return [cargo_bin, "run", "-r", "--manifest-path", str(cargo_toml), "--bin", "gen_one", "--"]

    def _generate_case_text(self, seed: int) -> str:
        cmd = list(self._gen_one_cmd) + ["--seed", str(int(seed))]
        if self.case_fixed_m > 0:
            cmd += ["--M", str(self.case_fixed_m)]
        if self.case_fixed_u > 0:
            cmd += ["--U", str(self.case_fixed_u)]
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return str(proc.stdout)

    @staticmethod
    def _parse_case_text(text: str) -> CaseData:
        toks = text.split()
        p = 0

        def get_int() -> int:
            nonlocal p
            v = int(toks[p])
            p += 1
            return v

        def get_float() -> float:
            nonlocal p
            v = float(toks[p])
            p += 1
            return v

        n = get_int()
        m = get_int()
        t = get_int()
        u = get_int()

        values = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                values[i, j] = get_int()

        start_xy: list[tuple[int, int]] = []
        for _ in range(m):
            x = get_int()
            y = get_int()
            start_xy.append((x, y))

        wa = np.zeros(m - 1, dtype=np.float64)
        wb = np.zeros(m - 1, dtype=np.float64)
        wc = np.zeros(m - 1, dtype=np.float64)
        wd = np.zeros(m - 1, dtype=np.float64)
        eps = np.zeros(m - 1, dtype=np.float64)
        for i in range(m - 1):
            wa[i] = get_float()
            wb[i] = get_float()
            wc[i] = get_float()
            wd[i] = get_float()
            eps[i] = get_float()

        r = np.zeros((m - 1, 2 * t), dtype=np.float64)
        for turn in range(t):
            for i in range(m - 1):
                r[i, 2 * turn] = get_float()
                r[i, 2 * turn + 1] = get_float()

        if p != len(toks):
            raise ValueError(f"parse token mismatch: consumed={p}, total={len(toks)}")

        return CaseData(
            n=n,
            m=m,
            t=t,
            u=u,
            values=values,
            start_xy=start_xy,
            wa=wa,
            wb=wb,
            wc=wc,
            wd=wd,
            eps=eps,
            r=r,
        )

    @staticmethod
    def _init_state(case: CaseData) -> RuntimeState:
        owner = np.full((case.n, case.n), -1, dtype=np.int16)
        level = np.zeros((case.n, case.n), dtype=np.int16)
        px = [0] * case.m
        py = [0] * case.m
        for player, (x, y) in enumerate(case.start_xy):
            px[player] = x
            py[player] = y
            owner[x, y] = player
            level[x, y] = 1
        return RuntimeState(owner=owner, level=level, px=px, py=py)

    @staticmethod
    def _cell_category(state: RuntimeState, case: CaseData, player: int, x: int, y: int) -> int:
        o = int(state.owner[x, y])
        if o == -1:
            return 0
        if o == player:
            return 2 if int(state.level[x, y]) >= case.u else 1
        return 3 if int(state.level[x, y]) == 1 else 4

    @staticmethod
    def _get_candidates(state: RuntimeState, case: CaseData, player: int) -> list[tuple[int, int]]:
        n = case.n
        m = case.m
        sx, sy = state.px[player], state.py[player]

        visited = np.zeros((n, n), dtype=np.uint8)
        q: list[tuple[int, int]] = [(sx, sy)]
        head = 0
        visited[sx, sy] = 1
        reachable: list[tuple[int, int]] = []

        while head < len(q):
            x, y = q[head]
            head += 1

            ok = True
            for i in range(m):
                if i != player and state.px[i] == x and state.py[i] == y:
                    ok = False
                    break
            if ok:
                reachable.append((x, y))

            if int(state.owner[x, y]) == player:
                for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and not visited[nx, ny]:
                        visited[nx, ny] = 1
                        q.append((nx, ny))

        return reachable

    @staticmethod
    def _decide_enemy_move_true(
        state: RuntimeState,
        case: CaseData,
        ai_idx: int,
        turn: int,
        *,
        cands: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int]:
        player = ai_idx + 1
        if cands is None:
            cands = AHC061LocalEnv._get_candidates(state, case, player)
        if not cands:
            return (state.px[player], state.py[player])

        scores: list[float] = []
        for x, y in cands:
            cat = AHC061LocalEnv._cell_category(state, case, player, x, y)
            val = float(case.values[x, y])
            if cat == 0:
                a = val * float(case.wa[ai_idx])
            elif cat == 1:
                a = val * float(case.wb[ai_idx])
            elif cat == 2:
                a = 0.0
            elif cat == 3:
                a = val * float(case.wc[ai_idx])
            else:
                a = val * float(case.wd[ai_idx])
            scores.append(a)

        r1 = float(case.r[ai_idx, 2 * turn])
        r2 = float(case.r[ai_idx, 2 * turn + 1])
        if r1 < float(case.eps[ai_idx]):
            idx = min(int(math.floor(r2 * len(cands))), len(cands) - 1)
            return cands[idx]

        mx = max(scores)
        tol = 1e-9 * max(abs(mx), 1.0)
        best = [i for i, sc in enumerate(scores) if sc >= mx - tol]
        idx = min(int(math.floor(r2 * len(best))), len(best) - 1)
        return cands[best[idx]]

    @staticmethod
    def _apply_turn(state: RuntimeState, case: CaseData, moves: list[tuple[int, int]]) -> RuntimeState:
        m = case.m
        ns = RuntimeState(
            owner=state.owner.copy(),
            level=state.level.copy(),
            px=list(state.px),
            py=list(state.py),
        )

        temp_pos = list(moves)
        move_counts: dict[tuple[int, int], int] = {}
        for mv in temp_pos:
            move_counts[mv] = move_counts.get(mv, 0) + 1

        collected = [False] * m
        for i in range(m):
            x, y = temp_pos[i]
            if move_counts[(x, y)] >= 2:
                owner = int(ns.owner[x, y])
                if i != owner:
                    collected[i] = True

        for i in range(m):
            if collected[i]:
                continue
            x, y = temp_pos[i]
            owner = int(ns.owner[x, y])
            if owner == -1:
                ns.owner[x, y] = i
                ns.level[x, y] = 1
            elif owner == i:
                if int(ns.level[x, y]) < case.u:
                    ns.level[x, y] = int(ns.level[x, y]) + 1
            else:
                ns.level[x, y] = int(ns.level[x, y]) - 1
                if int(ns.level[x, y]) == 0:
                    ns.owner[x, y] = i
                    ns.level[x, y] = 1
                else:
                    collected[i] = True

        for i in range(m):
            if collected[i]:
                temp_pos[i] = (state.px[i], state.py[i])

        for i in range(m):
            ns.px[i] = temp_pos[i][0]
            ns.py[i] = temp_pos[i][1]
        return ns

    @staticmethod
    def _score_all(state: RuntimeState, case: CaseData) -> np.ndarray:
        s = np.zeros(case.m, dtype=np.float64)
        owner_flat = state.owner.reshape(-1).astype(np.int64, copy=False)
        valid = owner_flat >= 0
        if not np.any(valid):
            return s

        owner_valid = owner_flat[valid]
        if np.any(owner_valid >= case.m):
            raise IndexError("owner index out of range in _score_all")

        weights = (case.values * state.level).reshape(-1).astype(np.float64, copy=False)
        s += np.bincount(owner_valid, weights=weights[valid], minlength=case.m)[: case.m]
        return s

    @staticmethod
    def _objective(scores: np.ndarray) -> float:
        s0 = max(1e-9, float(scores[0]))
        sa = max(1e-9, float(np.max(scores[1:])) if scores.shape[0] > 1 else 1.0)
        return math.log2(s0 / sa)

    @staticmethod
    def _game_defined_score(scores: np.ndarray) -> float:
        """AHC061 tester score: round(1e5 * log2(1 + S0 / SA))."""
        s0_i = int(max(0.0, float(scores[0])))
        sa_i = int(max(0.0, float(np.max(scores[1:]))) if scores.shape[0] > 1 else 1)
        if sa_i <= 0:
            return float(np.iinfo(np.int64).max)
        return float(round(1e5 * math.log2(1.0 + s0_i / sa_i)))

    def set_case_counter(self, counter: int) -> None:
        self._episode_counter = int(max(0, counter))

    def _choose_case_index(self) -> int:
        if self.case_select_mode == "sequential":
            idx = (self._episode_counter * self.vector_env_size + self.vector_env_rank) % self._case_count
            self._episode_counter += 1
        else:
            idx = int(self.np_random.integers(0, self._case_count))
        return int(idx)

    def _case_key(self, seed: int) -> str:
        return f"seed:{int(seed)}"

    def _load_case(self, seed: int) -> CaseData:
        key = self._case_key(seed)
        if self.cache_cases and key in self._case_cache:
            return self._case_cache[key]
        text = self._generate_case_text(seed)
        case = self._parse_case_text(text)
        if self.cache_cases:
            self._case_cache[key] = case
        return case

    def _build_action_mask(self) -> np.ndarray:
        assert self.case is not None and self.state is not None
        mask = np.zeros((MAX_N * MAX_N,), dtype=np.bool_)
        for x, y in self._get_candidates(self.state, self.case, 0):
            mask[x * MAX_N + y] = True
        return mask

    def _get_bayes_vector(self) -> np.ndarray:
        if self._bayes is None:
            return np.zeros((self._bayes_feat_dim,), dtype=np.float32)
        return self._bayes.posterior_feature_vector(max_enemies=MAX_M - 1, normalize=True)

    @staticmethod
    def _reachable_map_from_candidates(cands: list[tuple[int, int]]) -> np.ndarray:
        out = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        for x, y in cands:
            out[int(x), int(y)] = 1.0
        return out

    @staticmethod
    def _frontier_map_from_reachable(reach: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        rb = reach[:n, :n] > 0.5
        for x in range(n):
            for y in range(n):
                if rb[x, y]:
                    continue
                hit = False
                if x > 0 and rb[x - 1, y]:
                    hit = True
                elif x + 1 < n and rb[x + 1, y]:
                    hit = True
                elif y > 0 and rb[x, y - 1]:
                    hit = True
                elif y + 1 < n and rb[x, y + 1]:
                    hit = True
                if hit:
                    out[x, y] = 1.0
        return out

    def _dist_from_piece_map(self, px: int, py: int, n: int, *, clip_max: float) -> np.ndarray:
        out = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        d = np.abs(self._grid_x - int(px)) + np.abs(self._grid_y - int(py))
        out[:n, :n] = np.minimum(d[:n, :n], clip_max) / max(1e-6, clip_max)
        return out

    def _dist_to_reachable_map(self, reach: np.ndarray, n: int, *, clip_max: float) -> np.ndarray:
        out = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        pts = np.argwhere(reach[:n, :n] > 0.5)
        if pts.shape[0] <= 0:
            out[:n, :n] = 1.0
            return out
        gx = self._grid_x[:n, :n].astype(np.int16, copy=False)[:, :, None]
        gy = self._grid_y[:n, :n].astype(np.int16, copy=False)[:, :, None]
        px = pts[:, 0][None, None, :]
        py = pts[:, 1][None, None, :]
        md = np.abs(gx - px) + np.abs(gy - py)
        out[:n, :n] = np.minimum(np.min(md, axis=2), clip_max) / max(1e-6, clip_max)
        return out

    @staticmethod
    def _decode_bayes_from_normalized_row(row: np.ndarray) -> tuple[float, float, float, float]:
        rb = float(R_LO + np.clip(float(row[0]), 0.0, 1.0) * (R_HI - R_LO))
        rc = float(R_LO + np.clip(float(row[1]), 0.0, 1.0) * (R_HI - R_LO))
        rd = float(R_LO + np.clip(float(row[2]), 0.0, 1.0) * (R_HI - R_LO))
        eps = float(E_LO + np.clip(float(row[3]), 0.0, 1.0) * (E_HI - E_LO))
        return rb, rc, rd, eps

    def _enemy_next_prob_map(
        self,
        *,
        case: CaseData,
        state: RuntimeState,
        player: int,
        cands: list[tuple[int, int]],
        bayes_norm_row: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        if len(cands) <= 0:
            return out

        rb, rc, rd, eps = self._decode_bayes_from_normalized_row(bayes_norm_row)
        scores: list[float] = []
        for x, y in cands:
            cat = self._cell_category(state, case, player, int(x), int(y))
            v = float(case.values[int(x), int(y)])
            if cat == 0:
                a = v
            elif cat == 1:
                a = v * rb
            elif cat == 2:
                a = 0.0
            elif cat == 3:
                a = v * rc
            else:
                a = v * rd
            scores.append(a)

        mx = max(scores)
        tol = 1e-9 * max(abs(mx), 1.0)
        best = [i for i, sc in enumerate(scores) if sc >= mx - tol]
        eps = float(np.clip(eps, 1e-6, 1.0 - 1e-6))
        p_rand = eps / max(1, len(cands))
        p_greedy = (1.0 - eps) / max(1, len(best))
        best_set = set(best)

        for i, (x, y) in enumerate(cands):
            p = p_rand + (p_greedy if i in best_set else 0.0)
            out[int(x), int(y)] = np.float32(p)
        return out

    def _encode_obs_legacy(self) -> np.ndarray:
        assert self.case is not None and self.state is not None and self._last_scores is not None
        case = self.case
        st = self.state
        n = case.n
        value_scale = 1000.0

        vals = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        owner_self = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        owner_enemy = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        owner_neutral = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        level_norm = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        piece_self = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        piece_enemy = np.zeros((MAX_N, MAX_N), dtype=np.float32)

        sl = (slice(0, n), slice(0, n))
        vals[sl] = case.values
        vals[sl] /= np.float32(value_scale)

        owner_view = st.owner
        owner_self[sl] = (owner_view == 0)
        owner_neutral[sl] = (owner_view == -1)
        owner_enemy[sl] = (owner_view > 0)

        level_norm[sl] = st.level
        level_norm[sl] /= np.float32(max(1, case.u))

        piece_self[st.px[0], st.py[0]] = 1.0
        if case.m > 1:
            enemy_px = np.asarray(st.px[1 : case.m], dtype=np.int64)
            enemy_py = np.asarray(st.py[1 : case.m], dtype=np.int64)
            piece_enemy[enemy_px, enemy_py] = 1.0

        board_vec = np.concatenate(
            [
                vals.reshape(-1),
                owner_self.reshape(-1),
                owner_enemy.reshape(-1),
                owner_neutral.reshape(-1),
                level_norm.reshape(-1),
                piece_self.reshape(-1),
                piece_enemy.reshape(-1),
            ]
        )

        globals_base = np.zeros((21,), dtype=np.float32)
        globals_base[0] = float(self.turn / max(1, case.t))
        globals_base[1] = float((case.m - 2) / 6.0)
        globals_base[2] = float((case.u - 1) / 4.0)
        globals_base[3] = float(st.px[0] / max(1, n - 1))
        globals_base[4] = float(st.py[0] / max(1, n - 1))

        if case.m > 1:
            den = float(max(1, n - 1))
            enemy_xy = np.zeros((MAX_M - 1, 2), dtype=np.float32)
            e = case.m - 1
            enemy_xy[:e, 0] = np.asarray(st.px[1 : case.m], dtype=np.float32) / den
            enemy_xy[:e, 1] = np.asarray(st.py[1 : case.m], dtype=np.float32) / den
            globals_base[5:19] = enemy_xy.reshape(-1)

        s_cap = float(max(1, case.u * np.sum(case.values)))
        s0 = float(self._last_scores[0])
        sa = float(np.max(self._last_scores[1:])) if case.m > 1 else 1.0
        globals_base[19] = float(np.clip(s0 / s_cap, 0.0, 1.0))
        globals_base[20] = float(np.clip(sa / s_cap, 0.0, 1.0))

        bayes_vec = self._get_bayes_vector()
        return np.concatenate([board_vec, globals_base, bayes_vec])

    def _encode_obs_teacher_p0_v1_88ch(self) -> np.ndarray:
        assert self.case is not None and self.state is not None and self._last_scores is not None
        case = self.case
        st = self.state
        n = int(case.n)
        u_den = float(max(1, case.u))
        sl = (slice(0, n), slice(0, n))

        chs: list[np.ndarray] = []

        # global-like channels broadcasted to 10x10
        chs.append(np.full((MAX_N, MAX_N), np.float32(self.turn / max(1, case.t)), dtype=np.float32))
        chs.append(np.full((MAX_N, MAX_N), np.float32((case.m - 2) / 6.0), dtype=np.float32))
        chs.append(np.full((MAX_N, MAX_N), np.float32((case.u - 1) / 4.0), dtype=np.float32))

        value_norm = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        value_norm[sl] = np.asarray(case.values, dtype=np.float32) / np.float32(1000.0)
        chs.append(value_norm)

        owner_neutral = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        owner_neutral[sl] = (st.owner == -1)
        chs.append(owner_neutral)

        # self channels
        self_level = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        self_level[sl] = (np.asarray(st.level, dtype=np.float32) / np.float32(u_den)) * (st.owner == 0)
        self_pos = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        self_pos[int(st.px[0]), int(st.py[0])] = 1.0
        self_cands = self._get_candidates(st, case, 0)
        self_reach = self._reachable_map_from_candidates(self_cands)
        self_frontier = self._frontier_map_from_reachable(self_reach, n=n)
        self_dist_from_piece = self._dist_from_piece_map(int(st.px[0]), int(st.py[0]), n=n, clip_max=18.0)
        self_dist_to_reach = self._dist_to_reachable_map(self_reach, n=n, clip_max=10.0)
        chs.extend([self_level, self_pos, self_reach, self_frontier, self_dist_from_piece, self_dist_to_reach])

        # enemy precompute
        all_cands: list[list[tuple[int, int]]] = [self._get_candidates(st, case, p) for p in range(case.m)]
        enemy_players = list(range(1, case.m))
        enemy_players.sort(key=lambda p: (-float(self._last_scores[p]), int(p)))

        bayes_norm_by_player = np.zeros((MAX_M, 4), dtype=np.float32)
        bayes_vec = self._get_bayes_vector().reshape(MAX_M - 1, 4)
        for p in range(1, min(case.m, MAX_M)):
            bayes_norm_by_player[p, :] = bayes_vec[p - 1, :]

        zero_map = np.zeros((MAX_N, MAX_N), dtype=np.float32)
        for slot in range(MAX_M - 1):
            if slot < len(enemy_players):
                p = int(enemy_players[slot])
                enemy_level = np.zeros((MAX_N, MAX_N), dtype=np.float32)
                enemy_level[sl] = (np.asarray(st.level, dtype=np.float32) / np.float32(u_den)) * (st.owner == p)

                enemy_pos = np.zeros((MAX_N, MAX_N), dtype=np.float32)
                enemy_pos[int(st.px[p]), int(st.py[p])] = 1.0

                ecands = all_cands[p]
                enemy_reach = self._reachable_map_from_candidates(ecands)
                enemy_frontier = self._frontier_map_from_reachable(enemy_reach, n=n)
                enemy_dist_from_piece = self._dist_from_piece_map(int(st.px[p]), int(st.py[p]), n=n, clip_max=18.0)
                enemy_dist_to_reach = self._dist_to_reachable_map(enemy_reach, n=n, clip_max=10.0)
                enemy_next_prob = self._enemy_next_prob_map(
                    case=case,
                    state=st,
                    player=p,
                    cands=ecands,
                    bayes_norm_row=bayes_norm_by_player[p],
                )

                bwb = np.full((MAX_N, MAX_N), bayes_norm_by_player[p, 0], dtype=np.float32)
                bwc = np.full((MAX_N, MAX_N), bayes_norm_by_player[p, 1], dtype=np.float32)
                bwd = np.full((MAX_N, MAX_N), bayes_norm_by_player[p, 2], dtype=np.float32)
                beps = np.full((MAX_N, MAX_N), bayes_norm_by_player[p, 3], dtype=np.float32)

                chs.extend(
                    [
                        enemy_level,
                        enemy_pos,
                        enemy_reach,
                        enemy_frontier,
                        enemy_dist_from_piece,
                        enemy_dist_to_reach,
                        enemy_next_prob,
                        bwb,
                        bwc,
                        bwd,
                        beps,
                    ]
                )
            else:
                chs.extend([zero_map] * 11)

        if len(chs) != 88:
            raise RuntimeError(f"teacher_p0_v1_88ch channel count mismatch: {len(chs)}")
        return np.stack(chs, axis=0).reshape(-1)

    def _encode_obs(self) -> np.ndarray:
        if self.obs_mode == "teacher_p0_v1_88ch":
            return self._encode_obs_teacher_p0_v1_88ch()
        return self._encode_obs_legacy()

    def expert_action_main_greedy(self) -> int:
        """Greedy teacher action on current state (AHC061 local simulator)."""
        assert self.case is not None and self.state is not None
        case = self.case
        st = self.state

        cands = self._get_candidates(st, case, 0)
        if not cands:
            return st.px[0] * MAX_N + st.py[0]

        enemy_moves: list[tuple[int, int]] = []
        for ai_idx in range(case.m - 1):
            enemy_moves.append(self._decide_enemy_move_true(st, case, ai_idx, self.turn))

        best_mv = cands[0]
        best_obj = -1e100
        for mv0 in cands:
            moves: list[tuple[int, int]] = [mv0] + enemy_moves
            ns = self._apply_turn(st, case, moves)
            obj = self._objective(self._score_all(ns, case))
            if obj > best_obj + 1e-12 or (abs(obj - best_obj) <= 1e-12 and mv0 < best_mv):
                best_obj = obj
                best_mv = mv0
        return best_mv[0] * MAX_N + best_mv[1]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        case_idx = self._choose_case_index()
        case_seed = int(self._case_seeds[case_idx])
        case_key = self._case_key(case_seed)
        self.case = self._load_case(case_seed)
        self.state = self._init_state(self.case)
        self.turn = 0
        self.done = False
        self._last_scores = self._score_all(self.state, self.case)
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_illegal_penalty = 0.0
        self._episode_terminal_score = 0.0
        self._episode_terminal_game_score = 0.0
        self._episode_objective_delta = 0.0
        self._case_path = case_key

        bayes_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._bayes = create_opponent_bayes_estimator(
            n=self.case.n,
            m=self.case.m,
            u=self.case.u,
            num_particles=self.bayes_num_particles,
            resample_ess_frac=self.bayes_resample_ess_frac,
            seed=bayes_seed,
            backend=self.bayes_backend,
        )

        obs = self._encode_obs()
        info = {
            "case_path": str(case_key),
            "case_index": int(case_idx),
            "case_seed": int(case_seed),
            "action_mask": self._build_action_mask(),
            "bayes_params": self._get_bayes_vector(),
        }
        return obs, info

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on terminated episode; call reset() first")
        assert self.case is not None and self.state is not None and self._last_scores is not None
        case = self.case
        st = self.state

        action = int(action)
        tx, ty = divmod(action, MAX_N)

        candidates_all: list[list[tuple[int, int]]] = [self._get_candidates(st, case, p) for p in range(case.m)]
        legal = set(candidates_all[0])
        illegal = (tx, ty) not in legal
        if illegal:
            tx, ty = st.px[0], st.py[0]

        moves: list[tuple[int, int]] = [(tx, ty)]
        for ai_idx in range(case.m - 1):
            player = ai_idx + 1
            mv = self._decide_enemy_move_true(
                st,
                case,
                ai_idx,
                self.turn,
                cands=candidates_all[player],
            )
            moves.append(mv)

        if self._bayes is not None:
            self._bayes.update(
                values=case.values,
                owner_before=st.owner,
                level_before=st.level,
                observed_selected=moves,
                observed_candidates=candidates_all,
            )

        prev_obj = self._objective(self._last_scores)
        ns = self._apply_turn(st, case, moves)
        new_scores = self._score_all(ns, case)
        new_obj = self._objective(new_scores)

        objective_delta = new_obj - prev_obj
        illegal_penalty = self.illegal_action_penalty if illegal else 0.0

        self.state = ns
        self._last_scores = new_scores
        self.turn += 1
        self._episode_length += 1

        terminated = self.turn >= case.t
        truncated = False
        self.done = bool(terminated or truncated)

        terminal_score = self._objective(new_scores) if terminated else 0.0
        terminal_game_score = self._game_defined_score(new_scores) if terminated else 0.0
        if self.reward_mode == "delta_log_ratio":
            base_reward = objective_delta
        elif self.reward_mode == "final_log_ratio":
            base_reward = terminal_score
        else:
            raise ValueError(f"unknown reward_mode: {self.reward_mode}")
        reward = float(base_reward + illegal_penalty)

        self._episode_return += reward
        self._episode_illegal_penalty += float(illegal_penalty)
        self._episode_terminal_score += float(terminal_score)
        self._episode_terminal_game_score += float(terminal_game_score)
        self._episode_objective_delta += float(objective_delta)

        obs = self._encode_obs()
        scores_pad = np.zeros((MAX_M,), dtype=np.float32)
        scores_pad[: case.m] = new_scores.astype(np.float32)
        info: dict[str, Any] = {
            "action_mask": self._build_action_mask() if not terminated else np.zeros((MAX_N * MAX_N,), dtype=np.bool_),
            "is_legal_action": not illegal,
            "scores": scores_pad,
            "objective": float(new_obj),
            "bayes_params": self._get_bayes_vector(),
            "reward_illegal_penalty": float(illegal_penalty),
            "reward_terminal_score": float(terminal_score),  # ratio-based terminal component
            "reward_terminal_score_ratio": float(terminal_score),
            "reward_terminal_game_score": float(terminal_game_score),
            "reward_objective_delta": float(objective_delta),
        }
        if terminated:
            info["episode_raw"] = {"r": float(self._episode_return), "l": int(self._episode_length)}
            info["final_scores"] = scores_pad.copy()
            info["final_objective"] = float(new_obj)
            info["final_game_score"] = float(terminal_game_score)
            info["episode_illegal_penalty"] = float(self._episode_illegal_penalty)
            info["episode_terminal_score"] = float(self._episode_terminal_score)
            info["episode_terminal_game_score"] = float(self._episode_terminal_game_score)
            info["episode_objective_delta"] = float(self._episode_objective_delta)
        return obs, float(reward), bool(terminated), bool(truncated), info
