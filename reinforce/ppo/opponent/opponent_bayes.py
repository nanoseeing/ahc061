"""`opponent_bayes` に関する相手モデル処理。"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)
_CPP_CLASS_CACHE: type | None = None
_CPP_LOAD_ERROR: Exception | None = None


W_LO = 0.3
W_HI = 1.0
E_LO = 0.1
E_HI = 0.5
R_LO = W_LO / W_HI
R_HI = W_HI / W_LO

DX = (-1, 1, 0, 0)
DY = (0, 0, -1, 1)


@dataclass
class Particle:
    """`Particle` を表すデータクラス。

    Attributes:
        wa (float): フィールド値。
        wb (float): フィールド値。
        wc (float): フィールド値。
        wd (float): フィールド値。
        eps (float): フィールド値。
        w (float): フィールド値。
    """
    wa: float
    wb: float
    wc: float
    wd: float
    eps: float
    w: float


def _clip(x: float, lo: float, hi: float) -> float:
    """内部ヘルパー: `clip` を実行する。

    Args:
        x (float): 入力テンソル。
        lo (float): lo の値。
        hi (float): hi の値。

    Returns:
        float: 計算結果。
    """
    return min(hi, max(lo, x))


def _normalize_weights(ps: list[Particle]) -> None:
    """内部ヘルパー: `normalize_weights` を実行する。

    Args:
        ps (list[Particle]): ps の値。
    """
    s = float(sum(p.w for p in ps))
    if s <= 0.0:
        uni = 1.0 / max(1, len(ps))
        for p in ps:
            p.w = uni
        return
    inv = 1.0 / s
    for p in ps:
        p.w *= inv


def _effective_sample_size(ps: Sequence[Particle]) -> float:
    """内部ヘルパー: `effective_sample_size` を実行する。

    Args:
        ps (Sequence[Particle]): ps の値。

    Returns:
        float: 計算結果。
    """
    s2 = sum(p.w * p.w for p in ps)
    if s2 <= 1e-18:
        return 0.0
    return 1.0 / s2


def _systematic_resample(ps: Sequence[Particle], rng: random.Random) -> list[Particle]:
    """内部ヘルパー: `systematic_resample` を実行する。

    Args:
        ps (Sequence[Particle]): ps の値。
        rng (random.Random): rng の値。

    Returns:
        list[Particle]: 計算結果。
    """
    n = len(ps)
    cdf: list[float] = []
    acc = 0.0
    for p in ps:
        acc += p.w
        cdf.append(acc)

    out: list[Particle] = []
    u0 = rng.random() / max(1, n)
    j = 0
    for i in range(n):
        u = u0 + i / max(1, n)
        while j < n - 1 and cdf[j] < u:
            j += 1
        src = ps[j]
        out.append(
            Particle(
                wa=src.wa,
                wb=src.wb,
                wc=src.wc,
                wd=src.wd,
                eps=src.eps,
                w=1.0 / max(1, n),
            )
        )
    return out


def _jitter(p: Particle, rng: random.Random) -> None:
    """内部ヘルパー: `jitter` を実行する。

    Args:
        p (Particle): p の値。
        rng (random.Random): rng の値。
    """
    p.wa = _clip(p.wa * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.wb = _clip(p.wb * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.wc = _clip(p.wc * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.wd = _clip(p.wd * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.eps = _clip(p.eps + rng.gauss(0.0, 0.01), E_LO, E_HI)


def _cell_category(owner: np.ndarray, level: np.ndarray, u: int, player: int, x: int, y: int) -> int:
    """内部ヘルパー: `cell_category` を実行する。

    Args:
        owner (np.ndarray): owner の値。
        level (np.ndarray): level の値。
        u (int): u の値。
        player (int): player の値。
        x (int): 入力テンソル。
        y (int): 入力テンソル。

    Returns:
        int: 計算結果。
    """
    o = int(owner[x, y])
    if o == -1:
        return 0
    if o == player:
        return 2 if int(level[x, y]) >= u else 1
    return 3 if int(level[x, y]) == 1 else 4


def _get_candidates(
    n: int,
    m: int,
    owner: np.ndarray,
    px: Sequence[int],
    py: Sequence[int],
    player: int,
) -> list[tuple[int, int]]:
    """内部ヘルパー: `get_candidates` を実行する。

    Args:
        n (int): n の値。
        m (int): m の値。
        owner (np.ndarray): owner の値。
        px (Sequence[int]): px の値。
        py (Sequence[int]): py の値。
        player (int): player の値。

    Returns:
        list[tuple[int, int]]: 計算結果。
    """
    sx, sy = int(px[player]), int(py[player])
    seen = np.zeros((n, n), dtype=np.bool_)
    q: list[tuple[int, int]] = [(sx, sy)]
    seen[sx, sy] = True
    head = 0
    reachable: list[tuple[int, int]] = []

    while head < len(q):
        x, y = q[head]
        head += 1

        ok = True
        for p in range(m):
            if p != player and int(px[p]) == x and int(py[p]) == y:
                ok = False
                break
        if ok:
            reachable.append((x, y))

        if int(owner[x, y]) == player:
            for d in range(4):
                nx, ny = x + DX[d], y + DY[d]
                if 0 <= nx < n and 0 <= ny < n and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))

    return reachable


def _ai_eval(
    values: np.ndarray,
    owner: np.ndarray,
    level: np.ndarray,
    u: int,
    player: int,
    theta: Particle,
    x: int,
    y: int,
) -> float:
    """内部ヘルパー: `ai_eval` を実行する。

    Args:
        values (np.ndarray): values の値。
        owner (np.ndarray): owner の値。
        level (np.ndarray): level の値。
        u (int): u の値。
        player (int): player の値。
        theta (Particle): theta の値。
        x (int): 入力テンソル。
        y (int): 入力テンソル。

    Returns:
        float: 計算結果。
    """
    cat = _cell_category(owner, level, u, player, x, y)
    val = float(values[x, y])
    if cat == 0:
        return val * theta.wa
    if cat == 1:
        return val * theta.wb
    if cat == 2:
        return 0.0
    if cat == 3:
        return val * theta.wc
    return val * theta.wd


def _likelihood_observed_move(
    values: np.ndarray,
    owner: np.ndarray,
    level: np.ndarray,
    u: int,
    player: int,
    cands: Sequence[tuple[int, int]],
    observed: tuple[int, int],
    theta: Particle,
) -> float:
    """内部ヘルパー: `likelihood_observed_move` を実行する。

    Args:
        values (np.ndarray): values の値。
        owner (np.ndarray): owner の値。
        level (np.ndarray): level の値。
        u (int): u の値。
        player (int): player の値。
        cands (Sequence[tuple[int, int]]): cands の値。
        observed (tuple[int, int]): observed の値。
        theta (Particle): theta の値。

    Returns:
        float: 計算結果。
    """
    if not cands:
        return 1e-12

    scores: list[float] = []
    obs_idx = -1
    for i, (x, y) in enumerate(cands):
        if (x, y) == observed:
            obs_idx = i
        scores.append(_ai_eval(values, owner, level, u, player, theta, x, y))

    if obs_idx < 0:
        return 1e-12

    b = len(cands)
    eps = _clip(theta.eps, 1e-6, 1.0 - 1e-6)
    p_rand = eps / b

    max_score = max(scores)
    tol = 1e-9 * max(abs(max_score), 1.0)
    best = [i for i, s in enumerate(scores) if s >= max_score - tol]
    in_best = obs_idx in set(best)
    p_greedy = ((1.0 - eps) / max(1, len(best))) if in_best else 0.0
    return max(1e-12, p_rand + p_greedy)


class OpponentBayesEstimator:
    """`OpponentBayesEstimator` を表すクラス。"""

    def __init__(
        self,
        n: int,
        m: int,
        u: int,
        *,
        num_particles: int = 128,
        resample_ess_frac: float = 0.55,
        seed: int = 0,
    ) -> None:
        """インスタンスを初期化する。

        Args:
            n (int): n の値。
            m (int): m の値。
            u (int): u の値。
            num_particles (int): num_particles の値。
            resample_ess_frac (float): resample_ess_frac の値。
            seed (int): 乱数シード。
        """
        self.n = int(n)
        self.m = int(m)
        self.u = int(u)
        self.num_particles = int(max(8, num_particles))
        self.resample_ess_frac = float(_clip(resample_ess_frac, 0.05, 0.95))
        self.rng = random.Random(int(seed))

        self.particles: dict[int, list[Particle]] = {}
        for p in range(1, self.m):
            self.particles[p] = self._sample_prior_particles(self.num_particles)

    def _sample_prior_particles(self, k: int) -> list[Particle]:
        """内部ヘルパー: `sample_prior_particles` を実行する。

        Args:
            k (int): k の値。

        Returns:
            list[Particle]: 計算結果。
        """
        out: list[Particle] = []
        for _ in range(k):
            out.append(
                Particle(
                    wa=self.rng.uniform(W_LO, W_HI),
                    wb=self.rng.uniform(W_LO, W_HI),
                    wc=self.rng.uniform(W_LO, W_HI),
                    wd=self.rng.uniform(W_LO, W_HI),
                    eps=self.rng.uniform(E_LO, E_HI),
                    w=1.0 / k,
                )
            )
        return out

    def update(
        self,
        *,
        values: np.ndarray,
        owner_before: np.ndarray,
        level_before: np.ndarray,
        observed_selected: Sequence[tuple[int, int]],
        observed_candidates: Sequence[Sequence[tuple[int, int]]],
    ) -> None:
        """`update` を実行する。

        Args:
            values (np.ndarray): values の値。
            owner_before (np.ndarray): owner_before の値。
            level_before (np.ndarray): level_before の値。
            observed_selected (Sequence[tuple[int, int]]): observed_selected の値。
            observed_candidates (Sequence[Sequence[tuple[int, int]]]): observed_candidates の値。
        """
        if len(observed_candidates) < self.m:
            raise ValueError(f"observed_candidates length must be >= m ({self.m})")
        for p in range(1, self.m):
            raw = observed_candidates[p]
            cands = [(int(x), int(y)) for (x, y) in raw]
            obs = observed_selected[p]
            ps = self.particles[p]

            for pt in ps:
                like = _likelihood_observed_move(
                    values,
                    owner_before,
                    level_before,
                    self.u,
                    p,
                    cands,
                    obs,
                    pt,
                )
                pt.w *= like

            _normalize_weights(ps)
            ess = _effective_sample_size(ps)
            if ess < self.resample_ess_frac * len(ps):
                ps = _systematic_resample(ps, self.rng)
                for pt in ps:
                    _jitter(pt, self.rng)
                _normalize_weights(ps)
                self.particles[p] = ps

    def posterior_mean_raw(self, player: int) -> tuple[float, float, float, float, float]:
        """`posterior_mean_raw` を実行する。

        Args:
            player (int): player の値。

        Returns:
            tuple[float, float, float, float, float]: 計算結果。
        """
        if player <= 0 or player >= self.m:
            raise ValueError(f"invalid player id: {player}")
        ps = self.particles[player]
        wa = sum(p.wa * p.w for p in ps)
        wb = sum(p.wb * p.w for p in ps)
        wc = sum(p.wc * p.w for p in ps)
        wd = sum(p.wd * p.w for p in ps)
        eps = sum(p.eps * p.w for p in ps)
        return wa, wb, wc, wd, eps

    def posterior_mean_ratio(self, player: int) -> tuple[float, float, float, float]:
        """`posterior_mean_ratio` を実行する。

        Args:
            player (int): player の値。

        Returns:
            tuple[float, float, float, float]: 計算結果。
        """
        wa, wb, wc, wd, eps = self.posterior_mean_raw(player)
        if not math.isfinite(wa) or abs(wa) < 1e-12:
            wa = 1.0
        rb = _clip(wb / wa, R_LO, R_HI)
        rc = _clip(wc / wa, R_LO, R_HI)
        rd = _clip(wd / wa, R_LO, R_HI)
        e = _clip(eps, E_LO, E_HI)
        return rb, rc, rd, e

    def posterior_feature_vector(self, *, max_enemies: int = 7, normalize: bool = True) -> np.ndarray:
        """`posterior_feature_vector` を実行する。

        Args:
            max_enemies (int): max_enemies の値。
            normalize (bool): 有効化フラグ。

        Returns:
            np.ndarray: 計算結果。
        """
        feat = np.zeros((max_enemies * 4,), dtype=np.float32)

        def norm_ratio(v: float) -> float:
            """`norm_ratio` を実行する。

            Args:
                v (float): v の値。

            Returns:
                float: 計算結果。
            """
            return float((v - R_LO) / max(1e-12, (R_HI - R_LO)))

        def norm_eps(v: float) -> float:
            """`norm_eps` を実行する。

            Args:
                v (float): v の値。

            Returns:
                float: 計算結果。
            """
            return float((v - E_LO) / max(1e-12, (E_HI - E_LO)))

        for ei in range(max_enemies):
            p = ei + 1
            off = ei * 4
            if p < self.m:
                rb, rc, rd, eps = self.posterior_mean_ratio(p)
                if normalize:
                    feat[off + 0] = np.float32(_clip(norm_ratio(rb), 0.0, 1.0))
                    feat[off + 1] = np.float32(_clip(norm_ratio(rc), 0.0, 1.0))
                    feat[off + 2] = np.float32(_clip(norm_ratio(rd), 0.0, 1.0))
                    feat[off + 3] = np.float32(_clip(norm_eps(eps), 0.0, 1.0))
                else:
                    feat[off + 0] = np.float32(rb)
                    feat[off + 1] = np.float32(rc)
                    feat[off + 2] = np.float32(rd)
                    feat[off + 3] = np.float32(eps)
            else:
                feat[off : off + 4] = 0.0
        return feat


def _normalize_backend_name(backend: str) -> str:
    """内部ヘルパー: `normalize_backend_name` を実行する。

    Args:
        backend (str): backend の値。

    Returns:
        str: 計算結果。
    """
    b = str(backend).strip().lower()
    if not b:
        b = "auto"
    if b == "python":
        raise ValueError(
            "bayes_backend='python' is no longer supported; "
            "use bayes_backend='cpp' (or 'auto' which resolves to cpp)"
        )
    if b not in ("auto", "cpp"):
        raise ValueError(f"unsupported bayes backend: {backend!r}; expected auto|cpp")
    return b


def _load_cpp_estimator_class(*, build_if_missing: bool = True) -> type:
    """内部ヘルパー: `load_cpp_estimator_class` を実行する。

    Args:
        build_if_missing (bool): 有効化フラグ。

    Returns:
        type: 計算結果。
    """
    global _CPP_CLASS_CACHE, _CPP_LOAD_ERROR
    if _CPP_CLASS_CACHE is not None:
        return _CPP_CLASS_CACHE
    if _CPP_LOAD_ERROR is not None and not build_if_missing:
        raise _CPP_LOAD_ERROR

    from . import load_cpp_backend

    try:
        mod = load_cpp_backend(build_if_missing=build_if_missing)
        cls = getattr(mod, "OpponentBayesEstimator", None)
        if cls is None:
            raise RuntimeError("cpp bayes backend loaded but OpponentBayesEstimator class is missing")
        _CPP_CLASS_CACHE = cls
        _CPP_LOAD_ERROR = None
        return cls
    except Exception as e:
        _CPP_LOAD_ERROR = e
        raise


def create_opponent_bayes_estimator(
    *,
    n: int,
    m: int,
    u: int,
    num_particles: int = 128,
    resample_ess_frac: float = 0.55,
    seed: int = 0,
    backend: str = "auto",
    build_if_missing: bool = True,
) -> Any:
    """`opponent_bayes_estimator`を作成する。

    Args:
        n (int): n の値。
        m (int): m の値。
        u (int): u の値。
        num_particles (int): num_particles の値。
        resample_ess_frac (float): resample_ess_frac の値。
        seed (int): 乱数シード。
        backend (str): backend の値。
        build_if_missing (bool): 有効化フラグ。

    Returns:
        Any: 計算結果。
    """
    b = _normalize_backend_name(backend)
    if b not in ("auto", "cpp"):
        raise ValueError(f"unsupported bayes backend: {backend!r}; expected auto|cpp")
    cls = _load_cpp_estimator_class(build_if_missing=build_if_missing)
    return cls(
        n=n,
        m=m,
        u=u,
        num_particles=num_particles,
        resample_ess_frac=resample_ess_frac,
        seed=seed,
    )
