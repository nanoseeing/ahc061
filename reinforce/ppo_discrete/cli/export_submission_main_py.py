from __future__ import annotations

import argparse
import base64
import io
import json
import textwrap
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MAX_SOURCE_BYTES = 512 * 1024


TEMPLATE_MAIN_PY = """#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import math
import random
import sys
import zlib
from dataclasses import dataclass

import numpy as np


MODEL_META_JSON = __MODEL_META_JSON__
MODEL_BLOB_B85 = (
__MODEL_BLOB_B85_LINES__
)

DX = (-1, 1, 0, 0)
DY = (0, 0, -1, 1)
W_LO = 0.3
W_HI = 1.0
E_LO = 0.1
E_HI = 0.5
R_LO = W_LO / W_HI
R_HI = W_HI / W_LO


@dataclass
class IOConfig:
    n: int
    m: int
    t: int
    u: int
    values: np.ndarray
    start_xy: list[tuple[int, int]]


@dataclass
class CaseData:
    n: int
    m: int
    t: int
    u: int
    values: np.ndarray
    start_xy: list[tuple[int, int]]
    wa: np.ndarray
    wb: np.ndarray
    wc: np.ndarray
    wd: np.ndarray
    eps: np.ndarray
    r: np.ndarray


@dataclass
class RuntimeState:
    owner: np.ndarray
    level: np.ndarray
    px: list[int]
    py: list[int]


@dataclass
class Particle:
    wa: float
    wb: float
    wc: float
    wd: float
    eps: float
    w: float

def _load_arrays() -> dict[str, np.ndarray]:
    blob = "".join(MODEL_BLOB_B85)
    raw = zlib.decompress(base64.b85decode(blob.encode("ascii")))
    out: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        for k in z.files:
            arr = z[k]
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32, copy=False)
            out[str(k)] = arr
    return out


def _clip(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _act(name: str, x: np.ndarray) -> np.ndarray:
    k = str(name).strip().lower()
    if k in ("", "tanh"):
        return np.tanh(x, dtype=np.float32)
    if k == "relu":
        return np.maximum(x, 0.0, dtype=np.float32)
    if k == "silu":
        return x / (1.0 + np.exp(-x, dtype=np.float32))
    raise ValueError(f"unsupported activation: {name}")


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (w @ x + b).astype(np.float32, copy=False)


def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int) -> np.ndarray:
    if int(pad) > 0:
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    else:
        xp = x
    kh = int(w.shape[2])
    kw = int(w.shape[3])
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))
    y = np.tensordot(w, win, axes=([1, 2, 3], [0, 3, 4]))
    y = y + b[:, None, None]
    return y.astype(np.float32, copy=False)


def _layer_indices(arrays: dict[str, np.ndarray], prefix: str) -> list[int]:
    out: list[int] = []
    plen = len(prefix)
    for k in arrays.keys():
        if not (k.startswith(prefix) and k.endswith(".weight")):
            continue
        mid = k[plen : -len(".weight")]
        if mid.isdigit():
            out.append(int(mid))
    out.sort()
    return out


class StudentMPolicy:
    def __init__(self, arrays: dict[str, np.ndarray], model_kwargs: dict[str, object]) -> None:
        self.arr = arrays
        self.board_channels = int(model_kwargs.get("board_channels", 7))
        self.board_size = int(model_kwargs.get("board_size", 10))
        self.global_dim = int(model_kwargs.get("global_dim", 0))
        self.num_blocks = int(model_kwargs.get("num_blocks", 0))
        self.activation = str(model_kwargs.get("activation", "tanh")).lower()
        self.global_layers = _layer_indices(arrays, "global_mlp.")
        self.value_layers = _layer_indices(arrays, "value_head.")

    def _split_obs(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        board_dim = self.board_channels * self.board_size * self.board_size
        board = obs[:board_dim].reshape(self.board_channels, self.board_size, self.board_size)
        g = None
        if self.global_dim > 0:
            g = obs[board_dim : board_dim + self.global_dim]
        return board, g

    def _global_emb(self, g: np.ndarray | None) -> np.ndarray | None:
        if g is None:
            return None
        if not self.global_layers:
            return g.astype(np.float32, copy=False)
        x = g.astype(np.float32, copy=False)
        for idx in self.global_layers:
            x = _linear(x, self.arr[f"global_mlp.{idx}.weight"], self.arr[f"global_mlp.{idx}.bias"])
            x = _act(self.activation, x)
        return x

    def logits_and_value(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        board, g = self._split_obs(obs)
        h = _conv2d(board, self.arr["stem.0.weight"], self.arr["stem.0.bias"], pad=1)
        h = _act(self.activation, h)

        for bi in range(self.num_blocks):
            z = _conv2d(h, self.arr[f"blocks.{bi}.conv1.weight"], self.arr[f"blocks.{bi}.conv1.bias"], pad=1)
            z = _act(self.activation, z)
            z = _conv2d(z, self.arr[f"blocks.{bi}.conv2.weight"], self.arr[f"blocks.{bi}.conv2.bias"], pad=1)
            h = _act(self.activation, h + z)

        g_emb = self._global_emb(g)
        if g_emb is not None and "film.weight" in self.arr:
            film = _linear(g_emb, self.arr["film.weight"], self.arr["film.bias"])
            width = int(h.shape[0])
            gamma = film[:width].reshape(width, 1, 1)
            beta = film[width:].reshape(width, 1, 1)
            h = h * (1.0 + gamma) + beta

        logits = _conv2d(h, self.arr["policy_conv.weight"], self.arr["policy_conv.bias"], pad=0).reshape(-1)
        if g_emb is not None and "policy_global_bias.weight" in self.arr:
            logits = logits + _linear(g_emb, self.arr["policy_global_bias.weight"], self.arr["policy_global_bias.bias"])
        logits = logits.astype(np.float32, copy=False)

        v = h.mean(axis=(1, 2)).astype(np.float32, copy=False)
        if g_emb is not None:
            v = np.concatenate([v, g_emb.astype(np.float32, copy=False)], axis=0)
        if self.value_layers:
            for i, idx in enumerate(self.value_layers):
                v = _linear(v, self.arr[f"value_head.{idx}.weight"], self.arr[f"value_head.{idx}.bias"])
                if i + 1 < len(self.value_layers):
                    v = _act(self.activation, v)
            value = float(v.reshape(-1)[0])
        else:
            value = 0.0
        return logits, value


def _normalize_weights(ps: list[Particle]) -> None:
    s = float(sum(p.w for p in ps))
    if s <= 0.0:
        uni = 1.0 / max(1, len(ps))
        for p in ps:
            p.w = uni
        return
    inv = 1.0 / s
    for p in ps:
        p.w *= inv


def _effective_sample_size(ps: list[Particle]) -> float:
    s2 = sum(p.w * p.w for p in ps)
    if s2 <= 1e-18:
        return 0.0
    return 1.0 / s2


def _systematic_resample(ps: list[Particle], rng: random.Random) -> list[Particle]:
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
        out.append(Particle(src.wa, src.wb, src.wc, src.wd, src.eps, 1.0 / max(1, n)))
    return out


def _jitter(p: Particle, rng: random.Random) -> None:
    p.wa = _clip(p.wa * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.wb = _clip(p.wb * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.wc = _clip(p.wc * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.wd = _clip(p.wd * math.exp(rng.gauss(0.0, 0.04)), W_LO, W_HI)
    p.eps = _clip(p.eps + rng.gauss(0.0, 0.01), E_LO, E_HI)


def _cell_category(owner: np.ndarray, level: np.ndarray, u: int, player: int, x: int, y: int) -> int:
    o = int(owner[x, y])
    if o == -1:
        return 0
    if o == player:
        return 2 if int(level[x, y]) >= u else 1
    return 3 if int(level[x, y]) == 1 else 4


def _get_candidates(n: int, m: int, owner: np.ndarray, px: list[int], py: list[int], player: int) -> list[tuple[int, int]]:
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


def _ai_eval(values: np.ndarray, owner: np.ndarray, level: np.ndarray, u: int, player: int, theta: Particle, x: int, y: int) -> float:
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
    cands: list[tuple[int, int]],
    observed: tuple[int, int],
    theta: Particle,
) -> float:
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
    p_greedy = (1.0 - eps) / max(1, len(best)) if in_best else 0.0
    return max(1e-12, p_rand + p_greedy)


class OpponentBayesEstimator:
    def __init__(self, n: int, m: int, u: int, *, num_particles: int = 128, resample_ess_frac: float = 0.55, seed: int = 0) -> None:
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
        observed_selected: list[tuple[int, int]],
        observed_candidates: list[list[tuple[int, int]]],
    ) -> None:
        if len(observed_candidates) < self.m:
            raise ValueError(f"observed_candidates length must be >= m ({self.m})")
        for p in range(1, self.m):
            cands = observed_candidates[p]
            obs = observed_selected[p]
            ps = self.particles[p]
            for pt in ps:
                like = _likelihood_observed_move(values, owner_before, level_before, self.u, p, cands, obs, pt)
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
        ps = self.particles[player]
        wa = sum(p.wa * p.w for p in ps)
        wb = sum(p.wb * p.w for p in ps)
        wc = sum(p.wc * p.w for p in ps)
        wd = sum(p.wd * p.w for p in ps)
        eps = sum(p.eps * p.w for p in ps)
        return wa, wb, wc, wd, eps

    def posterior_mean_ratio(self, player: int) -> tuple[float, float, float, float]:
        wa, wb, wc, wd, eps = self.posterior_mean_raw(player)
        if not math.isfinite(wa) or abs(wa) < 1e-12:
            wa = 1.0
        rb = _clip(wb / wa, R_LO, R_HI)
        rc = _clip(wc / wa, R_LO, R_HI)
        rd = _clip(wd / wa, R_LO, R_HI)
        e = _clip(eps, E_LO, E_HI)
        return rb, rc, rd, e

    def posterior_feature_vector(self, *, max_enemies: int = 7, normalize: bool = True) -> np.ndarray:
        feat = np.zeros((max_enemies * 4,), dtype=np.float32)

        def norm_ratio(v: float) -> float:
            return float((v - R_LO) / max(1e-12, (R_HI - R_LO)))

        def norm_eps(v: float) -> float:
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


def _score_all(state: RuntimeState, case: CaseData) -> np.ndarray:
    s = np.zeros(case.m, dtype=np.float64)
    for i in range(case.n):
        for j in range(case.n):
            o = int(state.owner[i, j])
            if o >= 0:
                s[o] += float(case.values[i, j] * int(state.level[i, j]))
    return s


def _init_state(case: CaseData) -> RuntimeState:
    owner = np.full((case.n, case.n), -1, dtype=np.int16)
    level = np.zeros((case.n, case.n), dtype=np.int16)
    px = [0] * case.m
    py = [0] * case.m
    for p, (x, y) in enumerate(case.start_xy):
        px[p] = int(x)
        py[p] = int(y)
        owner[x, y] = p
        level[x, y] = 1
    return RuntimeState(owner=owner, level=level, px=px, py=py)


def _build_action_mask(cands: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((100,), dtype=np.bool_)
    for x, y in cands:
        mask[x * 10 + y] = True
    return mask


def _normalize_obs(obs: np.ndarray, arrays: dict[str, np.ndarray], meta: dict[str, object]) -> np.ndarray:
    if not bool(meta.get("use_vecnorm", False)):
        return obs.astype(np.float32, copy=False)
    vec = meta.get("vecnorm", {})
    if not isinstance(vec, dict):
        return obs.astype(np.float32, copy=False)
    mean = arrays.get("__vec_obs_mean")
    var = arrays.get("__vec_obs_var")
    if mean is None or var is None:
        return obs.astype(np.float32, copy=False)
    eps = float(vec.get("epsilon", 1e-8))
    clip_obs = float(vec.get("clip_obs", 10.0))
    out = (obs - mean) / np.sqrt(var + eps)
    out = np.clip(out, -clip_obs, clip_obs)
    return out.astype(np.float32, copy=False)


def _sample_from_logits(logits: np.ndarray) -> int:
    z = logits - float(np.max(logits))
    p = np.exp(z, dtype=np.float32)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0.0:
        return int(np.argmax(logits))
    p = p / s
    cs = np.cumsum(p)
    r = float(np.random.random())
    idx = int(np.searchsorted(cs, r, side="right"))
    if idx >= int(logits.shape[0]):
        idx = int(logits.shape[0] - 1)
    return idx


def _decode_action(a: int) -> tuple[int, int]:
    return int(a // 10), int(a % 10)


def _read_initial() -> IOConfig:
    n, m, t, u = map(int, input().split())
    values = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        row = list(map(int, input().split()))
        for j in range(n):
            values[i, j] = row[j]
    start_xy: list[tuple[int, int]] = []
    for _ in range(m):
        x, y = map(int, input().split())
        start_xy.append((x, y))
    return IOConfig(n=n, m=m, t=t, u=u, values=values, start_xy=start_xy)


def _read_turn_result(n: int, m: int) -> tuple[list[tuple[int, int]], RuntimeState]:
    selected_moves: list[tuple[int, int]] = []
    for _ in range(m):
        sx, sy = map(int, input().split())
        selected_moves.append((sx, sy))

    px = [0] * m
    py = [0] * m
    for p in range(m):
        x, y = map(int, input().split())
        px[p] = x
        py[p] = y

    owner = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        row = list(map(int, input().split()))
        for j in range(n):
            owner[i, j] = row[j]

    level = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        row = list(map(int, input().split()))
        for j in range(n):
            level[i, j] = row[j]

    return selected_moves, RuntimeState(owner=owner, level=level, px=px, py=py)


def _to_case(cfg: IOConfig) -> CaseData:
    m = cfg.m
    t = cfg.t
    return CaseData(
        n=cfg.n,
        m=cfg.m,
        t=cfg.t,
        u=cfg.u,
        values=cfg.values,
        start_xy=cfg.start_xy,
        wa=np.zeros((max(0, m - 1),), dtype=np.float64),
        wb=np.zeros((max(0, m - 1),), dtype=np.float64),
        wc=np.zeros((max(0, m - 1),), dtype=np.float64),
        wd=np.zeros((max(0, m - 1),), dtype=np.float64),
        eps=np.zeros((max(0, m - 1),), dtype=np.float64),
        r=np.zeros((max(0, m - 1), 2 * max(1, t)), dtype=np.float64),
    )


def _encode_obs(case: CaseData, st: RuntimeState, turn: int, scores: np.ndarray, bayes_vec: np.ndarray) -> np.ndarray:
    n = case.n
    vals = np.zeros((10, 10), dtype=np.float32)
    owner_self = np.zeros((10, 10), dtype=np.float32)
    owner_enemy = np.zeros((10, 10), dtype=np.float32)
    owner_neutral = np.zeros((10, 10), dtype=np.float32)
    level_norm = np.zeros((10, 10), dtype=np.float32)
    piece_self = np.zeros((10, 10), dtype=np.float32)
    piece_enemy = np.zeros((10, 10), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            vals[i, j] = float(case.values[i, j] / 1000.0)
            o = int(st.owner[i, j])
            if o == 0:
                owner_self[i, j] = 1.0
            elif o == -1:
                owner_neutral[i, j] = 1.0
            else:
                owner_enemy[i, j] = 1.0
            level_norm[i, j] = float(st.level[i, j] / max(1, case.u))

    piece_self[st.px[0], st.py[0]] = 1.0
    for p in range(1, case.m):
        piece_enemy[st.px[p], st.py[p]] = 1.0

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
    globals_base[0] = float(turn / max(1, case.t))
    globals_base[1] = float((case.m - 2) / 6.0)
    globals_base[2] = float((case.u - 1) / 4.0)
    globals_base[3] = float(st.px[0] / max(1, n - 1))
    globals_base[4] = float(st.py[0] / max(1, n - 1))

    off = 5
    for ei in range(7):
        if ei + 1 < case.m:
            p = ei + 1
            globals_base[off + 2 * ei] = float(st.px[p] / max(1, n - 1))
            globals_base[off + 2 * ei + 1] = float(st.py[p] / max(1, n - 1))

    s_cap = float(max(1, case.u * np.sum(case.values)))
    s0 = float(scores[0])
    sa = float(np.max(scores[1:])) if case.m > 1 else 1.0
    globals_base[19] = float(np.clip(s0 / s_cap, 0.0, 1.0))
    globals_base[20] = float(np.clip(sa / s_cap, 0.0, 1.0))

    if bayes_vec.shape != (28,):
        bayes_vec = np.zeros((28,), dtype=np.float32)
    return np.concatenate([board_vec, globals_base, bayes_vec.astype(np.float32)]).astype(np.float32)


def main() -> int:
    meta = json.loads(MODEL_META_JSON)
    arrays = _load_arrays()
    if str(meta.get("model_type", "")) != "StudentMBoardAgent":
        raise RuntimeError(f"unsupported model_type for generated main.py: {meta.get('model_type')}")

    policy = StudentMPolicy(arrays, dict(meta.get("model_kwargs", {})))
    deterministic = bool(meta.get("deterministic", True))
    use_action_mask = bool(meta.get("use_action_mask", True))
    bayes_particles = int(meta.get("bayes_num_particles", 128))
    bayes_seed = int(meta.get("bayes_seed", 0))
    bayes_resample_ess_frac = float(meta.get("bayes_resample_ess_frac", 0.55))

    cfg = _read_initial()
    case = _to_case(cfg)
    st = _init_state(case)
    bayes = OpponentBayesEstimator(
        n=case.n,
        m=case.m,
        u=case.u,
        num_particles=bayes_particles,
        resample_ess_frac=bayes_resample_ess_frac,
        seed=bayes_seed,
    )

    for turn in range(cfg.t):
        scores = _score_all(st, case)
        candidates_all = [_get_candidates(case.n, case.m, st.owner, st.px, st.py, p) for p in range(case.m)]
        bayes_vec = bayes.posterior_feature_vector(max_enemies=7, normalize=True)
        obs = _encode_obs(case, st, turn, scores, bayes_vec)
        obs = _normalize_obs(obs, arrays, meta)
        logits, _value = policy.logits_and_value(obs)

        mask = None
        if use_action_mask:
            mask = _build_action_mask(candidates_all[0])
            if np.any(mask):
                logits = np.where(mask, logits, -1.0e30)

        if deterministic:
            action = int(np.argmax(logits))
        else:
            action = _sample_from_logits(logits)

        if mask is not None and (action < 0 or action >= mask.shape[0] or not bool(mask[action])):
            cands = candidates_all[0]
            if cands:
                action = int(cands[0][0] * 10 + cands[0][1])
            else:
                action = int(st.px[0] * 10 + st.py[0])

        x, y = _decode_action(action)
        print(x, y, flush=True)

        selected_moves, next_st = _read_turn_result(cfg.n, cfg.m)

        bayes.update(
            values=case.values,
            owner_before=st.owner,
            level_before=st.level,
            observed_selected=selected_moves,
            observed_candidates=candidates_all,
        )

        st = next_st

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export AHC061 submission-ready single-file main.py with embedded model weights.")
    p.add_argument("--checkpoint", type=Path, required=True, help="path to checkpoint (*.pt)")
    p.add_argument("--output", type=Path, default=Path("submission_main.py"), help="output main.py path")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="dtype used to embed weights")
    p.add_argument("--compress-level", type=int, default=9, help="zlib compression level for embedded weights (0-9)")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-action-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bayes-num-particles", type=int, default=128)
    p.add_argument("--bayes-seed", type=int, default=0)
    p.add_argument("--bayes-resample-ess-frac", type=float, default=0.55)
    p.add_argument("--vecnorm-mode", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    p.add_argument("--strict-size-limit", action=argparse.BooleanOptionalAction, default=False)
    return p


def _state_dict_to_numpy(state_dict: dict[str, Any], *, dtype: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    f_dtype = np.float16 if dtype == "fp16" else np.float32
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        if not torch.is_tensor(v):
            continue
        arr = v.detach().cpu().numpy()
        if arr.dtype.kind == "f":
            arr = arr.astype(f_dtype, copy=False)
        out[k] = np.asarray(arr)
    return out


def _compact_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _pack_arrays_b85(arrays: dict[str, np.ndarray], *, compress_level: int) -> str:
    lv = int(max(0, min(9, compress_level)))
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    raw = buf.getvalue()
    comp = zlib.compress(raw, level=lv)
    return base64.b85encode(comp).decode("ascii")


def _blob_to_python_lines(blob: str, *, width: int = 120) -> str:
    chunks = textwrap.wrap(blob, width=width)
    if not chunks:
        return '    ""'
    return "\n".join(f'    "{c}"' for c in chunks)


def build_submission_source(*, meta: dict[str, Any], packed_blob_b85: str) -> str:
    src = TEMPLATE_MAIN_PY
    meta_json_literal = repr(_compact_json(meta))
    src = src.replace("__MODEL_META_JSON__", meta_json_literal)
    src = src.replace("__MODEL_BLOB_B85_LINES__", _blob_to_python_lines(packed_blob_b85))
    return src


def _extract_vecnorm_for_export(
    meta: dict[str, Any],
    *,
    dtype: str,
    mode: str,
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
    out_meta: dict[str, Any] = {}
    out_arrays: dict[str, np.ndarray] = {}
    m = str(mode).strip().lower()
    vec_state = meta.get("vecnormalize_state")
    has_state = isinstance(vec_state, dict)
    use = bool(m == "on" or (m == "auto" and has_state))
    if m == "on" and not has_state:
        raise ValueError("vecnorm-mode=on but checkpoint meta has no vecnormalize_state")
    if not use:
        return False, out_meta, out_arrays
    assert isinstance(vec_state, dict)
    obs_rms = vec_state.get("obs_rms")
    if not isinstance(obs_rms, dict):
        return False, out_meta, out_arrays
    mean = np.asarray(obs_rms.get("mean"), dtype=np.float32)
    var = np.asarray(obs_rms.get("var"), dtype=np.float32)
    if mean.size == 0 or var.size == 0:
        return False, out_meta, out_arrays
    f_dtype = np.float16 if dtype == "fp16" else np.float32
    out_arrays["__vec_obs_mean"] = mean.astype(f_dtype, copy=False)
    out_arrays["__vec_obs_var"] = var.astype(f_dtype, copy=False)
    out_meta = {
        "epsilon": float(vec_state.get("epsilon", 1e-8)),
        "clip_obs": float(vec_state.get("clip_obs", 10.0)),
    }
    return True, out_meta, out_arrays


def main() -> int:
    args = build_parser().parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint has no valid model_config")
    model_type = str(model_config.get("type", "")).strip()
    if model_type != "StudentMBoardAgent":
        raise ValueError(f"unsupported model type for export: {model_type!r} (expected StudentMBoardAgent)")
    model_kwargs = model_config.get("kwargs")
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}

    obs_shape = payload.get("obs_shape")
    action_dim = payload.get("action_dim")
    if not isinstance(obs_shape, (tuple, list)) or len(obs_shape) != 1:
        raise ValueError(f"unexpected obs_shape: {obs_shape}")
    if int(action_dim) != 100:
        raise ValueError(f"unexpected action_dim: {action_dim} (expected 100)")

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no model_state_dict")
    arrays = _state_dict_to_numpy(state_dict, dtype=str(args.dtype))

    raw_meta = payload.get("meta")
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    use_vecnorm, vec_meta, vec_arrays = _extract_vecnorm_for_export(
        raw_meta,
        dtype=str(args.dtype),
        mode=str(args.vecnorm_mode),
    )
    arrays.update(vec_arrays)

    export_meta: dict[str, Any] = {
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "obs_dim": int(np.prod(np.asarray(obs_shape, dtype=np.int64))),
        "action_dim": int(action_dim),
        "deterministic": bool(args.deterministic),
        "use_action_mask": bool(args.use_action_mask),
        "bayes_num_particles": int(args.bayes_num_particles),
        "bayes_seed": int(args.bayes_seed),
        "bayes_resample_ess_frac": float(args.bayes_resample_ess_frac),
        "use_vecnorm": bool(use_vecnorm),
        "vecnorm": vec_meta,
    }

    blob = _pack_arrays_b85(arrays, compress_level=int(args.compress_level))
    source = build_submission_source(meta=export_meta, packed_blob_b85=blob)
    source_bytes = len(source.encode("utf-8"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source, encoding="utf-8")

    print(f"[export] checkpoint: {ckpt}")
    print(f"[export] output: {out}")
    print(f"[export] source bytes: {source_bytes}")
    print(f"[export] vecnorm embedded: {use_vecnorm}")
    if int(args.max_source_bytes) > 0 and source_bytes > int(args.max_source_bytes):
        msg = (
            f"generated source exceeds limit: {source_bytes} > {int(args.max_source_bytes)} bytes "
            f"(output={out})"
        )
        if bool(args.strict_size_limit):
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
