from __future__ import annotations

import numpy as np

from reinforce.ppo_discrete.domains.ahc061 import opponent_bayes as ob
from reinforce.ppo_discrete.domains.ahc061.opponent_bayes import create_opponent_bayes_estimator, ensure_cpp_bayes_backend


def _build_small_state():
    n, m, u = 10, 4, 3
    values = np.arange(n * n, dtype=np.int32).reshape(n, n) % 100 + 1
    owner = np.full((n, n), -1, dtype=np.int16)
    level = np.zeros((n, n), dtype=np.int16)
    px = [1, 3, 5, 7]
    py = [1, 3, 5, 7]
    for p in range(m):
        owner[px[p], py[p]] = p
        level[px[p], py[p]] = 1
    candidates = [ob._get_candidates(n, m, owner, px, py, p) for p in range(m)]
    observed = [cands[0] if cands else (px[p], py[p]) for p, cands in enumerate(candidates)]
    return {
        "n": n,
        "m": m,
        "u": u,
        "values": values,
        "owner": owner,
        "level": level,
        "px": px,
        "py": py,
        "observed": observed,
        "candidates": candidates,
    }


def _run_once(backend: str):
    s = _build_small_state()
    e = create_opponent_bayes_estimator(
        n=s["n"],
        m=s["m"],
        u=s["u"],
        num_particles=32,
        seed=1234,
        backend=backend,
    )
    e.update(
        values=s["values"],
        owner_before=s["owner"],
        level_before=s["level"],
        observed_selected=s["observed"],
        observed_candidates=s["candidates"],
    )
    return e.posterior_feature_vector(max_enemies=7, normalize=True)


def test_auto_backend_observed_candidates_runs():
    if not ensure_cpp_bayes_backend(build_if_missing=False, force_build=False, verbose=False):
        return
    feat = _run_once("auto")
    assert feat.shape == (28,)
    assert np.all(np.isfinite(feat))


def test_cpp_backend_observed_candidates_runs():
    if not ensure_cpp_bayes_backend(build_if_missing=False, force_build=False, verbose=False):
        return
    f_cpp = _run_once("cpp")
    assert f_cpp.shape == (28,)
    assert np.all(np.isfinite(f_cpp))
