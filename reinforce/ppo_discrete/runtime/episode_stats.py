from __future__ import annotations

from typing import Any

import numpy as np


def extract_completed_episode_stats(infos: Any) -> list[dict[str, float]]:
    """Extract per-episode stats from vector-env infos.

    Returned dict keys:
    - total_return
    - length
    - illegal_penalty_return
    - terminal_score_return
    - terminal_game_score_return
    - final_self_score
    - final_enemy_max_score
    - final_score_ratio
    - final_score_log2
    - final_game_score
    """
    if not isinstance(infos, dict):
        return []

    stats = _extract_from_episode_arrays(infos)
    if stats:
        return stats
    return _extract_from_final_info(infos)


def _extract_from_episode_arrays(infos: dict[str, Any]) -> list[dict[str, float]]:
    ep = infos.get("episode")
    ep_mask = infos.get("_episode")
    if not isinstance(ep, dict) or ep_mask is None:
        return []

    r = _as_1d(ep.get("r"))
    l = _as_1d(ep.get("l"))
    m = _as_bool_1d(ep_mask)
    if r is None or m is None:
        return []

    illegal = _as_1d(infos.get("episode_illegal_penalty"))
    terminal = _as_1d(infos.get("episode_terminal_score"))
    terminal_game = _as_1d(infos.get("episode_terminal_game_score"))
    final_scores = _as_2d(infos.get("final_scores"))
    final_scores_mask = _as_bool_1d(infos.get("_final_scores"))
    out: list[dict[str, float]] = []
    n = min(r.shape[0], m.shape[0])
    for i in range(n):
        if not bool(m[i]):
            continue
        score_self, score_enemy_max, score_ratio, score_log2, score_game = _score_fields_from_array_row(
            final_scores,
            final_scores_mask,
            i,
        )
        terminal_game_score_return = (
            float(terminal_game[i])
            if terminal_game is not None and i < terminal_game.shape[0]
            else (float(score_game) if np.isfinite(float(score_game)) else 0.0)
        )
        out.append(
            {
                "total_return": float(r[i]),
                "length": float(l[i]) if l is not None and i < l.shape[0] else float("nan"),
                "illegal_penalty_return": float(illegal[i]) if illegal is not None and i < illegal.shape[0] else 0.0,
                "terminal_score_return": float(terminal[i]) if terminal is not None and i < terminal.shape[0] else 0.0,
                "terminal_game_score_return": terminal_game_score_return,
                "final_self_score": score_self,
                "final_enemy_max_score": score_enemy_max,
                "final_score_ratio": score_ratio,
                "final_score_log2": score_log2,
                "final_game_score": score_game,
            }
        )
    return out


def _extract_from_final_info(infos: dict[str, Any]) -> list[dict[str, float]]:
    finfos = infos.get("final_info")
    if not isinstance(finfos, (list, tuple, np.ndarray)):
        return []

    out: list[dict[str, float]] = []
    for fi in finfos:
        if not isinstance(fi, dict):
            continue
        episode = fi.get("episode")
        if not isinstance(episode, dict):
            continue
        total_return = _safe_float(episode.get("r"), float("nan"))
        length = _safe_float(episode.get("l"), float("nan"))
        score_self, score_enemy_max, score_ratio, score_log2, score_game = _score_fields_from_final_info(fi)
        out.append(
            {
                "total_return": total_return,
                "length": length,
                "illegal_penalty_return": _safe_float(fi.get("episode_illegal_penalty"), 0.0),
                "terminal_score_return": _safe_float(fi.get("episode_terminal_score"), 0.0),
                "terminal_game_score_return": _safe_float(
                    fi.get("episode_terminal_game_score"),
                    float(score_game) if np.isfinite(float(score_game)) else 0.0,
                ),
                "final_self_score": score_self,
                "final_enemy_max_score": score_enemy_max,
                "final_score_ratio": score_ratio,
                "final_score_log2": score_log2,
                "final_game_score": score_game,
            }
        )
    return out


def _as_2d(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _as_1d(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _as_bool_1d(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.bool_).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _safe_float(v: Any, default: float) -> float:
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return default
        v = np.asarray(v).reshape(-1)[0]
    try:
        return float(v)
    except Exception:
        return default


def _score_fields_from_array_row(
    final_scores: np.ndarray | None,
    final_scores_mask: np.ndarray | None,
    idx: int,
) -> tuple[float, float, float, float, float]:
    if final_scores is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    if idx >= final_scores.shape[0]:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    if final_scores_mask is not None and idx < final_scores_mask.shape[0] and not bool(final_scores_mask[idx]):
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    row = np.asarray(final_scores[idx], dtype=np.float64).reshape(-1)
    if row.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    self_score = float(row[0])
    enemy_max = float(np.max(row[1:])) if row.size > 1 else float("nan")
    return _score_fields_from_pair(self_score, enemy_max)


def _score_fields_from_final_info(fi: dict[str, Any]) -> tuple[float, float, float, float, float]:
    fs = fi.get("final_scores")
    if fs is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    row = np.asarray(fs, dtype=np.float64).reshape(-1)
    if row.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    self_score = float(row[0])
    enemy_max = float(np.max(row[1:])) if row.size > 1 else float("nan")
    return _score_fields_from_pair(self_score, enemy_max)


def _score_fields_from_pair(self_score: float, enemy_max: float) -> tuple[float, float, float, float, float]:
    if not np.isfinite(self_score) or not np.isfinite(enemy_max):
        return self_score, enemy_max, float("nan"), float("nan"), float("nan")
    ratio = self_score / max(1.0, enemy_max)
    log2_ratio = float(np.log2(max(ratio, 1e-12)))
    # Keep game_score definition consistent with env._game_defined_score:
    # round(1e5 * log2(1 + floor_nonneg(S0) / floor_nonneg(SA)))
    s0_i = int(max(0.0, float(self_score)))
    sa_i = int(max(0.0, float(enemy_max)))
    if sa_i <= 0:
        game_score = float(np.iinfo(np.int64).max)
    else:
        game_score = float(np.round(1e5 * np.log2(1.0 + float(s0_i) / float(sa_i))))
    return self_score, enemy_max, float(ratio), log2_ratio, game_score
