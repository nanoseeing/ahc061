"""`metrics` に関するユーティリティ。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ScalarSummary:
    """`ScalarSummary` を表すデータクラス。

    Attributes:
        count (int): フィールド値。
        mean (float): フィールド値。
        std (float): フィールド値。
        min (float): フィールド値。
        p25 (float): フィールド値。
        p50 (float): フィールド値。
        p75 (float): フィールド値。
        max (float): フィールド値。
    """
    count: int
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    max: float

    def as_dict(self) -> dict[str, float | int]:
        """`as_dict` を実行する。

        Returns:
            dict[str, float | int]: 計算結果。
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "max": self.max,
        }


def summarize(values: Iterable[float]) -> ScalarSummary:
    """`summarize` を実行する。

    Args:
        values (Iterable[float]): values の値。

    Returns:
        ScalarSummary: 計算結果。
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        nan = float("nan")
        return ScalarSummary(0, nan, nan, nan, nan, nan, nan, nan)
    return ScalarSummary(
        count=int(arr.size),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        max=float(np.max(arr)),
    )


def summarize_mean_variance(values: Sequence[float] | Iterable[float]) -> dict[str, float | int]:
    """`summarize_mean_variance` を実行する。

    Args:
        values (Sequence[float] | Iterable[float]): values の値。

    Returns:
        dict[str, float | int]: 計算結果。
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        nan = float("nan")
        return {
            "count": 0,
            "mean": nan,
            "variance": nan,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "variance": float(np.var(arr)),
    }


def group_score_mean_variance_by_m_u(
    *,
    scores: Sequence[float],
    m_values: Sequence[int],
    u_values: Sequence[int],
    m_key: str = "m",
    u_key: str = "u",
) -> dict[str, list[dict[str, float | int]]]:
    """`group_score_mean_variance_by_m_u` を実行する。

    Args:
        scores (Sequence[float]): scores の値。
        m_values (Sequence[int]): m_values の値。
        u_values (Sequence[int]): u_values の値。
        m_key (str): m_key の値。
        u_key (str): u_key の値。

    Returns:
        dict[str, list[dict[str, float | int]]]: 計算結果。
    """
    if not (len(scores) == len(m_values) == len(u_values)):
        raise ValueError(
            "grouped score length mismatch: "
            f"scores={len(scores)} m_values={len(m_values)} u_values={len(u_values)}"
        )

    by_m: dict[int, list[float]] = defaultdict(list)
    by_u: dict[int, list[float]] = defaultdict(list)
    by_m_u: dict[tuple[int, int], list[float]] = defaultdict(list)
    for score, m_val, u_val in zip(scores, m_values, u_values):
        s = float(score)
        if not np.isfinite(s):
            continue
        m = int(m_val)
        u = int(u_val)
        by_m[m].append(s)
        by_u[u].append(s)
        by_m_u[(m, u)].append(s)

    rows_m: list[dict[str, float | int]] = []
    for m in sorted(by_m.keys()):
        row: dict[str, float | int] = {str(m_key): int(m)}
        row.update(summarize_mean_variance(by_m[m]))
        rows_m.append(row)

    rows_u: list[dict[str, float | int]] = []
    for u in sorted(by_u.keys()):
        row = {str(u_key): int(u)}
        row.update(summarize_mean_variance(by_u[u]))
        rows_u.append(row)

    rows_m_u: list[dict[str, float | int]] = []
    for m_u in sorted(by_m_u.keys()):
        m, u = m_u
        row = {str(m_key): int(m), str(u_key): int(u)}
        row.update(summarize_mean_variance(by_m_u[m_u]))
        rows_m_u.append(row)

    return {
        "by_m": rows_m,
        "by_u": rows_u,
        "by_m_u": rows_m_u,
    }
