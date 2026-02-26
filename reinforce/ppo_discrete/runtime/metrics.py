from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ScalarSummary:
    count: int
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    max: float

    def as_dict(self) -> dict[str, float | int]:
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
