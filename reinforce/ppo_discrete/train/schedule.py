from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..ppo.config import PPOConfig

ScheduleFn = Callable[[float], float]


def validate_ppo_config(cfg: PPOConfig) -> None:
    # Field-level and batch constraints are now validated by PPOConfig (pydantic BaseModel).
    # This function is kept for backward compatibility; callers can still call it safely.
    pass


def resolve_vecnorm_gamma(*, vecnorm_gamma: float | None, ppo_gamma: float) -> float:
    if vecnorm_gamma is None:
        return float(ppo_gamma)
    return float(vecnorm_gamma)


def validate_vecnorm_config(
    *,
    enabled: bool,
    clip_obs: float,
    clip_reward: float,
    epsilon: float,
    vecnorm_gamma: float | None,
    ppo_gamma: float,
) -> None:
    if not bool(enabled):
        return
    if float(clip_obs) <= 0.0:
        raise ValueError(f"vecnorm_clip_obs must be positive: {clip_obs}")
    if float(clip_reward) <= 0.0:
        raise ValueError(f"vecnorm_clip_reward must be positive: {clip_reward}")
    if float(epsilon) <= 0.0:
        raise ValueError(f"vecnorm_epsilon must be positive: {epsilon}")
    vgamma = resolve_vecnorm_gamma(vecnorm_gamma=vecnorm_gamma, ppo_gamma=ppo_gamma)
    if not np.isfinite(vgamma) or vgamma <= 0.0:
        raise ValueError(f"vecnorm_gamma must be positive: {vgamma}")


def _scheduled_scalar(kind: str, start: float, end: float, progress: float) -> float:
    k = str(kind).strip().lower()
    if k == "constant":
        return float(start)
    if k == "linear":
        return float(start + (end - start) * progress)
    if k == "cosine":
        return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(f"unknown schedule kind: {kind}")


def _split_csv_args(text: str) -> list[str]:
    raw = str(text).strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_schedule_expr_impl(
    expr: str,
    *,
    default_kind: str,
    default_start: float,
    default_end: float,
) -> tuple[ScheduleFn, str]:
    text = str(expr).strip()
    if not text:
        k = str(default_kind).strip().lower()
        return (
            lambda p: float(_scheduled_scalar(k, float(default_start), float(default_end), float(p))),
            f"{k}({float(default_start):.12g},{float(default_end):.12g})",
        )

    low = text.lower()
    if low in ("constant", "linear", "cosine"):
        return (
            lambda p: float(_scheduled_scalar(low, float(default_start), float(default_end), float(p))),
            f"{low}({float(default_start):.12g},{float(default_end):.12g})",
        )
    if low == "exp":
        a = float(default_start)
        b = float(default_end)
        if a <= 0.0 or b <= 0.0:
            raise ValueError(f"exp schedule requires positive start/end, got ({a}, {b})")
        return (lambda p: float(a * ((b / a) ** float(np.clip(p, 0.0, 1.0)))), f"exp({a:.12g},{b:.12g})")

    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", text)
    if m is None:
        raise ValueError(
            f"invalid schedule expr={expr!r}; expected e.g. constant(0.1), linear(0.2,0.05), "
            "cosine(0.2,0.01), exp(3e-4,1e-5), piecewise(0:0.2,0.5:0.1,1:0.05)"
        )
    name = str(m.group(1)).strip().lower()
    args = _split_csv_args(m.group(2))

    if name == "constant":
        if len(args) > 1:
            raise ValueError(f"constant(...) expects 0 or 1 args, got: {expr}")
        v = float(default_start if len(args) == 0 else float(args[0]))
        return (lambda _p: float(v), f"constant({v:.12g})")

    if name in ("linear", "cosine", "exp"):
        if len(args) not in (0, 2):
            raise ValueError(f"{name}(...) expects 0 or 2 args, got: {expr}")
        a = float(default_start if len(args) == 0 else float(args[0]))
        b = float(default_end if len(args) == 0 else float(args[1]))
        if name == "exp":
            if a <= 0.0 or b <= 0.0:
                raise ValueError(f"exp schedule requires positive start/end, got ({a}, {b})")
            return (lambda p: float(a * ((b / a) ** float(np.clip(p, 0.0, 1.0)))), f"exp({a:.12g},{b:.12g})")
        return (
            lambda p: float(_scheduled_scalar(name, float(a), float(b), float(p))),
            f"{name}({a:.12g},{b:.12g})",
        )

    if name == "piecewise":
        if len(args) < 2:
            raise ValueError(f"piecewise(...) expects >=2 points, got: {expr}")
        pts: list[tuple[float, float]] = []
        for tok in args:
            if ":" not in tok:
                raise ValueError(f"invalid piecewise point={tok!r}; expected p:v")
            ptxt, vtxt = tok.split(":", 1)
            p = float(ptxt.strip())
            v = float(vtxt.strip())
            if not np.isfinite(p) or not np.isfinite(v):
                raise ValueError(f"invalid piecewise point={tok!r}; non-finite value")
            pts.append((p, v))
        pts.sort(key=lambda x: x[0])
        for i in range(1, len(pts)):
            if abs(pts[i][0] - pts[i - 1][0]) <= 1e-12:
                raise ValueError(f"piecewise has duplicate progress points: {pts[i][0]}")

        def _piecewise(progress: float) -> float:
            p = float(np.clip(progress, 0.0, 1.0))
            if p <= pts[0][0]:
                return float(pts[0][1])
            for j in range(1, len(pts)):
                p0, v0 = pts[j - 1]
                p1, v1 = pts[j]
                if p <= p1:
                    if p1 <= p0 + 1e-12:
                        return float(v1)
                    t = (p - p0) / (p1 - p0)
                    return float(v0 + t * (v1 - v0))
            return float(pts[-1][1])

        desc = "piecewise(" + ",".join(f"{p:.6g}:{v:.6g}" for p, v in pts) + ")"
        return _piecewise, desc

    raise ValueError(
        f"unknown schedule expr kind={name!r}; supported: constant, linear, cosine, exp, piecewise"
    )


@dataclass(frozen=True)
class ScalarSchedule:
    fn: ScheduleFn
    description: str
    nonnegative: bool = False

    def __post_init__(self) -> None:
        self.validate_samples()

    def __call__(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        v = float(self.fn(p))
        if not np.isfinite(v):
            raise ValueError(
                f"schedule produced non-finite value at progress={p}: {v} ({self.description})"
            )
        if self.nonnegative and v < 0.0:
            raise ValueError(
                f"schedule produced negative value at progress={p}: {v} ({self.description})"
            )
        return v

    def validate_samples(self) -> None:
        for p in (0.0, 0.5, 1.0):
            _ = self(p)

    @classmethod
    def from_expr(
        cls,
        expr: str,
        *,
        default_kind: str,
        default_start: float,
        default_end: float,
        nonnegative: bool,
    ) -> ScalarSchedule:
        fn, desc = _parse_schedule_expr_impl(
            expr,
            default_kind=default_kind,
            default_start=default_start,
            default_end=default_end,
        )
        return cls(fn=fn, description=desc, nonnegative=bool(nonnegative))


@dataclass(frozen=True)
class PPOScheduleSet:
    learning_rate: ScalarSchedule
    ent_coef: ScalarSchedule
    clip_coef: ScalarSchedule
    clip_range_vf: ScalarSchedule | None

    @classmethod
    def from_config(cls, cfg: PPOConfig) -> PPOScheduleSet:
        lr = ScalarSchedule.from_expr(
            str(cfg.learning_rate_schedule),
            default_kind="linear",
            default_start=float(cfg.learning_rate),
            default_end=0.0,
            nonnegative=True,
        )
        ent_start = float(cfg.ent_coef)
        ent_end = float(ent_start if cfg.ent_coef_final is None else cfg.ent_coef_final)
        ent = ScalarSchedule.from_expr(
            str(cfg.ent_coef_schedule_expr),
            default_kind=str(cfg.ent_coef_schedule),
            default_start=ent_start,
            default_end=ent_end,
            nonnegative=True,
        )
        clip_start = float(cfg.clip_coef)
        clip_end = float(clip_start if cfg.clip_coef_final is None else cfg.clip_coef_final)
        clip = ScalarSchedule.from_expr(
            str(cfg.clip_coef_schedule_expr),
            default_kind=str(cfg.clip_coef_schedule),
            default_start=clip_start,
            default_end=clip_end,
            nonnegative=True,
        )

        vf: ScalarSchedule | None = None
        if cfg.clip_range_vf is not None:
            vf_start = float(cfg.clip_range_vf)
            vf_end = float(vf_start if cfg.clip_range_vf_final is None else cfg.clip_range_vf_final)
            vf = ScalarSchedule.from_expr(
                str(cfg.clip_range_vf_schedule_expr),
                default_kind=str(cfg.clip_range_vf_schedule),
                default_start=vf_start,
                default_end=vf_end,
                nonnegative=True,
            )
        elif (
            str(cfg.clip_range_vf_schedule_expr).strip()
            or cfg.clip_range_vf_final is not None
            or str(cfg.clip_range_vf_schedule).strip().lower() != "constant"
        ):
            raise ValueError(
                "clip_range_vf schedule was configured but clip_range_vf is unset; "
                "set clip_range_vf to enable value clipping schedule"
            )

        return cls(
            learning_rate=lr,
            ent_coef=ent,
            clip_coef=clip,
            clip_range_vf=vf,
        )


def parse_schedule_expr(
    expr: str,
    *,
    default_kind: str,
    default_start: float,
    default_end: float,
) -> tuple[ScheduleFn, str]:
    sch = ScalarSchedule.from_expr(
        expr,
        default_kind=default_kind,
        default_start=default_start,
        default_end=default_end,
        nonnegative=False,
    )
    return sch, sch.description


def validate_schedule_args(cfg: PPOConfig) -> None:
    _ = PPOScheduleSet.from_config(cfg)


def schedule_progress(iteration: int, total_iterations: int) -> float:
    if total_iterations <= 1:
        return 1.0
    p = float(iteration - 1) / float(total_iterations - 1)
    return float(np.clip(p, 0.0, 1.0))
