"""PPO 学習で使う係数スケジュールの解決・検証ユーティリティ。"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..ppo.config import PPOConfig

ScheduleFn = Callable[[float], float]



def resolve_vecnorm_gamma(*, vecnorm_gamma: float | None, ppo_gamma: float) -> float:
    """VecNormalize 用 `gamma` を解決する。

    Args:
        vecnorm_gamma (float | None): VecNormalize 専用 `gamma`。`None` なら PPO 側を流用。
        ppo_gamma (float): PPO の割引率 `gamma`。

    Returns:
        float: 実際に使用する VecNormalize の `gamma`。
    """
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
    """VecNormalize 関連設定の妥当性を検証する。

    Args:
        enabled (bool): VecNormalize を使うか。
        clip_obs (float): 観測クリップ値。
        clip_reward (float): 報酬クリップ値。
        epsilon (float): 数値安定化のイプシロン。
        vecnorm_gamma (float | None): VecNormalize 専用 `gamma`。
        ppo_gamma (float): PPO の割引率 `gamma`。
    """
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
    """内部ヘルパー: 進捗に応じてスカラー係数を補間する。

    Args:
        kind (str): スケジュール種別（`constant` / `linear` / `cosine`）。
        start (float): 開始値。
        end (float): 終了値。
        progress (float): 進捗（`0.0` から `1.0`）。

    Returns:
        float: 補間後の値。
    """
    k = str(kind).strip().lower()
    if k == "constant":
        return float(start)
    if k == "linear":
        return float(start + (end - start) * progress)
    if k == "cosine":
        return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(f"unknown schedule kind: {kind}")


def _split_csv_args(text: str) -> list[str]:
    """内部ヘルパー: カンマ区切り引数文字列を分割する。

    Args:
        text (str): 分割対象文字列。

    Returns:
        list[str]: 空白除去済みのトークン列。
    """
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
    """内部ヘルパー: スケジュール式を実行関数と説明文字列へ変換する。

    Args:
        expr (str): スケジュール式。
        default_kind (str): 省略時の種別。
        default_start (float): 省略時の開始値。
        default_end (float): 省略時の終了値。

    Returns:
        tuple[ScheduleFn, str]: 評価関数と説明文字列。
    """
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
            """内部ヘルパー: piecewise 線形補間で値を返す。

            Args:
                progress (float): 進捗（`0.0` から `1.0`）。

            Returns:
                float: 補間後の値。
            """
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
    """単一スカラー係数のスケジュール定義。"""
    fn: ScheduleFn
    description: str
    nonnegative: bool = False

    def __post_init__(self) -> None:
        """生成時にサンプル点でスケジュールの妥当性を確認する。"""
        self.validate_samples()

    def __call__(self, progress: float) -> float:
        """進捗に対応するスケジュール値を返す。

        Args:
            progress (float): 進捗（`0.0` から `1.0`）。

        Returns:
            float: スケジュール値。
        """
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
        """代表点（0.0, 0.5, 1.0）でスケジュールを評価して検証する。"""
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
        """式文字列から `ScalarSchedule` を生成する。

        Args:
            expr (str): スケジュール式。
            default_kind (str): 省略時の種別。
            default_start (float): 省略時の開始値。
            default_end (float): 省略時の終了値。
            nonnegative (bool): 非負制約を有効にするか。

        Returns:
            ScalarSchedule: 生成したスケジュール。
        """
        fn, desc = _parse_schedule_expr_impl(
            expr,
            default_kind=default_kind,
            default_start=default_start,
            default_end=default_end,
        )
        return cls(fn=fn, description=desc, nonnegative=bool(nonnegative))


@dataclass(frozen=True)
class PPOScheduleSet:
    """PPO 学習で使う主要係数スケジュールの集合。"""
    learning_rate: ScalarSchedule
    ent_coef: ScalarSchedule
    clip_coef: ScalarSchedule
    clip_range_vf: ScalarSchedule | None

    @classmethod
    def from_config(cls, cfg: PPOConfig) -> PPOScheduleSet:
        """`PPOConfig` からスケジュール群を構築する。

        Args:
            cfg (PPOConfig): 設定オブジェクト。

        Returns:
            PPOScheduleSet: 学習率・エントロピー係数などのスケジュール集合。
        """
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


@dataclass(frozen=True)
class RuntimeScheduleCoefficients:
    """1 イテレーション時点で実際に適用するスケジュール係数。"""
    progress: float
    learning_rate: float
    ent_coef: float
    clip_coef: float
    clip_range_vf: float | None


class RuntimeScheduleResolver:
    """実行イテレーションに応じた係数を解決するヘルパー。"""

    def __init__(
        self,
        *,
        schedules: PPOScheduleSet,
        total_iterations: int,
        warmup_iterations: int,
    ) -> None:
        """ランタイム解決器を初期化する。

        Args:
            schedules (PPOScheduleSet): 解決対象のスケジュール群。
            total_iterations (int): 学習総イテレーション数。
            warmup_iterations (int): 学習率ウォームアップ期間。
        """
        self.schedules = schedules
        self.total_iterations = int(total_iterations)
        self.warmup_iterations = int(warmup_iterations)
        if self.total_iterations <= 0:
            raise ValueError(f"total_iterations must be > 0: {self.total_iterations}")
        if self.warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be >= 0: {self.warmup_iterations}")

    def resolve(self, *, iteration: int) -> RuntimeScheduleCoefficients:
        """指定イテレーションで適用する係数セットを返す。

        Args:
            iteration (int): 1 始まりの学習イテレーション。

        Returns:
            RuntimeScheduleCoefficients: 解決済み係数セット。
        """
        iteration_i = int(iteration)
        progress = schedule_progress(iteration_i, int(self.total_iterations))

        learning_rate, lr_progress = self._resolve_learning_rate(
            iteration=iteration_i,
        )
        self._validate_nonnegative_finite(learning_rate, name="learning_rate", progress=progress)

        ent_coef = float(self.schedules.ent_coef(progress))
        self._validate_nonnegative_finite(ent_coef, name="ent_coef", progress=progress)

        clip_coef = float(self.schedules.clip_coef(progress))
        self._validate_nonnegative_finite(clip_coef, name="clip_coef", progress=progress)

        clip_range_vf: float | None = None
        if self.schedules.clip_range_vf is not None:
            clip_range_vf = float(self.schedules.clip_range_vf(progress))
            self._validate_nonnegative_finite(clip_range_vf, name="clip_range_vf", progress=progress)

        return RuntimeScheduleCoefficients(
            progress=float(lr_progress),
            learning_rate=float(learning_rate),
            ent_coef=float(ent_coef),
            clip_coef=float(clip_coef),
            clip_range_vf=(None if clip_range_vf is None else float(clip_range_vf)),
        )

    def _resolve_learning_rate(self, *, iteration: int) -> tuple[float, float]:
        """内部ヘルパー: ウォームアップを考慮して学習率を解決する。

        Args:
            iteration (int): 1 始まりの学習イテレーション。

        Returns:
            tuple[float, float]: 学習率とスケジュール進捗。
        """
        if self.warmup_iterations <= 0:
            lr_progress = schedule_progress(int(iteration), int(self.total_iterations))
            return float(self.schedules.learning_rate(lr_progress)), float(lr_progress)

        warmup_iters = min(int(self.total_iterations), int(self.warmup_iterations))
        if int(iteration) <= warmup_iters:
            base_lr = float(self.schedules.learning_rate(0.0))
            warmup_ratio = float(min(1.0, float(iteration) / float(max(1, warmup_iters))))
            return float(base_lr * warmup_ratio), 0.0

        # Start annealing only after warmup window has finished.
        anneal_iteration = int(iteration) - warmup_iters
        anneal_total_iterations = max(1, int(self.total_iterations) - warmup_iters)
        lr_progress = schedule_progress(anneal_iteration, anneal_total_iterations)
        return float(self.schedules.learning_rate(lr_progress)), float(lr_progress)

    @staticmethod
    def _validate_nonnegative_finite(value: float, *, name: str, progress: float) -> None:
        """内部ヘルパー: 係数が有限かつ非負であることを検証する。

        Args:
            value (float): 検証対象値。
            name (str): 値の名前。
            progress (float): 進捗。
        """
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"invalid scheduled {name}={value} at progress={progress}")


def parse_schedule_expr(
    expr: str,
    *,
    default_kind: str,
    default_start: float,
    default_end: float,
) -> tuple[ScheduleFn, str]:
    """スケジュール式を解析して関数と説明文字列を返す。

    Args:
        expr (str): スケジュール式。
        default_kind (str): 省略時の種別。
        default_start (float): 省略時の開始値。
        default_end (float): 省略時の終了値。

    Returns:
        tuple[ScheduleFn, str]: 評価関数と説明文字列。
    """
    sch = ScalarSchedule.from_expr(
        expr,
        default_kind=default_kind,
        default_start=default_start,
        default_end=default_end,
        nonnegative=False,
    )
    return sch, sch.description


def validate_schedule_args(cfg: PPOConfig) -> None:
    """`PPOConfig` のスケジュール設定が解決可能かを検証する。

    Args:
        cfg (PPOConfig): 設定オブジェクト。
    """
    _ = PPOScheduleSet.from_config(cfg)


def schedule_progress(iteration: int, total_iterations: int) -> float:
    """イテレーション番号から `0.0..1.0` の進捗率を計算する。

    Args:
        iteration (int): 1 始まりの現在イテレーション。
        total_iterations (int): 学習総イテレーション数。

    Returns:
        float: クリップ済み進捗率。
    """
    if total_iterations <= 1:
        return 1.0
    p = float(iteration - 1) / float(total_iterations - 1)
    return float(np.clip(p, 0.0, 1.0))
