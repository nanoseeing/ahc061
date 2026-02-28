"""PPO 学習設定を定義するコンフィグモジュール。

Pydantic モデルで型と制約を管理し、学習開始前に不整合を検出する。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PPOConfig(BaseModel):
    """PPO 実験で使用するハイパーパラメータ群。

    バッチサイズ整合性や係数レンジをバリデーションし、学習実行時の
    設定ミスを早期に検出する。
    """

    model_config = ConfigDict(frozen=False)

    exp_name: str = "ahc061"
    seed: int = 1
    cuda: bool = True

    total_iterations: int = Field(default=500, gt=0)
    num_envs: int = Field(default=8, gt=0)
    num_steps: int = Field(default=128, gt=0)

    learning_rate: float = Field(default=2.5e-4, gt=0.0)
    learning_rate_schedule: str = ""
    gamma: float = Field(default=0.99, gt=0.0)
    gae_lambda: float = Field(default=0.95, gt=0.0)

    num_minibatches: int = Field(default=4, gt=0)
    update_epochs: int = Field(default=4, gt=0)
    norm_adv: bool = True
    clip_coef: float = Field(default=0.2, gt=0.0)
    clip_range_vf: Optional[float] = None
    clip_range_vf_schedule: Literal["constant", "linear", "cosine"] = "constant"
    clip_range_vf_final: Optional[float] = None
    clip_range_vf_schedule_expr: str = ""
    clip_vloss: bool = True
    ent_coef: float = Field(default=0.01, ge=0.0)
    ent_coef_schedule: Literal["constant", "linear", "cosine"] = "constant"
    ent_coef_final: Optional[float] = None
    ent_coef_schedule_expr: str = ""
    vf_coef: float = Field(default=0.5, ge=0.0)
    aux_opp_param_loss_coef: float = Field(default=0.0, ge=0.0)
    aux_opp_param_use_valid_mask: bool = True
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    target_kl: Optional[float] = None
    clip_coef_schedule: Literal["constant", "linear", "cosine"] = "constant"
    clip_coef_final: Optional[float] = None
    clip_coef_schedule_expr: str = ""

    warmup_iters: int = Field(default=0, ge=0)

    save_interval: int = Field(default=10, gt=0)

    @field_validator("clip_range_vf")
    @classmethod
    def _validate_clip_range_vf(cls, v: Optional[float]) -> Optional[float]:
        """`clip_range_vf` が設定されている場合に正値かを検証する。

        Args:
            v (Optional[float]): 検証指定値。

        Returns:
            Optional[float]: 検証済みの値。
        """
        if v is not None and float(v) <= 0.0:
            raise ValueError(f"clip_range_vf must be positive when set: {v}")
        return v

    @field_validator("clip_range_vf_final", "ent_coef_final", "clip_coef_final")
    @classmethod
    def _validate_optional_final_nonneg(cls, v: Optional[float]) -> Optional[float]:
        """スケジュール終端係数が設定されている場合に非負かを検証する。

        Args:
            v (Optional[float]): 検証指定値。

        Returns:
            Optional[float]: 検証済みの値。
        """
        if v is not None and float(v) < 0.0:
            raise ValueError(f"final coefficient must be >= 0 when set: {v}")
        return v

    @field_validator("target_kl")
    @classmethod
    def _validate_target_kl(cls, v: Optional[float]) -> Optional[float]:
        """`target_kl` が設定されている場合に正値かを検証する。

        Args:
            v (Optional[float]): 検証指定値。

        Returns:
            Optional[float]: 検証済みの値。
        """
        if v is not None and float(v) <= 0.0:
            raise ValueError(f"target_kl must be positive when set: {v}")
        return v

    @model_validator(mode="after")
    def _validate_batch_config(self) -> PPOConfig:
        """バッチ構成の整合性を検証する。

        `batch_size = num_envs * num_steps` が `num_minibatches` で割り切れること、
        かつ `norm_adv=True` 時にミニバッチサイズが極端に小さくないことを確認する。

        Returns:
            PPOConfig: 検証済みの設定自身。
        """
        batch_size = int(self.num_envs) * int(self.num_steps)
        if batch_size % int(self.num_minibatches) != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by num_minibatches={self.num_minibatches}; "
                "adjust num_envs, num_steps, or num_minibatches"
            )
        minibatch_size = batch_size // int(self.num_minibatches)
        if minibatch_size <= 1 and bool(self.norm_adv):
            raise ValueError(
                f"minibatch_size={minibatch_size} is too small for advantage normalization; "
                "increase num_envs*num_steps or set norm_adv=False"
            )
        return self

    @property
    def batch_size(self) -> int:
        """1 イテレーションで収集するサンプル数（`num_envs * num_steps`）を返す。

        Returns:
            int: バッチサイズ。
        """
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self) -> int:
        """1 ミニバッチあたりのサンプル数を返す。

        Returns:
            int: ミニバッチサイズ。
        """
        return int(self.batch_size // self.num_minibatches)

    @property
    def num_iterations(self) -> int:
        """学習イテレーション数を返す。互換性維持のための別名プロパティ。

        Returns:
            int: 学習イテレーション数。
        """
        return int(self.total_iterations)
