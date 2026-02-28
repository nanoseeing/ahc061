"""バッチ環境実装が満たすべき Protocol 定義。"""
from __future__ import annotations

from typing import Protocol, Sequence, cast, runtime_checkable

import torch

from .env import EnvSpec


@runtime_checkable
class BatchEnvProtocol(Protocol):
    """ロールアウト/評価ループが利用するバッチ環境インターフェース。"""

    @property
    def spec(self) -> EnvSpec:
        """環境の静的仕様を返す。"""
        ...

    @property
    def batch_size(self) -> int:
        """並列環境数を返す。"""
        ...

    @property
    def pf_enabled(self) -> bool:
        """PF 機能の有効状態を返す。"""
        ...

    @property
    def feature_id(self) -> str:
        """現在の観測特徴量 ID を返す。"""
        ...

    @property
    def board_size(self) -> int:
        """盤面サイズ `N` を返す。"""
        ...

    @property
    def action_dim(self) -> int:
        """離散行動数を返す。"""
        ...

    @property
    def feature_channels(self) -> int:
        """現在特徴量の観測チャネル数を返す。"""
        ...

    def feature_channels_of(self, feature_id: str) -> int:
        """指定特徴量 ID のチャネル数を返す。

        Args:
            feature_id (str): 参照対象の特徴量 ID。

        Returns:
            int: 観測チャネル数。
        """
        ...

    def set_pf_enabled(self, v: bool) -> None:
        """PF 機能の有効/無効を切り替える。

        Args:
            v (bool): 有効化状態。
        """
        ...

    def set_feature_id(self, feature_id: str) -> None:
        """観測特徴量 ID を切り替える。

        Args:
            feature_id (str): 新しい特徴量 ID。
        """
        ...

    def reset_random(self, seeds: torch.Tensor) -> None:
        """各環境をランダム seed で初期化する。

        Args:
            seeds (torch.Tensor): 環境ごとの seed（`[B]`）。
        """
        ...

    def reset_from_tools(self, paths: Sequence[str], pf_seeds_extra: torch.Tensor | None = None) -> None:
        """入力ファイル列から環境を初期化する。

        Args:
            paths (Sequence[str]): `tools/in` 形式の入力ファイルパス列。
            pf_seeds_extra (torch.Tensor | None): PF 用追加 seed（`[B]`）。
        """
        ...

    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        """現在状態の観測と合法手マスクを返す。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(board, mask)`。
        """
        ...

    def observe_into(self, board: torch.Tensor, mask: torch.Tensor) -> None:
        """現在状態の観測とマスクを既存テンソルへ書き込む。

        Args:
            board (torch.Tensor): 観測書き込み先。
            mask (torch.Tensor): 合法手マスク書き込み先。
        """
        ...

    def observe_into_feature(self, board: torch.Tensor, mask: torch.Tensor, feature_id: str) -> None:
        """指定特徴量で観測を生成して書き込む。

        Args:
            board (torch.Tensor): 観測書き込み先。
            mask (torch.Tensor): 合法手マスク書き込み先。
            feature_id (str): 生成に使う特徴量 ID。
        """
        ...

    def observe_pair_into(
        self,
        board_a: torch.Tensor,
        board_b: torch.Tensor,
        mask: torch.Tensor,
        *,
        feature_id_a: str,
        feature_id_b: str,
    ) -> None:
        """2 種類の特徴量観測を同時に生成して書き込む。

        Args:
            board_a (torch.Tensor): 特徴量 A 観測書き込み先。
            board_b (torch.Tensor): 特徴量 B 観測書き込み先。
            mask (torch.Tensor): 合法手マスク書き込み先。
            feature_id_a (str): 特徴量 A の ID。
            feature_id_b (str): 特徴量 B の ID。
        """
        ...

    def aux_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """補助学習ターゲットを返す。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                行動分布、相手パラメータ、有効マスク。
        """
        ...

    def aux_targets_into(self, move_dist: torch.Tensor, opp_param: torch.Tensor, opp_valid: torch.Tensor) -> None:
        """補助学習ターゲットを既存テンソルへ書き込む。

        Args:
            move_dist (torch.Tensor): 行動分布ターゲット書き込み先。
            opp_param (torch.Tensor): 相手パラメータ書き込み先。
            opp_valid (torch.Tensor): 有効マスク書き込み先。
        """
        ...

    def bayes_params(self) -> torch.Tensor:
        """ベイズ推定パラメータを返す。

        Returns:
            torch.Tensor: ベイズ推定パラメータ。
        """
        ...

    def bayes_params_into(self, bayes_params: torch.Tensor) -> None:
        """ベイズ推定パラメータを既存テンソルへ書き込む。

        Args:
            bayes_params (torch.Tensor): 書き込み先テンソル。
        """
        ...

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """環境を 1 ステップ進める。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(reward, done)`。
        """
        ...

    def step_into(self, actions: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        """1 ステップ進め、報酬と終了フラグを書き込む。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。
            reward (torch.Tensor): 報酬書き込み先。
            done (torch.Tensor): 終了フラグ書き込み先。
        """
        ...

    def step_observe_into(
        self,
        actions: torch.Tensor,
        board: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """次観測・マスク・報酬・終了フラグを同時に書き込む。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。
            board (torch.Tensor): 次観測書き込み先。
            mask (torch.Tensor): 次合法手マスク書き込み先。
            reward (torch.Tensor): 報酬書き込み先。
            done (torch.Tensor): 終了フラグ書き込み先。
        """
        ...

    def step_observe_aux_into(
        self,
        actions: torch.Tensor,
        board: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        move_dist: torch.Tensor,
        opp_param: torch.Tensor,
        opp_valid: torch.Tensor,
    ) -> None:
        """補助ターゲット込みで 1 ステップ遷移情報を書き込む。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。
            board (torch.Tensor): 次観測書き込み先。
            mask (torch.Tensor): 次合法手マスク書き込み先。
            reward (torch.Tensor): 報酬書き込み先。
            done (torch.Tensor): 終了フラグ書き込み先。
            move_dist (torch.Tensor): 行動分布ターゲット書き込み先。
            opp_param (torch.Tensor): 相手パラメータ書き込み先。
            opp_valid (torch.Tensor): 有効マスク書き込み先。
        """
        ...

    def step_observe_pair_into(
        self,
        actions: torch.Tensor,
        board_a: torch.Tensor,
        board_b: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        *,
        feature_id_a: str,
        feature_id_b: str,
    ) -> None:
        """2 特徴量観測を同時生成しつつ 1 ステップ進める。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。
            board_a (torch.Tensor): 特徴量 A 観測書き込み先。
            board_b (torch.Tensor): 特徴量 B 観測書き込み先。
            mask (torch.Tensor): 次合法手マスク書き込み先。
            reward (torch.Tensor): 報酬書き込み先。
            done (torch.Tensor): 終了フラグ書き込み先。
            feature_id_a (str): 特徴量 A の ID。
            feature_id_b (str): 特徴量 B の ID。
        """
        ...

    def pos0(self) -> torch.Tensor:
        """初期位置情報を返す。

        Returns:
            torch.Tensor: 初期位置情報テンソル。
        """
        ...

    def official_score(self) -> torch.Tensor:
        """公式スコアを返す。

        Returns:
            torch.Tensor: 各環境の公式スコア。
        """
        ...

    def score_s0_sa(self) -> torch.Tensor:
        """`score_s0_sa` 系の評価値を返す。

        Returns:
            torch.Tensor: 各環境の評価値。
        """
        ...

    def m_u(self) -> torch.Tensor:
        """問題インスタンスごとの `(m, u)` を返す。

        Returns:
            torch.Tensor: 形状 `[B, 2]` の `(m, u)`。
        """
        ...


def ensure_batch_env(env: object) -> BatchEnvProtocol:
    """`BatchEnvProtocol` を満たす環境オブジェクトとして扱えることを保証する。

    Args:
        env (object): 検証対象オブジェクト。

    Returns:
        BatchEnvProtocol: `BatchEnvProtocol` として扱える環境。

    Raises:
        TypeError: `BatchEnvProtocol` を満たさない場合。
    """
    if isinstance(env, BatchEnvProtocol):
        return cast(BatchEnvProtocol, env)
    # Better error than AttributeError deeper in rollout loop.
    raise TypeError(f"env does not satisfy BatchEnvProtocol: {type(env)!r}")
