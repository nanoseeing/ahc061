"""C++ 拡張 `BatchEnv` を Python から扱うラッパモジュール。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

from .cpp_ext import load_ext


@dataclass(frozen=True)
class EnvSpec:
    """AHC061 環境の静的仕様を表すデータクラス。

    Attributes:
        n (int): 盤面サイズ（`N`）。
        t_max (int): 1 エピソードあたりの最大ステップ数。
        m_max (int): 問題インスタンスの最大 `M` 値。
    """
    n: int = 10
    t_max: int = 100
    m_max: int = 8


class BatchEnv:
    """C++ 拡張のバッチ環境 API を提供するラッパクラス。"""

    def __init__(
        self,
        batch_size: int,
        *,
        feature_id: str = "submit_v1",
        pf_enabled: bool = True,
        verbose_build: bool = False,
    ):
        """バッチ環境を生成する。

        Args:
            batch_size (int): 並列環境数。
            feature_id (str): 観測特徴量 ID。
            pf_enabled (bool): PF（Particle Filter）機能を使うか。
            verbose_build (bool): 拡張モジュールのビルドログを詳細表示するか。
        """
        self._ext = load_ext(verbose=verbose_build)
        self._env = self._ext.BatchEnv(batch_size=batch_size, feature_id=str(feature_id), pf_enabled=pf_enabled)
        self.spec = EnvSpec()

    @property
    def batch_size(self) -> int:
        """並列環境数を返す。

        Returns:
            int: 並列環境数。
        """
        return int(self._env.batch_size)

    @property
    def pf_enabled(self) -> bool:
        """PF 機能が有効かを返す。

        Returns:
            bool: PF 有効化状態。
        """
        return bool(self._env.pf_enabled)

    @property
    def feature_id(self) -> str:
        """現在使用中の特徴量 ID を返す。

        Returns:
            str: 特徴量 ID。
        """
        return str(self._env.feature_id)

    @property
    def feature_channels(self) -> int:
        """現在の特徴量 ID に対するチャネル数を返す。

        Returns:
            int: 観測チャネル数。
        """
        return int(self._env.feature_channels())

    @property
    def board_size(self) -> int:
        """盤面サイズ `N` を返す。

        Returns:
            int: 盤面サイズ。
        """
        return int(self.spec.n)

    @property
    def action_dim(self) -> int:
        """離散行動数（`N*N`）を返す。

        Returns:
            int: 行動数。
        """
        n = int(self.spec.n)
        return int(n * n)

    def feature_channels_of(self, feature_id: str) -> int:
        """指定特徴量 ID のチャネル数を返す。

        Args:
            feature_id (str): 参照対象の特徴量 ID。

        Returns:
            int: チャネル数。
        """
        return int(self._env.feature_channels_of(str(feature_id)))

    def set_pf_enabled(self, v: bool) -> None:
        """PF 機能の有効/無効を切り替える。

        Args:
            v (bool): 有効化状態。
        """
        self._env.set_pf_enabled(bool(v))

    def set_feature_id(self, feature_id: str) -> None:
        """観測特徴量 ID を切り替える。

        Args:
            feature_id (str): 新しい特徴量 ID。
        """
        self._env.set_feature_id(str(feature_id))

    def reset_random(self, seeds: torch.Tensor) -> None:
        """各環境をランダム seed で初期化する。

        Args:
            seeds (torch.Tensor): 環境ごとの seed（`[B]`）。
        """
        self._env.reset_random(seeds.to(dtype=torch.int64, device="cpu"))

    def reset_from_tools(self, paths: Sequence[str], pf_seeds_extra: Optional[torch.Tensor] = None) -> None:
        """`tools/in` 形式の入力ファイルから環境を初期化する。

        Args:
            paths (Sequence[str]): 入力ファイルパス列。
            pf_seeds_extra (Optional[torch.Tensor]): PF 用追加 seed（`[B]`）。
        """
        if pf_seeds_extra is None:
            self._env.reset_from_tools(list(paths))
            return
        pf_seeds_extra_cpu = pf_seeds_extra.to(dtype=torch.int64, device="cpu")
        self._env.reset_from_tools_seeded(list(paths), pf_seeds_extra_cpu)

    def observe(self):
        """現在状態の観測と合法手マスクを返す。

        Returns:
            Any: 実装依存の観測結果（通常は `(board, mask)`）。
        """
        return self._env.observe()

    def observe_into(self, board: torch.Tensor, mask: torch.Tensor) -> None:
        """現在状態の観測とマスクを既存テンソルへ書き込む。

        Args:
            board (torch.Tensor): 観測書き込み先テンソル。
            mask (torch.Tensor): 合法手マスク書き込み先テンソル。
        """
        self._env.observe_into(board, mask)

    def observe_into_feature(self, board: torch.Tensor, mask: torch.Tensor, feature_id: str) -> None:
        """指定特徴量 ID で観測を生成して書き込む。

        Args:
            board (torch.Tensor): 観測書き込み先テンソル。
            mask (torch.Tensor): 合法手マスク書き込み先テンソル。
            feature_id (str): 生成に使う特徴量 ID。
        """
        self._env.observe_into_feature(board, mask, str(feature_id))

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
            board_a (torch.Tensor): 特徴量 A の観測書き込み先。
            board_b (torch.Tensor): 特徴量 B の観測書き込み先。
            mask (torch.Tensor): 合法手マスク書き込み先。
            feature_id_a (str): 特徴量 A の ID。
            feature_id_b (str): 特徴量 B の ID。
        """
        self._env.observe_pair_into(board_a, board_b, mask, str(feature_id_a), str(feature_id_b))

    def aux_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """補助学習ターゲットを返す。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                行動分布、相手パラメータ、有効マスク。
        """
        return self._env.aux_targets()

    def aux_targets_into(self, move_dist: torch.Tensor, opp_param: torch.Tensor, opp_valid: torch.Tensor) -> None:
        """補助学習ターゲットを既存テンソルへ書き込む。

        Args:
            move_dist (torch.Tensor): 行動分布ターゲット書き込み先。
            opp_param (torch.Tensor): 相手パラメータ書き込み先。
            opp_valid (torch.Tensor): 相手パラメータ有効マスク書き込み先。
        """
        self._env.aux_targets_into(move_dist, opp_param, opp_valid)

    def bayes_params(self) -> torch.Tensor:
        """ベイズ推定パラメータを返す。

        Returns:
            torch.Tensor: ベイズ推定パラメータ。
        """
        return self._env.bayes_params()

    def bayes_params_into(self, bayes_params: torch.Tensor) -> None:
        """ベイズ推定パラメータを既存テンソルへ書き込む。

        Args:
            bayes_params (torch.Tensor): 書き込み先テンソル。
        """
        self._env.bayes_params_into(bayes_params)

    def step(self, actions: torch.Tensor):
        """環境を 1 ステップ進める。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。

        Returns:
            Any: 実装依存の遷移結果（通常は `(reward, done)`）。
        """
        return self._env.step(actions.to(dtype=torch.int64, device="cpu"))

    def step_into(self, actions: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        """1 ステップ進め、報酬と終了フラグを書き込む。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。
            reward (torch.Tensor): 報酬書き込み先（`[B]`）。
            done (torch.Tensor): 終了フラグ書き込み先（`[B]`）。
        """
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_into(actions_cpu, reward, done)

    def step_observe_into(
        self,
        actions: torch.Tensor,
        board: torch.Tensor,
        mask: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """1 ステップ進めて次観測・マスク・報酬・終了フラグを同時に書き込む。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。
            board (torch.Tensor): 次観測書き込み先。
            mask (torch.Tensor): 次合法手マスク書き込み先。
            reward (torch.Tensor): 報酬書き込み先。
            done (torch.Tensor): 終了フラグ書き込み先。
        """
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_observe_into(actions_cpu, board, mask, reward, done)

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
        """`step_observe_into` に加えて補助ターゲットも同時に書き込む。

        Args:
            actions (torch.Tensor): 各環境の行動（`[B]`）。
            board (torch.Tensor): 次観測書き込み先。
            mask (torch.Tensor): 次合法手マスク書き込み先。
            reward (torch.Tensor): 報酬書き込み先。
            done (torch.Tensor): 終了フラグ書き込み先。
            move_dist (torch.Tensor): 行動分布ターゲット書き込み先。
            opp_param (torch.Tensor): 相手パラメータ書き込み先。
            opp_valid (torch.Tensor): 相手パラメータ有効マスク書き込み先。
        """
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_observe_aux_into(actions_cpu, board, mask, reward, done, move_dist, opp_param, opp_valid)

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
        """2 種類の特徴量観測を同時生成しつつ 1 ステップ進める。

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
        actions_cpu = actions.to(dtype=torch.int64, device="cpu")
        self._env.step_observe_pair_into(
            actions_cpu,
            board_a,
            board_b,
            mask,
            reward,
            done,
            str(feature_id_a),
            str(feature_id_b),
        )

    def pos0(self) -> torch.Tensor:
        """初期位置情報を返す。

        Returns:
            torch.Tensor: 初期位置情報テンソル。
        """
        return self._env.pos0()

    def official_score(self) -> torch.Tensor:
        """公式スコアを返す。

        Returns:
            torch.Tensor: 各環境の公式スコア。
        """
        return self._env.official_score()

    def score_s0_sa(self) -> torch.Tensor:
        """`score(s0, sa)` 系の評価値を返す。

        Returns:
            torch.Tensor: 各環境の `score_s0_sa`。
        """
        return self._env.score_s0_sa()

    def m_u(self) -> torch.Tensor:
        """各問題インスタンスの `(m, u)` を返す。

        Returns:
            torch.Tensor: 形状 `[B, 2]` の `(m, u)` テンソル。
        """
        return self._env.m_u()


def tools_input_paths(seed_begin: int, seed_end: int) -> list[str]:
    """`tools/in/{seed:04d}.txt` 形式の入力パス列を生成する。

    Args:
        seed_begin (int): 開始 seed（含む）。
        seed_end (int): 終了 seed（含む）。

    Returns:
        list[str]: seed 昇順の入力ファイルパス列。
    """
    repo_root = Path(__file__).resolve().parents[5]
    base = repo_root / "tools" / "in"
    out: list[str] = []
    for s in range(seed_begin, seed_end + 1):
        out.append(str(base / f"{s:04d}.txt"))
    return out
