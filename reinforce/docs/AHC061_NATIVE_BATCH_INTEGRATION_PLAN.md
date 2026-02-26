# AHC061 Native BatchEnv Integration Plan

最終更新: 2026-02-26

## 1. 目的

この計画は、AHC061 の PPO 学習環境を C++ native 化しつつ、
学習モデル・実験管理・運用は `reinforce` 側資産を主系として維持するための実装方針を定義する。

前提:

- 学習モデルは `reinforce/ppo_discrete/models/*` を使う
- 実験管理は `reinforce/ppo_discrete/runtime/*` と既存 CLI/config を使う
- `exp002` 側は「BatchEnv interface 設計の参照元」に限定する

## 2. スコープ

### 2.1 実施すること

- `reinforce` 配下に C++ native BatchEnv 層を持つ
- PPO 学習時の rollout 収集を native backend に切り替え可能にする
- 既存の checkpoint/metrics/run layout 運用を維持する

### 2.2 実施しないこと

- `exp002` 側の Python 学習スクリプトやモデル実装の直接利用
- `exp002` ディレクトリを runtime 依存として参照する設計
- `reinforce` の実験管理基盤を別系に置き換えること

## 3. 固定仕様

`exp002` の Env 仕様を次で固定する。

- 不正手: `step` で例外送出（補正 stay / penalty なし）
- 報酬: `reward = phi_{t+1} - phi_t`
- `phi`: `log(1 + s0 / (sa + 1e-9))`
- 行動マスク: C++ 側で合法手を返し、policy 側で mask 適用

## 4. アーキテクチャ

### 4.1 レイヤ分割

- Env core (C++):
  `reinforce/ppo_discrete/domains/ahc061/native_batch/cpp_core/include/ahc061/core/*.hpp`
- Torch extension (C++):
  `reinforce/ppo_discrete/domains/ahc061/native_batch/cpp_ext/src/ahc061_ext.cpp`
- Python wrapper:
  `reinforce/ppo_discrete/domains/ahc061/native_batch/env.py`
- Rollout collector:
  `reinforce/ppo_discrete/algorithms/ppo/native_rollout.py`
- Trainer entrypoint:
  `reinforce/ppo_discrete/cli/train_ppo.py`（最終統合先）

### 4.2 backend 切替方針

`train_ppo` に rollout backend を導入し、同一 CLI 内で分岐する。

- `rollout_backend=gym`（既存）
- `rollout_backend=native`（追加）

`native` 選択時のみ `BatchEnv` を生成し、`collect_native_rollout()` を呼ぶ。
PPO 更新 (`PPOTrainer`) と checkpoint 保存は共通化する。

## 5. `exp002` 思想との差分（2026-02-26時点）

### 5.1 既に一致している点

- C++ core + torch extension + Python wrapper を `reinforce` 配下へ内製化済み
- `BatchEnv` interface は `exp002` 準拠（`observe_into`, `step_observe_into`, `step_observe_aux_into`）
- 不正手例外・`delta phi` 報酬の仕様を固定済み
- `feature_id` 切替・`pf_enabled` 切替が可能

### 5.2 不足している点（重要）

- `train_ppo` 本体に native backend 分岐が未統合（現状は `train_ppo_native` 分離）
- `run_pipeline` から native PPO を選べない
- rollout の高性能化要素が不足
  - workspace 再利用
  - pinned memory
  - channels_last 方針
  - rollout cache device 方針
- periodic eval / fixed-seed eval の native 経路統合が未完
- resume strictness を native 経路で `train_ppo` 同等に未整備

### 5.3 進捗（2026-02-26）

- Phase A/B/C は実装済み（`train_ppo` native dispatch、`run_pipeline` 透過、rollout 高速化）
- Phase D の中核は実装済み
  - `train_ppo_native` に `--distributed {auto,off,on}` を追加
  - `torchrun` 実行時に rank/local_rank/world_size を解釈して local batch 分割
  - DDP wrap + all-reduce によるメトリクス集計
  - rank-aware seed、rank0 限定 checkpoint/manifest
- `train_ppo` / `run_pipeline` から native distributed 設定を透過可能
- model/feature 管理の整理を開始
  - model preset catalog: `reinforce/ppo_discrete/models/catalog.py`
  - native feature catalog: `reinforce/ppo_discrete/domains/ahc061/native_batch/feature_catalog.py`

### 5.4 まだ残る差分（統合上の要対応）

- `teacher_p0_v1_88ch` native feature は実装済み（C++ core）
- `TeacherP0V1BoardAgent` preset も native 対応済み（`model_preset=teacher_p0_v1_88ch`）
- collect/eval の Gym 依存 CLI は native 化未着手
### 5.5 方針上の扱い

- `exp002` の「Env高速化思想」は取り込む
- `exp002` の「学習ループ実装」は直接流用せず、`reinforce` の運用基盤へ統合する
- 追加機能はすべて `reinforce` 側 API/設定/ログ体系に合わせる

## 6. 再整理した実装ステップ

### Phase A（最優先）: `train_ppo` への統合

対象: `reinforce/ppo_discrete/cli/train_ppo.py`

- `rollout_backend=gym|native` を追加
- `native_feature_id`, `native_pf_enabled`, `native_amp` を追加
- rollout 収集のみ分岐し、更新/保存/ログは共通化

完了条件:

- 既存 `train_ppo` で `gym/native` の両経路が動く
- `train_ppo_native` は検証用サブコマンドへ降格できる状態

### Phase B: 実験管理統合

対象: `run_pipeline.py`, pipeline config

- `ppo_rollout_backend` 系キー追加
- native PPO 実行を pipeline から選択可能にする
- 既存 collect/BC/eval フローとの接続を維持

完了条件:

- `run_pipeline` 1コマンドで native PPO 学習まで回せる

### Phase C: `exp002` 水準の性能機能を移植

対象: native rollout 実装

- workspace 再利用
- pinned memory と non-blocking copy
- channels_last 適用方針
- rollout cache on CPU/GPU 選択
- （必要なら）fused step+observe+aux 経路の本格利用

完了条件:

- `gym` 比で明確な SPS 改善
- 設定で性能チューニング可能

### Phase D: DDP / マルチGPU対応

対象: `train_ppo` native backend

- `torchrun` 前提の rank 分割
- `local_batch_size` / `local_minibatch` 分割
- 集計値の `all_reduce` 整備
- 再現性 seed 設計（rank-aware）

完了条件:

- 単一GPUと複数GPUで同じ CLI 系で実行可能
- スケール時に品質退行がない

### Phase E: 検証・ベンチ標準化

#### E.1 正しさ検証

- native env smoke test
- fixed seed の短い rollout 再現テスト
- 不正手例外が発生しないことの監視

#### E.2 速度計測

- 指標: SPS, rollout時間, update時間
- 比較軸: `backend`, `num_envs`, `num_steps`, `num_minibatches`, `update_epochs`
- 結果を `reinforce/docs/THROUGHPUT_BENCH_*.md` へ追記

## 7. 変更対象ファイル（優先度順）

1. `reinforce/ppo_discrete/cli/train_ppo.py`
2. `reinforce/ppo_discrete/algorithms/ppo/native_rollout.py`
3. `reinforce/ppo_discrete/domains/ahc061/native_batch/*`
4. `reinforce/ppo_discrete/cli/run_pipeline.py`
5. `reinforce/configs/ppo_discrete/*.toml`
6. `reinforce/docs/*.md`

## 8. リスクと対策

- C++ extension build 失敗（`ninja`, compiler）
  - 対策: setup 手順の明文化、起動時エラーのガイド表示
- model と feature 次元不一致
  - 対策: config 固定 + 起動時バリデーション
- `train_ppo` 統合時の既存回帰
  - 対策: backend 分岐を rollout 収集点に限定
- DDP 導入時の集計不整合
  - 対策: `all_reduce` の対象メトリクスを先に固定してテスト化

## 9. 直近の実行順

1. Phase A（`train_ppo` native統合）
2. Phase B（pipeline統合）
3. Phase C（性能機能移植）
4. Phase E（ベンチ定点化）
5. Phase D（DDP対応）
