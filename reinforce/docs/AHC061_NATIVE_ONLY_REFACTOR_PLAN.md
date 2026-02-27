# AHC061 Native-Only Refactor Plan

最終更新: 2026-02-26

## 目的

- Gym 依存を段階的に排除し、AHC061 学習経路を native BatchEnv へ統一する。
- feature / model 管理を共通カタログ化し、重複実装を減らす。
- Python 実装の feature / model（StudentM, Teacher）を native 側と同じ単位で扱えるようにする。

## 現在の到達点

- PPO 学習は `rollout_backend=native` で実行可能（DDP 含む）。
- native feature catalog を追加:
  - `reinforce/ppo_discrete/domain/ahc061/native_batch/feature_catalog.py`
- model preset catalog を追加:
  - `reinforce/ppo_discrete/agents/catalog.py`
- `train_ppo_native` は `--model-preset` を受け付け、native 非対応 preset を明示エラー化。

## 残課題

1. Teacher 経路の native 完了
- `TeacherP0V1BoardAgent` を native feature で学習/評価可能にする
- preset と feature_id の組み合わせ検証を強化

2. CLI の native-only 化
- `train_ppo` / `run_pipeline` の Gym 前提分岐を段階的に削除
- collect/eval も native backend を主系化

3. モジュール整理
- feature spec（channels, submit可否, next_mode）を単一の定義源へ寄せる
- model preset と config file の責務を整理（preset 主体 + 上書き）

## 実装順（推奨）

1. native eval/collect の共通 runner を追加
2. Gym API 依存の削除（`env/factory.py`, `domains/ahc061/env.py` を分離または廃止）
3. feature/model catalog の責務整理を完了

## 進捗メモ（2026-02-26 追加）

- `teacher_p0_v1_88ch` を native C++ feature として実装済み
- `model_preset=teacher_p0_v1_88ch` は native 対応へ昇格済み
- `eval_policy` に `rollout_backend=native` を追加（AHC061 native BatchEnv で評価可能）
- `run_pipeline` の最終 `eval_policy` 呼び出しで、`ppo_rollout_backend=native` 時に native 評価経路を使用
- `collect_teacher` に `rollout_backend=native` を追加（native BatchEnv で教師データ収集可能）
- native 反復処理を `native_eval_runner.py` へ切り出し、`eval_policy` / `collect_teacher` で共通利用
- `train_ppo(native)` に resume / periodic val / schedule を統合し、`run_pipeline` から native resume を利用可能化
- native 経路専用の `native_vecnorm.py` を追加し、学習・評価の `vecnormalize_state` 復元を統合
