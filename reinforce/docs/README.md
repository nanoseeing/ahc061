# ppo_discrete

`reinforce/ppo_discrete` は、離散行動向け PPO 学習基盤です。
現行の実装対象は以下です。

- 汎用 Gymnasium 離散環境（例: CartPole）
- AHC061 ローカル環境（`AHC061Local-v0`）

## 主要エントリポイント

- `python -m reinforce.ppo_discrete.cli.run_pipeline`
  - collect -> BC -> PPO(val) -> eval を一括実行
- `python -m reinforce.ppo_discrete.cli.collect_teacher`
  - 教師データ収集（AHC061 は `--rollout-backend native` 必須）
- `python -m reinforce.ppo_discrete.cli.train_bc`
  - BC 事前学習
- `python -m reinforce.ppo_discrete.cli.train_ppo`
  - PPO 学習
- `python -m reinforce.ppo_discrete.cli.train_ppo_native`
  - AHC061 native C++ BatchEnv で PPO 学習（Gym 非依存）
- `python -m reinforce.ppo_discrete.cli.eval_policy`
  - 最終評価
- `python -m reinforce.ppo_discrete.cli.export_submission_main_py`
- `python -m reinforce.ppo_discrete.cli.export_submission_main_cpp`

補助的なスモーク用途 CLI:

- `python -m reinforce.ppo_discrete.smoke_cli.benchmark_infer`

## ドキュメント索引

- 設定テンプレート: `reinforce/docs/CONFIG.md`
- 実験管理: `reinforce/docs/EXPERIMENT_MANAGEMENT.md`
- モジュール構成: `reinforce/docs/MODULE_MAP.md`
- AHC061 モデル仕様（実装済み）: `reinforce/docs/AHC061_MODEL_DESIGN.md`
- AHC061 native BatchEnv 統合計画: `reinforce/docs/AHC061_NATIVE_BATCH_INTEGRATION_PLAN.md`
- AHC061 native-only リファクタ計画: `reinforce/docs/AHC061_NATIVE_ONLY_REFACTOR_PLAN.md`
- AHC061 Python native 統合作業Step: `reinforce/docs/AHC061_NATIVE_PYTHON_UNIFICATION_STEPS.md`
- スループット測定記録: `reinforce/docs/THROUGHPUT_BENCH_20260225.md`

## クイックスタート

```bash
python -m reinforce.ppo_discrete.cli.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml
```

AHC061 の軽量実行例:

```bash
python -m reinforce.ppo_discrete.cli.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.ahc061_gamma099_light.toml

# Native BatchEnv (Gym 非依存)
python -m reinforce.ppo_discrete.cli.train_ppo_native \
  --config-file reinforce/configs/ppo_discrete/train_ppo_native.default.toml

# Native BatchEnv multi-GPU (torchrun)
torchrun --standalone --nproc_per_node=4 -m reinforce.ppo_discrete.cli.train_ppo_native \
  --config-file reinforce/configs/ppo_discrete/train_ppo_native.default.toml \
  --distributed on
```

## 補足

- 本リポジトリでは checkpoint に `model_config`（`type + kwargs`）を保存し、
  復元時もこの情報を使ってモデル構造を再現します。
- native 側の feature 一覧は `native_batch/feature_catalog.py` で取得できます。
- モデル preset は `models/catalog.py`（例: `student_m_submit_v1`）で一元管理しています。
- `train_ppo --rollout-backend native` / `train_ppo_native` は periodic val と resume をサポートします。
- native backend の `eval_policy` も `vecnorm-mode`（auto/on/off）をサポートします。
- `run_pipeline` でも `ppo_rollout_backend=native` の resume を利用できます。
- `run_pipeline` は backend 組み合わせを早期検証し、不正構成は stage 実行前にエラー化します。
- `collect_teacher --rollout-backend native` は `bayes_params` を C++ PF posterior（28次元）で保存します（meta に source/policy を出力）。
- `collect_teacher --native-save-aux-targets` で native 補助教師信号（`opp_param_true` / `opp_valid`）を別キー保存できます。
- `train_bc --aux-opp-param-loss-coef` で `opp_param_true` / `opp_valid` を補助損失として学習に利用できます。
- `train_ppo_native --aux-opp-param-loss-coef`（または `train_ppo --rollout-backend native`）でも同じ補助信号を PPO 更新へ統合できます。
- 旧構成向けの設計メモ・未採用案は削除し、文書は現行実装に合わせています。
