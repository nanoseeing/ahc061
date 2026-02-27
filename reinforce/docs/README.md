# ppo_discrete

`reinforce/ppo_discrete` は、離散行動向け PPO 学習基盤です。
現行の実装対象は以下です。

- AHC061 ローカル環境（`AHC061Local-v0`）

## 主要エントリポイント

- `python -m reinforce.ppo_discrete.entrypoints.run_pipeline`
  - collect -> BC -> PPO(val) -> eval を一括実行
- `python -m reinforce.ppo_discrete.entrypoints.collect_teacher`
  - 教師データ収集
- `python -m reinforce.ppo_discrete.entrypoints.train_bc`
  - BC 事前学習
- `python -m reinforce.ppo_discrete.entrypoints.train_ppo`
  - PPO 学習
- `python -m reinforce.ppo_discrete.entrypoints.eval_policy`
  - 最終評価

実装構成の補足:

- `reinforce/ppo_discrete/entrypoints/*` は実行エントリのみ（5本）
- `reinforce/ppo_discrete/usecases/*` は collect/eval/ppo/pipeline 実装本体
- `reinforce/ppo_discrete/infra/*` は checkpoint / run管理 / 記録補助

## ドキュメント索引

- 設定テンプレート: `reinforce/docs/CONFIG.md`
- 実験管理: `reinforce/docs/EXPERIMENT_MANAGEMENT.md`
- モジュール構成: `reinforce/docs/MODULE_MAP.md`
- AHC061 モデル仕様（実装済み）: `reinforce/docs/AHC061_MODEL_DESIGN.md`
- AHC061 BatchEnv 統合計画: `reinforce/docs/AHC061_NATIVE_BATCH_INTEGRATION_PLAN.md`
- AHC061 リファクタ計画: `reinforce/docs/AHC061_NATIVE_ONLY_REFACTOR_PLAN.md`
- AHC061 Python 統合作業Step: `reinforce/docs/AHC061_NATIVE_PYTHON_UNIFICATION_STEPS.md`
- スループット測定記録: `reinforce/docs/THROUGHPUT_BENCH_20260225.md`

## クイックスタート

```bash
python -m reinforce.ppo_discrete.entrypoints.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml
```

AHC061 の軽量実行例:

```bash
python -m reinforce.ppo_discrete.entrypoints.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.ahc061_gamma099_light.toml

# BatchEnv (Gym 非依存)
python -m reinforce.ppo_discrete.entrypoints.train_ppo \
  --config-file reinforce/configs/ppo_discrete/train_ppo.default.toml

# BatchEnv multi-GPU (torchrun)
torchrun --standalone --nproc_per_node=4 -m reinforce.ppo_discrete.entrypoints.train_ppo \
  --config-file reinforce/configs/ppo_discrete/train_ppo.default.toml \
  --distributed on
```

## 補足

- 本リポジトリでは checkpoint に `model_config`（`type + kwargs`）を保存し、
  復元時もこの情報を使ってモデル構造を再現します。
- feature 一覧は `domains/ahc061/batch_env/feature_catalog.py` で取得できます。
- モデル preset は `agents/catalog.py`（例: `student_m_submit_v1`）で一元管理しています。
- `train_ppo` は periodic val と resume をサポートします。
- `eval_policy` は `vecnorm-mode`（auto/on/off）をサポートします。
- `run_pipeline` でも resume を利用できます。
- `collect_teacher` は `bayes_params` を C++ PF posterior（28次元）で保存します（meta に source/policy を出力）。
- `collect_teacher --save-aux-targets` で補助教師信号（`opp_param_true` / `opp_valid`）を別キー保存できます。
- `train_bc --aux-opp-param-loss-coef` で `opp_param_true` / `opp_valid` を補助損失として学習に利用できます。
- `train_ppo --aux-opp-param-loss-coef` でも同じ補助信号を PPO 更新へ統合できます。
- 旧構成向けの設計メモ・未採用案は削除し、文書は現行実装に合わせています。
