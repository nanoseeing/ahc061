# ppo_discrete

`reinforce/ppo_discrete` は、離散行動向け PPO 学習基盤です。
現行の実装対象は以下です。

- 汎用 Gymnasium 離散環境（例: CartPole）
- AHC061 ローカル環境（`AHC061Local-v0`）

## 主要エントリポイント

- `python -m reinforce.ppo_discrete.cli.run_pipeline`
  - collect -> BC -> PPO(val) -> eval を一括実行
- `python -m reinforce.ppo_discrete.cli.collect_teacher`
  - 教師データ収集
- `python -m reinforce.ppo_discrete.cli.train_bc`
  - BC 事前学習
- `python -m reinforce.ppo_discrete.cli.train_ppo`
  - PPO 学習
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
```

## 補足

- 本リポジトリでは checkpoint に `model_config`（`type + kwargs`）を保存し、
  復元時もこの情報を使ってモデル構造を再現します。
- 旧構成向けの設計メモ・未採用案は削除し、文書は現行実装に合わせています。
