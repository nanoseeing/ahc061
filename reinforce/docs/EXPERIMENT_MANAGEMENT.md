# Experiment Management

最終更新: 2026-02-26

この文書は `reinforce/ppo_discrete` の現行実験管理ルールをまとめます。

## Run Layout

`train_ppo` / `run_pipeline` は以下を自動生成します。  
`train_bc` / `collect_teacher` / `evaluate_policy` も `--run-root` 指定時に同じ構造を使います。

```text
<run_root>/<run_name>/
  manifest.json
  config/
  data/
  models/
  logs/
  reports/
  artifacts/
```

- `manifest.json`: 実験ステータス、設定、成果物パス
- `config/`: 実行時に解決された設定
- `models/`: 学習済み重み（`best.pt`, `last.pt`, `ppo_final.pt` など）
- `logs/`: `metrics.jsonl`, `events.jsonl`, `metrics.latest.json`, `tensorboard/`
- `reports/`: 評価サマリ JSON

## Config Management

config は `json/toml/yaml` に対応します。

- `reinforce/configs/ppo_discrete/train_ppo.default.toml`
- `reinforce/configs/ppo_discrete/train_bc.default.toml`
- `reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml`
- `reinforce/configs/ppo_discrete/collect_teacher.default.toml`
- `reinforce/configs/ppo_discrete/evaluate_policy.default.toml`

CLI 上書き:

```bash
--set key=value
```

## Logging / Tracking

- ローカル構造化ログ
  - `logs/metrics.jsonl`
  - `logs/events.jsonl`
- TensorBoard
  - `train_ppo` は既定で有効
  - `--no-tensorboard` で無効化
- MLflow
  - `--mlflow-tracking-uri` 指定時に有効

## Notes

- `run_pipeline` は PPO 出力からモデルを解決し、
  パイプライン run 配下の `models/ppo_final.pt` に集約コピーします。
- モデル探索順は `models/best.pt` -> `models/last.pt` -> `best.pt` -> `last.pt` です。

