# PPO Discrete Module Map (Current)

最終更新: 2026-02-26

この文書は `reinforce/ppo_discrete` の現行モジュール構成のみを記述します。

## ディレクトリ構成

```text
reinforce/ppo_discrete/
  algorithms/ppo/
    config.py
    native_rollout.py
    native_vecnorm.py
    rollout_buffer.py
    trainer.py
  env/
    api.py
    factory.py
    vec_normalize.py
  domains/ahc061/
    env.py
    native_batch/
      env.py
      interface.py
      feature_catalog.py
      cpp_ext.py
      cpp_ext/src/ahc061_ext.cpp
      cpp_core/include/ahc061/core/
        base/
        env/
        features/
        game/
        io/
        opponent/
    opponent_bayes.py
    native/
      opponent_bayes_cpp.cpp
      build_opponent_bayes_cpp.py
  models/
    discrete_board.py
    student_m.py
    catalog.py
    nn_init.py
    builder.py
    registry.py
  runtime/
    checkpoint.py
    config_utils.py
    episode_stats.py
    experiment.py
    log_utils.py
    metrics.py
    teacher_dataset.py
    tracking.py
  cli/
    collect_teacher.py
    train_bc.py
    train_ppo.py
    train_ppo_native.py
    native_ppo_runner.py
    native_eval_runner.py
    eval_policy.py
    run_pipeline.py
    ahc061_agent.py
    export_submission_main_py.py
    export_submission_main_cpp.py
  smoke_cli/
    benchmark_infer.py
```

## 役割

- `algorithms/ppo/*`
  - PPO 更新ロジックとハイパーパラメータ定義
- `env/*`
  - 環境生成、action mask 取り出し、VecNormalize
- `domains/ahc061/*`
  - AHC061 固有環境と敵パラメータ推定
- `models/*`
  - モデル定義、`type + kwargs` からの構築
- `runtime/*`
  - checkpoint、設定読込、run layout、ログ集計
- `cli/*`
  - 学習・評価・提出生成の本番エントリポイント
- `smoke_cli/*`
  - 補助的な計測コマンド（パイプライン本体外）

## パイプラインの実行フロー

`run_pipeline.py` は以下を順に呼び出します。

1. `collect_teacher.py`
2. `train_bc.py`
3. `train_ppo.py`（内部で periodic val を実行）
4. `eval_policy.py`（最終評価）

## 代表コマンド

```bash
# 一括実行
python -m reinforce.ppo_discrete.cli.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml

# 個別実行
python -m reinforce.ppo_discrete.cli.train_ppo \
  --config-file reinforce/configs/ppo_discrete/train_ppo.default.toml

# 最終評価
python -m reinforce.ppo_discrete.cli.eval_policy \
  --config-file reinforce/configs/ppo_discrete/evaluate_policy.default.toml
```
