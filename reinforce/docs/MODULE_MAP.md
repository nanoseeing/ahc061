# PPO Discrete Module Map (Current)

最終更新: 2026-02-27

この文書は `reinforce/ppo_discrete` の現行モジュール構成のみを記述します。

## ディレクトリ構成

```text
reinforce/ppo_discrete/
  core/ppo/
    config.py
    rollout.py
    vecnorm.py
    rollout_buffer.py
    trainer.py
    train_utils.py
  domain/ahc061/
    batch_env/
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
    teacher_dataset.py
    cpp_bayes/
      opponent_bayes_cpp.cpp
      build_opponent_bayes_cpp.py
  agents/
    discrete_board.py
    student_m.py
    catalog.py
    nn_init.py
    builder.py
    registry.py
  infra/
    checkpoint.py
    config_utils.py
    experiment.py
    log_utils.py
    metrics.py
    tracking.py
  usecases/
    eval_service.py
    model_checkpoint_service.py
    pipeline_commands.py
    pipeline_service.py
    ppo_requests.py
    ppo_service.py
    teacher_dataset_merge.py
  entrypoints/
    collect_teacher.py
    train_ppo.py
    eval_policy.py
    train_bc.py
    run_pipeline.py
```

## 役割

- `core/ppo/*`
  - PPO 更新ロジック、ハイパーパラメータ定義、学習スケジュール/設定検証
- `domain/ahc061/*`
  - AHC061 固有シミュレーション核・BatchEnv・敵パラメータ推定
- `agents/*`
  - モデル定義、`type + kwargs` からの構築
- `infra/*`
  - checkpoint I/O、設定読込、run layout、ログ集計
- `usecases/*`
  - collect/eval/ppo/pipeline の実処理
  - `model_checkpoint_service.py`: モデル復元/保存のアプリ層ロジック
  - `teacher_dataset_merge.py`: 教師 shard の merge/cleanup 共通処理
- `entrypoints/*`
  - 実行エントリポイント（5本）

## パイプラインの実行フロー

`entrypoints/run_pipeline.py` は parser のみを持ち、実処理は `usecases/pipeline_service.py` が担当します。処理順は以下です。

1. `collect_teacher.py`
2. `train_bc.py`
3. `train_ppo.py`（内部で periodic val を実行）
4. `eval_policy.py`（最終評価）

## 代表コマンド

```bash
# 一括実行
python -m reinforce.ppo_discrete.entrypoints.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml

# 個別実行
python -m reinforce.ppo_discrete.entrypoints.train_ppo \
  --config-file reinforce/configs/ppo_discrete/train_ppo.default.toml

# 最終評価
python -m reinforce.ppo_discrete.entrypoints.eval_policy \
  --config-file reinforce/configs/ppo_discrete/evaluate_policy.default.toml
```
