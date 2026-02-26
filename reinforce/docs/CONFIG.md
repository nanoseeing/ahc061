# ppo_discrete Config Templates

最終更新: 2026-02-26

## 構成ファイル

`reinforce/configs/ppo_discrete/`:

- ジョブ default
  - `train_ppo.default.toml`
  - `train_ppo_native.default.toml`
  - `train_bc.default.toml`
  - `collect_teacher.default.toml`
  - `evaluate_policy.default.toml`
- パイプライン
  - `pipeline/run_pipeline.default.toml`
  - `pipeline/run_pipeline.ahc061_*.toml`
- モデル定義
  - `model/model.ahc061_cnn_full.toml`
  - `model/model.ahc061_student_m.toml`
  - `model/model.ahc061_native_submit_v1_student_m.toml`
  - `model/model.ahc061_teacher_p0_v1.toml`
  - `model/model.cnn_fc.example.toml`

## 基本的な使い方

```bash
# train_ppo
python -m reinforce.ppo_discrete.cli.train_ppo \
  --config-file reinforce/configs/ppo_discrete/train_ppo.default.toml

# run_pipeline
python -m reinforce.ppo_discrete.cli.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml

# train_ppo_native (AHC061 native BatchEnv)
python -m reinforce.ppo_discrete.cli.train_ppo_native \
  --config-file reinforce/configs/ppo_discrete/train_ppo_native.default.toml
```

`--set key=value` で上書きできます。

```bash
python -m reinforce.ppo_discrete.cli.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml \
  --set env_id=AHC061Local-v0 \
  --set env_kwargs_json='{"bayes_num_particles":128,"bayes_backend":"cpp"}' \
  --set casegen_enable=true \
  --set casegen_num_cases=5000
```

## AHC061 でよく使うキー

- 収集・環境
  - `env_kwargs_json`
  - collect backend: `collect_rollout_backend`, `collect_native_feature_id`, `collect_native_pf_enabled`, `collect_native_amp`, `collect_native_save_aux_targets`
  - `casegen_enable`, `casegen_num_cases`, `casegen_seed_mode`, `casegen_seed_start`
- BC
  - `bc_aux_opp_param_loss_coef`, `bc_aux_opp_param_use_valid_mask`
- PPO
  - `ppo_total_timesteps`, `ppo_num_envs`, `ppo_num_steps`
  - `ppo_learning_rate`, `ppo_learning_rate_schedule`
  - `ppo_clip_coef`, `ppo_clip_range_vf`, `ppo_ent_coef`, `ppo_target_kl`
  - native aux loss: `ppo_aux_opp_param_loss_coef`, `ppo_aux_opp_param_use_valid_mask`
  - `ppo_num_minibatches`, `ppo_update_epochs`
  - native backend: `ppo_rollout_backend=native`, `ppo_native_feature_id`, `ppo_native_pf_enabled`
  - native perf/DDP: `ppo_native_memory_format`, `ppo_native_pin_memory`, `ppo_native_rollout_cache_device`, `ppo_native_distributed`
  - native model preset: `ppo_native_model_preset`
- PPO中 val
  - `ppo_val_interval_steps`, `ppo_val_episodes`, `ppo_val_seed_start`
  - `ppo_val_at_start`, `ppo_val_fixed_seeds`
- 最終 eval
  - `eval_episodes`, `eval_env_kwargs_json`, `skip_last_eval`
- モデル
  - `model_class`, `model_config_file`, `model_config_json`

## 運用メモ

- `model_config_file` / `model_config_json` は `type + kwargs` 形式です。
- native PPO では `model_preset`（例: `student_m_submit_v1`）でモデル定義を固定できます。
- `train_ppo_native` / `train_ppo --rollout-backend native` は `--resume` / `--resume-from` に対応し、`num_envs` / `num_steps` / `batch_size` 一致を厳格チェックします。
- native PPO は `eval_interval_steps` / `eval_episodes` / `eval_fixed_seeds` / `eval_at_start` による periodic val を利用できます。
- native PPO は `vecnorm`（obs/reward 正規化）に対応し、checkpoint の `vecnormalize_state` に保存・resume 復元されます。
- `run_pipeline` は backend 組み合わせを事前検証します（例: collect native は AHC061 + sync 必須、`ppo_native_distributed=on` は不可、native-only 評価で `eval_casegen_*` は不可）。
- `collect_teacher --rollout-backend native` は `bayes_params` を 28次元ゼロ埋めで保存します（BC互換のため、現状は学習で未使用）。
- `collect_teacher --native-save-aux-targets` を有効にすると、native 収集時に `opp_param_true` / `opp_valid`（別キー）を追加保存できます。
- `train_bc --aux-opp-param-loss-coef > 0` で `opp_param_true` / `opp_valid` を補助損失に使用できます（モデルに aux head を自動有効化）。
- `train_ppo_native --aux-opp-param-loss-coef > 0`（または `train_ppo --rollout-backend native` 同等引数）で、native rollout から `opp_param_true` / `opp_valid` を直接取り込み補助損失を適用できます。
- `bayes_backend=cpp` を使う場合は `pybind11` が必要です。
- `train_ppo` の resume は `num_envs` / `num_steps` を checkpoint 作成時と一致させてください。
- PPO は `batch_size=num_envs*num_steps` 単位で更新するため、`global_step` が `total_timesteps` を少し超えて終了する場合があります。
