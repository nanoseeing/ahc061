# ppo_discrete Config Templates

最終更新: 2026-02-26

## 構成ファイル

`reinforce/configs/ppo_discrete/`:

- ジョブ default
  - `train_ppo.default.toml`
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
python -m reinforce.ppo_discrete.entrypoints.train_ppo \
  --config-file reinforce/configs/ppo_discrete/train_ppo.default.toml

# run_pipeline
python -m reinforce.ppo_discrete.entrypoints.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml
```

`--set key=value` で上書きできます。

```bash
python -m reinforce.ppo_discrete.entrypoints.run_pipeline \
  --config-file reinforce/configs/ppo_discrete/pipeline/run_pipeline.default.toml \
  --set env_id=AHC061Local-v0 \
  --set env_kwargs_json='{"bayes_num_particles":128,"bayes_backend":"cpp"}'
```

## AHC061 でよく使うキー

- 収集・環境
  - `env_kwargs_json`
  - `collect_feature_id`, `collect_pf_enabled`, `collect_amp`, `collect_save_aux_targets`
- BC
  - `bc_aux_opp_param_loss_coef`, `bc_aux_opp_param_use_valid_mask`
- PPO
  - `ppo_total_timesteps`, `ppo_num_envs`, `ppo_num_steps`
  - `ppo_learning_rate`, `ppo_learning_rate_schedule`
  - `ppo_clip_coef`, `ppo_clip_range_vf`, `ppo_ent_coef`, `ppo_target_kl`
  - aux loss: `ppo_aux_opp_param_loss_coef`, `ppo_aux_opp_param_use_valid_mask`
  - `ppo_num_minibatches`, `ppo_update_epochs`
  - perf/DDP: `ppo_memory_format`, `ppo_pin_memory`, `ppo_rollout_cache_device`, `ppo_distributed`
  - model preset: `ppo_model_preset`
- PPO中 val
  - `ppo_val_interval_steps`, `ppo_val_episodes`, `ppo_val_seed_start`
  - `ppo_val_at_start`, `ppo_val_fixed_seeds`
- 最終 eval
  - `eval_episodes`, `eval_env_kwargs_json`, `skip_last_eval`
- モデル
  - `model_class`, `model_config_file`, `model_config_json`

## 運用メモ

- `model_config_file` / `model_config_json` は `type + kwargs` 形式です。
- `train_ppo` は `--resume` / `--resume-from` に対応し、`num_envs` / `num_steps` / `batch_size` 一致を厳格チェックします。
- `train_ppo` は `eval_interval_steps` / `eval_episodes` / `eval_fixed_seeds` / `eval_at_start` による periodic val を利用できます。
- `train_ppo` は `vecnorm`（obs/reward 正規化）に対応し、checkpoint の `vecnormalize_state` に保存・resume 復元されます。
- AHC061 (`env_id=AHC061Local-v0`) では `train_ppo` の `num_steps <= 100`（`env.spec.t_max`）が必須です。
- `collect_teacher` は `bayes_params` を C++ 側 PF posterior（28次元）で保存します。
- `collect_teacher --save-aux-targets` を有効にすると、`opp_param_true` / `opp_valid`（別キー）を追加保存できます。
- `train_bc --aux-opp-param-loss-coef > 0` で `opp_param_true` / `opp_valid` を補助損失に使用できます（モデルに aux head を自動有効化）。
- `train_ppo --aux-opp-param-loss-coef > 0` で、rollout から `opp_param_true` / `opp_valid` を取り込み補助損失を適用できます。
- AHC061 の `bayes_backend` は cpp 専用です（`auto` も cpp 解決）。`python` は未サポートです。
- cpp Bayes backend ビルドには `pybind11` が必要です。
- `train_ppo` の resume は `num_envs` / `num_steps` を checkpoint 作成時と一致させてください。
- PPO は `batch_size=num_envs*num_steps` 単位で更新するため、`global_step` が `total_timesteps` を少し超えて終了する場合があります。
