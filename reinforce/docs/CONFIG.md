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
  - `casegen_enable`, `casegen_num_cases`, `casegen_seed_mode`, `casegen_seed_start`
- PPO
  - `ppo_total_timesteps`, `ppo_num_envs`, `ppo_num_steps`
  - `ppo_learning_rate`, `ppo_learning_rate_schedule`
  - `ppo_clip_coef`, `ppo_clip_range_vf`, `ppo_ent_coef`, `ppo_target_kl`
  - `ppo_num_minibatches`, `ppo_update_epochs`
- PPO中 val
  - `ppo_val_interval_steps`, `ppo_val_episodes`, `ppo_val_seed_start`
  - `ppo_val_at_start`, `ppo_val_fixed_seeds`
- 最終 eval
  - `eval_episodes`, `eval_env_kwargs_json`, `skip_last_eval`
- モデル
  - `model_class`, `model_config_file`, `model_config_json`

## 運用メモ

- `model_config_file` / `model_config_json` は `type + kwargs` 形式です。
- `bayes_backend=cpp` を使う場合は `pybind11` が必要です。
- `train_ppo` の resume は `num_envs` / `num_steps` を checkpoint 作成時と一致させてください。
- PPO は `batch_size=num_envs*num_steps` 単位で更新するため、`global_step` が `total_timesteps` を少し超えて終了する場合があります。

