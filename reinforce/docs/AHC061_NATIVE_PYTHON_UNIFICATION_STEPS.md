# AHC061 Python Native Unification Steps

最終更新: 2026-02-26

## 目的

- Python 側の Gym/native 二重実装を段階的に整理し、native 主系の経路を増やす。
- 既存の `run_pipeline` 運用を維持したまま、collect/eval を native でも実行可能にする。

## Step 一覧

### Step 1. native eval の共通 runner 化

- 状態: `done`
- 実施:
  - `reinforce/ppo_discrete/cli/native_eval_runner.py` を追加
  - `run_native_policy_episodes()` を実装
  - `eval_policy` native 経路を runner 利用へ置換

### Step 2. collect_teacher に native backend 追加

- 状態: `done`
- 実施:
  - `collect_teacher` に `--rollout-backend`, `--native-*` を追加
  - native 経路を `native_eval_runner` 経由に統合
  - 既存 NPZ 形式（`obs/action/reward/done/episode/step/bayes_params`）を維持

### Step 3. run_pipeline から collect native を選択可能化

- 状態: `done`
- 実施:
  - `run_pipeline` に以下を追加:
    - `collect_rollout_backend`
    - `collect_native_feature_id`
    - `collect_native_pf_enabled`
    - `collect_native_amp`
  - collect worker 実行コマンドへ backend/native 引数を透過
  - AHC061 の `random -> ahc061_main_greedy` 自動補正を gym collect 時のみ適用

### Step 4. AHC061 casegen 必須条件の整理

- 状態: `done`
- 実施:
  - `run_pipeline` の AHC061 `casegen_enable` 必須条件を緩和
  - gym を使う stage（collect/ppo/final eval）が存在する場合のみ必須化
  - native-only collect 実行では casegen なしで実行可能に変更

### Step 5. config/docs 反映

- 状態: `done`
- 実施:
  - `run_pipeline.default.toml` に collect native 設定を追記
  - `CONFIG.md`, `README.md`, `MODULE_MAP.md`, `AHC061_NATIVE_ONLY_REFACTOR_PLAN.md` を更新

### Step 6. スモーク検証

- 状態: `done`
- 実施コマンド:
  - `python -m py_compile reinforce/ppo_discrete/cli/native_eval_runner.py reinforce/ppo_discrete/cli/eval_policy.py reinforce/ppo_discrete/cli/collect_teacher.py reinforce/ppo_discrete/cli/run_pipeline.py`
  - `python -m reinforce.ppo_discrete.cli.eval_policy --rollout-backend native ...`
  - `python -m reinforce.ppo_discrete.cli.collect_teacher --rollout-backend native ...`
  - `python -m reinforce.ppo_discrete.cli.collect_teacher --rollout-backend gym ...`
  - `python -m reinforce.ppo_discrete.cli.run_pipeline ... --set collect_rollout_backend=native ... --set skip_bc=true --set skip_ppo=true --set skip_last_eval=true`
  - `python -m reinforce.ppo_discrete.cli.run_pipeline ... --set collect_rollout_backend=gym ... --set skip_bc=true --set skip_ppo=true --set skip_last_eval=true`

### Step 7. train_ppo(native) の resume 統合

- 状態: `done`
- 実施:
  - `native_ppo_runner` に `--resume`, `--resume-from` を追加
  - checkpoint の `global_step` / `cfg_global` から再開し、`num_envs` / `num_steps` / `batch_size` 不一致をエラー化
  - `run_pipeline` の `ppo_rollout_backend=native` resume 制限を解除

### Step 8. train_ppo(native) の periodic val 統合

- 状態: `done`
- 実施:
  - `native_ppo_runner` に `eval_interval_steps`, `eval_episodes`, `eval_seed_start`, `eval_fixed_seeds`, `eval_at_start`, `eval_deterministic` を追加
  - `native_eval_runner.run_native_policy_episodes()` を使って periodic val を実行
  - `logs/periodic_val_metrics.jsonl` と `reports/train_summary.json` / `summary.json` を出力

### Step 9. schedule/運用出力の parity 拡張

- 状態: `done`
- 実施:
  - native PPO で `learning_rate` / `ent_coef` / `clip_coef` / `clip_range_vf` schedule を適用
  - `logs/train_metrics.jsonl` と `models/best.pt`, `models/last.pt` を更新
  - `train_ppo_native.default.toml` と docs を更新

### Step 10. native vecnorm 統合

- 状態: `done`
- 実施:
  - Gym 非依存 `native_vecnorm.py` を追加し、`collect_native_rollout` に接続
  - `train_ppo(native)` / `train_ppo_native` で `--vecnorm*` を有効化
  - checkpoint meta (`vecnormalize_state`) の保存/復元に native 経路を統合
  - `eval_policy` native 経路で `vecnorm-mode=auto|on|off` を有効化

### Step 11. run_pipeline backend 整合チェック強化

- 状態: `done`
- 実施:
  - `run_pipeline` 起動直後に backend 組み合わせを早期検証
  - 例:
    - collect native は `env_id=AHC061Local-v0` / `collect_vector_env=sync` 必須
    - `collect_policy=ahc061_main_greedy` は collect native で禁止
    - `ppo_native_distributed=on` は run_pipeline では禁止（torchrun未使用）
    - native-only eval 構成で `eval_casegen_*` 指定を禁止

### Step 12. native collect `bayes_params` 方針の固定

- 状態: `done`
- 実施:
  - `runtime/teacher_dataset.py` を追加し、`bayes_params` 既定形（28次元）を共通化
  - `collect_teacher` の native 経路は `bayes_params` をゼロ埋めで固定（NPZ互換維持）
  - `collect_teacher` meta に `bayes_param_sources` / `bayes_param_policy` を追加し、由来を明示
  - `run_pipeline` の shard merge 側も同じ既定形を参照するよう統一

### Step 13. native collect 補助教師信号の別キー拡張

- 状態: `done`
- 実施:
  - `collect_teacher --native-save-aux-targets` を追加（native backendのみ）
  - native収集時に `opp_param_true` / `opp_valid` を optional key として NPZ 保存
  - `run_pipeline` の collect worker merge でも optional key を保持
  - `bayes_params` は引き続きゼロ埋め（互換維持）、補助信号は別キーで扱う方針を固定

### Step 14. interface整合の型拡張とBC側可視化

- 状態: `done`
- 実施:
  - `NativeBatchEnvProtocol` を native wrapper/C++公開APIに合わせて拡張（aux系含む）
  - `train_bc` が dataset の optional native aux key（`opp_param_true` / `opp_valid`）を検知し、metaへ出力
  - 現状 `train_bc` は aux key を損失に使わないことをログ明示

### Step 15. BCのnative補助損失接続

- 状態: `done`
- 実施:
  - `train_bc --aux-opp-param-loss-coef` / `--aux-opp-param-use-valid-mask` を追加
  - `opp_param_true` / `opp_valid` を使った補助MSE損失を実装
  - `run_pipeline` から BC補助損失設定を透過 (`bc_aux_opp_param_*`)
  - モデル側に `get_aux_opp_param()` / `aux_opp_param_head`（StudentM/Teacher/DiscreteBoard）を追加

### Step 16. PPO(native) の補助損失接続

- 状態: `done`
- 実施:
  - `native_rollout` に optional aux 収集を追加（`step_observe_aux_into` 経由）
  - `RolloutBuffer`/`PPOTrainer` を `opp_param_true` / `opp_valid` 補助損失対応へ拡張
  - `train_ppo_native` / `train_ppo --rollout-backend native` に `--aux-opp-param-loss-coef` / `--aux-opp-param-use-valid-mask` を追加
  - `run_pipeline` から PPO補助損失設定を透過 (`ppo_aux_opp_param_*`)
  - 補助損失有効時はモデル aux head を自動有効化

## 残課題（次段）

- move_dist など追加補助信号を使う場合の loss 設計整理
