from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class ModelArgs(Protocol):
    """Attributes accessed by append_model_args / build_train_bc_cmd / build_train_ppo_cmd."""
    model_class: str
    model_config_file: Path | None
    model_config_json: str


class TrainBCArgs(ModelArgs, Protocol):
    seed: int
    bc_epochs: int
    bc_aux_opp_param_loss_coef: float
    bc_aux_opp_param_use_valid_mask: bool


class TrainPPOArgs(ModelArgs, Protocol):
    env_id: str
    seed: int
    ppo_total_timesteps: int
    ppo_num_envs: int
    ppo_num_steps: int
    ppo_learning_rate: float
    ppo_gamma: float
    ppo_gae_lambda: float
    ppo_num_minibatches: int
    ppo_update_epochs: int
    ppo_clip_coef: float
    ppo_clip_coef_schedule: str
    ppo_clip_coef_final: float | None
    ppo_clip_coef_schedule_expr: str
    ppo_ent_coef: float
    ppo_ent_coef_schedule: str
    ppo_ent_coef_final: float | None
    ppo_ent_coef_schedule_expr: str
    ppo_vf_coef: float
    ppo_aux_opp_param_loss_coef: float
    ppo_aux_opp_param_use_valid_mask: bool
    ppo_max_grad_norm: float
    ppo_checkpoint_interval_steps: int
    ppo_eval_interval_steps: int
    ppo_eval_episodes: int
    ppo_eval_seed_start: int
    ppo_log_interval_iters: int
    ppo_vecnorm_clip_obs: float
    ppo_vecnorm_clip_reward: float
    ppo_vecnorm_epsilon: float
    ppo_feature_id: str
    ppo_norm_adv: bool
    ppo_clip_vloss: bool
    ppo_eval_at_start: bool
    ppo_eval_fixed_seeds: bool
    ppo_eval_deterministic: bool
    ppo_vecnorm: bool
    ppo_vecnorm_norm_obs: bool
    ppo_vecnorm_norm_reward: bool
    ppo_vecnorm_eval_norm_reward: bool
    ppo_amp: bool
    ppo_pin_memory: bool
    ppo_pf_enabled: bool
    ppo_memory_format: str
    ppo_rollout_cache_device: str
    ppo_distributed: str
    ppo_model_preset: str
    ppo_learning_rate_schedule: str
    ppo_clip_range_vf: float | None
    ppo_clip_range_vf_schedule: str
    ppo_clip_range_vf_final: float | None
    ppo_clip_range_vf_schedule_expr: str
    ppo_target_kl: float | None
    ppo_vecnorm_gamma: float | None
    ppo_eval_env_kwargs_json: str
    use_action_mask: bool
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str
    tensorboard: bool


class EvalPolicyArgs(Protocol):
    env_id: str
    eval_episodes: int
    seed: int
    ppo_feature_id: str
    ppo_pf_enabled: bool
    use_action_mask: bool


class PipelineArgs(TrainPPOArgs, TrainBCArgs, EvalPolicyArgs, Protocol):
    """All attributes accessed by pipeline_service.run_pipeline and helpers."""
    run_root: Path
    run_name: str
    resume: bool
    env_kwargs_json: str
    eval_env_kwargs_json: str
    skip_collect: bool
    skip_bc: bool
    skip_ppo: bool
    skip_last_eval: bool
    collect_policy: str
    collect_workers: int
    collect_episodes: int
    collect_chunk_episodes: int
    collect_feature_id: str
    collect_pf_enabled: bool
    collect_amp: bool
    collect_save_aux_targets: bool
    bc_use_collect_shards: bool
    ppo_init_model: Path | None


def append_bool_flag(cmd: list[str], name: str, value: bool) -> None:
    cmd.append(f"--{name}" if bool(value) else f"--no-{name}")


def append_model_args(cmd: list[str], args: ModelArgs) -> None:
    model_class = str(args.model_class).strip()
    model_config_file = str(args.model_config_file).strip() if args.model_config_file is not None else ""
    model_config_json = str(args.model_config_json).strip()

    if model_class:
        cmd += ["--model-class", model_class]
    if model_config_file:
        cmd += ["--model-config-file", model_config_file]
    if model_config_json:
        cmd += ["--model-config-json", model_config_json]


def build_collect_teacher_cmd(
    *,
    py: str,
    env_id: str,
    episodes: int,
    seed: int,
    collect_policy: str,
    chunk_episodes: int,
    output_npz: Path,
    env_kwargs: dict[str, Any],
    feature_id: str,
    pf_enabled: bool,
    amp: bool,
    save_aux_targets: bool,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "reinforce.ppo_discrete.entrypoints.collect_teacher",
        "--env-id",
        str(env_id),
        "--episodes",
        str(int(episodes)),
        "--seed",
        str(int(seed)),
        "--policy",
        str(collect_policy),
        "--chunk-episodes",
        str(int(chunk_episodes)),
        "--output-npz",
        str(output_npz),
        "--env-kwargs-json",
        json.dumps(env_kwargs),
        "--feature-id",
        str(feature_id),
    ]
    append_bool_flag(cmd, "pf-enabled", bool(pf_enabled))
    append_bool_flag(cmd, "amp", bool(amp))
    append_bool_flag(cmd, "save-aux-targets", bool(save_aux_targets))
    return cmd


def build_train_bc_cmd(
    *,
    py: str,
    args: TrainBCArgs,
    output_model: Path,
    dataset_npz: Path | None = None,
    dataset_shards_glob: str = "",
) -> list[str]:
    cmd = [
        py,
        "-m",
        "reinforce.ppo_discrete.entrypoints.train_bc",
        "--output-model",
        str(output_model),
        "--seed",
        str(int(args.seed)),
        "--epochs",
        str(int(args.bc_epochs)),
        "--aux-opp-param-loss-coef",
        str(float(args.bc_aux_opp_param_loss_coef)),
    ]
    append_bool_flag(cmd, "aux-opp-param-use-valid-mask", bool(args.bc_aux_opp_param_use_valid_mask))
    if dataset_shards_glob:
        cmd += ["--dataset-shards-glob", str(dataset_shards_glob)]
    elif dataset_npz is not None:
        cmd += ["--dataset-npz", str(dataset_npz)]
    append_model_args(cmd, args)
    return cmd


def build_train_ppo_cmd(
    *,
    py: str,
    args: TrainPPOArgs,
    run_dir: Path,
    env_kwargs: dict[str, Any],
    ppo_val_env_kwargs_json: str,
    init_model: Path | None = None,
    resume_run_name: str = "",
    resume_from: Path | None = None,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "reinforce.ppo_discrete.entrypoints.train_ppo",
        "--env-id",
        str(args.env_id),
        "--run-dir",
        str(run_dir),
        "--seed",
        str(int(args.seed)),
        "--total-timesteps",
        str(int(args.ppo_total_timesteps)),
        "--num-envs",
        str(int(args.ppo_num_envs)),
        "--num-steps",
        str(int(args.ppo_num_steps)),
        "--learning-rate",
        str(float(args.ppo_learning_rate)),
        "--gamma",
        str(float(args.ppo_gamma)),
        "--gae-lambda",
        str(float(args.ppo_gae_lambda)),
        "--num-minibatches",
        str(int(args.ppo_num_minibatches)),
        "--update-epochs",
        str(int(args.ppo_update_epochs)),
        "--clip-coef",
        str(float(args.ppo_clip_coef)),
        "--clip-coef-schedule",
        str(args.ppo_clip_coef_schedule),
        "--ent-coef",
        str(float(args.ppo_ent_coef)),
        "--ent-coef-schedule",
        str(args.ppo_ent_coef_schedule),
        "--vf-coef",
        str(float(args.ppo_vf_coef)),
        "--aux-opp-param-loss-coef",
        str(float(args.ppo_aux_opp_param_loss_coef)),
        "--max-grad-norm",
        str(float(args.ppo_max_grad_norm)),
        "--checkpoint-interval-steps",
        str(int(args.ppo_checkpoint_interval_steps)),
        "--val-interval-steps",
        str(int(args.ppo_eval_interval_steps)),
        "--val-episodes",
        str(int(args.ppo_eval_episodes)),
        "--val-seed-start",
        str(int(args.ppo_eval_seed_start)),
        "--log-interval-iters",
        str(int(args.ppo_log_interval_iters)),
        "--vecnorm-clip-obs",
        str(float(args.ppo_vecnorm_clip_obs)),
        "--vecnorm-clip-reward",
        str(float(args.ppo_vecnorm_clip_reward)),
        "--vecnorm-epsilon",
        str(float(args.ppo_vecnorm_epsilon)),
        "--env-kwargs-json",
        json.dumps(env_kwargs),
        "--feature-id",
        str(args.ppo_feature_id),
    ]
    append_model_args(cmd, args)
    append_bool_flag(cmd, "norm-adv", bool(args.ppo_norm_adv))
    append_bool_flag(cmd, "clip-vloss", bool(args.ppo_clip_vloss))
    append_bool_flag(cmd, "val-at-start", bool(args.ppo_eval_at_start))
    append_bool_flag(cmd, "use-action-mask", bool(args.use_action_mask))
    append_bool_flag(cmd, "val-fixed-seeds", bool(args.ppo_eval_fixed_seeds))
    append_bool_flag(cmd, "val-deterministic", bool(args.ppo_eval_deterministic))
    append_bool_flag(cmd, "vecnorm", bool(args.ppo_vecnorm))
    append_bool_flag(cmd, "vecnorm-norm-obs", bool(args.ppo_vecnorm_norm_obs))
    append_bool_flag(cmd, "vecnorm-norm-reward", bool(args.ppo_vecnorm_norm_reward))
    append_bool_flag(cmd, "vecnorm-val-norm-reward", bool(args.ppo_vecnorm_eval_norm_reward))
    append_bool_flag(cmd, "aux-opp-param-use-valid-mask", bool(args.ppo_aux_opp_param_use_valid_mask))
    append_bool_flag(cmd, "pf-enabled", bool(args.ppo_pf_enabled))
    append_bool_flag(cmd, "amp", bool(args.ppo_amp))
    cmd += ["--memory-format", str(args.ppo_memory_format)]
    append_bool_flag(cmd, "pin-memory", bool(args.ppo_pin_memory))
    cmd += ["--rollout-cache-device", str(args.ppo_rollout_cache_device)]
    cmd += ["--distributed", str(args.ppo_distributed)]
    if str(args.ppo_model_preset).strip():
        cmd += ["--model-preset", str(args.ppo_model_preset).strip()]
    if str(args.ppo_learning_rate_schedule).strip():
        cmd += ["--learning-rate-schedule", str(args.ppo_learning_rate_schedule)]
    if args.ppo_clip_range_vf is not None:
        cmd += ["--clip-range-vf", str(float(args.ppo_clip_range_vf))]
    if str(args.ppo_clip_range_vf_schedule).strip():
        cmd += ["--clip-range-vf-schedule", str(args.ppo_clip_range_vf_schedule)]
    if args.ppo_clip_range_vf_final is not None:
        cmd += ["--clip-range-vf-final", str(float(args.ppo_clip_range_vf_final))]
    if str(args.ppo_clip_range_vf_schedule_expr).strip():
        cmd += ["--clip-range-vf-schedule-expr", str(args.ppo_clip_range_vf_schedule_expr)]
    if args.ppo_clip_coef_final is not None:
        cmd += ["--clip-coef-final", str(float(args.ppo_clip_coef_final))]
    if str(args.ppo_clip_coef_schedule_expr).strip():
        cmd += ["--clip-coef-schedule-expr", str(args.ppo_clip_coef_schedule_expr)]
    if args.ppo_ent_coef_final is not None:
        cmd += ["--ent-coef-final", str(float(args.ppo_ent_coef_final))]
    if str(args.ppo_ent_coef_schedule_expr).strip():
        cmd += ["--ent-coef-schedule-expr", str(args.ppo_ent_coef_schedule_expr)]
    if args.ppo_target_kl is not None:
        cmd += ["--target-kl", str(float(args.ppo_target_kl))]
    if args.ppo_vecnorm_gamma is not None:
        cmd += ["--vecnorm-gamma", str(float(args.ppo_vecnorm_gamma))]
    if str(ppo_val_env_kwargs_json).strip():
        cmd += ["--val-env-kwargs-json", str(ppo_val_env_kwargs_json)]
    if not bool(args.tensorboard):
        cmd.append("--no-tensorboard")
    if str(args.mlflow_tracking_uri).strip():
        cmd += ["--mlflow-tracking-uri", str(args.mlflow_tracking_uri)]
        if str(args.mlflow_experiment).strip():
            cmd += ["--mlflow-experiment", str(args.mlflow_experiment)]
        if str(args.mlflow_run_name).strip():
            cmd += ["--mlflow-run-name", str(args.mlflow_run_name)]

    if resume_from is not None and str(resume_run_name).strip():
        cmd += [
            "--run-name",
            str(resume_run_name).strip(),
            "--resume",
            "--resume-from",
            str(resume_from),
        ]
    elif init_model is not None:
        cmd += ["--init-model", str(init_model)]
    return cmd


def build_eval_policy_cmd(
    *,
    py: str,
    args: EvalPolicyArgs,
    model_path: Path,
    output_json: Path,
    env_kwargs: dict[str, Any],
) -> list[str]:
    cmd = [
        py,
        "-m",
        "reinforce.ppo_discrete.entrypoints.eval_policy",
        "--env-id",
        str(args.env_id),
        "--model-path",
        str(model_path),
        "--episodes",
        str(int(args.eval_episodes)),
        "--seed",
        str(int(args.seed)),
        "--output-json",
        str(output_json),
        "--deterministic",
        "--env-kwargs-json",
        json.dumps(env_kwargs),
        "--feature-id",
        str(args.ppo_feature_id),
    ]
    append_bool_flag(cmd, "pf-enabled", bool(args.ppo_pf_enabled))
    append_bool_flag(cmd, "use-action-mask", bool(args.use_action_mask))
    return cmd
