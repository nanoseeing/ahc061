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
    env_id: str
    seed: int
    bc_total_transitions: int
    bc_num_envs: int
    bc_num_steps: int
    bc_learning_rate: float
    bc_num_minibatches: int
    bc_max_grad_norm: float
    bc_temperature: float
    bc_teacher_model_path: Path | None
    ppo_feature_id: str
    ppo_pf_enabled: bool
    use_action_mask: bool


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
    ppo_compile: bool
    ppo_model_preset: str
    ppo_learning_rate_schedule: str
    ppo_warmup_iters: int
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
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_run_name: str
    tensorboard: bool


class PipelineArgs(TrainPPOArgs, TrainBCArgs, EvalPolicyArgs, Protocol):
    """All attributes accessed by pipeline_service.run_pipeline and helpers."""
    run_root: Path
    run_name: str
    resume: bool
    env_kwargs_json: str
    eval_env_kwargs_json: str
    skip_bc: bool
    skip_ppo: bool
    skip_last_eval: bool
    bc_teacher_model_path: Path | None
    ppo_init_model: Path | None


def append_bool_flag(cmd: list[str], name: str, value: bool) -> None:
    """Append a Hydra boolean override: name=true or name=false.

    ``name`` may use either hyphens or underscores; hyphens are converted to
    underscores to match Hydra config key names.
    """
    key = name.replace("-", "_")
    cmd.append(f"{key}={'true' if bool(value) else 'false'}")


def append_model_args(cmd: list[str], args: ModelArgs) -> None:
    model_class = str(args.model_class).strip()
    model_config_file = str(args.model_config_file).strip() if args.model_config_file is not None else ""
    model_config_json = str(args.model_config_json).strip()

    if model_class:
        cmd.append(f"model_class={model_class}")
    if model_config_file:
        cmd.append(f"model_config_file={model_config_file}")
    if model_config_json:
        # Wrap in OmegaConf single-quote string to prevent dict-syntax parsing
        cmd.append(f"model_config_json='{model_config_json}'")


def build_train_bc_cmd(
    *,
    py: str,
    args: TrainBCArgs,
    output_model: Path,
    teacher_model_path: Path,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "reinforce.ppo.entrypoints.train_bc",
        f"output_model={output_model}",
        f"teacher_model_path={teacher_model_path}",
        f"env_id={args.env_id}",
        f"seed={int(args.seed)}",
        f"total_transitions={int(args.bc_total_transitions)}",
        f"num_envs={int(args.bc_num_envs)}",
        f"num_steps={int(args.bc_num_steps)}",
        f"learning_rate={float(args.bc_learning_rate)}",
        f"num_minibatches={int(args.bc_num_minibatches)}",
        f"max_grad_norm={float(args.bc_max_grad_norm)}",
        f"temperature={float(args.bc_temperature)}",
        f"feature_id={args.ppo_feature_id}",
    ]
    append_bool_flag(cmd, "pf_enabled", bool(args.ppo_pf_enabled))
    append_bool_flag(cmd, "use_action_mask", bool(args.use_action_mask))
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
        "reinforce.ppo.entrypoints.train_ppo",
        f"env_id={args.env_id}",
        f"run_dir={run_dir}",
        f"seed={int(args.seed)}",
        f"total_timesteps={int(args.ppo_total_timesteps)}",
        f"num_envs={int(args.ppo_num_envs)}",
        f"num_steps={int(args.ppo_num_steps)}",
        f"learning_rate={float(args.ppo_learning_rate)}",
        f"gamma={float(args.ppo_gamma)}",
        f"gae_lambda={float(args.ppo_gae_lambda)}",
        f"num_minibatches={int(args.ppo_num_minibatches)}",
        f"update_epochs={int(args.ppo_update_epochs)}",
        f"clip_coef={float(args.ppo_clip_coef)}",
        f"clip_coef_schedule='{args.ppo_clip_coef_schedule}'",
        f"ent_coef={float(args.ppo_ent_coef)}",
        f"ent_coef_schedule='{args.ppo_ent_coef_schedule}'",
        f"vf_coef={float(args.ppo_vf_coef)}",
        f"aux_opp_param_loss_coef={float(args.ppo_aux_opp_param_loss_coef)}",
        f"max_grad_norm={float(args.ppo_max_grad_norm)}",
        f"checkpoint_interval_steps={int(args.ppo_checkpoint_interval_steps)}",
        f"eval_interval_steps={int(args.ppo_eval_interval_steps)}",
        f"eval_episodes={int(args.ppo_eval_episodes)}",
        f"eval_seed_start={int(args.ppo_eval_seed_start)}",
        f"log_interval_iters={int(args.ppo_log_interval_iters)}",
        f"vecnorm_clip_obs={float(args.ppo_vecnorm_clip_obs)}",
        f"vecnorm_clip_reward={float(args.ppo_vecnorm_clip_reward)}",
        f"vecnorm_epsilon={float(args.ppo_vecnorm_epsilon)}",
        f"env_kwargs_json='{json.dumps(env_kwargs)}'",
        f"feature_id={args.ppo_feature_id}",
        f"memory_format={args.ppo_memory_format}",
        f"rollout_cache_device={args.ppo_rollout_cache_device}",
        f"distributed={args.ppo_distributed}",
    ]
    append_bool_flag(cmd, "compile", bool(args.ppo_compile))
    append_model_args(cmd, args)
    append_bool_flag(cmd, "norm_adv", bool(args.ppo_norm_adv))
    append_bool_flag(cmd, "clip_vloss", bool(args.ppo_clip_vloss))
    append_bool_flag(cmd, "eval_at_start", bool(args.ppo_eval_at_start))
    append_bool_flag(cmd, "use_action_mask", bool(args.use_action_mask))
    append_bool_flag(cmd, "eval_fixed_seeds", bool(args.ppo_eval_fixed_seeds))
    append_bool_flag(cmd, "eval_deterministic", bool(args.ppo_eval_deterministic))
    append_bool_flag(cmd, "vecnorm", bool(args.ppo_vecnorm))
    append_bool_flag(cmd, "vecnorm_norm_obs", bool(args.ppo_vecnorm_norm_obs))
    append_bool_flag(cmd, "vecnorm_norm_reward", bool(args.ppo_vecnorm_norm_reward))
    append_bool_flag(cmd, "vecnorm_eval_norm_reward", bool(args.ppo_vecnorm_eval_norm_reward))
    append_bool_flag(cmd, "aux_opp_param_use_valid_mask", bool(args.ppo_aux_opp_param_use_valid_mask))
    append_bool_flag(cmd, "pf_enabled", bool(args.ppo_pf_enabled))
    append_bool_flag(cmd, "amp", bool(args.ppo_amp))
    append_bool_flag(cmd, "pin_memory", bool(args.ppo_pin_memory))
    append_bool_flag(cmd, "tensorboard", bool(args.tensorboard))
    if str(args.ppo_model_preset).strip():
        cmd.append(f"model_preset={args.ppo_model_preset.strip()}")
    if str(args.ppo_learning_rate_schedule).strip():
        cmd.append(f"learning_rate_schedule='{args.ppo_learning_rate_schedule}'")
    if int(args.ppo_warmup_iters) > 0:
        cmd.append(f"warmup_iters={int(args.ppo_warmup_iters)}")
    if args.ppo_clip_range_vf is not None:
        cmd.append(f"clip_range_vf={float(args.ppo_clip_range_vf)}")
    if str(args.ppo_clip_range_vf_schedule).strip():
        cmd.append(f"clip_range_vf_schedule='{args.ppo_clip_range_vf_schedule}'")
    if args.ppo_clip_range_vf_final is not None:
        cmd.append(f"clip_range_vf_final={float(args.ppo_clip_range_vf_final)}")
    if str(args.ppo_clip_range_vf_schedule_expr).strip():
        cmd.append(f"clip_range_vf_schedule_expr='{args.ppo_clip_range_vf_schedule_expr}'")
    if args.ppo_clip_coef_final is not None:
        cmd.append(f"clip_coef_final={float(args.ppo_clip_coef_final)}")
    if str(args.ppo_clip_coef_schedule_expr).strip():
        cmd.append(f"clip_coef_schedule_expr='{args.ppo_clip_coef_schedule_expr}'")
    if args.ppo_ent_coef_final is not None:
        cmd.append(f"ent_coef_final={float(args.ppo_ent_coef_final)}")
    if str(args.ppo_ent_coef_schedule_expr).strip():
        cmd.append(f"ent_coef_schedule_expr='{args.ppo_ent_coef_schedule_expr}'")
    if args.ppo_target_kl is not None:
        cmd.append(f"target_kl={float(args.ppo_target_kl)}")
    if args.ppo_vecnorm_gamma is not None:
        cmd.append(f"vecnorm_gamma={float(args.ppo_vecnorm_gamma)}")
    if str(ppo_val_env_kwargs_json).strip():
        cmd.append(f"eval_env_kwargs_json='{ppo_val_env_kwargs_json}'")
    if str(args.mlflow_tracking_uri).strip():
        cmd.append(f"mlflow_tracking_uri={args.mlflow_tracking_uri}")
        if str(args.mlflow_experiment).strip():
            cmd.append(f"mlflow_experiment={args.mlflow_experiment}")
        if str(args.mlflow_run_name).strip():
            cmd.append(f"mlflow_run_name={args.mlflow_run_name}")

    if resume_from is not None and str(resume_run_name).strip():
        cmd += [
            f"run_name={resume_run_name.strip()}",
            "resume=true",
            f"resume_from={resume_from}",
        ]
    elif init_model is not None:
        cmd.append(f"init_model={init_model}")
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
        "reinforce.ppo.entrypoints.eval_policy",
        f"env_id={args.env_id}",
        f"model_path={model_path}",
        f"episodes={int(args.eval_episodes)}",
        f"seed={int(args.seed)}",
        f"output_json={output_json}",
        "prefer_run_layout=false",
        "deterministic=true",
        f"env_kwargs_json='{json.dumps(env_kwargs)}'",
        f"feature_id={args.ppo_feature_id}",
    ]
    append_bool_flag(cmd, "pf_enabled", bool(args.ppo_pf_enabled))
    append_bool_flag(cmd, "use_action_mask", bool(args.use_action_mask))
    append_bool_flag(cmd, "tensorboard", bool(args.tensorboard))
    if str(args.mlflow_tracking_uri).strip():
        cmd.append(f"mlflow_tracking_uri={args.mlflow_tracking_uri}")
        if str(args.mlflow_experiment).strip():
            cmd.append(f"mlflow_experiment={args.mlflow_experiment}")
        if str(args.mlflow_run_name).strip():
            cmd.append(f"mlflow_run_name={args.mlflow_run_name}")
    return cmd
