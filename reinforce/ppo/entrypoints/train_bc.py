from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig

from ..train.online_bc_service import OnlineBCConfig, run_online_bc
from .common import cfg_to_namespace

_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


@hydra.main(version_base="1.3", config_path=_CONF_DIR, config_name="train_bc/default")
def main(cfg: DictConfig) -> None:
    args = _cfg_to_ns(cfg)
    raise SystemExit(_run(args))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    return cfg_to_namespace(
        cfg,
        optional_paths={
            "teacher_model_path": True,
            "init_model": True,
            "output_model": True,
            "model_config_file": True,
            "run_root": False,
        },
    )


def _run(args: SimpleNamespace) -> int:
    if args.teacher_model_path is None:
        raise ValueError("teacher_model_path is required")
    if args.output_model is None and args.run_root is None:
        raise ValueError("output_model is required (or set run_root)")

    output_model = args.output_model
    if output_model is None:
        from ..utils.experiment import create_run_layout, make_run_name
        run_name = args.run_name or make_run_name("online_bc", seed=args.seed)
        layout = create_run_layout(args.run_root, run_name)
        output_model = layout.models_dir / "bc_init.pt"

    cfg = OnlineBCConfig(
        env_id=str(args.env_id),
        feature_id=str(args.feature_id),
        pf_enabled=bool(args.pf_enabled),
        use_action_mask=bool(args.use_action_mask),
        amp=bool(args.amp),
        teacher_model_path=Path(args.teacher_model_path),
        model_class=str(args.model_class),
        model_config_file=args.model_config_file,
        model_config_json=str(args.model_config_json),
        model_preset=str(args.model_preset),
        init_model=args.init_model,
        output_model=Path(output_model),
        num_envs=int(args.num_envs),
        num_steps=int(args.num_steps),
        total_transitions=int(args.total_transitions),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        num_minibatches=int(args.num_minibatches),
        max_grad_norm=float(args.max_grad_norm),
        temperature=float(args.temperature),
        log_interval_iters=int(args.log_interval_iters),
        seed=int(args.seed),
        run_root=args.run_root,
        run_name=str(args.run_name),
    )

    run_online_bc(cfg)
    return 0


if __name__ == "__main__":
    main()
