from __future__ import annotations

from .native_ppo_runner import (
    build_parser,
    parse_args,
    run_native_ppo_from_args,
    run_native_ppo_from_train_ppo_args,
)


def run_from_args(args):
    return run_native_ppo_from_args(args)


def run_from_train_ppo_args(*, train_args, cfg, device, env_kwargs):
    return run_native_ppo_from_train_ppo_args(
        train_args=train_args,
        cfg=cfg,
        device=device,
        env_kwargs=env_kwargs,
    )


def main() -> int:
    args = parse_args()
    return int(run_native_ppo_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
