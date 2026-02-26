from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CleanRL's baseline discrete PPO script from the local vendored repository."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    parser.add_argument(
        "cleanrl_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to cleanrl/cleanrl/ppo.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]  # .../reinforce
    ppo_script = repo_root / "cleanrl" / "cleanrl" / "ppo.py"

    if not ppo_script.exists():
        raise FileNotFoundError(f"CleanRL ppo.py not found: {ppo_script}")

    forwarded = list(args.cleanrl_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    cmd = [sys.executable, str(ppo_script)] + forwarded
    print("[run_cleanrl_ppo_baseline]", " ".join(cmd))

    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

