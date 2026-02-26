from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..runtime.checkpoint import load_agent_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark inference throughput of saved discrete policy model.")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--batch-sizes", type=str, default="1,8,32,128")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def bench_once(
    agent, obs_shape: tuple[int, ...], batch_size: int, iters: int, device: torch.device
) -> dict[str, float]:
    x = torch.as_tensor(np.random.randn(batch_size, *obs_shape), dtype=torch.float32, device=device)
    for _ in range(20):
        _ = agent.get_logits(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = agent.get_logits(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    us_per_iter = 1e6 * elapsed / max(1, iters)
    us_per_sample = us_per_iter / max(1, batch_size)
    return {
        "batch_size": float(batch_size),
        "iters": float(iters),
        "elapsed_sec": float(elapsed),
        "us_per_iter": float(us_per_iter),
        "us_per_sample": float(us_per_sample),
        "samples_per_sec": float((batch_size * iters) / max(1e-12, elapsed)),
    }


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    agent, _meta = load_agent_checkpoint(args.model_path, device=device)
    agent.eval()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    results = [bench_once(agent, tuple(agent.obs_shape), bs, args.iters, device) for bs in batch_sizes]
    report = {"model_path": str(args.model_path), "device": str(device), "results": results}
    print(json.dumps(report, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[benchmark_infer] saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
