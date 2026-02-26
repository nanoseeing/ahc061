from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models import build_agent, load_model_config_from_sources, normalize_model_config
from ..runtime.checkpoint import save_agent_checkpoint
from ..runtime.experiment import coerce_optional_path, create_run_layout, make_run_name, resolve_config, to_jsonable, update_manifest
from ..runtime.log_utils import get_logger
from ..runtime.metrics import summarize
from ..runtime.tracking import MetricTracker

logger = get_logger("train_bc")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train behavior-cloning model for discrete board PPO warm-start.")
    p.add_argument("--config-file", type=Path, default=None, help="json/toml/yaml config file")
    p.add_argument("--config-section", type=str, default="train_bc", help="section key in config file")
    p.add_argument("--set", dest="set", action="append", default=[], help="override key=value (repeatable)")

    p.add_argument("--dataset-npz", type=Path, default=None)
    p.add_argument(
        "--dataset-shards-glob",
        type=str,
        default="",
        help="optional glob for shard npz files; when set, BC streams shards instead of loading one big npz",
    )
    p.add_argument("--output-model", type=Path, default=None)
    p.add_argument("--metrics-jsonl", type=Path, default=None)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--model-class", type=str, default="", help="registered model name or import path")
    p.add_argument("--model-config-file", type=Path, default=None, help="optional model config (json/toml/yaml)")
    p.add_argument("--model-config-json", type=str, default="", help="optional model config JSON override")
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--run-root", type=Path, default=None, help="optional run root for managed outputs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--prefer-run-layout", action=argparse.BooleanOptionalAction, default=True)
    return p


def _parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in parser._actions:
        if a.dest == "help":
            continue
        if a.default is argparse.SUPPRESS:
            continue
        out[a.dest] = a.default
    return out


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    pre, _unknown = parser.parse_known_args()

    base_defaults = _parser_defaults(parser)
    for key in ("config_file", "config_section", "set"):
        base_defaults.pop(key, None)

    cfg = resolve_config(
        defaults=base_defaults,
        config_file=pre.config_file,
        config_section=pre.config_section,
        overrides=list(pre.set or []),
    )
    unknown = sorted(k for k in cfg.keys() if k not in base_defaults.keys())
    if unknown:
        raise ValueError(f"unknown config keys for train_bc: {', '.join(unknown)}")

    parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.dataset_npz = coerce_optional_path(args.dataset_npz, dot_is_none=True)
    args.output_model = coerce_optional_path(args.output_model, dot_is_none=True)
    args.metrics_jsonl = coerce_optional_path(args.metrics_jsonl, dot_is_none=True)
    args.model_config_file = coerce_optional_path(args.model_config_file, dot_is_none=True)
    args.run_root = coerce_optional_path(args.run_root)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    shards_glob = str(args.dataset_shards_glob).strip()
    if args.dataset_npz is None and not shards_glob:
        raise ValueError("--dataset-npz or --dataset-shards-glob is required")
    if args.output_model is None and args.run_root is None:
        raise ValueError("--output-model is required (or set --run-root)")


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_model_config(args: argparse.Namespace) -> dict[str, Any]:
    explicit_cfg = load_model_config_from_sources(
        model_config_file=args.model_config_file,
        model_config_json=args.model_config_json,
    )
    model_class = str(args.model_class).strip()
    if explicit_cfg is not None and model_class:
        explicit_cfg["type"] = model_class

    return normalize_model_config(
        explicit_cfg,
        default_type=model_class or "DiscreteBoardAgent",
    )


def _batch_iter(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        end = min(len(indices), start + batch_size)
        yield indices[start:end]


def _resolve_dataset_paths(args: argparse.Namespace) -> list[Path]:
    shards_glob = str(args.dataset_shards_glob).strip()
    if shards_glob:
        paths = sorted(Path(p) for p in glob.glob(shards_glob))
        if not paths:
            raise FileNotFoundError(f"no shard files matched: {shards_glob}")
        return paths
    assert args.dataset_npz is not None
    return [Path(args.dataset_npz)]


def _count_dataset_rows(dataset_paths: list[Path]) -> tuple[int, list[int], tuple[int, ...], int]:
    total = 0
    sizes: list[int] = []
    obs_shape_ref: tuple[int, ...] | None = None
    action_dim_ref: int | None = None
    for p in dataset_paths:
        with np.load(p) as data:
            obs_shape = tuple(int(x) for x in np.asarray(data["obs_shape"]).tolist())
            action_dim = int(np.asarray(data["action_dim"])[0])
            n = int(np.asarray(data["action"]).shape[0])
            if obs_shape_ref is None:
                obs_shape_ref = obs_shape
                action_dim_ref = action_dim
            else:
                if obs_shape_ref != obs_shape:
                    raise ValueError(f"obs_shape mismatch in shard: {p}")
                if action_dim_ref != action_dim:
                    raise ValueError(f"action_dim mismatch in shard: {p}")
            total += n
            sizes.append(n)
    if obs_shape_ref is None or action_dim_ref is None:
        raise ValueError("empty dataset")
    return int(total), sizes, obs_shape_ref, int(action_dim_ref)


def _prepare_run(args: argparse.Namespace) -> tuple[Any, MetricTracker | None]:
    if args.run_root is None:
        return None, None

    run_name = args.run_name or make_run_name("train_bc", seed=args.seed)
    layout = create_run_layout(args.run_root, run_name)

    if args.prefer_run_layout or args.output_model is None:
        args.output_model = layout.models_dir / "bc_init.pt"
    if args.prefer_run_layout or args.metrics_jsonl is None:
        args.metrics_jsonl = layout.logs_dir / "train_bc.metrics.jsonl"

    config_snapshot = to_jsonable({"args": vars(args), "layout": layout.as_dict()})
    (layout.config_dir / "train_bc.args.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    tracker = MetricTracker(layout.root, run_name=run_name, enable_tensorboard=False, config=config_snapshot)
    update_manifest(
        layout,
        {
            "job": "train_bc",
            "status": "running",
            "run_name": run_name,
            "layout": layout.as_dict(),
            "config": config_snapshot,
            "timestamps": {"started_at": time.time()},
        },
    )
    return layout, tracker


def main() -> int:
    args = parse_args()
    _validate_args(args)

    layout = None
    tracker: MetricTracker | None = None
    try:
        layout, tracker = _prepare_run(args)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        dataset_paths = _resolve_dataset_paths(args)
        total_n, shard_sizes, obs_shape, action_dim = _count_dataset_rows(dataset_paths)
        if total_n < 2:
            raise ValueError(f"dataset is too small: transitions={total_n}")

        idx = np.arange(total_n, dtype=np.int64)
        np.random.shuffle(idx)
        n_valid = int(round(total_n * args.valid_ratio))
        n_valid = max(1, min(total_n - 1, n_valid))
        valid_mask = np.zeros((total_n,), dtype=np.bool_)
        valid_mask[idx[:n_valid]] = True
        train_count = int(total_n - n_valid)

        offsets: list[int] = []
        cur = 0
        for n in shard_sizes:
            offsets.append(cur)
            cur += int(n)

        logger.info(
            "dataset mode=%s shards=%d transitions=%d train=%d valid=%d",
            "shards" if len(dataset_paths) > 1 else "single_npz",
            int(len(dataset_paths)),
            int(total_n),
            int(train_count),
            int(n_valid),
        )

        device = _choose_device(args.device)
        model_config = _resolve_model_config(args)
        agent, model_config = build_agent(
            obs_shape=obs_shape,
            action_dim=action_dim,
            model_config=model_config,
        )
        agent = agent.to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_valid_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        train_loss_hist: list[float] = []
        valid_loss_hist: list[float] = []
        valid_acc_hist: list[float] = []
        metrics_path = args.metrics_jsonl or args.output_model.with_suffix(args.output_model.suffix + ".metrics.jsonl")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            agent.train()
            train_loss_sum = 0.0
            train_seen = 0

            shard_order = np.arange(len(dataset_paths), dtype=np.int32)
            np.random.shuffle(shard_order)
            for si in shard_order.tolist():
                shard_path = dataset_paths[int(si)]
                shard_n = int(shard_sizes[int(si)])
                off = int(offsets[int(si)])
                if shard_n <= 0:
                    continue
                with np.load(shard_path) as data:
                    obs = np.asarray(data["obs"], dtype=np.float32)
                    act = np.asarray(data["action"], dtype=np.int64)
                    local_valid_mask = valid_mask[off : off + shard_n]
                    local_train_idx = np.flatnonzero(~local_valid_mask)
                    if local_train_idx.size <= 0:
                        continue
                    np.random.shuffle(local_train_idx)
                    for b in _batch_iter(local_train_idx, args.batch_size):
                        xb = torch.as_tensor(obs[b], dtype=torch.float32, device=device)
                        yb = torch.as_tensor(act[b], dtype=torch.long, device=device)
                        logits = agent.get_logits(xb)
                        loss = F.cross_entropy(logits, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        bs = int(yb.numel())
                        train_loss_sum += float(loss.item()) * float(bs)
                        train_seen += bs

            agent.eval()
            valid_loss_sum = 0.0
            valid_correct = 0
            valid_seen = 0
            with torch.no_grad():
                for si, shard_path in enumerate(dataset_paths):
                    shard_n = int(shard_sizes[si])
                    off = int(offsets[si])
                    if shard_n <= 0:
                        continue
                    local_valid_idx = np.flatnonzero(valid_mask[off : off + shard_n])
                    if local_valid_idx.size <= 0:
                        continue
                    with np.load(shard_path) as data:
                        obs = np.asarray(data["obs"], dtype=np.float32)
                        act = np.asarray(data["action"], dtype=np.int64)
                        for b in _batch_iter(local_valid_idx, args.batch_size):
                            xb = torch.as_tensor(obs[b], dtype=torch.float32, device=device)
                            yb = torch.as_tensor(act[b], dtype=torch.long, device=device)
                            logits_val = agent.get_logits(xb)
                            valid_loss_sum += float(F.cross_entropy(logits_val, yb, reduction="sum").item())
                            pred = torch.argmax(logits_val, dim=-1)
                            valid_correct += int((pred == yb).sum().item())
                            valid_seen += int(yb.numel())

            mean_train_loss = float(train_loss_sum / train_seen) if train_seen > 0 else float("nan")
            valid_loss = float(valid_loss_sum / valid_seen) if valid_seen > 0 else float("nan")
            valid_acc = float(valid_correct / valid_seen) if valid_seen > 0 else float("nan")
            train_loss_hist.append(mean_train_loss)
            valid_loss_hist.append(valid_loss)
            valid_acc_hist.append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}

            row = {
                "epoch": int(epoch),
                "train_loss": mean_train_loss,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
            if tracker is not None:
                tracker.log_metrics(
                    epoch,
                    {
                        "losses/train_loss": mean_train_loss,
                        "losses/valid_loss": valid_loss,
                        "metrics/valid_acc": valid_acc,
                    },
                )

            logger.info(
                "epoch=%d/%d train_loss=%.6f valid_loss=%.6f valid_acc=%.4f",
                int(epoch),
                int(args.epochs),
                float(mean_train_loss),
                float(valid_loss),
                float(valid_acc),
            )

        if best_state is not None:
            agent.load_state_dict(best_state)

        meta = {
            "dataset_mode": ("shards" if len(dataset_paths) > 1 else "single_npz"),
            "dataset_npz": (str(args.dataset_npz) if args.dataset_npz is not None else ""),
            "dataset_shards_glob": str(args.dataset_shards_glob),
            "dataset_shards_count": int(len(dataset_paths)),
            "dataset_transitions": int(total_n),
            "dataset_train_transitions": int(train_count),
            "dataset_valid_transitions": int(n_valid),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "obs_shape": list(obs_shape),
            "action_dim": int(action_dim),
            "model_config": to_jsonable(model_config),
            "metrics_jsonl": str(metrics_path),
            "best_valid_loss": float(best_valid_loss),
            "train_loss_summary": summarize(train_loss_hist).as_dict(),
            "valid_loss_summary": summarize(valid_loss_hist).as_dict(),
            "valid_acc_summary": summarize(valid_acc_hist).as_dict(),
            "args": to_jsonable(vars(args)),
        }
        save_agent_checkpoint(args.output_model, agent, optimizer=None, meta=meta)

        report_path = args.output_model.with_suffix(args.output_model.suffix + ".report.json")
        report_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("saved model: %s", args.output_model)
        logger.info("report: %s", report_path)

        if tracker is not None and layout is not None:
            summary_path = layout.reports_dir / "train_bc_summary.json"
            summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            update_manifest(
                layout,
                {
                    "status": "completed",
                    "result": {
                        "output_model": str(args.output_model),
                        "report_json": str(report_path),
                        "summary_json": str(summary_path),
                        "best_valid_loss": float(best_valid_loss),
                    },
                    "timestamps": {"finished_at": time.time()},
                },
            )
            tracker.log_event("train_bc_complete", {"best_valid_loss": float(best_valid_loss)})

        return 0
    except Exception as e:
        if tracker is not None and layout is not None:
            update_manifest(layout, {"status": "failed", "error": str(e), "timestamps": {"failed_at": time.time()}})
            tracker.log_event("train_bc_failed", {"error": str(e)})
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())
