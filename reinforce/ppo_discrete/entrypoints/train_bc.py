from __future__ import annotations

import glob
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from ..models import build_agent, load_model_config_from_sources, normalize_model_config
from ..pipeline.model_checkpoint_service import save_agent_checkpoint
from ..ppo.trainer import aux_opp_param_mse
from ..utils.experiment import coerce_optional_path, create_run_layout, make_run_name, to_jsonable, update_manifest
from ..utils.log_utils import get_logger
from ..utils.metrics import summarize
from ..data.teacher_dataset import DATASET_KEY_OPP_PARAM_TRUE, DATASET_KEY_OPP_VALID
from ..utils.tracking import MetricTracker

logger = get_logger("train_bc")
_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


@hydra.main(version_base="1.3", config_path=_CONF_DIR, config_name="train_bc/default")
def main(cfg: DictConfig) -> None:
    args = _cfg_to_ns(cfg)
    raise SystemExit(_run(args))


def _cfg_to_ns(cfg: DictConfig) -> SimpleNamespace:
    d: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[assignment]
    d["dataset_npz"] = coerce_optional_path(d.get("dataset_npz"), dot_is_none=True)
    d["output_model"] = coerce_optional_path(d.get("output_model"), dot_is_none=True)
    d["metrics_jsonl"] = coerce_optional_path(d.get("metrics_jsonl"), dot_is_none=True)
    d["model_config_file"] = coerce_optional_path(d.get("model_config_file"), dot_is_none=True)
    d["run_root"] = coerce_optional_path(d.get("run_root"))
    return SimpleNamespace(**d)


def _validate_args(args: SimpleNamespace) -> None:
    shards_glob = str(args.dataset_shards_glob).strip()
    if args.dataset_npz is None and not shards_glob:
        raise ValueError("--dataset-npz or --dataset-shards-glob is required")
    if args.output_model is None and args.run_root is None:
        raise ValueError("--output-model is required (or set --run-root)")


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_model_config(args: SimpleNamespace) -> dict[str, Any]:
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


def _resolve_dataset_paths(args: SimpleNamespace) -> list[Path]:
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


def _inspect_aux_keys(dataset_paths: list[Path]) -> dict[str, Any]:
    has_aux: bool | None = None
    opp_param_shape: tuple[int, ...] | None = None
    opp_valid_shape: tuple[int, ...] | None = None
    for p in dataset_paths:
        with np.load(p) as data:
            present = (DATASET_KEY_OPP_PARAM_TRUE in data.files) and (DATASET_KEY_OPP_VALID in data.files)
            if has_aux is None:
                has_aux = bool(present)
            elif bool(has_aux) != bool(present):
                raise ValueError(f"aux key presence mismatch across dataset shards: {p}")
            if not present:
                continue
            ps = tuple(int(x) for x in np.asarray(data[DATASET_KEY_OPP_PARAM_TRUE]).shape[1:])
            vs = tuple(int(x) for x in np.asarray(data[DATASET_KEY_OPP_VALID]).shape[1:])
            if opp_param_shape is None:
                opp_param_shape = ps
                opp_valid_shape = vs
            else:
                if opp_param_shape != ps:
                    raise ValueError(f"{DATASET_KEY_OPP_PARAM_TRUE} shape mismatch across shards: {p}")
                if opp_valid_shape != vs:
                    raise ValueError(f"{DATASET_KEY_OPP_VALID} shape mismatch across shards: {p}")
    if has_aux is None:
        has_aux = False
    return {
        "available": bool(has_aux),
        "keys": ([DATASET_KEY_OPP_PARAM_TRUE, DATASET_KEY_OPP_VALID] if bool(has_aux) else []),
        "opp_param_true_shape": (list(opp_param_shape) if opp_param_shape is not None else []),
        "opp_valid_shape": (list(opp_valid_shape) if opp_valid_shape is not None else []),
    }



def _prepare_run(args: SimpleNamespace) -> tuple[Any, MetricTracker | None]:
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

    tracker = MetricTracker(layout.root, run_name=run_name, config=config_snapshot)
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


class BCTrainer:
    """Behavioural Cloning trainer. Encapsulates dataset setup, model building, and the training loop."""

    def __init__(self, args: SimpleNamespace) -> None:
        self.args = args

    # ------------------------------------------------------------------
    # Dataset setup
    # ------------------------------------------------------------------

    def _setup_dataset(self) -> tuple[list[Path], int, list[int], tuple[int, ...], int, dict[str, Any], np.ndarray, list[int], int, int]:
        args = self.args
        dataset_paths = _resolve_dataset_paths(args)
        total_n, shard_sizes, obs_shape, action_dim = _count_dataset_rows(dataset_paths)
        aux_info = _inspect_aux_keys(dataset_paths)
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
        if bool(aux_info["available"]):
            logger.info(
                "dataset contains optional aux keys (%s), but train_bc currently ignores them",
                ",".join(str(k) for k in aux_info["keys"]),
            )
        return dataset_paths, total_n, shard_sizes, obs_shape, action_dim, aux_info, valid_mask, offsets, train_count, n_valid

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        aux_info: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], torch.optim.Optimizer, float, bool, torch.device]:
        args = self.args
        device = _choose_device(args.device)
        model_config = _resolve_model_config(args)
        aux_coef = float(max(0.0, float(args.aux_opp_param_loss_coef)))
        aux_active = bool(aux_coef > 0.0)
        if aux_active and not bool(aux_info["available"]):
            raise ValueError(
                "aux_opp_param_loss_coef > 0 requires dataset keys "
                f"{DATASET_KEY_OPP_PARAM_TRUE}/{DATASET_KEY_OPP_VALID}"
            )
        if aux_active:
            model_config = dict(model_config)
            kwargs = dict(model_config.get("kwargs") or {})
            kwargs.setdefault("aux_opp_param_head", True)
            model_config["kwargs"] = kwargs
        agent, model_config = build_agent(obs_shape=obs_shape, action_dim=action_dim, model_config=model_config)
        agent = agent.to(device)
        if aux_active and not callable(getattr(agent, "get_aux_opp_param", None)):
            raise ValueError(
                "aux_opp_param_loss_coef > 0 requires model to implement get_aux_opp_param(obs) -> [B,7,5]"
            )
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        return agent, model_config, optimizer, aux_coef, aux_active, device

    # ------------------------------------------------------------------
    # Training & validation passes
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        *,
        agent: Any,
        optimizer: torch.optim.Optimizer,
        dataset_paths: list[Path],
        shard_sizes: list[int],
        offsets: list[int],
        valid_mask: np.ndarray,
        aux_active: bool,
        aux_coef: float,
        device: torch.device,
    ) -> tuple[float, float, int]:
        """Run one training epoch. Returns (train_loss_sum, train_aux_loss_sum, train_seen)."""
        args = self.args
        agent.train()
        train_loss_sum = 0.0
        train_aux_loss_sum = 0.0
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
                opp_param = np.asarray(data[DATASET_KEY_OPP_PARAM_TRUE], dtype=np.float32) if aux_active else None
                opp_valid = np.asarray(data[DATASET_KEY_OPP_VALID], dtype=np.uint8) if aux_active else None
                local_train_idx = np.flatnonzero(~valid_mask[off : off + shard_n])
                if local_train_idx.size <= 0:
                    continue
                np.random.shuffle(local_train_idx)
                for b in _batch_iter(local_train_idx, args.batch_size):
                    xb = torch.as_tensor(obs[b], dtype=torch.float32, device=device)
                    yb = torch.as_tensor(act[b], dtype=torch.long, device=device)
                    logits = agent.get_logits(xb)
                    bc_loss = F.cross_entropy(logits, yb)
                    loss = bc_loss
                    aux_loss_v = float("nan")
                    if aux_active:
                        assert opp_param is not None and opp_valid is not None
                        tb = torch.as_tensor(opp_param[b], dtype=torch.float32, device=device)
                        vb = torch.as_tensor(opp_valid[b], dtype=torch.uint8, device=device)
                        aux_loss = aux_opp_param_mse(
                            agent.get_aux_opp_param(xb), tb, vb,
                            use_valid_mask=bool(args.aux_opp_param_use_valid_mask),
                        )
                        aux_loss_v = float(aux_loss.item())
                        loss = loss + (aux_coef * aux_loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    bs = int(yb.numel())
                    train_loss_sum += float(bc_loss.item()) * float(bs)
                    if np.isfinite(aux_loss_v):
                        train_aux_loss_sum += float(aux_loss_v) * float(bs)
                    train_seen += bs
        return train_loss_sum, train_aux_loss_sum, train_seen

    def _validate_epoch(
        self,
        *,
        agent: Any,
        dataset_paths: list[Path],
        shard_sizes: list[int],
        offsets: list[int],
        valid_mask: np.ndarray,
        aux_active: bool,
        device: torch.device,
    ) -> tuple[float, float, int, int]:
        """Run one validation pass. Returns (valid_loss_sum, valid_aux_loss_sum, valid_correct, valid_seen)."""
        args = self.args
        agent.eval()
        valid_loss_sum = 0.0
        valid_aux_loss_sum = 0.0
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
                    opp_param = np.asarray(data[DATASET_KEY_OPP_PARAM_TRUE], dtype=np.float32) if aux_active else None
                    opp_valid = np.asarray(data[DATASET_KEY_OPP_VALID], dtype=np.uint8) if aux_active else None
                    for b in _batch_iter(local_valid_idx, args.batch_size):
                        xb = torch.as_tensor(obs[b], dtype=torch.float32, device=device)
                        yb = torch.as_tensor(act[b], dtype=torch.long, device=device)
                        logits_val = agent.get_logits(xb)
                        valid_loss_sum += float(F.cross_entropy(logits_val, yb, reduction="sum").item())
                        if aux_active:
                            assert opp_param is not None and opp_valid is not None
                            tb = torch.as_tensor(opp_param[b], dtype=torch.float32, device=device)
                            vb = torch.as_tensor(opp_valid[b], dtype=torch.uint8, device=device)
                            aux_val = aux_opp_param_mse(
                                agent.get_aux_opp_param(xb), tb, vb,
                                use_valid_mask=bool(args.aux_opp_param_use_valid_mask),
                            )
                            valid_aux_loss_sum += float(aux_val.item()) * float(yb.numel())
                        pred = torch.argmax(logits_val, dim=-1)
                        valid_correct += int((pred == yb).sum().item())
                        valid_seen += int(yb.numel())
        return valid_loss_sum, valid_aux_loss_sum, valid_correct, valid_seen

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, *, layout: Any, tracker: MetricTracker | None) -> int:
        args = self.args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        (dataset_paths, total_n, shard_sizes, obs_shape, action_dim,
         aux_info, valid_mask, offsets, train_count, n_valid) = self._setup_dataset()

        agent, model_config, optimizer, aux_coef, aux_active, device = self._setup_model(
            obs_shape, action_dim, aux_info
        )

        best_valid_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        train_loss_hist: list[float] = []
        valid_loss_hist: list[float] = []
        valid_acc_hist: list[float] = []
        train_aux_loss_hist: list[float] = []
        valid_aux_loss_hist: list[float] = []
        metrics_path = args.metrics_jsonl or args.output_model.with_suffix(args.output_model.suffix + ".metrics.jsonl")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            train_loss_sum, train_aux_loss_sum, train_seen = self._train_epoch(
                agent=agent, optimizer=optimizer,
                dataset_paths=dataset_paths, shard_sizes=shard_sizes, offsets=offsets,
                valid_mask=valid_mask, aux_active=aux_active, aux_coef=aux_coef, device=device,
            )
            valid_loss_sum, valid_aux_loss_sum, valid_correct, valid_seen = self._validate_epoch(
                agent=agent,
                dataset_paths=dataset_paths, shard_sizes=shard_sizes, offsets=offsets,
                valid_mask=valid_mask, aux_active=aux_active, device=device,
            )

            mean_train_loss = float(train_loss_sum / train_seen) if train_seen > 0 else float("nan")
            valid_loss = float(valid_loss_sum / valid_seen) if valid_seen > 0 else float("nan")
            valid_acc = float(valid_correct / valid_seen) if valid_seen > 0 else float("nan")
            train_aux_loss = float(train_aux_loss_sum / train_seen) if (aux_active and train_seen > 0) else float("nan")
            valid_aux_loss = float(valid_aux_loss_sum / valid_seen) if (aux_active and valid_seen > 0) else float("nan")
            train_loss_hist.append(mean_train_loss)
            valid_loss_hist.append(valid_loss)
            valid_acc_hist.append(valid_acc)
            if aux_active:
                train_aux_loss_hist.append(train_aux_loss)
                valid_aux_loss_hist.append(valid_aux_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}

            row = {
                "epoch": int(epoch),
                "train_loss": mean_train_loss,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "train_aux_opp_param_loss": (train_aux_loss if aux_active else None),
                "valid_aux_opp_param_loss": (valid_aux_loss if aux_active else None),
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
                        "losses/train_aux_opp_param_loss": (train_aux_loss if aux_active else float("nan")),
                        "losses/valid_aux_opp_param_loss": (valid_aux_loss if aux_active else float("nan")),
                    },
                )

            if aux_active:
                logger.info(
                    "epoch=%d/%d train_loss=%.6f valid_loss=%.6f valid_acc=%.4f train_aux=%.6f valid_aux=%.6f",
                    int(epoch), int(args.epochs), float(mean_train_loss), float(valid_loss),
                    float(valid_acc), float(train_aux_loss), float(valid_aux_loss),
                )
            else:
                logger.info(
                    "epoch=%d/%d train_loss=%.6f valid_loss=%.6f valid_acc=%.4f",
                    int(epoch), int(args.epochs), float(mean_train_loss), float(valid_loss), float(valid_acc),
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
            "dataset_aux": aux_info,
            "aux_opp_param_loss_coef": float(aux_coef),
            "aux_opp_param_use_valid_mask": bool(args.aux_opp_param_use_valid_mask),
            "aux_opp_param_active": bool(aux_active),
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
            "train_aux_opp_param_loss_summary": (summarize(train_aux_loss_hist).as_dict() if aux_active else {}),
            "valid_aux_opp_param_loss_summary": (summarize(valid_aux_loss_hist).as_dict() if aux_active else {}),
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


def _run(args: SimpleNamespace) -> int:
    _validate_args(args)
    layout = None
    tracker: MetricTracker | None = None
    try:
        layout, tracker = _prepare_run(args)
        return BCTrainer(args).run(layout=layout, tracker=tracker)
    except Exception as e:
        if tracker is not None and layout is not None:
            update_manifest(layout, {"status": "failed", "error": str(e), "timestamps": {"failed_at": time.time()}})
            tracker.log_event("train_bc_failed", {"error": str(e)})
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
