"""`jsonl_to_tensorboard` スクリプト。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _safe_float(v: object) -> float | None:
    """内部ヘルパー: `safe_float` を実行する。

    Args:
        v (object): v の値。

    Returns:
        float | None: 計算結果。
    """
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    try:
        f = float(v)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


def _flatten(obj: object, prefix: str = "") -> dict[str, float]:
    """内部ヘルパー: `flatten` を実行する。

    Args:
        obj (object): obj の値。
        prefix (str): prefix の値。

    Returns:
        dict[str, float]: 計算結果。
    """
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}.{i}" if prefix else str(i)))
    else:
        v = _safe_float(obj)
        if v is not None and prefix:
            out[prefix] = v
    return out


# ---------------------------------------------------------------------------
# per-file converters
# ---------------------------------------------------------------------------

def _convert_train_metrics(jsonl_path: Path, writer) -> int:
    """内部ヘルパー: `convert_train_metrics` を実行する。

    Args:
        jsonl_path (Path): 対象パス。
        writer (Any): writer の値。

    Returns:
        int: 計算結果。
    """
    skip_keys = {"iteration", "global_step", "time"}
    count = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(row.get("global_step", row.get("step", 0)))
            for k, v in row.items():
                if k in skip_keys:
                    continue
                fv = _safe_float(v)
                if fv is not None:
                    writer.add_scalar(f"train/{k}", fv, global_step=step)
            count += 1
    return count


def _convert_periodic_val_metrics(jsonl_path: Path, writer) -> int:
    """内部ヘルパー: `convert_periodic_val_metrics` を実行する。

    Args:
        jsonl_path (Path): 対象パス。
        writer (Any): writer の値。

    Returns:
        int: 計算結果。
    """
    skip_keys = {"global_step", "eval_round", "seed_base", "fixed_seeds", "time"}
    count = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(row.get("global_step", 0))
            flat = _flatten(row)
            for k, v in flat.items():
                # skip_keys を前方一致で除外
                if any(k == sk or k.startswith(sk + ".") for sk in skip_keys):
                    continue
                writer.add_scalar(f"eval/{k}", v, global_step=step)
            count += 1
    return count


def _convert_flat_metrics(jsonl_path: Path, writer, tag_prefix: str = "eval_final") -> int:
    """内部ヘルパー: `convert_flat_metrics` を実行する。

    Args:
        jsonl_path (Path): 対象パス。
        writer (Any): writer の値。
        tag_prefix (str): tag_prefix の値。

    Returns:
        int: 計算結果。
    """
    skip_keys = {"step", "time"}
    count = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(row.get("step", 0))
            for k, v in row.items():
                if k in skip_keys:
                    continue
                fv = _safe_float(v)
                if fv is not None:
                    writer.add_scalar(f"{tag_prefix}/{k}", fv, global_step=step)
            count += 1
    return count


# ---------------------------------------------------------------------------
# discovery & conversion
# ---------------------------------------------------------------------------

def _make_writer(tb_dir: Path):
    """内部ヘルパー: `make_writer` を実行する。

    Args:
        tb_dir (Path): 対象パス。

    Returns:
        Any: 計算結果。
    """
    from torch.utils.tensorboard import SummaryWriter
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def convert_ppo_run(run_dir: Path) -> bool:
    """`convert_ppo_run` を実行する。

    Args:
        run_dir (Path): 対象パス。

    Returns:
        bool: 計算結果。
    """
    logs_dir = run_dir / "logs"
    train_metrics = logs_dir / "train_metrics.jsonl"
    periodic_val = logs_dir / "periodic_val_metrics.jsonl"
    if not train_metrics.exists() and not periodic_val.exists():
        return False

    tb_dir = logs_dir / "tensorboard"
    print(f"  [ppo_run] {run_dir.name}")
    writer = _make_writer(tb_dir)
    try:
        if train_metrics.exists():
            n = _convert_train_metrics(train_metrics, writer)
            print(f"    train_metrics.jsonl        → {n} steps")
        if periodic_val.exists():
            n = _convert_periodic_val_metrics(periodic_val, writer)
            print(f"    periodic_val_metrics.jsonl → {n} eval rounds")
    finally:
        writer.flush()
        writer.close()
    print(f"    -> {tb_dir}")
    return True


def convert_eval_run(run_dir: Path) -> bool:
    """`convert_eval_run` を実行する。

    Args:
        run_dir (Path): 対象パス。

    Returns:
        bool: 計算結果。
    """
    metrics = run_dir / "logs" / "metrics.jsonl"
    if not metrics.exists():
        return False

    tb_dir = run_dir / "logs" / "tensorboard"
    print(f"  [eval_run] {run_dir.name}")
    writer = _make_writer(tb_dir)
    try:
        n = _convert_flat_metrics(metrics, writer, tag_prefix="eval_final")
        print(f"    metrics.jsonl → {n} rows")
    finally:
        writer.flush()
        writer.close()
    print(f"    -> {tb_dir}")
    return True


def convert_pipeline_run(pipeline_dir: Path) -> None:
    """`convert_pipeline_run` を実行する。

    Args:
        pipeline_dir (Path): 対象パス。
    """
    converted = 0

    # ppo_runs/*/
    ppo_runs_dir = pipeline_dir / "artifacts" / "ppo_runs"
    if ppo_runs_dir.is_dir():
        for d in sorted(ppo_runs_dir.iterdir()):
            if d.is_dir() and convert_ppo_run(d):
                converted += 1

    # eval_runs/*/
    eval_runs_dir = pipeline_dir / "artifacts" / "eval_runs"
    if eval_runs_dir.is_dir():
        for d in sorted(eval_runs_dir.iterdir()):
            if d.is_dir() and convert_eval_run(d):
                converted += 1

    # トップレベルに直接 train_metrics.jsonl がある場合（単一 run）
    if not converted:
        if convert_ppo_run(pipeline_dir):
            converted += 1
        if convert_eval_run(pipeline_dir):
            converted += 1

    if converted == 0:
        print("  変換対象の JSONL ファイルが見つかりませんでした。")
    else:
        print(f"\n合計 {converted} ディレクトリを変換しました。")
        print(f"\n起動コマンド:")
        print(f"  tensorboard --logdir '{pipeline_dir}'")


def main() -> None:
    """メイン処理を実行する。"""
    parser = argparse.ArgumentParser(description="JSONL → TensorBoard 変換")
    parser.add_argument("run_dir", type=Path, help="パイプライン run ディレクトリ（または単一 run）")
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: ディレクトリが見つかりません: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"変換対象: {run_dir}\n")
    convert_pipeline_run(run_dir)


if __name__ == "__main__":
    main()
