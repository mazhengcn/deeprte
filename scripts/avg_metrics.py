#!/usr/bin/env python3
"""
Compute average MSE and RMSE across subfolders containing metrics.json files.

It looks for subfolders (example_* or sample_*) under the given base path
and parses metrics.json with keys like psi_mse, phi_mse, psi_rmse, phi_rmse.
Numeric strings are tolerated. A combined overall mean is also reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def load_metrics(mfile: Path) -> Dict[str, Optional[float]]:
    try:
        data = json.loads(mfile.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to read {mfile}: {e}") from e

    return {
        "psi_mse": to_float(data.get("psi_mse")),
        "phi_mse": to_float(data.get("phi_mse")),
        "psi_rmse": to_float(data.get("psi_rmse")),
        "phi_rmse": to_float(data.get("phi_rmse")),
        # optional plain keys if present
        "mse": to_float(data.get("mse")),
        "rmse": to_float(data.get("rmse")),
    }


def find_metric_dirs(base_dir: Path) -> List[Path]:
    # Prefer explicit patterns, but also accept any immediate subdir that has metrics.json
    candidates = set(base_dir.glob("example_*")) | set(base_dir.glob("sample_*"))
    # Add any subdir containing metrics.json
    for child in base_dir.iterdir():
        if child.is_dir() and (child / "metrics.json").is_file():
            candidates.add(child)
    return sorted([p for p in candidates if (p / "metrics.json").is_file()])


def compute_summary(base_dir: Path) -> Dict[str, Any]:
    subdirs = find_metric_dirs(base_dir)
    records: List[Dict[str, Optional[float]]] = []
    for d in subdirs:
        mfile = d / "metrics.json"
        try:
            rec = load_metrics(mfile)
        except Exception as e:
            print(f"Warning: {e}")
            continue
        rec["example"] = d.name  # type: ignore[index]
        records.append(rec)

    psi_mse_mean = safe_mean([r.get("psi_mse") for r in records])
    phi_mse_mean = safe_mean([r.get("phi_mse") for r in records])
    psi_rmse_mean = safe_mean([r.get("psi_rmse") for r in records])
    phi_rmse_mean = safe_mean([r.get("phi_rmse") for r in records])

    # overall means: average across psi and phi values altogether (ignore None)
    mse_pool = [r.get("psi_mse") for r in records] + [r.get("phi_mse") for r in records]
    rmse_pool = [r.get("psi_rmse") for r in records] + [
        r.get("phi_rmse") for r in records
    ]
    overall_mse_mean = safe_mean(mse_pool)
    overall_rmse_mean = safe_mean(rmse_pool)

    plain_mse_mean = safe_mean([r.get("mse") for r in records])
    plain_rmse_mean = safe_mean([r.get("rmse") for r in records])

    summary: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "num_examples": len(records),
        "psi_mse_mean": psi_mse_mean,
        "phi_mse_mean": phi_mse_mean,
        "overall_mse_mean": overall_mse_mean,
        "psi_rmse_mean": psi_rmse_mean,
        "phi_rmse_mean": phi_rmse_mean,
        "overall_rmse_mean": overall_rmse_mean,
        "plain_mse_mean": plain_mse_mean,
        "plain_rmse_mean": plain_rmse_mean,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average MSE/RMSE across metrics.json files in subdirectories."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "S-g0.8-sinrv",
        help="Base directory containing example_*/sample_* subfolders with metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON summary (default: <base>/metrics_summary.json)",
    )
    args = parser.parse_args()

    base_dir: Path = args.base.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory does not exist: {base_dir}")

    summary = compute_summary(base_dir)

    # Print a concise summary
    print(json.dumps(summary, indent=2))

    out_file = args.output or (base_dir / "metrics_summary.json")
    try:
        out_file.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to: {out_file}")
    except Exception as e:
        print(f"Warning: failed to write summary to {out_file}: {e}")


if __name__ == "__main__":
    main()
