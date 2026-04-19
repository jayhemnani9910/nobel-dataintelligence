#!/usr/bin/env python3
"""
Benchmark Runner

Runs VibroPredict + all registered baselines on both the random split and the
OOD (EC holdout) split, then writes:

  - benchmarks/benchmarks.json         — machine-readable results
  - docs/future/PHASE4_BENCHMARKS.md   — human-readable report
  - benchmarks/plots/*.png             — scatter + residual plots per model

Supports three operating modes:

  1. Full run (with checkpoint + KinHub + VDOS cache):
        python scripts/run_benchmarks.py \\
            --checkpoint checkpoints/vibropredict_best.pt \\
            --kinhub data/kaggle/kinhub.csv \\
            --vdos-dir data/spectral/

  2. Dry run (no checkpoint, no data — produces a structural report with
     every VibroPredict row marked `pending`, useful for shaping the report):
        python scripts/run_benchmarks.py --dry-run

  3. Baselines only (skip VibroPredict; useful when no checkpoint exists yet
     but you want to verify baseline integrations run end-to-end):
        python scripts/run_benchmarks.py --skip-vibropredict \\
            --kinhub data/kaggle/kinhub.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment and metrics helpers
# ---------------------------------------------------------------------------


def _get_env_dump() -> dict:
    """Capture environment info for reproducibility."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        env["pip_freeze"] = result.stdout.strip().split("\n")
    except Exception:
        env["pip_freeze"] = ["<unavailable>"]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        env["git_sha"] = result.stdout.strip()
    except Exception:
        env["git_sha"] = "<unavailable>"

    return env


def _compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    from vibropredict.training.metrics import compute_all_metrics

    return compute_all_metrics(predictions, targets)


def _fmt(value: Any, spec: str = ".4f") -> str:
    """Format a value as a number if possible, otherwise as a string."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return format(value, spec)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _run_vibropredict(
    checkpoint_path: str,
    test_df: pd.DataFrame,
    vdos_dir: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run VibroPredict inference on test data."""
    import torch

    from vibropredict.models.vibropredict_hybrid import VibroPredictHybrid

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = VibroPredictHybrid(fusion_dim=512, dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    predictions: list[float] = []
    targets: list[float] = []

    for _, row in test_df.iterrows():
        uniprot_id = str(row["uniprot_id"])
        sequence = str(row.get("sequence", ""))
        log_kcat = float(row["log_kcat"])
        substrate_smiles = str(row.get("substrate_smiles", ""))

        vdos_path = Path(vdos_dir) / f"{uniprot_id}_vdos.npy"
        if vdos_path.exists():
            vdos = np.load(vdos_path).astype(np.float32)
            if len(vdos) < 1000:
                vdos = np.pad(vdos, (0, 1000 - len(vdos)))
            else:
                vdos = vdos[:1000]
        else:
            vdos = np.zeros(1000, dtype=np.float32)

        vdos_tensor = torch.tensor(vdos, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logkcat, _ = model(
                sequences=[sequence],
                vdos=vdos_tensor.to(device),
                substrate_smiles=[substrate_smiles],
            )

        predictions.append(logkcat.squeeze().item())
        targets.append(log_kcat)

    return np.array(predictions), np.array(targets)


def _run_baselines(test_df: pd.DataFrame) -> dict:
    """Run all registered baselines on test data."""
    from vibropredict.evaluation.baselines import get_baseline, list_baselines

    results = {}
    for name in list_baselines():
        logger.info(f"Running baseline: {name}")
        try:
            model = get_baseline(name)
            model.setup()

            predictions = []
            targets = []
            for _, row in test_df.iterrows():
                sequence = str(row.get("sequence", ""))
                smiles = str(row.get("substrate_smiles", ""))
                log_kcat = float(row["log_kcat"])

                pred, _ = model.predict(sequence, smiles)
                predictions.append(pred)
                targets.append(log_kcat)

            preds = np.array(predictions)
            tgts = np.array(targets)
            metrics = _compute_metrics(preds, tgts)

            results[name] = {
                "predictions": preds.tolist(),
                "targets": tgts.tolist(),
                "metrics": metrics,
            }
            logger.info(f"  {name}: R²={metrics['r_squared']:.4f}")
        except Exception as exc:
            logger.warning(f"  {name} failed: {exc}")
            results[name] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _generate_plots(
    output_dir: Path,
    vibropredict_results: dict,
    baseline_results: dict,
) -> list[Path]:
    """Write scatter + residual plots per (model, split) pair.

    Returns the list of plot paths written. Silently skips if matplotlib is
    not installed (it's not a hard dependency).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot generation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Collect (model_name, split_name, predictions, targets) tuples
    rows: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for split_name, split_data in vibropredict_results.items():
        if "predictions" in split_data and "targets" in split_data:
            rows.append(
                (
                    "VibroPredict",
                    split_name,
                    np.asarray(split_data["predictions"]),
                    np.asarray(split_data["targets"]),
                )
            )
    for baseline_name, splits in baseline_results.items():
        for split_name, split_data in splits.items():
            if (
                isinstance(split_data, dict)
                and "predictions" in split_data
                and "targets" in split_data
            ):
                rows.append(
                    (
                        baseline_name,
                        split_name,
                        np.asarray(split_data["predictions"]),
                        np.asarray(split_data["targets"]),
                    )
                )

    for model_name, split_name, preds, targets in rows:
        if len(preds) == 0 or len(targets) == 0:
            continue

        safe_name = f"{model_name}_{split_name}".replace("/", "_").replace(" ", "_")

        # Scatter: predicted vs actual
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(targets, preds, alpha=0.4, s=12)
        lim_lo = float(min(targets.min(), preds.min()))
        lim_hi = float(max(targets.max(), preds.max()))
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, label="y = x")
        ax.set_xlabel("Actual log(k_cat)")
        ax.set_ylabel("Predicted log(k_cat)")
        ax.set_title(f"{model_name} — {split_name}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        scatter_path = output_dir / f"scatter_{safe_name}.png"
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=120)
        plt.close(fig)
        written.append(scatter_path)

        # Residual histogram
        residuals = preds - targets
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(residuals, bins=30, alpha=0.8, edgecolor="black")
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (pred − actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"{model_name} — {split_name} residuals")
        ax.grid(True, alpha=0.3)
        residual_path = output_dir / f"residuals_{safe_name}.png"
        fig.tight_layout()
        fig.savefig(residual_path, dpi=120)
        plt.close(fig)
        written.append(residual_path)

    logger.info(f"Wrote {len(written)} plots to {output_dir}")
    return written


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_json_report(
    output_path: Path,
    vibropredict_results: dict,
    baseline_results: dict,
    env_info: dict,
    mode: str,
):
    """Write machine-readable benchmarks.json."""
    rows = []

    for split_name, split_data in vibropredict_results.items():
        if "metrics" in split_data:
            for metric_name, metric_value in split_data["metrics"].items():
                rows.append(
                    {
                        "model": "VibroPredict",
                        "split": split_name,
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )

    for baseline_name, splits in baseline_results.items():
        for split_name, split_data in splits.items():
            if isinstance(split_data, dict) and "metrics" in split_data:
                for metric_name, metric_value in split_data["metrics"].items():
                    rows.append(
                        {
                            "model": baseline_name,
                            "split": split_name,
                            "metric": metric_name,
                            "value": metric_value,
                        }
                    )

    report = {
        "generated_at": env_info.get("timestamp", ""),
        "git_sha": env_info.get("git_sha", ""),
        "mode": mode,
        "results": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report written to {output_path}")


def _table_row(model_label: str, metrics: dict | None, bold: bool) -> str:
    """Render one row of a model/metrics table."""
    if metrics is None:
        cells = ["pending", "pending", "pending", "pending"]
    else:
        cells = [
            _fmt(metrics.get("r_squared", "N/A")),
            _fmt(metrics.get("rmse", "N/A")),
            _fmt(metrics.get("mae", "N/A")),
            _fmt(metrics.get("spearman", "N/A")),
        ]
    name = f"**{model_label}**" if bold else model_label
    if bold and metrics is not None:
        cells[0] = f"**{cells[0]}**"
    return f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"


def _split_section(
    title: str,
    heading_note: str | None,
    split_key: str,
    vibropredict_results: dict,
    baseline_results: dict,
) -> list[str]:
    """Assemble one split's results table."""
    lines: list[str] = [f"## {title}", ""]
    if heading_note:
        lines.extend(["> [!NOTE]", f"> {heading_note}", ""])
    lines.extend(
        [
            "| Model | R² | RMSE | MAE | Spearman |",
            "|-------|----|------|-----|----------|",
        ]
    )

    vp_entry = vibropredict_results.get(split_key, {})
    vp_metrics = vp_entry.get("metrics") if isinstance(vp_entry, dict) else None
    lines.append(_table_row("VibroPredict", vp_metrics, bold=True))

    for name, splits in baseline_results.items():
        split_data = splits.get(split_key) if isinstance(splits, dict) else None
        metrics = split_data.get("metrics") if isinstance(split_data, dict) else None
        lines.append(_table_row(name, metrics, bold=False))

    lines.append("")
    return lines


def _write_markdown_report(
    output_path: Path,
    vibropredict_results: dict,
    baseline_results: dict,
    env_info: dict,
    mode: str,
    plot_paths: list[Path],
):
    """Write human-readable PHASE4_BENCHMARKS.md."""
    is_pending = mode in {"dry-run", "skip-vibropredict"}

    lines = [
        "---",
        "title: Phase 4 Benchmarks — Head-to-Head SOTA Comparison",
        "tags:",
        "  - benchmarks",
        "  - phase-1",
        f"date: {datetime.date.today().isoformat()}",
        "---",
        "",
        "# Phase 4 Benchmarks",
        "",
    ]

    # TL;DR
    if is_pending:
        lines.extend(
            [
                "> [!WARNING]",
                f"> **Status: PENDING (mode={mode!r})** — This report was generated without a trained VibroPredict checkpoint.",
                "> Every VibroPredict row below shows `pending`. Re-run `scripts/run_benchmarks.py` with `--checkpoint` to populate.",
                "",
            ]
        )
    else:
        summaries: list[str] = []
        for split_name in ["random", "ood_ec_holdout"]:
            entry = vibropredict_results.get(split_name, {})
            metrics = entry.get("metrics") if isinstance(entry, dict) else None
            if metrics and "r_squared" in metrics:
                summaries.append(f"VibroPredict on {split_name}: R²={_fmt(metrics['r_squared'])}")
        tldr = " | ".join(summaries) if summaries else "No results available."
        lines.extend(["> [!IMPORTANT]", f"> **TL;DR** — {tldr}", ""])

    # Result tables
    lines.extend(
        _split_section(
            "Random Split Results",
            None,
            "random",
            vibropredict_results,
            baseline_results,
        )
    )
    lines.extend(
        _split_section(
            "OOD (EC Holdout) Split Results",
            "This is the moment of truth — does the model generalize to unseen enzyme families?",
            "ood_ec_holdout",
            vibropredict_results,
            baseline_results,
        )
    )

    # Plots
    if plot_paths:
        lines.extend(["## Plots", ""])
        for p in plot_paths:
            rel = (
                p.relative_to(output_path.parent.parent.parent)
                if output_path.parent.parent in p.parents
                else p
            )
            lines.append(f"![{p.stem}](../../{rel})")
        lines.append("")
    else:
        lines.extend(
            [
                "## Plots",
                "",
                "_No plots generated (either no predictions available or matplotlib not installed)._",
                "",
            ]
        )

    # What's missing
    lines.extend(
        [
            "## What's Missing (Phase 2)",
            "",
            "The following baselines and benchmarks are **not yet included**:",
            "",
            "- **KcatNet** (2025) — Geometric deep learning; requires vendoring their code.",
            "- **RealKcat** (2025) — Dataset overlap unresolved; see [DATASET_AUDIT.md](DATASET_AUDIT.md).",
            "- **CataPro** (2025) — Not yet integrated into the baseline harness.",
            "- **ProteinGym v1.3** — Community benchmark with 250+ DMS assays.",
            "- **Modality ablation on OOD split** — Does VDOS matter more on OOD than random?",
            "",
            "## Reproduction Instructions",
            "",
            "```bash",
            f"# Commit SHA: {env_info.get('git_sha', '<unknown>')}",
            f"# Generated: {env_info.get('timestamp', '<unknown>')}",
            "python scripts/run_benchmarks.py \\",
            "    --checkpoint checkpoints/vibropredict_best.pt \\",
            "    --kinhub data/kaggle/kinhub.csv \\",
            "    --vdos-dir data/spectral/",
            "```",
            "",
            "### Environment",
            "",
            f"- Python: `{env_info.get('python_version', '<unknown>').splitlines()[0]}`",
            f"- Platform: `{env_info.get('platform', '<unknown>')}`",
            "",
            "## Links",
            "",
            "- [ROADMAP.md](ROADMAP.md) — Strategic context",
            "- [DATASET_AUDIT.md](DATASET_AUDIT.md) — KinHub vs RealKcat overlap",
            "- [benchmarks.json](../../benchmarks/benchmarks.json) — Machine-readable results",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown report written to {output_path}")


# ---------------------------------------------------------------------------
# Dry-run helpers
# ---------------------------------------------------------------------------


def _dry_run_pipeline() -> tuple[dict, dict, str]:
    """Produce empty result structures for a dry run (no checkpoint / no data).

    Returns:
        (vibropredict_results, baseline_results, mode)
    """
    vibropredict_results: dict = {
        "random": {"status": "pending", "note": "dry run — no checkpoint provided"},
        "ood_ec_holdout": {"status": "pending", "note": "dry run — no checkpoint provided"},
    }
    baseline_results: dict = {}
    return vibropredict_results, baseline_results, "dry-run"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run VibroPredict benchmarks.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to VibroPredict checkpoint (.pt file). Required unless --dry-run or --skip-vibropredict.",
    )
    parser.add_argument(
        "--kinhub",
        type=str,
        default="data/kaggle/kinhub.csv",
        help="Path to KinHub CSV file.",
    )
    parser.add_argument(
        "--vdos-dir",
        type=str,
        default="data/spectral/",
        help="Directory containing VDOS .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/",
        help="Output directory for JSON and plots.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip running baseline models.",
    )
    parser.add_argument(
        "--skip-vibropredict",
        action="store_true",
        help="Skip VibroPredict inference; run only baselines.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate a structural report with every VibroPredict row marked 'pending', without running anything.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference ('cpu' or 'cuda').",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env_info = _get_env_dump()
    logger.info(f"Git SHA: {env_info['git_sha']}")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    if args.dry_run:
        logger.info("Dry run — skipping all inference.")
        vibropredict_results, baseline_results, mode = _dry_run_pipeline()
        plot_paths: list[Path] = []
    else:
        if not Path(args.kinhub).exists():
            logger.error(f"KinHub not found at {args.kinhub} — use --dry-run or supply --kinhub.")
            sys.exit(1)

        kinhub_df = pd.read_csv(args.kinhub)
        logger.info(f"Loaded KinHub: {len(kinhub_df)} rows")

        from vibropredict.data.splits import ECHoldoutSplit, RandomSplit

        random_splits = RandomSplit(seed=42).split(kinhub_df)
        ood_splits = ECHoldoutSplit(seed=42).split(kinhub_df)

        vibropredict_results = {}
        baseline_results = {}
        mode = "full"
        if args.skip_vibropredict:
            mode = "skip-vibropredict"
        elif args.checkpoint is None:
            logger.error(
                "No --checkpoint provided. Pass --dry-run to generate a "
                "pending report or --skip-vibropredict to run baselines only."
            )
            sys.exit(1)

        for split_label, test_df in [
            ("random", random_splits["test"]),
            ("ood_ec_holdout", ood_splits["test"]),
        ]:
            logger.info("=" * 60)
            logger.info(f"Split: {split_label} ({len(test_df)} test samples)")
            logger.info("=" * 60)

            if not args.skip_vibropredict:
                try:
                    preds, targets = _run_vibropredict(
                        args.checkpoint, test_df, args.vdos_dir, args.device
                    )
                    metrics = _compute_metrics(preds, targets)
                    vibropredict_results[split_label] = {
                        "predictions": preds.tolist(),
                        "targets": targets.tolist(),
                        "metrics": metrics,
                    }
                    logger.info(f"VibroPredict on {split_label}: R²={metrics['r_squared']:.4f}")
                except Exception as exc:
                    logger.error(f"VibroPredict failed on {split_label}: {exc}")
                    vibropredict_results[split_label] = {"error": str(exc)}
            else:
                vibropredict_results[split_label] = {"status": "skipped"}

            if not args.skip_baselines:
                split_baselines = _run_baselines(test_df)
                for name, result in split_baselines.items():
                    baseline_results.setdefault(name, {})[split_label] = result

        plot_paths = _generate_plots(plots_dir, vibropredict_results, baseline_results)

    _write_json_report(
        output_dir / "benchmarks.json",
        vibropredict_results,
        baseline_results,
        env_info,
        mode,
    )
    _write_markdown_report(
        Path("docs/future/PHASE4_BENCHMARKS.md"),
        vibropredict_results,
        baseline_results,
        env_info,
        mode,
        plot_paths,
    )

    logger.info("Benchmark run complete.")


if __name__ == "__main__":
    main()
