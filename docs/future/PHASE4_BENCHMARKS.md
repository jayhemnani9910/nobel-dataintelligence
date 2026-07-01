---
title: Phase 4 Benchmarks — Head-to-Head SOTA Comparison
tags:
  - benchmarks
  - phase-1
date: 2026-04-19
---

# Phase 4 Benchmarks

> [!WARNING]
> **Status: PENDING (mode='dry-run')** — This report was generated without a trained VibroPredict checkpoint.
> Every VibroPredict row below shows `pending`. Re-run `scripts/run_benchmarks.py` with `--checkpoint` to populate.

## Random Split Results

| Model | R² | RMSE | MAE | Spearman |
|-------|----|------|-----|----------|
| **VibroPredict** | pending | pending | pending | pending |

## OOD (EC Holdout) Split Results

> [!NOTE]
> This is the moment of truth — does the model generalize to unseen enzyme families?

| Model | R² | RMSE | MAE | Spearman |
|-------|----|------|-----|----------|
| **VibroPredict** | pending | pending | pending | pending |

## Plots

_No plots generated (either no predictions available or matplotlib not installed)._

## What's Missing (Phase 2)

The following baselines and benchmarks are **not yet included**:

- **KcatNet** (2025) — Geometric deep learning; requires vendoring their code.
- **RealKcat** (2025) — Dataset overlap unresolved; see [DATASET_AUDIT.md](DATASET_AUDIT.md).
- **CataPro** (2025) — Not yet integrated into the baseline harness.
- **ProteinGym v1.3** — Community benchmark with 250+ DMS assays.
- **Modality ablation on OOD split** — Does VDOS matter more on OOD than random?

## Reproduction Instructions

```bash
# Commit SHA: ecb4072135a2f545e85548a20fd812dbbeacb4b5
# Generated: 2026-04-19T09:39:10.432386+00:00
python scripts/run_benchmarks.py \
    --checkpoint checkpoints/vibropredict_best.pt \
    --kinhub data/kaggle/kinhub.csv \
    --vdos-dir data/spectral/
```

### Environment

- Python: `3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]`
- Platform: `Linux-6.17.0-20-generic-x86_64-with-glibc2.39`

## Links

- [ROADMAP.md](ROADMAP.md) — Strategic context
- [DATASET_AUDIT.md](DATASET_AUDIT.md) — KinHub vs RealKcat overlap
- [benchmarks.json](../../benchmarks/benchmarks.json) — Machine-readable results
