# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
There are no tagged releases yet.

## [Unreleased]

### Added
- Demo GIF, CONTRIBUTING.md, issue and PR templates, `.env.example`.
- `ruff.toml` with `target-version = "py310"`, so lint matches the oldest Python CI tests.

### Changed
- README rewritten around a runnable PETase example and the physics-signal result.
- CI also runs `vibropredict/tests/`.
- `.gitignore` covers every `.env*` file except `.env.example`.

### Fixed
- `scripts/run_benchmarks.py` crashed on Python 3.10 (`datetime.UTC`).

## 2026-10-06

### Fixed
- Crashes in `real_world_petase.py`, `run_benchmarks.py`, the dataset loaders and the GNN default edge dimension.
- `fit()` restores the best validation checkpoint. Novozymes updates no longer blank rows. Several loss, metric and pooling bugs.
- The Python 3.12 CI job: `graphein` is skipped there, since its pins have no 3.12 wheels and the code does not use it.

### Changed
- The demo page no longer shows results that were never run.
- Ruff lint and format pass across the tree.

## 2026-07-01 to 2026-07-02

### Fixed
- VDOS eigenvalue to wavenumber conversion in both the ANM and GNM paths. Every spectrum had collapsed to 0 cm⁻¹.
- The ablation runner.
- Inference loads real checkpoints and keeps coordinates and features aligned. Missing checkpoints give a clear error.

### Added
- Physics-signal validation on 27 thermophile/mesophile pairs. The result is negative.
- `docs/BENCHMARK_RUNBOOK.md`, `AUDIT.md`, `AUDIT_REVIEW.md`.

## 2026-04-19

### Added
- Enzyme-family out-of-distribution split (`ECHoldoutSplit`).
- Baseline harness with CatPred (local only) and a stub.
- Optional Weights & Biases logging in `TrainerWithMMDrop`.
- Benchmark runner with `--dry-run`.
- KinHub vs RealKcat overlap check.
- CI lint job, Python 3.10 to 3.12 matrix and a training smoke test.

## 2026-03-16

### Added
- VibroPredict: tri-modal enzyme kinetics model (ProtT5, VDOS, ChemBERTa) with spectral dropout training.
- Inference pipeline, `predict-stability` and `predict-kcat` CLI commands, Colab training notebooks.
- Project page in `demo/`, deployed to GitHub Pages.
- CI test workflow.

## 2025-12-12 to 2025-12-19

### Added
- First version: NMA and VDOS generation, GNN and spectral CNN models, Novozymes and CAFA 5 pipelines with CLI submissions, notebooks.
- GitHub Pages workflow.
