# Changes from independent audit (feat/phase1-fixes)

## Source fixes
- **src/inference.py**
  - `predict_stability`: infer `num_go_terms` from the checkpoint's
    `cafa_head.6.weight` instead of hardcoding 100. Fixes a `RuntimeError`
    (cafa_head size mismatch) that made the real bundled checkpoints unloadable.
  - `predict_stability`: when a structure has more Cα atoms than the sequence
    length, clip **both** coords and per-residue features to the common minimum
    (was truncating only features to the larger value → `IndexError` in
    GATv2Conv).
  - Corrected the checkpoint docstring + missing-checkpoint hint: epoch1-3 are
    2-param smoke-test stubs; epoch6/9/10 are real 5.76M-param
    VibroStructuralModel weights (untrained on real data).

## Tests
- **tests/test_inference.py**: added
  `test_predict_stability_loads_real_checkpoint_and_runs` (guards both fixes;
  skipped if the real checkpoint / torch_geometric are absent).

## Docs
- **README.md**: Quick Start now includes `pip install -e .` and notes
  `PYTHONPATH=.` / `KMP_AFFINITY=disabled` so the documented `python -m src.cli`
  examples actually run.

## Environment
- Installed `torch_geometric`, `xxhash`, `psutil` into the `qdd` env (all
  already in requirements.txt; env was out of sync). This un-skips the GNN tests.

## New audit artifacts (repo root)
- **AUDIT_REVIEW.md** — the independent audit + re-verification of prior claims.
- **audit_vdos_check.png** — VDOS unit-fix re-verification figure.
- **e2e_run.log** — end-to-end run log.

## Test suite: 180 passed / 10 skipped  →  191 passed / 0 skipped
