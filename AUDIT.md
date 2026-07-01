# Full Repository Audit — Quantum Data Decoder / VibroPredict

*Whole-folder audit. Every source module, test, script, and doc was inventoried;
findings verified by running code where possible. Supersedes the earlier
`AUDIT_REPORT.md` (which covered only the first physics bug + test fixes).*

**Environment:** conda env `qdd` (Python 3.11, CPU PyTorch, ProDy, RDKit,
scikit-learn). Sandbox note: PyTorch import requires `KMP_AFFINITY=disabled`.

---

## Verdict

Strong, honest, well-tested engineering with a **recurring critical physics bug
that appeared in two independent VDOS code paths** — both now fixed. The
evaluation scaffolding (baselines, SOTA comparison, splits) is genuinely honest:
no fabricated numbers, missing baselines raise rather than fake. The core
scientific claim (physics-informed features help) remains **unproven** — a
controlled validation came back negative (see `docs/PHYSICS_SIGNAL_VALIDATION.md`).

**Test suite:** 5 failing at audit start → **0 failing** (177 passed,
10 skipped; 187 collected), up from 177 original tests. Added regression
guards for both physics bugs and the ablation runner.

---

## Fixed during this audit

### CRITICAL ×2 — VDOS eigenvalue→wavenumber collapse

The same broken conversion `1/(2*pi*29979.2458)` (~2×10⁷ too small) appeared in
**both** VDOS implementations, collapsing every spectrum to a delta at 0 cm⁻¹
and making the spectral branch — the project's core novelty — protein-independent.

1. **`src/nma_analysis.py`** (`ANMAnalyzer`, Phase 1-2 path). Replaced with a
   first-principles constant `ENM_FREQ_CM1_PER_SQRT_EIGVAL ≈ 108.59` (γ=1
   kcal/mol/Å², unit mass → base ω²=4.184×10²⁶ s⁻²). PETase-vs-LCC VDOS cosine
   0.99999999 → 0.984; thermostable LCC correctly shows lower entropy + stiffer
   softest mode. Guard: `tests/test_nma.py` (5 tests).
2. **`vibropredict/spectra/vdos_engine.py`** (`VibroEnzymePipeline`, the Phase-3
   path that feeds the tri-modal model). Identical bug on the GNM eigenvalues.
   Fixed to import the *same* shared constant (single source of truth). Verified:
   6eqe-vs-6ths VDOS cosine ~1.0 → 0.990, peaks at real wavenumbers not bin 0.
   Guard: `vibropredict/tests/test_vdos_engine.py` (3 tests).

**Root cause of survival:** neither VDOS path had any test before this audit.

### HIGH — Ablation runner did not ablate (`vibropredict/evaluation/ablation.py`)

`no_sequence` / `no_chemical` variants re-ran the full model, so 3 of 4 rows were
identical. Added real `drop_sequence` / `drop_chemical` flags to
`VibroPredictHybrid.forward` and rewired the loop. Verified end-to-end: all four
variants now give distinct metrics. Guard: `vibropredict/tests/test_ablation.py`.

### LOW — fixed

- **Flaky ranking-loss test** — rewritten to use a provably-violated ranking.
- **4 wandb tests** — mock lacked `__spec__`; added a valid spec.
- **Entropy unit mislabel** in `scripts/real_world_petase.py` — "kB" → J/(mol·K)
  (value is R·Σ); JSON key renamed.
- **README test count** — "103+" → "180+" (actual: 187).

---

## Open findings (not yet fixed — prioritized)

| # | Severity | Item | Location |
|---|---|---|---|
| 1 | HIGH | ~~Headline "R²>0.75" unsubstantiated~~ **RECONCILED** — README + both roadmap docs now flag it as an unmet target | README, `docs/future/VIBROPREDICT_*` |
| 2 | HIGH | Physics thesis **tested and negative** — powered curated homolog test (n=27) confirms no signal; documented, not hidden | `docs/PHYSICS_SIGNAL_VALIDATION.md` |
| 3 | MEDIUM | ~~Inference CLI non-functional~~ **FIXED** — real NMA VDOS via `--pdb`; honest checkpoint error; warned length-only fallback | `src/inference.py`, `src/cli.py` |
| 4 | LOW | Bundled data is placeholder (4-row CSV, `HEADER DUMMY` PDB) | `data/` |
| 5 | LOW | ~~10 bundled checkpoints don't match CLI names~~ **ADDRESSED** — resolver now raises an honest error naming the toy stubs | `checkpoints/` |

Modules read and found structurally clean (no correctness bug detected): GNN
GATv2 encoder (`src/models/gnn.py`), structure pipeline (ESMFold / SIFTS / QC),
baselines (`catpred` raises rather than fakes), `sota_comparison` (literature
numbers flagged `source: published`), split logic (`splits.py` EC-holdout is a
valid OOD design), `spectral_generation.py`.

---

## Documentation cleanup

The repo has 56 `.md` files. Most are the intentional Obsidian **vault**
(`vault/**`, 40 files) and planning docs with YAML frontmatter — these are
deliberate, keep them. Recommended actions:

**Delete (safe — generated/cruft):**
- `.pytest_cache/README.md` — auto-generated, already git-ignored; not a repo doc.

**Reconcile (contain now-false claims, edit rather than delete):**
- `README.md`, `docs/future/VIBROPREDICT_PHASE2_REPORT.md`,
  `docs/future/VIBROPREDICT_INTEGRATION_ROADMAP.md` — all assert R²>0.75 with no
  run behind it. Update to reflect the dry-run reality + the physics-validation
  result.

**Consolidate (redundant audit docs):**
- `AUDIT_REPORT.md` (this session's first-pass) is now fully subsumed by **this**
  `AUDIT.md`. Recommend deleting `AUDIT_REPORT.md` and keeping `AUDIT.md`.

No other `.md` is dead: the `docs/future/*` reports are historical planning
records, and `DATASET_AUDIT.md` / `ROADMAP.md` are the project's own honest
self-audits worth preserving.

---

## What to work on next (backlog, roughly by value)

1. **Reconcile the R²>0.75 claim** (HIGH, cheap) — make README + 2 docs honest.
   Stops the repo advertising a number nothing produces.
2. **Curated physics validation** (HIGH, medium) — ≥30–50 length/construct-matched
   thermophile/mesophile pairs (ProThermDB / Meltome), ideally conformer
   ensembles. The definitive test of the core thesis. n=8 was underpowered.
3. **Fix the inference CLI** (MEDIUM) — wire real NMA VDOS into the kcat path,
   resolve checkpoint-name mismatch, so `predict-*` actually runs.
4. **Tri-modal benchmark with honest ablation** (MEDIUM, **BLOCKED on data**) —
   the runner + ablation are now correct, but the real labelled kcat split
   (KinHub/RealKcat) is not in the repo (bundled data is 4-row toy CSVs) and the
   encoders need ProtT5-XL (~11 GB) + ChemBERTa + GPU-hours. Deliberately NOT run
   on toy data — that would manufacture exactly the unsubstantiated number this
   audit flags. Needs: real dataset added, then a GPU run measuring the spectral
   branch's gate weight and marginal R².
5. **Strengthen the physics** (research) — per-residue fluctuation profiles or
   conformer-ensemble VDOS instead of scalar moments; re-test.
6. **Package a small real demo dataset** (LOW) — replace placeholder data so the
   pipeline is runnable end-to-end out of the box.

---

## File inventory (source)

- `src/` (Phase 1-2 QDD): 14 modules — nma_analysis, spectral_generation,
  models/{gnn,cnn,multimodal,losses}, training, datasets, cli, inference,
  data_acquisition, utils.
- `vibropredict/` (Phase 3): 30 modules — models/{sequence,chemical,fusion,hybrid},
  spectra/{gnm_calculator,vdos_engine}, structures/{esmfold,sifts,qc},
  data/{kinhub,splits,standardization,enzyextract,dataset},
  evaluation/{ablation,sota_comparison,visualization,baselines/*}, training/*.
- Tests: 17 files, 187 tests (8 in `tests/`, 9 in `vibropredict/tests/`).
- Scripts: `real_world_petase.py`, `run_benchmarks.py`, `audit_kinhub_vs_realkcat.py`.
- Notebooks: 4 in `notebooks/`, 2 Colab.
