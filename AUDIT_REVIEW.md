# Independent Audit Review — Quantum Data Decoder / VibroPredict

*Independent, from-scratch re-audit of the whole repository. The pre-existing
`AUDIT.md` was treated as an **untrusted set of claims** and every material
assertion in it was re-verified by reading the code and running it. This file
records what I found independently; it does not inherit `AUDIT.md`'s
conclusions.*

**Auditor environment:** conda env `qdd` — Python 3.11.15, torch 2.12.1,
numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, prody 2.6.1, rdkit 2026.03.3,
biopython 1.87. **Not present in the env:** `transformers`, `torch_geometric`
(installed during audit), `openmm`. Sandbox note: the env aborts on import
unless `KMP_AFFINITY=disabled` (OpenMP `pthread_setaffinity_np` is blocked); the
package is also not installed editable, so `python -m src.cli` needs
`PYTHONPATH=.`.

---

## Verdict

This is careful, honest, well-tested research engineering. The central physics
machinery (VDOS from normal-mode analysis) is **correct**, the evaluation
harness is **genuinely honest** (no fabricated numbers; missing baselines raise
rather than fake), and the headline scientific result is reported against
itself: the physics/VDOS branch is shown to add **no** predictive signal, and
the repo says so plainly rather than hiding it.

The defects that remain are **integration/packaging** issues, not scientific
fraud or broken math:

1. the one usable bundled checkpoint cannot be loaded by the inference CLI
   because of a `num_go_terms` shape mismatch (CRITICAL for "run end-to-end");
2. `torch_geometric` (a declared requirement) is absent from the env, so the
   GNN/stability path raises on import until installed (HIGH);
3. the CLI is not runnable as documented (`python -m src.cli` fails without
   `PYTHONPATH=.` / editable install) (MEDIUM);
4. `inference.py` and the prior `AUDIT.md` **mischaracterize the bundled
   checkpoints** — they claim *all* `best_model_epoch*.pt` are "2-parameter
   linear stubs," but epoch 6/9/10 are real 47 MB `VibroStructuralModel`
   checkpoints with 5.76 M parameters (MEDIUM — accuracy of documentation).

**Test suite at audit start:** 180 passed, 10 skipped (190 collected). No
failures. I reproduced this exactly.

---

## Re-verification of the existing `AUDIT.md` claims

| Prior claim (AUDIT.md) | My independent finding | Status |
|---|---|---|
| VDOS eigenvalue→wavenumber bug fixed in `src/nma_analysis.py` with constant ≈108.59 | Re-derived the constant from first principles (γ=1 kcal/mol/Å², unit mass): **108.59135850**, exact match. Old factor `1/(2π·29979.2458)` is **2.045×10⁷×** too small. | **CONFIRMED** |
| Same fix in `vibropredict/spectra/vdos_engine.py`, single shared constant | `vdos_engine.py` imports `ENM_FREQ_CM1_PER_SQRT_EIGVAL` from `nma_analysis` — single source of truth. Verified. | **CONFIRMED** |
| VDOS spectra now protein-distinct | Ran both paths on real PDBs 6eqe (PETase) vs 6ths: ANM VDOS cosine **0.939**, GNM **0.990** (both <1, peaks off bin 0). Thermostable 6ths correctly stiffer softest mode (134 vs 31 cm⁻¹) and lower S_vib (357 vs 406 J/mol/K). | **CONFIRMED** |
| Ablation runner now truly ablates (was re-running full model) | `forward()` zeros each modality embedding; loop passes distinct `drop_*` flags. Independent probe: all 4 variants give distinct predictions; `no_sequence` flattens output (sequence dominance). | **CONFIRMED** |
| Baselines honest — CatPred raises rather than fakes | `baselines/catpred.py` `predict()` raises `RuntimeError` with setup instructions when unavailable; `sota_comparison.py` tags literature numbers `source: "published"`. | **CONFIRMED** |
| Headline R²=0.45 random / 0.07 OOD; physics ΔR²≈0 | `benchmarks/benchmarks.json` internally consistent; ablation `no_spectral`≈`full` ⇒ ΔR²(spectral)=+0.005/−0.002. **But** these numbers were produced on Colab A100 and are **asserted**, not reproducible in this env (no GPU/transformers/ProtT5). | **CONFIRMED (internally); NOT independently reproducible here** |
| "All bundled `best_model_epoch*.pt` are 2-parameter smoke-test stubs" | **False.** epoch1–3 are 2-param stubs (3.9 KB); **epoch4 is a 2-class head; epoch6/9/10 are real 47 MB VibroStructuralModel checkpoints (5.76 M params, cafa_head=10000).** | **REFUTED** |
| Inference CLI "FIXED — real NMA VDOS via `--pdb`; honest checkpoint error" | Real-NMA path and honest missing-checkpoint error are present and work. **But** loading the real epoch10 checkpoint fails with a `num_go_terms` size mismatch — so the CLI is *not* actually runnable with the one real bundled weight. | **PARTIAL** |
| `data/kaggle` is placeholder (4-row CSV, HEADER DUMMY PDB) | Confirmed: `train.csv` has 4 toy rows (`ACDE,7.0,50.0`), PDB is `HEADER DUMMY`. | **CONFIRMED** |
| 7,057 precomputed CatPred VDOS are real | Confirmed: 7057 `.npy` files, all shape (1000,), peaks spread bins 721–981, none at bin 0, pairwise cosine 0.92–0.998 — protein-distinct, non-degenerate. | **CONFIRMED** |

---

## Findings (independent), severity-ranked

### CRITICAL — inference CLI cannot load the one real bundled checkpoint
**`src/inference.py:121`** hardcodes `num_go_terms=100` when reconstructing
`VibroStructuralModel`, but the real checkpoints (`best_model_epoch6/9/10.pt`)
were trained with `num_go_terms=10000` (and the model default is 10000).
Result:
```
RuntimeError: Error(s) in loading state_dict for VibroStructuralModel:
  size mismatch for cafa_head.6.weight: checkpoint [10000, 256] vs model [100, 256]
```
So `predict-stability --checkpoint checkpoints/best_model_epoch10.pt` fails.
**Repro:** `PYTHONPATH=. python -m src.cli predict-stability --sequence AAAA
--checkpoint checkpoints/best_model_epoch10.pt --pdb data/pdb/6eqe.pdb`.
**Fix:** infer `num_go_terms` from `checkpoint["model_state_dict"]
["cafa_head.6.weight"].shape[0]` rather than hardcoding. Verified: with the
inferred value the state_dict loads and the novozymes head runs end-to-end.

### HIGH — declared dependency `torch_geometric` absent → GNN/stability path dead on import
`requirements.txt` lists `torch-geometric>=2.2.0`, but it (and `psutil`,
`xxhash` it needs) were not in the `qdd` env. `src/models/gnn.py` raises
`ImportError` on import, taking down `predict-stability`, the novozymes
pipeline's model, and anything importing the GNN. Installed
`torch_geometric==2.8.0 + xxhash + psutil` during the audit to unblock. The
env should be reconciled with `requirements.txt` (this is why several
`test_models` GNN tests are **skipped**, not run).

### MEDIUM — CLI not runnable as documented
README shows `python -m src.cli …`, but the package is not installed editable
and `pyproject`'s console-script is `qdd`. Out of the box `python -m src.cli`
gives `ModuleNotFoundError: No module named 'src'`; it only works with
`PYTHONPATH=.` or `pip install -e .`. Either document `PYTHONPATH=.`/`qdd …`
or ship an editable install step.

### MEDIUM — checkpoints mischaracterized in code + prior audit
`src/inference.py` `_resolve_checkpoint` docstring and the hint string tell the
user *all* bundled `best_model_epoch*.pt` are "toy smoke-test stubs (a
2-parameter linear stub)." That is true only for epoch1–3. epoch6/9/10 are real
5.76 M-param `VibroStructuralModel` weights. This misleads a user away from the
one checkpoint that (after the CRITICAL fix) actually loads. `AUDIT.md` repeats
the same false claim.

### LOW — env/reproducibility hygiene
- `transformers` and `openmm` (both in `requirements.txt`) are missing from
  `qdd`; the ProtT5/ChemBERTa tri-modal path and any OpenMM feature are
  unimportable locally. Tests mock these, so the suite passes, but the tri-modal
  model cannot be instantiated locally.
- The suite works only with `KMP_AFFINITY=disabled`; worth documenting in the
  test runbook / a `conftest.py` env guard.
- `data/kaggle` placeholder means the documented `novozymes --data-dir
  ./data/kaggle` runs but on 4 toy rows → `Metric: nan` (constant-variance
  Spearman). Not a bug, but the "runnable out of the box" story is toy-only.

### Not defects (checked and clean)
`TriModalFusion` (softmax gating), `MutantRankingLoss` (margin-ranking on sign
of target diff), `metrics.py` (rmse/R²/pearson/spearman/top-k), MM-Drop
`TrainerWithMMDrop` (grad clip, early stopping, checkpointing), `losses.py`
(Pearson/Focal/Contrastive/Combined; the misnamed-Spearman class is explicitly
aliased and commented), `spectral_generation.py`, GNM/ANM analyzers, split
logic. The 39 targeted model/training/ablation/loss tests pass.

---

## Fixes applied

| Finding | Fix | File(s) | Verified by |
|---|---|---|---|
| CRITICAL — `num_go_terms=100` hardcode broke real-checkpoint load | Infer `num_go_terms` from `checkpoint["model_state_dict"]["cafa_head.6.weight"].shape[0]` | `src/inference.py` | epoch10 now loads + runs |
| CRITICAL follow-on (found while fixing) — coords/features length mismatch → `IndexError` in `GATv2Conv` when sequence shorter than CA count | Clip **both** coords and features to `min(len)`; the old code truncated only `features` to the larger value (a no-op) | `src/inference.py` | `predict-stability` runs with short seq + real PDB |
| HIGH — declared dep `torch_geometric` absent from env | Installed `torch_geometric 2.8.0 + xxhash + psutil` into `qdd` | env `qdd` | GNN imports; 8 previously-skipped `test_models` GNN tests now run + pass |
| MEDIUM — checkpoints mischaracterized as "all 2-param stubs" | Corrected docstring + missing-checkpoint hint to distinguish epoch1-3 stubs from epoch6/9/10 real weights | `src/inference.py` | text review |
| MEDIUM — CLI not runnable as documented | Added `pip install -e .` + `PYTHONPATH=.`/`KMP_AFFINITY` note to Quick Start | `README.md` | `python -m src.cli` / `qdd` both work |
| Regression guard | New test loads the real epoch10 checkpoint and exercises the short-seq alignment path | `tests/test_inference.py` | passes |

**Regression pass:** full suite **191 passed, 0 skipped** (was 180 passed / 10
skipped at audit start). Net: +1 new regression test, and all 10 formerly-skipped
tests now execute (0 skipped) — 8 of them the `test_models` GNN tests unblocked by
installing torch_geometric — and pass. No pre-existing test broke.

**Not changed (deliberately):**
- Placeholder `data/kaggle` and `data/cafa5` left as-is — they let the pipeline
  run out of the box; replacing them with real datasets is a separate data task.
- The headline Colab-A100 kcat benchmark was not re-run (no local GPU /
  `transformers` / ProtT5 weights). Its numbers were checked for internal
  consistency and honest labeling, not independently reproduced.
- The scientific conclusion (physics/VDOS branch adds no signal) is left intact:
  it is well-supported by the code and two independent negative results, and is
  reported honestly.

## Final verdict

After independent re-verification and fixes, the repository is **scientifically
honest and now runnable end-to-end locally** for the CPU-only surface (NMA/VDOS,
`predict-stability` with the real bundled checkpoint, the novozymes pipeline on
toy data, and the benchmark dry-run). The physics core is correct, the
evaluation harness does not fabricate, and the one genuinely broken user path
(inference on the real checkpoint) has been fixed with a regression guard. The
remaining limitations are honest data/compute constraints (placeholder datasets;
GPU-only tri-modal benchmark), not defects in the code.
