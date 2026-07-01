# Benchmark Runbook — running the VibroPredict tri-modal benchmark

*How to take the tri-modal kcat benchmark from "blocked on data" to a real,
honest R² on a held-out split. Every path, column name, and command below was
read from the current source; nothing here is aspirational.*

This is the one audit item that could not be completed in-sandbox (see
[`AUDIT.md`](../AUDIT.md) item #4): the code paths are correct and the ablation
now genuinely ablates, but there is **no real labelled data in the repo** (the
bundled `data/kaggle/*.csv` are 4-row toy files) and the encoders need large
model downloads + GPU. Follow these steps on a GPU box with the data in hand.

---

## 0. Prerequisites

- A CUDA GPU (ProtT5-XL inference is the bottleneck; ~11 GB model, fits in
  16 GB VRAM at batch 1–4 with fp16).
- The `qdd`-equivalent environment plus HuggingFace `transformers` and
  `sentencepiece` (for the ProtT5 `T5Tokenizer`).
- Two pretrained encoders download on first use:
  - Sequence: `Rostlab/prot_t5_xl_uniref50` (~11 GB)
  - Chemical: `seyonec/ChemBERTa-zinc-base-v1` (~150 MB)

```bash
export KMP_AFFINITY=disabled OMP_NUM_THREADS=4 PYTHONPATH=.
pip install transformers sentencepiece
```

---

## 1. Get the data into the schema the loaders expect

The KinHub loader (`vibropredict/data/kinhub.py`) validates these **required
columns** and drops rows with nulls in them:

```
REQUIRED_COLUMNS = ["uniprot_id", "k_cat", "substrate_smiles"]
```

Duplicate `(uniprot_id, substrate_smiles)` pairs are collapsed by **geometric
mean of `k_cat`** automatically. Useful optional columns:

| Column | Used by | Purpose |
|---|---|---|
| `product_smiles` | chemical encoder (DRFP) | reaction fingerprint; falls back to substrate-only if absent |
| `ec_number` / `ec` | `ECHoldoutSplit` | derives `ec_column` for the OOD split (first `ec_level` digits, default 2) |
| `sequence` | sequence encoder | if absent, sequence must be resolvable from `uniprot_id` upstream |

Source options (both flagged in `docs/future/DATASET_AUDIT.md`):
- **KinHub** — the project's own curated kcat set.
- **RealKcat** (27,176 entries) — larger, but **audit the overlap with KinHub
  first** (`scripts/audit_kinhub_vs_realkcat.py`) or the OOD split leaks.

Save the cleaned table as e.g. `data/kinhub/kinhub.csv`.

---

## 2. Precompute VDOS spectra (one .npy per protein)

The dataset (`vibropredict/data/enzyme_kinetics_dataset.py`) loads a VDOS vector
per row from:

```
{vdos_dir}/{uniprot_id}_vdos.npy
```

Missing files are silently zero-filled with `has_vdos=False` (so a run with an
empty `vdos_dir` trains a de-facto bimodal model — check this flag if the
spectral gate looks dead). Generate the real spectra with the **now-fixed**
pipeline — one structure per `uniprot_id` (AlphaFold or PDB):

```python
import numpy as np, pathlib
from vibropredict.spectra.vdos_engine import VibroEnzymePipeline  # corrected GNM scaling

pipe = VibroEnzymePipeline(n_points=1000, freq_max=500.0, broadening=5.0)
out = pathlib.Path("data/spectral"); out.mkdir(parents=True, exist_ok=True)
for uniprot_id, pdb_path in structures.items():        # you supply this mapping
    vdos, _ = pipe.generate_vdos(str(pdb_path))
    np.save(out / f"{uniprot_id}_vdos.npy", vdos.astype("float32"))
```

> The eigenvalue→wavenumber scaling here was the critical bug fixed this audit
> (`ENM_FREQ_CM1_PER_SQRT_EIGVAL`). Sanity-check: two different proteins must
> give **distinct** spectra (cosine < 0.999) — if they don't, you are on an old
> checkout.

---

## 3. Choose the split

`vibropredict/data/splits.py` provides two, both returning
`{"train","val","test"}` DataFrames:

- `RandomSplit(train_ratio=0.8, val_ratio=0.1, seed=42)` — the easy IID number.
- `ECHoldoutSplit(ec_level=2)` — **the honest one.** Holds out whole EC
  subclasses so test enzymes are unlike anything trained on. Report this as the
  headline; the random split flatters every model.

Always report **both**, and never quote the random-split R² without the OOD one
beside it.

---

## 4. Train

Training is driven from `colab/train_vibropredict.ipynb` (and
`vibropredict/notebooks/07_model_training.ipynb`) using
`TrainerWithMMDrop` (`vibropredict/training/trainer.py`). The model is
`VibroPredictHybrid(fusion_dim=512, dropout=0.2)`. Key knobs:

- **MM-Drop**: modality dropout during training — the regularizer that forces
  the fusion gate not to collapse onto one branch. Keep it on.
- **Ranking loss**: `MutantRankingLoss(margin=0.1)`, weighted `lambda_rank=0.1`.
- Log to W&B by setting `WANDB_PROJECT=vibropredict` (optional; guarded).

Save the checkpoint as a dict with a `model_state_dict` key (what
`predict_kcat` / `run_benchmarks.py` expect) — e.g.
`checkpoints/vibropredict_best.pt`.

---

## 5. Run the benchmark

`scripts/run_benchmarks.py` compares VibroPredict against the published
baselines and writes `benchmarks/benchmarks.json` + plots.

```bash
python scripts/run_benchmarks.py \
    --checkpoint checkpoints/vibropredict_best.pt \
    --kinhub     data/kinhub/kinhub.csv \
    --vdos-dir   data/spectral/ \
    --output-dir benchmarks/ \
    --device     cuda
```

Flags: `--dry-run` (what `benchmarks.json` currently is — pending rows, no
compute), `--skip-vibropredict` (baselines only), `--skip-baselines`. The
baselines live in `vibropredict/evaluation/baselines/` and **raise** if their
setup isn't present rather than fabricating a number (CatPred needs its own
install; see `catpred.py`).

---

## 6. The measurement that actually matters — honest ablation

A single R² does not tell you whether the **physics branch earns its place**.
The audit fixed the ablation runner so this is now a real experiment. Run
`vibropredict/evaluation/ablation.py` (`run_ablation`) with the trained model:

- It evaluates 4 variants via the real drop flags: full, `drop_spectral`,
  `drop_sequence`, `drop_chemical` — each zeroing one modality before fusion.
- Report **ΔR² (full − drop_spectral)** on the **EC-holdout** split. This is the
  marginal contribution of the VDOS branch. Also report the mean **fusion gate
  weight** on the spectral branch.

Decision rule, stated up front (pre-register it):
- ΔR²_spectral ≤ ~0 on the OOD split ⇒ the physics branch is **not helping**;
  the honest headline is a bimodal (sequence+chemical) model, and the negative
  [`PHYSICS_SIGNAL_VALIDATION.md`](PHYSICS_SIGNAL_VALIDATION.md) result (n=27,
  all p>0.39) predicts exactly this. Do not bury it.
- ΔR²_spectral clearly > 0 **and** it survives the EC-holdout ⇒ genuine signal;
  now you have something publishable, and it directly contradicts the n=27
  single-structure result in an interesting way worth explaining (e.g. the
  learned SpectralCNN extracts more than the scalar moments we tested).

---

## 7. Update the claims

Whatever the number is, propagate it honestly:
- `README.md` benchmark table (currently marked "not yet benchmarked").
- The status banners in `docs/future/VIBROPREDICT_PHASE2_REPORT.md` and
  `VIBROPREDICT_INTEGRATION_ROADMAP.md`.
- `AUDIT.md` items #1 and #4.

The whole point of the audit was to stop the repo advertising an unearned
R²>0.75. Replace it with a measured number and its OOD counterpart, or with a
plain statement that the physics branch did not help — either is a real result.
