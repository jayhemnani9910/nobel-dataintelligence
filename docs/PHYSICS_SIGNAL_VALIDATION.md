# Physics-Signal Validation (Phase 1 of the benchmark)

*Does the corrected VDOS / vibrational-entropy feature actually carry a
thermostability signal? This is the make-or-break test before investing in a
full tri-modal training run.*

## Design

A within-family thermophile-vs-mesophile comparison, which needs no recalled
melting temperatures — organism thermophilicity is the ground truth and the
direction is the test. For 8 protein families we pulled one thermophile and one
mesophile structure from RCSB (X-ray, ≤2.5 Å, verified organism metadata),
restricted each to its first protein chain for a fair monomer comparison, and
computed the corrected physics features (fixed eigenvalue→wavenumber scaling):

- **Vibrational entropy per residue** — hypothesis: thermophile *lower*.
- **Softest-mode wavenumber** — hypothesis: thermophile *stiffer* (higher).
- **Mean VDOS frequency** — hypothesis: thermophile *higher*.

Pairs: rubredoxin, cold-shock protein, adenylate kinase, GAPDH, superoxide
dismutase, triosephosphate isomerase, ribonuclease H, phosphofructokinase.

## Result — negative

| Metric | Pairs matching hypothesis | Wilcoxon p |
|---|---|---|
| Entropy/residue (thermo lower) | 4 / 8 | 0.42 |
| Softest mode (thermo stiffer) | 3 / 8 | 0.88 |
| Mean VDOS freq (thermo higher) | 2 / 8 | 0.97 |

Direction agreement is at or below chance on every metric; no test approaches
significance. Restricting to the 6 pairs with well-matched chain lengths
(ratio < 1.15) does not improve it (2/6 on each metric). See
`physics_signal_validation.png` and `physics_signal_features.csv`.

## Interpretation

Two distinct conclusions, kept separate:

1. **The units-bug fix is still correct and necessary.** Post-fix, VDOS spectra
   are protein-specific (PETase vs LCC cosine 0.984 vs the old 0.99999999), and
   the regression tests confirm distinct folds give distinct spectra. That bug
   had to be fixed regardless.

2. **Single-structure coarse-grained ANM vibrational entropy is not, on its own,
   a usable thermostability predictor here.** This is consistent with the NMA
   literature: a single X-ray snapshot, a uniform 15 Å ENM with γ=1, and no
   solvent / ligand / oligomeric-state modelling discard most of what sets
   thermostability. The confounds in this test (differing constructs, ligands,
   resolutions) are real and would need controlling in a larger study.

## Follow-up: stronger features (also negative)

Because scalar entropy moments discard spatial information, we tested
better-motivated features on the same 8 pairs:

| Feature (hypothesis) | Pairs matching | Wilcoxon p |
|---|---|---|
| Mean MSF — thermophile more rigid | 5 / 8 | 0.53 |
| Top-10% MSF — rigid flexible loops | 3 / 8 | 0.77 |
| Spectral gap λ₂/λ₁ (scale-free) — higher | 6 / 8 | 0.38 |
| Contact density (8 Å) — thermophile denser | 3 / 8 | 0.63 |

None reaches significance. Notably, **contact density — a non-vibrational
feature the stability literature does support — also fails here (3/8)**. That a
feature which *should* separate thermophiles doesn't, points to the test being
**underpowered and confounded** (n=8; pairs differ in construct, bound ligands,
oligomeric state, and resolution) as much as to weakness in the physics features
themselves. Continuing to try features on the same 8 noisy pairs would be
fishing, so we stop here.

**What a real test needs:** a curated pair set of ≥30–50 thermophile/mesophile
homologs controlled for length, construct, and ligand state, and ideally
conformer ensembles rather than single X-ray snapshots. **This was subsequently
built (n=27 validated pairs) — see the powered follow-up below, which confirms
the negative.**

## Powered follow-up (n=27) — negative confirmed

The n=8 test above was underpowered, so we built the curated set it called for:
**27 thermophile/mesophile enzyme homolog pairs**, each PDB identity **verified
by its RCSB polymer-entity name** (full-text search alone produced 18/28 false
positives — wrong protein under the right keyword — all filtered out). Structures
are X-ray ≤2.5 Å, restricted to the first chain. Organisms span 5 thermophiles
(*Thermotoga maritima*, *Thermus thermophilus*, *Pyrococcus furiosus*,
*Pyrococcus horikoshii*, *Methanocaldococcus jannaschii*) vs 2 mesophiles
(*Escherichia coli*, *Clostridium pasteurianum*).

| Feature (hypothesis) | Pairs matching | Wilcoxon p |
|---|---|---|
| Entropy/res — thermophile lower | 12 / 27 | 0.83 |
| Softest mode — thermophile stiffer | 13 / 27 | 0.47 |
| Mean MSF — thermophile more rigid | 14 / 27 | 0.46 |
| Mean VDOS freq — thermophile higher | 13 / 27 | 0.39 |
| Contact density — thermophile denser | 13 / 27 | 0.81 |

**Every feature sits at chance (44–52 %); no test approaches significance.** A
length-matched subset (ratio < 1.10, n=16) is no better. See
`curated_physics_validation.png` and `curated_physics_features.csv`.

The result is now robust, not a small-sample artifact. Even **contact density**
— a feature the stability literature genuinely supports — fails at n=27, which
points to the common cause: **a single static X-ray snapshot analysed with a
uniform coarse-grained network model discards the determinants of
thermostability** (electrostatics, packing detail, solvent, sequence-level
substitutions), regardless of which scalar summary is extracted. This does *not*
refute the units-bug fix (spectra are correctly protein-specific now); it says
single-structure NMA summaries are the wrong *feature*, not that the code is wrong.

## Implication for Phase 2 (full tri-modal benchmark)

The physics branch shows **no univariate thermostability signal** in this test.
That does not strictly prove it adds nothing inside a trained fusion model — a
learned SpectralCNN could extract structure the raw moments miss — but it
substantially lowers the prior that "physics-informed beats sequence-only" will
hold, and it means a multi-hour ProtT5/ChemBERTa training run is a speculative
investment rather than a validation of a demonstrated signal.

**Recommendation:** treat the physics branch as *unproven* rather than
*validated*. Before a full training run, the higher-value work is either
(a) strengthening the physics itself (ensemble over multiple conformers,
solvent/ligand-aware ENM, or per-residue fluctuation profiles instead of scalar
moments), or (b) running the tri-modal model with an *honest ablation* (now
fixed) on a real labelled split and reporting whether the spectral branch's gate
weight and marginal R² are non-trivial — a cheaper test than assuming it helps.
