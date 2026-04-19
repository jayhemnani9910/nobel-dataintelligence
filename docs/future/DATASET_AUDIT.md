---
title: Dataset Audit — KinHub vs RealKcat
aliases:
  - Dataset Audit
  - KinHub Audit
tags:
  - audit
  - dataset
  - phase-1
date: 2026-04-19
---

# Dataset Audit: KinHub vs RealKcat

> [!IMPORTANT]
> **TL;DR** — RealKcat (2025) reports 27,176 curated entries, suspiciously close to our KinHub ~27k. This audit attempts to quantify overlap. **If overlap is high (>50%), benchmark comparisons against RealKcat are invalid** because our test set is not held-out relative to their training data.

## Background

| Dataset | Entries | Source | Paper |
|---------|---------|--------|-------|
| **KinHub** (this project) | ~27,000 | Curated from BRENDA, SABIO-RK, literature | Internal |
| **RealKcat** (2025) | 27,176 | Gradient-based framework; curated from BRENDA + SABIO-RK + UniProt | [PMC11844551](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844551/) |

Both datasets source from the same upstream databases (BRENDA, SABIO-RK), so overlap is **expected** and must be measured before any fair benchmark.

## Methodology

The audit script (`scripts/audit_kinhub_vs_realkcat.py`) performs:

1. **Download** — Attempts to fetch RealKcat's supplementary dataset from known public URLs (GitHub, Zenodo).
2. **Normalize** — Canonicalizes substrate SMILES via RDKit (if available) to prevent false negatives from SMILES representation differences.
3. **Join** — Outer-merges on `(uniprot_id, substrate_smiles)` to identify entries present in one or both datasets.
4. **Report** — Writes `data/audits/kinhub_realkcat_overlap.csv` with per-entry membership flags and k_cat values from each source.

### Running the audit

```bash
python scripts/audit_kinhub_vs_realkcat.py \
    --kinhub data/kaggle/kinhub.csv \
    --output data/audits/kinhub_realkcat_overlap.csv
```

The script is **idempotent** — re-running produces identical output.

## Findings

> [!WARNING]
> **Status: BLOCKED** — As of 2026-04-19, no public URL for RealKcat's raw (enzyme, substrate, k_cat) table has been located. The paper ([PMC11844551](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844551/)) references supplementary data, but a download link verified to serve the dataset has not been found at time of writing.

### What to check when updating this audit

When re-running the audit, check these sources (none of these have been verified to host the dataset at time of writing — confirm before trusting):

- The paper's supplementary materials page on the PMC / journal site.
- A data-availability statement in the published PDF.
- The RealKcat authors' institutional / lab GitHub organization (search for recent repositories).
- Zenodo / Figshare — search by paper DOI.
- Contact the corresponding author directly if no public source is found.

The audit script (`scripts/audit_kinhub_vs_realkcat.py`) accepts a ``--realkcat-path`` argument — point it at a local copy once obtained, and the overlap analysis runs without any network calls.

### Minimal-risk assumption

Given that both KinHub and RealKcat source from BRENDA + SABIO-RK:

- **Assume high overlap** (>60%) until proven otherwise.
- **Flag all benchmark results** against RealKcat as potentially invalid.
- When RealKcat's data becomes available, re-run the audit and update this document.

## Recommendations

1. **Contact the RealKcat authors** to request their supplementary dataset for a proper overlap analysis.
2. **Do not claim superiority over RealKcat** in any publication until overlap is measured and accounted for.
3. When overlap is measured:
   - If **<20%**: datasets are sufficiently independent; proceed with benchmark.
   - If **20–60%**: report overlap percentage alongside benchmark numbers; consider a deduplicated evaluation.
   - If **>60%**: benchmark is invalid without a fully deduplicated OOD evaluation.

## Output schema

`data/audits/kinhub_realkcat_overlap.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `uniprot_id` | str | UniProt accession |
| `substrate_smiles` | str | Substrate SMILES string |
| `in_kinhub` | bool | Present in our KinHub dataset |
| `in_realkcat` | bool | Present in RealKcat (None if data unavailable) |
| `kcat_kinhub` | float | k_cat value from KinHub |
| `kcat_realkcat` | float | k_cat value from RealKcat (None if data unavailable) |

## Links

- Audit script: [`scripts/audit_kinhub_vs_realkcat.py`](../../scripts/audit_kinhub_vs_realkcat.py)
- KinHub dataset note: [`vault/datasets/KinHub.md`](../../vault/datasets/KinHub.md)
- RealKcat paper: [PMC11844551](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844551/)
- Roadmap context: [ROADMAP.md § Tier 1.5](ROADMAP.md)
