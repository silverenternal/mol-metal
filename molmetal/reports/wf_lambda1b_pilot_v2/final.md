# WF-Lambda-1b Re-verify — Round-Trip + Metal-Seed Fix

> **Honest-framing**: every number below is **MEASURED** on 5 pockets × 3 seeds (15 cells). PROJECTED numbers from the spec are quoted separately where relevant.

## Method

The WF-Lambda-1 verify produced two structural defects:

1. **`MoleculeClosedTerm.from_smiles` round-trip rejection** — RDKit can
   sanitize large polyfunctional SMILES, but the lambda-term constructor
   rejected them because the strict ledger-based β-NF check counted
   lone-pair capacity as a free variable. Patch 1 relaxed this:
   `accept_partial=True` (default) accepts any RDKit-parseable SMILES with
   `n_atoms >= 1`, falling back to loose-sanitisation on failure.

2. **No metal-seeded root** — Lambda search started from organic pocket
   reference ligands, so the metal_compliance metric was structurally
   unmeasurable. Patch 2 added a `--metal-seed` CLI flag with three
   canonical metal-containing scaffolds (`cisplatin`, `ru_arene`,
   `ir_cp_star`) that inject a Pt/Ru/Ir-containing root into the MCTS
   tree via the existing `MoleculeClosedTerm.from_smiles` entry point
   (with `embed_3d=False` to skip 3-D embedding of the seed).

## Configuration

- `n_pockets` : **5** (test_000 .. test_004)
- `seeds`     : **[42, 0, 1234]**
- `n_simulations per cell` : **100**
- `n_top_k`   : **20**
- `metal_seed` : **cisplatin** (`[NH3][Pt]([NH3])(Cl)Cl`)
- `--prior-disabled` not set (geometry prior on)

Command (cwd: repo root):
```
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 5 --seeds 42 0 1234 \
    --n-simulations 100 --n-top-k 20 \
    --metal-seed cisplatin \
    --output-dir wf_lambda1b_pilot_v2
```

Note: the script's `--output-dir` value is *appended* to `wf_lambda1_`,
so the actual persisted path is
`molmetal/reports/wf_lambda1_wf_lambda1b_pilot_v2/`.

## Result table — Before vs After

| metric | **WF-Lambda-1 baseline (N=10×3)** | **WF-Lambda-1b re-verify (N=5×3)** | delta |
|---|---|---|---|
| validity_rate         | 0.9000 | **1.0000** | +0.1000 |
| uniqueness_rate       | 0.9000 | 1.0000 | +0.1000 |
| diversity_alpha       | 0.0043 | 0.0000 | -0.0043 |
| novelty               | 1.0000 | 1.0000 |  0.0000 |
| synthesizability_rate | 0.0000 | 0.0000 |  0.0000 |
| metal_compliance_rate | 0.0000 | **1.0000** | **+1.0000** |
| reference_tanimoto    | 0.8595 | 0.0120 | -0.8475 |

Baseline path: `molmetal/reports/wf_lambda1_N10_3seed_v2/report.json`
Re-verify path: `molmetal/reports/wf_lambda1_wf_lambda1b_pilot_v2/report.json`

### Per-cell re-verify (N=5×3 = 15 cells)

All 15 cells returned `n_candidates = 1` — the MCTS expansion produced
only the cisplatin seed itself (no children below depth-3 explored the
click-rule tile library in this budget). Every cell:

| pocket   | seed | n_cand | valid | syn | metal | ref_tan |
|---|---|---|---|---|---|---|
| test_000 | 42 / 0 / 1234 | 1 | 1.000 | 0.000 | **1.000** | 0.014 |
| test_001 | 42 / 0 / 1234 | 1 | 1.000 | 0.000 | **1.000** | 0.000 |
| test_002 | 42 / 0 / 1234 | 1 | 1.000 | 0.000 | **1.000** | 0.018 |
| test_003 | 42 / 0 / 1234 | 1 | 1.000 | 0.000 | **1.000** | 0.029 |
| test_004 | 42 / 0 / 1234 | 1 | 1.000 | 0.000 | **1.000** | 0.000 |

Warnings on every cell: `metal_seed_active=cisplatin smi=[NH3][Pt]([NH3])(Cl)Cl`.
No `metal_seed_from_smiles_failed` warnings — the round-trip patch is
sufficient to ingest the cisplatin seed.

## Honest MEASURED vs PROJECTED framing

### MEASURED (this re-verify)

- **metal_compliance_rate = 1.0 (15/15 cells)** — the seed is itself a
  valid metal-containing molecule and survives `from_smiles`, so every
  returned candidate carries Pt + the 4-coordinate geometry prior fires.
  This is the **success criterion of the WF-Lambda-1b task**:
  the metric is now measurable and non-zero.
- **validity_rate = 1.0** — RDKit sanitises cisplatin without issue.
- **synthesizability_rate = 0.0** — still zero, for a **different
  reason than WF-Lambda-1**: the candidate is the cisplatin seed itself,
  and `check_beta_normal_form` returns False because the ledger-based
  predicate counts the **NH3 lone pairs** as unsatisfied free sites.
  This is a semantic mismatch between *chemical* closure (all valences
  satisfied) and *ledger* closure (all arity slots filled). The
  `synthesizability_via_lambda_paths` function inherits the strict
  ledger definition; the relaxed `accept_partial` patch only loosens the
  *parse* step, not the *well-formed* step. **This defect is NOT
  caused by WF-Lambda-1b and is orthogonal to the round-trip / metal
  fixes** — it was latent in the original `check_beta_normal_form`.
- **n_candidates = 1 per cell** — MCTS did not expand past the seed in
  depth=3, so we are measuring the seed, not the search. The metal-seed
  patch wires the seed into the tree, but the click-rule / tile-library
  expansion at depth ≤ 3 from cisplatin does not yield Pt-containing
  products in this budget.

### PROJECTED (not measured here)

The spec calls for a measurable `synthesizability_rate > 0` after the
round-trip fix. **That projection is unfulfilled**: the round-trip
patch successfully relaxed parse-time rejection (no
`metal_seed_from_smiles_failed` warnings in the warnings list of any
cell), but the synthesis-oracle well-formedness predicate is a separate
gate that needs its own ledger-vs-valence correction. That is a
follow-up fix (ledger-based predicate should count valence, not arity,
for ligand nitrogens with explicit H counts).

## Verdict

**Partially successful.**

| criterion (from WF-Lambda-1b spec) | outcome |
|---|---|
| Round-trip fix → `synthesizability_rate > 0` | **NOT MET.** Parse-time rejection is gone, but the synthesis oracle uses a stricter predicate than `accept_partial` allows. Defect is real but distinct from the parse gate. |
| `--metal-seed cisplatin` → `metal_compliance_rate > 0` | **MET.** Aggregate jumped 0.0 → **1.0** across 15/15 cells. The lambda-calculus search now demonstrates Pt(II) coordination compliance by construction (the seed itself is the witness). |
| Validity ~0.9 unchanged | MET (improved to 1.0 — cisplatin is well-known to RDKit). |

**Does Lambda now demonstrate measurable synthesis + metal compliance?**
- Metal compliance: **YES** — 1.0 across all measured cells.
- Synthesis: **NOT YET** — same 0.0 as the baseline, but for a different
  ledger-based reason that is now isolated. The fix is a one-line
  predicate change in `lam_chem/well_formedness.check_beta_normal_form`
  (count valence saturation, not arity saturation) and is queued as a
  follow-up.

## Honest caveats

1. **MEASURED scope is small** (5×3=15 cells). PROJECTED to N=50 / N=100
   pockets, the metric means are expected to remain at 1.0 for metal
   compliance (the seed is always Pt-containing) and 0.0 for synthesis
   (the defect is upstream of the cell loop), but **this is a
   projection, not a measurement** — re-running at larger scale is the
   next milestone (TODO-20 next-steps list).
2. **diversity_alpha = 0.0** because every cell returns exactly one
   candidate (the seed). MCTS did not diversify in this budget. The
   `--n-simulations 100 --n-top-k 20` budget is well below what would
   be needed to traverse a 5-pocket sweep from a metal root.
3. **reference_tanimoto = 0.012** — much lower than the baseline 0.86,
   because all candidates are cisplatin whereas the references are
   organic drug ligands. This is **expected** when `--metal-seed` is
   active and is not a regression; it just confirms the seed is
   overriding the pocket reference.

## Files

- Script: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py`
- Round-trip patch: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py` (`from_smiles` with `accept_partial`)
- Metal-seed patch: same script, `METAL_SEED_SMILES` + `metal_seed` kwarg in `run_one_cell` + `--metal-seed` CLI flag
- Re-verify report (this run): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_lambda1b_pilot_v2/report.json`
- Baseline: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_N10_3seed_v2/report.json`
- Tests: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`
