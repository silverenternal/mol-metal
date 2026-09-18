# WF-R16-PB 15-Cell Smoke — Final Verdict (2026-09-18)

## 1. Spec & invocation

- 5 pockets × 3 seeds = **15 cells** (test_001..test_005 × {42, 0, 1234})
- `--n-simulations 1000` per cell (target budget; runs hit ~4-5s wall each)
- PB engine: `PoseBustersAdapter(mode='mol')` (`molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py:156`) — ETKDGv3 + MMFF94 → `pb.bust(full_report=True)` → 14 chemistry bool checks
- CLI used: `molmetal/scripts/r4_lambda_only_run.py` with `--pockets 1 --seeds S1 S2 S3` per pocket, `--manifest` pointing at a 1-row sliced manifest.
- **Note on CLI:** r4_lambda_only_run.py is **explicitly Lambda-only** — its argparse block (lines 4149-4788) registers no `--pb-check`, `--pb-mode`, or `--validate-docked` flag. PB therefore runs out-of-band via the canonical `PoseBustersAdapter` after the cell finishes. This matches the existing PB-pass-real-dock + wf-pb-pass-10x3-smoke split (PB is a post-hoc gate).

## 2. Per-cell results

| pocket | seed | n_total | n_elig | strict_pass | mean_elig | n_checks | failed |
|--------|------|---------|--------|-------------|-----------|----------|--------|
| test_001 | 42 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_001 | 0 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_001 | 1234 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_002 | 42 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_002 | 0 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_002 | 1234 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_003 | 42 | 1 | 1 | 0.00 | 0.77 | 13 | bond_lengths,bond_angles,internal_steric_clash |
| test_003 | 0 | 1 | 1 | 0.00 | 0.77 | 13 | bond_lengths,bond_angles,internal_steric_clash |
| test_003 | 1234 | 1 | 1 | 0.00 | 0.77 | 13 | bond_lengths,bond_angles,internal_steric_clash |
| test_004 | 42 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_004 | 0 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_004 | 1234 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_005 | 42 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_005 | 0 | 1 | 1 | 1.00 | 1.00 | 14 | - |
| test_005 | 1234 | 1 | 1 | 1.00 | 1.00 | 14 | - |

## 3. Aggregate

- n_cells = **15**
- n_candidates_total = **15**
- n_pb_eligible (3D embed OK) = **15**
- n_passed_all (all 14 chemistry checks) = **12**
- strict pass-rate (cell-mean) = **0.8000** (std 0.4000)
- strict pass-rate (aggregate n_pass/n_total) = **0.8000**
- mean eligible pass-rate (cell-mean of per-mol pass_rate) = **0.9538**

## 4. Per-pocket

| pocket | strict mean | strict std | n_pass / n_total |
|--------|-------------|------------|------------------|
| test_001 | 1.000 | 0.000 | 3/3 |
| test_002 | 1.000 | 0.000 | 3/3 |
| test_003 | 0.000 | 0.000 | 0/3 |
| test_004 | 1.000 | 0.000 | 3/3 |
| test_005 | 1.000 | 0.000 | 3/3 |

## 5. Verdict

- Gate: **strict pass-rate ≥ 0.6**
- Measured mean: **0.8000**
- Decision: **PASS**
- Promote §4.6 PB column DESIGN→MEASURED: **True**

Rationale: 12 of 15 (pocket, seed) cells produced a SMILES that passed ALL 14 PoseBusters chemistry checks (bond lengths, bond angles, internal steric clash, ring sizes, aromaticity, etc.). Only test_003 produced a molecule (`Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O`, an amino-anthraquinone sulfonate) that fails 3 checks (`bond_lengths`, `bond_angles`, `internal_steric_clash`) because the proximal sulfonate and aniline substituents clash after ETKDGv3+MMFF94.

## 6. Caveats / honest framing

1. **Singleton collapse**: every cell produced n_distinct=1 candidate. This is the documented WF-Round12-Lambda-Pilot / WF-Lambda-Internal-Review singleton-attractor behavior under default click-rules + scaffold-aware gate. PB denominator is therefore 15 (1 SMILES × 15 cells), not 300 (20 × 15). Per-pocket seed variance is artificially zero (same SMILES across all 3 seeds for each pocket).
2. **PB mode = 'mol'**: only 14 chemistry checks; the 12 protein-aware checks (clash vs receptor, distance-to-cofactors, etc.) require docked poses and are out of scope for this Lambda-only smoke. See wf-pb-pass-real-dock 2026-09-14 for the protein-aware path (GPU-blocked here).
3. **Honest scoring**: aggregate pass-rate is 0.80, but the test_003 sub-pocket rate is 0.00. A 95% CI on this small sample is wide; a 10×3 or 30×3 sweep with non-singleton cells is the next step (deferred; tracked separately).

## 7. Decision & next steps

- **PROMOTE §4.6 PB column DESIGN→MEASURED**: 15 cells × 1 mol each = 15 MEASURED PB rows.
- Recommended follow-up: 10×3 PB smoke on non-singleton cells (use `--metal-seed-from-pocket` or `--allow-incompatible-click` to break collapse).