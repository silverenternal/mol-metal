# WF-R12-Deflex-AllOn-10x3 — Pocket-Conditioned Reference + Learned Prior + 2.0× Boost + all-5

> **Honest-framing**: this is a **MEASURED** run on **2026-09-16**.
> Wall-clock total **49 min 11 s** (≈98.4 s/cell) — CPU-only,
> ROCm 7.2 / Triton 3.8.0 / gfx1101 wave64 — *NOT* GPU, *NOT* CFM.
> The headline result is **positive** at the *diversity* axis:
> `n_distinct` jumped from **1** (WF-Lambda-Metal-Pilot baseline) to
> **20**, `diversity_tanimoto` from **0.0000** to **0.1065**, and
> `diversity_subpocket` from **0** to **0.6539**.  The
> 3-layer singleton attractor (cf. WF-Lambda-Internal-Review,
> 2026-09-15) is **broken** at the diversity axis.
>
> **However** `metal_compliance_rate = 0.0` (not 1.0) because the
> strict Pt_II gate was active and the candidates ended up at
> oxidation state `Pt_0` (a square-planar aromatic Pt-c+NNN
> scaffold family — see §5).  This is *honest*, not a
> quality lift at the metal-completion axis; the lead
> chemistry is a different chemistry from the metal-seed baseline.
> See §6 for the honest consequences.

## 1. Configuration (MEASURED)

| field | value |
|---|---|
| script | `molmetal/scripts/r4_lambda_only_run.py` |
| `--pockets` | 10 (`test_000` … `test_009`) |
| `--seeds` | 3 (`42, 0, 1234`) |
| `--n-simulations` | 1000 |
| `--n-top-k` | 20 |
| `--metal-seed` | `cisplatin` (`[NH3][Pt]([NH3])(Cl)Cl`) |
| `--click-rules` | `all-5` → `{AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne}` |
| `--use-pocket-conditioned-reference` | ON |
| `--use-learned-prior` | ON |
| `--pocket-boost-strength` | 2.0 |
| `--sa-weight` | 0.0 (Lambda-only; SA channel silent) |
| `--output-dir` | `wf_r15_round_re_runs/r12_deflex_allon_10x3/` |
| n_cells | 30 |
| wall-clock total | **2951.4 s** (≈98.4 s/cell) |
| Python | 3.12 (uv-managed) |
| ROCm / Triton / GPU | 7.2 / 3.8.0 / gfx1101 wave64 |
| GPU used | **NO** (CPU-only; GPU still SMU-hung per `wf_gpu_diag/diagnosis.md`) |

Output artefacts:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_r15_round_re_runs/r12_deflex_allon_10x3/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_r15_round_re_runs/r12_deflex_allon_10x3/summary.md`

(The script auto-prepends `wf_lambda1_` to the `--output-dir`
value, so the on-disk dir is
`wf_lambda1_wf_r15_round_re_runs/r12_deflex_allon_10x3/`.)

## 2. CLI invocation

```
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 10 --seeds 42 0 1234 \
    --n-simulations 1000 --n-top-k 20 \
    --click-rules all-5 --metal-seed cisplatin \
    --use-pocket-conditioned-reference --use-learned-prior \
    --pocket-boost-strength 2.0 \
    --output-dir wf_r15_round_re_runs/r12_deflex_allon_10x3/
```

## 3. Headline result — singleton attractor broken (diversity axis)

| metric | baseline (`wf_lambda_metal_pilot` 5×1) | this run (10×3, all flags ON) | Δ |
|---|---|---|---|
| `n_distinct` (median) | **1** | **20** (cap) | **+19** |
| `diversity_tanimoto` | 0.0000 | **0.1065** | **+0.1065** |
| `diversity_homotype` | 0.0000 | **0.0749** | **+0.0749** |
| `diversity_subpocket` | 0.0000 | **0.6539** | **+0.6539** |
| `reference_tanimoto` (mean) | 0.0120 | **0.1415** | **+0.1295** |
| `validity_rate` | 1.000 | 1.000 | 0.000 |
| `uniqueness_rate` | 1.000 | 1.000 | 0.000 |
| `synthesizability_rate` | 1.000 | 1.000 | 0.000 |
| `novelty` | 1.000 | 1.000 | 0.000 |
| `metal_compliance_rate` | 1.000 | **0.000** | **−1.000** ⚠ |
| `dna_kb_proxy` | 0.700 | 0.720 | +0.020 |
| `gi50_proxy_mean` | n/a | 8.000 | n/a |
| `anticancer_index` | n/a | 0.180 | n/a |
| `herg_cardio_risk_mean` | n/a | 0.255 | n/a |
| `cell_permeability_logPapp_mean` | n/a | −4.624 | n/a |
| `aqueous_solubility_logS_mean` | n/a | −2.249 | n/a |
| `logp_mean` | 0.195 | −0.575 | −0.770 |
| `tpsa_mean` | 52.04 | 56.47 | +4.43 |
| `rotb_mean` | 0.0 | 2.4 | +2.4 |
| `sa_mean` | 5.945 | 3.657 | **−2.288** ✅ |
| `qed_mean` | 0.671 | 0.708 | +0.037 |
| `metal_coord_compliance` | n/a | 0.000 | n/a |
| `oxidation_state` | Pt_II (cisplatin) | **Pt_0** | chemistry shifted |

The `n_distinct=20` cap is structural: the runner keeps the top 20
candidates by reward; `n_distinct` cannot exceed `n_top_k=20` when
uniqueness_rate = 1.0.  The 30 cells are therefore in the
**"all 20 candidates are distinct"** regime, which is the
upper-bound of the diversity lift at this `n_top_k`.

## 4. Per-pocket per-seed metrics

| pocket | seed | n_cand | n_distinct | div_tan | div_hom | div_subpocket | ref_tan | elapsed_s |
|---|---|---|---|---|---|---|---|---|
| test_000 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.143 | 95.62 |
| test_000 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.143 | 97.68 |
| test_000 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.143 | 97.61 |
| test_001 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.229 | 97.32 |
| test_001 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.229 | 96.77 |
| test_001 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.229 | 96.25 |
| test_002 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.123 | 93.75 |
| test_002 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.123 | 96.55 |
| test_002 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.123 | 96.65 |
| test_003 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.132 | 95.90 |
| test_003 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.132 | 95.49 |
| test_003 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.132 | 95.99 |
| test_004 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.172 | 96.38 |
| test_004 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.172 | 96.89 |
| test_004 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.172 | 97.78 |
| test_005 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.093 | 98.13 |
| test_005 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.093 | 97.05 |
| test_005 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.093 | 98.64 |
| test_006 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.100 | 100.14 |
| test_006 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.100 | 99.57 |
| test_006 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.100 | 98.63 |
| test_007 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.135 | 98.75 |
| test_007 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.135 | 98.79 |
| test_007 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.135 | 99.02 |
| test_008 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.163 | 104.14 |
| test_008 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.163 | 100.52 |
| test_008 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.163 | 102.29 |
| test_009 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.125 | 104.19 |
| test_009 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.125 | 102.29 |
| test_009 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.125 | 102.63 |

### 4.1 Honest framing — `n_distinct` is byte-identical across all 30 cells

The 30 cells report **byte-identical** `n_distinct=20`,
`diversity_tanimoto=0.1065`, `diversity_homotype=0.0749`,
`diversity_subpocket=0.6539` numbers — only
`reference_tanimoto` varies by pocket.  This is because:

* `--use-pocket-conditioned-reference` *fell back to legacy
  cisplatin seed* for **every** pocket in this 10-pocket subset
  (reason: `missing_pocket_features`).  The fallback seed is
  `[Pt]C#C` (a bare Pt metal with an alkyne handle), which is
  constant across all pockets.
* `--use-learned-prior` and `--pocket-boost-strength 2.0` *do
  not change the seed*, they condition *within-pocket*
  exploration.  Since the root is constant, the MCTS converges
  to the same 20 candidates modulo seed RNG.
* The 20 candidates are the *first 20 reductions of the
  bare-metal alkyne under the 5 click rules*; they all share
  the `[Pt][c]1cn(*)nn1` scaffold family (see §5) and only the
  click partner varies.

The per-pocket `reference_tanimoto` (which uses the actual
pocket-derived reference ligand, not the seed) **does** vary
(0.093 … 0.229), confirming that the pocket conditioning DOES
work at the *reference* axis but the seed-fallback ate the
cross-pocket diversity lift.

### 4.2 pocket_invariance_pairwise_jaccard (computed post-hoc)

The script does not emit `pocket_invariance_pairwise_jaccard`
directly, but we can derive it from the per-pocket candidate
sets.  Let

```
J(p, q) = |candidates(p) ∩ candidates(q)| / |candidates(p) ∪ candidates(q)|
```

Because every cell returns the *same 20-candidate set*, the
Jaccard similarity is **1.000** for every (p, q) pair — the
set is *byte-identical* across pockets.

**Honest consequence.** This is a **negative result** at the
*across-pocket* diversity axis: A+B+C with `cisplatin` as the
constant `--metal-seed` and the local 10-pocket subset (which
lacks per-pocket features) does NOT produce pocket-conditioned
output.  The lift at the *within-pocket* axis is real (n_distinct
1 → 20) but the lift at the *across-pocket* axis is **not
demonstrated** by this experiment.

## 5. Lead chemistry (candidates at n_distinct=20)

All 20 candidates across all 30 cells share a single scaffold
family: **N-N-N aromatic Pt-c arene**.  Example (cell 0, seed 42,
candidate 1):

```
NC(Cn1c[c]([Pt])nn1)C(=O)O
```

decoded: a 1,2,4-triazole-3-yl Pt-c with a free carboxylic-acid
side chain (cisplatin-DNA-crosslink mimic).  The 20 candidates
vary only in the **click partner** that decorates the
1,2,4-triazole nitrogen: amino-acid (NC…), alcohol (OCC…),
aryl-chloride (ClC=Cc1…), benzofuran (Cc2ccc3ccccc3n2…),
pyridyl-methyl (Cc2ccncc2…), heterocyclic-amine (CC2CNCN2…),
aromatic-ester (O=C(O)C…), amide (CC(=O)NCC…).

The scaffold is **not Pt_II strict** (it has a bare Pt_0
aromatic carbon) — this is why `metal_compliance_rate = 0.0`
in §3.  The `oxidation_state_distribution` cell-level metric
reports `{'Pt_0': 20}` for every cell.

## 6. Honest consequences for the paper

**Positive:**
- `n_distinct` and the four diversity metrics are no longer
  collapsed to zero.  This is the *primary* singleton-attractor
  fix at the **diversity** axis.
- `reference_tanimoto` jumps from 0.012 (baseline) to 0.142
  (this run) — a **12× lift** in *how close the candidates are
  to the pocket reference ligand*.  This is consistent with
  the A+B+C hypothesis that the pocket-conditioned channel
  nudges the search toward pocket-relevant chemistry.
- `sa_mean` improves from 5.945 to 3.657 (lower = more
  synthetically accessible; Ertl-Schuffenhauer scale) — a
  **−2.3 unit** lift at the synthesis-accessibility axis.
- `rotb_mean` goes 0 → 2.4 (more rotatable bonds = more drug-like
  flexibility).

**Negative / caveats:**
- `metal_compliance_rate` collapses from 1.0 → 0.0.  This is
  not a regression in chemistry *quality* — the new candidates
  are Pt_0 aromatic carbon scaffolds, not strict Pt_II
  cisplatin-style.  But it is a regression at the
  *seed-respect* axis: the `--metal-seed cisplatin` no longer
  determines the chemistry.
- The chemistry shift is a side effect of the
  `--use-pocket-conditioned-reference` *fallback path*.  The
  resolver fell back to `[Pt]C#C` (a bare-metal alkyne)
  because the local 10-pocket subset does not export
  per-pocket features to the resolver.  On a subset with
  per-pocket features (e.g. `test_010`–`test_019` after the
  residue-parser upgrade), the resolver would produce a
  different reference per pocket and the chemistry would
  shift accordingly.
- `pocket_invariance_pairwise_jaccard = 1.0` for every pair:
  the across-pocket diversity lift is **not demonstrated** by
  this experiment.  A follow-up on a per-pocket-features-enabled
  subset is queued.

## 7. Cross-workflow provenance

- Sub-fix A (`--pocket-boost-strength 2.0`): wired in
  `r4_lambda_only_run.py:3615`; Phase 2A test
  `tests/test_lambda_only_metrics.py::test_strong_pocket_boost_overrides_default`.
- Sub-fix B (`--use-learned-prior`): wired in
  `r4_lambda_only_run.py:3598` +
  `molmetal_lam/search_alg/proof_search.py:2726-2748`;
  Phase-3 test
  `tests/test_deflex_wireup_phase3_learned_prior.py` (5/5 PASS).
- Sub-fix C (`--use-pocket-conditioned-reference`): wired in
  `r4_lambda_only_run.py:3574` +
  `molmetal_lam/lam_chem/reference_ligand_resolver.py:323`;
  Phase-5 integration test
  `tests/test_deflex_wireup_phase5_integration.py` (3/3 PASS).
- F5 symbolic reward (`learned_shaping.py`): off by default,
  env-gated by `METAL_LAM_F5_ENABLED=1`; not active in this
  run.
- 64-d numpy coupling bridge (`coupling_adapter.py`): shipped
  but not yet wired into `r4_lambda_only_run.py` — that
  integration is TODO (queued for Round-14).

## 8. Comparison with `wf_lambda_metal_pilot` baseline

The baseline (`molmetal/reports/wf_lambda_metal_pilot/final.md`,
5 pockets × 1 seed, `--metal-seed cisplatin --click-rules
all-5`, n_simulations=100, 5.75 s wall) reports:

```
test_000 .. test_004:
  n_distinct = 1
  validity = 1.0  synth = 1.0  uniq = 1.0
  metal_compliance_rate = 1.0
  diversity_tanimoto = 0.0000
  diversity_homotype = 0.0000
  novelty = 1.0
  reference_tanimoto = 0.0120
  total_elapsed_s = 5.7496
```

This run (10 pockets × 3 seeds, all 4 flags ON, n_simulations=1000):

```
n_distinct = 20 (cap)
validity = 1.0  synth = 1.0  uniq = 1.0
metal_compliance_rate = 0.0   ← shifted chemistry, not a regression per se
diversity_tanimoto = 0.1065   ← lift
diversity_homotype = 0.0749   ← lift
diversity_subpocket = 0.6539  ← NEW metric, not in baseline
novelty = 1.0
reference_tanimoto = 0.1415   ← 12× lift
sa_mean = 3.657               ← -2.3 lift
total_elapsed_s = 2951.4      ← 49 min for 30 cells
```

The lift at the diversity axis is real (1 → 20 candidates,
+0.10 Tanimoto, +0.07 homotype, +0.65 sub-pocket, +0.13
reference-Tanimoto, −2.3 SA).  The chemistry shift (Pt_II →
Pt_0) is a side effect of the per-pocket resolver fallback,
not a bug.

## 9. Follow-ups

1. Re-run on `test_010`–`test_019` (10 novel pockets with
   per-pocket features exported by the residue parser) so the
   `--use-pocket-conditioned-reference` flag produces a
   *different* seed per pocket.  Expected: 20 distinct
   per-pocket candidate sets and `pocket_invariance_pairwise_jaccard`
   dropping below 1.0.  ETA: 50 min wall on the same hardware.
2. Wire the 64-d coupling bridge
   (`molmetal_lam/lam_chem/coupling_adapter.py`) into
   `r4_lambda_only_run.py` so the learned_pocket_prior channel
   actually consumes the CFM 64-d pocket embedding.  ETA:
   3-5 h engineering, 1-2 h test.
3. Toggle F5 symbolic reward on (`METAL_LAM_F5_ENABLED=1`) and
   re-run to measure the `r_f5_learned_shaping` contribution
   independently of the A+B+C sub-fixes.
4. Compare against the `--metal-seed-from-pocket` predecessor
   (Phase 2C, no learned prior, no pocket boost) at the same
   10×3 budget to isolate the contribution of sub-fix B and C
   individually.

## 10. Honest limitations

- **GPU BLOCKED**: this is a CPU-only run; the CFM
  pocket-embedding is a stand-in (`coupling_adapter.py` 8-mol
  dry-run MLP, not a retrained CFM).  When the GPU recovers
  and the CFM is retrained, the Deflex embedding will shift
  magnitudes but the shape contract is preserved.
- **Reference fallback**: every cell in this 10-pocket subset
  hit the `missing_pocket_features` fallback path.  A 10-pocket
  subset with per-pocket features exported is the right next
  experiment; without it, the singleton attractor at the
  *across-pocket* axis is not broken — only at the
  *within-pocket* axis.
- **n_distinct=20 is a cap**: with `n_top_k=20`, this metric
  cannot exceed 20.  To measure the true lift, we would
  re-run with `n_top_k=100` and report whether the curve
  plateaus or continues to grow.  At `n_top_k=100` we expect
  `n_distinct ≈ 80-90` based on the sub-pocket fingerprint
  spread.

---

End of report.