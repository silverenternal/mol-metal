# WF-Lift-EV1 — Across-Pocket Diversity on test_010..019 (sub-fix A+B+C re-verify)

> **Verdict: NEGATIVE-result reproduction (MEASURED 2026-09-17, CPU-only, 30 cells).**
> **Across-pocket Jaccard = 1.000 ± 0.000** across all 45 (p, q) pocket pairs.
> Identical to the test_000..009 baseline (`wf_r15_round_re_runs/r12_deflex_allon_10x3/final.md` §4.2):
> the singleton attractor is *not broken* on test_010..019 either, because the
> `--use-pocket-conditioned-reference` flag falls back to the same legacy
> `[Pt]C#C` metal-alkyne handle for *every* pocket.  Root cause:
> `molmetal_lam/lam_chem/reference_ligand_resolver.py` returns
> `missing_pocket_features` because the local CrossDocked2020 manifest does
> **not** export per-pocket residue features, so the discriminator receives
> the zero vector and the fallback bucket fires.

## 1. Configuration (MEASURED)

| field | value |
|---|---|
| script | `molmetal/scripts/r4_lambda_only_run.py` |
| `--pockets` | 10 (`test_010` … `test_019`) |
| `--seeds` | 3 (`42, 0, 1234`) |
| `--n-simulations` | 1000 |
| `--n-top-k` | 20 |
| `--metal-seed` | `cisplatin` (`[NH3][Pt]([NH3])(Cl)Cl`) |
| `--click-rules` | `all-5` → `{AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne}` |
| `--use-pocket-conditioned-reference` | ON |
| `--use-learned-prior` | ON |
| `--pocket-boost-strength` | 2.0 |
| `--sa-weight` | 0.0 (Lambda-only) |
| `--manifest` | `molmetal/data/test_010_019_subset_manifest.csv` (10 rows) |
| `--output-dir` | `wf_lift_ev1/` (script auto-prepends `wf_lambda1_`) |
| n_cells | 30 |
| wall-clock total | **5185.5 s ≈ 86 min 26 s** (≈172 s/cell) |
| Python | 3.12 (uv-managed) |
| GPU used | **NO** (CPU-only; GPU still SMU-hung per `wf_gpu_diag/diagnosis.md`) |

Output artefacts:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_lift_ev1/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_lift_ev1/summary.md`

## 2. CLI invocation

```
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 10 --seeds 42 0 1234 \
    --n-simulations 1000 --n-top-k 20 \
    --click-rules all-5 --metal-seed cisplatin \
    --use-pocket-conditioned-reference --use-learned-prior \
    --pocket-boost-strength 2.0 \
    --manifest molmetal/data/test_010_019_subset_manifest.csv \
    --output-dir wf_lift_ev1/
```

A custom 10-row subset manifest was created at
`molmetal/data/test_010_019_subset_manifest.csv` (rows 11–20 of
`crossdocked100_manifest.csv` filtered to `test_010`…`test_019`).
The `--pockets` integer counts the rows; the manifest is the row selector.

## 3. Headline result — across-pocket Jaccard unchanged (NEGATIVE)

| metric | baseline (`r12_deflex_allon_10x3` on test_000..009) | this run (test_010..019) | Δ |
|---|---|---|---|
| `pocket_invariance_pairwise_jaccard` (mean) | **1.000** | **1.000** | **0.000** ⚠ |
| `pocket_invariance_pairwise_jaccard` (std) | 0.000 | 0.000 | 0.000 |
| `n_distinct` (median) | 20 (cap) | 20 (cap) | 0 |
| `diversity_tanimoto` | 0.1065 | 0.1065 | 0.0000 |
| `diversity_homotype` | 0.0749 | 0.0749 | 0.0000 |
| `diversity_subpocket` | 0.6539 | 0.6539 | 0.0000 |
| `reference_tanimoto` (mean) | 0.1415 | 0.1511 | +0.0096 |
| `validity_rate` | 1.000 | 1.000 | 0.000 |
| `uniqueness_rate` | 1.000 | 1.000 | 0.000 |
| `synthesizability_rate` | 1.000 | 1.000 | 0.000 |
| `novelty` | 1.000 | 1.000 | 0.000 |
| `metal_compliance_rate` | 0.000 | 0.000 | 0.000 ⚠ |
| `sa_mean` | 3.657 | 3.657 | 0.000 |
| `qed_mean` | 0.708 | 0.708 | 0.000 |
| `logp_mean` | -0.575 | -0.575 | 0.000 |
| `tpsa_mean` | 56.47 | 56.47 | 0.000 |
| `rotb_mean` | 2.4 | 2.4 | 0.000 |

**All 30 cells are byte-identical to the test_000..009 cells** — same 20
SMILES, same per-metric numbers, except `reference_tanimoto` which varies
modestly per pocket (0.085–0.256) because the *pocket-derived reference
ligand* differs even though the *MCTS root* is constant.

## 4. Per-pocket per-seed metrics

| pocket | seed | n_cand | n_distinct | div_tan | div_hom | div_subpocket | ref_tan | elapsed_s |
|---|---|---|---|---|---|---|---|---|
| test_010 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.085 | 100.4 |
| test_010 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.085 | 99.2 |
| test_010 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.085 | 99.6 |
| test_011 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.117 | 99.4 |
| test_011 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.117 | 98.2 |
| test_011 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.117 | 99.0 |
| test_012 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.103 | 99.6 |
| test_012 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.103 | 100.4 |
| test_012 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.103 | 99.6 |
| test_013 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.137 | 172.7 |
| test_013 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.137 | 167.0 |
| test_013 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.137 | 168.0 |
| test_014 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.256 | 226.4 |
| test_014 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.256 | 240.1 |
| test_014 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.256 | 240.1 |
| test_015 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.256 | 178.4 |
| test_015 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.256 | 188.0 |
| test_015 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.256 | 189.0 |
| test_016 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.100 | 196.4 |
| test_016 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.100 | 199.0 |
| test_016 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.100 | 202.2 |
| test_017 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.132 | 114.0 |
| test_017 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.132 | 115.0 |
| test_017 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.132 | 117.0 |
| test_018 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.149 | 184.0 |
| test_018 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.149 | 184.0 |
| test_018 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.149 | 187.0 |
| test_019 | 42 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.175 | 191.7 |
| test_019 | 0 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.175 | 190.0 |
| test_019 | 1234 | 20 | 20 | 0.1065 | 0.0749 | 0.6539 | 0.175 | 190.0 |

Wall-clock per cell varies because of crossdock pocket-size effects on the
3D-conformer attachment step (`n_coords_3d_attached_total = 600` = 30×20
constant).  The metric columns are byte-identical across all 30 rows.

## 5. Pocket-pairwise Jaccard table (45 pairs)

For each (p, q) pocket pair, `J(p, q) = |candidates(p) ∩ candidates(q)| / |candidates(p) ∪ candidates(q)|` where candidates are the top-20 deduplicated SMILES returned by `--n-top-k 20`.

| (p, q) | J | (p, q) | J | (p, q) | J |
|---|---|---|---|---|---|
| (010, 011) | 1.0000 | (011, 014) | 1.0000 | (013, 016) | 1.0000 |
| (010, 012) | 1.0000 | (011, 015) | 1.0000 | (013, 017) | 1.0000 |
| (010, 013) | 1.0000 | (011, 016) | 1.0000 | (013, 018) | 1.0000 |
| (010, 014) | 1.0000 | (011, 017) | 1.0000 | (013, 019) | 1.0000 |
| (010, 015) | 1.0000 | (011, 018) | 1.0000 | (014, 015) | 1.0000 |
| (010, 016) | 1.0000 | (011, 019) | 1.0000 | (014, 016) | 1.0000 |
| (010, 017) | 1.0000 | (012, 013) | 1.0000 | (014, 017) | 1.0000 |
| (010, 018) | 1.0000 | (012, 014) | 1.0000 | (014, 018) | 1.0000 |
| (010, 019) | 1.0000 | (012, 015) | 1.0000 | (014, 019) | 1.0000 |
| (011, 012) | 1.0000 | (012, 016) | 1.0000 | (015, 016) | 1.0000 |
| (011, 013) | 1.0000 | (012, 017) | 1.0000 | (015, 017) | 1.0000 |
| (012, 018) | 1.0000 | (012, 019) | 1.0000 | (015, 018) | 1.0000 |
| (013, 014) | 1.0000 | (012, 018) | 1.0000 | (015, 019) | 1.0000 |
| (013, 015) | 1.0000 | (013, 016) | 1.0000 | (016, 017) | 1.0000 |
| (016, 018) | 1.0000 | (016, 019) | 1.0000 | (017, 018) | 1.0000 |
| (017, 019) | 1.0000 | (018, 019) | 1.0000 | — | — |

**mean = 1.0000, std = 0.0000, median = 1.0000, min = 1.0000, max = 1.0000.**

The full machine-readable 45-pair table is at
`molmetal/reports/wf_lift_ev1/jaccard_table.json`.

## 6. Comparison with baseline (test_000..009)

| metric | `r12_deflex_allon_10x3` (test_000..009) | this run (test_010..019) | Δ |
|---|---|---|---|
| pocket-pair Jaccard mean | 1.000 | 1.000 | 0.000 |
| pocket-pair Jaccard std | 0.000 | 0.000 | 0.000 |
| ref_tanimoto mean | 0.1415 | 0.1511 | +0.0096 |
| diversity_tanimoto | 0.1065 | 0.1065 | 0.0000 |
| diversity_subpocket | 0.6539 | 0.6539 | 0.0000 |
| metal_compliance_rate | 0.000 | 0.000 | 0.000 |
| sa_mean | 3.657 | 3.657 | 0.000 |
| qed_mean | 0.708 | 0.708 | 0.000 |
| wall-clock | 49 min (98 s/cell) | 86 min (172 s/cell) | +37 min |

The `+37 min` wall-clock difference is **not** a chemistry / metric change; it
reflects the larger crossdock pocket files for test_010..019 (different
extracted subdirs) increasing 3D-conformer attachment cost.  Per-cell metric
behaviour is byte-identical to the baseline.

## 7. Root-cause analysis (honest)

The 3-layer singleton attractor identified in
`wf_lambda_internal_review/audit.md` is fully intact on this 10-pocket
subset:

1. **Chemistry layer (click SMARTS)** — every cell starts the MCTS from the
   same `[Pt]C#C` bare-metal alkyne handle (returned by
   `_build_fallback_record` in `reference_ligand_resolver.py:323` when
   `pocket_features` is None), then explores 5 click reductions on the same
   Pt_0 scaffold family.  This produces the same 20-candidate set every
   time.
2. **MCTS cache layer** — `_unreactive_states` is permanent per
   (pocket_id, seed) but with the same root, the cache hits the same
   dead-end set in the same iteration order.
3. **Reward prior layer** — `metal_geometry_prior_bonus` is hard-gated; with
   the fallback root, it collapses to the same aromatic-Pt-c family.

The expected fix path (per `wf_lambda_internal_review/diagnose.md` F2a,
"MetalLigandExchange SMARTS rule") is **not** enabled by the CLI flags
chosen here — that would require a separate follow-up that ships a
metal-conditional click partition before the reference fallback fires.

**`reference_tanimoto` is the only metric that varies by pocket** (0.085
for test_010 vs 0.256 for test_014/015) because the pocket-derived
reference ligand differs even when the MCTS root does not.  This is a
**reference-axis lift only**; the *generator* output is constant.

## 8. Honest limitations

- **No per-pocket features loaded.** The `crossdocked100_manifest.csv`
  pocket rows lack the `pocket_residues` column that
  `pocket_features_from_binding_site` would consume.  The 10-row subset
  manifest inherits the same limitation.  A follow-up would extend the
  manifest schema to carry residue names, atom counts, and
  hydrophobic/hydrophilic pocket centroids — that is the *real* fix for
  the across-pocket diversity axis.
- **`--use-pocket-conditioned-reference` did not actually condition.** All
  30 cells hit the `missing_pocket_features` fallback path (verbatim
  warning in `run.log`).  A user who reads the CLI help text would expect
  a different seed per pocket; the current implementation silently
  returns the same `[Pt]C#C` fallback.  **Documentation gap**: the
  `--use-pocket-conditioned-reference` help string does not warn that
  the local CrossDocked2020 manifest does not carry pocket features.
- **`metal_compliance_rate = 0.000`** is a chemistry shift (Pt_0 aromatic
  carbon scaffold) caused by the same root, not a regression in chemistry
  *quality* per se.  The lead candidates are reproducible and reasonable
  triazole scaffolds; they are not Pt_II strict.
- **`pocket_invariance_pairwise_jaccard = 1.000`** is the *across-pocket*
  metric that was supposed to lift with sub-fix C.  The expected lift
  (`1.000 → 0.5–0.8` from the EV-1 spec) is **not demonstrated** on
  test_010..019 with the current data + flags.

## 9. Follow-ups (prioritized)

1. **Extend the manifest schema to carry per-pocket residue features.**
   Adds 5 columns (`pocket_residues`, `n_pocket_residues`, `n_hydrophobic`,
   `n_polar`, `centroid_x/y/z`) to `crossdocked100_manifest.csv`.  Re-run
   the same 10×3 sweep on test_010..019.  Expected Jaccard lift
   `1.000 → 0.4–0.7`.  ETA: 2h engineering (residue parser already
   exists in `warm_start.py`), 1.5h test.
2. **Document the `--use-pocket-conditioned-reference` data dependency**
   in the CLI help text.  Add a `--pocket-features-source` flag that
   accepts `manifest` (current), `sdf-infer` (parse ligand SDF for residue
   hints), or `external-csv` (user-supplied).  ETA: 1h engineering.
3. **Run the same 10×3 on test_020..029** (the next 10 crossdocked
   pockets) as a third control.  If Jaccard=1.000 there too, the
   conclusion generalises: PCR requires manifest data that the current
   CrossDocked2020 extract does not carry.  ETA: 90 min CPU.
4. **Wire F2(a) MetalLigandExchange into the click pool** so that even
   the fallback `[Pt]C#C` root can produce diverse click products
   independent of the reference ligand.  ETA: 4h engineering (already
   shipped in `pt_metal_ligand_exchange.py` per
   `wf_lambda_fix_full_path_v2`); 1h test.

## 10. Cross-workflow provenance

- Sub-fix A (`--pocket-boost-strength 2.0`): wired in
  `r4_lambda_only_run.py:3615`; tests in
  `tests/test_lambda_only_metrics.py::test_strong_pocket_boost_overrides_default`.
- Sub-fix B (`--use-learned-prior`): wired in
  `r4_lambda_only_run.py:3598` + `proof_search.py:2726-2748`; tests in
  `tests/test_deflex_wireup_phase3_learned_prior.py` (5/5 PASS).
- Sub-fix C (`--use-pocket-conditioned-reference`): wired in
  `r4_lambda_only_run.py:3574` + `reference_ligand_resolver.py:323`; tests
  in `tests/test_deflex_wireup_phase5_integration.py` (3/3 PASS).
- All 3 sub-fixes **are** active; the across-pocket Jaccard lift is
  blocked at the *data layer* (no pocket features in the manifest), not
  at the *code layer* (all 3 sub-fixes run).

## 11. Honest verdict

**EV-1 (across-pocket diversity) — `WEAK` status UNCHANGED on
test_010..019.**  The 3 sub-fixes A+B+C are wired and active but require
a per-pocket-feature data layer that the current
`crossdocked100_manifest.csv` does not provide.  The 30 cells produced
in this 86 min CPU run are byte-identical to the test_000..009 baseline
at the candidate-set level; the only across-pocket lift is in
`reference_tanimoto` (0.085 → 0.256 range), which is a reference-axis
metric and not a generator-output diversity metric.

**Net M1-T23 (Weak → Strong) status update:** EV-1 is *necessary but
not sufficient*; the across-pocket axis still requires (a) manifest
schema extension, (b) F2(a) MetalLigandExchange rule wire, and (c) a
re-run on a per-pocket-features-enabled subset.  Estimated combined
ETA: 5-6h engineering + 2h CPU.

---

End of report.
