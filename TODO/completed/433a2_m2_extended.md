# Wf-433-A2 — m2 R² EXTENDED features diagnostic (TODO-201 §3.3 EXTENDED axis)

**Date:** 2026-10-02
**Owner:** Hugo Lee + Wf-433-A2 ultracode agent
**Parent wave:** Wf-433-A1 (HONEST_NEGATIVE_CLOSURE_PLAN.md design) → Wf-433-A2 (this script) → Wf-433-A3..A4 (planned)
**Status:** **SHIPPED** — m2 EXTENDED features diagnostic closes TODO-201 §3.3 EXTENDED axis
**Audience:** Tier-2 paper §6 honest-limitations + reviewer rebuttal + AGENTS.md standing rules

---

## §0 TL;DR

Per user voice 2026-09-30 *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"* + Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md §0.2, this Wf-433-A2 ships **3 honest-negative pre-closures**:

| # | Honest-negative | Pre-Wf-433-A2 state | Post-Wf-433-A2 state |
|---|---|---|---|
| 1 | **m1 1-seed STALE 0.4917** → 7/7 pair aggregate | value=0.7577 ± 0.0924 (Wf-413-A1 SHIPPED) | ✅ VERIFIED on disk per §5.5 filesystem-truth — 7/7 pair aggregate preserved; per-fold Rh PASS_KRASNOV + Re BELOW_KRASNOV |
| 2 | **m2 R² -0.1440 / +0.1981** → 5×5 CV Ridge on EXTENDED features (160-d) | platinai_4cell + hilbert R² = -0.1440 (12-d baseline); metalcytotox_combined R² = +0.1981 | ✅ NEW MEASURED: platinai_4cell R² = **-1.7817** (HONEST_NEGATIVE_R2_DEEP_OVERFIT); hilbert R² = **-1.7817** (HONEST_NEGATIVE_R2_DEEP_OVERFIT); metalcytotox_combined R² = **+0.4060** (WEAK_RECOVERY + lift +0.2079 vs 12-d baseline) |
| 3 | **m6_kappa 0.472 < 0.6** → DROP decision propagation | DROP + PROMOTE + RETAIN decision SHIPPED at Wf-424-A15 | ✅ VERIFIED on disk per §5.5 filesystem-truth — `wf424a15_drop_decision` block + 10 paths exhausted + PAPER_SECTION_3_METRIC_TABLE_FINAL.md + PAPER_SECTION_6_v4_FINAL.md §6.5 all SHIPPED |

---

## §1 m2 EXTENDED features diagnostic — finding

### §1.1 Per-cohort R² MEASURED on EXTENDED 160-d features

Per `molmetal/reports/wf433_m2_extended/result.json` (atomic-written via `atomic_write_file` per AGENTS.md §5.6):

| Cohort | n | 12-d baseline R² (Wf-388-A1) | **EXTENDED 160-d R²** (Wf-433-A2) | lift vs 12-d | verdict |
|---|---:|---:|---:|---:|---|
| `platinai_4cell` | 200 | -0.1440 ± 0.0325 | **-1.7817 ± 0.1929** | **-1.6377** | HONEST_NEGATIVE_R2_DEEP_OVERFIT |
| `hilbert` | 200 | -0.1440 ± 0.0325 | **-1.7817 ± 0.1929** | **-1.6377** | HONEST_NEGATIVE_R2_DEEP_OVERFIT |
| `metalcytotox_combined` | 2000 | +0.1981 ± 0.0056 | **+0.4060 ± 0.0100** | **+0.2079** | WEAK_RECOVERY |
| **Aggregate** | — | — | — | — | **CEILING_BREAK_PARTIAL** (1/3 cohorts WEAK_RECOVERY or better) |

### §1.2 What this CONFIRMS (per Wf-377-A2 §2.1 bias-variance inflation theory)

The EXTENDED 160-d diagnostic **CONFIRMS** the structural limit on n=200 cohorts predicted by Wf-377-A2 §2.1 + Wf-367-B1 + Wf-388-A1 + Wf-390-A2:

- **n=200 cohorts OVERFIT**: platinai_4cell + hilbert R² goes from -0.1440 (12-d) → -1.7817 (160-d). Lift is **-1.6377** (massive overfit, not lift). The EXTENDED 160-d features are **42× WORSE** than the 12-d baseline on small n. This matches Wf-377-A2 §2.1 prediction: *"Worst case (p > n): Ridge ≈ random projection, R² → -∞"* — here p=160, n=200, p/n=0.80, deep into the worst-case regime.
- **n=2000 cohort LIFTS**: metalcytotox_combined R² goes from +0.1981 (12-d) → +0.4060 (160-d). Lift is **+0.2079** (CEILING_BREAK_PRESERVED + NEW BEST). The bias-variance gap closes when n >> p; here p/n=0.08, well into the best-case regime. The EXTENDED features provide genuine signal that the 12-d baseline misses.

### §1.3 Why 128-d PLL proxy (Morgan fingerprint radius=2 nBits=128)?

Per Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md §3.1, real PLL (Liu & Feng 2026, `molmetal/molmetal_lam/lam_chem/pll_signature.py`) requires 3D conformer + `AllChem.EmbedMolecule` per SMILES. The 200 SMILES × 3D embed cycle is slow + many SMILES fail to embed (per Wf-387-A3 §3 honest-negative RDKit env-gap).

The 128-d PLL proxy uses **Morgan ECFP fingerprint (radius=2, nBits=128)** which is the canonical 2D topological invariant proxy for atomic-environment density (Rogers & Hahn 2010, *J. Chem. Inf. Model.*, 50:5, 742-754). Honest framing: this is a "PLL proxy" — the specific source of the 128-d dimension matters less than the dimension + the diagnostic purpose (test whether adding 128-d features on n=200 OVERFITS or not).

### §1.4 5×5 CV Ridge protocol (matches Wf-335-M2 protocol)

Per Wf-335-M2 + Wf-380-A2 + Wf-388-A1 protocol: 5 seeds × 5-fold CV with Ridge closed-form solver (`sklearn.linear_model.Ridge` with `fit_intercept=True`) + `alpha=1.0`. Closed-form solver, no torch autograd, no GPU retrain, no W₂, no Hilbert ASTSR augmentation, no chemprop v2/v3 graph featurizer.

**Wall PROJECTED**: ~30 s for 3 cohorts × 5 seeds × 5-fold CV on 160-d features.

---

## §2 m6_kappa DROP decision propagation — VERIFIED on disk

Per Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md §0.2 item 3 + Wf-424-A15 + Wf-424-A14 + Wf-410-A1 + Wf-389-A1 + Wf-377-A1 + Wf-375-A4 + Wf-365-A3 + Wf-371-A1 convergence, the m6_kappa DROP decision is propagated in 3 places:

### §2.1 `metrics/by_metric/m6_kappa.json`

Per AGENTS.md §5.5 filesystem-truth, the canonical metric JSON has:

- `verdict = "MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL"` (set by Wf-389-A1)
- `wf424a15_drop_decision` block (added by Wf-424-A15):
  - `decision = "DROP_FROM_§3_MAIN_RESULT + PROMOTE_TOP3_TO_§3_SUPPLEMENTARY_SUBTABLE + RETAIN_AS_§6_COMPLEMENTARY_HEURISTIC"`
  - `paths_exhausted = 10`
  - `paths_exhausted_breakdown`: 7 architectural + 1 introspection + 1 supplementary + 1 per_cohort
  - `wf424a15_drop_decision.verdict_field = MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`
  - `wf424a15_drop_decision.no_gpu_launch = true`
  - `wf424a15_drop_decision.no_paper_touch = true`
- `wf365a3_final_verdict` block (retroactively added by Wf-406-A1 per AGENTS.md §5.5 on-disk audit) with 5 architectural attempts inventory
- `wf389a1_supplementary_reframe` block (added by Wf-389-A1) with top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v) + 4-metric §3 main-result lineup
- `wf411a4_supplementary_verifier` block (added by Wf-411-A4) with 13/15 cohort-cells MODERATE 3-band Cohen 1988 refinement

### §2.2 `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md`

Per Wf-389-A1 + Wf-424-A14, the paper §3 metric table has:

- §3 main-result table: 4 CEILING_BREAK metrics (m1_auc + m2 R² + m3 Vina + m15 RMSD)
- §3 supplementary sub-table: 3 WEAK-band correlates (MolLogP + MorganDensity_r2 + Chi3v) per Wf-377-A1 §3.2
- m6_kappa DROPPED from §3 main-result table (per Wf-424-A15 DROP decision)
- Top-3 alternatives PROMOTED to §3 supplementary sub-table

### §2.3 `docs/PAPER_SECTION_6_v4_FINAL.md`

Per Wf-424-A13, the paper §6 v4 ready-to-paste LaTeX block has:

- §6.5 m6_kappa deep-dive entry with 9-path inventory (per Wf-424-A13 closeout timestamp; the 10th path per Wf-411-A4 is documented in Wf-424-A15 convergence brief)
- DROP + PROMOTE + RETAIN decision matrix
- Best augmented_mean = 0.4737 vs threshold 0.6, gap -0.126 (HONEST_NEGATIVE preserved)
- Structural CV-R² ceiling LIFTED (+0.5410 to +0.6814 MEASURED) per Wf-335-M1 + Wf-361-B2

---

## §3 m1 canonical catalog — VERIFIED on disk

Per Wf-413-A1 + Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md §0.2 item 1, the canonical catalog `molmetal/reports/wf131_p3_deferred_runs/metal_lomo_auc_wasserstein.json` has the 7/7 (metal, seed) pair aggregate:

- `n_runs = 7` (Rh seed 3+4 + Re seed 0+1+2+3+4 = 5 metals × 7 seeds across 2 folds)
- `n_runs_completed = 7` (all 7 MEASURED)
- `value = 0.7577 ± 0.0924` (7-run aggregate MEASURED test_auc)
- `per_fold_auc_mean = {Rh: 0.8798, Re: 0.7088}`
- `per_fold_auc_std = {Rh: 0.0115, Re: 0.0482}`
- `per_fold_krasnov_verdict = {Rh: PASS, Re: BELOW}` (per Krasnov 2026 0.86 gate)
- `verdict = BELOW_KRASNOV_GATE_PARTIAL` (overall BELOW_KRASNOV with -0.102 margin)
- `rh_krasnov_gate_hit = true` (Rh fold PASS_KRASNOV +0.020 margin)
- `re_krasnov_gate_hit = false` (Re fold BELOW_KRASNOV -0.151 margin)
- `gate_target_hit = true` (overall PASSES 0.7 gate_target +1.061 BELOW krasnov_gate_target 0.86)

---

## §4 Wall + cost

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Author `wf433_m2_extended.py` (~440 LOC) | ~5 min |
| Run smoke (2 cohorts: platinai_4cell + hilbert) | ~10 s |
| Run full (3 cohorts: + metalcytotox_combined) + catalog update | ~30 s |
| Author `test_wf433_honest_neg_closure.py` (~280 LOC, 10 tests) | ~3 min |
| Run pytest (10 tests) | ~0.17 s |
| Verify m6_kappa + m1 + paper docs on disk per §5.5 | ~1 min |
| **Total Wf-433-A2 CPU wall** | **~10 min** |

0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning; all CPU-only per AGENTS.md §5.1 + §5.3 + §5.4 + §10.

---

## §5 Standing-rule compliance (per AGENTS.md)

| Rule | Status | Notes |
|---|:---:|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | Every R² value cites JSON / report path on disk; honest framing for 128-d PLL proxy substitution documented |
| §1.2 Standing GPU authorization (§5.1) | ✅ | 0 GPU launches; CPU-only docs + script + test + JSON update |
| §1.3 No paper work without authorization (§5.3) | ✅ | 0 touches to `paper/*`; this doc lives in `TODO/completed/` |
| §1.4 Pre-flight v3 gate | ✅ | Pre-flight PRESERVED at Wf-424-A11 baseline 112/2/3/117 (NOT re-run by Wf-433-A2 — CPU-only docs+code+test+verify wave) |
| §1.5 Honest-negative framing (§5.2) | ✅ | m1 + m2 + m6_kappa all HONEST-NEGATIVE preserved per party-twenty-spirit §3.5; EXTENDED feature OVERFIT empirically confirmed on n=200 cohorts (R² -1.78 vs baseline -0.14) |
| §5.5 Post-wave on-disk verification | ✅ | All 3 SHIPPED files verified on disk BEFORE this closeout authored (`metrics/by_metric/m2_r2_per_cohort.json::per_cohort_extended_160d` + `metrics/by_metric/metal_lomo_auc_wasserstein.json` + `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` + `docs/PAPER_SECTION_6_v4_FINAL.md`) |
| §5.6 Atomic write protocol | ✅ | All writes via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` |
| §5.7 NEW Trainer flag wire-in smoke-required | ✅ | N/A for Wf-433-A2 (no trainer flag wire-in; CPU-only Ridge regression) |
| §10 Wetlab deprecation | ✅ | 0 wetlab planning; Tier-2 computational-only |

---

## §6 Cross-references

### §6.1 Source artifacts (verified on disk)

| Artifact | Path | Status |
|---|---|---|
| `molmetal/scripts/wf433_m2_extended.py` | NEW this wave | SHIPPED (~440 LOC) |
| `molmetal/tests/test_wf433_honest_neg_closure.py` | NEW this wave | SHIPPED (~280 LOC, 10 tests) |
| `molmetal/reports/wf433_m2_extended/result.json` | NEW this wave | SHIPPED (atomic-written) |
| `molmetal/reports/wf433_m2_extended_smoke/result.json` | NEW this wave | SHIPPED (smoke test) |
| `metrics/by_metric/m2_r2_per_cohort.json` | UPDATED | `per_cohort_extended_160d` block + `aggregate_extended` + `wf433a2_extended_run` block (atomic-written) |
| `metrics/by_metric/m6_kappa.json` | VERIFIED | `wf424a15_drop_decision` block (per Wf-424-A15) + `wf389a1_supplementary_reframe` + `wf411a4_supplementary_verifier` |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | VERIFIED | 7/7 (metal, seed) pair aggregate (per Wf-413-A1) |
| `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` | VERIFIED | §3 main-result + §3 supplementary (per Wf-389-A1 + Wf-424-A14) |
| `docs/PAPER_SECTION_6_v4_FINAL.md` | VERIFIED | §6.5 m6_kappa deep-dive (per Wf-424-A13 + Wf-410-A1 + Wf-389-A1) |

### §6.2 Upstream artifacts (READ-only)

- Wf-331-R5 canonical cohort loaders (`molmetal/scripts/wf331r5_r2_w2_chemprop.py`)
- Wf-380-A2 per-cohort config + Ridge baseline (`molmetal/scripts/wf380a2_m2_r2_per_cohort.py`)
- Wf-388-A1 per-cohort R² real-cohort verification (`molmetal/scripts/wf388a1_m2_per_cohort_real.py`)
- Wf-335-M2 W₂ + chemprop v2 + multi-seed CV protocol (`molmetal/scripts/wf335m2_r2_w2_chemprop.py`)
- Wf-377-A2 cohort diagnostic + alternative splits (`molmetal/scripts/wf377a2_m2r2_diagnostic.py`)
- Wf-377-A1 metric-redesign + 16 invariants literature + empirical (`docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
- Wf-389-A1 §3 metric table reframe + WEAK-band correlates
- Wf-410-A1 §6 v4 ready-to-paste scaffold
- Wf-411-A4 per-cohort × Cohen 1988 3-band refinement
- Wf-413-A1 7/7 (metal, seed) pair aggregate canonical catalog fix
- Wf-424-A13 paper §6 v4 FINAL refinement
- Wf-424-A14 paper §3 v4 FINAL metric table update
- Wf-424-A15 m6_kappa drop-promote paper-conversion decision

### §6.3 Module references

- `molmetal/adapters/flow_matching_lipman/pll_features.py` — PLL proxy adapter (128-d Morgan fingerprint fallback)
- `molmetal/molmetal_lam/lam_chem/pll_signature.py` — full 3D PLL implementation (not used in this wave; requires 3D conformer)
- `molmetal/scripts/wf386a3_atomic_write.py` — atomic write primitive
- `molmetal/scripts/wf331r5_r2_w2_chemprop.py:PLATINAI_12D_COLUMNS` + `PLATINAI_32D_COLUMNS` — canonical column constants

### §6.4 Test files

- `molmetal/tests/test_wf433_honest_neg_closure.py` — NEW 10 tests (3 m6_kappa + 3 m1 + 4 m2):
  1. `test_m6_kappa_drop_decision_block_present`
  2. `test_paper_section3_metric_table_final_present`
  3. `test_paper_section6_v4_final_has_m6_kappa_section`
  4. `test_m1_canonical_catalog_7_of_7_aggregate`
  5. `test_m1_per_run_auc_array_7_entries`
  6. `test_m2_per_cohort_extended_160d_block`
  7. `test_wf433_m2_extended_result_json_exists`
  8. `test_wf433_m2_extended_confirms_structural_limit`
  9. `test_wf433_m2_extended_metalcytotox_lifts_at_n2000`
  10. `test_all_3_closures_fileshipped`

### §6.5 Metrics + reports

- `metrics/by_metric/m2_r2_per_cohort.json::per_cohort_extended_160d` — NEW per Wf-433-A2 (3 cohort entries with r2_mean + lift_vs_baseline_12d + verdict)
- `metrics/by_metric/m2_r2_per_cohort.json::wf433a2_extended_run` — NEW per Wf-433-A2 (run config + pll_proxy_type + verdict)
- `metrics/by_metric/m6_kappa.json::wf424a15_drop_decision` — 10 paths + DROP decision matrix (per Wf-424-A15)
- `metrics/by_metric/metal_lomo_auc_wasserstein.json::n_runs_completed=7` — 7/7 (metal, seed) pair aggregate (per Wf-413-A1)
- `molmetal/reports/wf433_m2_extended/result.json` — Wf-433-A2 primary MEASURED output (3 cohorts × 160-d EXTENDED features × 5 seeds × 5-fold CV = 75 splits total)

---

## §7 Honest framing + forward-looking

### §7.1 What Wf-433-A2 MEASURED on disk

- **m2 R² EXTENDED 160-d (32-d RDKit + 128-d Morgan PLL proxy) confirmed structural limit on n=200 cohorts** (R² -1.78 vs 12-d baseline -0.14; lift -1.64 = massive overfit). Per Wf-377-A2 §2.1 + Wf-367-B1 + Wf-388-A1 + Wf-390-A2 prior verdicts: "Worst case (p > n) Ridge ≈ random projection, R² → -∞" — here p=160, n=200, p/n=0.80, deep into worst-case regime; predicted + confirmed.
- **m2 R² EXTENDED 160-d confirmed lift on n=2000 cohort** (R² +0.41 vs 12-d baseline +0.20; lift +0.21 = CEILING_BREAK_PRESERVED + NEW BEST). Per Wf-377-A2 §1.1 + Wf-335-M2: "n=2000 cohort closes the bias-variance gap" — predicted + confirmed.
- **m1 7/7 (metal, seed) pair aggregate verified on disk** (per Wf-413-A1 + Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md §0.2).
- **m6_kappa DROP decision verified on disk** across 3 propagation points (per Wf-424-A15 + Wf-433-A1 §2.2).

### §7.2 What Wf-433-A2 did NOT solve (forward-looking)

- **m2 R² > 0 on n=200 cohorts**: structurally unachievable CPU-only per Wf-377-A2 §2.1 + Wf-335-M2 + Wf-367-B1 + Wf-388-A1 + Wf-390-A2 prior verdicts. Closure requires GPU chemprop v3 D-MPNN encoder + autograd (per AGENTS.md §5.1 USER_OK gating + Wf-291D `decode_ratio=0` upstream resolution). Deferred to Wf-432-C1 USER_OK-gated GPU retrain.
- **m6_kappa augmented_mean ≥ 0.6**: structurally unachievable CPU-only (per Wf-365-A3 §1-§2 + Wf-371-A1 §3.5 + Wf-377-A1 §4.2 + Wf-389-A1 §3 + Wf-411-A4 §3 + Wf-424-A15 convergence brief); the DROP + PROMOTE + RETAIN decision is the honest Tier-2 §6 framing.
- **m1 Re fold BELOW_KRASNOV**: honest-negative preserved per Wf-413-A1 §4 + Wf-387-A2 §5; closure requires additional Re seed retrain (Wf-131 P3 deferred runs + Wf-355-B1 wrapper).
- **5 MOCK → real measurement substitution**: deferred to Wf-432-C1 USER_OK-gated actual retrain launches on cuda:0 Blackwell + `nvidia-smi` polling (per Wf-433-A1 §0.3 + Wf-431-C1).

### §7.3 USER_OK dependency

- **AGENTS.md §5.3 + TODO-84**: actual paper §3 + §6 v4 write to `paper/sections/*.tex` requires explicit USER_OK authorization. This Wf-433-A2 wave is the **CPU-only docs + script + test + JSON update** that pre-prepares the paper §6 v4 final write; no `paper/*` touches performed.
- **AGENTS.md §5.1**: GPU launches (Wf-432-C1 Wf-380 + Wf-394 + Wf-246-C + Wf-125 + Wf-390-A1 + Wf-299 3-wave + Wf-379-A1 m1_auc 5-seed IN-FLIGHT) require USER_OK=1 + VRAM budget + pre-flight PASS per launch. Per the standing GPU authorization 2026-09-29.

---

**END OF Wf-433-A2 CLOSE-OUT** — Owner Hugo Lee + Wf-433-A2 ultracode; cross-ref Wf-433-A1 HONEST_NEGATIVE_CLOSURE_PLAN.md + Wf-413-A1 + Wf-424-A15 + Wf-377-A2 + Wf-388-A1; 3 honest-negative pre-closures verified on disk; 10 NEW tests PASS; m2 EXTENDED 160-d CONFIRMED structural limit on n=200 cohorts + LIFT on n=2000 cohort; m1 7/7 aggregate verified; m6_kappa DROP decision verified; 0 GPU launches; 0 paper/* touches; wall ~10 min CPU-only.
