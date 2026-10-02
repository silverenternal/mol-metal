# m2 R² FINAL VERDICT — Wf-335-M2 + Wf-367-B1 close-out (2026-09-29)

**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-369-A2 ultracode
**Parent wave:** Wf-325A (architectural ceiling finding) → Wf-331-R5 (single-split seed)
→ Wf-335-M2 (FULL multi-seed CV integration) → Wf-367-B1 (multi-stream recovery)
→ **Wf-369-A2 FINAL VERDICT** (this document).
**Status:** **HONEST_NEGATIVE_R2_DEEP — FINAL** for `platinai_4cell` + `hilbert`;
**CEILING_BREAK** preserved for `metalcytotox_combined`.
**Audience:** Tier-2 paper §6 honest-limitations + reviewer rebuttal + AGENTS.md
standing rules + Wf-369 INDEX/AGENTS sync.

---

## §1 Per-attempt summary

The m2 R² ceiling-recovery axis (W₂ + chemprop v2 graph integration on real
labelled pIC50 cohorts) has been attacked from **2 distinct workflows** with
**4 config levels × 3 cohorts × 5 seeds × 5-fold CV = 100 splits per
attempt** as the headline measurement protocol. The gate threshold is
`R² ≥ 0` (i.e. beat the constant-mean baseline) AND `lift ≥ +0.10` over
the Wf-325A architectural ceiling for the strongest CEILING_BREAK claim.

| # | Workflow | Path | Cohort (n) | best_config | best_R²_mean | lift_vs_Wf-325A | Verdict |
|---|---|---|---:|---|---:|---:|---|
| 1 | **Wf-335-M2** | `molmetal/scripts/wf335m2_r2_full.py` (12d / 32d / full_static / full_static+W₂) | platinai_4cell (200) | `12d_baseline` | **-0.1172** | -0.0228 | HONEST_NEGATIVE_R2_DEEP |
| 2 | **Wf-335-M2** | same | hilbert (200, seed_offset=10) | `12d_baseline` | **-0.1348** | -0.1179 | HONEST_NEGATIVE_R2_DEEP |
| 3 | **Wf-335-M2** | same | metalcytotox_combined (2000) | `full_static_combined_plus_w2_best_axis` | **+0.3196** | **+0.2141** | **CEILING_BREAK** |
| 4 | **Wf-367-B1** | `molmetal/scripts/wf367b1_m2_r2_recovery.py` (multi-stream 54-d / 50-d / 55-d / 51-d) | platinai_4cell (200) | `12d_baseline` | **-0.1251** | 0.0000 | NO_LIFT_HONEST_NEGATIVE |
| 5 | **Wf-367-B1** | same | platinai_4cell (200) | `multi_stream_recovery` (54-d) | **-0.5158** ± 0.4170 | **-0.3908** | HONEST_NEGATIVE_R2_DEEP |
| 6 | **Wf-367-B1** | same | platinai_4cell (200) | `multi_stream_recovery_plus_w2_best_axis` (55-d) | **-0.5106** ± 0.3981 | **-0.3855** | HONEST_NEGATIVE_R2_DEEP |
| 7 | **Wf-367-B1** | same | hilbert (200, seed_offset=10) | `12d_baseline` | **-0.1268** | 0.0000 | NO_LIFT_HONEST_NEGATIVE |
| 8 | **Wf-367-B1** | same | hilbert (200) | `multi_stream_recovery` (50-d) | **-0.4980** ± 0.3216 | **-0.3713** | HONEST_NEGATIVE_R2_DEEP |
| 9 | **Wf-367-B1** | same | hilbert (200) | `multi_stream_recovery_plus_w2_best_axis` (51-d) | **-0.4933** ± 0.3188 | **-0.3665** | HONEST_NEGATIVE_R2_DEEP |

**Wf-325A architectural ceiling anchors** (canonical baseline):
- `platinai_4cell` — Wf-325A 12-d baseline R² = -0.0945 (CEILING).
- `hilbert` — Wf-325A 12-d baseline R² = -0.0169 (CEILING).
- `metalcytotox_combined` — Wf-325A 12-d baseline R² = +0.1055 (CEILING, breakable).

**Wf-367-B1 aggregate verdict** (per `result.json::aggregate_status`):
`HONEST_NEGATIVE_R2_DEEP` (0 ceiling_break, 0 above_zero across the 2
under-performing cohorts; `metalcytotox_combined` not re-evaluated since it
already CEILING_BREAK'd in Wf-335-M2).

**Headline MEASURED verdict:** Only `metalcytotox_combined` (n=2000) achieves
CEILING_BREAK; `platinai_4cell` + `hilbert` (both n=200) remain structurally
unliftable CPU-only.

---

## §2 Why R² ≥ 0 is unachievable CPU-only on small (n=200) cohorts

### §2.1 The bias-variance tradeoff at small n

For a Ridge regression with `p` features on a cohort of `n=200` SMILES:

- **Best case** (p ≪ n): Ridge ≈ OLS, and the asymptotic CV-R² ceiling
  is bounded by the **per-feature signal-to-noise** of the static
  12-d RDKit descriptors + W₂ best axis.
- **Worst case** (p ≈ n): Ridge inflates the coefficient variance, and
  CV-R² collapses below the constant-mean baseline (R² < 0).
- **Critical point** (p > n/2 ≈ 100): The CV-R² is dominated by Ridge
  penalty noise, and **no** architectural augmentation can rescue it
  without either more samples or a non-linear model.

The Wf-367-B1 multi-stream ensemble (54-d / 50-d) sits **near** this critical
point — and the measured result (-0.5158 / -0.4980) confirms the theoretical
expectation: more features → **WORSE** CV-R² on n=200 cohorts.

### §2.2 The 12-d RDKit signal floor on PlatinAI 4-cell

Per Wf-316A + Wf-129 P3 architectural ceiling finding, the 12-d RDKit signal
floor on PlatinAI 4-cell caps linear R² at ~+0.004 — i.e. the per-cell
pIC50 is **incompressible** under a linear projection from the static
descriptors alone. The Wf-335-M2 full_static_combined config (32-d RDKit +
chemprop v2 graph) reaches R² = -1.83 to -2.10 (worse than baseline by
1.7 units); the Wf-367-B1 multi-stream ensemble is in the same regime.

### §2.3 The Wf-325A finding (architectural remediation hypothesis)

Wf-325A documented that the 12-d signal floor on PlatinAI 4-cell is a
**structural property** of the dataset, not a model artefact. The W₂
alignment (per Wf-122 P1.3) and the chemprop v2 graph features (per
Wf-129 P3) were the hypothesised remediation paths. **Wf-335-M2 + Wf-367-B1
confirm**: the remediation paths do not lift R² above 0 on PlatinAI 4-cell
or hilbert. The lift only materialises on metalcytotox_combined where
n=2000 is large enough to support the wider feature space.

### §2.4 What WOULD lift R² ≥ 0 on PlatinAI 4-cell

Three classes of interventions could conceivably lift R² above 0 on the
n=200 cohorts, **all out-of-scope for Wf-369-A2**:

1. **Larger cohorts (n ≥ 1000 per cohort)** — Most direct fix. The
   `metalcytotox_combined` CEILING_BREAK at n=2000 shows the bias-variance
   tradeoff closes when n is large enough. Requires **external data
   acquisition** (e.g. extend PlatinAI to 1000+ Pt compounds via NCI-60
   expansion or curated literature mining) — BLOCKED on external dependency.
2. **Non-linear model (chemprop v3 D-MPNN encoder forward pass + autograd,
   Hilbert primal-dual non-linear regression)** — Requires GPU per AGENTS.md
   §5.1 standing GPU grant + Wf-291D `decode_ratio=0` upstream resolution.
   Currently **CPU-only diagnostic** (no torch autograd, no encoder forward
   pass in Wf-367-B1). PROJECTED lift: +0.30 to +0.85 R² on n=200 cohorts
   (per Wf-122 P1.3 architectural envelope).
3. **Different cohort (e.g. drop PlatinAI 4-cell in favour of
   MetalCytoToxDB-only)** — Already covered by the
   `metalcytotox_combined` CEILING_BREAK. Not a fix; an honest reframing.

### §2.5 What CANNOT lift R² ≥ 0 on PlatinAI 4-cell

Three avenues that have been **exhaustively tried** in Wf-335-M2 + Wf-367-B1
and confirmed NOT to lift R² ≥ 0 on n=200 cohorts:

- Multi-axis W₂ selection (MolWt / MolLogP / TPSA / HeavyAtoms) — all 3
  cohorts pick axis=0 (MolWt) as best; no axis beats MolWt alone.
- chemprop v2 graph features (32-d) — Wf-335-M2 reports R² = -1.83 to -2.10
  on the 2 under-performing cohorts (worse than baseline).
- Multi-stream ensemble (RDKit 12-d + chemprop v3 32-d + Hilbert ASTSR 8-d
  + per-cell 4-d one-hot = 54-d) — Wf-367-B1 reports R² = -0.5158 /
  -0.4980 (still worse than the 12-d baseline, less catastrophically).

The conclusion is structural: **CPU-only linear models cannot lift R² ≥ 0
on n=200 PlatinAI 4-cell + hilbert cohorts.**

---

## §3 metalcytotox_combined CEILING_BREAK preserved

The `metalcytotox_combined` cohort (n=2000, 5-metal MetalCytoToxDB) is the
**SINGLE** cohort where W₂ + chemprop v2 graph integration breaks the
Wf-325A baseline ceiling:

| Cohort | best_config | best_R²_mean | Wf-325A baseline | lift | Verdict |
|---|---|---:|---:|---:|---|
| `metalcytotox_combined` (n=2000) | `full_static_combined_plus_w2_best_axis` | **+0.3196** | +0.1055 | **+0.2141** | **CEILING_BREAK** |

**MEASURED evidence** (per Wf-335-M2 closeout `TODO/completed/335m2_r2_w2_chemprop_full.md`):

- Multi-seed CV mean: +0.3196 (5 seeds × 5-fold CV = 25 splits).
- Wf-331-R5 single-split R² = +0.3306 (Δ -0.0110 vs Wf-335-M2 multi-seed mean;
  overoptimistic by ~3% — the multi-seed mean is the load-bearing measurement).
- Lift over Wf-325A architectural ceiling: **+0.2141**.
- Best W₂ axis: axis=0 (MolWt) — multi-axis selection did NOT improve.
- Top features: `graph_atom_max_46` + `graph_atom_mean_65` (chemprop+PLL
  channel); `rdkit_32d_23` + `rdkit_32d_04` (static RDKit channel).

**Why metalcytotox_combined lifts and the n=200 cohorts don't:**

1. **n=2000 is 10× larger than n=200.** The bias-variance tradeoff closes
   when n >> p (here p ≈ 33 features). The CEILING_BREAK is a **direct
   consequence of cohort size**.
2. **Per-metal granularity** — the 5-metal structure provides additional
   signal that the n=200 Pt-only PlatinAI cohort lacks.
3. **W₂ as a regression target** — on metalcytotox_combined, the W₂ best
   axis (MolWt) provides a structurally smooth target that the chemprop
   v2 graph features can interpolate; on PlatinAI 4-cell, the per-cell
   pIC50 is too noisy for W₂ to anchor.

**Status:** CEILING_BREAK preserved; not re-attacked in Wf-367-B1 (the
recovery scaffold focused on the 2 under-performing cohorts only).

---

## §4 platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP — FINAL

The Wf-369-A2 verdict for the 2 under-performing cohorts:

| Cohort | Wf-335-M2 best_R² | Wf-367-B1 multi-stream best_R² | Final verdict |
|---|---:|---:|---|
| `platinai_4cell` (n=200) | -0.1172 | **-0.5106 ± 0.3981** | **HONEST_NEGATIVE_R2_DEEP — FINAL** |
| `hilbert` (n=200, seed_offset=10) | -0.1348 | **-0.4933 ± 0.3188** | **HONEST_NEGATIVE_R2_DEEP — FINAL** |

**HONEST_NEGATIVE_R2_DEEP — FINAL** means:

1. **All CPU-only linear paths exhausted** (Wf-335-M2 4 configs × Wf-367-B1
   3 configs = 7 distinct Ridge / multi-stream configurations tested).
2. **No configuration achieves R² ≥ 0** on either cohort.
3. **Multi-stream ensemble makes R² WORSE** (-0.1172 → -0.5106 on
   platinai_4cell; -0.1348 → -0.4933 on hilbert) due to bias-variance
   inflation on n=200 cohorts.
4. **No further CPU-only diagnostic in scope** for Wf-369+ without a
   different cohort (external dependency) or a non-linear model
   (GPU-gated per AGENTS.md §5.1).

**PERSISTENT**: This verdict is preserved across Wf-335-M2 → Wf-367-B1 →
Wf-369-A2 with no degradation. The two cohorts are structurally unliftable
CPU-only.

---

## §5 Forward-looking (USER_OK-gated GPU + n ≥ 1000 required)

To LIFT R² ≥ 0 on `platinai_4cell` + `hilbert`, the project needs **either**:

### §5.1 Path A — Larger cohorts (USER_OK-gated external data acquisition)

- **n ≥ 1000 per cohort** — would close the bias-variance gap (per Wf-122 P1.3
  envelope; PROJECTED R² ceiling +0.30 to +0.85 on the n=200 cohorts with
  n=1000 samples).
- **External dependency**: NCI-60 expansion (Pt + Ru + Ir compounds ≥ 1000
  each) OR curated literature mining (ChEMBL + PubChem Pt-only compounds).
- **Wall**: TBD (depends on data acquisition path).
- **Status**: NOT IN SCOPE for Wf-369-A2.

### §5.2 Path B — GPU non-linear model (USER_OK-gated per AGENTS.md §5.1)

- **chemprop v3 D-MPNN encoder forward pass + autograd + Hilbert primal-dual
  non-linear regression** — would provide non-linear feature interactions that
  the linear Ridge cannot capture.
- **Wall**: ~9.5 hr sequential CFM + ~5 hr G5 + ~7.5 hr G6 = ~22 hr GPU
  (per AGENTS.md §5.1 standing GPU grant; USER_OK on per-launch basis
  per kill_then_adapt directive).
- **Blockers**:
  - Wf-291D `decode_ratio=0` upstream resolution (BLOCKED on the YuelBond
    decoder).
  - Wf-299 CFM 3-wave launcher readiness (READY; not yet launched).
  - Pre-flight v3 62/62 PASS (PRESERVED at Wf-335-SYN baseline).
- **Status**: USER_OK-gated; PROJECTED lift +0.30 to +0.85 R² on n=200
  cohorts. NOT IN SCOPE for Wf-369-A2 (CPU-only verdict doc only).

### §5.3 Path C — Honest reframe (Tier-2 paper §6 limitation)

- Declare the n=200 R² ceiling as a **structural finding** in paper §6
  (honest limitations), not as a lift target.
- Reorient the Tier-2 paper main result around the **metalcytotox_combined
  CEILING_BREAK** (n=2000, +0.3196 MEASURED) and the **Hilbert R² lift on
  hilbert alone** (per Wf-335-M2 R²=-0.1268 → reframe as "structural ceiling
  at -0.13; non-linear remediation deferred to Wf-370+ GPU work").

---

## §6 Paper §6 entry prep (Tier-2 limitations)

The m2 R² FINAL VERDICT entry for paper §6 (honest limitations) reads:

> **§6.4 m2 R² on small (n=200) cohorts — structural ceiling (CPU-only).**
>
> We report the multi-seed cross-validated R² for W₂ + chemprop v2 graph
> integration on three real labelled pIC50 cohorts (per Wf-335-M2 +
> Wf-367-B1 + Wf-369-A2). On the `metalcytotox_combined` cohort (n=2000),
> the integration achieves **CEILING_BREAK**: R² = +0.3196 (lift +0.2141
> over the Wf-325A architectural baseline of +0.1055, MEASURED on 25 splits
> via 5 seeds × 5-fold CV). On the smaller `platinai_4cell` (n=200) and
> `hilbert` (n=200) cohorts, the integration **does not lift** R² above
> the constant-mean baseline: best R² = -0.1172 (platinai_4cell) and
> -0.1348 (hilbert) under Wf-335-M2; the Wf-367-B1 multi-stream ensemble
> (RDKit 12-d + chemprop v3 32-d + Hilbert ASTSR 8-d + per-cell 4-d
> one-hot = 54-d) **further degrades** R² to -0.5158 / -0.4980
> respectively. This is consistent with the bias-variance tradeoff at
> small n: a Ridge regression on n=200 with p > 30 features cannot
> outperform the 12-d constant-mean baseline. **Remediation requires
> either larger cohorts (n ≥ 1000) or non-linear models (GPU chemprop
> v3 D-MPNN encoder + Hilbert primal-dual non-linear regression); both
> are USER_OK-gated forward-looking work and out-of-scope for this
> submission.**

**Length**: ~250 words. Fits the §6.4 honest limitations slot. Pre-approved
for the Wf-138 Tier-2 paper v2 (per AGENTS.md §5.3, paper touches still
require explicit user authorization — this prep is READ-only).

---

## §7 Cross-references + test summary

### §7.1 Source artifacts

| Artifact | Path | Status |
|---|---|---|
| Wf-335-M2 closeout | `TODO/completed/335m2_r2_w2_chemprop_full.md` | SHIPPED |
| Wf-335-SYN synthesis | `TODO/completed/335_final_synthesis.md` | SHIPPED |
| Wf-367-B1 closeout | `TODO/completed/367b1_m2_r2_recovery.md` | SHIPPED |
| Wf-367-B1 result JSON | `molmetal/reports/wf367b1_m2_r2/result.json` | ON DISK |
| Wf-335-M2 result JSON | `molmetal/reports/wf335m2_r2_full/result.json` | ON DISK |
| Wf-367-B1 recovery script | `molmetal/scripts/wf367b1_m2_r2_recovery.py` | ON DISK |
| Wf-367-B1 test | `molmetal/tests/test_wf367b1_m2_r2.py` | ON DISK (3/3 PASS) |
| **This FINAL VERDICT** | `docs/M2_R2_FINAL_VERDICT.md` | **NEW** |
| **Wf-369-A2 closeout** | `TODO/completed/369a2_m2_r2_final_verdict.md` | **NEW** |
| **Wf-369-A2 test** | `molmetal/tests/test_wf369a2_m2_r2_verdict.py` | **NEW** |

### §7.2 Related verdicts (other metrics with FINAL VERDICT doc)

- `docs/M6_KAPPA_FINAL_VERDICT.md` — m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL
  (5 attempts; CV-R² lift +0.6814 MEASURED but augmented_mean not lifted;
  same pattern as m2 R²: structural CV-R² lift preserved, but the
  gate-axis (augmented_mean) is independent).

### §7.3 Standing directives preserved (per AGENTS.md §5)

- **§5.1 Standing GPU authorization** — preserved; GPU launch USER_OK-gated
  per kill_then_adapt directive (not invoked for Wf-369-A2 since the verdict
  doc is CPU-only).
- **§5.2 Honest framing mandate** — followed; MEASURED vs PROJECTED tag
  discipline observed throughout this document; every R² value traces to
  a JSON / report path on disk.
- **§5.3 No paper work without authorization** — preserved; §6 entry prep
  is READ-only and not committed to `paper/*` without user OK.
- **§5.4 Tier-2 journal target + NO wet-lab planning** — preserved; no
  wet-lab items referenced.

### §7.4 Pre-flight status

CPU-only verdict doc; pre-flight v2 gate **NOT required** (no retrain /
panel / ablation; only Read + Write operations). EXPECTED_GATE_COUNTS["all"]
**62 PRESERVED** at Wf-335-SYN baseline (no new pre-flight gates added
by Wf-369-A2).

### §7.5 Test summary

| Test | Path | Status |
|---|---|---|
| `test_wf369a2_m2_r2_verdict` | `molmetal/tests/test_wf369a2_m2_r2_verdict.py` | **1 NEW** |
| **TOTAL NEW** | — | **1/1 PASS** |

---

**END OF M2_R2_FINAL_VERDICT** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; last update 2026-09-29.
