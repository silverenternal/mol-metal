# Wf-327-F12 — m6_kappa BELOW threshold root-cause investigation (VERIFIED — already closed by Wf-325D)

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Parent:** Wf-325D / TODO-130 (G2 m6_kappa validation) / Wf-252 F2-M6Kappa / Wf-253D validation / Wf-314A honest-negative close
**Status:** ✅ VERIFIED — Wf-325D has already completed this investigation. Investigation = REPLAY + CONFIRM. No new work needed.

---

## §0 TL;DR — investigation status

| Item | Value |
|---|---|
| Wf-325D status | ✅ SHIPPED (`TODO/completed/325d_m6_kappa_final.md`, 11:59 timestamp) |
| Pre-flight gate `p3_m6_kappa_closeout_documented` | ✅ PASS (per Wf-325D §6) |
| Pre-flight gate `p3_admet_m6_kappa_honest_negative` | ✅ PASS (per Wf-309A / Wf-314A) |
| m6_kappa MEASURED cohort mean | **0.4719** (n=449, metallo_drugs_500_train) |
| Threshold | ≥ 0.6 (ChemAxon PMC11387870 druglikeness Shape Kappa Index) |
| Delta | **-0.128** (FAIL honest-negative) |
| MetalCytoToxDB global Pearson r | **0.0813** (NEGATIVE_WEAK, |r| < 0.1) |
| MetalCytoToxDB global Spearman ρ | **0.1114** (NEGATIVE_WEAK, ρ < 0.2) |
| Root cause category | **METRIC_DEFINITION** + **HONEST_NEGATIVE_BLOCKED** |
| CPU-only fixable? | **PARTIAL** — closeout + framing already done (Wf-325D). Metric lift to |r| ≥ 0.3 requires GPU work (chemprop v2 + W₂, USER_OK-gated). |
| Verdict | **VERIFIED** — skip Wf-327-F12 fresh investigation; Wf-325D is the canonical closeout. |

**One-line verdict:** Wf-327-F12 confirms Wf-325D already investigated and shipped. m6_kappa BELOW threshold is a structural metric-definition limitation (Shape Kappa is a topological index, not a cytotoxicity predictor) + HONEST_NEGATIVE_BLOCKED on chemprop v2 / W₂ GPU recovery path. Tier-2 acceptance does NOT require m6_kappa ≥ 0.6.

---

## §1 Honest framing (`party-twenty-spirit` §3.5)

Per `AGENTS.md` §1 standing rules:

1. **MEASURED-vs-PROJECTED tag discipline** — every numeric value below is MEASURED, traced to `metrics/by_metric/m6_kappa.json` (Sep 28) + `TODO/completed/325d_m6_kappa_final.md` (Sep 29) + `molmetal/reports/wf253d_m6_kappa_validation/` (Wf-253D). Zero fabricated / aspirational numbers.
2. **"Kill then adapt" GPU directive** — `mol-metal-gpu-execution` §0. Wf-327-F12 is REPLAY + CONFIRM; NO GPU work performed. Recovery path remains USER_OK-gated.
3. **No paper work without authorization** — TODO-84 active; zero touches to `paper/*`. Wf-325D explicitly avoids paper work.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207) — `p3_m6_kappa_closeout_documented` PASS, `p3_admet_m6_kappa_honest_negative` PASS. Both already shipped.
5. **Honest-negative framing** — `HONEST_NEGATIVE: true` + `honest_negative_reason` + `honest_framing_note` preserved on JSON per Wf-312C R02 / Wf-314A. Wf-325D §6 mirrors these.

---

## §2 What is m6_kappa measuring? (replay from Wf-325D §4)

**Source:** `metrics/by_metric/m6_kappa.json::description` + `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles`

| Field | Value |
|---|---|
| Metric | **Shape Kappa Index** |
| Definition | Mean of normalised RDKit Kappa1/2/3 in [0, 1] (ChemAxon PMC11387870 convention) |
| Source code | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` |
| Unit | ratio (0-1, higher = more branched kappa shape) |
| Scope | drug-likeness panel (metrics_v2 Wf-252 F2 add-on) |
| Sister metric | RDKit Bickerton 2012 QED (`metrics/by_metric/qed.json::qed_value=0.6123` PASS) |

**Architectural fact:** Shape Kappa Index is a **normalised topological index** derived from RDKit Kappa1/2/3 shape descriptors. It measures molecular SHAPE / branching complexity — **NOT pharmacological activity**.

---

## §3 Threshold + MEASURED values (replay from Wf-325D §2 + §3)

### §3.1 Threshold

| Threshold | Source | Rationale |
|---|---|---|
| **≥ 0.6** | ChemAxon PMC11387870 druglikeness convention | Drug-like molecules in the ChemAxon reference set show Shape Kappa mean ≈ 0.6 |

### §3.2 MEASURED cohort (metallo_drugs_500_train)

| Statistic | Value | Verdict |
|---|---:|---|
| `value` (mean) | **0.4719** | ❌ FAIL |
| `std` | 0.1588 | — |
| `n_samples` | 449 | — |
| `n_parsed` | 449 | 0 parse failures (RDKit canonical SMILES graceful) |
| `n_failed` | 0 | — |
| Delta vs threshold | **-0.128** | BELOW threshold |

### §3.3 MetalCytoToxDB validation (5,460 rows)

Source: `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.md` (Wf-253D / TODO-130 G2)

| Statistic | Value | Verdict |
|---|---:|---|
| `n_total_rows` | 5,460 | MEASURED (≥ 5,000 floor) ✅ |
| Global Pearson r | **0.0813** | NEGATIVE_WEAK (|r| < 0.1) |
| 95% CI | [0.0549, 0.1076] | — |
| Global Spearman ρ | **0.1114** | NEGATIVE_WEAK (ρ < 0.2) |
| Per-metal Ru Pearson r | +0.0970 | WEAK |
| Per-metal Ir Pearson r | +0.0425 | WEAK |
| Per-metal Os Pearson r | +0.1377 | MODERATE on small n=87 (Spearman 0.317) |
| Per-metal Rh Pearson r | -0.0335 | NEGATIVE_WEAK |
| Per-metal Re Pearson r | **-0.1843** | NEGATIVE (Re shows inverse trend) |

---

## §4 Root-cause analysis (Wf-325D §4-§5 categorization)

### §4.1 Primary root cause: METRIC_DEFINITION

Shape Kappa Index is **NOT a learned metallodrug score** — it is a topological index measuring molecular shape/branching complexity. Per Wf-325D §4:

> Shape Kappa is side-by-side with RDKit Bickerton 2012 QED in paper §3 as a complementary druglikeness signal. It is NOT a cytotoxicity predictor — a near-zero Pearson r vs pIC50 is the EXPECTED outcome, NOT a regression.

Implication:
- Shape Kappa does NOT predict pIC50 — a Pearson r near zero is by-design.
- The BELOW-threshold cohort mean (0.4719 < 0.6) reflects the structural nature of metallo-drugs (high branching / square-planar geometry) vs the ChemAxon drug-like reference set.
- The `metrics/by_metric/m6_kappa.json::qed_comparison.honest_framing` field carries the cross-reference: "do NOT claim as a learned metallodrug score without validation against MetalCytoToxDB labelled pIC50."

### §4.2 Secondary root cause: HONEST_NEGATIVE_BLOCKED on metric lift

Per Wf-325D §5, the recovery path that COULD lift the correlation requires GPU work:

| Step | Action | Owner / status |
|---|---|---|
| 1 | chemprop v2 graph features (D-MPNN over MolCLR-style atom-bond encoder) | CPU scaffold SHIPPED per Wf-219 / Wf-243 |
| 2 | W₂ Wasserstein alignment on pIC50 distribution | CPU portion SHIPPED per Wf-243 |
| 3 | 5-seed × 50-epoch full GPU sweep | **BLOCKED** — USER_OK-gated per `mol-metal-gpu-execution` §0; Wf-310C / TODO-179 launch script on disk |
| 4 | If GPU run lifts m6_kappa correlation to |r| ≥ 0.3 → re-promote honest-negative to MEASURED_PASS | post-USER_OK only |

**Important:** Tier-2 acceptance bar does NOT require m6_kappa ≥ 0.6 (per `AGENTS.md` §5.4). m6_kappa is a druglikeness panel metric, NOT a cytotoxicity prediction. Honest-negative close-out is the SHIP state for this snapshot.

---

## §5 Is it CPU-fixable? — analysis

### §5.1 What is already CPU-done (Wf-325D)

1. ✅ Closeout document: `TODO/completed/325d_m6_kappa_final.md` (Sep 29)
2. ✅ Pre-flight gate `p3_m6_kappa_closeout_documented` (CPU, < 10 ms)
3. ✅ Pre-flight gate `p3_admet_m6_kappa_honest_negative` (Wf-309A / Wf-314A)
4. ✅ JSON `HONEST_NEGATIVE: true` + `honest_negative_reason` + `honest_framing_note` tags
5. ✅ MetalCytoToxDB per-metal Pearson + Spearman JSON (`molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.json`)
6. ✅ `claim_wf311e_m6_kappa_honest_negative` HONEST_NEGATIVE_CATALOG entry in `molmetal/scripts/paper_dashboard.py:607-612`

### §5.2 What requires GPU (NOT CPU-fixable)

- **Metric lift to |r| ≥ 0.3**: requires chemprop v2 graph features + W₂ alignment 5-seed × 50-epoch GPU sweep. USER_OK-gated per `mol-metal-gpu-execution` §0. Launch script: `scripts/launch_wf310c_chemprop_v2_gpu.sh`. ETA TBD.

**Conclusion:** m6_kappa CPU work is COMPLETE (closeout + framing + JSON tags + pre-flight gates). Metric lift requires GPU work and is USER_OK-gated. **CPU-only fix is NOT APPLICABLE** because the metric is already CPU-shut (properly closed + framed + tagged).

---

## §6 Honest-negative framing preservation (Wf-325D §6 contract)

The canonical tags on `metrics/by_metric/m6_kappa.json` are preserved per the Wf-308B / Wf-312C / Wf-314A / Wf-315A contract:

```json
{
  "metric": "m6_kappa",
  "value": 0.4719079955824899,
  "status": "MEASURED",
  "HONEST_NEGATIVE": true,
  "honest_negative_reason": "Mean cohort m6_kappa = 0.472 < 0.6 ... by -0.128; ... Pearson r 0.081, Spearman 0.111; honest-negative close per Wf-312C R02 druglikeness panel.",
  "honest_framing_note": "Shape Kappa Index emission. ... This is the side-by-side complement to RDKit Bickerton 2012 QED; do NOT claim it as a learned metallodrug score without validation against MetalCytoToxDB labelled pIC50.",
  "metalcytotox_validation": {
    "global_pearson_r": 0.08126287494327507,
    "global_spearman_rho": 0.11138252920995872,
    "verdict": "NEGATIVE_WEAK (Shape Kappa has negligible linear association with pIC50)"
  }
}
```

Pre-flight gates (both PASS):
- `p3_admet_m6_kappa_honest_negative` PASS (Wf-309A / Wf-314A) — verifies `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason`.
- `p3_m6_kappa_closeout_documented` PASS (Wf-325D ship) — verifies this closeout doc exists + carries 5 §0..§4 anchors + HONEST_NEGATIVE preservation note.

---

## §7 What Wf-327-F12 ships

1. `TODO/completed/327f12_m6_kappa_investigation.md` — this verification document (replay + confirm of Wf-325D).
2. **VERIFIED** — Wf-325D investigation is canonical and complete. No new investigation needed.
3. **NO** new pre-flight gate required (Wf-325D's `p3_m6_kappa_closeout_documented` already covers this).
5. **NO** paper `paper/*` touches (TODO-84 active).
6. **NO** GPU launches (USER_OK-gated per `mol-metal-gpu-execution` §0).
7. **NO** metrics JSON edits (Wf-325D already set `HONEST_NEGATIVE: true` + tags).

---

## §8 References

- `TODO/completed/325d_m6_kappa_final.md` — canonical Wf-325D closeout (this doc mirrors it).
- `metrics/by_metric/m6_kappa.json` — canonical JSON with `HONEST_NEGATIVE=true` + `metalcytotox_validation` block.
- `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.md` — Wf-253D detailed MetalCytoToxDB validation report.
- `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.json` — per-metal Pearson + Spearman JSON.
- `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validate.py` — validation script (CPU-only).
- `molmetal/scripts/paper_dashboard.py:607-612` — `claim_wf311e_m6_kappa_honest_negative` HONEST_NEGATIVE_CATALOG entry.
- `molmetal/scripts/pre_flight_check.py:1246-1300` — `gate_admet_honest_negative_tag` (used by `p3_admet_m6_kappa_honest_negative`).
- `TODO/completed/312_honest_framing_35plus.md` — Wf-312C R02 honest-negative close precedent (3 ADMET druglikeness gates: hepatotox / logp7_4 / m6_kappa).
- `TODO/completed/314a_wf311e_part4_deterministic_edits.md` — Wf-314A closeout for the deterministic edits that added `HONEST_NEGATIVE` tags.
- `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` — Wf-322F closeout template (Wf-325D + this doc follow the same structural pattern).
- `TODO/completed/309a_wf309_honest_negative_tag_audit.md` — pre-flight gate audit precedent.

---

**END OF Wf-327-F12 INVESTIGATION** — VERIFIED that Wf-325D already completed the m6_kappa BELOW threshold root-cause analysis. m6_kappa is HONEST_NEGATIVE preserved (Shape Kappa Index is a topological index, not a learned metallodrug score; recovery requires GPU work that is USER_OK-gated). Tier-2 acceptance bar does NOT require m6_kappa ≥ 0.6.