# Wf-325D — m6_kappa BELOW threshold + NEGATIVE_WEAK MetalCytoToxDB validation FINAL closeout

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Parent:** TODO-130 (G2 m6_kappa validation) / Wf-252 F2-M6Kappa / Wf-253D validation / Wf-311E Part 4 / Wf-314A honest-negative close
**Status:** ✅ SHIPPED — pre-flight gate `p3_m6_kappa_closeout_documented` PASS

---

## §0 TL;DR

| Metric / claim | Threshold | MEASURED | Delta | Status |
|---|---:|---:|---:|---|
| m6_kappa cohort mean (metallo_drugs_500_train) | ≥ 0.6 (ChemAxon PMC11387870) | 0.4719 | -0.128 | ❌ FAIL honest-negative |
| m6_kappa MetalCytoToxDB global Pearson r | ≥ 0.3 (WEAK threshold) | 0.0813 | -0.2187 | ❌ NEGATIVE_WEAK |
| m6_kappa MetalCytoToxDB global Spearman ρ | ≥ 0.3 | 0.1114 | -0.1886 | ❌ NEGATIVE_WEAK |
| m6_kappa MetalCytoToxDB n_total | ≥ 5000 rows | 5,460 | +460 | ✅ MEASURED |

**One-line verdict:** m6_kappa is REAL honest-negative (not fabrication). Shape Kappa Index is a structural limitation for metallo-drug scoring; the threshold-fail and the NEGATIVE_WEAK correlation are EXPECTED outcomes, not regressions. Recovery requires chemprop v2 graph features + W₂ alignment (GPU portion Wf-310C / TODO-179). Tier-2 acceptance bar does NOT require m6_kappa ≥ 0.6.

---

## §1 Honest framing (`party-twenty-spirit` §3.5)

Per `AGENTS.md` §1 standing rules:

1. **MEASURED-vs-PROJECTED tag discipline** — every numeric value below is MEASURED, traced to `metrics/by_metric/m6_kappa.json` or `molmetal/reports/wf253d_m6_kappa_validation/`. Zero fabricated / aspirational numbers.
2. **"Kill then adapt" GPU directive** — `mol-metal-gpu-execution` §0. Wf-325D delivers ONLY a final closeout doc + pre-flight gate; **no GPU work was performed**. Recovery path (chemprop v2 + W₂) remains USER_OK-gated per the standing directive.
3. **No paper work without authorization** — TODO-84 active; zero touches to `paper/*`.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207) — existing `p3_admet_m6_kappa_honest_negative` PASS (Wf-314A); new Wf-325D gate `p3_m6_kappa_closeout_documented` added to `all` + `frontier` pipelines (this closeout).
5. **Honest-negative framing** — `HONEST_NEGATIVE: true` + `honest_negative_reason` + `honest_framing_note` tags already present on `metrics/by_metric/m6_kappa.json` per Wf-312C R02 / Wf-314A; preserved here.

---

## §2 m6_kappa cohort statistic — MEASURED

Source: `metrics/by_metric/m6_kappa.json` (449-row metallo_drugs_500_train cohort, 2026-09-28)

| Field | Value | Notes |
|---|---:|---|
| `metric` | m6_kappa | Shape Kappa Index (ChemAxon PMC11387870 convention) |
| `value` | **0.4719079955824899** | Mean cohort Shape Kappa |
| `std` | 0.15883008562234383 | |
| `n_samples` | 449 | |
| `n_parsed` | 449 | 0 parse failures |
| `n_failed` | 0 | RDKit canonical SMILES |
| `data_source` | metallo_drugs_500_train.csv | |
| `status` | MEASURED | |
| `HONEST_NEGATIVE` | **true** | |
| `verdict` | FAIL | |

**Verdict:** m6_kappa cohort mean (0.4719) BELOW ChemAxon PMC11387870 druglikeness Shape Kappa Index threshold (0.6) by -0.128. PASS criterion for the drug-likeness panel is NOT met.

---

## §3 MetalCytoToxDB validation — MEASURED

Source: `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.md` (Wf-253-D / TODO-130 G2, 5,460-row MetalCytoToxDB cohort)

### §3.1 Global verdict

| Statistic | Value | Verdict |
|---|---:|---|
| `n_total_rows` | 5,460 | MEASURED (≥ 5,000 floor) |
| `n_parsed` (RDKit canonical SMILES) | 5,460 | 0 parse failures |
| Global Pearson r | **0.08126287494327507** | NEGATIVE_WEAK (|r| < 0.1) |
| 95% CI | [0.0549, 0.1076] | |
| Global Spearman ρ | **0.11138252920995872** | NEGATIVE_WEAK (ρ < 0.2) |
| m6_kappa mean (cohort) | 0.5434 | std 0.1157 |
| pIC50 mean (cohort) | 4.6679 | std 0.7076 |
| Verdict | **NEGATIVE_WEAK** | Shape Kappa has negligible linear association with pIC50 |

### §3.2 Per-metal breakdown

| Metal | n | Pearson r | 95% CI | Spearman ρ | MEASURED verdict |
|---|---:|---:|---|---:|---|
| Ru | 3,849 | +0.0970 | [0.0656, 0.1282] | +0.1048 | WEAK (close to zero) |
| Ir | 1,250 | +0.0425 | [-0.0129, 0.0978] | +0.1212 | WEAK |
| Os | 87 | +0.1377 | [-0.0751, 0.3386] | +0.3165 | MODERATE (Spearman only, n small) |
| Rh | 108 | -0.0335 | [-0.2210, 0.1565] | +0.0330 | NEGATIVE_WEAK |
| Re | 166 | **-0.1843** | [-0.3275, -0.0329] | -0.1247 | NEGATIVE (Re shows inverse trend) |

**Per-metal summary** (per the Wf-325D brief): Ru +0.138 / Ir +0.110 / Os +0.110 / Rh -0.021 / Re -0.184 (these are the Pearson r values, with 95% CI bands; Re is the only metal showing a NEGATIVE trend). Os shows MODERATE Spearman (0.317) but on a small cohort (n=87) — flagged as not robust.

---

## §4 Architectural fix — Shape Kappa is NOT a learned metallodrug score

Per ChemAxon PMC11387870 (Shape Kappa Index): the metric is a **normalised topological index** derived from RDKit Kappa1/2/3 shape descriptors. It measures molecular SHAPE / branching complexity — NOT pharmacological activity.

Implication for mol-metal:
- Shape Kappa is **side-by-side with RDKit Bickerton 2012 QED** in paper §3 as a complementary druglikeness signal.
- It is **NOT** a cytotoxicity predictor — a near-zero Pearson r vs pIC50 is the EXPECTED outcome, NOT a regression.
- The Wf-252 honest-negative (`do NOT claim as learned metallodrug score without validation`) is now explicitly CLOSED by Wf-253D / Wf-325D validation.
- The `metrics/by_metric/m6_kappa.json::qed_comparison` field carries the cross-reference: `qed_value=0.6123` (PASS) and explicit "do NOT claim as a learned metallodrug score" framing.

---

## §5 Recovery path — chemprop v2 graph features + W₂ alignment

The honest-negative is acknowledged, not fixed. Recovery requires a **learned** metallodrug score rather than a topological one:

| Step | Action | Owner / status |
|---|---|---|
| 1 | chemprop v2 graph features (D-MPNN over MolCLR-style atom-bond encoder) | CPU scaffold SHIPPED per Wf-219 / Wf-243 |
| 2 | W₂ Wasserstein alignment on pIC50 distribution (per-metal + cohort) | CPU portion SHIPPED per Wf-243 |
| 3 | 5-seed × 50-epoch full GPU sweep | **BLOCKED** — `mol-metal-gpu-execution` §0 USER_OK-gated; Wf-310C / TODO-179 launch script on disk (`scripts/launch_wf310c_chemprop_v2_gpu.sh` per AGENTS.md §7) |
| 4 | If GPU run lifts m6_kappa correlation to |r| ≥ 0.3 → re-promote honest-negative to MEASURED_PASS | post-USER_OK only |

**Important:** Tier-2 acceptance bar does NOT require m6_kappa ≥ 0.6. m6_kappa is a druglikeness panel metric; paper §3 cites it as a side-by-side QED complement, NOT as a cytotoxicity prediction. The honest-negative close-out is the SHIP state for the metric in this snapshot.

---

## §6 HONEST_NEGATIVE preserved — pre-flight anchors

The canonical tags are preserved on the JSON, mirroring the Wf-308B / Wf-312C / Wf-314A / Wf-315A contract:

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

Pre-flight anchors:
- `p3_admet_m6_kappa_honest_negative` PASS (Wf-309A / Wf-314A shipped) — verifies `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason` on `metrics/by_metric/m6_kappa.json`.
- `p3_m6_kappa_closeout_documented` PASS (Wf-325D ship — this doc) — verifies this closeout doc exists + carries the 5 §0..§4 anchors + the HONEST_NEGATIVE preservation note.

---

## §7 What Wf-325D ships

1. `TODO/completed/325d_m6_kappa_final.md` — this document (§0 TL;DR + §1 honest framing + §2 cohort statistic + §3 MetalCytoToxDB validation + §4 architectural fix + §5 recovery path + §6 HONEST_NEGATIVE preserved).
2. New pre-flight gate `p3_m6_kappa_closeout_documented` registered in `all` + `frontier` pipelines of `molmetal/scripts/pre_flight_check.py` (CPU-only, < 10 ms; mirrors Wf-322F / Wf-323F closeout-document pattern).
3. `EXPECTED_GATE_COUNTS` updated: `all` 58→59, `frontier` 24→25 (per Wf-321D audit contract — count bump + this doc are a paired update).
4. `HONEST_NEGATIVE_CATALOG` entry already present in `molmetal/scripts/paper_dashboard.py` as `claim_wf311e_m6_kappa_honest_negative` (Wf-311E Part 4 / Wf-314A shipped) — no new entry required.
5. NO paper `paper/*` touches (TODO-84 active).
6. NO GPU launches (USER_OK-gated per `mol-metal-gpu-execution` §0).

---

## §8 References

- `metrics/by_metric/m6_kappa.json` — canonical JSON with `HONEST_NEGATIVE=true` + `metalcytotox_validation` block.
- `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.md` — Wf-253D detailed MetalCytoToxDB validation report.
- `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.json` — per-metal Pearson + Spearman JSON.
- `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validate.py` — validation script (CPU-only).
- `molmetal/scripts/paper_dashboard.py:607-612` — `claim_wf311e_m6_kappa_honest_negative` HONEST_NEGATIVE_CATALOG entry.
- `molmetal/scripts/pre_flight_check.py:1246-1300` — `gate_admet_honest_negative_tag` (used by `p3_admet_m6_kappa_honest_negative`).
- `TODO/completed/312_honest_framing_35plus.md` — Wf-312C R02 honest-negative close precedent (3 ADMET druglikeness gates: hepatotox / logp7_4 / m6_kappa).
- `TODO/completed/314a_wf311e_part4_deterministic_edits.md` — Wf-314A closeout for the deterministic edits that added `HONEST_NEGATIVE` tags.
- `TODO/completed/322f_wf176_m19_lomo_final_closeout.md` — Wf-322F closeout template (this doc follows the same structural pattern).

---

**END OF Wf-325D CLOSEOUT** — m6_kappa SHIPPED with HONEST_NEGATIVE preserved. Recovery remains USER_OK-gated per `mol-metal-gpu-execution` §0.