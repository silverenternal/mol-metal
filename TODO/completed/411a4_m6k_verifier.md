# Wf-411-A4 — m6_kappa Tier-2 supplementary verifier (per-cohort empirical check)

**Date:** 2026-09-30 (Wf-411-A4 timestamp)
**Wave:** Wf-411 ultracode CPU-only docs+code+test wave (parallel coverage with Wf-412..Wf-419)
**Sub-flow:** Wf-411-A4 (m6_kappa §3 supplementary sub-table per-cohort verifier)
**Task reference:** Wf-389-A1 §3 supplementary reframe + Wf-377-A1 §4.2 top-3 promotion + Wf-365-A3 verdict doc
**Owner:** Hugo Lee + Wf-411-A4 ultracode
**Type:** DOCS + CODE + TEST (CPU-only per AGENTS.md §5.1)

**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §10

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **Verdict** | **STRUCTURAL REFINEMENT** of Wf-389-A1 §3 (per-cohort × 3-band Cohen 1988 check) | ✅ DONE (Wf-411-A4 §3) |
| **Empirical MIXED verdict** | 13/15 cohort-cells = MODERATE (|r| ≥ 0.1) on 3-band scheme | ✅ MEASURED |
| **0 STRONG bands** | No cohort-cell reaches STRONG (|r| ≥ 0.5) | ✅ MEASURED |
| **NEW script** | `molmetal/scripts/wf411a4_m6k_supp_verifier.py` ~290 LOC | ✅ SHIPPED |
| **NEW tests** | `molmetal/tests/test_wf411a4_m6k_supp.py` 2 NEW tests PASS in 2.16 s wall CPU-only | ✅ |
| **NEW result.json** | `molmetal/reports/wf411a4_m6k_supp_verifier/result.json` | ✅ DONE |
| **JSON update** | `metrics/by_metric/m6_kappa.json::wf411a4_supplementary_verifier` block added | ✅ DONE |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Pre-flight v3 status** | 89/2/3/94 PRESERVED at Wf-391-B1 baseline (NOT re-run by Wf-411-A4 docs+code+test) | ✅ PRESERVED |

**One-line verdict**: Wf-411-A4 produces a **STRUCTURAL REFINEMENT** of the Wf-389-A1 §3
supplementary sub-table recommendation by running a per-cohort × Cohen 1988 3-band
verifier on n=5460 MetalCytoToxDB labelled pIC50 split into 5 cohorts (NSCLC +
breast + melanoma + Krebs + other). The empirical finding is that 13/15 cohort-cells
reach MODERATE band (|r| ≥ 0.1) on the 3-band scheme, NOT pure WEAK as the Wf-389-A1
§3 verdict stated under a 4-band scheme. This is an **HONEST-NEGATIVE refinement** —
the supplementary sub-table is exploratory + mostly MODERATE, not pure WEAK.

---

## §1 What Wf-411-A4 does

### §1.1 Task brief

Per Wf-411-A4 task brief:
> "m6_kappa Tier-2 supplementary verifier:
> 1. On n=5460 MetalCytoToxDB labelled pIC50:
> 2. Compute per-cohort r2 (Krebs + NSCLC + breast + melanoma + other) for MolLogP + MorganDensity_r2 + Chi3v
> 3. Compute WEAK/MODERATE/STRONG band per Cohen 1988 (|r| < 0.1 / 0.1-0.5 / > 0.5)
> 4. Compute Cohen 1988 WEAK band verdict: all 3 fall in WEAK band (per Wf-389-A1)"

### §1.2 What Wf-411-A4 actually produces

Wf-411-A4 runs the verifier script + tests + JSON update. The empirical result is:

- **Cohort distribution** (n=5460): other=3558, NSCLC=1083, breast=726, Krebs=60, melanoma=33
- **Per-invariant per-cohort table** (15 cohort-cells): see §2 below
- **Cohen 1988 3-band classification**: 0 STRONG, 13 MODERATE, 2 WEAK
- **Aggregate verdict**: **MIXED** (NOT pure WEAK as the task brief assumed)

The MIXED verdict is the **STRUCTURAL REFINEMENT** of Wf-389-A1 §3. Per the task brief's
4-band Cohen 1988 scheme (where 0.1 ≤ |r| < 0.3 was called WEAK), the supplementary
sub-table is exploratory + WEAK. Per the 3-band Cohen 1988 scheme (where 0.1 ≤ |r| < 0.5
is MODERATE), the supplementary sub-table is exploratory + mostly MODERATE. Both
classifications agree that m6_kappa is correctly DROPPED from §3 main-result; they
disagree on the precise label for the supplementary sub-table.

---

## §2 Per-cohort empirical findings

### §2.1 Cohort definitions

| Cohort | Cell lines | n |
|---|---|---:|
| **NSCLC** | A549 + NCI-H460 + BEAS-2B + 16HBE | 1083 |
| **breast** | MCF-7 + MDA-MB-231 + MCF-10A | 726 |
| **melanoma** | A375 | 33 |
| **Krebs** | HL-60 + L1210 + Molt4/C8 (German "Krebs" = cancer; interpreted as leukemia) | 60 |
| **other** | everything else (HepG2, HCT-116, A2780, SK-OV-3, HEK293, MRC-5, Caco-2, etc.) | 3558 |
| **Total** | | 5460 |

### §2.2 Per-cohort × per-invariant table (Pearson r, n=5460 MetalCytoToxDB)

| Cohort | n | MolLogP | MorganDensity_r2 | Chi3v |
|---|---:|---:|---:|---:|
| NSCLC | 1083 | **+0.216** (MODERATE) | **+0.155** (MODERATE) | **+0.281** (MODERATE) |
| breast | 726 | **+0.192** (MODERATE) | **+0.036** (WEAK) | **+0.144** (MODERATE) |
| melanoma | 33 | **−0.233** (MODERATE) | **+0.020** (WEAK) | **+0.239** (MODERATE) |
| Krebs | 60 | **+0.322** (MODERATE) | **+0.229** (MODERATE) | **+0.365** (MODERATE) |
| other | 3558 | **+0.200** (MODERATE) | **+0.193** (MODERATE) | **+0.133** (MODERATE) |

### §2.3 Aggregate verdict (Cohen 1988 3-band scheme per Wf-411-A4 task brief)

| Verdict | Threshold | n cohort-cells |
|---|---|---:|
| STRONG | \|r\| ≥ 0.5 | **0** |
| MODERATE | 0.1 ≤ \|r\| < 0.5 | **13** (87%) |
| WEAK | \|r\| < 0.1 | **2** (13%) |
| **Total** | | **15** |

**Aggregate verdict = MIXED** (NOT pure WEAK as the task brief assumed).

The 2 WEAK cohort-cells are:
- **breast × MorganDensity_r2** (|r| = 0.036) — n=726, breast cancer cohort
- **melanoma × MorganDensity_r2** (|r| = 0.020) — n=33, melanoma cohort (small sample)

The 13 MODERATE cohort-cells include all MolLogP cells + most MorganDensity_r2 cells
+ all Chi3v cells.

---

## §3 Structural refinement of Wf-389-A1 §3

### §3.1 The 3-band vs 4-band Cohen 1988 difference

Wf-389-A1 §3 used the Wf-377-A1 4-band Cohen 1988 scheme:
- |r| < 0.1 → NEGLIGIBLE
- 0.1 ≤ |r| < 0.3 → WEAK
- 0.3 ≤ |r| < 0.5 → MODERATE
- |r| ≥ 0.5 → STRONG

Wf-411-A4 uses the task brief's 3-band Cohen 1988 scheme:
- |r| < 0.1 → WEAK
- 0.1 ≤ |r| < 0.5 → MODERATE
- |r| ≥ 0.5 → STRONG

These two schemes agree at the boundary (|r| ≥ 0.5 STRONG; |r| < 0.1 WEAK/NEGLIGIBLE)
but classify 0.1-0.3 differently: 4-band calls it WEAK; 3-band calls it MODERATE.

### §3.2 Implication for the supplementary sub-table

The Wf-389-A1 §3 supplementary sub-table (MolLogP + MorganDensity_r2 + Chi3v) was
labeled **WEAK band** because all 3 had |r| in 0.1-0.3 on the n=5460 cohort.

Per Wf-411-A4 3-band scheme on per-cohort breakdown:
- MolLogP: ALL 5 cohorts are MODERATE (|r| range 0.192-0.322)
- MorganDensity_r2: 3 MODERATE, 2 WEAK
- Chi3v: ALL 5 cohorts are MODERATE (|r| range 0.133-0.365)

The supplementary sub-table is **exploratory + mostly MODERATE per the 3-band scheme**,
not pure WEAK. This is an HONEST-NEGATIVE refinement of the Wf-389-A1 §3 verdict.

### §3.3 What the refinement does NOT change

- m6_kappa remains DROPPED from §3 main-result (per Wf-365-A3 verdict + Wf-377-A1 §4.2)
- The supplementary sub-table lineup is preserved (MolLogP + MorganDensity_r2 + Chi3v)
- All 3 alternatives remain statistically stronger correlates than m6_kappa on n=5460
  (per Wf-389-A1 §2: 2.46× / 2.02× / 1.90× lift factor)
- m6_kappa remains as §6 honest-limitations complementary heuristic

---

## §4 Deliverables

### §4.1 NEW script: `molmetal/scripts/wf411a4_m6k_supp_verifier.py`

~290 LOC, 5 sections:
1. Module docstring (with cohort definitions + Cohen 1988 3-band scheme)
2. `_safe` + `_morgan_density` helpers (RDKit feature extraction)
3. `_cohen1988_band(abs_r)` helper (3-band classification)
4. `load_cohort` + `parse_mols` + `compute_top3_invariants` (data pipeline)
5. `per_cohort_r2` + `per_invariant_summary` + `main` (verifier + CLI)

CLI usage:
```
python molmetal/scripts/wf411a4_m6k_supp_verifier.py
python molmetal/scripts/wf411a4_m6k_supp_verifier.py --max-samples 1000 --out-dir /tmp/wf411a4_smoke
```

### §4.2 NEW test file: `molmetal/tests/test_wf411a4_m6k_supp.py`

~210 LOC, 2 tests PASS in 2.16 s wall CPU-only:

1. `test_per_cohort_weak_band` — verify the 3-band Cohen 1988 boundary logic
   (|r| < 0.1 WEAK / 0.1-0.5 MODERATE / >= 0.5 STRONG) on synthetic cases
   + verify empirical result.json per-cohort table structure (15 cohort-cells)

2. `test_aggregate_supplementary` — verify the aggregate verdict computation
   (n_cohort_strong_total=0, n_cohort_moderate_total + n_cohort_weak_total = 15,
   aggregate_verdict in {MIXED, ALL_WEAK}, all_three_invariants_no_strong = True)

Test design honest framing:
- Tests do NOT assert aggregate_verdict == "ALL_WEAK" — that would be a false-positive
  test contradicting the empirical MIXED verdict
- Tests assert all structural invariants (no STRONG bands, 15 cohort-cells,
  cohort definitions correct, etc.) without being brittle to the WEAK vs MODERATE
  boundary classification
- Tests verify both `Wf-411-A4` `_cohen1988_band` helper and the empirical result.json

### §4.3 NEW result.json: `molmetal/reports/wf411a4_m6k_supp_verifier/result.json`

Contains:
- Cohort definitions + n_per_cohort
- Cohen 1988 3-band scheme documentation
- Per-invariant per-cohort table (3 × 5 = 15 cohort-cells with Pearson r + band)
- Aggregate counters (n_cohort_strong/moderate/weak total)
- Aggregate verdict (MIXED)
- Interpretation (per Wf-411-A4 3-band scheme)

### §4.4 JSON update: `metrics/by_metric/m6_kappa.json`

Added `wf411a4_supplementary_verifier` block with:
- Cohort definitions + n_per_cohort
- Cohen 1988 3-band scheme documentation
- Per-invariant summary (band_counts + strongest/weakest_band + verdict)
- Aggregate verdict (MIXED) + counters (0 STRONG, 13 MODERATE, 2 WEAK)
- `structural_refinement_of_wf389a1: True` (HONEST-NEGATIVE refinement)
- `wf389a1_used_4_band_cohen1988: True` + `wf411a4_used_3_band_cohen1988: True`
- Cross-refs (verifier script + result.json + test file + closeout)

---

## §5 Cross-references

### §5.1 Source verdicts (Wf-411-A4 dependencies)

| Workflow | Source path | Verdict |
|---|---|---|
| Wf-377-A1 | `TODO/completed/377a1_m6_kappa_redesign.md` | n=5460 design + top-3 promotion recommendation |
| Wf-389-A1 | `TODO/completed/389a1_m6k_supp_reframe.md` | §3 supplementary sub-table lineup |
| Wf-365-A3 | `TODO/completed/365a3_m6_kappa_final.md` | 5 architectural paths exhausted FINAL |
| Wf-381-A4 | `TODO/completed/381a4_m6kappa_finalize.md` | 8 paths exhausted FINAL FINAL |
| Wf-375-A4 | `TODO/completed/375a4_m6_kappa_reframe.md` | 4 candidate reframings |
| Wf-335-M1 | `metrics/by_metric/m35chemprop_v3_pll_cv.json` | chemprop+PLL CV-R² +0.6814 CEILING_BREAK |

### §5.2 Output files (Wf-411-A4 SHIPPED)

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/wf411a4_m6k_supp_verifier.py` | ~290 | Cohen 1988 3-band per-cohort verifier + CLI |
| `molmetal/tests/test_wf411a4_m6k_supp.py` | ~210 | 2 NEW tests PASS in 2.16 s wall |
| `molmetal/reports/wf411a4_m6k_supp_verifier/result.json` | ~3 KB | Per-cohort × per-invariant table + aggregate verdict |
| `metrics/by_metric/m6_kappa.json` | +1 block | `wf411a4_supplementary_verifier` block added |
| `TODO/completed/411a4_m6k_verifier.md` | ~150 | This closeout |

### §5.3 Standing directives re-confirmed

- §5.1 GPU grant: 0 GPU launches (CPU-only docs+code+test wave)
- §5.2 Honest framing: HONEST-NEGATIVE refinement surfaced (3-band MIXED vs 4-band WEAK)
- §5.3 No paper work: 0 `paper/*` touches per TODO-84
- §5.4 Tier-2 + no wet-lab: 0 wet-lab work per AGENTS.md §10
- §5.5 Post-wave on-disk verification: `ls -la` of all 4 SHIPPED files verified
- §5.6 Atomic write protocol: NOT used (no closeout write attempted; only JSON updates)
- §5.7 Trainer flag wire-in smoke: N/A (no trainer flag wire-in)
- §10 Wetlab deprecation FAQ: N/A (no wetlab involvement)

---

## §6 Honest framing (party-twenty-spirit §3.5)

### §6.1 What Wf-411-A4 DID彻底解决

1. **Per-cohort × per-invariant table** — 15 cohort-cells with empirical Pearson r computed
   on n=5460 MetalCytoToxDB labelled pIC50 split into 5 cohorts (NSCLC + breast + melanoma +
   Krebs + other)
2. **Cohen 1988 3-band classification** — applied per task brief verbatim (|r| < 0.1 WEAK,
   0.1-0.5 MODERATE, >= 0.5 STRONG)
3. **Aggregate verdict** — MIXED (13 MODERATE, 2 WEAK, 0 STRONG)
4. **JSON update** — `metrics/by_metric/m6_kappa.json::wf411a4_supplementary_verifier` block
5. **Tests** — 2 NEW tests PASS (verifier helper + result.json structure)
6. **STRUCTURAL REFINEMENT** of Wf-389-A1 §3 — surfaced the 3-band vs 4-band Cohen 1988
   classification difference as an HONEST-NEGATIVE

### §6.2 What Wf-411-A4 did NOT彻底解决 (preserved honest-negatives)

1. **m6_kappa gate remains HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-381-A4 +
   Wf-377-A1 + Wf-365-A3 + Wf-371-A1 + Wf-375-A4; 9 paths exhausted; best aug_mean 0.472
   vs threshold 0.6)
2. **m2 R² gate remains HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-388-A1 + Wf-377-A2 +
   Wf-380-A2; metalcytotox_combined n=2000 CEILING_BREAK + platinai_4cell + hilbert cohorts
   R² < 0 on real data)
3. **2 NEW pre-flight FAILs preserved** (loi_send_ready Tier-2 reclassification cascade +
   wf343c4_anchor_freshness_24h wf269c_report.json STALE — NOT Wf-411-A4-introduced)
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (wf343c4_yuelbond_decoder_env +
   wf343c4_wave_budget_remaining_nonneg + data_2_1_6_bindingdb_external_honest_negative)
5. **The supplementary sub-table is exploratory + mostly MODERATE per 3-band scheme** —
   this is the HONEST-NEGATIVE refinement; not pure WEAK as Wf-389-A1 §3 implied
6. **Wf-411-A4 3-band scheme differs from Wf-389-A1 4-band scheme** — both valid Cohen 1988
   classifications; the task brief's 3-band mapping is the canonical Wf-411-A4 verdict
7. **No STRONG bands on any cohort-cell** — supplementary table is exploratory, NOT headline
8. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per AGENTS.md §5.1 + Wf-368-B1 §7.2)
9. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min
   install per Wf-368-B1 §0)

### §6.3 User voice answer

Wf-411-A4 did NOT 彻底解决 the structural/metric gates (m6_kappa + m2 R² remain
HONEST_NEGATIVE FINAL). It DID 彻底解决 the Wf-389-A1 §3 supplementary sub-table
verification gap by running per-cohort × per-invariant × Cohen 1988 3-band classification
on n=5460 MetalCytoToxDB labelled pIC50, surfacing the HONEST-NEGATIVE refinement that
the supplementary sub-table is exploratory + mostly MODERATE per the 3-band scheme (NOT
pure WEAK as the Wf-389-A1 §3 verdict implied under the 4-band scheme).

NEW ultracode Wf-412+ is the recommended follow-up for any additional per-cohort
verifications + the Tier-2 §3 v4 paper update from `docs/PAPER_SECTION_3_METRIC_TABLE.md`.

---

## §7 CHANGELOG

- 2026-09-30: Initial Wf-411-A4 SHIPPED (this closeout)
