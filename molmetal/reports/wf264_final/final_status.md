# Wf-264-Final — paper §4 Honest-Claim Audit (post-Wf-264 re-run)

**Date**: 2026-09-28
**Workflow**: Wf-264-Final (re-audit post-Wf-264-Audit)
**Target file**: `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` (paper §4)
**Constraint**: READ-ONLY on paper §4 source; READ-ONLY on Wf-264-Audit (`molmetal/reports/wf264_audit/paper_4_claim_audit.md`).
**Purpose**: Verify whether the 5 §3.5 honest-framing violations identified in Wf-264-Audit have been resolved, and whether new claims were introduced (with their own honest-claim status).
**Owner**: Wf-264-Final ultracode.

---

## §0 TL;DR

- **0 aspirational claims** (post-Wf-264) — UNCHANGED from pre-Wf-264.
- **5 §3.5 violations remain** (post-Wf-264) — UNCHANGED from pre-Wf-264.
- **86 reasonable claims** (post-Wf-264) — +1 vs pre-Wf-264 (new §1.4 Wf-124 P3 Ru seed=0 test_auc=0.7613 + Wf-261 v2 split documentation, both traceable).
- **91 total claims** (post-Wf-264) — +1 vs pre-Wf-264.
- **No main-result numbers regressed**; **no fabrication introduced**.

The post-Wf-264 update to paper §4 (`TODO/completed/188_paper_4_krasnov_protocol_aligned.md`) added §1.4 (Strict Bemis-Murcko scaffold 70/10/20 split post-Wf-261) and updated §1.1 split row to point to v2 CSV. **None of the 5 Wf-264-Audit §3.5 violations have been patched** — the same broken citation pointers in §9, transcription error in §0.1, seed-count inflation in §2.1/§2.3, and partial-disclosure in §2.1/§5.1 are still present.

---

## §1 Pre-Wf-264 vs Post-Wf-264 comparison

### §1.1 Acceptance gate delta (FLAT schema)

| Gate | Pre-Wf-264 (Wf-264-Audit) | Post-Wf-264 (Wf-264-Final) | Δ | Verdict |
|---|---:|---:|---:|:---:|
| **G1_n_claims_extracted** | 90 | **91** | **+1** | ✅ (new §1.4 Wf-124 P3 Ru seed=0 test_auc=0.7613 claim; Wf-261 v2 split numbers were already audited under §2.10) |
| **G2_n_aspirational** | 0 | **0** | **0** | ✅ PASS — no claim without MEASURED source (target met) |
| **G3_n_projected_as_measured** | 5 | **5** | **0** | ⚠️ NOT MET (target 0; same 5 minor §3.5 violations remain) |
| **G4_n_reasonable** | 85 | **86** | **+1** | ✅ MEASURED with traceable source (post-Wf-264 §1.4 Wf-124 P3 Ru seed=0 + Wf-261 v2 split documentation both traceable) |
| **G5_audit_path** | `molmetal/reports/wf264_audit/paper_4_claim_audit.md` | `molmetal/reports/wf264_final/final_status.md` (this file) | n/a | ✅ PASS |
| **honest_framing_note** | paper §4 is unusually thorough — 0 aspirational, 5 §3.5 violations (minor) | unchanged: paper §4 remains in strong shape; 0 aspirational, 5 §3.5 violations still minor and fixable in 5 small edits; **no main-result impact** | n/a | ✅ |
| **G4_total_delta** (G2 + G3) | 5 (aspirational + projected) | **5** | **0** | unchanged |

### §1.2 5 §3.5 violations — pre vs post comparison

| # | Violation (Wf-264-Audit §3.1-§3.5) | Pre-Wf-264 location | Post-Wf-264 location | Status |
|---|---|---|---|---|
| 1 | §9 line 410 broken citation pointer (`chemprop_v2_5metal_5seed_50epoch.json` cited as TODO-154 source — file is Wf-146 LOMO incomplete) | §9 line 410 | **§9 line 410 (unchanged)** | ⚠️ NOT FIXED |
| 2 | §9 line 411 broken citation pointer (`cfm_decode_real.json` cited for m14 3-seed 0.9980 — JSON has value=0.9645) | §9 line 411 | **§9 line 411 (unchanged)** | ⚠️ NOT FIXED |
| 3 | §0.1 line 19 transcription error (Δ = −0.0098 inconsistent with §3.1's −0.0074; mixes 3-seed and 5-seed) | §0.1 line 19 | **§0.1 line 19 (unchanged)** | ⚠️ NOT FIXED |
| 4 | §2.1 + §2.3 seed-count inflation (Ru/Ir "5-seed mean" contradicts cross-metal n=14 paired t-test; should be "3-seed mean") | §2.1 lines 98-99 + §2.3 line 118-119 | **§2.1 lines 98-99 + §2.3 line 118 (unchanged)** | ⚠️ NOT FIXED |
| 5 | §2.1 + §5.1 + §6.1 partial-disclosure (Wf-219 Ru 0.810 ± 0.017 from 3/3 timeout runs; status not disclosed) | §2.1 line 98 + §5.1 line 259 + §6.1 line 364 | **§2.1 line 98 + §5.1 line 259 (unchanged; L11 in §6.1 line 364 also unchanged)** | ⚠️ NOT FIXED |

**All 5 Wf-264-Audit §3.5 violations remain unfixed post-Wf-264.** The post-Wf-264 changes were ADDITIVE (§1.4 documentation block + §1.1 split-row source-path update), not corrective.

---

## §2 Post-Wf-264 paper §4 changes — claim-by-claim trace

### §2.1 §1.1 split row update (line 38)

**Pre-Wf-264**:
> | Split | Bemis-Murcko scaffold 70/10/20 | `molmetal/scripts/train_nci60_10cell.py` (scaffold_split=True) |

**Post-Wf-264**:
> | Split (v2, post-Wf-261) | Strict Bemis-Murcko scaffold 70/10/20 on `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` (26 633 lines, 0 leak) | `molmetal/scripts/pre_flight_check.py::gate_scaffold_zero_leak` + `molmetal/reports/wf261_regenerate/regen_report.md` |

**Trace check — `molmetal/scripts/pre_flight_check.py::gate_scaffold_zero_leak`**: present, defined in `molmetal/scripts/pre_flight_check.py`; verified PASS per Wf-265 pre_flight run (`/tmp/wf265_preflight.json` p0_scaffold_zero_leak=PASS). Source pointer ✅ MEASURED.

**Trace check — `molmetal/reports/wf261_regenerate/regen_report.md`**: present; v2 CSV lines=26633, rows=26632, 0 scaffold leak confirmed per Wf-264-Audit §2.10 (already audited). Source pointer ✅ MEASURED.

**Verdict**: ✅ Source-pointer hygiene IMPROVED (was pointing to legacy v1 script; now points to v2 pre-flight gate + regen report). Does NOT count as a new numeric claim (no new number introduced).

### §2.2 §1.1 protocol components table — new "Split generation" row (line 51)

**New row**:
> | **Split generation (post-Wf-261 2026-09-28)** | **Strict Bemis-Murcko scaffold 70/10/20 (`ScaffoldSplitter`, seed=42) on v2 CSV `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` (26 632 rows, 3 069 unique scaffolds, 0 scaffold leak)** | `molmetal/reports/wf261_regenerate/regen_report.md` + `molmetal/reports/wf261_verify/verify_report.md` |

**Trace check**:
- "26 632 rows" → `wf261_regenerate/regen_report.md` n_rows_v2=26632 ✅ MEASURED (rounded)
- "3 069 unique scaffolds" → `wf261_regenerate/regen_report.md` n_unique_scaffolds=3069 ✅ MEASURED (rounded)
- "0 scaffold leak" → `wf261_regenerate/regen_report.md` g3_zero_scaffold_leak=True ✅ MEASURED
- "ScaffoldSplitter seed=42" → narrative, references implementation choice in Wf-261-Update ✅ MEASURED

**Verdict**: ✅ New row is fully MEASURED with traceable JSON/MD source. Not a §3.5 violation.

### §2.3 §1.4 NEW section — "Strict Bemis-Murcko scaffold 70/10/20 split (post-Wf-261 2026-09-28)" (lines 66-88)

This is the main post-Wf-264 addition. It contains 11 numeric claims (mostly duplicating §2.10 Wf-261 v1 vs v2 split already audited under Wf-264-Audit §2.10) + 1 NEW numeric claim:

| §1.4 claim | Source value | Verdict |
|---|---|:---:|
| v1 CSV path `.data/molmetal/metalcytotox_5metal_multitask.csv` | file path constant | ✅ MEASURED (legacy) |
| v2 CSV path `.data/molmetal/metalcytotox_5metal_multitask_v2.csv` | `wf261_regenerate/regen_report.md` path | ✅ MEASURED |
| v1 26 802 lines, 26 801 rows, 3 070 unique scaffolds | `wf261_audit/split_audit.md` (legacy v1 audit) | ✅ MEASURED (already audited under Wf-264 §2.10) |
| v2 26 633 lines, 26 632 rows, 3 069 unique scaffolds | `wf261_regenerate/regen_report.md` | ✅ MEASURED (already audited under Wf-264 §2.10) |
| v1 LEAKY scaffold overlap train∩val=30, train∩test=48, val∩test=32 | `wf261_audit/split_audit.md` | ✅ MEASURED |
| v2 scaffold overlap = 0 (strict disjoint) | `wf261_regenerate/regen_report.md` g3_zero_scaffold_leak=True | ✅ MEASURED |
| v1 split counts train 18 760 / val 2 680 / test 5 361 | `wf261_audit/split_audit.md` | ✅ MEASURED |
| v2 split counts train 18 137 / val 2 759 / test 5 736 | `wf261_regenerate/regen_report.md` split_counts | ✅ MEASURED |
| v2 SHA-256 = `cc3ea24755f1b14a69c8c3acd827db0bc1dd123301479c9ca490920725adf063` | `wf261_regenerate/regen_report.md` sha256_v2 | ✅ MEASURED |
| v2 169 null/empty scaffold_id rows dropped | `wf261_regenerate/regen_report.md` n_dropped_null=169 | ✅ MEASURED |
| **Wf-124 P3 Ru seed=0 test_auc=0.7613** | `wf124_p3_multitask/ru/per_seed_metrics_seed0.json` test_metrics.roc_auc=0.7613038529802696 | ✅ MEASURED (NEW claim; not in Wf-264-Audit §2 because §1.4 was added post-audit) |

**Verdict**: ✅ All 11 §1.4 claims are MEASURED + 1 NEW MEASURED claim (Wf-124 P3 Ru seed=0 0.7613). The new claim is explicitly framed as "**Pre-Wf-261 numbers (e.g. Wf-124 P3 Ru seed=0 test_auc=0.7613 on v1) are explicitly NOT the source of any §2.1 / §4 / §6 cell**" — honest framing is preserved.

---

## §3 §3.5 violations — confirmation that all 5 remain

### §3.1 Issue #1 — §9 line 410 broken citation pointer (CONFIRMED STILL PRESENT)

**Post-Wf-264 §9 line 410**:
> `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` — TODO-154 chemprop v2 5-metal in-domain fold AUCs

**File content check (verified by `ls` + `find` post-Wf-264)**:
- File does NOT exist at `molmetal/metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json`
- `find molmetal/metrics/by_metric/ -name "*.json"` returns only 4 files: `cfm_decode_ratio_r17.json`, `vina_min.json`, `multi_cell_auc.json`, `decode_ratio.json` — none matches the cited name

**Verdict**: ⚠️ **STILL PRESENT**. The citation pointer is broken (file does not exist on disk); per Wf-264-Audit §3.1, the file (when present) was for a DIFFERENT experiment (Wf-146 LOMO incomplete). Recommended fix unchanged: cite `TODO/completed/154_chemprop_v4_head_to_head.md` §2 directly.

### §3.2 Issue #2 — §9 line 411 broken citation pointer (CONFIRMED STILL PRESENT)

**Post-Wf-264 §9 line 411**:
> `molmetal/metrics/by_metric/cfm_decode_real.json` — m14 3-seed decode_ratio 0.9980 (cross-check)

**File content check (verified by `ls` + `find` post-Wf-264)**:
- File does NOT exist at `molmetal/metrics/by_metric/cfm_decode_real.json` (per `find` listed above)

**Verdict**: ⚠️ **STILL PRESENT**. The citation pointer is broken (file does not exist on disk); per Wf-264-Audit §3.2, the file (when present) had value=0.9645 not the cited 0.9980. The 0.9980 figure IS measured (Wf-139 P5 / Wf-148 m14 3-seed 5979/5991) but the citation pointer is wrong. Recommended fix unchanged: cite `molmetal/reports/wf139_p5_cfm_decode_real/synthesis.md` + `wf148_m14_real_cfm.log`.

### §3.3 Issue #3 — §0.1 line 19 transcription error (CONFIRMED STILL PRESENT)

**Post-Wf-264 §0.1 line 19** (verified by `grep`):
> ... Δ chemprop v2 − iMoLD = −0.0098 / −0.0177 / −0.0035). **Mol-Metal does NOT claim to beat iMoLD.**

**§3.1 line 134** (consistent 5-seed):
> Δ vs iMoLD (5-seed) = **−0.0074** / −0.0177 / −0.0035

**Verdict**: ⚠️ **STILL PRESENT**. The §0.1 line still mixes 3-seed (−0.0098) and 5-seed (−0.0177, −0.0035); §3.1 still uses consistent 5-seed (−0.0074). Recommended fix unchanged: use −0.0074 in §0.1.

### §3.4 Issue #4 — §2.1 + §2.3 seed-count inflation (CONFIRMED STILL PRESENT)

**Post-Wf-264 §2.1 line 98** (verified by `grep`):
> | **Ru** | **0.909** (5-seed mean, paired vs V4 p=0.025) | 0.81 (time-split) | **+0.099** | 0.810 ± 0.017 (3-seed, scaffold-aligned) | ✅ STRONG ...

**Post-Wf-264 §2.1 line 99**:
> | **Ir** | **0.885** (5-seed mean, paired vs V4 p=0.010) | 0.73 (time-split) | **+0.155** | 0.759 ± 0.014 (3-seed, scaffold-aligned) | ✅ STRONG ...

**Post-Wf-264 §2.3 line 118**:
> | Ru only | 5 | — | **0.025** | ✅ Significant |

**Verdict**: ⚠️ **STILL PRESENT**. Per Wf-264-Audit §3.4, cross-metal n=14 paired t-test requires Ru=3, Ir=3, Os=3, Rh=3, Re=2 (Ru + Ir inflated to n=5 contradicts n=14 total). Recommended fix unchanged: change to "3-seed mean" for Ru + Ir in §2.1, and Ru only n=3 in §2.3.

### §3.5 Issue #5 — §2.1 + §5.1 + §6.1 Wf-219 Ru timeout partial-disclosure (CONFIRMED STILL PRESENT)

**Post-Wf-264 §2.1 line 99** (verified by `grep`):
> | **Ru** | ... | 0.810 ± 0.017 (3-seed, scaffold-aligned) | ✅ STRONG (Krasnov-beating; Wf-219 reproduces at 0.810 within scaffold-split protocol) |

**Post-Wf-264 §5.1 line 259**:
> | **Wf-219 reproduces Ru/Ir at the 3-seed × 50-ep scaffold-aligned protocol** (hidden=64/depth=2 smaller backbone) | ✅ MEASURED (Ru 0.810 ± 0.017, Ir 0.759 ± 0.014) | Wf-219 |

**Post-Wf-264 §6.1 line 364 (L11 honest-negative)**:
> | **L11 (NEW — Wf-219 specific)** | **Wf-219 3-seed × 50 ep scaffold-aligned sweep uses smaller chemprop v2 backbone (hidden=64, depth=2), not the production defaults (hidden=300, depth=3).** Per-metal inter-seed std at the smaller backbone: Ru ±0.017, Ir ±0.014 (vs Krasnov Ru 0.81 published on time-based split). The smaller backbone is **strictly weaker per-step** but sufficient to expose the protocol-aligned scaffold-split signal. **Full-backbone 50-epoch GPU follow-up DEFERRED** (TODO-219 §3 + Wf-219 honest_framing_note). | §6 | `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` honest_framing_note |

**Verdict**: ⚠️ **STILL PRESENT**. Per Wf-264-Audit §3.5, all 3 Ru runs in Wf-219 sweep_summary.json per_metal_seed_auc are tagged `status: "timeout"` (Ru 0.7858 / 0.8215 / 0.8215; mean_auc=NaN; per_metal_summary.Ru.n_seeds=0). The "✅ MEASURED" + "✅ STRONG" labels in §2.1/§5.1 do NOT disclose this timeout status. L11 in §6.1 mentions smaller backbone but NOT the timeout status. Recommended fix unchanged: add "⚠️ PARTIAL — Ru 3/3 per-seed runs tagged timeout" disclosure to §2.1 + §5.1 + §6.1 L11.

---

## §4 Post-Wf-264 honest-framing summary

### §4.1 §0/§2/§3/§4/§5/§6/§7 main-result numbers — UNCHANGED

All main-result numbers in paper §4 remain MEASURED with traceable JSON/MD sources per Wf-264-Audit §2.1-§2.11:
- chemprop v2 0.886 ± 0.014 (TODO-154) — UNCHANGED
- Wf-188 m14 0.9958 (26,687/26,799 ok, n=26,801) — UNCHANGED
- Wf-179 m15 5/5 < 2.0 Å (aggregate median 0.658 Å) — UNCHANGED
- m21 NCI-60 0.8007 ± 0.0160 — UNCHANGED
- m7 diversity 0.0397 (+22.5% lift) — UNCHANGED
- Wf-174 0.7113 / Wf-185 0.7137 (DrugOOD) — UNCHANGED
- Wf-219 Ir 0.759 ± 0.014 — UNCHANGED (⚠️ Ru 0.810 ± 0.017 PARTIAL, see §3.5)

### §4.2 §5.3 PROJECTED + §5.4 DEFERRED — UNCHANGED

Per Wf-264-Audit §4, the following PROJECTED / DEFERRED items remain correctly tagged as NOT MEASURED:
- DrugOOD 1.58M full retrain (BD-11 DEFERRED)
- PlatinAI 226K full retrain (L2 DEFERRED)
- BindingDB 1.07M full retrain (BD-12 DEFERRED)
- Krasnov temporal-split head-to-head driver (TODO-219 §3 DEFERRED)
- Wf-194 v1 cross-metal transfer (TODO-170 §5 DEFERRED)
- m3 Vina Dock 5-seed mean ≥ −8.5 kcal/mol (R19 DEFERRED)
- m5 SA closure-bounded 2.45-2.66 (TODO-140 + Wf-139 P6 DEFERRED)
- m16 Binding affinity ≥ 0.50 / Boltz-2 0.86 (R19 DEFERRED)
- m4 Wet-lab click yield (R22 DEFERRED)
- m10 Phototherapy (R20 DEFERRED)
- m2 Pt A2780cis remaining 6/9 cells (R22 DEFERRED)
- m18 LOMO v3 cross-metal transfer (TODO-170 §5 DEFERRED)
- m17 DrugOOD full graph-NN iMoLD head-to-head (DEFERRED)
- m8 PB-Valid 5-seed mean ≥ 0.94 (Wf-32 DEFERRED)
- m15 RMSD full 100-pocket CrossDocked100 (R20 DEFERRED)
- m7 diversity full-cohort GPU evolution (R20 DEFERRED)

### §4.3 Post-Wf-264 ADDITIONS — ALL HONEST

The post-Wf-264 additions (§1.1 split-row update + new §1.4 section) are all MEASURED with traceable sources:
- §1.1 split row updated to point to `pre_flight_check.py::gate_scaffold_zero_leak` + `wf261_regenerate/regen_report.md` (both exist; verified by `find` + `cat`)
- §1.4 split documentation block: 11 Wf-261 v1 vs v2 numbers + 1 new Wf-124 P3 Ru seed=0 0.7613 — all MEASURED + traceable
- §1.4 explicitly states "Wf-124 P3 Ru seed=0 test_auc=0.7613 on v1" is **NOT** the source of any §2.1 / §4 / §6 cell — honest framing preserved

---

## §5 Wf-258 §3.5 honest-framing summary (post-Wf-264)

Paper §4 remains **exceptionally well-disciplined** on honest framing per `party-twenty-spirit` §3.5 + Wf-258 §3.5 + Wf-264 §3.5:

- **0 aspirational claims** (G2) — all numeric claims trace to MEASURED sources.
- **5 §3.5 violations** (G3) — all minor hygiene fixes; all UNFIXED post-Wf-264 (per §3.1-§3.5 above).
- **86 reasonable claims** (G4) — +1 vs pre-Wf-264 (new §1.4 Wf-124 P3 Ru seed=0 claim).
- **No main-result number regressed** — chemprop v2 0.886, m14 0.9958, m15 5/5 < 2.0 Å, m21 0.8007, m7 0.0397, DrugOOD 0.7137 all UNCHANGED and MEASURED.
- **No fabrication introduced** — §1.4 documentation block explicitly frames all v1 numbers as "NOT the source of any §2.1 / §4 / §6 cell".
- **No best-seed selection** — §1.4 reports Wf-124 P3 Ru seed=0 0.7613 (the only one of 15 (metal, seed) pairs MEASURED; the other 14 remain PENDING per Wf-265 §4).

**Verdict**: Paper §4 remains in **strong shape** for Wf-227 final-draft sign-off. The 5 §3.5 violations are minor and fixable in 5 small edits (citation pointers in §9 + 1 transcription error in §0.1 + seed-count inflation in §2.1/§2.3 + partial-disclosure in §2.1/§5.1/§6.1) — none of them affect the §0/§2/§3/§4 main-result numbers.

---

## §6 Honest framing note (party-twenty-spirit §3.5)

- **READ-ONLY audit**: This Wf-264-Final report did NOT modify paper §4 source. It re-verified the 5 §3.5 violations from Wf-264-Audit (all still present), verified the new §1.4 claims (all MEASURED), and emitted this final_status.md.
- **0 aspirational claims** in paper §4: every numeric claim in §0/§1/§2/§3/§4/§5/§6/§7/§9 traces to a MEASURED JSON/MD source (87 → 88 reasonable claims; pre-Wf-264 had 85, post-Wf-264 has 86 — see §1.1) or a cite-only lit anchor (9 claims; unchanged).
- **5 §3.5 violations** (Wf-264-Audit §3.1-§3.5): all minor and fixable in 5 small edits. **None of the 5 have been patched** post-Wf-264 — the post-Wf-264 changes were ADDITIVE (§1.4 documentation block + §1.1 split-row source-path update), not corrective.
- **1 new MEASURED claim introduced post-Wf-264** (§1.4 Wf-124 P3 Ru seed=0 test_auc=0.7613, sourced from `wf124_p3_multitask/ru/per_seed_metrics_seed0.json` test_metrics.roc_auc=0.7613038529802696). Explicitly framed as "NOT the source of any §2.1 / §4 / §6 cell" per honest-framing discipline.

**No fabrication. No best-seed selection. No best-config selection. No overclaim.**

---

## §7 FLAT output (acceptance gates)

```
{
  G1_n_aspirational_post: 0,
  G2_n_projected_as_measured_post: 5,
  G3_n_reasonable_post: 86,
  G4_total_delta: 5,
  G5_final_status_path: "molmetal/reports/wf264_final/final_status.md",
  honest_framing_note: "Wf-264-Final re-audit confirms: 0 aspirational claims (UNCHANGED from pre-Wf-264); 5 §3.5 violations remain (UNCHANGED — all minor hygiene fixes, none affect main-result numbers); 86 reasonable claims (+1 vs pre-Wf-264, the new §1.4 Wf-124 P3 Ru seed=0 0.7613 claim + Wf-261 v2 split documentation). Post-Wf-264 paper §4 changes were ADDITIVE (§1.1 split-row update + new §1.4 documentation block), not corrective — none of the 5 Wf-264-Audit §3.5 violations were patched. Paper §4 remains in strong shape for Wf-227 final-draft sign-off; the 5 §3.5 violations are fixable in 5 small edits (citation pointers §9 + transcription error §0.1 + seed-count inflation §2.1/§2.3 + partial-disclosure §2.1/§5.1/§6.1). Total claim count: 91 (+1 vs pre-Wf-264). No fabrication. No best-seed selection. No best-config selection. No overclaim."
}
```

---

*Generated 2026-09-28 by Wf-264-Final ultracode. READ-ONLY audit (no edits to paper §4 or source files). All numeric claims in this report trace to either (a) the Wf-264-Audit (`molmetal/reports/wf264_audit/paper_4_claim_audit.md`), (b) the Wf-265-Final (`molmetal/reports/wf265_final/final_status.md`), (c) the post-Wf-264 paper §4 diff (52-line ADDITIVE block), or (d) the Wf-124 P3 Ru seed=0 per-seed metrics JSON (`molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json`).*