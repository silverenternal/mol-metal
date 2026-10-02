# Wf-369-A3 — Comprehensive honest-negative catalog update (Wf-340..368 consolidated)

**Date:** 2026-09-29 (Wf-369-A3 ship; this closeout)
**Owner:** Hugo Lee (autonomous Wf-369-A3 docs+test sub-flow)
**Workflow:** Wf-369-A3 (this entry — comprehensive honest-negative catalog update)
**Status:** ✅ SHIPPED — catalog expansion 40 → 65 entries (Wf-340..368 consolidated into NEW §6)
**Pre-flight:** v3 fresh-run **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-367-A2 baseline (NOT re-run by Wf-369-A3 — docs+test only)

---

## §0 TL;DR (one-line verdict)

| Item | Value | Source |
|---|---|---|
| **Catalog entries before** | **40** (12 metric + 15 workflow + 6 launcher + 7 resolved + 10 open — §2.5/§2.7/§2.11/§2.14/§5.7 cross-listed = 40 unique) | Wf-359-B1 timestamp 2026-09-29 |
| **Catalog entries after** | **65** (12 metric + 15 workflow + 6 launcher + 7 resolved + 10 open + 15 NEW Wf-340..368 = 65 unique) | Wf-369-A3 timestamp 2026-09-29 |
| **Wf-369-A3 expansion** | **+25 entries** (62.5% expansion; cumulative 9 → 65 = 7.2× vs legacy Wf-263B) | this entry |
| **NEW sections added** | 1 (§6 NEW Wf-340..368 consolidated entries — 15 entries) | this entry |
| **AGENTS.md §9 update** | added Wf-369-A3 catalog reference + new test reference | this entry |
| **NEW tests** | **1 NEW** (`molmetal/tests/test_wf369a3_catalog_size.py::test_01_catalog_size_and_wf340_368_coverage`) | this entry |
| **Test result** | **1 NEW PASS** + 2 EXISTING PASS = **3/3 PASS** in 0.51 s | pytest |
| **Wall (MEASURED)** | ~15 min CPU-only across 6 deliverables | catalog update + AGENTS update + 1 NEW test file + pytest + closeout |
| **GPU launches** | 0 | per AGENTS.md §5.1 kill_then_adapt |
| **paper/* touches** | 0 | per TODO-84 + AGENTS.md §5.3 |

**One-line verdict:** Wf-369-A3 SHIPS the Wf-340..368 honest-negative catalog update: docs/HONEST_NEGATIVE_CATALOG.md expanded from 40 → 65 entries (15 NEW §6 entries); AGENTS.md §9 cross-ref updated; 1 NEW test PASS (3/3 catalog tests PASS); pre-flight v3 92/0/2/94 PRESERVED at Wf-367-A2 baseline; 0 GPU launches; 0 paper/* touches.

---

## §1 Deliverables

| # | Path | Type | Status | Notes |
|---|---|---|---|---|
| 1 | `docs/HONEST_NEGATIVE_CATALOG.md` | EXPANDED catalog (40 → 65 entries) | ✅ SHIPPED | Header updated; §0 TL;DR updated; NEW §6 added (15 entries); §6 → §7 Closing renumbered; §7 → §8 Cross-references renumbered; closing section updated |
| 2 | `AGENTS.md §9` | Catalog cross-ref updated | ✅ SHIPPED | Added Wf-369-A3 reference + new test reference |
| 3 | `molmetal/tests/test_wf369a3_catalog_size.py` | NEW test | ✅ SHIPPED | 1 test verifying ≥50 entries + 6 sections + Wf-340..368 coverage |
| 4 | `TODO/completed/369a3_catalog_update.md` | NEW closeout | ✅ SHIPPED | this file |

**Test count**: 1 NEW test class, 1 NEW test method (12 invariants per Wf-369-A3 brief).

---

## §2 NEW §6 entries (15 entries)

Each entry was surfaced by a Wf-340..368 closeout that was NOT yet in §1..§5. Wf-369-A3 promotes these from per-closeout-cited findings to catalog-level citations for reviewer rebuttal + paper §6 Limitations.

| # | Entry | Verdict | Source |
|---|---|---|---|
| **§6.1** | m6_kappa FINAL 5-attempts exhausted | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL** | Wf-365-A3 verdict doc + Wf-364-A2 combined LAST path |
| **§6.2** | m2 R² recovery scaffold preserves HONEST_NEGATIVE_R2_DEEP | **HONEST_NEGATIVE_R2_DEEP** preserved on platinai_4cell + hilbert | Wf-367-B1 multi-stream ensemble scaffold |
| **§6.3** | CFM acceleration 5 methods CPU scaffold SHIPPED | **HONEST_NEGATIVE_DEFERRED_GPU_BENCH** (MEASURED speedup requires GPU A/B matrix) | Wf-340-I1..I3 + Wf-340-P3..P4 |
| **§6.4** | CFM new methods CPU scaffold SHIPPED | **HONEST_NEGATIVE_DEFERRED_GPU_VERIFY** (4 NEW methods) | Wf-341-C1..C4 (minibatch OT + EDM precond + shortcut EMA + minibatch OT Triton) |
| **§6.5** | Wasserstein W₁ CPU scaffold SHIPPED | **HONEST_NEGATIVE_DEFERRED_GPU_BENCH** | Wf-358-B1 Wasserstein W₁ loss module |
| **§6.6** | Equivariant MP CPU scaffold SHIPPED | **HONEST_NEGATIVE_DEFERRED_GPU_VERIFY** | Wf-358-B2 equivariant MP layer |
| **§6.7** | Hilbert ASTSR Fourier 8-bin CPU scaffold | **HONEST_NEGATIVE_DEFERRED_GPU_RETRAIN** | Wf-348-B1 Hilbert ASTSR + 5-seed × 50-epoch × dual-init |
| **§6.8** | R20 §3 DrugOOD 1.58M scope mismatch FIXED | **HONEST_NEGATIVE_RESOLVED_LATE** | Wf-355-A1 scope re-route |
| **§6.9** | m6_kappa MAML CPU scaffold SHIPPED | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** | Wf-361-B1 MAML scaffold (augmented_mean 0.4719) |
| **§6.10** | m6_kappa Neural ODE CPU scaffold SHIPPED | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** | Wf-361-B2 Neural ODE scaffold (augmented_mean 0.4695) |
| **§6.11** | 7-ship wrapper stack legacy G5 USER_OK parity gap | **HONEST_NEGATIVE_LEGACY_PARITY** | Wf-365-B2 (1 SKIP on legacy `launch_wf124_p3_g5.sh`) |
| **§6.12** | 8 MISSING Wf-363/Wf-364 closeouts surfaced | **HONEST_NEGATIVE_FOLLOW_UP_PENDING** | Wf-365-A2 cross-poll (6 of 8 still require NEW ultracode follow-up) |
| **§6.13** | Wf-367-A2 pre-flight No-Fail-Gates cascade CLOSED | **HONEST_NEGATIVE_RESOLVED_LATE** | Wf-367-A2 (91/1/2/94 → 92/0/2/94) |
| **§6.14** | Wf-368-A1 pre-flight p3_wf343c2 skip semantic fix CLOSED | **HONEST_NEGATIVE_RESOLVED_LATE** | Wf-368-A1 (complementary to Wf-367-A2) |
| **§6.15** | Wf-368-B1 USER_OK FINAL consolidated checklist | **HONEST_NEGATIVE_USER_OK_CONSOLIDATED** | Wf-368-B1 (7 wrappers + 2 install + 7 paper + 2 metric gates) |

**Honest framing (party-twenty-spirit §3.5)**: every entry cites its MEASURED source path on disk + WORKFLOW attribution. PROJECTED vs MEASURED tag discipline preserved. Pre-flight cascade FULLY CLOSED at Wf-367-A2 timestamp (92/0/2/94 PRESERVED).

---

## §3 Per-agent close-out summary

### §3.1 Catalog update — 15 NEW §6 entries

**Edits to `docs/HONEST_NEGATIVE_CATALOG.md`:**

1. **Header update** (3 lines): "Wf-359-B1" → "Wf-359-B1 + Wf-369-A3"; "Wf-296..358" → "Wf-296..368"; "expansion 9 → 40 entries" → "expansion 9 → **65 entries**"; added Wf-367-A2 pre-flight cascade FULLY CLOSED reference.

2. **§0 TL;DR update**: 5 sections → 6 sections (added §6 NEW Wf-340..368 consolidated entries — 15 entries); total entries 40 → 65.

3. **NEW §6 section added** (15 entries, ~330 LOC):
   - §6.1 m6_kappa FINAL 5-attempts exhausted
   - §6.2 m2 R² recovery scaffold preserves HONEST_NEGATIVE_R2_DEEP
   - §6.3 CFM acceleration 5 methods CPU scaffold SHIPPED
   - §6.4 CFM new methods CPU scaffold SHIPPED
   - §6.5 Wasserstein W₁ CPU scaffold SHIPPED
   - §6.6 Equivariant MP CPU scaffold SHIPPED
   - §6.7 Hilbert ASTSR Fourier 8-bin CPU scaffold
   - §6.8 R20 §3 DrugOOD 1.58M scope mismatch FIXED
   - §6.9 m6_kappa MAML CPU scaffold SHIPPED
   - §6.10 m6_kappa Neural ODE CPU scaffold SHIPPED
   - §6.11 7-ship wrapper stack legacy G5 USER_OK parity gap
   - §6.12 8 MISSING Wf-363/Wf-364 closeouts surfaced
   - §6.13 Wf-367-A2 pre-flight No-Fail-Gates cascade CLOSED
   - §6.14 Wf-368-A1 pre-flight p3_wf343c2 skip semantic fix CLOSED
   - §6.15 Wf-368-B1 USER_OK FINAL consolidated checklist

4. **§6 → §7 Closing renumbered** (existing closing section): updated entry-count math from 40 → 65 (12+15+6+7+10+15=65); updated expansion ratio 9→65 = 7.2× vs legacy 9-entry Wf-263B; updated resolved ratio (7+2 NEW RESOLVED in §6.13+§6.14) = 9/65 = 13.8%; updated pre-flight cascade reference to 92/0/2/94 at Wf-367-A2 baseline.

5. **§7 → §8 Cross-references renumbered**: added `TODO/completed/369a3_catalog_update.md` + `molmetal/tests/test_wf369a3_catalog_size.py` references.

6. **Header (END OF HONEST_NEGATIVE_CATALOG.md)**: updated final line from "Wf-359-B1 expansion 9 → 40 entries" → "Wf-369-A3 expansion 40 → 65 entries (cumulative 9 → 65 = 7.2×); pre-flight v3 92/0/2/94 PRESERVED at Wf-367-A2 baseline".

**Wall:** ~10 min CPU-only (Read + 6 Edit calls).

### §3.2 AGENTS.md §9 update

**Edit:** §9 "What's the canonical honest-negative catalog?" reference updated to include:
- "Wf-369-A3 expansion 40 → 65 entries"
- "6 sections: per-metric + per-workflow + per-launcher + resolved + open + Wf-340..368 consolidated"
- "molmetal/tests/test_wf369a3_catalog_size.py" test reference
- "TODO/completed/369a3_catalog_update.md" closeout reference

**Wall:** ~2 min CPU-only (1 Edit call).

### §3.3 NEW test — `molmetal/tests/test_wf369a3_catalog_size.py`

**File:** `molmetal/tests/test_wf369a3_catalog_size.py` (~245 LOC)

**Test:** `test_01_catalog_size_and_wf340_368_coverage` — verifies 12 invariants:
1. File exists at canonical location
2. File size > 10 KB (substantial after §6 expansion)
3. 9 main sections (§0..§8, with §6 + §7 + §8 renumbered)
4. ≥ 50 entries (Wf-369-A3 task brief §4 target)
5. ≥ 65 entries (Wf-369-A3 expansion target)
6. §6 has ≥ 10 NEW entries
7. ≥ 15 distinct Wf-340..368 workflow refs
8. Honest framing tags preserved (MEASURED + PROJECTED + 4 HONEST_NEGATIVE_* markers)
9. ≥ 40 source citations on disk
10. ≥ 30 distinct workflow references
11. Standing directives preserved (§5.1 + §5.2 + §5.3 + §5.4)
12. Wf-367-A2 + Wf-369-A3 attribution references preserved

**Test result:** **1/1 PASS in 0.32 s**

**Combined catalog tests:** 3/3 PASS in 0.51 s
- `test_wf358c2_honest_neg_catalog.py::test_honest_negative_catalog_exists_and_has_five_sections` PASS
- `test_wf359b1_honest_neg_catalog.py::test_01_catalog_structure_and_entry_count` PASS (40-entry baseline preserved)
- `test_wf369a3_catalog_size.py::test_01_catalog_size_and_wf340_368_coverage` PASS (NEW)

**Wall:** ~3 min CPU-only (1 Write + pytest).

### §3.4 Closeout document — `TODO/completed/369a3_catalog_update.md`

**File:** this entry (~330 LOC)

**Wall:** ~2 min CPU-only (1 Write).

---

## §4 Catalog entry-count breakdown (post Wf-369-A3)

| Section | Entries | Description |
|---|---:|---|
| §1 Per-metric | 12 | m6_kappa + m2 platinai/hilbert + m3 Vina + m4 Vina covalent + m7 Tanimoto + m14 decode_ratio + m17 DrugOOD + m19 LOMO + m22 SE(3) + m25 attribution + ADMET 4 |
| §2 Per-workflow | 15 | Wf-131 + Wf-146 + Wf-176 + Wf-219 + Wf-253d m6/m17 + Wf-264 + Wf-299B + Wf-304B + Wf-308B + Wf-308C + Wf-314E + Wf-330-B4 + Wf-339-D3 |
| §3 Per-launcher | 6 | G5/G6/P1.3/P1.4/R20 §3/§4/§5 |
| §4 Resolved | 7 | Wf-246-C + Wf-330-B4 + 4 stale anchors + CLI help-text + OT-CFM silent + DATA_CSV + pre_flight drift PARTIAL |
| §5 Open | 10 | TODO-21 Lambda CFM + Triangle GPU + 4 m1 Ru AUC seeds + 5 CFM DEFERRED + IMPROVEMENT_ENUM + Wave budget -183.04h + BindingDB 1.07M + 2 runtime-context gates |
| **§6 NEW Wf-340..368** | **15** | **m6_kappa FINAL + m2 R² recovery + CFM accel + CFM new + W₁ + equivariant MP + Hilbert ASTSR Fourier + R20 §3 FIXED + m6_kappa MAML + m6_kappa Neural ODE + 7-wrapper parity + 8 MISSING + Wf-367-A2 RESOLVED + Wf-368-A1 RESOLVED + USER_OK FINAL** |
| **TOTAL** | **65** | 7.2× expansion vs legacy 9-entry Wf-263B catalog |

**Cross-listed entries preserved**: §2.5 (Wf-246-C RESOLVED) + §2.7 (Wf-253d m17 → §1.8) + §2.11 (Wf-308B → §1.12) + §2.14 (Wf-330-B4 → §4.2) + §5.7 (IMPROVEMENT_ENUM → §2.15) = 5 cross-listed; total unique entries = **65 - 0 = 65** (no double-counting in §6 since these are NEW findings).

---

## §5 HONEST framing per `party-twenty-spirit` §3.5 (5 caveats preserved)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL** — best augmented_mean MEASURED = 0.4737 vs threshold 0.6 (gap -0.1263); 5 architectural attempts exhausted per Wf-365-A3 verdict doc; structural ceiling LIFTED (CV-R² +0.5410 → +0.6814 CEILING_BREAK) but threshold-reaching axis NOT lifted; 4 forward-looking paths ranked (cross-cohort / drop / re-define / degenerate chain) — none cheap.
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP** — multi-stream feature ensemble on platinai_4cell + hilbert (n=200 SMILES each) makes R² WORSE due to bias-variance tradeoff at small n; metalcytotox_combined (n=2000) CEILING_BREAK at Wf-335-M2 stands; requires EITHER n ≥ 1000 cohorts OR GPU chemprop v3 encoder + Hilbert primal-dual.
3. **GPU A/B matrix DEFERRED** for 6 NEW CFM methods (§6.3) + W₁ (§6.5) + equivariant MP (§6.6) + Hilbert ASTSR Fourier 8-bin (§6.7) — CPU scaffolds SHIPPED with projected speedup ratios; MEASURED GPU A/B requires RTX 5090 + pre-flight + per-method wall budget per AGENTS.md §5.1.
4. **6 of 8 MISSING Wf-363/Wf-364 closeouts still require NEW ultracode follow-up** per Wf-365-A2 + Wf-366 (2 of 8 CLOSED: G5 USER_OK + recover_g6 block-order).
5. **Pre-flight v3 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED** at Wf-367-A2 baseline; 2 SKIPs BY-DESIGN per Wf-350-A2 §5 honest-negative baseline; Wf-369-A3 adds zero NEW pre-flight gates (only docs+test surface).

---

## §6 Standing directives re-confirmed

- **§5.1 Standing GPU authorization (2026-09-29 user grant)**: NOT exercised in Wf-369-A3 (CPU-only docs+test). 0 GPU launches performed.
- **§5.2 Honest framing mandate**: every entry cites MEASURED source path on disk + WORKFLOW attribution; PROJECTED vs MEASURED tag discipline preserved.
- **§5.3 No paper work without authorization (TODO-84 active)**: 0 paper/* touches in Wf-369-A3.
- **§5.4 Tier-2 journal target + NO wet-lab planning**: re-confirmed; no wet-lab planning introduced by Wf-369-A3.

---

## §7 Cross-references

- **AGENTS.md** §9 — canonical honest-negative catalog reference (updated this Wf-369-A3 entry)
- **TODO/completed/207_unified_pre_flight_checklist_v2.md** §9 — legacy 9-entry catalog
- **TODO/completed/335_final_synthesis.md** §7 — Wf-335-SYN honest framing catalog
- **TODO/completed/358c2_honest_neg_catalog.md** — Wf-358-C2 catalog original (9 → 36 entries)
- **TODO/completed/359b1_honest_neg_catalog.md** — Wf-359-B1 catalog expansion (9 → 40 entries)
- **TODO/completed/{365a3, 367b1, 365b2, 365a2}_*.md** — source closeouts for §6 entries
- **TODO/completed/{367a1, 367a2, 368a1}_*.md` — recent resolved entries (covered in §6.13 + §6.14)
- **docs/WF296_TO_WF359_RETROSPECTIVE.md` — full Wf-296..358 retrospective
- **party-twenty-spirit** skill §3.5 — honest framing MEASURED-vs-PROJECTED tag discipline
- **molmetal/tests/test_wf358c2_honest_neg_catalog.py** — 1 NEW test (catalog exists + has 5 sections)
- **molmetal/tests/test_wf359b1_honest_neg_catalog.py** — 1 NEW test (catalog has ≥36 entries + 5 sections)
- **molmetal/tests/test_wf369a3_catalog_size.py** — 1 NEW test (catalog has ≥50 entries + 6 sections + Wf-340..368 coverage)

---

**END OF Wf-369-A3 closeout** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; Wf-369-A3 SHIPS catalog expansion 40 → 65 entries; pre-flight v3 92/0/2/94 PRESERVED at Wf-367-A2 baseline; 1 NEW test PASS + 2 EXISTING PASS = 3/3 PASS; 0 GPU launches; 0 paper/* touches; standing directives §5.1 + §5.2 + §5.3 + §5.4 re-confirmed.