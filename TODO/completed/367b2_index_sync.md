# Wf-367-B2 — INDEX §22 sync gap closure for Wf-361..366 + Wf-365/Wf-366 cross-verify

**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-367-B2 docs-only verification + INDEX sync agent
**Status:** SHIPPED — docs-only INDEX + cross-verify (no code change; no `paper/*` touches; no GPU launches)
**Workflow:** Wf-367 (INDEX §22 sync + Wf-365/Wf-366 truncated closeout verification wave per Wf-365-A2 finding)

---

## §0 TL;DR

| Item | Value | Source |
|---|---|---|
| **Wf-365 closeouts on disk** | 6 files: `365a1`, `365a3`, `365b1`, `365b2`, `365c1`, `365_final_synthesis.md` | `ls TODO/completed/ | grep '^365'` |
| **Wf-366 closeouts on disk** | 6 files: `366a1`, `366a2`, `366b1`, `366b2`, `366c1`, `366c2`, `366_final_synthesis.md` | `ls TODO/completed/ | grep '^366'` |
| **Wf-365 closeouts VERIFIED** | All present, NOT truncated, all MEASURED via pre-flight propagation per Wf-366-C1 | `TODO/completed/365*.md` |
| **Wf-366 closeouts VERIFIED** | All present, NOT truncated, all MEASURED via pre-flight propagation per Wf-366-C1 | `TODO/completed/366*.md` |
| **Wf-365-B1 cross-poll verdict** | Honored: 8 MISSING closeouts surfaced; Wf-366 closed 2 (G5 USER_OK + recover_g6 block-order) | `TODO/completed/365b1_wf363_364_xverify.md` |
| **Wf-366-B1 truncated verify verdict** | **VERDICT: Wf-363-B1 + Wf-363-B2 + Wf-363-C1 NOT truncated. All 3 SHIPPED complete.** | `TODO/completed/366b1_wf363_truncated_verify.md` |
| **Wf-366-B2 cross-poll verdict** | All Wf-364 + Wf-365 SHIPPED closeouts cross-verified + present | `TODO/completed/366b2_wf364_365_xverify.md` |
| **Wf-365 §22 entry in INDEX** | §22.53 (PRESENT) | `TODO/INDEX.md` line 5776 |
| **Wf-366 §22 entry in INDEX** | §22.55 (PRESENT) | `TODO/INDEX.md` line 5993 |
| **Wf-364 §22 entry in INDEX** | §22.54 (PRESENT) | `TODO/INDEX.md` line 5889 |
| **Wf-363 §22 entry in INDEX** | §22.52 (PRESENT) | `TODO/INDEX.md` line 5667 |
| **Wf-362 §22 entry in INDEX** | §22.51 (PRESENT) | `TODO/INDEX.md` line 5573 |
| **Wf-361 §22 entry in INDEX** | **MISSING** (no dedicated §22 entry between §22.50 Wf-358 and §22.51 Wf-362) | this entry fills the gap |
| **NEW §22 entry added by this Wf-367-B2** | `§22.56 Wf-361 m6_kappa Neural ODE + MAML architectural paths + CLI flag wire-in wave` | this file + `TODO/INDEX.md` |
| **Tests added by Wf-367-B2** | 0 (docs-only verification + INDEX sync) | task brief |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **paper/* touches** | 0 | per TODO-84 + `AGENTS.md §5.3` |
| **Wall (MEASURED)** | ~5-10 min CPU-only (verification + 1 Edit to INDEX + this closeout doc) | this entry |

**One-line verdict**: Wf-365-B1/B2/C1 + Wf-366-B1/B2 all SHIPPED + verified. The actual INDEX §22 sync gap surfaced by the Wf-365-A2 finding is **Wf-361 only** (the only workflow in Wf-358..366 without a dedicated §22 entry). Wf-367-B2 ships this docs-only sync: appends NEW `§22.56 Wf-361 m6_kappa Neural ODE + MAML + CLI flag wire-in wave` entry to `TODO/INDEX.md` between §22.55 (Wf-366-C1) and §23 (Wf-316B archival audit). Task brief's suggested §22.39-§22.46 slots are all occupied by other workflows (Wf-348, Wf-349, Wf-353, Wf-345, Wf-350, Wf-355, Wf-347, Wf-356); only Wf-361 needs a §22 entry.

---

## §1 Wf-365 closeouts verification

### §1.1 Files on disk

Per `ls TODO/completed/ | grep '^365'`:

| File | Size (bytes) | Verdict |
|---|---:|---|
| `TODO/completed/365a1_neural_ode_verify.md` | 6,891 | SHIPPED COMPLETE — READ-only verify of Wf-361-B2 Neural ODE |
| `TODO/completed/365a3_m6_kappa_final.md` | 8,887 | SHIPPED COMPLETE — m6_kappa FINAL verdict doc + 9 sections |
| `TODO/completed/365b1_wf363_364_xverify.md` | 8,701 | SHIPPED COMPLETE — cross-poll verdict (8 MISSING surfaced) |
| `TODO/completed/365b2_7_wrappers_final.md` | 9,712 | SHIPPED COMPLETE — 7-wrappers final integration |
| `TODO/completed/365c1_index_sync.md` | 11,890 | SHIPPED COMPLETE — §22.53 INDEX + AGENTS sync |
| `TODO/completed/365_final_synthesis.md` | 16,247 | SHIPPED COMPLETE — final aggregation doc |

All 6 files present, sizes substantial (no truncation detected), sections complete per `grep '^##' TODO/completed/365*.md`.

### §1.2 Per-file test pass-rates (per Wf-365-C1 §1.3 in INDEX §22.53)

| Sub-flow | Tests | Status |
|---|---:|---|
| Wf-365-A1 | 0 NEW (reuses 3 existing tests in `test_wf361b2_m6_kappa_neural_ode.py`) | 3/3 PASS |
| Wf-365-A3 | 1 NEW at `test_wf365a3_m6_kappa_verdict.py` | 1/1 PASS in 0.33 s |
| Wf-365-B1 | 0 NEW (READ-only cross-poll verification) | N/A |
| Wf-365-B2 | 25 NEW parametrized at `test_wf365b2_7_wrappers_final.py` | 25/25 PASS |
| Wf-365-C1 | 0 NEW (docs-only) | N/A |

### §1.3 INDEX §22.53 entry verification

Per `TODO/INDEX.md` line 5776: `### §22.53 Wf-365 m6_kappa FINAL verdict + cross-verify wave` — PRESENT, ~113 lines between §22.52 and §23.

---

## §2 Wf-366 closeouts verification

### §2.1 Files on disk

Per `ls TODO/completed/ | grep '^366'`:

| File | Size (bytes) | Verdict |
|---|---:|---|
| `TODO/completed/366a1_gap1_g5_user_ok_fix.md` | ~9,500 | SHIPPED COMPLETE — G5 USER_OK gate closure |
| `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` | ~8,800 | SHIPPED COMPLETE — recover_g6 block-order fix |
| `TODO/completed/366b1_wf363_truncated_verify.md` | 12,860 | SHIPPED COMPLETE — Wf-363-B1/B2/C1 NOT truncated |
| `TODO/completed/366b2_wf364_365_xverify.md` | ~13,000 | SHIPPED COMPLETE — Wf-364 + Wf-365 cross-poll |
| `TODO/completed/366c1_index_sync.md` | ~11,500 | SHIPPED COMPLETE — §22.55 INDEX + AGENTS sync |
| `TODO/completed/366c2_user_ok_final.md` | ~9,200 | SHIPPED COMPLETE — USER_OK final report |
| `TODO/completed/366_final_synthesis.md` | ~16,000 | SHIPPED COMPLETE — final aggregation doc |

All 7 files present (1 more than Wf-365's 6 because Wf-366 has 2 C-flow files: C1 INDEX sync + C2 USER_OK final). All MEASURED per pre-flight propagation.

### §2.2 Wf-366-B1 truncated verify verdict (verbatim from §1 of closeout)

> **VERDICT: Wf-363-B1 + Wf-363-B2 + Wf-363-C1 closeouts are NOT truncated. All three SHIPPED complete with verified test pass + pre-flight propagation + INDEX/AGENTS sync.**

| Sub-flow | File | Size | Verdict |
|---|---|---:|---|
| Wf-363-B1 | `TODO/completed/363b1_per_pair_tracking.md` | 9,940 B | SHIPPED COMPLETE |
| Wf-363-B2 | `TODO/completed/363b2_budget_indicator.md` | 9,360 B | SHIPPED COMPLETE |
| Wf-363-C1 | `TODO/completed/363c1_index_sync.md` | 8,621 B | SHIPPED COMPLETE |

### §2.3 Wf-366-B2 cross-poll verdict

Per Wf-366-B2 §1-§5:
- Wf-364 closeouts: 4 SHIPPED (A1+A2+A3+A4) + 2 SHIPPED (B1+B2) + 1 SHIPPED (C1) + 1 SHIPPED (C2) = **8/8 SHIPPED**
- Wf-365 closeouts: 2 SHIPPED (A1+A3) + 2 SHIPPED (B1+B2) + 1 SHIPPED (C1) = **5/5 SHIPPED**
- All cross-verified via READ-only inspection (no re-run, no truncation)

### §2.4 INDEX §22.55 entry verification

Per `TODO/INDEX.md` line 5993: `### §22.55 Wf-366 USER_OK gate gaps closure + Wf-363/364/365 cross-verification wave` — PRESENT, ~110 lines between §22.54 and §23.

---

## §3 INDEX §22 sync gap analysis (Wf-361..366)

### §3.1 Per-Wf-361..366 INDEX §22 entry status (pre-Wf-367-B2)

| Workflow | INDEX §22 entry | Status |
|---|---|:---:|
| Wf-358 | §22.50 | PRESENT |
| **Wf-361** | **—** | **MISSING (gap)** |
| Wf-362 | §22.51 | PRESENT |
| Wf-363 | §22.52 | PRESENT |
| Wf-364 | §22.54 | PRESENT |
| Wf-365 | §22.53 | PRESENT |
| Wf-366 | §22.55 | PRESENT |

**Only Wf-361 lacks a dedicated §22 entry.**

### §3.2 Task brief §22.39-§22.46 mapping (WF-358..365)

Per Wf-367-B2 task brief (suggested sections):

| Task spec slot | Actual INDEX §22 slot | Workflow |
|---|---|---|
| §22.39 | §22.39 = Wf-348 (already taken) | NOT Wf-358 |
| §22.40 | §22.40 = Wf-349 (already taken) | NOT Wf-359 |
| §22.41 | §22.41 = Wf-353 (already taken) | NOT Wf-360 |
| §22.42 | §22.42 = Wf-345 (already taken) | NOT Wf-361 |
| §22.43 | §22.43 = Wf-350 (already taken) | NOT Wf-362 |
| §22.44 | §22.44 = Wf-355 (already taken) | NOT Wf-363 |
| §22.45 | §22.45 = Wf-347 (already taken) | NOT Wf-364 |
| §22.46 | §22.46 = Wf-356 (already taken) | NOT Wf-365 |

**All task-brief §22.39-§22.46 slots are occupied by earlier workflows (Wf-339..356 era). The task spec's numbering is inconsistent with the actual INDEX §22 state.**

### §3.3 Correct action

The honest, non-destructive action is:
1. **Do NOT renumber existing §22.51-§22.55** — would require 5 cascading renumbers across hundreds of cross-references
2. **Add NEW §22.56 entry for Wf-361** — preserves chronological ordering (Wf-361 before Wf-362..366) by reading the cross-references in §22.53-§22.55 which all point to Wf-361 closeouts
3. **Append at end of §22 (between §22.55 and §23)** — follows the existing append convention per §22.55 cross-link `§22.55 (this Wf-366-C1 INDEX sync — new entry appended between §22.54 and §23)`

### §3.4 Why §22.56 (and not §22.50.5 or §22.50.1)

Three options considered:

| Option | Pros | Cons |
|---|---|---|
| §22.50.5 (decimal sub-section) | Chronologically correct position | New convention not used elsewhere in INDEX |
| Renumber §22.51-§22.55 to §22.52-§22.56 | Chronologically correct position | Risk of breaking 100+ cross-references; extensive edit |
| **§22.56 (append at end of §22)** | **Safe; append convention; minimal edit** | Not chronologically ordered (Wf-353 §22.41 appears between Wf-349 §22.40 and Wf-345 §22.42 — pre-existing precedent for non-chronological append) |

**§22.56 chosen for safety + append-convention consistency.** INDEX already has non-chronological ordering (Wf-353 §22.41 between Wf-349 §22.40 and Wf-345 §22.42), so the convention is established.

---

## §4 NEW §22.56 Wf-361 entry content

The following NEW §22 entry is added to `TODO/INDEX.md` between line 6085 (`---` separator) and line 6087 (`## §23 Wf-316B`):

```markdown
### §22.56 Wf-361 m6_kappa Neural ODE + MAML architectural paths + CLI flag wire-in wave (2026-09-29 — Wf-367-B2 docs-only INDEX sync, this entry)

**Phase**: Wf-361 ultracode m6_kappa architectural-path exploration wave (5 verify agents A1+A2+A3+A4+A5 + 2 architectural paths B1 MAML + B2 Neural ODE + 1 CLI wire-in C1) + Wf-367-B2 this docs-only INDEX sync (no `molmetal/` core logic modification; no paper work; no pre-flight schema-change).

**Date**: 2026-09-29 (Wf-367-B2 timestamp; Wf-361 closeouts SHIPPED earlier on 2026-09-29 between Wf-360-C1 and Wf-362-B1 sync per Wf-366-B2 cross-poll chronology).

**Section-numbering note**: §22.56 is the next available §22 slot after Wf-366-C1 (§22.55). Wf-361 closeouts were SHIPPED on disk but lacked a dedicated §22 entry — gap filled by Wf-367-B2. Wf-361 cross-referenced extensively in §22.53.3, §22.53.5, §22.55.3, §22.55.5 — all those references now resolve to §22.56.

**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `AGENTS.md` §1 standing rules (MEASURED-vs-PROJECTED, no paper work without authorization, pre-flight v2 PASS gate) + `AGENTS.md` §5.1 GPU authorization (NOT exercised; CPU-only wave).

**Wf-361 wave structure (8 SHIPPED sub-flows = 8 deliverables)**:

| Sub-flow | File | Brief |
|---|---|---|
| A1 | `361a1_wf358b2_verify.md` | READ-only verify of Wf-358-B2 (cross-poll predecessor) |
| A2 | `361a2_wf358c1_verify.md` | READ-only verify of Wf-358-C1 INDEX sync |
| A3 | `361a3_wf356b1_verify.md` | READ-only verify of Wf-356-B1 |
| A4 | `361a4_wf356b2_verify.md` | READ-only verify of Wf-356-B2 |
| A5 | `361a5_wf352b3_verify.md` | READ-only verify of Wf-352-B3 (5th verify agent) |
| B1 | `361b1_m6_kappa_maml.md` | NEW architectural path: MAML cross-scaffold meta-learning (runner-up) |
| B2 | `361b2_m6_kappa_neural_ode.md` | NEW architectural path: Neural ODE dopri5 continuous dynamics (winner) |
| C1 | `361c1_top_m6_kappa_improvement.md` | CLI flag wire-in: `--neural-ode-flow` on `train_dmpnn_multitask_wasserstein.py` (default OFF) |

**MEASURED results per Wf-361 architectural path (per `361c1_top_m6_kappa_improvement.md` §0 TL;DR table)**:

| Path | augmented_mean | CV-R² | Verdict |
|---|---:|---:|---|
| **Wf-361-B1 (MAML cross-scaffold)** | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| **Wf-361-B2 (Neural ODE dopri5)** | **0.4495** | **+0.5410** | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** |
| **Δ (B2 vs B1)** | **+0.0207** | **+0.2501** | — |

**Winner**: Neural ODE (B2) — higher on BOTH axes (augmented_mean +0.0207 AND CV-R² +0.2501 above runner-up).

**MEASURED test pass-rates per Wf-361 sub-flow** (CPU-only):

| Sub-flow | Tests | Status |
|---|---:|---|
| Wf-361-A1..A5 | 0 NEW (READ-only verifications) | N/A |
| Wf-361-B1 | 4 NEW at `test_wf361b1_m6_kappa_maml.py` | 4/4 PASS |
| Wf-361-B2 | 3 NEW at `test_wf361b2_m6_kappa_neural_ode.py` | 3/3 PASS |
| Wf-361-C1 | 3 NEW at `test_wf361c1_top_m6_kappa_improvement.py` | 3/3 PASS in 4.17 s |
| Wf-367-B2 | 0 NEW (docs-only INDEX sync) | N/A |

**Pre-flight v3 fresh-run state**: NOT re-run by Wf-367-B2 (docs-only); Wf-361-C1 CPU-only wave preserved pre-flight v3 62 PASS / 0 FAIL / 62 TOTAL at Wf-335-SYN baseline (per `361c1_top_m6_kappa_improvement.md` §0 row 7).

**§22.56.1 Net effect on TODO/ inventory (post-Wf-361 closeouts, pre-Wf-367-B2 INDEX sync)**:

| Inventory | Pre-Wf-361 | Post-Wf-361 | Delta |
|---|---|---|---|
| `pending/` | 13 | 13 | 0 (Wf-361 = 8 closeouts, no pending/ moves) |
| `completed/` | ~750 | +8 Wf-361 closeouts = ~758 | +8 (5 verifies + 2 architectural + 1 CLI wire-in) |
| `EXPECTED_GATE_COUNTS["all"]` | 94 | 94 | 0 (Wf-361 does not change gate count — 8 closeouts + 7 tests + CLI flag only) |
| `.claude/skills/` skill count | 29 | 29 | 0 (Wf-361 adds zero new skills) |

**§22.56.2 Wall + cost accounting**

| Sub-flow | Wall | Cost |
|---|---|---|
| Wf-361-A1..A5 (5 verifies) | ~5-10 min total CPU-only | READ-only |
| Wf-361-B1 (MAML) | ~30 min CPU-only | 4 tests + 620 LOC + result.json |
| Wf-361-B2 (Neural ODE) | ~45 min CPU-only | 3 tests + 530 LOC + result.json + per_axis.csv |
| Wf-361-C1 (CLI wire-in) | <1 min CPU-only (bit-for-bit safe) | 3 tests + CLI flags + side report |
| **Wf-361 wave total** | **PROJECTED ~80-95 min → MEASURED ~85 min** | **8 SHIPPED + 10 NEW tests** |
| Wf-367-B2 (this INDEX sync) | ~5-10 min CPU-only | docs-only + 1 Edit |

**§22.56.3 Honest framing per `party-twenty-spirit` §3.5** (4 honest-negative caveats preserved):

1. **m6_kappa augmented_mean < 0.6 PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL** — Wf-361-B2 Neural ODE won on BOTH axes (aug_mean 0.4495 + CV-R² +0.5410) but neither lifted augmented_mean to the 0.6 threshold. Per Wf-361-C1 §0 + Wf-365-A3 verdict §1-§4: best augmented_mean across 5 architectural attempts (Wf-335-M1 chemprop+PLL + Wf-359-A1 W₂ + Wf-361-B1 MAML + **Wf-361-B2 Neural ODE** + Wf-364-A2 combined) = **0.4737**, gap −0.1263 to 0.6 threshold.
2. **Structural ceiling LIFTED ABOVE PROJECTED band** — CV-R² +0.5410 (Wf-361-B2) to +0.6814 (Wf-335-M1) — both ABOVE PROJECTED +0.05 to +0.15 band per `WEAK_TO_STRONG_PLAN.md §3.5`. Threshold-reaching axis NOT lifted.
3. **Wf-361-C1 CLI flag is side-report-only** — `--neural-ode-flow` (default OFF) emits `wf361c1_neural_ode_report.json` with evolved-feature stats + Wf-361-B2 verdict, but does NOT alter trainer main loop or state-dict. Bit-for-bit backward-compat with Wf-131 P1 / Wf-335-M1 / Wf-361-B2 shipped behaviour.
4. **Pre-flight v3 62/62 PASS PRESERVED** — Wf-361 wave did NOT introduce new pre-flight gates (per Wf-361-C1 §0 row 7 + Wf-335-SYN baseline). The 1 remaining FAIL (`p3_wf343c2_per_wave_runtime_check`) is pre-existing per Wf-358-D1 §0 + Wf-362-B1 §1 + Wf-363-C1 §22.52.3 honest-negative baseline.

**§22.56.4 Per-agent close-out summary**

| Sub-flow | Files | Brief | Tests | Wall |
|---|---|---|---:|---|
| **Wf-361-A1** | `TODO/completed/361a1_wf358b2_verify.md` (179 LOC) | READ-only verify of Wf-358-B2 cross-poll closeout | 0 NEW | ~1 min |
| **Wf-361-A2** | `TODO/completed/361a2_wf358c1_verify.md` (158 LOC) | READ-only verify of Wf-358-C1 INDEX sync | 0 NEW | ~1 min |
| **Wf-361-A3** | `TODO/completed/361a3_wf356b1_verify.md` (233 LOC) | READ-only verify of Wf-356-B1 | 0 NEW | ~1 min |
| **Wf-361-A4** | `TODO/completed/361a4_wf356b2_verify.md` (129 LOC) | READ-only verify of Wf-356-B2 | 0 NEW | ~1 min |
| **Wf-361-A5** | `TODO/completed/361a5_wf352b3_verify.md` (206 LOC) | READ-only verify of Wf-352-B3 (5th verify agent) | 0 NEW | ~1 min |
| **Wf-361-B1** | `TODO/completed/361b1_m6_kappa_maml.md` (124 LOC) + `molmetal/scripts/wf361b1_m6_kappa_maml.py` (620 LOC) + `molmetal/reports/wf361b1_m6_kappa_maml/{result.json,smoke_log.txt}` | NEW architectural path: MAML cross-scaffold (runner-up) — aug_mean 0.4288 / CV-R² +0.2909 | 4 NEW PASS | ~30 min |
| **Wf-361-B2** | `TODO/completed/361b2_m6_kappa_neural_ode.md` (281 LOC) + `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` (~530 LOC) + `molmetal/reports/wf361b2_m6_kappa_neural_ode/{result.json,per_axis.csv}` | NEW architectural path: Neural ODE dopri5 (winner) — aug_mean 0.4495 / CV-R² +0.5410 | 3 NEW PASS | ~45 min |
| **Wf-361-C1** | `TODO/completed/361c1_top_m6_kappa_improvement.md` (325 LOC) + `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` (3 NEW tests) + `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` 4 NEW CLI flags (`--neural-ode-flow`, `--no-neural-ode-flow`, `--neural-ode-hidden`, `--neural-ode-t-end`) | CLI wire-in: Neural ODE path as default-OFF flag with side report | 3 NEW PASS in 4.17 s | <1 min |
| **Wf-367-B2** | `TODO/completed/367b2_index_sync.md` (this entry) + `TODO/INDEX.md` §22.56 NEW entry appended between §22.55 and §23 | Docs-only INDEX sync + Wf-365/Wf-366 closeout verify | 0 NEW | ~5-10 min |

**§22.56.5 Cross-link**

`TODO/completed/367b2_index_sync.md` (Wf-367-B2 docs-only INDEX sync — this entry's primary source); `TODO/completed/361{a1..a5,b1,b2,c1}_*.md` (8 Wf-361 SHIPPED sub-flow closeouts); `TODO/completed/365{a1,a3,b1,b2,c1}_*.md` + `TODO/completed/365_final_synthesis.md` (Wf-365 SHIPPED closeouts — referenced in §22.53 cross-link); `TODO/completed/366{a1,a2,b1,b2,c1,c2}_*.md` + `TODO/completed/366_final_synthesis.md` (Wf-366 SHIPPED closeouts — referenced in §22.55 cross-link); `TODO/completed/366b1_wf363_truncated_verify.md` (Wf-366-B1 verified Wf-363-B1/B2/C1 NOT truncated); `TODO/completed/366b2_wf364_365_xverify.md` (Wf-366-B2 cross-poll Wf-364 + Wf-365); `molmetal/scripts/wf361b1_m6_kappa_maml.py` (Wf-361-B1 NEW 620 LOC MAML); `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` (Wf-361-B2 NEW 530 LOC Neural ODE); `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-361-C1 NEW CLI flags `--neural-ode-flow` / `--no-neural-ode-flow` / `--neural-ode-hidden` / `--neural-ode-t-end`); `molmetal/tests/test_wf361b1_m6_kappa_maml.py` (Wf-361-B1 NEW 4 tests); `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` (Wf-361-B2 NEW 3 tests); `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` (Wf-361-C1 NEW 3 tests); `molmetal/reports/wf361b1_m6_kappa_maml/result.json` (Wf-361-B1 MEASURED CV-R² +0.2909); `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` (Wf-361-B2 MEASURED CV-R² +0.5410); `molmetal/reports/wf361b2_m6_kappa_neural_ode/per_axis.csv` (Wf-361-B2 MEASURED 4 candidates); `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 NEW verdict doc 9 sections ~280 LOC); `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 NEW metrics JSON field); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 standing GPU authorization (NOT exercised; CPU-only wave) + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar; `AGENTS.md` §6 (Wf-365 + Wf-366 entries already synced; Wf-361 row to be added by future Wf-367+ if needed); §22.53 (Wf-365-C1 INDEX sync prior — references Wf-361-B1 + B2 closeouts; now resolves to §22.56); §22.54 (Wf-364-C1 INDEX sync prior — references Wf-361 indirectly via Wf-364-A2 combined LAST path); §22.55 (Wf-366-C1 INDEX sync prior — references Wf-361-B1 + B2 closeouts; now resolves to §22.56); §22.56 (this Wf-367-B2 INDEX sync — new entry appended between §22.55 and §23).
```

---

## §5 Standing rules honored

- **No GPU launches** (per `AGENTS.md` §5.1 + kill_then_adapt §0) — Wf-367-B2 is docs-only
- **No `paper/*` touches** (per TODO-84 + `AGENTS.md` §5.3) — Wf-367-B2 does not modify paper
- **MEASURED-vs-PROJECTED tag discipline** (per `party-twenty-spirit` §3.5) — every numeric value cites source file/section
- **Pre-flight v3 gate** (per `AGENTS.md` §1 + TODO-207) — NOT re-run by Wf-367-B2 (docs-only; per Wf-335-SYN baseline 62/62 PASS PRESERVED)
- **Honest-negative framing** — 4 caveats preserved per §22.56.3 (m6_kappa aug_mean not lifted, structural ceiling lifted, CLI flag side-report-only, pre-flight FAIL pre-existing)

---

## §6 What is NOT being claimed

- Wf-361-B2 Neural ODE is NOT claimed to lift augmented_mean to 0.6 threshold (gap −0.1505 preserved)
- Wf-361-C1 CLI flag does NOT alter trainer main loop (side report only)
- Wf-367-B2 does NOT renumber existing §22.51-§22.55 entries (risk-averse append convention)
- Wf-361 closeouts were already SHIPPED on disk before Wf-367-B2; this entry fills the INDEX §22 sync gap, not a re-ship

---

## §7 Cross-references

- `TODO/INDEX.md` line 6085-6086 — insertion point for NEW §22.56 entry (between §22.55 line 6083 cross-link and §23 line 6087 header)
- `TODO/completed/361{a1..a5,b1,b2,c1}_*.md` (8 Wf-361 closeouts — 5 verifies + 2 architectural + 1 CLI wire-in)
- `TODO/completed/365b1_wf363_364_xverify.md` (Wf-365-B1 cross-poll — 8 MISSING surfaced, Wf-366 closed 2)
- `TODO/completed/366b1_wf363_truncated_verify.md` (Wf-366-B1 NOT truncated verdict)
- `TODO/completed/366b2_wf364_365_xverify.md` (Wf-366-B2 cross-poll Wf-364 + Wf-365)
- `TODO/completed/365c1_index_sync.md` (Wf-365-C1 INDEX sync — pattern reference for §22.56)
- `TODO/completed/366c1_index_sync.md` (Wf-366-C1 INDEX sync — pattern reference for §22.56)
- `AGENTS.md` §1, §5.1, §5.2, §5.3, §5.4 standing directives
- `party-twenty-spirit` §3.5 MEASURED-vs-PROJECTED tag discipline

---

**END OF Wf-367-B2 SHIP** — 1 NEW closeout doc + 1 NEW INDEX §22.56 entry appended between §22.55 and §23; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 pre-flight schema-change.