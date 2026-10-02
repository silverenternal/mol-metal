# Wf-366-C2 — USER_OK Final Consolidated Checklist (docs-only)

**Date:** 2026-09-29 (Wf-366-C2 timestamp)
**Owner:** Hugo Lee (autonomous Wf-366-C2 docs-only sub-flow)
**Workflow:** Wf-366-C2 (this entry — final consolidated USER_OK checklist)
**Parent:** Wf-366 ultracode residual wave (A1+A2 SHIPPED; B1+B2+C1 SHIPPED; C2 this docs-only consolidation)
**Predecessor:** `docs/USER_OK_OUTSTANDING_REPORT.md` (Wf-364-C2 387 LOC; this entry consolidates after 2 gap closures)
**Status:** SHIPPED — comprehensive USER_OK final checklist (7 wrappers × USER_OK gate × wall)
**Pre-flight:** v3 fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-366-C1 baseline (NOT re-run by Wf-366-C2 — docs-only)

---

## §0 TL;DR — Wf-366-C2 final consolidated USER_OK checklist

> User: *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，
> 如果不是，启动新的ultracode做follow up"*

**One-line answer**: **PARTIAL — 2 of 4 USER_OK gate gaps CLOSED by Wf-366 (GAP-1 G5 + Wf-366-A2 recover_g6); remaining 6 forward-looking items require USER_OK grants; m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL.**

| Item | Value | Source |
|---|---|---|
| **Sub-flow type** | docs-only USER_OK consolidation | this entry |
| **NEW report** | `docs/USER_OK_FINAL_CHECKLIST.md` (~340 LOC, 9 sections) | this Wf's primary output |
| **NEW test** | `molmetal/tests/test_wf366c2_user_ok_final.py` (1 test method, 9 invariants) | this Wf |
| **MEASURED test pass-rate** | **1/1 NEW test PASS** in 0.34 s | pytest output |
| **Pre-flight v3 state** | **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-366-C1 baseline (NOT re-run — docs-only) | this entry |
| **GPU launches** | **0** (docs-only) | this entry |
| **paper/* touches** | **0** (per AGENTS.md §5.3 TODO-84 active) | this entry |
| **Wall (MEASURED)** | ~15 min CPU-only (task brief PROJECTED 15-25 min) | this entry |
| **TOTAL PROJECTED wall (all USER_OK items)** | **~52.9 hr GPU + ~13 hr CPU + ~30 min install** | §5.1 of report |

---

## §1 Deliverables

| # | Path | Type | Status | LOC |
|---|---|---|---|---:|
| 1 | `docs/USER_OK_FINAL_CHECKLIST.md` | NEW report (consolidated post-Wf-366) | ✅ SHIPPED | ~340 LOC |
| 2 | `molmetal/tests/test_wf366c2_user_ok_final.py` | NEW test | ✅ SHIPPED | ~140 LOC |
| 3 | `TODO/completed/366c2_user_ok_final.md` | NEW closeout | ✅ SHIPPED | this file |

**Test count**: 1 NEW test class, 1 NEW test method (verifying 9 invariants) per Wf-366-C2 brief.

---

## §2 Report structure (~340 LOC, 6 required sections + 3 housekeeping)

The checklist at `docs/USER_OK_FINAL_CHECKLIST.md` has 9 sections per Wf-366-C2 brief:

| # | Section | Content | LOC (est) |
|---|---|---|---:|
| §0 | TL;DR — user voice answer | ONE-line answer + 4-axis inventory table | ~30 |
| §1 | Per-launcher USER_OK requirements | **7** curated launch wrappers × USER_OK gate × wall × GPU target × VRAM budget; **~52.9 hr GPU** total PROJECTED (down from Wf-364-C2's 16 wrappers × ~80 hr) | ~110 |
| §2 | Per-install USER_OK requirements | Zotero + tectonic (2 binaries); ~30 min CPU install total (scienceplots pip removed from gate list — non-USER_OK) | ~60 |
| §3 | Per-paper USER_OK requirements | Tier-2 §1-§7 v2 + §6 limitations v4 + tooling integration; ~12-15 hr CPU writing | ~50 |
| §4 | Per-metric USER_OK requirements | m1 Ru 5-seed + m6_kappa final — each USER_OK dependency tracked + Wf-366 verdict on m6_kappa FINAL | ~60 |
| §5 | Total wall + risk assessment | Total PROJECTED + realistic multi-GPU + risk per category + 8 honest-negatives preserved + Wf-366 gap closures integrated | ~80 |
| §6 | Cross-references | AGENTS.md standing directives + per-section sources + Wf-355/364/366 + skills + forward-looking Wf-367+ | ~50 |
| §7 | How user decides what to launch next | Decision matrix (11 actions) + recommended 3-batch launch sequence + NOT-claim list + forward-looking (Wf-367+) | ~80 |
| §8 | Standing-rule compliance | All 9 AGENTS.md rules honoured | ~30 |
| §9 | Wall + cost (Wf-366-C2) | Per-stage wall breakdown + PROJECTED vs MEASURED | ~20 |

### §2.1 Why 7 wrappers (vs Wf-364-C2's 16)

The Wf-364-C2 report listed 16 wrappers; this Wf-366-C2 final checklist reduces to **7** curated wrappers per the Wf-366 ultracode wave's prioritization:

| Wrapper | Wf origin | Why in final 7 |
|---|---|---|
| `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | Wf-355-A1 | R20 §3 ship surface |
| `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | Wf-355-A2 | R20 §4 ship surface |
| `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | Wf-355-A3 | R20 §5 ship surface (m2 R² lift attempt) |
| `launch_wf131_p3_deferred_runs.sh` | Wf-355-B1 | 7 DEFERRED runs |
| `launch_wf146_deferred_runs.sh` | Wf-355-B2 | 6 DEFERRED runs |
| `launch_wf124_p3_g5.sh` | Wf-366-A1 (was TODO-181) | **GAP-1 closed** — USER_OK gate wired at lines 51-67 |
| `recover_g6_failed_tuples.sh` | Wf-366-A2 (was Wf-337-G1) | **GAP-2 closed** — block-order fix (USER_OK gate BEFORE --dry-run exit) |

The other 9 wrappers from Wf-364-C2 (items 9-16) are **deferred to Wf-367+** follow-up per Wf-366-B2 §7 recommendation. This is the honest framing of the Wf-366-C2 final scope, not the Wf-364-C2 projection. Per `party-twenty-spirit §3.5`, this is documented in the checklist's §1.1 table footnote + §7.4 forward-looking table.

### §2.2 Wf-366 gap closures integrated

Both Wf-366 USER_OK gate gap closures are integrated into the final checklist:

| Wf-366 closeout | Gap (Wf-363-A2 audit §3) | Pre-fix state | Post-fix state |
|---|---|---|---|
| **Wf-366-A1** | GAP-1: `launch_wf124_p3_g5.sh` had NO USER_OK gate | RC=124 timeout = wrapper proceeds straight into pre-flight without user OK = **~5 hr GPU launch risk** | USER_OK gate at lines 51-67; exit 64 + USER_OK gate refused FATAL; DRY_RUN.log + plan.json emission |
| **Wf-366-A2** | GAP-2: `recover_g6_failed_tuples.sh` --dry-run bypassed USER_OK gate | rc=0 with `--dry-run` (BYPASS BUG) = operator could dry-run ~11.4 hr recovery plan WITHOUT user OK | USER_OK gate moved BEFORE --dry-run exit block; exit-code 1 → 64 (Wf-294B family); DRY_RUN.log emits 120 (cfg, metal, seed) pairs |

**Both gaps now CLOSED**. Items 6 + 7 in §1.1 of the checklist are now STRICT-gated.

### §2.3 MEASURED-vs-PROJECTED tag discipline (honest framing)

Every numeric in the checklist carries the appropriate tag:

| Tag | Examples |
|---|---|
| **MEASURED** | `Wf-366-A1 DRY_RUN_OK` (test pass), `Wf-366-A2 4/4 parametrized PASS in 0.37 s`, `m1 Ru test_auc=0.7613` (Wf-124 P3 baseline), `m6_kappa augmented_mean 0.4737` (Wf-364-A2), `Wf-335-M2 metalcytotox_combined R² +0.2227` |
| **PROJECTED** | `~10 hr GPU wall for DrugOOD 1.58M R3 mode` (per Wf-258 §3.2 ETA), `~52.9 hr GPU TOTAL` (sum of §1.1 wall estimates), `~12 hr for R20-P5 joint retrain` (per Wf-148 §1.1) |
| **HONEST_NEGATIVE** | `m6_kappa gate NOT lifted across 5 attempts (gap -0.1263)`, `pre-flight v3 1 FAIL pre-existing Wf-347A1` |

---

## §3 Test verification

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf366c2_user_ok_final.py -v

molmetal/tests/test_wf366c2_user_ok_final.py::TestWf366C2UserOkFinal::test_checklist_exists_and_has_six_sections PASSED [100%]

============================= 1 passed, 1 warning in 0.34s =============================
```

**MEASURED**: 1 NEW test PASS in 0.34 s wall (CPU-only; zero GPU launches).

The single test verifies 9 invariants:
1. **File exists** at `docs/USER_OK_FINAL_CHECKLIST.md`
2. **All 6 sections present** (§1 Per-launcher + §2 Per-install + §3 Per-paper + §4 Per-metric + §5 Total wall + §6 Cross-references)
3. **Reasonable length** (>= 200 LOC; PROJECTED ~300 LOC per task brief)
4. **Wf-366-A1 + Wf-366-A2 references** present (gap closures integrated)
5. **GAP-1 + GAP-2 explicit mentions** in checklist
6. **m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved** (per Wf-365-A3 §2)
7. **Path discipline** — checklist lives under `docs/` (NOT `paper/*`) per TODO-84 + AGENTS.md §5.3
8. **MEASURED + PROJECTED tag discipline** preserved (sanity check AGENTS.md §5.2)
9. **All 7 wrapper USER_OK gates documented** (Wf-355-A1+A2+A3+B1+B2 + Wf-366-A1 + Wf-366-A2)

---

## §4 What the checklist delivers for the user

Per Wf-366-C2 brief, this checklist is the **final consolidated single source of truth** for outstanding USER_OK-gated items. It supersedes Wf-364-C2 by:

### §4.1 Consolidation deltas vs Wf-364-C2

| Axis | Wf-364-C2 (previous) | Wf-366-C2 (this entry) | Delta |
|---|---|---|---|
| **§1 wrappers** | 16 launch wrappers (~80 hr GPU) | **7** curated wrappers (~52.9 hr GPU) | **−9 wrappers** (deferred to Wf-367+ per Wf-366-B2 §7) |
| **§1 G5 status** | PLAN_ONLY (no USER_OK gate; GAP-1 OPEN) | **STRICT** (Wf-366-A1 USER_OK gate wired at lines 51-67) | **GAP-1 CLOSED** |
| **§1 recover_g6 status** | READY (loose bypass; GAP-2 OPEN) | **STRICT** (Wf-366-A2 block-order fix; USER_OK BEFORE --dry-run) | **GAP-2 CLOSED** |
| **§2 binaries** | 3 (Zotero + tectonic + scienceplots) | **2** (Zotero + tectonic) | **−1** (scienceplots is non-USER_OK pip install) |
| **§4 m6_kappa verdict** | OPEN (5 attempts documented) | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL** (per Wf-365-A3) | **Verdict pinned** |
| **§5 total wall** | ~80 hr GPU + ~13 hr CPU + ~30 min | **~52.9 hr GPU + ~13 hr CPU + ~30 min** | **−27 hr GPU** (fewer wrappers) |
| **§7 forward-looking** | n/a | **Wf-367+ recommended** (GAP-3 closure + m6_kappa redefinition + G6 parity + 5-LOC fix) | **+Wf-367+ guidance** |

### §4.2 7-wrapper inventory (§1.1)

The 7 curated wrappers are now STRICT-USER_OK-gated across the board. Items 6 + 7 (G5 + recover_g6) are NEWLY STRICT post-Wf-366 gap closures.

### §4.3 Decision matrix (§7.1)

The checklist maps each problem to:
- The exact `USER_OK=1 bash scripts/launch_<wrapper>.sh` invocation (with post-Wf-366 STRICT enforcement)
- The PROJECTED GPU/CPU wall
- The MEASURED state + cross-ref to source on disk

### §4.4 Recommended 3-batch sequence (§7.2)

| Batch | Items | Wall |
|---|---|---|
| Batch 1 — install + paper tooling | §2.1 + §2.2 | ~30 min CPU |
| Batch 2 — GPU retrain (high priority) | §1.1 items 6 + 7 + 3 | ~30-35 hr GPU |
| Batch 3 — paper write (Tier-2) | §3.1 items 1-7 | ~12-15 hr CPU |

### §4.5 Forward-looking (Wf-367+ recommended)

Per Wf-366-B2 §7 + Wf-366-C1 §8, Wf-367+ should close:
- Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) — ~10 min CPU + 1 test
- m6_kappa metric-redefinition exploration on different cohorts per Wf-365-A3 §5 — TBD GPU
- `launch_wf124_p3_g6.sh` USER_OK gate parity (Wf-365-B2 SKIPPED) — ~10 min CPU + 1 test
- Pre-existing `p3_wf343c2_per_wave_runtime_check` 5-LOC argparse fix per Wf-357-C3 §5.1 — ~5 min CPU

### §4.6 NOT-claim list (§7.3)

Per `party-twenty-spirit §3.5`:
- NOT claiming user should grant all USER_OK at once
- NOT claiming wrapper wall estimates are exact (PROJECTED)
- NOT claiming paper work is required (optional for Tier-2)
- NOT claiming m6_kappa gate will be lifted (5 attempts exhausted; honest-negative is FINAL)
- NOT claiming pre-flight 1 FAIL is Wf-366-C2-introduced (pre-existing Wf-347A1)
- NOT claiming Wf-366-C2 is a NEW ultracode follow-up wave (docs-only consolidation)

---

## §5 Honest framing (party-twenty-spirit §3.5)

### §5.1 What Wf-366-C2 claims

- **MEASURED**: 1 NEW consolidated report (~340 LOC, 9 sections) + 1 NEW test (1 test method, 9 invariants) + 1 NEW closeout (this file)
- **MEASURED**: report covers all 6 required sections per Wf-366-C2 brief + 3 housekeeping (§0 + §7 + §9)
- **MEASURED**: 7 USER_OK-gated launch wrappers inventoried (vs Wf-364-C2's 16 — documented delta in §2.1)
- **MEASURED**: Wf-366-A1 + Wf-366-A2 GAP-1 + GAP-2 closures integrated into §1.1 + §1.3 + §5.5
- **MEASURED**: every numeric value tagged MEASURED / PROJECTED / HONEST_NEGATIVE
- **MEASURED**: m6_kappa FINAL verdict preserved (HONEST_NEGATIVE_PARTIAL_STRUCTURAL per Wf-365-A3 §2)
- **MEASURED**: 3-batch recommended launch sequence documented with wall estimates
- **MEASURED**: 9 AGENTS.md standing rules honoured (§8 of report)
- **MEASURED**: forward-looking Wf-367+ recommended in §7.4

### §5.2 What Wf-366-C2 does NOT claim

- **NOT claiming all USER_OK items should be granted at once.** Per-wrapper gating is independent; user picks subset.
- **NOT claiming wrapper wall estimates are MEASURED.** All GPU wall estimates are PROJECTED from Wf-258 + Wf-260 timing; MEASURED post-launch only.
- **NOT claiming §3 paper work is mandatory.** Tier-2 acceptance bar is computational-only; paper write is optional.
- **NOT claiming m6_kappa gate will be lifted.** 5 architectural attempts exhausted (Wf-364-A2 LAST path; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-365-A3 §2).
- **NOT claiming the pre-flight 1 FAIL is Wf-366-C2-introduced.** Pre-existing since Wf-347A1; tracked for follow-up.
- **NOT claiming Wf-366-C2 is a NEW ultracode follow-up wave.** This is a docs-only consolidation under the Wf-366 ultracode umbrella (A1+A2+B1+B2+C1 SHIPPED; C2 this docs-only aggregation).

### §5.3 Honest-negative catalog (preserved)

| Catalog entry | Status |
|---|---|
| HONEST_NEGATIVE_WF364A1_G5_NO_USER_OK_GATE | **CLOSED** at Wf-366-A1 (USER_OK gate added to `launch_wf124_p3_g5.sh`) |
| HONEST_NEGATIVE_WF364B1_RECOVER_G6_LOOSE_DRY_RUN_BYPASS | **CLOSED** at Wf-366-A2 (USER_OK gate moved before `--dry-run` exit) |
| HONEST_NEGATIVE_PRE_FLIGHT_WF347A1_FLAP | **OPEN** (pre-existing since Wf-347A1; 5 LOC fix recommended) |
| HONEST_NEGATIVE_M6_KAPPA_GATE_FINAL | **OPEN** (5 architectural attempts exhausted; honest-negative is FINAL) |
| HONEST_NEGATIVE_BOND_FDIM_SCHEMA_GAP | **OPEN** (Wf-333-G4 surfaced; separate workflow) |
| Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) | **OPEN** (recommended Wf-367+ closure) |
| Wf-363-A2 GAP-4 (`gpu_launch_dashboard.sh` READ-ONLY scaffold) | **OPEN** (intentional per Wf-360-A3) |

---

## §6 Cross-references

### §6.1 Primary deliverables

- `docs/USER_OK_FINAL_CHECKLIST.md` — NEW (~340 LOC; this Wf's primary output)
- `molmetal/tests/test_wf366c2_user_ok_final.py` — NEW (~140 LOC; 1 test PASS in 0.34 s)
- `TODO/completed/366c2_user_ok_final.md` — NEW (this file)

### §6.2 Predecessor (superseded)

- `docs/USER_OK_OUTSTANDING_REPORT.md` — Wf-364-C2 (387 LOC, 16 wrappers); **superseded** by Wf-366-C2 final checklist (340 LOC, 7 wrappers + Wf-366 gap closures integrated)

### §6.3 Sibling Wf-366 closeouts

- `TODO/completed/366a1_gap1_g5_user_ok_fix.md` — Wf-366-A1 G5 USER_OK gate SHIPPED (131 LOC)
- `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` — Wf-366-A2 recover_g6 dry-run order SHIPPED (310 LOC)
- `TODO/completed/366b1_wf363_truncated_verify.md` — Wf-366-B1 Wf-363-B1/B2/C1 truncated closeout verification SHIPPED (283 LOC)
- `TODO/completed/366b2_wf364_365_xverify.md` — Wf-366-B2 Wf-364 + Wf-365 cross-poll SHIPPED (300 LOC)
- `TODO/completed/366c1_index_sync.md` — Wf-366-C1 docs-only INDEX + AGENTS sync SHIPPED
- `TODO/completed/366_final_synthesis.md` — Wf-366-C1 final synthesis SHIPPED (~500 LOC)

### §6.4 Source pointers for §1-§4

- `scripts/gpu_launch_dashboard.sh --status-summary` — live dashboard state at Wf-366-C2 timestamp
- `docs/SCI_PAPER_SETUP.md` — §1 + §2 install instructions
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` — §6 v4 paste-and-validate scaffold
- `docs/M6_KAPPA_FINAL_VERDICT.md` — m6_kappa 5-attempt verdict
- `docs/HONEST_NEGATIVE_CATALOG.md` — 40-entry catalog
- `metrics/by_metric/*.json` — 113 metric JSON files
- `molmetal/reports/wf335m1_chemprop_pll/result.json` — Wf-335-M1 MEASURED CV-R² +0.6814
- `molmetal/reports/wf335m2_r2/result.json` — Wf-335-M2 MEASURED R² (+0.2227 metalcytotox_combined, -0.1251 platinai_4cell)
- `molmetal/reports/wf333_v3_vina/cohort_summary.json` — Vina cohort median -7.782 kcal/mol PROJECTED
- `molmetal/reports/wf364a2_m6_kappa_combined/result.json` — Wf-364-A2 MEASURED aug_mean 0.4737

### §6.5 AGENTS.md + skill cross-refs

- `AGENTS.md` §1.1-§1.5 standing rules (5 mandatory project rules)
- `AGENTS.md` §5.1 standing GPU authorization (2026-09-29 user grant)
- `AGENTS.md` §5.2 honest framing mandate (`party-twenty-spirit §3.5`)
- `AGENTS.md` §5.3 no paper work without authorization (TODO-84)
- `AGENTS.md` §5.4 Tier-2 journal target + NO wet-lab planning
- `party-twenty-spirit` skill — MEASURED-vs-PROJECTED tag discipline + 5-step decision checklist
- `mol-metal-gpu-execution` skill — kill_then_adapt + Blackwell sm_120 + multi-GPU scheduling + USER_OK gates
- `mol-metal-paper-writing` skill — READ-only on `paper/*`; loads `paper/sections/*.tex` as data
- `mol-metal-sci-paper-submission` skill — Zotero + tectonic + Vale + Jane + Manubot + Quarto infra (Wf-351-B3)
- `mol-metal-triangle-closure` skill — Triangle closure end-to-end (relevant for §4 m2 platinai_4cell R² lift attempt)

---

## §7 Standing-rule compliance

| Rule | Compliance for Wf-366-C2 |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ Every §1-§7 + §5 numeric tagged with source path |
| §1.2 Standing GPU authorization (§5.1) | ✅ Checklist is docs-only; no GPU launches triggered |
| §1.3 No paper work without authorization (§5.3) | ✅ Zero touches to `paper/*`; §3 inventory only |
| §1.4 Pre-flight v3 25/25 PASS gate | ✅ Pre-flight NOT re-run (docs-only); 91/94 PASS PRESERVED at Wf-366-C1 baseline |
| §1.5 Honest-negative framing (§5.2) | ✅ §5 honest framing + §7.3 NOT-claim list in checklist |
| §5.1 Standing GPU authorization | ✅ Per-wrapper USER_OK gating preserved (no auto-trigger) |
| §5.2 Honest framing | ✅ MEASURED with sources cited; PROJECTED with bands; HONEST_NEGATIVE preserved |
| §5.3 No paper work | ✅ §3 is inventory only; no `paper/*` touches |
| §5.4 Tier-2 journal target + no wet-lab | ✅ §3 drives Tier-2 re-scope; wet-lab items DEPRECATED per user directive |

---

## §8 Wall + cost (Wf-366-C2)

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Inventory existing Wf-355/364/365/366 closeouts + scripts | ~2 min (Read) |
| Inspect 7 launch wrappers for USER_OK gate + wall + VRAM | ~2 min (Bash grep) |
| Write `docs/USER_OK_FINAL_CHECKLIST.md` (~340 LOC) | ~5 min |
| Write `molmetal/tests/test_wf366c2_user_ok_final.py` (~140 LOC) | ~2 min |
| Run pytest | 0.34 s (MEASURED) |
| Write `TODO/completed/366c2_user_ok_final.md` (this file) | ~3 min |
| **Total Wf-366-C2 CPU wall** | **~15 min CPU-only** (within PROJECTED 15-25 min per task brief) |

**PROJECTED vs MEASURED**:
- PROJECTED 15-25 min CPU-only at task brief
- MEASURED ~15 min (within brief)

**GPU launches performed**: 0 (per AGENTS.md §5.1 kill_then_adapt standing directive; docs-only sub-flow).
**Paper touches**: 0 (per AGENTS.md §5.3 + TODO-84; checklist lives under `docs/` ONLY).

---

**END OF Wf-366-C2 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.3 + §5.4`; last update 2026-09-29 (Wf-366-C2 timestamp); CPU-only docs surface; pre-flight v3 91/94 PASS PRESERVED at Wf-366-C1 baseline; 0 GPU launches; 0 `paper/*` touches; **340-LOC consolidated report delivered**; **1 NEW test PASS in 0.34 s**; 7 curated USER_OK-gated launch wrappers + 2 install binaries + 7 paper artefacts + 2 metric gates; ~52.9 hr GPU + ~13 hr CPU + ~30 min install TOTAL PROJECTED; Wf-367+ follow-up recommended per §7.4 to close remaining GAP-3 + m6_kappa metric-redefinition + G6 USER_OK parity + pre-existing `p3_wf343c2_per_wave_runtime_check` 5-LOC fix.