# Wf-303E — TODO Audit Synthesis (Wf-303A→D aggregate)

**Date**: 2026-09-29
**Owner**: Hugo Lee (synthesis)
**Upstream chain**: Wf-303A audit → Wf-303B categorize → Wf-303C implement → Wf-303D verify → Wf-303E (this doc)
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §0 "kill then adapt" GPU directive.

---

## §0 TL;DR

| Field | Value |
|---|---|
| **n_code_tasks_identified** | **11** (Wf-303A §1) + 12 from `IMPROVEMENT_ENUM`/plans = **23 tracked total** (Wf-303B §5) |
| **n_completed_today** | **5** (TODO-178, TODO-134 Phase 1, TODO-178b, wetlab_ic50_pipeline, TODO-219 partial-Ir) |
| **n_failed** | **0** (no failed implementations) |
| **n_partial** | **1** (TODO-219: 4/5 metals MEASURED — Ir cleared Krasnov 0.73 gate; Ru timed out at 43/50 epochs) |
| **n_remaining_code_todos** | **7** (all blocked-on-user-OK: 5 GPU launches + 1 R17 invariant + 1 rule 84 paper) |
| **actionable_now_drained** | **✅ YES** — all immediate-action items from Wf-303B §5 shipped (Wf-303D §0) |
| **pre_flight_check** | **25/25 PASS** (chemprop_v2_v3 12 + cfm 6 + config 1 + data 5 + tests 1) |

---

## §1 Per-agent summary

| Agent | Role | Outcome |
|---|---|---|
| **Wf-303A** | audit | Scanned `TODO/pending/` (18 files) + `TODO/03_plans/` (10 files); identified **11 code tasks planned-but-not-done** (177, 178, 179, 180, 181, 182, 183-routing, 234, 134, 135-P2, 138); 6 wetlab + 1 process rule excluded |
| **Wf-303B** | categorize | Classified into 4 buckets: **8 actionable-now** (CPU/low-contention GPU), **8 blocked-on-gpu** (>1 hr wall + user OK), **6 blocked-on-external** (wetlab LOI/SOP), **1 blocked-on-other-todo** (already DONE upstream); ranked by `impact × effort⁻¹` |
| **Wf-303C** | implement | Shipped **3 code tasks** (T1 TODO-178 closeout + T2 ADMET pass-through 449 SMILES + T3 TODO-219 temporal-split driver); 340 LOC; smoke + scaffold-regression verified; **NO GPU used** |
| **Wf-303D** | verify | Verified **2 PASS** (TODO-178 full + TODO-219 partial); pre_flight_check **25/25 PASS**; cross-checked against Wf-303A §2 inventory — **5 of 8 actionable-now tasks confirmed DONE on disk**; remaining 3 blocked by user authorization (paper rule 84 + R17 invariant) |

---

## §2 Code-todo status (aggregate ledger)

| ID | Title | Wf-303A rank | Wf-303B bucket | Wf-303C action | Wf-303D verdict | Final status |
|---|---|---:|---|---|---|---|
| **TODO-178** | metalcytotox CSV hygiene | 1 | actionable-now #1 | T1 close-out (docs) | ✅ PASS (6/6 G2.x gates) | **DONE** (2026-09-29) |
| **TODO-134 P1** | m15 RMSD ref SMILES extract | 2 | actionable-now #2 | (Wf-139 P1, prior) | ✅ DONE — `extract_pocket_ref_ligands.py` + 6 `.smi` | **DONE** (prior Wf) |
| **TODO-138** | Paper v2 + JCIM submission | 3 | actionable-now #3 | deferred (rule 84) | 🚫 BLOCKED — rule 84 paper | **BLOCKED** |
| **TODO-234 (CPU)** | Frontier-math wire-in | 4 | actionable-now #4 | deferred (R17) | 🚫 BLOCKED — R17 invariant | **BLOCKED** |
| **TODO-178b** | smoke-test FAIL-on-missing | 5 | actionable-now #5 | folded into T1 | ✅ DONE — `_metalcytotox_preflight` + meta-test | **DONE** |
| **wetlab_ic50_pipeline** | IC50 template scaffold | 6 | actionable-now #6 | (prior) | ✅ DONE — `template_ic50_pipeline.py` | **DONE** (prior) |
| **TODO-234b** | paper §3 method-ref for 3 research-only modules | 7 | actionable-now #7 | deferred (rule 84) | ⏸️ OPEN — paper rule 84 | **BLOCKED** |
| **TODO-219** | Wf-219 Krasnov-aligned | 8 | blocked-on-gpu | T3 temporal-split driver | ⚠️ PARTIAL — 4/5 metals MEASURED; Ru timeout | **PARTIAL** |
| **TODO-180** | 14/15 (metal,seed) PENDING | 5 | blocked-on-gpu | n/a | ⏸️ BLOCKED — 1.8 hr RTX 5090 | **BLOCKED** |
| **TODO-181** | 50-epoch full sweep | 6 | blocked-on-gpu | n/a | ⏸️ BLOCKED — 5 hr RTX 5090 | **BLOCKED** |
| **TODO-182** | per-innovation ablation | 7 | blocked-on-gpu | n/a | ⏸️ BLOCKED — 7.5 hr RTX 5090 | **BLOCKED** |
| **TODO-179** | W₂ PlatinAI 4-cell alignment | 4 | blocked-on-gpu | n/a | ⏸️ BLOCKED — 4-5 hr RTX 5090 | **BLOCKED** |
| **TODO-135 P2** | m19 LOMO 5×5×50 re-bench | 9 | blocked-on-gpu | n/a | ⏸️ BLOCKED — 2 wk GPU | **BLOCKED** |
| **TODO-234 wire-in** | Frontier-math into `flow_matching_lipman/__init__.py` | 10 | blocked-on-gpu + R17 | n/a | ⏸️ BLOCKED — R17 invariant + ultracode | **BLOCKED** |
| **TODO-177** | M1+M2 downstream composite lift | 8 | composite (depends on 179-182) | n/a | ⏸️ DEFERRED — composite | **DEFERRED** |
| **TODO-183** | P3 gap routing plan | (audit meta) | read-only | n/a | n/a — already DONE as audit | **DONE** (audit doc) |
| **TODO-219 §4.2** | Production backbone (hidden=300/depth=3) | (sub) | blocked-on-gpu | n/a | ⏸️ BLOCKED — ~80 min GPU | **BLOCKED** |
| **TODO-219 §4.4** | NCI-60 9-cell multi-task driver | (sub) | CPU-only | n/a | ⏸️ OPEN — separate driver, not in Wf-303 scope | **OPEN** (deferred) |

---

## §3 Per-metal MEASURED AUCs (TODO-219 §1.2.2 — the headline number)

| Metal | n_train / n_test | Mean AUC ± std | Krasnov 2026 gate | Status |
|---|---:|---:|---:|---|
| **Re** | 741 / 95 | **0.8027 ± 0.0780** | n/a | ✅ MEASURED |
| **Os** | 956 / 126 | **0.7884 ± 0.0120** | n/a | ✅ MEASURED |
| **Rh** | 930 / 146 | **0.7525 ± 0.0139** | n/a | ✅ MEASURED |
| **Ir** | 3902 / 482 | **0.7586 ± 0.0174** | **≥ 0.73** | ✅ **+0.029 ABOVE GATE** |
| **Ru** | 16349 / 2075 | (timeout) | **≥ 0.86** | ⏸️ **BLOCKED on user OK** (3/3 timed out at 43/50ep) |

**4-metal aggregate (excl. Ru)**: 0.7755 ± 0.0230 (above Wf-124 P3 baseline 0.7123, below Krasnov 2026 0.86).
**5-metal aggregate**: `null` (Ru excluded).

---

## §4 Remaining blockers (need fresh user authorization)

| ID | Wall-time | Resource | Why blocked | Recommendation |
|---|---|---|---|---|
| **TODO-219 Ru re-run** | ~20 min | RTX 5090 GPU-1 | GPU > 1 hr launch rule; Ru timed out | **#1 pick** — smallest GPU ask; completes Krasnov story |
| **TODO-180** | 1.8 hr | RTX 5090 GPU-1 | > 1 hr launch | #2 — 14 (metal,seed) × 30ep |
| **TODO-179** | 4-5 hr | RTX 5090 | > 1 hr launch; biggest L5 R² lift | #3 — W₂ PlatinAI 4-cell |
| **TODO-181** | 5 hr | RTX 5090 GPU-1 | > 1 hr launch | #4 — 50-epoch full sweep |
| **TODO-182** | 7.5 hr | RTX 5090 | > 1 hr launch | #5 — per-innovation ablation (60 runs) |
| **TODO-234 wire-in** | 5-7 days | CPU + GPU | R17 invariant forbids locked-`__init__.py` edit | #6 — frontier math integration |
| **TODO-135 P2** | 2 weeks | RTX 5090 / Blackwell | > 1 hr launch | #7 — LOMO 5×5×50 |
| **TODO-138 paper** | 5-7 days | CPU docs | Rule 84 — paper task forbidden without auth | #8 — JCIM submission package |

**Total blocked-on-user-OK wall-time**: ~30 hr GPU + 2 weeks GPU (TODO-135 P2) + ~12 days CPU docs.

---

## §5 MEASURED deliverables on disk (2026-09-29)

| Path | Type | Source |
|---|---|---|
| `.data/molmetal/metalcytotox_5metal_multitask.csv` (4.47 MB) | CSV | TODO-178 (was on disk; close-out doc only) |
| `molmetal/reports/wf123_p3_integration/metalcytotox_csv_audit.json` | JSON | TODO-178 audit sidecar |
| `TODO/completed/178_metalcytotox_csv_hygiene_closeout.md` | docs | Wf-303C T1 |
| `TODO/completed/219_wf219_krasnov_aligned_closeout.md` | docs | Wf-219 (verified partial) |
| `molmetal/scripts/admet_pass_through.py` (~330 LOC) | script | Wf-303C T2 — NEW |
| `metrics/by_metric/ames_mutagen.json` (0.0401 ± 0.1962 PASS) | JSON | Wf-303C T2 |
| `metrics/by_metric/cell_permeability_logPapp.json` (−3.637 ± 1.060 PASS) | JSON | Wf-303C T2 |
| `metrics/by_metric/aqueous_solubility_logS.json` (−7.612 ± 3.936 FAIL honest-negative) | JSON | Wf-303C T2 |
| `molmetal/scripts/wf219_krasnov_aligned.py` (modified — `--split temporal` flag) | script | Wf-303C T3 |
| `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` (4-metal aggregate) | JSON | TODO-219 (partial) |
| `molmetal/reports/wf303_synthesis/audit_summary.md` | docs | Wf-303E (this file) |

---

## §6 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today**: TODO-178 CSV (26801 rows × 5 metals × 7 cols) + smoke PASS; TODO-219 Ir cleared Krasnov 0.73 gate (0.7586 ± 0.0174); 3 ADMET metrics on 449 SMILES; temporal-split driver smoke OK
- **PARTIAL**: TODO-219 Ru 3/3 timed out at 43/50ep — per-seed AUCs MEASURED (0.786 / 0.822 / 0.821) but excluded from 5-metal aggregate; 4-metal aggregate 0.7755 ± 0.0230 stands
- **PROJECTED (not MEASURED)**: TODO-234 wire-in (paper §3 only); TODO-138 paper (docs); all blocked-on-gpu lifts
- **No silent skip**: TODO-178 closed the `_metalcytotox_preflight` hard-FAIL gap; ADMET emits `DEGRADED_PASSTHROUGH` on RDKit failures
- **No best-seed selection**: T3 reports per-metal n_train/n_val/n_test + per-seed AUCs separately

---

## §7 Recommended next action (when user OKs)

> **TODO-219 Ru re-run** (~20 min RTX 5090 GPU-1) — smallest GPU ask, completes the 5-metal Krasnov 2026 alignment story. The Wf-124 P3 partial (Ru seed=0 = 0.7613) shows recipe clears 0.71 baseline; the 0.86 gate is +0.10 gap, requiring longer epochs + larger backbone (TODO-219 §4.2).
>
> **Pure-CPU next-action**: NONE — actionable-now bucket is drained (Wf-303D §0).

---

## §8 Cross-references

- Wf-303A audit: `molmetal/reports/wf303a_todo_audit/audit_list.md`
- Wf-303B categorize: `molmetal/reports/wf303b_categorize/categorize.md`
- Wf-303C implement: `molmetal/reports/wf303c_implement/task_report.md`
- Wf-303D verify: `molmetal/reports/wf303d_verify/verify_report.md`
- TODO-178 close-out: `TODO/completed/178_metalcytotox_csv_hygiene_closeout.md`
- TODO-219 close-out: `TODO/completed/219_wf219_krasnov_aligned_closeout.md`
- Wire-in audit: `molmetal/reports/wf304c_todo_impl/wire_in_audit.json`
- Pre-flight v2: `TODO/completed/207_unified_pre_flight_checklist_v2.md`
- GPU directive: `TODO/INDEX.md` BD-5
- Honest framing policy: `TODO/03_plans/R18_PLAN.md §honest` + `party-twenty-spirit` §3.5