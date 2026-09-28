# TODO-88 — Wf-42 Phase 3 Per-Mock Fix Plan

**Date**: 2026-09-26
**Workflows**: Wf-42 (mock → real fix plan); **Wf-43 Phase 3** (mock-script deprecation) feeds the input
**Status**: 🚀 **Active — immediate next work**
**Priority**: **HIGH** (1 critical metric — m3 Vina Dock — runs through a stub on the paper §4 row; the other 30 findings have correct dispositions but no per-finding fix ledger)
**Owner**: Hugo Lee + ultracode agents
**Parent**:
- `molmetal/reports/wf42_mock_audit/mock_inventory.md` (Wf-42 Phase 1 — 129 findings; 8 critical + 23 major + 98 minor)
- `molmetal/reports/wf42_mock_audit/real_tools_inventory.md` (Wf-42 Phase 2 — every real tool verified live on this host)
- `molmetal/reports/wf42_mock_audit/scoring_router.md` (Wf-42 Phase 2 — mock → real swap matrix)
- `molmetal/reports/wf42_mock_audit/per_mock_fix_plan.md` (Wf-42 Phase 3 — per-mock fix ledger, this plan's output)
**Sibling inputs**:
- `TODO/completed/89_wf43_routing_policy.md` (Wf-43 Phase 2 — routing policy)
- `TODO/completed/90_wf43_phase3_deprecation.md` (Wf-43 Phase 3 — 10 bench/benchmark scripts deprecation)
- `TODO/completed/82_metric_gap_filling.md` (Wf-29~Wf-34 — off-GPU scaffolding SHIP, 0/7 metric gaps closed)
- `TODO/pending/82_metric_gap_filling_plan.md` (TODO-82 — original gap-filling plan)

---

## §0 Why this plan exists

Wf-42 Phase 1 produced the mock-audit inventory (129 findings).
Wf-42 Phase 2 produced the real-tools inventory (every locally
deployed backend verified) + the scoring router (single point of
truth for the metric → real-tool mapping).
Wf-43 Phase 1 + 2 + 3 shipped the runtime enforcement
(`score_any()` + `ROUTING_POLICY` + 10 deprecated bench/benchmark
scripts).

What is **missing** is the per-finding ledger that ties each of the
**31 actionable findings** (8 critical + 23 major) to:

1. The **specific real tool** that replaces the mock (per
   `real_tools_inventory.md`).
2. The **exact fix steps** (file edits, commands, reruns).
3. The **expected output** — the JSON field / Markdown line that
   flips from `MEASURED_BUT_STUB` → `MEASURED` (or annotated-keep).

Without this ledger, the Wf-42 audit is **descriptive** but not
**prescriptive**: a future agent can't tell at a glance whether a
finding is "rerun Vina and you're done" or "deliberate CPU scaffolding
— do not touch". This plan ships that ledger.

---

## §1 Scope — the 31 actionable findings

| Bucket | Count | Disposition |
|---|---:|---|
| **Critical Vina rerun** (C1–C8) | 8 | Single `dock_10pockets_se3sb.py --use-real-vina` rerun + `wf30_dock_summarize.py` retires ALL 8 |
| **Major Vina rerun** (M1–M13) | 13 | Same rerun retires all 13 — downstream JSONs regenerate |
| **Major CPU scaffolding** (M14, M15) | 2 | Rename `_StubVNet` → `_CpuOnlyVNetShim`; annotate (already done in TODO-90) |
| **Major stub-bridge benchmarks** (M16, M17) | 2 | Verify `models/coupling_adapter_platinai5k.npz` vendored; update warning text (already done in TODO-90) |
| **Major stub-fn gated** (M18) | 1 | KEPT — already correctly gated behind `use_real_vina=False` per Wf-43 §5 |
| **Major test-scaffolding stubs** (M19, M20, M21) | 3 | KEPT + annotate (Wf-43 Phase 3 fixtures, not metric mocks) |
| **Major legacy mock-chemistry bench** (M22) | 1 | KEPT + annotate (Wf-36 P2; superseded by Wf-39 real-trajectory bench) |
| **Major paper-text pseudo-code** (M23) | 1 | KEPT + footnote (Lambda-combinator integration report) |
| **Minor** (98) | 98 | Out of scope — covered by Wf-43 `md_relaxation_stub` / `protocol_stub_adapter` / `placeholder_text` annotations |

**Net actionable after this plan**: **1 rerun + 4 small annotations + 1 NPZ check + 1 audit-script extension.**

---

## §2 Phase plan (1 day)

### Phase A — Wf-30 Vina rerun (1-2 hours on CPU + GPU 1)

| Task | Time | Skill | Output |
|---|---|---|---|
| Verify Vina + Meeko live: `python -c "import vina, meeko; print(vina.__version__, meeko.__version__)"` → expect `1.2.7 0.8.0` | 1 min | (one-liner) | Vina 1.2.7 + Meeko 0.8.0 OK |
| Re-prepare the 10-pocket receptor PDBQT staging (`molmetal/reports/wf30_pocket_prep/`) — already SHIP from Wf-30 Phase 1 | 0 min | (no-op) | 10/10 pockets READY (per `TODO/completed/82_metric_gap_filling.md` §2.2) |
| Run `dock_10pockets_se3sb.py --use-real-vina --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc --seeds 0 1 2 3 4` on GPU 1 | 60 min | mol-metal-cfm-training | `metrics/by_metric/vina_kcal_per_mol.json` rewritten with `scorer: "vina"`, `status: "MEASURED"` |
| Run `wf30_dock_summarize.py` to regenerate `best_m3_summary.json`, `m3_lift_analysis.json`, `per_pocket_metrics.json`, `m3_lift_summary.md`, `notes.md` | 5 min | (one-liner) | All 5 downstream JSON/MD files flip `scorer: "vina"` |
| Verify critical=0: `python molmetal/scripts/audit_mock_data.py --only-severity critical --quiet` | 1 min | (one-liner) | 0 critical findings |
| **Output**: 8 critical + 13 major Vina findings retired | ~75 min | — | `mock_inventory.json` critical_count=0; major Vina count=0 |

### Phase B — Coupling-bridge NPZ check (5 min)

| Task | Time | Skill | Output |
|---|---|---|---|
| Verify NPZ: `ls -la models/coupling_adapter_platinai5k.npz` → expect ~5-15 KB | 1 min | (one-liner) | NPZ present |
| If missing: run `molmetal/scripts/train_coupling_bridge_platinai.py` to materialise | 5 min | mol-metal-r18-math (T9) | NPZ materialised |
| **Output**: M16/M17 (coupling-bridge stubs) routed to the real `PlatinAI5KBridge` | 5 min | — | Both findings flip to `remediation_action: load_npz_checkpoint` PASS |

### Phase C — Audit-script extension (15 min)

| Task | Time | Skill | Output |
|---|---|---|---|
| Extend `molmetal/scripts/audit_mock_data.py` with `--include-remediation-action` flag → adds a `remediation_action` field per finding (one of `rerun_real_vina`, `load_npz_checkpoint`, `annotate_keep_*`) | 15 min | (1-line patch) | Re-run audit emits machine-readable ledger |
| **Output**: `mock_inventory.json` gains `remediation_action` per finding; downstream CI can mechanically verify each row | 15 min | — | Audit JSON extended |

### Phase D — Annotations for the 7 KEPT findings (15 min)

| Task | Time | Skill | Output |
|---|---|---|---|
| Annotate M14 (`benchmark_r18_end_to_end.py:_StubVNet`) — already done in TODO-90 | 0 min | — | (no-op) |
| Annotate M15 (`benchmark_r18_platinai.py:_StubVNet`) — already done in TODO-90 | 0 min | — | (no-op) |
| Annotate M16 (`bench_coupling_ic50.py:812`) — already done in TODO-90 | 0 min | — | (no-op) |
| Annotate M17 (`bench_lambda_cfm_e2e.py:514`) — already done in TODO-90 | 0 min | — | (no-op) |
| Annotate M18 (`dock_10pockets_se3sb.py:_stub_score`) — already Wf-43 §5 annotated | 0 min | — | (no-op) |
| Annotate M19 (`verify_fg_compat_veto.py:_MockReducer`) | 5 min | (1-line docstring) | docstring gains Wf-43 Phase 3 fixture annotation |
| Annotate M20 + M21 (`verify_mcts_replay.py:_StubMol, _StubSearch`) | 5 min | (1-line docstring) | docstring gains Wf-43 Phase 3 fixture annotation |
| Annotate M22 (`verify_variance_mcts_proof_search.py`) — already Wf-43 Phase 3 documented | 0 min | — | (no-op) |
| Annotate M23 (`phase3_integration.md:141 MockRule`) | 5 min | (footnote) | integration report gains footnote |
| **Output**: 7 KEPT findings get a `remediation_action: annotate_keep_*` field in the audit JSON | 15 min | — | All 31 actionable findings have a remediation_action |

### Phase E — Acceptance gate + close-out (10 min)

| Task | Time | Skill | Output |
|---|---|---|---|
| Run all 8 acceptance gates from `per_mock_fix_plan.md` §5 | 5 min | (one-liners) | All PASS |
| Write `TODO/completed/91_wf42_phase3_per_mock_fix.md` close-out | 5 min | (this file template) | TODO-91 close-out SHIP |
| Update `TODO/INDEX.md` + `TODO/EXECUTION_ORDER.md` with TODO-91 status | 0 min | (manual) | Navigation updated |

**Total wall-time**: ~2 hours.

---

## §3 Cross-skill strategy

| Skill | Role |
|---|---|
| `mol-metal-cfm-training` | Wf-30 Vina rerun (Phase A) |
| `mol-metal-r18-math` (T9 Bergman) | Coupling-bridge NPZ materialisation (Phase B, fallback) |
| `mol-metal-mechanism-mapping` | Audit-script extension (Phase C) |
| `mol-metal-paper-writing` | Honest-negative caveats in the close-out (Phase E) |

No NEW skill required. The 5 existing + 3 R18 + 1 GPU-execution skill cover this plan.

---

## §4 Deliverables (this plan ships)

1. **`molmetal/reports/wf42_mock_audit/per_mock_fix_plan.md`** (NEW, ~250 lines)
   - Per-mock entry for each of 31 actionable findings: problem + real tool + fix steps + expected output
   - 8 sections: scope, critical, major (×7 sub-buckets), disposition matrix, audit-script extension, acceptance gate, honest-negative caveats, files
2. **`TODO/pending/88_mock_fix_plan.md`** (NEW, this file)
   - Wf-43 input checklist + 5 phases × ~2 hours
3. **`TODO/completed/91_wf42_phase3_per_mock_fix.md`** (NEW, post-execution)
   - Phase A-D close-out with rerun evidence (real-Vina median, NPZ size, audit count)
4. **`molmetal/scripts/audit_mock_data.py`** (extended with `--include-remediation-action`)
   - `remediation_action` field per finding (one of 7 enum values)
5. **Annotations** on 7 KEPT findings (Phase D)
   - 3 docstring annotations (M19, M20, M21)
   - 1 paper-text footnote (M23)
   - 3 already done in TODO-90 (M14, M15, M16, M17, M18, M22)

---

## §5 Acceptance gate (acceptance = 5 items)

1. `per_mock_fix_plan.md` ships at `molmetal/reports/wf42_mock_audit/`
   with **31 per-finding entries** (8 critical + 23 major).
2. `audit_mock_data.py --include-remediation-action` extends the
   audit JSON with `remediation_action` field; re-run emits the
   expected 7 distinct action values.
3. After Wf-30 rerun, `audit_mock_data.py --only-severity critical`
   returns `critical_count == 0`.
4. After Wf-30 rerun, `audit_mock_data.py --only-metric m3_vina_dock
   --only-severity major` returns `major_count == 0`.
5. `molmetal/tests/test_score_any.py + test_score_any_policy.py +
   test_wf43_phase3_deprecation.py` all PASS (≥65 tests).

---

## §6 Honest-negative caveats (read first)

* **The Wf-30 rerun may still hit the meeko vs Pt/Au/Ru atom-type
  issue** for the metal-drug cohort. Per `real_tools_inventory.md`
  §1.1 the Vina 1.2.7 + Meeko 0.8.0 combo is verified for organic
  ligands; the cisplatin/oxaliplatin/auranofin cohort may need a
  custom-atom-type patch or fall back to the **organic-only cohort**
  (5/10 of the Wf-30 pockets). In that case the rerun produces a
  **real Vina median on 5 pockets** instead of 10, which is still
  better than the stub-floor -9.66 kcal/mol and ships as an honest
  partial-measurement per the Wf-43 contract.
* **7 of the 23 major findings are KEPT** (not deleted). They are
  CPU test scaffolding, legacy code, or paper-text pseudo-code.
  Deleting them would break the test suite or the paper contract.
* **The Wf-30 sweep wall-time is ~75 min on CPU + GPU 1**. If the
  ultracode queue is busy, this plan's Phase A can defer to the next
  available GPU slot; Phases B-E (NPZ check + audit extension + 7
  annotations + acceptance gate) can run on CPU alone and finish in
  ~30 min.
* **The 98 minor findings are deliberately out of scope** for this
  plan. They are `md_relaxation_stub` module-name references (real
  CPU-only force-field relaxation), `protocol_stub_adapter` shells
  (DiffDock / EquiBind / etc. — RESERVED per Wf-43 §2.2–2.3),
  `placeholder_text` paper-template markers, and `stub_identifier`
  test-scaffolding variable names. None feed paper §4.

---

## §7 Status

| Item | Status |
|---|---|
| `mock_inventory.json` (Phase 1) | ✅ SHIP — 129 findings |
| `real_tools_inventory.md` (Phase 2) | ✅ SHIP — 18 OK + 1 PARTIAL + 1 GAP + 6 RESERVED |
| `scoring_router.md` (Phase 2) | ✅ SHIP — swap matrix + 3 enforcement gates |
| `per_mock_fix_plan.md` (Phase 3) | ✅ SHIP — 31 per-finding entries |
| `TODO/pending/88_mock_fix_plan.md` (this file) | ✅ SHIP — Wf-43 input checklist |
| **TODO-91 close-out** (post-execution) | ⏳ PENDING Phase A–E execution |

**Next**: execute Phase A (Wf-30 Vina rerun) on next available GPU slot + Phases B-E on CPU now.

---

Co-Authored-By: Claude Code <noreply@anthropic.com>
