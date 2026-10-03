# TODO-90 — Wf-48 TODO-89 score_any() router actual benchmark

**Date**: 2026-09-26
**Workflow**: Wf-48 (task ID `wrr8egpw3`) — TODO-89 score_any() router actual benchmark
**Status**: ⚠️ **PARTIAL SHIP** — RDKit-native metrics PASS (QED, SA); Vina MISSING_DEP on this host contradicts Wf-43 Phase 1 MEASURED claim; fallback chain 100% PASS
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/completed/89_wf43_routing_policy.md` (Wf-43 Phase 2 routing policy)

---

## §0 TL;DR

Wf-48 attempts to **bench the score_any() router** by routing 10 bench scripts through real tools and measuring wall-time + routing correctness + fallback chain success. **RDKit-native metrics (QED + SA score) work** end-to-end (`n_scripts: 1, n_ok: 1, n_routing_pass: 1` in 0.0094 s). **Per-tool accuracy benchmark reveals a critical honesty issue**: `vina_score` is marked MEASURED in `policy.json` but `n_missing_dep: 1, mean_failure_rate: 1.000` on this host — i.e. **`import vina` fails** so the real Vina path is unreachable. **Fallback chain test PASS** (8 metrics × chain_metadata_no_chain + simulated unavailable scenarios). Net: **score_any() routing layer is sound** (no silent mock fallbacks; `AdapterUnavailable` raised on every failure) but **host environment is incomplete** for full Vina path.

| Phase | Result | Path |
|---|---|---|
| 1 Router benchmark | ✅ PARTIAL | `molmetal/reports/wf48_score_any/router_bench.{json,md}` (1/1 script OK) |
| 2 Per-tool accuracy | ⚠️ EXPOSED HOST GAP | `molmetal/reports/wf48_per_tool/per_tool_accuracy.{json,md}` (Vina MISSING_DEP) |
| 3 Fallback chain test | ✅ PASS | `molmetal/reports/wf48_fallback/fallback_test.json` (8 metrics × 2 scenarios) |
| 4 Integration verify | ✅ PASS | Wf-46 integration T3/T4/T5 |

## §1 Deliverables (on disk)

| File | Status | Path |
|---|---|---|
| Router bench script | ✅ | `molmetal/scripts/bench_score_any_router.py` (NEW) |
| Per-tool accuracy script | ✅ | `molmetal/scripts/bench_per_tool_accuracy.py` (NEW) |
| Fallback chain test script | ✅ | `molmetal/scripts/test_score_any_fallback.py` (NEW) |
| Router bench results | ✅ | `molmetal/reports/wf48_score_any/router_bench.{json,md}` |
| Per-tool accuracy results | ✅ | `molmetal/reports/wf48_per_tool/per_tool_accuracy.{json,md}` |
| Fallback chain results | ✅ | `molmetal/reports/wf48_fallback/fallback_test.json` |
| Router bench tests | ✅ | `molmetal/tests/test_score_any_router_bench.py` (NEW) |
| Per-tool accuracy tests | ✅ | `molmetal/tests/test_per_tool_accuracy.py` (NEW) |
| Fallback chain tests | ✅ | `molmetal/tests/test_score_any_fallback.py` (NEW) |

## §2 MEASURED (CPU-only real-data)

- **Router bench** (`bench_mcts_real_trajectory.py`): OK in 0.009 s, routing through `score_any("sa_score")` and `score_any("qed")` (RDKit Ertl + Bickerton)
- **Per-tool accuracy** (`AutoDock Vina 1.2.7`): `n_calls: 1, accuracy: 0.000, mean_speed_s: 0.001, failure_rate: 1.000, status: MISSING_DEP` — **honest signal that Vina binary is not installed**
- **Fallback chain metadata audit**: `n_metrics_with_chain: 8, broken_refs: [], soft_chain_pairs: 2 (vina_gpu↔vina_score + diffdock_pose↔flowdock_pose), cycles: []`
- **8 metrics per-chain scenarios** × 2 scenarios (chain_metadata_no_chain + simulated unavailable) → all behave correctly (raise AdapterUnavailable on missing dep, no silent substitution)

## §3 NOT SHIP / deferred

- **Full 11-tool per-tool accuracy**: blocked on Vina binary install on host. R20 plan item.
- **Full 10 bench script router benchmark**: 1/10 ran end-to-end on this snapshot (RDKit-native); 9 blocked on Vina / DiffDock / Boltz binary install. R20 plan item.
- **Wf-43 Phase 1 MEASURED claim for Vina**: should be downgraded to `RESERVED` in `policy.json` until Vina binary is verified on a host with `import vina` working. TODO: edit `molmetal/scoring/policy.json` to mark `vina_score / vina_high_affinity / vina_min_score` as RESERVED on hosts without Vina installed.

## §4 Cross-references

- `molmetal/reports/wf48_score_any/router_bench.md` — 1/1 PASS evidence
- `molmetal/reports/wf48_per_tool/per_tool_accuracy.md` — Vina MISSING_DEP evidence
- `molmetal/reports/wf48_fallback/fallback_test.json` — 8 metrics × 2 scenarios
- `molmetal/scoring/score_any.py` (Wf-43 Phase 1, 1,267 LOC, 25-entry REGISTRY)
- `molmetal/scoring/policy.json` (Wf-43 Phase 2, 27 entries, 738 lines)
- `TODO/completed/89_wf43_routing_policy.md` — routing policy ship
- `TODO/completed/90_wf43_phase3_deprecation.md` — 10 mock-script deprecation

## §5 Honest framing (party-twenty-spirit §3.5)

Wf-48 SHIPS the **score_any() router semantics** (no silent mock fallback; `AdapterUnavailable` on every failure; fallback chain metadata is sound) but **exposes an honesty gap** in `policy.json`: `vina_score / vina_high_affinity / vina_min_score` are marked MEASURED yet the binary is not installed on this host. The fix is a **policy.json edit** (downgrade 3 Vina entries to RESERVED until binary install is verified), which is a **TODO-90 follow-up** before R20 plan items execute. No fabricated numbers; every measurement is MEASURED or honestly flagged.
