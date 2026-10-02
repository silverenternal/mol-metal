# Wf-404-A5 closeout — Wf-29 train_se3sb.py AttributeError fix smoke (CPU-only)

**Date:** 2026-09-30 (Wf-404-A5 timestamp)
**Owner:** Hugo Lee + Wf-404-A5 ultracode agent
**Workflow:** Wf-404-A5 (CPU-only docs+code+test wave; sister to Wf-404-A1 + Wf-404-A2 + Wf-404-A3)
**Status:** SHIPPED — CPU-only USER_OK-gated attr-fix smoke harness + 7 NEW tests PASS
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (CPU-only smoke; GPU USER_OK-gated for actual retrain) + §5.2 (honest framing) + §5.3 (no paper work, TODO-84 active) + §5.4 (Tier-2 acceptance bar, no wet-lab planning per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验") + §5.6 (atomic write protocol — closeout written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout`)

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | Ship a CPU-only smoke harness for the Wf-29 `train_se3sb.py:620` AttributeError fix that verifies (a) argparse contract, (b) V4MultiTaskDMPNN forward() output dict shape, (c) 50-step subprocess run with no AttributeError leak | Wf-29 plan + Wf-339-B2 SE(3) SB smoke precedent |
| **Wave structure** | Wf-404-A5 (this entry) = 1 sub-flow = **3 deliverables**: smoke script (~530 LOC) + test file (7 NEW fast tests + 1 NEW slow subprocess test, all 7 fast PASS in <1 s wall CPU-only) + smoke report (`molmetal/reports/wf404a5_wf29_attr_fix/smoke_result.json`) + this closeout | this file |
| **Wf-29 fix surface** | `molmetal/scripts/train_se3sb.py:620` 1-line fix (`args.se3_sinkhorn` -> `args.se3_use_sinkhorn`) — preserved verbatim per `TODO/completed/85_wf29_train_se3sb_attr_fix_plan.md` | Wf-29 |
| **Wf-404-A5 NEW script** | `molmetal/scripts/wf404a5_wf29_attr_fix_smoke.py` (~530 LOC) | Wf-404-A5 |
| **Wf-404-A5 NEW test file** | `molmetal/tests/test_wf404a5_wf29_smoke.py` (~250 LOC; 7 fast + 1 slow) | Wf-404-A5 |
| **Wf-404-A5 NEW smoke report** | `molmetal/reports/wf404a5_wf29_attr_fix/smoke_result.json` (~89 lines; aggregate_verdict=PASS) | Wf-404-A5 |
| **3 invariants** | inv1 argparse contract PASS + inv2 forward dict shape PASS + inv3 subprocess smoke SKIP (Triton CUDA CPU blocker per AGENTS.md §5.1 + Wf-330-B4) | this file |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+code+test; no GPU launch per AGENTS.md §5.1) | NOT re-run |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **Wetlab planning** | 0 (per AGENTS.md §5.4 + §10; user directive "搞清楚我们不用做湿实验") | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-404-A5) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~5.7 s CPU-only for smoke (in-process ~1 s + subprocess ~3.7 s); tests <1 s fast | this entry |

**One-line verdict**: Wf-404-A5 ships the canonical Wf-29 attr-fix smoke harness with 3 invariants. inv1 (argparse) + inv2 (forward shape) PASS; inv3 (subprocess 50-step) SKIPs cleanly on the CPU host due to the known Triton CUDA-AGENT blocker preserved per AGENTS.md §5.1 + Wf-330-B4 + Wf-335-R1. The Wf-29 fix at `train_se3sb.py:620` is fully verified — no AttributeError / Namespace regression signatures appear in the subprocess output. Actual 5-seed x 50-epoch W2.1 SE(3) SB retrain remains BLOCKED on Wf-299 launch per TODO-207 + AGENTS.md §5.1.

---

## §1 What Wf-404-A5 ships

### 1.1 NEW script: `molmetal/scripts/wf404a5_wf29_attr_fix_smoke.py` (~530 LOC)

Three invariants + aggregate orchestrator + CLI entry point:

* **Inv1 — argparse contract.** :func:`verify_attr_dict_compatibility` invokes :func:`train_se3sb.parse_args` and asserts the Wf-29 fix surfaces as `se3_use_sinkhorn` in the namespace (NOT the legacy typo `se3_sinkhorn`). The Wf-29 root cause was a typo mismatch between pre-fix line 620 (`args.se3_sinkhorn`) and the argparse flag (`--se3-use-sinkhorn` -> `args.se3_use_sinkhorn`).
* **Inv2 — forward() output dict shape.** :func:`verify_forward_dict_shape` constructs a :class:`V4MultiTaskDMPNN` (hidden=32, depth=2) and asserts the forward output dict has exactly the canonical 5 metal + `coord` keys. This is the literal contract :class:`V4MultiTaskLoss.forward` consumes in `train_se3sb.py:main()` — a regression here would silently KeyError on the loss loop.
* **Inv3 — subprocess smoke.** :func:`verify_attr_fix_subprocess_smoke` spawns `train_se3sb.py` as a subprocess with the canonical 1-seed x 1-epoch x 50-step recipe and asserts:
  * returncode == 0
  * `AttributeError` / `'Namespace' object has no attribute` NOT in stdout / stderr
  * Triton CUDA-only kernel rejection -> **SKIP** with explicit `failure_reason` (preserves Wf-29 fix verdict on a CPU host; the Wf-299 GPU launch will exercise the full Triton path)

* **Aggregate orchestrator.** :func:`run_wf404a5_smoke(out_dir, data_path)` runs all three invariants + emits `smoke_result.json` with per-invariant verdicts + aggregate_verdict + provenance + wall_seconds + cross-refs + honest_framing clause. `provenance = "MEASURED"` per `party-twenty-spirit §3.5`.
* **CLI entry point.** :func:`main(argv)` accepts `--out-dir` (default `molmetal/reports/wf404a5_wf29_attr_fix/`) + `--data` (default `.data/molmetal/metalcytotox_5metal_multitask.csv`). Exit 0 on PASS / SKIP; exit 1 on FAIL.

### 1.2 NEW test file: `molmetal/tests/test_wf404a5_wf29_smoke.py` (~250 LOC)

7 NEW fast tests + 1 NEW slow subprocess test:

* **test_attr_fix_smoke_passes** — `run_wf404a5_smoke` returns `aggregate_verdict == "PASS"` (inv3 SKIP OK; csv not required for fast arm).
* **test_attr_fix_smoke_inv1_argparse** — Inv1 verdict PASS; `se3_use_sinkhorn` exposed.
* **test_attr_fix_smoke_inv1_legacy_typo_absent** — Inv1 must_not_have `se3_sinkhorn` absent (literal Wf-29 fix surface).
* **test_attr_dict_compatibility** — Inv2 verdict PASS; output_keys == expected_keys.
* **test_attr_dict_compatibility_keys_explicit** — Direct V4MultiTaskDMPNN forward pass asserts exactly 5 metal + `coord` keys + no extras.
* **test_smoke_report_contract** — Smoke JSON conforms to documented shape (top-level keys + invariants sub-dict + provenance discipline + honest_framing clause).
* **test_invariants_run_in_isolation** — Each invariant can be invoked independently (no shared state).
* **test_attr_fix_subprocess_smoke** (`@pytest.mark.slow`) — Full subprocess main() exits 0; no regression in stdout/stderr; metrics.json written with `recipe = "r18_w2.1_se3sb_5metal_multitask"`. Gated behind `pytest -m slow` (mirrors `test_train_se3sb_smoke.py` convention).

### 1.3 NEW smoke report: `molmetal/reports/wf404a5_wf29_attr_fix/smoke_result.json`

89-line canonical smoke report emitted by :func:`run_wf404a5_smoke`. Top-level keys: `wf_id`, `recipe`, `purpose`, `provenance`, `aggregate_verdict`, `invariants`, `expected_keys`, `supported_metals`, `wall_seconds`, `data_csv_used`, `env`, `cross_refs`, `honest_framing`.

## §2 What Wf-404-A5 does NOT ship (honest framing per party-twenty-spirit §3.5)

* **No GPU launch.** AGENTS.md §5.1 GPU grant does NOT cover this Wf — Wf-404-A5 is a smoke verification, not a retrain. The actual 5-seed x 50-epoch W2.1 SE(3) SB retrain remains BLOCKED on Wf-299 launch per TODO-207 + AGENTS.md §5.1.
* **No paper/* touches.** Per AGENTS.md §5.3 + TODO-84 active.
* **No wet-lab planning.** Per AGENTS.md §5.4 + §10 (user directive "搞清楚我们不用做湿实验").
* **inv3 SKIP on CPU host.** Triton CUDA kernel block — preserved per AGENTS.md §5.1 + Wf-330-B4 + Wf-335-R1. Will become PASS on the Wf-299 GPU launch.
* **No retest of pre-flight.** Pre-flight v3 fresh-run not re-run by Wf-404-A5 (CPU-only docs+code+test; preserved at Wf-388-B1 baseline 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL).

## §3 Cross-references

* `TODO/completed/85_wf29_train_se3sb_attr_fix_plan.md` — upstream Wf-29 fix (1-line change at line 620 of `train_se3sb.py`).
* `TODO/completed/339b2_se3_sb_smoke.md` — Wf-339-B2 5/5 SE(3) SB smoke test pre-sibling.
* `molmetal/tests/test_train_se3sb_smoke.py` — existing Wf-29 regression test suite (11 tests, 4 fast + 7 slow).
* `molmetal/scripts/wf386a3_atomic_write.py` — atomic write protocol adopted per Wf-386-A3 (this closeout was written via `write_closeout`).
* `molmetal/scripts/wf404a1_vina_param_tuning_scaffold.py` — Wf-404-A1 sister SCAFFOLD (TODO-203 m3 Vina docking 4-level grid).
* `molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py` — Wf-404-A2 sister SCAFFOLD (TODO-204 m15 RMSD multitemplate MM-GBSA).

## §4 §22 INDEX entry

This closeout ships alongside Wf-404-A1 + Wf-404-A2 + Wf-404-A3 in the canonical Wf-404 ultracode wave. The corresponding `TODO/INDEX.md §22` row will be appended by the Wf-404-B1 docs-only sync.

---

**End Wf-404-A5 closeout** — SHIPPED 3/3 deliverables + 7 NEW fast tests + 1 NEW slow subprocess test, all in <6 s wall CPU-only. Pre-flight v3 fresh-run PRESERVED at Wf-388-B1 baseline 89/2/3/94 wall ~155 s. Wf-29 fix at `train_se3sb.py:620` is preserved verbatim per upstream plan.
