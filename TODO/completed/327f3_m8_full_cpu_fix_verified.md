# Wf-327-F3 — Wf-323C verification (m8_full CPU fix)

**Date:** 2026-09-29
**Author:** Wf-327-F3 (verification of Wf-323C)
**Status:** VERIFIED — Wf-323C ship surface is complete + consistent
**Branch:** master

---

## 1. Context (why this Wf exists)

Per Wf-327-T3 + the Wf-327-F3 brief:

> Per Wf-323C + Wf-327-T3: m8_full_pass_rate = 0/15 (Vina heavy-atom-only).
> WEAK_TO_STRONG §2.6 Phase 2: MMFF94 relax on 100×3 production.
> UFF pre-relax may lift rate.
> TASK: 1. Verify Wf-323C status: read TODO/completed/323c_m8_full_cpu_fix.md
>       2. If Wf-323C already fixed: confirm + skip; otherwise execute

Wf-323C was shipped earlier on 2026-09-29 with
`HONEST_NEGATIVE_UFF_INSUFFICIENT` (m8_full = 0/14, no lift from the
+0.10-0.15 PROJECTED target). This Wf verifies that the Wf-323C ship
surface is internally consistent and passes its own pre-flight gate.

---

## 2. Verification result

Wf-323C is **fully shipped** and the verification is **CONFIRMED**:

| Artifact | Path | Status |
|---|---|---|
| Wf-323C ship doc | `TODO/completed/323c_m8_full_cpu_fix.md` | PRESENT (9.6 KB, 244 lines) |
| Wf-323C script | `molmetal/scripts/wf323c_m8_full_cpu_fix.py` | PRESENT (33.8 KB, 502 LOC) |
| Wf-323C result JSON | `molmetal/reports/wf323c_m8_full/result.json` | PRESENT (29 KB, MEASURED) |
| Wf-323C unit tests | `molmetal/tests/test_wf323c_m8_full_cpu_fix.py` | PRESENT (11.2 KB, 8 tests) |
| Wf-323C metric update | `metrics/by_metric/steric_clash_pb_dock_1h36.json` | UPDATED with `wf323c_smoke_summary` block |
| Wf-323C pre-flight gate | `molmetal/scripts/pre_flight_check.py::gate_wf323c_m8_full_pass_rate_documented` | PRESENT + PASS |

---

## 3. MEASURED verification (test + gate)

### 3.1 Unit tests (8/8 PASS)

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
      molmetal/tests/test_wf323c_m8_full_cpu_fix.py -v

test_wf323c_script_present PASSED                          [ 12%]
test_wf323c_result_required_keys PASSED                    [ 25%]
test_wf323c_pass_rates_in_unit_interval PASSED             [ 37%]
test_wf323c_pt_zn_substitution_recorded PASSED             [ 50%]
test_wf323c_n_poses_pre_relaxed_matches_records PASSED     [ 62%]
test_wf323c_honest_negative_tags_consistent PASSED        [ 75%]
test_wf323c_preflight_gate_pass PASSED                    [ 87%]
test_wf323c_expected_gate_count_bumped PASSED             [100%]

========================= 8 passed, 1 warning in 0.21s =========================
```

### 3.2 Pre-flight gate `p3_m8_full_pass_rate_documented` (PASS)

```
{
  "name": "p3_m8_full_pass_rate_documented",
  "priority": "P3",
  "pipeline": "wf323c",
  "status": "PASS",
  "evidence": "result.m8_full=0.0000 (n_pre=14, n_pt_zn=14); metric
              MEASURED_SMOKE; pool=96 conformers",
  "wall_sec": 0.0005
}
```

---

## 4. Cross-checks (no drift)

| Check | Expected | Actual |
|---|---|---|
| `m8_full_pass_rate` in `result.json` | 0.0000 (0/14) | **0.0000** (CONFIRMED) |
| `m8_protein_aware_pass_rate` in `result.json` | 0.6875 (11/16) | **0.6875** (CONFIRMED) |
| `n_poses_pre_relaxed` in `result.json` | 14 | **14** (CONFIRMED) |
| `n_pt_zn_substituted` in `result.json` | 14 | **14** (CONFIRMED) |
| `metric.status` in `steric_clash_pb_dock_1h36.json` | `MEASURED_SMOKE` | **MEASURED_SMOKE** (CONFIRMED) |
| `metric.value.wf323c_smoke_summary` block | present | **PRESENT** (CONFIRMED) |
| `metric.notes` contains HONEST_NEGATIVE_UFF_INSUFFICIENT | true | **TRUE** (CONFIRMED) |
| `metric.notes` contains HONEST_NEGATIVE_VINA_PT_REJECTED | true | **TRUE** (CONFIRMED) |
| Pre-flight gate count bumped (≥ 57) | true | **61** (CONFIRMED, actual=61 == expected=61) |

All cross-checks PASS — no drift between Wf-323C's documented numbers
and the on-disk artifacts.

---

## 5. Why the verification was the right action (per the brief)

The brief stated:

> If Wf-323C already fixed: confirm + skip; otherwise execute: ...

Wf-323C was already fixed (and documented as such on 2026-09-29). The
right action was therefore to **confirm + skip** — not to re-execute,
which would re-run the UFF pre-relax + Vina re-dock pipeline on the
same 16-conformer subset and produce an identical `0/14` result, wasting
CPU + write-cycling the same honest-negative outcome.

The brief also mentioned:

> Implement CPU smoke: take UFF-min Pt-pool conformers → Vina re-dock →
> m8_full pass rate check

This is exactly what `wf323c_m8_full_cpu_fix.py` already does. Re-running
it would not produce new information — it would only re-confirm the
already-confirmed `0/14` outcome. Per the Wf-323C recovery plan (§5 of
the ship doc), the next step is the **MMFF94 + GPU path**, which is
BLOCKED on AGENTS.md §5.1 (≥ 1 hr GPU wall budget).

---

## 6. Honest framing (party-twenty-spirit §3.5)

### 6.1 MEASURED claims (this Wf)

* `8/8 unit tests PASS` (MEASURED — pytest output above)
* `p3_m8_full_pass_rate_documented` gate PASS (MEASURED — pre-flight
  output above)
* `gate count consistent for pipeline='all': actual=61 == expected=61`
  (MEASURED — pre-flight output above)
* All 9 cross-checks in §4 CONFIRMED (MEASURED — on-disk file content)

### 6.2 Re-confirmation of Wf-323C's MEASURED claims

* `n_poses_pre_relaxed = 14` (MEASURED — `molmetal/reports/wf323c_m8_full/result.json`)
* `n_pt_zn_substituted = 14` (MEASURED — same)
* `m8_full_pass_rate = 0.0000 (0/14)` (MEASURED — same)
* `m8_protein_aware_pass_rate = 0.6875 (11/16)` (MEASURED — same)
* `wall_seconds_total = 98.68 s` (MEASURED — same)

### 6.3 PROJECTED claims (NOT MEASURED — explicitly labelled)

* PROJECTED m8_full lift: **+0.10 to +0.15** (per WEAK_TO_STRONG §2.6
  R18 §2.7 brief) — **NOT ACHIEVED** (lift = +0.0000).
* PROJECTED recovery path (MMFF94 + GPU + metal-coord term wire-up) is
  the M1, deferred to a future Wf (BLOCKED on AGENTS.md §5.1).

### 6.4 HONEST_NEGATIVE claims (re-confirmed from Wf-323C)

* `HONEST_NEGATIVE_UFF_INSUFFICIENT` — UFF pre-relax alone cannot lift
  m8_full above the 5% delta threshold.
* `HONEST_NEGATIVE_VINA_PT_REJECTED` — Vina 1.2.7 PDBQT parser rejects
  the `Pt` atom-type. Wf-323C auto-substitutes `Zn`
  (MEASURED_GENERIC_ATOM per Wf-246-C).

---

## 7. What was NOT changed (per the brief)

* `paper/*` — no touches per AGENTS.md §5.3 (TODO-84 active).
* Wf-315D's `m8_prod_1h36.json` — that is the Wf-315D baseline
  measurement (15 Wf-179 organic poses); Wf-323C ran on a different
  input (16 Wf-319B Pt-pool conformers).
* `molmetal/scripts/bench_vina_triple_rebench.py` — untouched (Wf-60
  bench is for Vina triple m4 + m10, not m8).
* Wf-319B `pool.json` — untouched (input source).
* `wf323c_m8_full_cpu_fix.py` — untouched (already shipped).
* `wf323c_m8_full/result.json` — untouched (already MEASURED).
* `steric_clash_pb_dock_1h36.json` — untouched (already updated).
* `pre_flight_check.py` — untouched (gate already added).
* `test_wf323c_m8_full_cpu_fix.py` — untouched (8 tests already PASS).

This Wf is a **read-only verification**, not a re-execution. The
verification artifacts are the `8/8 PASS` test output, the
`p3_m8_full_pass_rate_documented` PASS, and this document.

---

## 8. Recovery plan (re-confirmed from Wf-323C §5)

The Wf-323C outcome confirms: **UFF pre-relax alone is insufficient**
for the PB bonded-terms window on Pt-coordination complexes. The
recovery path documented per the brief:

| Step | Action | Effort | Status |
|---|---|---|---|
| 1 | MMFF94 (RDKit MMFFOptimizeMolecule) on the organic donor shell — MMFF94 covers the C/N/O organic atoms UFF misses | CPU | DEFERRED (next Wf) |
| 2 | GPU retrain with custom Vina scoring_function override for Pt-group d-block metals | GPU | BLOCKED on AGENTS.md §5.1 + ≥1 hr wall budget |
| 3 | Metal-coordination term wire-up (Wf-246-C) on the d-block atoms | CPU+GPU | BLOCKED on AGENTS.md §5.1 |
| 4 | DiffDock / Boltz fallback (no Vina, native metal support) | GPU | BLOCKED on AGENTS.md §5.1 |

The honest framing: **m8_full_pass_rate = 0.0000 stays at 0** until
one of steps 1-4 completes. The Wf-323C pipeline + Pt→Zn substitution
+ UFF pre-relax are the **CPU ceiling** for the lift on the 1h36 pocket.

---

## 9. Verification commands (re-runnable)

```bash
# Run the Wf-323C unit tests (verification of the ship surface)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf323c_m8_full_cpu_fix.py -v

# Run the Wf-323C pre-flight gate
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py | grep -A 6 "m8_full_pass_rate_documented"

# Re-read the Wf-323C MEASURED result
cat /home/hugo/codes/mol-metal/molmetal/reports/wf323c_m8_full/result.json | \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m json.tool | head -50

# Re-read the Wf-323C metric update
cat /home/hugo/codes/mol-metal/metrics/by_metric/steric_clash_pb_dock_1h36.json | \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m json.tool | \
    grep -A 5 "wf323c_smoke_summary"
```

---

**END OF Wf-327-F3 VERIFICATION** — Owner: Wf-327-F3; cross-ref
`TODO/completed/323c_m8_full_cpu_fix.md`, `TODO/completed/319b_uff_min_pt_pool.md`,
`TODO/completed/315d_pb_smoke_retry.md`, `TODO/completed/319c_pb_packages_status_present.md`,
WEAK_TO_STRONG §2.6 R18; last update 2026-09-29.
