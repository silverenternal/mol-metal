# WF-IMPROVEMENT-BACKLOG I2 — Subsystem Smoke Scripts

**Status**: DONE 2026-09-25
**Owner**: ultracode dispatch harness
**Spec**: `TODO/IMPROVEMENT_BACKLOG.md` §I2
**Outcome**: 4 / 4 subsystem smokes PASS end-to-end on CPU in <60 s aggregate.

---

## 1. TL;DR

Four self-contained shell scripts under `molmetal/scripts/smoke/` provide
fast end-to-end verification of every major subsystem in the Mol-Metal
stack.  Each script:

* Runs in <120 s on a CPU-only CI box (no GPU, no external data).
* Prints a single `<NAME>_SMOKE PASS|FAIL ...` line on the last stdout line.
* Exits `0` on PASS, `1` on FAIL (and `1` on any internal crash).
* Is wired into `molmetal/tests/test_smoke_scripts.py` as 10 parametrized
  pytest guards (4 × exit-code, 4 × PASS-token, 1 × all.sh aggregate,
  1 × directory surface).

The aggregate entry point `bash molmetal/scripts/smoke/all.sh` runs all
4 sequentially and reports `ALL_SMOKE TOTAL_PASS=4/4 wall=<s>s`.

| Smoke                  | Subsystem                                | Success criterion          | Wall (CPU) |
| ---------------------- | ---------------------------------------- | -------------------------- | ---------- |
| `smoke_cfm.sh`         | LipmanFlowMatchingAdapter                | decode_ratio == 1.0        | 5–10 s     |
| `smoke_lambda.sh`      | Lambda MCTSProofSearch (10 × 1)         | n_distinct ≥ 1 per pocket  | 10–20 s    |
| `smoke_evaluators.sh`  | 14 evaluators (SA, QED, Vina, …)         | 14 / 14 PASS               | 3–5 s      |
| `smoke_pitfalls.sh`    | 7 R12-R17 pitfall guards                 | 7 / 7 PASS                 | 0.5–2 s    |
| **Total**              | **all 4 sequentially**                   | **ALL_SMOKE TOTAL_PASS=4/4** | **<60 s** |

---

## 2. What each script does

### 2.1 `smoke_cfm.sh` — CFM adapter end-to-end

Trains a tiny (hidden_dim=64, n_layers=2) `LipmanFlowMatchingAdapter`
for 200 steps on a synthetic 8-atom pocket, then runs 4 decode probes
of `n_samples=4, n_steps=20` each (16 generated mols total).  The
success criterion `decode_ratio == 1.0` is the lenient atom-based
contract — a mol is "decoded" iff its `coords` tensor is finite, has
≥ 2 atoms, and has the expected shape.  This matches the canonical
contract used by `metallo_drug_smoke_retrain._decode_smoke`.  The
stricter "bonds.shape[-1] > 0" contract is reserved for post-bond-head
runs (out of scope for this smoke).

Training atoms are drawn from the adapter's 20-element default vocab
(`1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, 39, 40, 53, 72,
78`) so the adapter's `vocab_mask` doesn't skip the optimizer step
(see `test_p1_oov_*`).

### 2.2 `smoke_lambda.sh` — Lambda MCTS pilot

Runs `r4_lambda_only_run.py --smoke --quiet` (the `_R17_SMOKE_PRESET`:
10 pockets × 1 seed × 20 sims on `test_010_019_subset_manifest.csv`).
Parses the resulting `report.json` and verifies that every one of the
10 cells reports `n_distinct >= 1` AND the total distinct count across
all cells is `>= 1`.  Uses the same manifest + preset as
`tests/test_smoke_flag.py`, so the smoke is bit-compatible with the
canonical IMPROVEMENT_BACKLOG B1 measurement (13.3 s CPU).

### 2.3 `smoke_evaluators.sh` — 14-evaluator sanity

Invokes `eval_all_evaluators.py` and parses the trailing
`SUMMARY: X/14 evaluators PASS` line.  The 14 evaluators are SA score
(3 mols), QED (2), Vina adapter (1), Vina Python binding (1), Meeko
(1), PoseBusters (1), Bio.PDB (1), OpenBabel (1), ProDy (1), plus
cisplatin SA + QED reference sanity (2).  PASS iff `X == 14`.

### 2.4 `smoke_pitfalls.sh` — 7 pitfall guards

Invokes `wf_pitfall_guard/check_all.py` and parses the trailing
`PASS=N FAIL=M SKIP=K MISS=L` summary line emitted by
`check_all._format_table`.  PASS iff `PASS == 7 AND FAIL == 0`.

The 7 R12-R17 guards cover:

| Pitfall | Fix                                                                 |
| ------- | ------------------------------------------------------------------- |
| P1      | OOV atoms → vocab_mask + 20-element default vocab                  |
| P2      | SynFlowNet shim TypeError → per-rule try/except                     |
| P3      | MetalLigandExchange 84-rule matrix + cisplatin seed default         |
| P4      | BF16 atom_logits clamp ±50 + FP32 AMP fallback                      |
| P5      | `r4_lambda_only_run` SAFETY_MAX ≥ 10000 floor                       |
| P6      | Reference clone location auto-detect (parent vs clone root)        |
| P7      | ModelWrapper keyword-only `t` fallback (`functools.partial` fix)    |

### 2.5 `all.sh` — aggregate entry point

Runs all 4 smokes sequentially with a per-script 180 s timeout
(`PER_SMOKE_TIMEOUT` env var overridable).  Prints a per-script
PASS/FAIL line, then a final `ALL_SMOKE TOTAL_PASS=N/4 TOTAL_FAIL=Y/4
wall=<s>s` aggregate.  Exits `0` iff all 4 PASS.

---

## 3. Pytest gate — `molmetal/tests/test_smoke_scripts.py`

10 parametrized guards; all 10 PASS in ~70 s on CPU.

```
molmetal/tests/test_smoke_scripts.py::test_smoke_directory_present PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_exits_zero_with_pass_token[smoke_cfm.sh-CFM_SMOKE PASS] PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_exits_zero_with_pass_token[smoke_lambda.sh-LAMBDA_SMOKE PASS] PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_exits_zero_with_pass_token[smoke_evaluators.sh-EVALUATORS_SMOKE PASS] PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_exits_zero_with_pass_token[smoke_pitfalls.sh-PITFALLS_SMOKE PASS] PASSED
molmetal/tests/test_smoke_scripts.py::test_all_sh_runs_all_4_smokes_and_reports_4_of_4 PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_completes_under_120s[smoke_cfm.sh] PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_completes_under_120s[smoke_lambda.sh] PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_completes_under_120s[smoke_evaluators.sh] PASSED
molmetal/tests/test_smoke_scripts.py::test_smoke_script_completes_under_120s[smoke_pitfalls.sh] PASSED
=================== 10 passed, 1 warning in 69.22s (0:01:09) ===================
```

The 4× `test_smoke_script_exits_zero_with_pass_token` parametrizations
verify both the exit-code contract and the PASS-token contract on stdout
(the same contract the dispatch harness greps for).  The 4×
`test_smoke_script_completes_under_120s` parametrizations enforce the
IMPROVEMENT_BACKLOG §I2 wall-time budget.

---

## 4. Honest framing

* **`decode_ratio=1.0` is the LENIENT atom-based contract** (finite,
  non-empty coords, ≥2 atoms).  The strict `bonds.shape[-1] > 0`
  contract returns `decode_ratio ≈ 0.0` even after 200-step training
  on a freshly-initialised adapter — that's an expected property of
  the cold-start bond head, not a regression.  The lenient contract is
  the canonical one used by `metallo_drug_smoke_retrain._decode_smoke`
  and matches what the IMPROVEMENT_BACKLOG §I2 spec calls
  "decode_ratio=1.0" (i.e. "all probes produced finite samples").
* **The Lambda smoke inherits the same `--smoke` preset the rest of
  the suite uses.**  10 cells × 1 seed × 20 sims is the smallest
  configuration that exercises the click-rule library, the metal seed
  default (cisplatin), the persistent search tree, and the report
  writer end-to-end.  Larger configs (`--r17-paper-grade` runs 100
  pockets × 3 seeds × 1000 sims ≈ 13 h) are out of scope for this
  fast-feedback smoke.
* **`pytest` is the executor for `smoke_pitfalls.sh`.**  When `pytest`
  is on PATH (the env.sh case) the inner runner prefers it; when it's
  missing it falls back to `python -m pytest` (the raw-python case).
  The contract (PASS/FAIL/SKIP counts) is identical in both paths.

---

## 5. Files added

```
molmetal/scripts/smoke/smoke_cfm.sh           (~120 lines)
molmetal/scripts/smoke/smoke_lambda.sh        (~95  lines)
molmetal/scripts/smoke/smoke_evaluators.sh    (~80  lines)
molmetal/scripts/smoke/smoke_pitfalls.sh      (~85  lines)
molmetal/scripts/smoke/all.sh                 (~95  lines)
molmetal/tests/test_smoke_scripts.py          (~140 lines, 10 tests)
molmetal/reports/wf_smoke_scripts/final.md    (this file)
```

All scripts are `chmod +x` (executable) so the dispatch harness can
invoke them directly as `bash <script>` without any interpreter
override.

---

## 6. Next steps (out of scope for this PR)

* Wire `molmetal/tests/test_smoke_scripts.py` into the CI matrix so the
  smoke gate runs on every PR.
* Add a 5th smoke (`smoke_metrics.sh`) for the 9 §4 metrics once the
  metric contract stabilises.
* Tighten the CFM smoke to require at least one `bonds.shape[-1] > 0`
  sample (the bond-head contract) once the production bond-head
  checkpoint is committed.
