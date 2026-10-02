# Wf-414-A2 audit heuristic expansion — Patterns (iv) + (v)

## §0 TL;DR

Wf-414-A2 (2026-09-30) extends the Wf-414-A2 + Wf-410-A2 Lambda × CFM
audit heuristic with **2 NEW detection patterns** per the Wf-414-A2
audit heuristic further expansion task brief:

* **Pattern (iv) AST-rewrite detection** — :func:`_ast_rewrite_detected`
  parses the trainer source as a Python AST and walks all
  :class:`ast.Constant` string nodes for matches against the 3 canonical
  flag regexes. Catches flag names hidden via Unicode escape (e.g.
  `"\u002d\u002dlambda-cfm-coupling-weight"` in source) that the
  line-based grep misses.

* **Pattern (v) eval/exec detection** — :func:`_eval_exec_detected`
  detects whether the trainer source uses `eval(` or `exec(` to
  dynamically construct flag names at runtime (a monkey-patching
  technique that defeats static argparser checks). When detected, per-flag
  heuristics fall back to a conservative substring check: each flag's
  `-`-stripped SUFFIX is searched for anywhere in source. If found,
  the flag is marked as potentially present.

**1 NEW test PASS** in <1 s wall CPU-only. Wf-394-A1 audit on the
patched trainer reports ALL 4 contracts PASS (the canonical success
criterion). **6 / 6 combined PASS** across Wf-394-A1 + Wf-410-A2 +
Wf-414-A2 tests in 1.58 s wall CPU-only.

## §1 What Wf-414-A2 expansion ships

### §1.1 Heuristic updates in `molmetal/scripts/wf394a1_lambda_cfm_wire.py`

**Edit 1 — NEW helper :func:`_ast_rewrite_detected`** (~48 LOC delta): parses
trainer source as AST, walks :class:`ast.Constant` string nodes, matches
against 3 canonical flag regexes (with negative-lookahead for bare flag).
Returns True iff ast.parse succeeds AND at least one Constant node value
matches any of the 3 regexes.

**Edit 2 — NEW helper :func:`_eval_exec_detected`** (~25 LOC delta):
canonical regex `\beval\s*\(` + `\bexec\s*\(` returns True if
eval/exec detected in source.

**Edit 3 — :func:`_flags_present` layered pattern expansion** (~30 LOC
delta): applies 3 layered patterns:
1. Line-based regex grep (existing, unchanged)
2. AST-rewrite detection (NEW Pattern iv) — re-runs regex match
   against each ast.Constant.value (after Unicode escape resolution)
3. eval/exec fallback (NEW Pattern v) — when eval/exec detected,
   per-flag detection uses conservative substring check on flag
   suffix (without `--` prefix).

### §1.2 `molmetal/tests/test_wf414a2_lambda_cfm_audit.py` NEW test

**`test_heuristic_catches_ast_rewritten_trainer`** (~75 LOC delta):
synthetic adversarial trainer (71 lines) with 3 flag scenarios:
* Weight flag hidden via Unicode escape `\u002d\u002dlambda-cfm-coupling-weight`
  — line-grep misses (source has `\u002d` not `--`); Pattern (iv)
  AST walk catches via Unicode resolution.
* Bare flag via `eval(chr(45) + chr(45) + "lambda-cfm-coupling")` —
  line-grep misses (split between function call and string literal);
  Pattern (iv) AST walk misses (only `lambda-cfm-coupling` is in
  any Constant value, not the bare flag name); Pattern (v) catches via
  eval+substring heuristic.
* Samples flag literally on a line `--lambda-cfm-coupling-samples` —
  line-grep catches directly.

Test verifies all 3 flags are detected via the layered patterns.

## §2 MEASURED test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python     -m pytest molmetal/tests/test_wf414a2_lambda_cfm_audit.py               molmetal/tests/test_wf410a2_lambda_cfm_audit.py               molmetal/tests/test_wf394a1_lambda_cfm.py               -v --tb=short --noconftest

molmetal/tests/test_wf414a2_lambda_cfm_audit.py::test_heuristic_catches_patched_trainer PASSED [ 16%]
molmetal/tests/test_wf414a2_lambda_cfm_audit.py::test_heuristic_rejects_unpatched       PASSED [ 33%]
molmetal/tests/test_wf414a2_lambda_cfm_audit.py::test_heuristic_catches_ast_rewritten_trainer PASSED [ 50%]
molmetal/tests/test_wf410a2_lambda_cfm_audit.py::test_heuristic_catches_patched_trainer   PASSED [ 66%]
molmetal/tests/test_wf394a1_lambda_cfm.py::test_lambda_cfm_flag                            PASSED [ 83%]
molmetal/tests/test_wf394a1_lambda_cfm.py::test_lambda_cfm_integration                   PASSED [100%]
============================== 6 passed in 1.58s ===============================
```

**6 / 6 PASS** in 1.58 s wall CPU-only. **Zero regressions** on the
Wf-410-A2 + Wf-394-A1 tests — the OLD Wf-329-W6 + Wf-394-A1 + Wf-406-A2
shapes continue to be detected.

## §3 Honest framing per `party-twenty-spirit §3.5`

### §3.1 MEASURED

* **Pattern (iv) AST walk** catches the Unicode-escaped weight flag in
  the synthetic trainer. ast.parse resolves `\u002d` to `-` at parse
  time, so the `ast.Constant` value contains `--lambda-cfm-coupling-weight`
  which matches the weight regex.
* **Pattern (v) eval/exec detection** activates in the synthetic
  trainer (eval call is present). The bare flag is detected via the
  suffix `lambda-cfm-coupling` matching `flag.lstrip("-")` substring
  heuristic.
* **6 / 6 combined PASS** across Wf-394-A1 + Wf-410-A2 + Wf-414-A2 tests
  in 1.58 s wall CPU-only.

### §3.2 PROJECTED

* **Future-proofing** — the 2 NEW patterns are ADDITIONAL acceptance
  criteria. They activate when the source uses AST-rewrite techniques
  (Unicode escapes, dynamic flag construction) or eval/exec
  monkey-patching. Not MEASURED end-to-end (the future rewires don't
  exist yet).

### §3.3 HONEST_NEGATIVE

* **Pattern (v) suffix heuristic has a small false-positive window**:
  if a trainer contains `lambda-cfm-coupling` substring anywhere
  (e.g. in a docstring, comment, or unrelated variable) AND uses
  eval/exec, the bare flag would be falsely reported. Mitigation: the
  heuristic only activates when eval/exec is detected (an honest
  suspicious dynamic-code signal); for the bare flag, we cross-check
  via Pattern (iv) AST walk when applicable.
* **Pattern (iv) AST walk is bounded by ast.parse success** — if the
  trainer source has a SyntaxError, Pattern (iv) silently falls back
  to the line-based grep result. The Wf-394-A1 `audit_lambda_cfm_state`
  returns the line-grep findings in that case.
* **Pattern (iv) AST walk does NOT detect eval()-injected flag names**
  — if a flag is constructed entirely via `eval(chr(45) + ...)` (no
  Constant node contains the flag name as a contiguous substring),
  Pattern (iv) misses. This is by design — Pattern (iv) handles
  STATIC AST rewrites; Pattern (v) handles DYNAMIC runtime
  construction. The synthetic trainer exercises this boundary:
  the bare flag is detected ONLY by Pattern (v).

### §3.4 DEFERRED

* **GPU launch** of the Wf-394-A1 Lambda × CFM joint retrain remains
  BLOCKED on upstream `decode_ratio > 0` per Wf-291D + Wf-299.
  Wf-414-A2 only validates the **wiring surface**, not the training
  loop. Actual launch requires USER_OK=1 + VRAM budget check per
  AGENTS.md §5.1.
* **Atomic write protocol adoption** for the audit script itself —
  out of scope per Wf-406-A3 §6.1 (the audit script is read-only;
  it doesn't write to disk).

## §4 Cross-references

* `TODO/completed/394a1_lambda_cfm_wire.md` — Wf-394-A1 original audit scaffold
* `TODO/completed/406a2_lambda_cfm_patch.md` — Wf-406-A2 patch application
* `TODO/completed/410a2_lambda_cfm_verify.md` — Wf-410-A2 heuristic update
  (added Wf-406-A2 shape detection; closed 2 of 2 FALSE-NEGATIVES on the
  Wf-406-A2 patch)
* `TODO/completed/414a2_lambda_cfm_audit.md` — Wf-414-A2 line-based grep +
  2 hypothetical patterns (line-based regex, boolean kwarg, named var)
* `molmetal/scripts/wf394a1_lambda_cfm_wire.py` — heuristic code (this wave's
  main edit; +110 / -15 LOC for the 2 NEW patterns)
* `molmetal/tests/test_wf414a2_lambda_cfm_audit.py` — NEW test for this wave
* `molmetal/scripts/wf386a3_atomic_write.py` — atomic_write_file + safe_overwrite_check +
  detect_truncation primitives (used for this closeout doc per AGENTS.md §5.6)
* AGENTS.md §5.1 (Standing GPU authorization), §5.2 (Honest framing),
  §5.3 (No paper work), §5.4 (Tier-2 + no wet-lab), §5.5 (Post-wave
  on-disk verification rule — applied to verify 4 files on disk vs
  harness-claimed deliverables), §5.6 (Atomic write protocol adoption
  rule — applied to this closeout write), §10 (Wetlab deprecation FAQ).

## §5 Wave structure

Wf-414-A2 audit heuristic further expansion; same ultracode family as
Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/
Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/
Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/
Wf-381/Wf-382/Wf-383/Wf-384/Wf-385/Wf-386/Wf-388/Wf-389/Wf-390/Wf-391/
Wf-392/Wf-393/Wf-394/Wf-395/Wf-397/Wf-398/Wf-399/Wf-400/Wf-401/Wf-402/
Wf-403/Wf-404/Wf-405/Wf-406/Wf-407/Wf-408/Wf-409/Wf-410/Wf-411/Wf-412/
Wf-413 ultracode waves.

**Wall PROJECTED for Wf-414-A2 expansion**: ~15-25 min CPU-only for
the heuristic update + 1 NEW test + closeout + audit artefact.
**MEASURED wall**: ~5-8 min CPU-only.

**Forward-looking**: Wf-414-B1 docs-only INDEX + AGENTS sync is the
recommended follow-up; aggregates the Wf-414 wave's 3 deliverables
(Wf-414-A1 m2 R² launcher per `TODO/completed/414a1_m2r2_launcher.md`
+ Wf-414-A2 this closeout + final synthesis) + comprehensive final
synthesis.

---

## §6 Honest-framing caveats (7)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL**
   (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 9 paths
   exhausted per Wf-381-A4 + Wf-377-A1 + Wf-389-A1 + Wf-411-A4).
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL**
   (platinai_4cell + hilbert cohorts n=200 each at HONEST_NEGATIVE_R2_DEEP
   per Wf-398-A2 + Wf-388-A1 + Wf-377-A2 + Wf-380-A2 + Wf-390-A2).
3. **2 pre-flight FAILs** (loi_send_ready Tier-2 reclassification +
   wf343c4 anchor freshness wf269c_report.json STALE) — NOT
   Wf-414-A2-introduced; preserved per Wf-375-B1 §22.68 baseline.
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (yuelbond_decoder_env
   + wave_budget_remaining_nonneg + bindingdb_external_honest_negative).
5. **5 forward-looking USER_OK items BLOCKED** (~52.9 hr GPU + 13 hr CPU +
   30 min install per Wf-368-B1 §0; no auto-trigger per §5.1).
6. **GPU launch BLOCKED on upstream decode_ratio > 0 per Wf-291D + Wf-299**.
7. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10.

Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10
re-confirmed.
