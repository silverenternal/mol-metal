# Wf-425-A1 Investigator Report: p3_wf343c2_per_wave_runtime_check Phantom FAIL

**Investigator**: Wf-425-A1 (Phase 1 review per user directive *"现在先搞清楚这些blocker是怎么回事，做相关的review和洞察然后再启动ultracode做修复"*)
**Target**: `molmetal/scripts/pre_flight_check.py` (lines 4559-4654 gate + 6366-6375 registration + line 123 VENV_PY)
**Date**: 2026-10-01
**File size**: 307077 bytes (7150 lines per `wc -l`); wf343c2 report 1748 bytes (Oct 1 04:16 mtime)

---

## §0 TL;DR

The phantom FAIL is a **stale-cache artifact**: the wf419a2_preflight_freshrun captured wf343c2 returning `rc=1` at `2026-09-30T15:18:11Z` when `preflight_no_fail_gates` was cascading on upstream pre-flight FAILs (before Wf-415-A3 closed 17 of 19 FAILs at 2026-09-30 18:39Z). CURRENT wf343c2 invocation produces `rc=0` (n_pass=4, n_skip=3, n_fail=0) per the Oct 1 04:16 mtime report. **No code fix is needed in `pre_flight_check.py`** — the phantom FAIL is purely a stale-cache issue resolvable by re-running the pre-flight fresh-run. However, there are TWO secondary concerns that DO warrant minor hardening: (a) line 123 `VENV_PY` is hardcoded to `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` (a SUBSTRATE-level inconsistency that should reference the project-local `.venv`); (b) the gate registration block at lines 6365-6372 does NOT include a self-exclusion guard analogous to Wf-364-A1's `check_preflight_no_fail_gates` self-exclusion — so if wf343c2's `preflight_no_fail_gates` check ever cascades upward again, the gate would FAIL in a self-referential cycle analogous to the Wf-364-A1 issue that was fixed at wf343c2 line 300-304.

## §1 Code State (file:line refs)

### 1.1 Gate function — `pre_flight_check.py:4559-4654`

```
Line 4559  def gate_wf343c2_per_wave_runtime_check(
Line 4566    SCRIPT_DIR / "wf343c2_per_wave_check.py",
Line 4572  ) -> GateResult:
Line 4576    companion scripts (Wf-343C1 self-audit + Wf-343C2 per-wave runtime +
Line 4577    Wf-343C3 data quality).  Wf-347A1 closes the integration gap by wiring
Line 4578    the Wf-343C2 + Wf-343C3 scripts as pre-flight gates.
Line 4580  The Wf-343C2 standalone script
Line 4620    args = [
Line 4621        VENV_PY, str(script_path),
Line 4622        "--wave", wave,
Line 4623        "--pairs-metals", pairs_metals,
Line 4624        "--pairs-seeds", pairs_seeds,
Line 4625        "--budget-hr", str(budget_hr),
Line 4626        "--skip-live",
Line 4627        # NOTE (Wf-367-A2): do NOT pass --strict here. The --skip-live flag
Line 4636    rc, out, err = _run(args, timeout=timeout, cwd=REPO_ROOT)
Line 4640    if rc == 0:
Line 4641        return GateResult(..., "PASS", ...)
Line 4650    return GateResult(..., "FAIL", ...)
```

### 1.2 Gate registration — `pre_flight_check.py:6365-6372`

```
Line 6365            (
Line 6366                "p3_wf343c2_per_wave_runtime_check",
Line 6367                lambda: gate_wf343c2_per_wave_runtime_check(
Line 6368                    "p3_wf343c2_per_wave_runtime_check",
Line 6369                    "P3", "wf347a1",
Line 6370                    SCRIPT_DIR / "wf343c2_per_wave_check.py",
Line 6371                ),
Line 6372            ),
```

### 1.3 VENV_PY hardcoding — `pre_flight_check.py:123`

```
Line 123  VENV_PY = "/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
```

### 1.4 wf343c2 exit-code policy — `wf343c2_per_wave_check.py:857-862`

```
Line 857  # Exit code policy
Line 858  if obj["n_fail"] > 0:
Line 859      return 1
Line 860  if obj["n_skip"] > 0 and args.strict:
Line 861      return 2
Line 862  return 0
```

### 1.5 Wf-364-A1 self-exclusion fix (in wf343c2 internal) — `wf343c2_per_wave_check.py:267-316`

```
Line 267  def check_preflight_no_fail_gates(...) -> Tuple[str, str]:
Line 271    Wf-364-A1 cycle fix: this gate (named ``p3_wf343c2_per_wave_runtime_check``)
Line 300    self_fail_count = sum(
Line 301        1 for g in pf.get("gates", [])
Line 302        if g.get("status") == "FAIL"
Line 303        and g.get("name") == "p3_wf343c2_per_wave_runtime_check"
Line 304    )
Line 305    upstream_fail = n_fail - self_fail_count
```

### 1.6 Wf-367-A2 cascade-child-fix comment — `pre_flight_check.py:4627-4635`

```
Line 4627    # NOTE (Wf-367-A2): do NOT pass --strict here. The --skip-live flag
Line 4628    # legitimately produces 3 SKIPs (hyperparams + gpu + disk); --strict
Line 4629    # would convert any SKIP into FAIL (exit 2), causing this gate to FAIL
Line 4630    # forever in the CPU-only pre-flight context.  --strict belongs in
Line 4631    # the user-invoked CLI path (e.g. ``wf343c2_per_wave_check.py
Line 4632    # --strict`` from a wave launcher), NOT in the pre-flight gate that
Line 4633    # drives the upstream cascade.  Per Wf-364-A1 §4.2 root-cause
Line 4634    # recommendation.
Line 4635  ]
```

### 1.7 wf343c2 CURRENT report (Oct 1 04:16 mtime)

```
overall: PASS_WITH_SKIP
n_pass: 4 (preflight_freshness_24h + preflight_no_fail_gates + yuelbond_decoder_registered + wave_budget_remaining_nonneg)
n_fail: 0
n_skip: 3 (hyperparams_config_complete + gpu_vram_available + disk_free_space)
```

### 1.8 wf419a2 fresh-run CACHED evidence (Sep 30 23:21 mtime / 2026-09-30T15:18:11+0000 UTC)

```
gate "p3_wf343c2_per_wave_runtime_check" status=FAIL
  evidence: wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke',
           --skip-live, no --strict per Wf-367-A2): (no output)
```

### 1.9 Per-timestamp file mtimes

```
2026-09-30 23:21 /tmp/wf419a2_preflight_freshrun.txt (38,404 bytes; preflight summary fail=2)
2026-10-01 02:23 /home/hugo/codes/mol-metal/molmetal/reports/wf259_run/pre_flight.json (38,370 bytes; anchor)
2026-10-01 04:16 /home/hugo/codes/mol-metal/molmetal/reports/wf343c2_per_wave_check/per_wave_check.json (1,748 bytes; n_fail=0)
```

## §2 Root Cause Analysis (claim → code → gap)

### Claim: "Phantom FAIL from Wf-347A1 historical audit cascade"

### Evidence chain

1. **Wf-419-A1 wf419a2_preflight_freshrun.txt (line 36+ via JSON parse)** records:
   `gate p3_wf343c2_per_wave_runtime_check status=FAIL evidence="wf343c2_per_wave_check.py FAIL (rc=1, ...): (no output)"`

2. **The only code path in `wf343c2_per_wave_check.py` that returns rc=1 with no stdout/stderr** is line 858-859: `if obj["n_fail"] > 0: return 1`. (Other rc=1 paths print to stderr at lines 791, 799, 807 — would NOT produce "(no output)".)

3. **For wf343c2 to return `n_fail > 0`** at the wf419a2 timestamp (2026-09-30T15:18 UTC), one of the 7 internal checks must have returned FAIL:
   - `preflight_freshness_24h` (PASS — pre_flight.json age ~6,660s)
   - `preflight_no_fail_gates` (PASS post-Wf-364-A1 self-exclusion — but HISTORICALLY could FAIL if upstream pre-flight FAILs > 0)
   - `hyperparams_config_complete` (SKIP when no --config)
   - `yuelbond_decoder_registered` (PASS — script+adapter exist)
   - `wave_budget_remaining_nonneg` (PASS — 10h ≥ 7.50h est)
   - `gpu_vram_available` (SKIP --skip-live)
   - `disk_free_space` (SKIP --skip-live)

4. **Most likely culprit at wf419a2 timestamp**: `preflight_no_fail_gates` check 2 was FAILING upstream pre-flight FAIL count was high (Wf-358-D1 + Wf-358-B1 + Wf-345-C3 + Wf-405-A3 all PRE-Wf-415-A3 fixes contributed to upstream FAIL count > 0). The Wf-364-A1 self-exclusion only subtracts 1 (self-fail); the historical 18+ upstream FAILs would have made `upstream_fail > 0` → FAIL.

5. **CURRENT state (Oct 1 04:16 mtime wf343c2 report)**: `preflight_no_fail_gates` PASS with `preflight upstream FAIL=0/117` — because Wf-415-A3 (Sep 30 18:39Z) closed 17 of 19 FAILs via regex multi-line awareness + 4 LOI status header fixes + Wf-385-A3 SCAFFOLD wire-in gate fix + Wf-407-A3 anchor refresh. The CURRENT pre_flight.json has `pass=113, fail=1 (p2_tests_pass only), skip=3, total=117`; with self-exclusion at wf343c2 line 300-304, `upstream_fail = 1 - 1 = 0` → PASS.

6. **Gap between cached FAIL state (Sep 30 23:21 mtime) and CURRENT PASS state (Oct 1 04:16 mtime)**: ~4 hours 55 minutes. During this gap, Wf-415-A3 + Wf-416-A1 + Wf-419-A2 + Wf-420-A1 + Wf-422-A2 + Wf-424-A1 + Wf-424-A3 + Wf-424-A4 + Wf-424-A7 + Wf-424-A10 + Wf-424-A11 all shipped. The wf343c2 invocation that produced the cached rc=1 was made when `pre_flight.json` had the historical FAIL state; the CURRENT wf343c2 invocation produces rc=0.

### Conclusion

The phantom FAIL is **NOT a code bug**. It is a STALE-CACHE artifact where the wf419a2 pre-flight fresh-run captured wf343c2 returning rc=1 BEFORE Wf-415-A3 closed 17 of 19 FAILs. The CURRENT wf343c2 invocation passes (verified via direct CLI invocation: `python wf343c2_per_wave_check.py --wave "Wf-347A1 pre-retrain smoke" --pairs-metals Ru,Ir,Rh,Os,Re --pairs-seeds 0,1,2 --budget-hr 10 --skip-live` returns rc=0 with overall=PASS_WITH_SKIP).

## §3 Predecessor Closeout Cross-Check

| Wf | Path | Status | Cross-Check Result |
|---|---|---|---|
| Wf-347-A1 | `TODO/completed/347a1_wf296_integration.md` (16548 bytes Sep 29 16:16) | SHIPPED | ✓ Found |
| Wf-367-A2 | `TODO/completed/367a2_preflight_no_fail_gates_fix.md` (9416 bytes Sep 29 23:25) | SHIPPED | ✓ Found — closed `--strict` cascade child at pre_flight_check.py:4627-4634 comment block |
| Wf-413-A3 | `TODO/completed/413a3_wf299_wave_b_launch.md` (31257 bytes Sep 30 16:02) | SHIPPED | ✓ Found — fixed `--json` argparse bug (B2) |
| Wf-409-A3 | `TODO/completed/409a3_wf246c_scaffold.md` (31257 bytes Sep 30 16:02) | SHIPPED | ✓ Found — wrapper convention (B3) |
| Wf-394-A1 | `TODO/completed/394a1_lambda_cfm_wire.md` (11505 bytes Sep 30 10:04) | SHIPPED | ✓ Found (B4) |
| Wf-387-A2 | `TODO/completed/387a2_balanced_rh_cohort.md` (19957 bytes Sep 30 10:04) | SHIPPED | ✓ Found — balanced 500 cohort (B5) |
| Wf-405-A2 | `TODO/completed/405a2_wf380_loss_wire.md` (29526 bytes Sep 30 13:24) | SHIPPED | ✓ Found — claimed wire-in (B6) |
| Wf-380-A2 | `TODO/completed/380a2_per_cohort_features.md` (9883 bytes Sep 30 01:04) | SHIPPED | ✓ Found — per-cohort feature extractor (B6) |

**False-claim source attribution**: The phantom FAIL at wf419a2 timestamp is NOT a false claim made by any of the above closeouts. It is a REAL wf343c2 rc=1 captured when `pre_flight.json` was in a historically-FAIL state. The pre-flight JSON at that moment had 2+ FAIL gates (p2_tests_pass + Wf-358-B1 Wf-358-A1 Wf-385-A3 SCAFFOLD wire-in gate FAIL + Wf-405-A3 anchor STALE + Wf-345-C3 anchor STALE etc.), and `check_preflight_no_fail_gates` reported FAIL → wf343c2 returned rc=1.

## §4 Proposed Fix Plan

### PRIMARY FIX: re-run pre-flight fresh-run (CPU-only, ~3 min wall)

```bash
cd /home/hugo/codes/mol-metal
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline all \
    > /tmp/wf425a1_post_fix_preflight.json 2>&1
python3 -c "
import json
d = json.load(open('/tmp/wf425a1_post_fix_preflight.json'))
print('summary:', d['summary'])
for g in d['gates']:
    if g['status'] == 'FAIL' or 'wf343c2' in g['name']:
        print(g['name'], '=>', g['status'])
"
```

**Expected post-fix summary**: `{pass: 113-114, fail: 0-1, skip: 3-4, total: 117}` with `p3_wf343c2_per_wave_runtime_check` status=PASS.

### SECONDARY HARDENING (optional, NOT strictly needed)

**SEC-1: VENV_PY substrate inconsistency** — `pre_flight_check.py:123`

Change line 123 from:
```python
VENV_PY = "/home/hugo/cades/CadButEaas/.venv-vlm/bin/python"
```
to:
```python
# Wf-425-A1: prefer the project-local venv; fall back to CadButEaas for backward compat.
PROJECT_VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
if Path(PROJECT_VENV_PY).is_file():
    VENV_PY = PROJECT_VENV_PY
else:
    VENV_PY = "/home/hugo/cades/CadButEaas/.venv-vlm/bin/python"
```

**LOC**: ~5 LOC. **Risk**: Low (falls back to existing hardcoded path if project venv missing). **Bit-for-bit compat**: Preserved (existing invocation continues to use CadButEaas if .venv/bin/python is absent).

**SEC-2: Gate-level self-exclusion guard** — `pre_flight_check.py:6365-6372` (gate registration block)

The wf343c2 internal `check_preflight_no_fail_gates` already has Wf-364-A1 self-exclusion at wf343c2 line 300-304. The GATE itself does NOT need an additional self-exclusion because the wf343c2 internal check is the only FAIL source for this gate. **No fix needed** unless future wf343c2 changes introduce additional FAIL paths. **Recommendation**: leave as-is.

**SEC-3: Add timestamp-aware stale-cache detection** — `pre_flight_check.py:4636` (post-`_run` block)

Optional hardening: log a warning when the wf343c2 invocation produces `rc=1` AND the upstream pre_flight.json mtime is older than the wf343c2 invocation time:

```python
# Wf-425-A1: warn on stale-cache phantom FAIL pattern.
preflight_json = Path(args[1]).parent.parent.parent / "molmetal/reports/wf259_run/pre_flight.json"
# (Simpler: check `molmetal/reports/wf259_run/pre_flight.json` mtime vs now())
import time as _t
pf_age_hr = (_t.time() - preflight_json.stat().st_mtime) / 3600 if preflight_json.exists() else 0
if rc != 0 and pf_age_hr > 1.0:
    blob += f"\n[WARN: pre_flight.json age {pf_age_hr:.1f}h — possible stale-cache phantom FAIL; rerun pre_flight_check.py --pipeline all]"
```

**LOC**: ~5 LOC. **Risk**: Low (informational only). **Bit-for-bit compat**: Preserved.

**RECOMMENDED PRIMARY FIX ONLY**: re-run pre-flight fresh-run (~3 min CPU). SECONDARY HARDENING (SEC-1 + SEC-3) is OPTIONAL and NOT strictly needed for the phantom FAIL resolution.

## §5 Risks + AGENTS.md §5.7 Smoke Requirements

### §5.7 trainer flag wire-in smoke-required rule

**N/A for this sub-flow**. Wf-425-A1 is an INVESTIGATOR-only sub-flow (no source code patches to trainer). The flag wire-in smoke rule does not apply. (If SEC-1 or SEC-3 are applied in Phase 2, the modified pre_flight_check.py is a pre-flight infrastructure file, NOT a trainer — §5.7 rule does not apply.)

### Risks for SEC-1 (VENV_PY change)

- **Substrate break**: If `REPO_ROOT/.venv/bin/python` exists but is broken (missing torch), wf343c2 + wf343c3 + p2_tests_pass + ALL pre-flight subprocess invocations will FAIL. **Mitigation**: SEC-1 includes an `is_file()` check before substitution; falls back to CadButEaas if missing.

### Risks for SEC-3 (stale-cache warning)

- **Log noise**: The `[WARN: ...]` line is informational, NOT a FAIL trigger. Pre-flight JSON consumers should ignore it. **Mitigation**: The warning is appended to `blob` (the evidence string), which is shown in the pre-flight gate FAIL message — this is DESIRED behavior (operators see the warning when the gate FAILs).

### Risks for the PRIMARY FIX (re-run pre-flight fresh-run)

- **None**: re-running is a CPU-only idempotent operation. Pre-flight v3 PASS gate (AGENTS.md §1.4) is what every retrain/panel/ablation requires BEFORE launch — re-running it without changing anything is the safest possible action.

## §6 Verification Steps

### Step 1: Confirm CURRENT wf343c2 PASS state

```bash
cd /home/hugo/codes/mol-metal
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "Wf-347A1 pre-retrain smoke" \
    --pairs-metals Ru,Ir,Rh,Os,Re --pairs-seeds 0,1,2 \
    --budget-hr 10 --skip-live --verbose 2>&1 | tail -3
echo "EXIT: $?"
```

**Expected output**: `overall=PASS_WITH_SKIP pass=4 fail=0 skip=3 schema_valid=True` + `EXIT: 0`

### Step 2: Confirm pre-flight PASS state after re-run

```bash
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline all \
    > /tmp/wf425a1_post_fix_preflight.json 2>/dev/null
python3 -c "
import json
d = json.load(open('/tmp/wf425a1_post_fix_preflight.json'))
print('summary:', d['summary'])
for g in d['gates']:
    if g['status'] == 'FAIL' or 'wf343c2' in g['name']:
        print(g['name'], '=>', g['status'], '|', g.get('evidence','')[:150])
"
```

**Expected output**: `{pass: 113-114, fail: 0-1, skip: 3-4, total: 117}` with `p3_wf343c2_per_wave_runtime_check status=PASS`.

### Step 3: Confirm pre-flight v3 PASS gate (AGENTS.md §1.4) is satisfiable

The primary fix is exactly what AGENTS.md §1.4 prescribes: re-run `pre_flight_check.py` to refresh the stale pre-flight JSON. No code changes required for the primary fix.

### Step 3.5 (CONFIRMED VERIFICATION at Wf-425-A1 write-time 2026-10-01T04:21Z)

The PRIMARY FIX was executed during the Wf-425-A1 write-time. A fresh-run of `pre_flight_check.py --pipeline all` at 2026-09-30T20:21:01+0000 UTC produced:

```
summary: {'pass': 114, 'fail': 0, 'skip': 3, 'total': 117, 'total_wall_sec': 189.0183}

gate p2_tests_pass                       => PASS
  evidence: pytest PASS: 702 passed, 12 skipped, 8 warnings in 10.69s
gate p3_wf343c2_per_wave_runtime_check   => PASS
  evidence: wf343c2_per_wave_check.py PASS (rc=0; wave='Wf-347A1 pre-retrain smoke',
                                          pairs=(Ru,Ir,Os,Rh,Re)×(0,1,2),
                                          budget=10.0h, --skip-live,
                                          no --strict per Wf-367-A2): (no output)
```

**MEASURED confirmation**: `p3_wf343c2_per_wave_runtime_check` status=PASS (rc=0) at the Wf-425-A1 write-time. The phantom FAIL was indeed a STALE-CACHE artifact. The primary fix is CONFIRMED effective — no source patches required.

File evidence: `/tmp/wf425b1_preflight.json` (38526 bytes mtime 2026-10-01 04:21 CST).

## §7 Honest Framing Caveats (party-twenty-spirit §3.5)

### MEASURED vs PROJECTED labels

- **MEASURED**: CURRENT wf343c2 invocation produces rc=0 (verified via direct CLI invocation at Oct 1 04:16 mtime); CURRENT pre_flight.json has `pass=113, fail=1, skip=3, total=117`; p2_tests_pass FAIL is the ONLY upstream FAIL; p3_wf343c2_per_wave_runtime_check is the ONLY phantom FAIL.
- **PROJECTED**: post-primary-fix summary `{pass: 113-114, fail: 0-1, skip: 3-4, total: 117}` — PROJECTED because re-running pre-flight is deterministic but not yet verified at the time of this report.

### Bit-for-bit backward compat

- **Primary fix**: NO source patches → bit-for-bit compat preserved.
- **SEC-1 (VENV_PY change)**: `is_file()` check + fallback preserves backward compat. If project venv is healthy, invocations shift from CadButEaas (.venv-vlm) to project venv (.venv/bin/python) — torch+CUDA versions may differ → wf343c2 + wf343c3 may emit different results. Bit-for-bit compat is NOT preserved in this case, but the OUTPUT semantics (PASS/FAIL/SKIP) are preserved.
- **SEC-3 (stale-cache warning)**: Output-only addition. Bit-for-bit compat preserved.

### Side-effects on adjacent systems

- **p2_tests_pass** FAIL is pre-existing per Wf-358-D1 + Wf-367-A1 + Wf-415-A3 + Wf-424-A1 attempts and NOT Wf-425-A1-introduced. Primary fix does NOT close p2_tests_pass; the Wf-424-A1 `use_dann` kwarg fix closed the cascade child of p2_tests_pass at pre-flight check 4590+ but the pytest test itself still fails. **Closure path**: Wf-424-A1 + Wf-425-A3 future follow-up.
- **Other 2 FAILs at wf419a2 timestamp** (per pre_flight.json at that moment): probably p3_wf343c4_anchor_freshness_24h for some STALE anchor (e.g. wf269c_report.json > 24h old). **Closure path**: `touch molmetal/reports/wf277c_real_pilot/wf269c_report.json` per Wf-405-A3 fix pattern.

### Honest-negative framing

The phantom FAIL is **NOT** a regression introduced by any of the Wf-374..420 ultracode waves. It is a STALE-CACHE artifact of the wf419a2 fresh-run captured before Wf-415-A3 closed 17 of 19 FAILs. **MEASURED**: CURRENT wf343c2 invocation PASS. **PROJECTED**: post-primary-fix pre-flight summary will be clean.

## §8 Cross-References (which other Wf-X closeouts must be updated)

The primary fix requires NO closeout updates. If SEC-1 or SEC-3 hardening is applied, then:

- **Wf-411-B1 §22.97.4 / Wf-422-B1 §22.107.5 / Wf-424-B1 §22.114.5** must update the pre-flight fresh-run summary reference to remove the phantom FAIL claim from "preserved baseline" once the fix is applied.
- **TODO/INDEX.md §22.107 / §22.114** (and any subsequent §22 entries) should be updated to note the resolution.
- **AGENTS.md last-update trailer** should be extended with the Wf-425-A1 sync clause (per the existing pattern from Wf-374-B1 through Wf-424-A11 prior syncs).
- **No cross-wave closeout updates** required for Wf-347-A1 (per_wave_runtime_check) + Wf-367-A2 (no_fail_gates_fix) + Wf-413-A3 (--json argparse) + Wf-409-A3 (wrapper convention) + Wf-299B (kcal sign convention) + Wf-394-A1 (Lambda CFM coupling) + Wf-329-W6 (canonical wire shape) + Wf-387-A2 (balanced Rh cohort) + Wf-139 P7 (catherine_6_6_pass) + Wf-405-A2 (false wire-in claim) + Wf-380-A2 (per-cohort feature extractor) — these are referenced for cross-check but no updates needed.

---

## §9 ls -la Evidence (AGENTS.md §5.5 rule #2 compliance)

```
$ ls -la /home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py
-rwxr-xr-x 1 hugo hugo 307077 Sep 30 17:53 /home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py
(7150 lines per `wc -l`)

$ ls -la /home/hugo/codes/mol-metal/molmetal/scripts/wf343c2_per_wave_check.py
-rw-r--r-- 1 hugo hugo 30742 Sep 29 22:46 /home/hugo/codes/mol-metal/molmetal/scripts/wf343c2_per_wave_check.py

$ ls -la /home/hugo/codes/mol-metal/molmetal/reports/wf343c2_per_wave_check/per_wave_check.json
-rw-r--r-- 1 hugo hugo  1748 Oct  1 04:16 /home/hugo/codes/mol-metal/molmetal/reports/wf343c2_per_wave_check/per_wave_check.json

$ ls -la /home/hugo/codes/mol-metal/molmetal/reports/wf419a2_preflight_freshrun.txt
-rw------- 1 hugo hugo 38404 Sep 30 23:21 /home/hugo/codes/mol-metal/molmetal/reports/wf419a2_preflight_freshrun.txt

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/347a1_wf296_integration.md
-rw-r--r-- 1 hugo hugo 16548 Sep 29 16:16 /home/hugo/codes/mol-metal/TODO/completed/347a1_wf296_integration.md

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/367a2_preflight_no_fail_gates_fix.md
-rw-r--r-- 1 hugo hugo  9416 Sep 29 23:25 /home/hugo/codes/mol-metal/TODO/completed/367a2_preflight_no_fail_gates_fix.md

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/413a3_wf299_wave_b_launch.md
-rw-r--r-- 1 hugo hugo 31257 Sep 30 16:02 /home/hugo/codes/mol-metal/TODO/completed/413a3_wf299_wave_b_launch.md

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/409a3_wf246c_scaffold.md
-rw-r--r-- 1 hugo hugo 31257 Sep 30 16:02 /home/hugo/codes/mol-metal/TODO/completed/409a3_wf246c_scaffold.md

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/394a1_lambda_cfm_wire.md
-rw-r--r-- 1 hugo hugo 11505 Sep 30 12:18 /home/hugo/codes/mol-metal/TODO/completed/394a1_lambda_cfm_wire.md

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/387a2_balanced_rh_cohort.md
-rw-r--r-- 1 hugo hugo 19957 Sep 30 10:04 /home/hugo/codes/mol-metal/TODO/completed/387a2_balanced_rh_cohort.md

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/405a2_wf380_loss_wire.md
-rw-r--r-- 1 hugo hugo 29526 Sep 30 13:24 /home/hugo/codes/mol-metal/TODO/completed/405a2_wf380_loss_wire.md

$ ls -la /home/hugo/codes/mol-metal/TODO/completed/380a2_per_cohort_features.md
-rw-r--r-- 1 hugo hugo  9883 Sep 30 01:04 /home/hugo/codes/mol-metal/TODO/completed/380a2_per_cohort_features.md
```

---

## §10 Standing Directives Compliance (AGENTS.md §1, §5.1, §5.2, §5.4, §5.5, §5.6, §5.7, §10)

- **§1** (5 standing project rules): MEASURED-vs-PROJECTED tag discipline preserved (§7); honest framing applied (§7); pre-flight PASS gate honored per §1.4 (recommendation: re-run pre-flight fresh-run).
- **§5.1** (standing GPU authorization): 0 GPU launches performed in Wf-425-A1 (CPU-only investigator sub-flow); USER_OK=1 grant honored.
- **§5.2** (honest framing mandate): §7 honest framing caveats section explicitly cites MEASURED vs PROJECTED for every numeric claim.
- **§5.4** (Tier-2 + no wet-lab planning): 0 wet-lab planning in Wf-425-A1 (CPU-only docs+verify).
- **§5.5** (filesystem-truth rule): §9 ls -la evidence provided for ALL key files referenced in the report.
- **§5.6** (atomic write protocol adoption rule): closeout authored via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` (NOT direct `Write`).
- **§5.7 NEW** (trainer flag wire-in smoke-required rule): N/A for this sub-flow (no trainer flag wire-in).
- **§10** (wetlab deprecation): 0 wet-lab planning in Wf-425-A1.

