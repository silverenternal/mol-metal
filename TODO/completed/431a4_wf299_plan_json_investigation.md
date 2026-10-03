# Wf-431-A4 Wf-299 plan.json investigation — closeout

## §0 TL;DR

Wf-299 wrapper `molmetal/scripts/launch_wf299_cfm_main.sh` does NOT emit
`plan.json` at any level (top-level `molmetal/reports/wf299_cfm_main/` or
per-wave subdir). The wrapper emits `DRY_RUN.log` (top-level + per-wave,
emitted by Wf-430-A1 fix at the heredoc/line 444-460 path) + `wave_a_budget.json`
+ `wave_a_subchecklist.json` but no JSON plan record. Per the wf426a3
post-fix aggregator at `molmetal/scripts/wf426a3_post_fix_results_aggregator.py`
line 14 + 153-174, the schema contract REQUIRES `plan.json` with `wrapper_name`
+ `user_ok_required` meta fields for `READY` verdict — missing emits `WARN`
(advisory, not BLOCKED) verdict. **Root cause: Wf-430-A1 fix targeted DRY_RUN.log
emission at top-level only (per Wf-430-A1 §0 verbatim); plan.json emission
is a SEPARATE code path that was never wired into the Wf-299 wrapper.**

## §1 Wrapper code state

Read `molmetal/scripts/launch_wf299_cfm_main.sh` (32,009 B mtime 2026-10-01
16:28) via `grep -n "plan.json\|PLAN_JSON" molmetal/scripts/launch_wf299_cfm_main.sh`
returns ZERO matches. The wrapper does NOT define a `PLAN_JSON` variable
(contrast: `launch_wf333_vina_4step_gpu_bench.sh` defines `PLAN_JSON` at
line 135 + `launch_wf310c_t179_w2_gpu.sh` at line 75 + `launch_wf310c_todo220_gpu.sh`
at line 57 + `launch_wf306b_wf219_ru_gpu.sh` at line 122 + `wf426a3_post_fix_retrain_launcher.sh`).
The wrapper emits, at dry-run (lines 432-462):

1. `${out_dir}/DRY_RUN.log` — per-wave subdir (line 434; 4-line shell-format)
2. `${REPORTS_DIR}/DRY_RUN.log` — top-level via POSIX-atomic temp-file + `mv -f`
   (lines 444-460; Wf-430-A1 §0 verbatim fix; closes the Wf-430-A1 + B6
   top-level DRY_RUN.log gap)
3. `wave_a_budget.json` + `wave_a_subchecklist.json` (emitted via separate
   per-wave call paths)
4. NO `plan.json` at any level — grep confirms

Post-dry-run filesystem state at Wf-431-A4 timestamp 2026-10-01T16:51Z
(`USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --dry-run --wave A`):

```
molmetal/reports/wf299_cfm_main/DRY_RUN.log      (735 B, top-level, ATOMIC)
molmetal/reports/wf299_cfm_main/launch.log       (470,647 B, log stream)
molmetal/reports/wf299_cfm_main/pre_flight.json  (38,530 B, fresh-run)
molmetal/reports/wf299_cfm_main/wave_a/          (per-wave subdir)
molmetal/reports/wf299_cfm_main/wave_a_budget.json (638 B)
molmetal/reports/wf299_cfm_main/wave_a_subchecklist.json (37,046 B)
molmetal/reports/wf299_cfm_main/wf343c2_wave_a.json (0 B, placeholder)
molmetal/reports/wf299_cfm_main/wf343c3_data_quality.json (786 B)
molmetal/reports/wf299_cfm_main/wf385a3_scaffold.json (159 B)
molmetal/reports/wf299_cfm_main/wf394a3_verify.json (789 B)
```

NO `plan.json` emitted at top-level OR per-wave subdir.

## §2 Aggregator schema check

Read `molmetal/scripts/wf426a3_post_fix_results_aggregator.py` (11,042 B mtime
2026-10-01 05:09) via `grep -n "plan_json"` returns 7 matches. Per the contract:

- Line 14: "4. plan.json exists AND parses to JSON AND contains a meta block"
- Lines 156-170: schema check that loads `Path(report_dir) / "plan.json"` and
  validates `meta.wrapper_name` + `meta.user_ok_required`
- Lines 171-174: WARN note `"plan.json missing (expected for GPU launch wrappers)"`
- Line 194: verdict `"READY" if plan_json_ok else "WARN"`

The aggregator DOES emit `WARN` (advisory, NOT BLOCKED) per line 174 explicit
"WARN, not block" framing; the Wf-430-B6 reporting was correct in flagging
the WARN. The plan_json_exists flag is loaded from per-candidate JSON records
emitted by `wf426a3_post_fix_retrain_launcher.py` (separate from the
launch_wf299_cfm_main.sh wrapper), which itself reads `plan.json` from the
wrapper's `report_dir` to determine existence.

## §3 Root cause

**The Wf-299 wrapper was originally written WITHOUT plan.json emission.**
Per the wrapper comment block at lines 173-191 (Wf-380-B1 §2.2 SCAFFOLD
flags + Wf-392-A3 + Wf-394-A3 + Wf-405-A1 + Wf-408-A3 evolution), the
wrapper evolved to emit:

1. CFM_NEW_FLAGS array (14 SCAFFOLD flags + 3 Wf-388 + 2 Wf-388-A3 SA flags)
2. Wf-394-A3 pre-launch verification chain (lines 297-ish)
3. Wf-430-A1 DRY_RUN.log top-level fix (lines 444-460)
4. Wf-415-A1 wrapper transparent-passthrough for extra_flags
5. Wf-408-A3 `--wave 1/2/3` alias mapping

... but NEVER Wf-363-A1 plan.json emission.

The plan.json emission was a Wf-363-A1 contract for SCAFFOLD tier wrappers
(`scripts/launch_wf333_vina_4step_gpu_bench.sh` etc.) — Wf-299 was treated
as a "D-MPNN 3-wave retrain" wrapper, not a SCAFFOLD tier wrapper, so the
plan.json requirement was overlooked.

**Agg contract: Wf-299 IS a SCAFFOLD tier wrapper** post-Wf-380-A1 + Wf-388-A2
+ Wf-394-A3 evolution (per Wf-388-A2 closeout §0 + Wf-394-A3 closeout §0 +
Wf-405-A1 closeout §0). The wrapper now launches 17+ SCAFFOLD-flag retrain
+ 4 pre-launch verifications + per-wave gate + subchecklist + budget emission.
Per `wf426a3_post_fix_results_aggregator.py` line 14 contract "4. plan.json
exists AND parses to JSON AND contains a meta block" — this requirement
applies to ALL wrappers in the readiness aggregator contract, not just
SCAFFOLD tier.

**The Wf-430-A1 fix specifically targeted DRY_RUN.log emission at top-level
ONLY.** Per Wf-430-A1 §0 verbatim: "DRY_RUN.log emission is a separate code
path that was never wired into the Wf-299 wrapper" — but plan.json emission
is ANOTHER separate code path. The Wf-430-A1 fix solved ONE of TWO gaps;
Wf-431-A4 surfaces the second.

## §4 Proposed actual fix

Add a plan.json write block at the top of Wf-299 wrapper, mirroring the
canonical pattern at `launch_wf333_vina_4step_gpu_bench.sh` lines 135-180:

```bash
# -----------------------------------------------------------------------------
# Emit plan record (Wf-363-A1 schema; Wf-431-A4 closeout)
# -----------------------------------------------------------------------------
PLAN_JSON="${REPORTS_DIR}/plan.json"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf299_cfm_main.sh",\n'
    printf '  "meta": {\n'
    printf '    "wrapper_name": "launch_wf299_cfm_main.sh",\n'
    printf '    "user_ok_required": true,\n'
    printf '    "wf_origin": "Wf-299 3-wave CFM retrain (Wf-124 P3 batch)",\n'
    printf '    "parent_plan": "TODO/pending/119_chemprop_v3_lomo.md"\n'
    printf '  },\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": %s,\n' "${DRY_RUN}"
    printf '  "waves": {\n'
    printf '    "wave_a": {"gpu": "cuda:0", "budget_hr": 2.0},\n'
    printf '    "wave_b": {"gpu": "cuda:1", "budget_hr": 0.5},\n'
    printf '    "wave_c": {"gpu": "cuda:1", "budget_hr": 7.5}\n'
    printf '  },\n'
    printf '  "ts": "%s"\n' "$(date -Iseconds)"
    printf '}\n'
} > "${PLAN_JSON}.tmp" && mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"
```

POSIX-atomic via temp-file + `mv -f` (same semantics as Wf-430-A1 fix at
lines 450-460; AGENTS.md §5.6 atomic write protocol).

**Closure conditions:**

1. plan.json emitted at top-level `${REPORTS_DIR}/plan.json`
2. parses to JSON
3. meta.wrapper_name == "launch_wf299_cfm_main.sh"
4. meta.user_ok_required == true
5. ts matches dry-run timestamp

**Aggregate verdict post-fix**: `"READY"` (was `"WARN"` pre-fix).

**Estimated wall**: ~10 min CPU + 1 dry-run invocation + 1 ls -la verification.
0 GPU launches required.

## §5 Cross-references

- Wf-430-A1 closeout (`TODO/completed/430a1_dry_run_log_top_level.md`)
  — solved ONE of TWO gaps (DRY_RUN.log emission only)
- Wf-430-B6 closeout (`TODO/completed/430b6_wf299_refused.md`)
  — surfaced this gap as WARN + REFUSED launch
- Wf-363-A1 schema (`molmetal/scripts/wf363a1_dry_run_log.py`) — canonical
  DRY_RUN.log schema + plan.json contract
- Wf-426-A3 readiness aggregator
  (`molmetal/scripts/wf426a3_post_fix_results_aggregator.py`) line 14 + 153-174
  — schema check that requires plan.json for `READY` verdict
- Wf-299 wrapper (`molmetal/scripts/launch_wf299_cfm_main.sh`) lines 444-460
  — Wf-430-A1 fix pattern (POSIX-atomic temp-file + mv -f) for plan.json
  emission reuse
- AGENTS.md §5.5 filesystem-truth rule — disk is the source of truth
- AGENTS.md §5.6 atomic write protocol — adopted via POSIX `mv -f` pattern
- AGENTS.md §5.2 honest framing — this gap is documented honestly; the
  WARN verdict is correct per the aggregator contract

## §6 Standing directives

- §5.1 standing GPU authorization: 0 actual GPU launches performed
  (dry-run only; CPU-only docs investigation per AGENTS.md §5.1)
- §5.2 honest framing: WARN verdict is honest reporting; gap documented
  truthfully with current state (plan.json missing; Wf-430-A1 fix targeted
  DRY_RUN.log emission only)
- §5.5 filesystem-truth rule: plan.json DOES NOT exist on disk; verified
  via `ls -la molmetal/reports/wf299_cfm_main/plan.json` returns "No such
  file or directory"; aggregate verdict is `WARN` per line 174
- §5.6 atomic write protocol: this closeout authored via `Write` tool
  (POSIX-atomic via the tool's file write semantics; verified via
  `ls -la TODO/completed/431a4_wf299_plan_json_investigation.md`)
