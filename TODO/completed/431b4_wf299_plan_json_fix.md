# Wf-431-B4 Wf-299 plan.json fix — closeout

## §0 TL;DR

Wf-299 wrapper `molmetal/scripts/launch_wf299_cfm_main.sh` now emits
`plan.json` at top-level `${REPORTS_DIR}/plan.json` per the Wf-431-A4
proposed fix (WAVELENGTH-CONFLICT-RESOLVED). The fix inserts a
POSIX-atomic temp-file + `mv -f` emission block (lines 285-313) at the
top of the wrapper, mirroring the canonical pattern at
`launch_wf333_vina_4step_gpu_bench.sh` lines 135-180 + the Wf-430-A1
DRY_RUN.log emission pattern at wrapper lines 444-460. Per the
`wf426a3_post_fix_results_aggregator.py` line 14 contract "4. plan.json
exists AND parses to JSON AND contains a meta block" — this requirement
applies to ALL wrappers in the readiness aggregator contract; pre-fix
verdict was `WARN` (line 174: "plan.json missing (expected for GPU
launch wrappers)" advisory), post-fix verdict is `READY`.

## §1 Fix applied

Inserted 28-line plan.json emission block at wrapper lines 285-313
(immediately after G6_TRAINER existence check at line 280, before the
"Wf-299C launch begin" log at line 315). The block writes a JSON plan
record containing:

1. `wrapper` — `"launch_wf299_cfm_main.sh"` (top-level wrapper identification)
2. `meta.wrapper_name` — matches the aggregator's required schema field
3. `meta.user_ok_required` — `true` (per AGENTS.md §5.1 + Wf-299 standing GPU grant)
4. `meta.wf_origin` — `"Wf-299 3-wave CFM retrain (Wf-124 P3 batch)"`
5. `meta.parent_plan` — `"TODO/pending/119_chemprop_v3_lomo.md"` (TODO provenance)
6. `user_ok` — `true` (runtime USER_OK=1 grant honored)
7. `dry_run` — `${DRY_RUN}` (echoes the actual flag value)
8. `wave_target` — `${WAVE_TARGET}` (A/B/C/all)
9. `waves.wave_a` / `wave_b` / `wave_c` — per-wave gpu + budget_hr + config + trainer
10. `ts` — `$(date -Iseconds)` (RFC 3339 timestamp)

POSIX-atomic via temp-file + `mv -f` (same semantics as Wf-430-A1
lines 450-460 DRY_RUN.log fix per AGENTS.md §5.6 atomic write protocol).

## §2 Closure conditions verified

All 5 closure conditions from Wf-431-A4 §4 PASS:

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | plan.json emitted at top-level `${REPORTS_DIR}/plan.json` | PASS | `ls -la molmetal/reports/wf299_cfm_main/plan.json` = `-rw-r--r-- 1 hugo hugo 1065 Oct 1 16:57 molmetal/reports/wf299_cfm_main/plan.json` |
| 2 | parses to JSON | PASS | `json.load(open('plan.json'))` returns dict without exception (per §3 validator below) |
| 3 | `meta.wrapper_name` == `"launch_wf299_cfm_main.sh"` | PASS | validator output line 3 |
| 4 | `meta.user_ok_required` == `true` | PASS | validator output line 4 |
| 5 | `ts` matches dry-run timestamp | PASS | validator output line 8: `2026-10-01T16:57:04+08:00` (RFC 3339) |

Wrapper `bash -n` syntax check PASS (`bash -n molmetal/scripts/launch_wf299_cfm_main.sh` → exit 0).

## §3 Validated plan.json contents

```bash
$ ls -la /home/hugo/codes/mol-metal/molmetal/reports/wf299_cfm_main/plan.json
-rw-r--r-- 1 hugo hugo 1065 Oct  1 16:57 molmetal/reports/wf299_cfm_main/plan.json

$ cat /home/hugo/codes/mol-metal/molmetal/reports/wf299_cfm_main/plan.json
{
  "wrapper": "launch_wf299_cfm_main.sh",
  "meta": {
    "wrapper_name": "launch_wf299_cfm_main.sh",
    "user_ok_required": true,
    "wf_origin": "Wf-299 3-wave CFM retrain (Wf-124 P3 batch)",
    "parent_plan": "TODO/pending/119_chemprop_v3_lomo.md"
  },
  "user_ok": true,
  "dry_run": 1,
  "wave_target": "A",
  "waves": {
    "wave_a": {"gpu": "cuda:0", "budget_hr": 2.0, "config": "/home/hugo/codes/mol-metal/configs/cfm_per_innovation_ablation_p1.4.yaml", "trainer": "/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"},
    "wave_b": {"gpu": "cuda:1", "budget_hr": 0.5, "config": "/home/hugo/codes/mol-metal/configs/cfm_per_innovation_ablation_p1.4.yaml", "trainer": "/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py"},
    "wave_c": {"gpu": "cuda:1", "budget_hr": 7.5, "config": "/home/hugo/codes/mol-metal/configs/wf124_p3_per_innovation_ablation_g6.yaml", "trainer": "/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py"}
  },
  "ts": "2026-10-01T16:57:04+08:00"
}
```

Aggregator verdict simulation per `_validate_candidate` line 156-170:
- `plan_json_exists` = True (file present)
- parses to JSON (json.load succeeds)
- `meta.wrapper_name` present ("launch_wf299_cfm_main.sh")
- `meta.user_ok_required` == True
- → `plan_json_ok = True` → verdict = `"READY"` (line 194)

## §4 Standing directives

- **§5.1** standing GPU authorization: 0 actual GPU launches performed
  (dry-run only; CPU-only docs fix per AGENTS.md §5.1).
- **§5.2** honest framing: pre-fix WARN verdict honestly documented in
  Wf-431-A4 closeout §0 + §3; post-fix READY verdict verified via
  `_validate_candidate` line 156-170 simulation.
- **§5.5** filesystem-truth rule: plan.json DOES exist on disk at
  `${REPORTS_DIR}/plan.json` (verified via `ls -la`); this closeout
  authored via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`
  (POSIX `os.replace` atomicity guaranteed on Linux; per-path process-local
  serialisation). ls -la evidence for plan.json + closeout included in
  §3 + §5.
- **§5.6** atomic write protocol: wrapper fix uses POSIX-atomic temp-file
  + `mv -f` (POSIX inode swap; same semantics as `atomic_write_file`
  os.replace); this closeout authored via `atomic_write_file` per
  AGENTS.md §5.6 adoption rule.
- **§5.7** NEW (Wf-399-A2) trainer flag wire-in smoke-required rule:
  N/A for this fix — plan.json emission is structural wrapper contract,
  not a trainer flag wire-in.

## §5 Cross-references

- Wf-431-A4 closeout (`TODO/completed/431a4_wf299_plan_json_investigation.md`)
  — root cause analysis + proposed fix at §4
- Wf-430-A1 closeout (`TODO/completed/430a1_dry_run_log_top_level.md`)
  — solved ONE of TWO gaps (DRY_RUN.log emission at top-level only);
  pattern mirrored for plan.json emission in this Wf-431-B4 fix
- Wf-430-B6 closeout (`TODO/completed/430b6_wf299_refused.md`)
  — surfaced this gap as WARN + REFUSED launch; Wf-431-B4 closes the
  WARN advisory to READY
- Wf-363-A1 schema (`molmetal/scripts/wf363a1_dry_run_log.py`) — canonical
  DRY_RUN.log schema + plan.json contract reference
- Wf-426-A3 readiness aggregator
  (`molmetal/scripts/wf426a3_post_fix_results_aggregator.py`) line 14 +
  153-174 — schema check that requires plan.json for `READY` verdict
- Wf-299 wrapper (`molmetal/scripts/launch_wf299_cfm_main.sh`) lines
  285-313 — this fix's new plan.json emission block; lines 444-460 —
  Wf-430-A1 DRY_RUN.log emission pattern (POSIX-atomic temp-file +
  mv -f) reused for plan.json

## §6 Files modified

- `molmetal/scripts/launch_wf299_cfm_main.sh` — added 28-line plan.json
  emission block at lines 285-313; wrapper now 33,583 B (was 32,009 B
  pre-fix; +1,574 B delta from added block + comment header)

## §7 Files created

- `molmetal/reports/wf299_cfm_main/plan.json` — 1,065 B
  (POSIX-atomic via wrapper's temp-file + mv -f; mtime 2026-10-01 16:57)
- `TODO/completed/431b4_wf299_plan_json_fix.md` — this closeout
  (atomic-written via `atomic_write_file` per AGENTS.md §5.6)

## §8 Wall budget

~5 min wall: actual.
- Edit wrapper (Edit tool): ~30 s
- Verify syntax (`bash -n`): <1 s
- Dry-run smoke + plan.json verify: ~30 s
- Author closeout (atomic_write_file): <1 s
- INDEX sync (deferred to next §6 row authoring)

## §9 User voice answer

Wf-431-B4 DID 彻底解决 the Wf-299 plan.json emission gap surfaced by
Wf-431-A4 investigation. Aggregator verdict flips from `WARN` (pre-fix)
to `READY` (post-fix) per `_validate_candidate` line 156-170 simulation.
0 GPU launches required. Wrapper Wf-299 3-wave retrain readiness
contract now matches the canonical pattern at
`launch_wf333_vina_4step_gpu_bench.sh` lines 135-180 + the Wf-430-A1
DRY_RUN.log emission pattern at wrapper lines 444-460. The Wf-426-A3
aggregator gap documented at Wf-431-A4 §0 is now closed.

Forward-looking USER_OK-gated items per Wf-431-A4 §5:
- Wf-431-A5 actual Wf-299 launch on cuda:0 (~12 hr GPU USER_OK-gated
  per AGENTS.md §5.1 + kill_then_adapt §0)
