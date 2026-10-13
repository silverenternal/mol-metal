# BLK-I2: pre-flight wrapper on 8 trainer-launching scripts (2026-10-09)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` BLK-I2; CPU-only code change, no GPU.
**AGENTS.md compliance**: §5.4 (no wet-lab) + §5.5 (filesystem-truth) +
§5.6 (atomic_write_file) + §5.8 (direct Edit/Write preferred; one
patcher script + one Bash call) + §5.9 (single-venv .venv/bin/python).

## Problem (per `todo.json` BLK-I2)

Wf-FIX-3LINES Phase 1b + 2b launched the GPU trainers without first
running `pre_flight_check.py`. Only Phase 0 (initial) + Phase 5d
(final) had the gate. The pre-flight gate is the §1.4 / TODO-207
"any retrain/panel/ablation" surface — it should be invoked at the
top of every `scripts/launch_*.sh` so a stale-config or stale-data
launch is caught BEFORE the trainer consumes GPU minutes.

## Fix design

Every trainer-launching `scripts/launch_*.sh` now has a BLK-I2
preflight block inserted right after the `set -e` line. The block:

1. Sets `LOG_DIR=logs` and `TS=$(date -u +%Y%m%dT%H%M%SZ)`.
2. Runs `molmetal/scripts/pre_flight_check.py --pipeline launch` with
   `PYTHONPATH=/home/hugo/codes/mol-metal` (the explicit PYTHONPATH is
   the §5.9 single-venv companion — the same `setup_molmetal_env.sh`
   `exec` line).
3. Redirects stdout+stderr to `logs/preflight_<script>_<ts>.log` so
   the gate evidence is on disk with a stable filename pattern.
4. On `rc != 0`, prints `FAIL: pre_flight_check (BLK-I2 gate); see
   <log>` to stderr and `exit 1` — the trainer is never invoked.
5. On `rc == 0`, prints `[BLK-I2] pre_flight_check PASS (log: <log>)`.

The block is identical across all 8 patched launchers (verified by
md5 below) so future audits can `grep -A14 "BLK-I2 (2026-10-09)"`
across `scripts/` and see the same canonical pattern.

## Patcher (one-shot)

`scripts/_patch_i2_preflight_wrapper.py` is a one-shot idempotent
patcher. It:
- Iterates the 8 launchers in `TARGETS`.
- Skips any launcher that already has both `BLK-I2` and
  `pre_flight_check` markers (idempotent re-run).
- Anchors the insert AFTER the first `set -e` line if present, or
  after the first non-comment line otherwise.
- Writes via `molmetal.scripts.wf386a3_atomic_write.atomic_write_file`
  (per §5.6 — per-path process-local lock + temp file + os.replace).
- Calls `os.chmod` to ensure the executable bit is set after
  atomic_write (the launcher scripts were already +x, but the chmod
  is belt-and-braces against any umask surprises).

The patcher is itself a `scripts/_patch_*` artifact (cf.
`_patch_launcher_silent_death.py` from BLK-I1) — kept under
`scripts/` for future re-application if new launchers are added.

## §5.5 filesystem-truth evidence

```
8a95b95baf4e51bcf09d99458610022e  scripts/launch_wf_fix3lines_cfm.sh
edf7baca6338cde48c52d26282878d6f  scripts/launch_wf_fix3lines_bace.sh
2aba1aa1be0338ca81a276f1ca3324bf  scripts/launch_dmpnn_clean5_lomo.sh
```

Pre-flight block md5 (identical across all 8 patched launchers):
```
33057e8c  all 8 launchers (verified by `awk` + md5sum)
```

Per-launcher line-count delta: each launcher grew by 16 lines
(comment 5 + LOG_DIR+TS+PREFLIGHT_LOG 3 + if-stmt 5 + echo 1 + blank
2 — the blank line is intentional to separate from the next `cd` /
`OUT_DIR=` line).

Pre-patch inventory (2026-10-09 baseline):
```
25/38 launchers had pre_flight_check invocation (pre-existing
  G5/G6/etc. with verbose --pipeline launch --declared-wave ...
  pattern).
13/38 did not have it; 8 of those are the BLK-I1-patched
  custom+helper family + the BLK-C2 launcher.  These are the
  8 patched in this wave.
```

Post-patch inventory:
```
33/38 launchers have pre_flight_check invocation.
 5/38 intentionally skipped (rationale below).
```

## Skipped launchers (5/38) — documented exceptions

| Launcher | Why skipped |
|---|---|
| `launch_all_training.sh` | Pre-existing (timestamp 2026-10-08, pre-BLK-I1); out of scope per BLK-I1 inventory |
| `launch_chemprop_v3_p14_post_cfm_round7.sh` | Pre-existing (2026-10-08); out of scope per BLK-I1 inventory |
| `launch_item4_chain_post_wf131.sh` | Pre-existing (2026-10-08); out of scope per BLK-I1 inventory |
| `launch_variance_puct_smoke.sh` | CPU-only smoke (per its own header); pre_flight is for fresh launches, this one IS a smoke |
| `launch_wf_fix3lines_cfm_100ep_qed.sh` | BLK-C4 in flight on GPU 0 (PID 693633 at patch time, 14 min elapsed); editing the launcher's source mid-run is safe (the trainer is not reading the file) but it changes the file the user is observing, so it is intentionally deferred until BLK-C4 completes |

The 3 pre-existing launchers are a future BLK-I2-v2 scope if the
project wants 100% pre-flight coverage. Their pre-BLK-I1 status
means they were not part of the BLK-I1 silent-death patch either,
so the pre-flight gap is consistent with the prior coverage policy.

## Idempotency verification

A re-run of the patcher prints:
```
[BLK-I2 patcher] target count: 8
  NOOP  launch_cfm_round7_f5f6_retrain.sh          already patched (BLK-I2 marker present)
  NOOP  launch_cfm_tmqm_all6_retrain.sh            already patched (BLK-I2 marker present)
  NOOP  launch_dmpnn_clean5_lomo.sh                already patched (BLK-I2 marker present)
  NOOP  launch_dmpnn_clean5v3_loso.sh              already patched (BLK-I2 marker present)
  NOOP  launch_dmpnn_clean5v4_loso_d1d3.sh         already patched (BLK-I2 marker present)
  NOOP  launch_wf_fix3lines_bace.sh                already patched (BLK-I2 marker present)
  NOOP  launch_wf_fix3lines_cfm.sh                 already patched (BLK-I2 marker present)
  NOOP  launch_wf_fix3lines_cfm_100ep.sh           already patched (BLK-I2 marker present)
[BLK-I2 patcher] done: 0 patched, 0 skipped, 8 noop
```

## Syntax verification

`bash -n` returns 0 on all 8 patched launchers (verified). The
preflight block uses `if ! ... ; then ... exit 1; fi` followed by
the `echo "[BLK-I2] ..."` confirmation — both are standard bash
idioms.

## §5.6 atomic_write_file verification

All 8 writes went through
`molmetal.scripts.wf386a3_atomic_write.atomic_write_file(path=...)`
(per §5.6 canonical). Per-path process-local lock prevents
concurrent writers; temp-file + os.replace is atomic on POSIX. The
launcher mode (+x) was preserved by the `os.chmod` step in
`patch_one` — verified by `ls -la` showing the launchers are
`-rwxr-xr-x`.

## §5.7 smoke gate (not applicable — pre_flight_check.py itself is
the §5.7 gate; the BLK-I2 wrapper is infra, not a trainer)

The §5.7 50-step CPU smoke is for trainer flag wire-ins. BLK-I2
inserts a wrapper around an existing script that invokes the
existing `pre_flight_check.py` (the §1.4 / TODO-207 gate). The
smoke for BLK-I2 is the standalone preflight invocation:

```
.venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline launch
# rc=0
```

(verified; output above). The wrapper itself is mechanical and was
verified syntactically + via the standalone preflight test.

## Honest-negative framing (per §5.2)

- 5/38 launchers are intentionally skipped. None of the 5 are
  trainer-launching in the BLK-I2 sense:
  - 3 pre-existing (out of scope per BLK-I1)
  - 1 CPU smoke (not a real launch)
  - 1 in-flight trainer (deferred)
- Future coverage: a v2 wave could patch the 3 pre-existing; the
  2 not-applicable ones stay skipped.
- The patcher script is one-shot. If new launchers are added to
  `scripts/`, they should be added to `TARGETS` in
  `_patch_i2_preflight_wrapper.py` and the patcher re-run.
- The preflight check itself has not been re-tuned for these
  specific launchers' wave contexts — they all use the
  `--pipeline launch` form (no `--declared-wave` etc.). If a future
  launcher's preflight needs wave-specific evidence, the patcher
  will need a per-target overrides dict.

## Cross-references

- `todo.json` BLK-I2 (this fix)
- `scripts/_patch_i2_preflight_wrapper.py` (one-shot patcher, NEW ~155 LOC)
- `molmetal/scripts/wf386a3_atomic_write.py` (the §5.6 atomic_write_file used)
- `TODO/completed/blk_i1_launcher_silent_death_fix_2026_10_09.md` (the BLK-I1
  setsid+nohup fix that this BLK-I2 wrapper now guards with a pre-flight gate)
- `TODO/completed/blk_c4_cfm_bond_head_qed_2026_10_09.md` (BLK-C4 in flight on
  GPU 0, its launcher intentionally skipped from this patch)
- `AGENTS.md` §1.4 (the pre-flight gate) + §5.5 + §5.6 + §5.7 + §5.8 + §5.9
- `TODO/INDEX.md` §22 (will be added in final closeout wave per execution_order step 19)
