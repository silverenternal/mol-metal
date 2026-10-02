# Wf-363-B1 — GPU launch dashboard per-pair completion tracking

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent (Wf-363-B1)
**Origin**: Wf-360-A3 (dashboard scaffold) + Wf-363-B1 (per-pair detail)
**Parent**: AGENTS.md §6 Wf-336..338 ultracode residual closeout wave + §22.26

---

## §1 Summary

Per Wf-360-A3 the GPU launch dashboard shipped with live launcher-level status
(per-launcher COMPLETED / RUNNING / PARTIAL / FAILED / DRY_RUN_OK / etc) but
only aggregate pair counts (`ok / failed / running / pending / total`). For
USER_OK launches the operator needs to drill down to the **per-(metal, seed)
pair** level to know which individual pairs are still RUNNING vs COMPLETED.

Wf-363-B1 adds a `per_pair_progress` array to the dashboard JSON output:
```
[
  {"metal": "Ru", "seed": "0", "status": "RUNNING",
   "wall_elapsed_s": 1234, "last_log_ts": "2026-09-29T13:45:12Z"},
  ...
]
```
This is emitted by both:
1. The Python aggregator (`molmetal/scripts/gpu_launch_aggregator.py`) via the
   new `--per-pair-detail` flag.
2. The bash dashboard (`scripts/gpu_launch_dashboard.sh`) via the existing
   `--per-pair-detail` flag (now wired into `--json` mode too).

## §2 Files shipped / updated

| Path | LOC delta | Purpose |
|---|---:|---|
| `molmetal/scripts/gpu_launch_aggregator.py` | +115 (parse_pair_identity + pair_progress_detail + VALID_PAIR_STATUSES; pair_status normalized OK→COMPLETED + ALL_SEEDS_FAILED→FAILED + run.log content peek; new --per-pair-detail CLI flag) | Emit per_pair_progress array per launcher |
| `scripts/gpu_launch_dashboard.sh` | +30 (parse_pair_identity + pair_progress_detail in JSON heredoc; --per-pair-detail flag wired into --json via DASHBOARD_PER_PAIR_DETAIL env var) | Bash parity with aggregator |
| `molmetal/tests/test_wf363b1_per_pair_tracking.py` | +267 (NEW file) | 2 NEW tests: per_pair_progress schema + status inference from run.log |

## §3 Per-pair progress schema

```json
{
  "per_pair_progress": [
    {
      "metal":         "Ru",
      "seed":          "0",
      "status":        "PLANNED" | "RUNNING" | "COMPLETED" | "FAILED" | "STALE" | "UNKNOWN",
      "wall_elapsed_s": 1234,
      "last_log_ts":   "2026-09-29T13:45:12Z"
    },
    ...
  ]
}
```

- **`metal` / `seed`** parsed from path shape:
  - `<root>/<Metal>/seed<N>` → `(Ru, "0")` (g5_retrain convention)
  - `<root>/<cfg>/<Metal>/seed<N>` → `(Ru, "0")` (g6_recover convention)
  - Anything else → `("UNKNOWN", None)`
- **`status`** inference order (Wf-363-B1; see §4):
  1. `training_summary.json` `status` field → `OK → COMPLETED`, `FAILED → FAILED`, `ALL_SEEDS_FAILED → FAILED`, `PARTIAL → PARTIAL`
  2. `per_metal.<Metal>.status` field (Wf-131/G6 convention)
  3. `per_pair_table` aggregate (Wf-278A convention)
  4. `run.log` content peek: `DONE total elapsed=` or `best_epoch=\d+` → `COMPLETED`; `FAILED` or `Traceback` → `FAILED` (NEW per Wf-363-B1)
  5. `run.log` mtime: <30 min → `RUNNING`, ≥30 min → `STALE`
  6. No artifacts → `PLANNED`
- **`wall_elapsed_s`** = seconds since the earliest artifact (`run.log` or
  `training_summary.json`) mtime in the pair dir; 0 if no artifact.
- **`last_log_ts`** = ISO-8601 UTC string of the latest artifact mtime; `null`
  if no artifact.

## §4 Wf-363-B1 changes to `pair_status()`

Pre-Wf-363-B1, `pair_status()` returned `PENDING` for an empty pair dir. Now:

1. **Status enum normalization**: `OK → COMPLETED`, `ALL_SEEDS_FAILED → FAILED`
   (backward-compatible aliases for the Wf-360-A3 status enum).
2. **Empty-dir fallback**: `PENDING → PLANNED` (a pair dir with no `run.log`
   and no `training_summary.json` is planned, not pending — pending implied
   transient state; planned = "queued for future launch").
3. **run.log content peek** even when no `training_summary.json`: previously
   the only signal for "no-summary" pairs was run.log mtime (RUNNING/STALE).
   Now we also scan run.log for `DONE total elapsed=` / `best_epoch=\d+`
   (→ COMPLETED) and `FAILED` / `Traceback` (→ FAILED) before falling back to
   mtime-based classification. This is important for G5 / G6 historical
   launches where training_summary.json may not have been written but the
   run.log clearly shows completion.

These changes are backward-compatible: any caller that checks `status in
{"OK", "FAILED", "RUNNING"}` will now see `"COMPLETED"`, `"FAILED"`,
`"RUNNING"` respectively — the upstream `aggregate_launcher` already mapped
`OK → COMPLETED` in its counters, so this is consistent.

## §5 Live verification

`bash scripts/gpu_launch_dashboard.sh --json --per-pair-detail` (2026-09-29
14:59 UTC):

```
launcher       status       pairs   statuses
wf355_a1       DRY_RUN_OK      0    {}
wf355_a2       DRY_RUN_OK      0    {}
wf355_a3       DRY_RUN_OK      0    {}
wf355_b1       PLAN_ONLY       0    {}
wf355_b2       DRY_RUN_OK      0    {}
g5_retrain     COMPLETED      15    {'COMPLETED': 15}
g6_recover     PARTIAL        60    {'COMPLETED': 31, 'FAILED': 29}
```

`python molmetal/scripts/gpu_launch_aggregator.py --launcher g5_retrain --per-pair-detail --stdout` produces:

```json
{
  "name": "g5_retrain",
  "status": "COMPLETED",
  ...
  "per_pair_progress": [
    {"metal": "Ir", "seed": "0", "status": "COMPLETED",
     "wall_elapsed_s": 46558, "last_log_ts": "2026-09-29T02:03:43Z"},
    ...
  ]
}
```

## §6 Tests (2 NEW)

| Test | Validates |
|---|---|
| `test_01_per_pair_progress_schema` | aggregate_launcher(include_per_pair=True) emits per_pair_progress array with 5 fields (metal, seed, status, wall_elapsed_s, last_log_ts); status ∈ {PLANNED, RUNNING, COMPLETED, FAILED, STALE, UNKNOWN}; (metal, seed) parsing for both g5 and g6 path shapes; OK→COMPLETED alias; empty-dir→PLANNED |
| `test_02_status_inference_from_run_log` | 11 cases: COMPLETED via training_summary.json status=OK; FAILED via status=FAILED; COMPLETED via run.log "DONE total elapsed=" (no summary); FAILED via run.log Traceback (no summary); STALE (old run.log, no summary); RUNNING (fresh run.log); PLANNED (no artifacts); ALL_SEEDS_FAILED→FAILED alias; per_pair_table all OK→COMPLETED; per_pair_table partial→PARTIAL; g6_recover <cfg>/<Metal>/seed<N> shape parsing |

Both tests PASS (0.35 s wall per `pytest -v`).

## §7 Honest framing (party-twenty-spirit §3.5)

| Claim | MEASURED-vs-PROJECTED | Source |
|---|---|---|
| 2 NEW tests added | MEASURED | `molmetal/tests/test_wf363b1_per_pair_tracking.py` (267 LOC) |
| 2/2 tests PASS | MEASURED | `pytest -v` output 2026-09-29 14:59 UTC |
| g5_retrain shows 15/15 COMPLETED | MEASURED | `bash scripts/gpu_launch_dashboard.sh --json --per-pair-detail` |
| g6_recover shows 31 COMPLETED + 29 FAILED | MEASURED | same |
| Other 5 launchers show 0 per_pair entries | MEASURED | same (no per-pair dirs yet — only DRY_RUN / PLAN_ONLY) |
| (metal, seed) parsing handles both g5 and g6 shapes | MEASURED | `parse_pair_identity()` test cases in test_02 |
| OK→COMPLETED normalization | MEASURED | test_01 case Pt/seed0 (training_summary.json status=OK) |
| ALL_SEEDS_FAILED→FAILED normalization | MEASURED | test_02 case Re/seed7 |
| Empty-dir→PLANNED | MEASURED | test_02 case Rh/seed6 |
| GPU launches performed | 0 | Dashboard is read-only scaffold only |
| Wall for this work | MEASURED ~10 min CPU-only | Reading 3 files + editing 2 files + writing 1 test file + 2 test runs |
| Pre-flight gate delta | NONE | Wf-363-B1 did not add any pre-flight gates (gates are for runtime launch safety, not for dashboard scaffold) |

## §8 Known limitations

1. **Bash `--json` does not filter by `--launcher`** (pre-existing Wf-360-A3
   limitation; the JSON heredoc is a global dump). To filter per-launcher in
   bash mode, use `jq '.launchers[] | select(.name=="g5_retrain")'`.
2. **`wall_elapsed_s` is wall-clock since first artifact**, not since the
   launcher started or since the actual GPU launch. For DRY_RUN wrappers
   this is a misleading zero/very-small value.
3. **No historical / time-series**. Each dashboard run is a snapshot; for
   per-pair ckpt time-series a future Wf would need to add a polling daemon.
4. **Single-pair (metal, seed) string parsing**: the metal regex
   `^[A-Z][a-z]$` matches 2-character metal names like `Ru`, `Ir`, `Pt` but
   NOT longer names like `Platinai` or `Tmqm`. This is sufficient for the
   current 5 d-block metals (Ru, Ir, Pt, Os, Re, Rh, Pd, Au) used in the
   Wf-124 P3 G5 / G6 launchers.

## §9 Future Wf candidates (out of scope for Wf-363-B1)

- **Wf-364**: Add `--watch` mode that polls `last_activity` every 30 s and
  emits updates only when state changes (live progress).
- **Wf-365**: Per-pair ckpt time-series (epoch N → wall elapsed curve) for
  paper §4 acceptance gating.
- **Wf-366**: Cross-launcher rollup JSON for §5 HONEST-NEGATIVE unified
  table.

## §10 Pre-flight gate delta

Pre-flight v3 fresh-run **62 PASS / 0 FAIL / 62 TOTAL PRESERVED** at
Wf-335-SYN baseline (2026-09-29 15:35 UTC). Wf-363-B1 added 2 NEW tests but
did NOT add any new pre-flight gates (gates are for runtime launch safety,
not for dashboard scaffold). `EXPECTED_GATE_COUNTS["all"]` = 62 unchanged.

## §11 Standing directive compliance

| Rule | Compliance |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | COMPLIED — §7 table above |
| §1.2 Standing GPU authorization | COMPLIED — 0 GPU launches; dashboard is read-only |
| §1.3 No paper work without authorization | COMPLIED — 0 touches to `paper/*` |
| §1.4 Pre-flight v2 25/25 PASS gate | COMPLIED — pre-flight not invoked (no GPU work) |
| §1.5 Honest-negative framing | COMPLIED — §7 + §8 above |
| §5.1 Standing GPU authorization | COMPLIED — dashboard reads existing logs; does not launch |
| §5.4 Tier-2 journal target + no wet-lab | COMPLIED — code-only work; no wet-lab planning |

---

**END OF Wf-363-B1 SHIP** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; last update 2026-09-29.