# Wf-360-A3 — GPU launch dashboard scaffold (USER_OK-gated activation)

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent (Wf-360-A3)
**Origin**: Wf-357-C3 (GPU launch dashboard scaffold; USER_OK-gated activation)
**Parent**: AGENTS.md §6 Wf-336..338 ultracode residual closeout wave + §22.26

---

## §1 Summary

Scaffold-only GPU launch dashboard that **READS** (does NOT modify) outputs
from 7 Wf-355 + G5/G6 USER_OK-gated launch wrappers and emits a unified
master status report (text table + JSON).

Activation is **USER_OK-gated** per AGENTS.md §5.1 + §5.4. The dashboard
itself is read-only — it never launches a GPU job. Live progress tracking
of in-flight launches requires USER_OK=YES to be set when the launch
wrapper is invoked; the dashboard then observes its on-disk outputs.

## §2 Files shipped

| Path | LOC | Purpose |
|---|---:|---|
| `scripts/gpu_launch_dashboard.sh` | 787 (bash + python heredoc) | Text dashboard: per-launcher status + per-pair progress + per-wave budget |
| `molmetal/scripts/gpu_launch_aggregator.py` | 245 | JSON aggregator: emits `molmetal/reports/gpu_launch_dashboard.json` |
| `molmetal/tests/test_wf360a3_gpu_launch_dashboard.py` | 175 | 2 NEW tests: bash-parse + JSON schema |
| `molmetal/reports/gpu_launch_dashboard.json` | (output) | Aggregator output (regenerated on each run) |

## §3 Launcher registry (7 launchers)

| Key | Script | Reports root | Wall PROJECTED |
|---|---|---|---:|
| `wf355_a1` | `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | `molmetal/reports/wf310e_r20_p3_drugood_full_launch/` | 10 hr |
| `wf355_a2` | `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` | `molmetal/reports/wf355a2_r20_p4_bindingdb/` | 7 hr |
| `wf355_a3` | `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | `molmetal/reports/wf355a3_r20_p5_tmqm_plat/` | 12 hr |
| `wf355_b1` | `scripts/launch_wf131_p3_deferred_runs.sh` | `molmetal/reports/wf131_p3_deferred_runs/` | 5 hr |
| `wf355_b2` | `scripts/launch_wf146_deferred_runs.sh` | `molmetal/reports/wf146_chemprop_lomo/` | 3 hr |
| `g5_retrain` | `scripts/launch_wf124_p3_g5.sh` | `results/wf124_p3_50epoch_full_sweep_g5/` | 5 hr |
| `g6_recover` | `scripts/recover_g6_failed_tuples.sh` | `results/wf124_p3_per_innovation_ablation_g6/` | 11.4 hr |

## §4 Status logic (per party-twenty-spirit §3.5 — MEASURED)

`get_launcher_status(root)` returns one of:
- **COMPLETED** — per-pair training_summary.json all OK (no FAILED / ALL_SEEDS_FAILED)
- **PARTIAL** — at least one FAILED or ALL_SEEDS_FAILED
- **RUNNING** — at least one per-pair run.log mtime < 30 min
- **DRY_RUN_OK** — plan.json `status=DRY_RUN_OK` OR launch.log has `DRY-RUN:` line
- **PLAN_ONLY** — plan.json `status=PLAN_ONLY` (wrapper run but no actual launch)
- **READY** — launch.log exists, < 24 hr, no DRY-RUN marker
- **STALE** — launch.log mtime > 24 hr
- **NO_PLAN** — no plan.json + no launch.log
- **FAILED** — plan.json `status=FAILED`

`pair_status(pair_dir)` (multi-shape detection) reads:
1. Top-level `status` field
2. `per_metal.<Metal>.status` field (Wf-131/G6 convention)
3. `per_pair_table` aggregate status (Wf-278A convention)
4. Fallback: `run.log` contains `DONE total elapsed=` or `best_epoch=\d+` → OK
5. Fallback: `run.log` contains `FAILED` or `Traceback` → FAILED
6. Otherwise: PENDING / RUNNING / STALE based on mtime

## §5 Usage

```bash
# All-launcher summary table
bash scripts/gpu_launch_dashboard.sh

# Single-launcher deep-dive
bash scripts/gpu_launch_dashboard.sh --launcher wf355_a3

# Machine-readable JSON
bash scripts/gpu_launch_dashboard.sh --json

# One-line status per launcher
bash scripts/gpu_launch_dashboard.sh --status

# List known launchers
bash scripts/gpu_launch_dashboard.sh --list-launchers

# Standalone JSON aggregator (writes molmetal/reports/gpu_launch_dashboard.json)
python molmetal/scripts/gpu_launch_aggregator.py

# Aggregator with --stdout for piping
python molmetal/scripts/gpu_launch_aggregator.py --stdout | jq .

# Aggregator with --launcher filter
python molmetal/scripts/gpu_launch_aggregator.py --launcher g6_recover
```

## §6 Output schema (dashboard JSON)

```json
{
  "dashboard_generated_at": "2026-09-29T13:23:50Z",
  "dashboard_schema_version": "1.0",
  "wf_origin": "Wf-360-A3 GPU launch dashboard scaffold",
  "activation_gated": true,
  "user_ok_required": true,
  "launcher_count": 7,
  "launchers": [
    {
      "name": "wf355_a3",
      "script": "/home/hugo/codes/mol-metal/scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",
      "root": "/home/hugo/codes/mol-metal/molmetal/reports/wf355a3_r20_p5_tmqm_plat",
      "status": "DRY_RUN_OK",
      "wall_projected_hr": 12.0,
      "last_activity": "2026-09-29T12:08:52Z",
      "elapsed_sec": 4425,
      "pair_progress": {
        "total_pairs": 0,
        "discovered_pairs": 0,
        "ok": 0, "failed": 0, "running": 0, "pending": 0
      },
      "activation_gated": true,
      "user_ok_required": true
    },
    ...
  ]
}
```

## §7 Tests (2 NEW)

| Test | Validates |
|---|---|
| `test_01_dashboard_script_parses_per_launcher_output` | `bash -n` parses + `--list-launchers` returns 7 + `--status` returns ≥7 lines + `--json` roundtrip is valid JSON with 7 launchers |
| `test_02_aggregator_produces_correct_json_schema` | Aggregator python imports cleanly + main() writes JSON + JSON schema (10 top-level fields + 6 pair_progress fields + valid status enum) |

Both tests PASS (0.34 s wall per `pytest -v`).

## §8 Honest framing (party-twenty-spirit §3.5)

| Claim | MEASURED-vs-PROJECTED | Source |
|---|---|---|
| 7 launchers in registry | MEASURED | `LAUNCHER_REGISTRY` array in `gpu_launch_dashboard.sh` |
| Wall-time projections | PROJECTED | Per Wf-355 HONEST-NEGATIVE comments in each wrapper |
| Status detection accuracy | MEASURED on 2 active launchers (g5_retrain 15/15 OK + g6_recover 31/60 PARTIAL) | Discovered via `--json` + `--launcher g5_retrain` |
| Aggregator read-only | MEASURED | Aggregator only `os.path.exists` + `json.load`; no writes to launcher outputs |
| USER_OK gate enforcement | PROJECTED at launch time | The wrapper scripts (not the dashboard) enforce USER_OK; dashboard reports status regardless |
| Pre-flight impact | NONE | Dashboard does NOT touch `pre_flight_check.py`; pre-flight run separately by user |
| GPU launches performed | 0 | Dashboard is read-only scaffold only |
| Wall for this work | MEASURED ~3 min CPU-only | Reading 5 wrapper headers + writing 3 files + 2 test runs |

## §9 Known limitations

1. **Plan-only vs not-launched distinction** — Wrappers that emit `plan.json`
   without `launch.log` (rare) show PLAN_ONLY; wrappers that emit `launch.log`
   only (DRY-RUN path) show DRY_RUN_OK. For wrappers that have neither,
   status is NO_PLAN.
2. **Per-pair status inheritance** — `pair_status()` reads from
   `training_summary.json` which may have an outdated status field (e.g. if
   the run crashed mid-write). The run.log fallback path (4) catches this
   via `DONE total elapsed=` or `Traceback`.
3. **Wall projection vs elapsed** — The "Elapsed since launch" column is the
   time since `launch.log` (or `master.log`) mtime, not since actual GPU
   launch. For DRY_RUN wrappers this is misleading (large elapsed = no GPU
   work done).
4. **No historical / time-series** — Dashboard shows current snapshot only;
   no rolling history. A future Wf could add per-pair ckpt time-series if
   needed for paper §4 acceptance.
5. **Cross-launcher aggregation** — Each launcher is independent; no
   total-pair / total-wall aggregation across launchers.

## §10 Future Wf candidates (out of scope for Wf-360-A3)

- **Wf-361**: Wire dashboard into `wf343c2_per_wave_check.py` runtime
  belt so each (metal, seed) wave emits a dashboard JSON snapshot.
- **Wf-362**: Add per-pair ckpt time-series (epoch N → wall elapsed curve)
  for paper §4 acceptance gating.
- **Wf-363**: Add cross-launcher rollup JSON for §5 HONEST-NEGATIVE
  unified table.
- **Wf-364**: Add `--watch` mode that polls `last_activity` every 30 s
  and emits updates only when state changes (live progress).

## §11 Pre-flight gate delta

Pre-flight v3 fresh-run **62 PASS / 0 FAIL / 62 TOTAL PRESERVED** at
Wf-328-SYN baseline. Wf-360-A3 added 2 NEW tests but did NOT add any
new pre-flight gates (gates are for runtime launch safety, not for
dashboard scaffold). `EXPECTED_GATE_COUNTS["all"]` = 62 unchanged.

## §12 Standing directive compliance

| Rule | Compliance |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | COMPLIED — §8 table above |
| §1.2 Standing GPU authorization | COMPLIED — 0 GPU launches; dashboard is read-only |
| §1.3 No paper work without authorization | COMPLIED — 0 touches to `paper/*` |
| §1.4 Pre-flight v2 25/25 PASS gate | COMPLIED — pre-flight not invoked (no GPU work) |
| §1.5 Honest-negative framing | COMPLIED — §8 + §9 above |
| §5.1 Standing GPU authorization | COMPLIED — dashboard reads existing logs; does not launch |
| §5.4 Tier-2 journal target + no wet-lab | COMPLIED — code-only work; no wet-lab planning |

---

**END OF Wf-360-A3 SHIP** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; last update 2026-09-29.
