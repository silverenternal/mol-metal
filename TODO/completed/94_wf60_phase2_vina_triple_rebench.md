# Wf-60 Phase 2 — Vina triple m4 + m10 min affinity re-bench

**Date**: 2026-09-26
**Workflow**: Wf-60 Phase 2 (Vina re-bench — closes the m4 MISSING_DEP / m10 min gap surfaced in Wf-47 partial ship).
**Status**: ✅ **SHIP** — bench script + tests + per-pocket JSONs + binary-status + synthesis all on disk; honest-MISSING_DEP framing preserved (no fabricated Vina scores).
**Priority**: HIGH (closes the Vina-m4-m10 audit ledger row; WEAK_REAL upgrade is the Wf-30 binary install, NOT Wf-60 scope).

---

## §0 TL;DR

| Deliverable | File | Status |
|---|---|---|
| `molmetal/scripts/bench_vina_triple_rebench.py` (~565 LOC) | ✅ SHIP |
| `molmetal/tests/test_vina_triple_rebench.py` (15 tests, all pass) | ✅ SHIP |
| `molmetal/reports/wf60_vina_triple/{10 pockets}/metrics.json` (10/10) | ✅ SHIP |
| `molmetal/reports/wf60_vina_triple/vina_binary_status.json` | ✅ SHIP (MISSING_DEP) |
| `molmetal/reports/wf60_vina_triple/synthesis.json` (10 pockets × 5 cands × 3 seeds = 150 cells) | ✅ SHIP |
| 15/15 unit tests pass on CPU (no GPU, no Vina binary required) | ✅ SHIP |
| **m4 triple > 0** | ⚠️ DEFERRED (real Vina binary missing on this host — `which vina` → not found, no `wf47_vina_fix` report) |
| **m10 min maintained** | ⚠️ DEFERRED (real Vina binary missing) |
| **Honest warning** when Vina missing | ✅ SHIP — `vina_binary_status.json: status=MISSING_DEP`, all 150 cells `STUB_FALLBACK` |

**Honest framing (party-twenty-spirit §3.5)**: The real Vina binary is NOT installed on this host. Every (pocket, cand, seed) cell carries `score_kcal_mol=null`, `pb_pass=null`, `qed=null`, `triple_success=null`, `status="STUB_FALLBACK"`. The bench ledger **never fabricates a Vina score**. The m4 / m10 aggregates are `null` (not 0.0) until the binary lands — `MOLMETAL_VINA_BIN=/path/to/vina` env var or symlink `/usr/local/bin/vina` enables the real path; the bench will then re-run with the same CLI surface.

---

## §1 Why this Wf exists

Wf-47 partial ship (`molmetal/reports/wf47_real_runs/script_pb_bench_per_tool_accuracy/per_tool_accuracy.md` row 22) recorded Vina as `MISSING_DEP`. Wf-30 staged **10/10 pockets READY** (830c, 1h36, 3eml, 3ny8, 4aaw, 4rlu, 1g9v, 2hnx, 4rn0, 1fmc — all `receptor.pdbqt` present in `molmetal/reports/wf30_pocket_prep/`). The m4 (Vina Triple Success Rate) and m10 (Vina min affinity) audit cells stayed WEAK_REAL because the bench script and binary path were missing.

Wf-60 Phase 2 ships the bench **without fabricating scores**:

* `--use-real-vina` flag → tries real Vina if the binary is on PATH (or `MOLMETAL_VINA_BIN`)
* Default → STUB_FALLBACK + honest `MISSING_DEP` per cell
* Per-pocket summary aggregates only over cells with real numbers (`m4_n_real` is exposed in the JSON)

---

## §2 What shipped

### §2.1 Bench script — `molmetal/scripts/bench_vina_triple_rebench.py` (~565 LOC)

CLI surface:

```
python -m molmetal.scripts.bench_vina_triple_rebench
    [--pockets 830c 1h36 ...]      # default: 10 Wf-30 staged
    [--cands SMILES ...]            # default: 5 cisplatin analogs
    [--seeds 0 1 2 ...]             # default: 0 1 2
    [--use-real-vina]               # default: STUB_FALLBACK
    [--pocket-status-json PATH]     # default: wf30_pocket_prep/pocket_status.json
    [--out-dir PATH]                # default: molmetal/reports/wf60_vina_triple
    [--dry-run]                     # print plan only
```

Key design choices (party-twenty-spirit §3.5 — no fabrication):

* **Binary detection** (`_detect_vina_binary()`) honours `MOLMETAL_VINA_BIN`, `shutil.which("vina")`, `qvina02`, `smina`, and the well-known prefixes `/usr/local/bin/vina` + `/opt/vina/bin/vina`. Records the search trail in `vina_binary_status.json: checked_paths`.
* **Stub fallback** (`_stub_score()`) returns `score_kcal_mol=None`, `triple_success=None`, `status="STUB_FALLBACK"`. Never fabricates a kcal/mol number.
* **Optional probes** (best-effort, never fail the run): `_try_qed()` and `_try_pb_pass()` are wrapped in try/except so RDKit / PoseBusters absence does not block the bench.
* **Per-pocket summary** (`_summarise_pocket()`) — uses set-membership to pick `OK` / `STUB_FALLBACK` / `PARTIAL` / `MISSING_DEP`; NaN-safe aggregation via `m4_n_real` count + `m10_min_kcal_mol` only over real scores.
* **Top-level synthesis** (`synthesis.json`) reports `vina_binary_status`, `n_total_cells`, `n_with_real_score`, `m4_global_triple_rate`, `m10_global_min_kcal_mol`, plus per-pocket roll-up.

### §2.2 Tests — `molmetal/tests/test_vina_triple_rebench.py` (15 tests, all pass)

| Class | Tests |
|---|---|
| CLI / dry-run | `test_cli_parses_defaults`, `test_cli_overrides`, `test_dry_run_prints_plan` |
| Binary detection | `test_binary_detection_no_env`, `test_binary_detection_with_env`, `test_binary_detection_env_missing_file` |
| Stub semantics | `test_stub_score_returns_none` |
| Per-pocket aggregation | `test_summarise_pocket_with_all_stubs`, `test_summarise_pocket_with_real_scores`, `test_summarise_pocket_partial`, `test_summarise_pocket_failing_gate` |
| End-to-end | `test_end_to_end_stub_path`, `test_end_to_end_real_receptor_stub` |
| Subprocess smoke | `test_subprocess_help`, `test_subprocess_dry_run` |

Run:

```
python -m pytest molmetal/tests/test_vina_triple_rebench.py -v
```

Result: **15 passed in 0.24s** (CPU-only, no GPU / Vina / RDKit / PoseBusters needed).

### §2.3 Per-pocket + synthesis JSON on disk

| Path | n_pockets | n_cells | m4 (triple rate) | m10 (min kcal/mol) |
|---|---:|---:|---|---|
| `molmetal/reports/wf60_vina_triple/` | 10 | 150 | **null** (binary MISSING_DEP) | **null** |

Per-pocket schema (`metrics.json`):

```json
{
  "pocket": "830c",
  "status": "STUB_FALLBACK",
  "n_cells": 15,
  "status_counts": {"STUB_FALLBACK": 15},
  "m4_triple_rate": null,
  "m4_n_real": 0,
  "m4_cutoff_kcal": -7.0,
  "m4_qed_min": 0.5,
  "m10_min_kcal_mol": null,
  "m10_median_kcal_mol": null,
  "m10_gate_kcal_mol": -4.0,
  "m10_pass": null,
  "cells": [
    {"cand_smiles": "...", "seed": 0, "score_kcal_mol": null, ...}
  ]
}
```

`vina_binary_status.json`:

```json
{
  "status": "MISSING_DEP",
  "bin": null,
  "version": "",
  "reason": "No executable Vina/Q-Vina2/smina binary found on PATH or at well-known prefixes. Set MOLMETAL_VINA_BIN to point at the binary to enable real docking. Without the binary, every (pocket, cand, seed) cell falls back to STUB_FALLBACK.",
  "checked_paths": ["/usr/local/bin/vina", "/opt/vina/bin/vina"]
}
```

---

## §3 Verify

| Spec requirement | Status |
|---|---|
| `molmetal/scripts/bench_vina_triple_rebench.py` (~250 LOC) | ✅ ~565 LOC (over-spec; full coverage of CLI + aggregation + binary detection + per-pocket + synthesis) |
| `molmetal/tests/test_vina_triple_rebench.py` (~15 tests) | ✅ 15 tests, all pass |
| `molmetal/reports/wf60_vina_triple/{pocket}/metrics.json` per-pocket | ✅ 10/10 pockets × `metrics.json` |
| `molmetal/reports/wf60_vina_triple/vina_binary_status.json` records real vs stub | ✅ SHIP with `status=MISSING_DEP` + reason + checked_paths |
| m4 triple > 0 if real Vina installed | ⚠️ DEFERRED — real Vina binary missing on this host |
| m10 min maintained | ⚠️ DEFERRED — real Vina binary missing |
| Honest warning if Vina missing | ✅ SHIP — `vina_binary_status.json: status=MISSING_DEP` + per-cell `STUB_FALLBACK` |
| GPU-0 Blackwell for docking | n/a (no GPU usage; Vina binary is CPU) |
| CPU-only fallback | ✅ default mode; runs in 0.05s for 10 pockets × 5 cands × 3 seeds |
| Honest framing about Vina install | ✅ per `party-twenty-spirit` §3.5 — no fabricated scores |

---

## §4 To enable real Vina (one-time setup)

When the host gains a Vina binary, the bench will pick it up via:

1. Set `MOLMETAL_VINA_BIN=/path/to/vina` (env var), **or**
2. Symlink to one of: `/usr/local/bin/vina`, `/opt/vina/bin/vina`, or any `vina` / `qvina02` / `smina` on PATH.

Then re-run with `--use-real-vina`:

```
python -m molmetal.scripts.bench_vina_triple_rebench \
    --use-real-vina \
    --out-dir molmetal/reports/wf60_vina_triple_real
```

The Python `vina` binding (PyPI `vina`) + `meeko` are also needed for the script's `_real_vina_score()` path. If the Python binding is missing, each cell falls back to `status="MISSING_DEP"` (separate reason string) — never fabricates.

---

## §5 Cross-references

- Wf-30 pocket staging: `molmetal/reports/wf30_pocket_prep/pocket_status.json` (10/10 READY)
- Wf-47 partial ship baseline: `molmetal/reports/wf47_real_runs/script_pb_bench_per_tool_accuracy/per_tool_accuracy.md` (Vina row 22 = MISSING_DEP)
- Wf-48 score_any router: `TODO/completed/90_wf48_score_any_router.md` (the `vina_score_gpu` wrapper remains `AdapterUnavailable` until the binary lands)
- Wf-50 GPU wrappers: `TODO/completed/100_gpu_scoring_acceleration.md` §8.1 — Vina-GPU RESERVED (no CUDA binary on host)
- Wf-60 Phase 1 (m1/m2/m19): `TODO/completed/94_wf60_multi_seed_phase1.md` (parallel dispatch pattern; ultracode analog notes)
- Honest framing: `party-twenty-spirit` §3.5, `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4

---

**Date**: 2026-09-26 · **Verdict**: SHIP — bench infrastructure + 15 tests + 10-pocket stub ledger on disk; m4/m10 numeric lift awaits real Vina binary install (Wf-30 follow-up, out of Wf-60 scope) · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
