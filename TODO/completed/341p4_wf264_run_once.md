# Wf-341-P4 — Wf-264 CFM property panel run-once wrapper audit + tests

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Source:** Wf-341 ultracode residual-gap closure wave — user directive 2026-09-29
*"基于刚才的洞察，可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"*
**Parent:** Wf-334-U3 (Wf-264 launcher WRITTEN) + Wf-336-U3 (USER_OK-gated)
            + Wf-338-F1 (run-once wrapper WRITTEN)
**Status:** **SHIPPED** (4/4 tests PASS; CPU-only edit; no GPU work launched)

---

## §1 What was done

Wf-341-P4 is the Wf-341 ultracode layer over the Wf-264 CFM property panel
run-once wrapper. The wrapper itself was authored at Wf-338-F1
(2026-09-29) — this P4 work audits it + writes 4 NEW tests
at `molmetal/tests/test_wf341p4_wf264_run_once.py`.

### §1.1 Files touched (this wave)

| Path | Action | Description |
|---|---|---|
| `scripts/run_wf264_cfm_property_panel.sh` | (already existed from Wf-338-F1) | Run-once wrapper — USER_OK-gated, pre-flight-gated, refuses --dry-run, --status query supported |
| `scripts/launch_wf264_cfm_property_panel.sh` | (already existed from Wf-334-U3) | Full Wf-264 launcher (dry-run + USER_OK gate + pre-flight + real launch) |
| `molmetal/tests/test_wf341p4_wf264_run_once.py` | **CREATED** | 4 NEW tests: parse + pre-flight + Wf-328 HONEST_NEGATIVE + USER_OK gate |
| `TODO/completed/341p4_wf264_run_once.md` | **CREATED** | This document |

### §1.2 Test results (MEASURED 2026-09-29)

```
molmetal/tests/test_wf341p4_wf264_run_once.py::test_run_once_launcher_parses_and_structure PASSED
molmetal/tests/test_wf341p4_wf264_run_once.py::test_preflight_integration_wired PASSED
molmetal/tests/test_wf341p4_wf264_run_once.py::test_wf328_honest_negative_for_rdkit_post_hoc PASSED
molmetal/tests/test_wf341p4_wf264_run_once.py::test_user_ok_gate_enforced_before_real_launch PASSED
============================== 4 passed in 0.17s ===============================
```

---

## §2 What the run-once wrapper does

Per AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing:

- **USER_OK-gated** (kill_then_adapt standing directive). Refuses real
  launch without `USER_OK=1` (exit code 3 = USER_OK_GATE_DENIED).
- **Pre-flight-gated** (TODO-207 §1). Runs `pre_flight_check.py --pipeline all`
  before invoking the full launcher; aborts on pre-flight FAIL.
- **Refuses `--dry-run`** (run-once is real-only). For dry-run, use the
  full `scripts/launch_wf264_cfm_property_panel.sh --dry-run` path.
- **`--status` query** is USER_OK-exempt (read-only state — emits JSON
  with `user_ok_required`, `dry_run_supported`, Vina honest-negative
  caveat).
- **CPU-only** (RDKit SA + QED on pre-existing SMILES from the
  Wf-188 MetalCytoToxDB ligand population, n=26,801).
- **No GPU launches**, no `torch.compile` / BF16 / TF32 wiring, no
  DataLoader.  Those Wf-328 flags apply to the D-MPNN Wasserstein
  trainer (Wf-334-U1 wired `--dmpnn-backend` there), NOT to this
  post-hoc property panel.

---

## §3 Honest framing (party-twenty-spirit §3.5)

| Claim | Status | Source |
|---|---|---|
| Wrapper exists at `scripts/run_wf264_cfm_property_panel.sh` | **MEASURED** | `ls -la scripts/run_wf264_cfm_property_panel.sh` (2026-09-29) |
| Wrapper bash syntax-clean | **MEASURED** | `bash -n scripts/run_wf264_cfm_property_panel.sh` → rc=0 |
| USER_OK=1 unset → exit 3 | **MEASURED** | End-to-end test invocation: rc=3, "USER_OK_GATE_DENIED" in stdout |
| `--status` query → exit 0 | **MEASURED** | End-to-end test invocation: rc=0, JSON with Vina caveat |
| Wrapper refuses `--dry-run` | **MEASURED** | End-to-end test invocation: rc=2 ("unknown arg") |
| Wf-328 flags NOT silently wired | **MEASURED** | `--torch-compile / --bf16 / --tf32 / --w2-gradient-mode` absent from run-once source |
| Wf-264 panel ran successfully on n=26,801 | **MEASURED (2026-09-28)** | `molmetal/reports/wf264_panel/panel_report.md` |
| Vina sub-panel MEASURED 0/100 docked | **MEASURED (2026-09-28)** | panel_report.md §2 + Wf-264 §2 honest-negative |
| Wf-328 NA contract documented in full launcher | **MEASURED** | `grep "Wf-328 optimization flags: NA" scripts/launch_wf264_cfm_property_panel.sh` |
| Pre-flight v3 42 PASS / 15 FAIL / 57 TOTAL HONEST-NEGATIVE | **MEASURED (2026-09-29)** | AGENTS.md §6 Wf-324D |
| `~/.venv/bin/python` venv convention | **MEASURED** | Wrapper hardcodes `/home/hugo/codes/mol-metal/.venv/bin/python` |
| Full Wf-264 panel re-run wall | **PROJECTED** (~10 min CPU) | RDKit SA + QED on n=26,801 SMILES (single-process loop) |
| Wf-328 num-workers 8 wired into panel | **HONEST_NEGATIVE** | Panel script has no DataLoader; SMILES processed sequentially |

---

## §4 Wf-328 HONEST_NEGATIVE contract (per AGENTS.md §5.2)

Per Wf-334-U3 + Wf-338-F1 documentation, the Wf-328 optimization flags do
NOT apply to `cfm_property_panel.py` (RDKit SA + QED post-hoc scorer):

| Flag | Status | Reason |
|---|---|---|
| `--torch-compile` | **NA** | RDKit compute, no torch.compile path |
| `--bf16` | **NA** | RDKit compute, no autocast path |
| `--tf32` | **NA** | RDKit compute, no matmul path |
| `--w2-gradient-mode` | **NA** | No W₂ loss in property panel |
| `--num-workers` | **NA** | Sequential SMILES loop, no DataLoader |

The full launcher documents this in its header (lines 14-21, 63-68) +
dry-run echo (lines 96-100). The run-once wrapper delegates to the
full launcher, so the contract is preserved transitively.

`test_wf328_honest_negative_for_rdkit_post_hoc` asserts:
1. None of `--torch-compile / --bf16 / --tf32 / --w2-gradient-mode`
   appear in the run-once source (CPU-only panel).
2. If `--num-workers` appears, it's in a comment, not active code.
3. The full launcher DOES mention all 5 flags (with `NA` or `do NOT`
   framing language + the "Wf-328 optimization flags: NA for this
   panel" dry-run echo).

---

## §5 Pre-flight v3 + USER_OK gate ordering (kill_then_adapt invariant)

Per AGENTS.md §5.1 + Wf-300C + Wf-306D, the ordering is:

1. **USER_OK gate fires FIRST** — fail fast without burning pre-flight
   time if user hasn't authorized.
2. **Pre-flight check fires SECOND** — `pre_flight_check.py --pipeline all`
   (TODO-207 §1 gate).
3. **Real panel launch fires THIRD** — `USER_OK=1 bash launch_wf264_cfm_property_panel.sh`
   (the full launcher re-enforces pre-flight for orchestration
   determinism — defence in depth).

`test_user_ok_gate_enforced_before_real_launch` asserts:
- Gate-first ordering: `[ "${USER_OK:-0}" != "1" ]` check appears BEFORE
  `pre_flight_check.py --pipeline all` invocation.
- `kill_then_adapt` reference present.
- End-to-end: invocation without USER_OK → rc=3, "USER_OK_GATE_DENIED"
  in stdout.
- `--status` query is USER_OK-exempt and emits JSON with the Vina
  honest-negative caveat (`BLOCKED on meeko` or `Wf-299B`).

---

## §6 Vina sub-panel — HONEST_NEGATIVE (per AGENTS.md §5.4)

Vina sub-panel is **BLOCKED on meeko+Pt parameterisation** per AGENTS.md
§5.4 + Wf-299B. MEASURED 0/100 docked ok on 1h36 (meeko 0.8.0
metal-atom limitation, honest-negative per Wf-264 §2).

The `--status` query surfaces this caveat:
```json
{
  "honest_negative_caveat": "Vina sub-panel is BLOCKED on meeko+Pt per AGENTS.md §5.4 + Wf-299B"
}
```

---

## §7 No paper work (per AGENTS.md §5.3)

TODO-84 active. No touches to `paper/*`. This work is wrapper + tests +
documentation only.

---

## §8 What is NOT in scope (TODO for Wf-335+)

- **Per_molecule / per_metal / per_pocket JSON split.** Currently the
  panel script emits only `aggregate.json`; per_molecule / per_metal /
  per_pocket splits are derived in `panel_report.md` §5. Future
  extension: have the panel script emit all three files directly.
- **Vina parameterisation.** BLOCKED on meeko+Pt per Wf-299B. Re-scope
  to in-silico metric lift per AGENTS.md §5.4 Tier-2 strategy.
- **Wf-328 num-workers wire-in.** HONEST_NEGATIVE — panel script has
  no DataLoader; SMILES processed sequentially.

---

## §9 Files referenced

| Path | Purpose |
|---|---|
| `scripts/run_wf264_cfm_property_panel.sh` | Run-once wrapper (Wf-338-F1) |
| `scripts/launch_wf264_cfm_property_panel.sh` | Full launcher (Wf-334-U3) |
| `molmetal/scripts/cfm_property_panel.py` | Panel script (RDKit SA + QED) |
| `molmetal/scripts/pre_flight_check.py` | Pre-flight v2/v3 (TODO-207) |
| `molmetal/tests/test_wf341p4_wf264_run_once.py` | 4 NEW tests (Wf-341-P4, this wave) |
| `molmetal/tests/test_wf334u3_wf264_launcher.py` | 2 prior tests (Wf-334-U3) |
| `molmetal/tests/test_wf338f1_w2_wf264_smoke.py` | 11 prior tests (Wf-338-F1) |
| `molmetal/reports/wf264_panel/panel_report.md` | Wf-264 MEASURED 2026-09-28 |
| `TODO/INDEX.md` | Status index (single source of truth) |
| `AGENTS.md` | Agent-facing entry point |

---

**END OF Wf-341-P4** — 4/4 tests PASS; run-once wrapper audited +
honest-negative framing preserved; lock deliverables for Wf-335+
follow-up.
