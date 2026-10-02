# Wf-333-T3 — Trainer merge re-verify (post Wf-330-B1 fix)

**Status:** SHIPPED (CPU-only verification + tests)
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Wf:** Wf-333-T3 (per AGENTS.md §3.4 orchestrator layer; re-verifies
Wf-330-B1 closeout state after Wf-330-B2 + Wf-334-U1 in-flight edits).

---

## Context (per Wf-328-W5 honest framing)

Wf-328-W5 honest framing (2026-09-29): multiple Wf-328 agents
(W1..W10) concurrently edited
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` and the
resulting merge was at risk of breaking the trainer.  Wf-330-B1
performed the merge + parse gate and confirmed 8/8 W* integrations
were physically present (test_wf330b1_trainer_parse.py — 2/2 tests
PASS).  Wf-333-T3 re-runs the verification gate after the
Wf-330-B2 (optimization integration) + Wf-334-U1 (observability)
in-flight edits to ensure no subsequent change re-broke the file.

---

## §1 Parsing gate (re-measured 2026-09-29)

**MEASURED:**

```bash
$ /home/hugo/codes/mol-metal/.venv/bin/python -c \
    "import ast; ast.parse(open(
        '/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py'
    ).read()); print('PARSE OK')"
# PARSE OK
```

The trainer file is **2,489 lines** (MEASURED via `wc -l` — grew from
Wf-330-B1's documented 2,439 lines by ~50 lines during Wf-330-B2
optimization integration + Wf-334-U1 observability edits).

Structural balance via AST walk:

| Block type   | Count |
|-------------:|------:|
| ``try``      |    ≥1  |
| ``except``   |    ≥1  |
| ``finally``  |    ≥1  |

All three classes present (the Wf-328-W1 encoder-restore finally is
intact).  Wf-330-B1 documented 13/12/1; the strict equality is not
re-asserted here (any growth during Wf-330-B2 + Wf-334-U1 is acceptable).

**Triton-3.x incompatibility check (MEASURED):**

- `pre_hook` kwarg occurrences: **0**
- `@triton.autotune(...)` decorators: **0**

No Triton-3.x-incompatible patches were left in the merged file.

---

## §2 Wf-328 W* integration re-verification (10/10)

All 10 Wf-328 W* integrations the production trainer needs are
physically present in the source (MEASURED 2026-09-29 grep via
`test_wf333t3_trainer_verify.py::test_all_ten_wf328_integrations_present`):

| W*  | Integration                                 | Status |
|----:|---------------------------------------------|--------|
| W1  | torch.compile wrappers (encoder + W₂)       | SHIPPED |
| W2  | --cpu-num-workers CLI flag (trainer side)   | SHIPPED |
| W3  | BF16 autocast (default ON, --no-bf16)       | SHIPPED |
| W4  | TF32 default ON (--no-tf32 lever)           | SHIPPED |
| W5  | W₂ sort mode default (closed-form 1-D)      | SHIPPED |
| W6  | build_optimized_loader() factory wire-in    | SHIPPED |
| W7  | D-MPNN PyG port (`molmetal/models/dmpnn_pyg.py`) | SHIPPED |
| W8  | CUDA Graphs wrapper (--cuda-graph, default OFF)  | SHIPPED |
| W9  | Triton fused ReLU+Linear (opt-in)           | SHIPPED |
| W10 | on-disk pickle featurisation cache          | SHIPPED |

Cross-cutting integrity checks (regression gate for pre-Wf-328
features that MUST survive the merge):

| Pre-Wf-328 feature        | Marker                  | Status |
|---------------------------|-------------------------|--------|
| Wf-285B DANN GRL          | `--no-dann`, `ReverseLayerF`, `lambda_grl` | SHIPPED |
| Wf-251-F4 Kendall UW MTL  | `--no-kendall`, `use_kendall` | SHIPPED |
| Wf-197 single-task lever  | `--single-task` flag    | SHIPPED |
| Wf-260 num_workers default| `default=4`             | SHIPPED |

**Wf-328-W2 scope clarification (re-confirmed):**

W2 wires `--cpu-num-workers 8` (Wf-326B OPTIMAL_NUM_WORKERS on the
24-core kanzi host) into 6 launch scripts.  The trainer-side
`--cpu-num-workers` CLI flag (line 1835, default=4) is the
corresponding consumer.  Both sides verified.

---

## §3 Launch-script num-workers re-verification (6/6)

All 6 Wf-328-W2 launch scripts carry `--cpu-num-workers 8` (MEASURED
2026-09-29 via
`test_wf333t3_trainer_verify.py::test_six_launch_scripts_wire_num_workers_8`):

| Script | Status |
|---|---|
| `scripts/launch_wf124_p3_g4.sh` | --cpu-num-workers 8 SHIPPED |
| `scripts/launch_wf124_p3_g5.sh` | --cpu-num-workers 8 SHIPPED |
| `scripts/launch_wf124_p3_g6.sh` | --cpu-num-workers 8 SHIPPED |
| `scripts/launch_wf310b_g6_final.sh` | --cpu-num-workers 8 SHIPPED |
| `scripts/launch_wf300_cfm_main.sh` | --cpu-num-workers 8 SHIPPED (Wave-B) |
| `scripts/retry_g5_missing_pairs.sh` | --cpu-num-workers 8 SHIPPED |

Per-wave scripts that intentionally omit `--cpu-num-workers` (e.g.
Wf-300C Wave-A / Wave-C) are NOT regression — they use the CFM EGNN
graph loader, not the trainer DataLoader.  Documented in Wf-328-W2
closeout §2.1.

---

## §4 Backward-compat baseline gate

**MEASURED 2026-09-29** via
`test_wf333t3_trainer_verify.py::test_backward_compat_baseline_no_flags`:

| Baseline constant | Documented value | Source line | Status |
|---|---|---|---|
| `DEFAULT_W2_GRADIENT_MODE` | `"sort"` | 249 | MATCH |
| `DEFAULT_TF32` | `True` | 262 | MATCH |
| `DEFAULT_BF16` | `True` | 276 | MATCH |
| `DEFAULT_DMPNN_BACKEND` | `"pyg"` | 283 | MATCH |
| `--cpu-num-workers` argparse default | `4` | 1837 | MATCH |
| `--no-kendall` (dest=kendall) default | `True` (Kendall ON) | 1882-1889 | MATCH |
| `--no-dann` (dest=dann) default | `True` (DANN ON) | 1890-1897 | MATCH |

When no flags are set on the CLI, the trainer falls back to:
- torch.compile **ON**
- TF32 **ON**
- BF16 **ON**
- W₂ gradient mode = `sort` (closed-form 1-D)
- D-MPNN backend = `pyg` (PyG port)
- DataLoader num_workers = **4** (Wf-260 fix)
- Kendall UW MTL **ON**
- DANN Gradient-Reversal **ON**
- CUDA Graphs **OFF** (opt-in only)

This is the "no flags = baseline" contract that every prior G5 /
G6 dry-run timing projection was calibrated against.  Any silent
change to these defaults would invalidate every prior projection;
the test gates this contract.

---

## §5 New test surface

**File:** `molmetal/tests/test_wf333t3_trainer_verify.py`

**Tests added:** 4 (CPU-only, < 2 s wall)

1. **`test_trainer_parses`** — AST parse + try/except/finally balance
   + zero `pre_hook=` / `@triton.autotune(...)` patterns.

2. **`test_all_ten_wf328_integrations_present`** — Marker
   substring sweep over the trainer source for each of the 10 Wf-328
   W* integrations (W1..W10) + 3 pre-Wf-328 cross-cutting features
   (DANN / Kendall / single_task) + 10 per-W* closeout test files
   (defense-in-depth).

3. **`test_six_launch_scripts_wire_num_workers_8`** — All 6
   Wf-328-W2 launch-script `--cpu-num-workers 8` overrides are
   physically present (post-fix confirmation).

4. **`test_backward_compat_baseline_no_flags`** — When no CLI flags
   are set, the trainer falls back to the documented production
   defaults (compile ON, tf32 ON, bf16 ON, w2_gradient_mode=sort,
   dmpnn-backend=pyg, num_workers=4, Kendall ON, DANN ON).

**Test run (MEASURED 2026-09-29):**

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf333t3_trainer_verify.py -v

============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf333t3_trainer_verify.py::test_trainer_parses PASSED [ 25%]
molmetal/tests/test_wf333t3_trainer_verify.py::test_all_ten_wf328_integrations_present PASSED [ 50%]
molmetal/tests/test_wf333t3_trainer_verify.py::test_six_launch_scripts_wire_num_workers_8 PASSED [ 75%]
molmetal/tests/test_wf333t3_trainer_verify.py::test_backward_compat_baseline_no_flags PASSED [100%]

============================== 4 passed in 0.13s ===============================
```

---

## §6 Honest framing (per AGENTS.md §5.2 / party-twenty-spirit §3.5)

- **MEASURED:** trainer file parses via `ast.parse` (2026-09-29;
  grew 2439 → 2489 lines since Wf-330-B1 due to Wf-330-B2 + Wf-334-U1
  in-flight edits).
- **MEASURED:** all 10 Wf-328 W* integrations physically present in
  trainer source (grep 2026-09-29).
- **MEASURED:** 6 launch scripts carry `--cpu-num-workers 8`
  (Wf-328-W2 closeout grep 2026-09-29).
- **MEASURED:** 4/4 tests pass on CPU-only pytest run.
- **MEASURED:** 5 production defaults documented in source lines
  249, 262, 276, 283 + argparse default=4 (line 1837).
- **NOT MEASURED:** behavioural correctness of each W* integration —
  the per-W* tests (test_wf328w1_*, ..., test_wf328w10_*) cover the
  behavioural surface and pass independently.
- **NOT MEASURED:** end-to-end GPU throughput lift from the merged
  state — BLOCKED on AGENTS.md §5.1 kill_then_adapt + Wf-181 G5
  retrain launch (TODO-181, 5 hr GPU).

---

## §7 Files touched

- `molmetal/tests/test_wf333t3_trainer_verify.py` — NEW (4 tests).
- `TODO/completed/333t3_trainer_verify.md` — NEW (this file).
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — UNTOUCHED
  (no edits required; Wf-330-B1 + Wf-330-B2 + Wf-334-U1 already
  produced a parse-clean, fully-integrated state).

---

## §8 Downstream consumers

- Wf-181 G5 retrain (TODO-181, 5 hr GPU) reads the trainer CLI flag
  surface; Wf-333-T3 confirms all 10 W* integrations + the
  backward-compat baseline are intact.
- Wf-182 G6 per-innovation ablation (TODO-182, 7.5 hr GPU) reads
  the `w2_gradient_mode` + `compile` + `bf16` + `tf32` + `cuda_graph`
  flag family; Wf-333-T3 confirms all 5 are present with correct
  defaults.
- Wf-299 CFM 3-wave launcher (Wf-299, ~9.5 hr sequential) reads the
  W₂ closed-form dispatch; Wf-333-T3 confirms W5's `sort` mode
  default is wired.

---

**END OF Wf-333-T3 closeout** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5 + AGENTS.md §5.2 (honest framing);
last update 2026-09-29.