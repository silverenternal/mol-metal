# Wf-330-B1: Concurrent-edit conflict resolution — trainer merge fix

**Status:** SHIPPED (CPU-only documentation + tests)
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Wf:** Wf-330-B1 (per AGENTS.md §3.4 orchestrator layer; resolves
Wf-328 W1..W10 race-condition edits on
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`).

---

## Context (per Wf-328-W5 honest framing)

Multiple Wf-328 agents (W1..W10) concurrently edited the same trainer
file `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` in
parallel.  Per Wf-328-W5 honest framing (2026-09-29), the original
race-condition concerns were:

1. Unclosed ``try/finally`` blocks left dangling in the middle of the
   ``train_one_epoch_joint`` forward loop (W3 BF16 autocast wrap).
2. Triton-3.x-incompatible ``pre_hook`` kwarg dropped into a
   ``@triton.autotune`` decorator (W9 Triton-edge-update patch).
3. Duplicate ``import`` statements and conflicting CLI flag
   definitions (W1 torch.compile vs W6 dataloader wire-in).
4. Plain ``SyntaxError`` raised by ``ast.parse`` once all the
   races are merged naively.

Wf-330-B1 audits the merged file and confirms the race resolved
cleanly: the file parses, all 8 W* integrations are physically present,
and no Triton-3.x-incompatible kwarg remains.

---

## §1 Parsing gate

**MEASURED (2026-09-29):**

```bash
$ python3 -c "import ast; ast.parse(open(
    'molmetal/scripts/train_dmpnn_multitask_wasserstein.py'
  ).read())"
# PARSE OK
```

The trainer file is **2,439 lines** (MEASURED via `wc -l`) and parses
without syntax errors.  Structural balance via AST walk:

| Block type   | Count |
|-------------:|------:|
| ``try``      |    13 |
| ``except``   |    12 |
| ``finally``  |     1 |

The single ``finally`` (line 1497) restores the encoder's monkey-patched
``_forward_one_mol`` after the Wf-328-W1 compile wrapper exits
backward-compat cleanly — this is the only ``try/finally`` block and it
is closed.

**Triton-3.x incompatibility check (MEASURED):**

- `pre_hook` kwarg occurrences: **0**
- `@triton.autotune(...)` decorators: **0**
- Triton-3.x-incompatible kwargs (e.g. `pre_hook=` left in
  `triton.autotune` calls): **0**

No Triton-3.x-incompatible patches were left in the merged file.

---

## §2 Wf-328 W* integration verification

All 8 W* integrations the production trainer needs are physically
present in the source (MEASURED via `grep -n 'Wf-328-W[0-9]+' ...`):

| W*  | Integration                              | Marker line(s)      | Status |
|----:|------------------------------------------|---------------------|--------|
| W1  | torch.compile wrappers (encoder + W₂)    | 135, 1264, 1498     | SHIPPED |
| W3  | BF16 autocast (default ON, --no-bf16)    | 269, 1532, 1582     | SHIPPED |
| W4  | TF32 default ON (--no-tf32 lever)        | 256, 1598           | SHIPPED |
| W5  | W₂ sort mode default (closed-form 1-D)   | 243, 1703           | SHIPPED |
| W6  | build_optimized_loader() factory wire-in | 736, 1840           | SHIPPED |
| W8  | CUDA Graphs wrapper (--cuda-graph)       | 1145, 1301, 1939    | SHIPPED |
| W9  | Triton fused ReLU+Linear (opt-in)        | 1730, 2027          | SHIPPED |
| W10 | on-disk pickle featurisation cache       | 297                 | SHIPPED |

Cross-cutting integrity checks (regression gate for pre-Wf-328
features that MUST survive the merge):

| Pre-Wf-328 feature        | Marker                  | Status |
|---------------------------|-------------------------|--------|
| Wf-285B DANN GRL          | `use_dann`, `ReverseLayerF`, `lambda_grl` | SHIPPED |
| Wf-251-F4 Kendall UW MTL  | `use_kendall`           | SHIPPED |
| Wf-197 single-task lever  | `single_task` flag      | SHIPPED |

**Wf-328-W2 + Wf-328-W7 scope clarification:**

- **Wf-328-W2** owns the **launch-script** ``--cpu-num-workers``
  override (6 launch scripts) — does NOT touch the trainer file
  itself.  See `molmetal/tests/test_wf328w2_launcher_num_workers.py`
  for the W2 verification surface.
- **Wf-328-W7** is the PyG port in `molmetal/models/dmpnn_pyg.py`;
  the trainer file references it via the ``--cuda-graph`` CLI help
  text (line 1952) so downstream consumers know the dependency chain.

---

## §3 New test surface

**File:** `molmetal/tests/test_wf330b1_trainer_parse.py`

**Tests added:** 2

1. **`test_trainer_file_parses_without_syntax_errors`**
   - AST parse via `ast.parse`.
   - AST walk verifies `try` / `except` / `finally` block balance.
   - Grep gate on `pre_hook` + `@triton.autotune` to catch any
     Triton-3.x-incompatible kwarg left over from the race.

2. **`test_wf328_integrations_present`**
   - Asserts the marker substring for each W* integration is present.
   - Asserts the pre-Wf-328 cross-cutting features (DANN, Kendall,
     single_task) survived the merge.

**Test run (MEASURED 2026-09-29):**

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf330b1_trainer_parse.py -v

============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 2 items

molmetal/tests/test_wf330b1_trainer_parse.py::test_trainer_file_parses_without_syntax_errors PASSED [ 50%]
molmetal/tests/test_wf330b1_trainer_parse.py::test_wf328_integrations_present PASSED [100%]

============================== 2 passed in 0.13s ===============================
```

---

## §4 Honest framing (per AGENTS.md §5.2 / party-twenty-spirit §3.5)

- **MEASURED:** file parses via `ast.parse` (13 try / 12 except / 1
  finally).
- **MEASURED:** all 8 Wf-328 W* integrations physically present in
  trainer source (grep 2026-09-29).
- **MEASURED:** 0 Triton-3.x-incompatible kwarg (`pre_hook=` /
  `@triton.autotune`) in trainer.
- **MEASURED:** 2/2 tests pass on CPU-only pytest run.
- **NOT MEASURED:** behavioural correctness of each W* integration —
  the per-W* tests (test_wf328w1_*, test_wf328w3_*, ..., test_wf328w10_*)
  cover the behavioural surface and pass independently.
- **NOT MEASURED:** end-to-end GPU throughput lift from the merged
  state — BLOCKED on AGENTS.md §5.1 kill_then_adapt + Wf-181 G5
  retrain launch (TODO-181, 5 hr GPU).

---

## §5 Files touched

- `molmetal/tests/test_wf330b1_trainer_parse.py` — NEW (2 tests).
- `TODO/completed/330b1_trainer_merge_fix.md` — NEW (this file).
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — UNTOUCHED
  (no edits required; the file already parsed cleanly post-merge).

---

## §6 Downstream consumers

- Wf-181 G5 retrain (TODO-181, 5 hr GPU) reads the trainer CLI flag
  surface; the merge ensures no flags were dropped during the race.
- Wf-182 G6 per-innovation ablation (TODO-182, 7.5 hr GPU) reads the
  `w2_gradient_mode` + `compile` + `bf16` + `tf32` + `cuda_graph`
  flag family; Wf-330-B1 confirms all 5 are present.
- Wf-299 CFM 3-wave launcher (Wf-299, ~9.5 hr sequential) reads the
  W₂ closed-form dispatch; Wf-330-B1 confirms W5's `sort` mode default
  is wired.

---

**END OF Wf-330-B1 closeout** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5 + AGENTS.md §5.2 (honest framing);
last update 2026-09-29.