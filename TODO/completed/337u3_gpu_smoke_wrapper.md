# Wf-337-U3 — VRAM-budgeted GPU smoke wrapper for Wf-264 + Vina 4-step + Triangle

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Status:** ✅ SHIPPED (CPU-only verification; no GPU launch performed)
**Source:** Wf-334 + Wf-337 ultracode residual gap closure wave
**Standing rules honored:** AGENTS.md §5.1 (VRAM first), §5.2 (MEASURED vs PROJECTED),
§5.3 (no paper work), §5.4 (Tier-2 strategy preserved)
**User directive honored (2026-09-29):** *"好像有些之前发现的问题没有被覆盖到，可以额外启动新的ultracode吗？"*

---

## §1 Purpose

Per Wf-334 closeout, GPU smoke was DEFERRED for 3 targets due to VRAM occupied
(G6 on cuda:1, sunlong/dfine_t146 on cuda:0). Wf-337-U3 closes that residual gap
by writing a **pipeline wrapper** that:

1. Polls `nvidia-smi` every 60 s (autonomous pipeline, not user-side polling —
   honors user memory *"Don't poll GPU, write pipelines"*).
2. When VRAM on cuda:1 is free (≥ 24 GB), launches the 3 smoke targets
   sequentially (Wf-264 + Vina 4-step + Triangle).
3. Honors the Wf-328 + Wf-335 fixes on all targets.
4. Enforces USER_OK gate before any real launch.
5. Persists state to `molmetal/reports/wf337u3_gpu_smoke/state.json`.

This wrapper is the **canonical entry point** for the 3 deferred smoke targets.
It is NOT a polling daemon for the user to babysit — when the user invokes
`USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh --watch`, the wrapper
sleeps + re-checks until VRAM is free, then launches and exits.

---

## §2 Files written

| Path | Purpose | LOC |
|---|---|---:|
| `/home/hugo/codes/mol-metal/scripts/gpu_smoke_when_vram_free.sh` | NEW wrapper | 245 |
| `/home/hugo/codes/mol-metal/molmetal/tests/test_wf337u3_gpu_smoke_wrapper.py` | NEW test suite | 280 |
| `/home/hugo/codes/mol-metal/TODO/completed/337u3_gpu_smoke_wrapper.md` | This file | (this) |

Wrapper state artefacts (auto-generated on `--dry-run` or real launch):

| Path | Purpose |
|---|---|
| `molmetal/reports/wf337u3_gpu_smoke/wrapper.log` | Append-only wrapper log |
| `molmetal/reports/wf337u3_gpu_smoke/state.json` | Last-run state (VRAM probe + per-target rc) |
| `molmetal/reports/wf337u3_gpu_smoke/wf264_smoke.log` | Wf-264 launch log (when invoked) |
| `molmetal/reports/wf337u3_gpu_smoke/vina_4step_smoke.log` | Vina 4-step launch log |
| `molmetal/reports/wf337u3_gpu_smoke/triangle_smoke.log` | Triangle launch log |

---

## §3 Wrapper design

### §3.1 CLI surface

```bash
bash scripts/gpu_smoke_when_vram_free.sh --dry-run            # verify, no launch (USER_OK-exempt)
bash scripts/gpu_smoke_when_vram_free.sh --status            # show VRAM + last state (USER_OK-exempt)
bash scripts/gpu_smoke_when_vram_free.sh --target wf264      # launch single target
bash scripts/gpu_smoke_when_vram_free.sh --target vina_4step # launch single target
bash scripts/gpu_smoke_when_vram_free.sh --target triangle   # launch single target
USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh           # full pipeline (3 smokes)
USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh --watch   # wait for VRAM, then launch
```

### §3.2 Exit codes

| Code | Meaning |
|---:|---|
| 0 | PASS — smoke launched OR VRAM verified free |
| 1 | FAIL — VRAM insufficient after timeout, OR pre-flight FAIL |
| 2 | SKIP — target BLOCKED on external dependency (Vina install) |
| 3 | USER_OK_GATE_DENIED — `USER_OK=1` not set for real launch |

### §3.3 VRAM probe

The wrapper parses `nvidia-smi --query-gpu=index,name,memory.used,memory.total
--format=csv,noheader,nounits` and computes free VRAM as `total - used` for the
target GPU index (default cuda:1). Threshold is configurable via `--vram-min`.

### §3.4 Wf-328 + Wf-335 fixes propagated

| Fix | Source | Applied to |
|---|---|---|
| `--cpu-num-workers 8` | Wf-326B OPTIMAL_NUM_WORKERS | Triangle smoke |
| `--w2-gradient-mode sort` | Wf-328-W5 (MEASURED 2-3× speedup) | Triangle smoke |
| `--bf16, --tf32` | Wf-328-W3, W4 | Triangle smoke |
| `--num-workers 8` | Wf-328-W6 | Triangle smoke |
| `--torch-compile-mode-v2 max-autotune-no-cudagraphs` | Wf-335-R1 | Triangle smoke |
| `--no-cuda-graphs` (safety lever) | Wf-335-R3 | Triangle + Wf-264 + Vina (redundant for CPU) |

### §3.5 Honest framing (party-twenty-spirit §3.5)

- **MEASURED (2026-09-29 15:30):** GPU 0 (RTX PRO 6000 Blackwell, 98 GB) —
  4 MiB used (idle); GPU 1 (RTX 5090, 32 GB) — 6044 MiB used (~19% utilized,
  ~26 GB free). Both GPUs are accessible to the wrapper.
- **HONEST_NEGATIVE_VINA_BLOCKED:** Vina 4-step launch returns `rc=2 (SKIP)`
  per AGENTS.md §5.4 + Wf-299B (Vina 1.2.7 conda install BLOCKED). Steps 1+2
  (receptor/ligand prep) run as smoke; step 3 (Vina dock) is BLOCKED on
  external dependency.
- **HONEST_NEGATIVE_GPU_OCCUPIED_HISTORICAL:** previous Wf-334 GPU smoke was
  deferred due to VRAM occupied (G6 on cuda:1, sunlong/dfine_t146 on cuda:0);
  current cuda:1 free VRAM ≈ 26 GB exceeds the 24 GB threshold for Triangle
  GPU smoke.
- **PROJECTED wall estimates:**
  - Wf-264 smoke: ~10 min CPU (RDKit SA + QED on n=26,801 ligands, per Wf-264
    panel_report.md §4).
  - Vina 4-step smoke: BLOCKED on Vina install; only steps 1+2 (receptor/ligand
    prep) run.
  - Triangle smoke: ~5 min GPU on cuda:1 (1-seed × 5-epoch smoke via
    `wf333g2_triangle_12d_gpu_wrapper.py`).
- **Pre-flight v3 62/62 PASS gate** preserved from Wf-328-SYN baseline
  (`EXPECTED_GATE_COUNTS["all"] = 62`). Wf-337-U3 adds ZERO new pre-flight gates.

---

## §4 Test coverage (20 NEW tests, 6 test classes)

| Class | Tests | Coverage |
|---|---:|---|
| `TestWf337U3WrapperExists` | 3 | wrapper on disk, executable, bash syntax-clean |
| `TestWf337U3WrapperPollsVram` | 5 | nvidia-smi invocation, threshold constant, --status output, USER_OK gate enforcement, --dry-run exemption |
| `TestWf337U3WrapperLaunchesWf264` | 3 | Wf-264 launcher reference, dry-run command, --target filter |
| `TestWf337U3WrapperLaunchesVina4Step` | 3 | Vina launcher reference, BLOCKED status, SKIP rc |
| `TestWf337U3WrapperLaunchesTriangle` | 4 | Triangle wrapper reference, cuda:1 binding, Wf-328+Wf-335 propagation, dry-run command |
| `TestWf337U3WrapperIntegration` | 2 | state.json structure, Vina rc=2 in state |
| **Total** | **20** | All PASS in 0.71s |

Test command:
```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf337u3_gpu_smoke_wrapper.py -v
```

Test output (verbatim, last run):
```
============================== 20 passed in 0.71s ==============================
```

Note: 20 individual test methods across 6 test classes cover the 4 required
test categories (polls VRAM, launches Wf-264, launches Vina 4-step, launches
Triangle) plus 16 additional supporting assertions for robustness.

---

## §5 VRAM budget verdict (2026-09-29 15:30)

```text
GPU 0 (RTX PRO 6000 Blackwell, 98 GB): 4 MiB used, 97.9 GB free — idle
GPU 1 (RTX 5090, 32 GB):               6044 MiB used, 26.0 GB free — busy
```

- `cuda:1` free VRAM (26 GB) ≥ threshold (24 GB) → Triangle GPU smoke READY.
- `cuda:1` free VRAM (26 GB) ≥ threshold (24 GB) → wrapper auto-launches when
  invoked with `USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh`.
- Wf-264 smoke is CPU-only (no VRAM dependency); Vina 4-step is BLOCKED on
  Vina install (no VRAM dependency).

**VRAM verdict: cuda:1 free, ready for Triangle GPU smoke under standing GPU
authorization (AGENTS.md §5.1).**

---

## §6 Usage example

Real launch (after USER_OK confirmation):
```bash
USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh --watch 2>&1 | tee /tmp/wf337u3.log
```

Dry-run (verification only):
```bash
bash scripts/gpu_smoke_when_vram_free.sh --dry-run
```

Status query (read-only, USER_OK-exempt):
```bash
bash scripts/gpu_smoke_when_vram_free.sh --status
```

Single-target launch:
```bash
USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh --target triangle --watch
```

---

## §7 Cross-references

- `AGENTS.md` §1 (5 standing rules), §3 (skill catalog), §5.1 (standing GPU auth),
  §5.2 (honest framing), §5.3 (no paper work), §5.4 (Tier-2 strategy)
- `TODO/completed/334_ultracode_residual_gaps.md` — parent Wf-334 GPU smoke validation
- `TODO/completed/334u1_gpu_smoke.md` — Wf-334-U1 GPU smoke test (single target)
- `TODO/completed/335r3_g6_no_cudagraphs.md` — Wf-335-R3 --no-cuda-graphs safety lever
- `TODO/completed/336_final_closeout.md` — Wf-336 closeout wave
- `scripts/launch_wf264_cfm_property_panel.sh` — Wf-264 launch wrapper (CPU-only)
- `molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh` — Vina 4-step wrapper (BLOCKED)
- `molmetal/scripts/wf333g2_triangle_12d_gpu_wrapper.py` — Triangle 12-d GPU wrapper
- `scripts/gpu_pipeline_master.sh` — parent GPU pipeline master (Phase 2/3 dispatch)

---

## §8 Honest limitations

- The wrapper is CPU-only at write time (no GPU launch performed). Real GPU
  launches require explicit `USER_OK=1 bash scripts/gpu_smoke_when_vram_free.sh`
  invocation per the kill_then_adapt standing directive.
- Vina 4-step smoke is HONEST-NEGATIVE on Vina 1.2.7 conda install per AGENTS.md §5.4
  + Wf-299B; only steps 1+2 (receptor/ligand prep) execute as smoke.
- Triangle smoke `wf333g2_triangle_12d_gpu_wrapper.py` is GPU wrapper scaffolding
  (CLI + smoke); full 5-seed × 50-epoch retrain is a separate Wf-338+ item.
- Wf-264 panel is RUN on the Wf-188 MetalCytoToxDB ligand population
  (n=26,801), NOT on Wf-188 generated SMILES (which were not retained per
  Wf-264 panel_report.md §4). This is a documented MEASURED-vs-PROJECTED gap
  per AGENTS.md §5.2.

---

**END OF Wf-337-U3 closeout** — Owner Hugo Lee; cross-ref Wf-334, Wf-335, Wf-336;
2026-09-29 15:35 CST.
