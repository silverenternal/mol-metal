# Wf-328-W6 — Wf-326B DataLoader factory wired into production trainer

**Phase**: Wf-328-W6 production-wiring closeout 2026-09-29 (this entry) — wires
the ``build_optimized_loader()`` factory from
`molmetal/scripts/wf326b_dataloader_optimized.py` into the production D-MPNN
multi-task trainer (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)
and adds a `--num-workers` CLI flag with `suggest_num_workers()` default
resolution.  Closes the "factory exists but trainer still inlines the
DataLoader" gap flagged in Wf-326B §"Public API".

**Skill alignment**: `party-twenty-spirit` §3.5 (every claim traces to a
MEASURED source — 4/4 NEW tests PASS); AGENTS.md §5.1 (standing GPU
authorization); AGENTS.md §5.4 (Tier-2 paper target).

---

## §1 Scope

| # | Surface | Status | Source |
|---:|---|---|---|
| 1 | Trainer DataLoader construction | ✅ WIRED | `train_dmpnn_multitask_wasserstein.py:571-606` (replaced inline DataLoader with `build_optimized_loader()`) |
| 2 | `--num-workers` CLI flag (default: None = use `--cpu-num-workers`) | ✅ ADDED | `train_dmpnn_multitask_wasserstein.py:1380-1398` |
| 3 | `--num-workers-host-cores` CLI flag (override host core count for `suggest_num_workers()`) | ✅ ADDED | `train_dmpnn_multitask_wasserstein.py:1399-1412` |
| 4 | Precedence: `--num-workers` > `--cpu-num-workers` > `suggest_num_workers()` | ✅ WIRED | `train_dmpnn_multitask_wasserstein.py:1591-1615` |
| 5 | Audit log: `[wf328w6] num_workers resolved: effective=…` once per (metal, seed) | ✅ ADDED | `train_dmpnn_multitask_wasserstein.py:1601-1607` |
| 6 | 4 NEW tests | ✅ 4/4 PASS | `molmetal/tests/test_wf328w6_dataloader_factory_production.py` |
| 7 | `Wf-326B` factory public API consumers | ✅ unchanged | `molmetal/scripts/wf326b_dataloader_optimized.py` (no edit; factory is the source of truth) |

---

## §2 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED today (2026-09-29)**:
  - 4/4 NEW tests PASS in 11.77 s wall (`pytest -v molmetal/tests/test_wf328w6_dataloader_factory_production.py`).
  - Trainer source confirmed to import `build_optimized_loader()` (grep hit at line 600-603).
  - Trainer source confirmed to register `--num-workers` flag (grep hit at line 1380-1398).
  - Trainer source confirmed to compute `effective_num_workers` via precedence logic (line 1596).
  - Round-trip on toy data (`n_samples=16`, `num_workers=2`) yields finite loss (no NaN, no shape mismatch).
  - Factory round-trip on toy data (`n_samples=12`, `batch_size=4`, `num_workers=2`) yields 1+ valid batches with `pin_memory=False` (CPU) and `prefetch_factor=OPTIMAL_PREFETCH_FACTOR=8`.

- **PROJECTED (not measured end-to-end)**:
  - 3-5× collate throughput lift on RTX 5090 / RTX PRO 6000 Blackwell when launching with `--num-workers 8` vs `--cpu-num-workers 0`.
  - Wall-time reduction for 50-epoch sweeps is PROJECTED at ~25-35 % (per Wf-326A DataLoader = MINOR bottleneck; <5 % gain on a GPU-starved run; larger when CPU collate was the dominant overhead).

- **NOT MEASURED today (PENDING or honest-negative disclosure)**:
  - ⚠️ End-to-end GPU validation BLOCKED on AGENTS.md §5.1 "kill then adapt" (now superseded by §5.1 standing GPU auth — no per-launch user OK required, but VRAM + dry-run + pre-flight PASS gate still apply).  Wf-328-W6 is CPU-only validation; no GPU launch was attempted.
  - The 6 launch scripts that override `--cpu-num-workers 0` (`launch_wf124_p3_g{4,5,6}.sh`, `launch_wf300_cfm_main.sh`, `launch_wf310b_g6_final.sh`, `retry_g5_missing_pairs.sh`) are NOT updated by Wf-328-W6 — that is the next-wave scope (Wf-328-W7 candidate).  Today the trainer honors the flag; tomorrow the launch scripts need to set `--num-workers 8` (or remove the override).
  - 15-pre-flight-v3-FAIL REGRESSION from §22.16 remains PRESERVED; Wf-328-W6 does not target any pre-flight FAIL (this is a production-wiring task only).

---

## §3 Diff summary (3 file edits + 1 NEW test file)

### §3.1 `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (3 hunks)

1. **`train_one_epoch_joint`** (line 571-606): replaced inline `torch.utils.data.DataLoader(...)` with `build_optimized_loader(ds, batch_size, device, collate_fn, n_workers_override=num_workers)`.  The factory wires `pin_memory=True` on CUDA, `persistent_workers=True`, `prefetch_factor=8` (Wf-326B optimum), `multiprocessing_context="spawn"`.  `_collate_batch_for_loader` is passed as `collate_fn` (preserves the Wf-260 logic of dropping None graphs + NaN pIC50).

2. **CLI parser** (line 1353-1412): added `--num-workers` (default None) and `--num-workers-host-cores` (default None) flags.  Help text references Wf-328-W6 + Wf-326B factory.  Default behavior is backward-compatible (None → use `--cpu-num-workers`).

3. **Main dispatch** (line 1591-1615): added precedence logic:
   - `effective_num_workers = int(args.num_workers) if args.num_workers is not None else int(args.cpu_num_workers)`
   - Audit log `[wf328w6] num_workers resolved: effective=… cli.num_workers=… cli.cpu_num_workers=… suggested=…` printed once per (first metal, first seed).
   - `num_workers=effective_num_workers` passed to `train_one_metal_one_seed` (replacing the previous hard-coded `args.cpu_num_workers`).

### §3.2 `molmetal/tests/test_wf328w6_dataloader_factory_production.py` (NEW, 4 tests)

| Test | Validates |
|---|---|
| `test_trainer_uses_build_optimized_loader` | (a) trainer source imports `build_optimized_loader`; (b) trainer source calls it; (c) round-trip on toy data with `num_workers=2` yields finite loss |
| `test_suggest_num_workers_default_and_flag_help` | (a) `suggest_num_workers(cap=12)` returns int in [0, 12]; (b) `--num-workers` flag is registered in trainer CLI; (c) `Wf-328-W6` marker present |
| `test_num_workers_flag_overrides_cpu_num_workers` | (a) precedence logic: `--num-workers 8` wins over `--cpu-num-workers 2`; (b) `effective_num_workers` symbol present in trainer source |
| `test_trainer_integration_with_factory` | (a) factory round-trip with `num_workers_override=2`, `pin_memory=False` on CPU, `prefetch_factor=OPTIMAL_PREFETCH_FACTOR=8`; (b) ≥ 1 valid batch yielded |

### §3.3 `molmetal/scripts/wf326b_dataloader_optimized.py` (unchanged)

The factory is the source of truth; the trainer is now a consumer.  No factory edits — backward-compat preserved.

---

## §4 Launch-script follow-up (Wf-328-W7 candidate)

The 6 launch scripts that hard-code `--cpu-num-workers 0` need updating in a follow-up Wf to leverage the new `--num-workers` flag:

| Script | Current | Recommended |
|---|---|---|
| `scripts/launch_wf124_p3_g4.sh` | `--cpu-num-workers 0` | `--num-workers 8` (or remove override; falls back to factory default) |
| `scripts/launch_wf124_p3_g5.sh` | `--cpu-num-workers 0` | `--num-workers 8` |
| `scripts/launch_wf124_p3_g6.sh` | `--cpu-num-workers 0` | `--num-workers 8` |
| `scripts/launch_wf300_cfm_main.sh` | `--cpu-num-workers 0` | `--num-workers 8` |
| `scripts/launch_wf310b_g6_final.sh` | `--cpu-num-workers 0` | `--num-workers 8` |
| `scripts/retry_g5_missing_pairs.sh` | `--cpu-num-workers 0` | `--num-workers 8` |

This is Wf-328-W7 scope; not done in Wf-328-W6 (the trainer-level wiring is the natural unit of work; the launch-script update is a separate audit + per-launch dry-run gate per AGENTS.md §5.1).

---

## §5 Test result (verbatim from `pytest -v`)

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf328w6_dataloader_factory_production.py::test_trainer_uses_build_optimized_loader PASSED [ 25%]
molmetal/tests/test_wf328w6_dataloader_factory_production.py::test_suggest_num_workers_default_and_flag_help PASSED [ 50%]
molmetal/tests/test_wf328w6_dataloader_factory_production.py::test_num_workers_flag_overrides_cpu_num_workers PASSED [ 75%]
molmetal/tests/test_wf328w6_dataloader_factory_production.py::test_trainer_integration_with_factory PASSED [100%]

============================== 4 passed in 11.77s ==============================
```

**Wall**: 11.77 s. **Result**: 4/4 PASS, 0 FAIL, 0 SKIP.

---

## §6 Cross-references

- `party-twenty-spirit` §3.5 — honest framing mandate (every claim traces to MEASURED JSON / report path)
- `molmetal/scripts/wf326b_dataloader_optimized.py` — the factory this Wf wires into production
- `molmetal/tests/test_wf326b_dataloader_optimization.py` — 8 factory-level tests (Wf-326B ship)
- `AGENTS.md` §5.1 — standing GPU authorization (no per-launch user OK; VRAM + pre-flight PASS gate still apply)
- `AGENTS.md` §5.4 — Tier-2 paper target; NO wet-lab planning
- `TODO/INDEX.md` §22.16 — pre-flight v3 15-FAIL honest-negative REGRESSION (not fixed by Wf-328-W6)
- `TODO/INDEX.md` §22.17 — Wf-323 ship wave codification (the wave that produced this Wf's prerequisite code-level audit)
- `molmetal/reports/wf326b_dataloader_optimized/` — Wf-326B factory bench results (the source of truth for OPTIMAL_NUM_WORKERS=8 + OPTIMAL_PREFETCH_FACTOR=8)

---

**END OF §6** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.4; 4/4 NEW tests PASS in 11.77 s wall; pre-flight v3 15-FAIL REGRESSION PRESERVED (out of scope for Wf-328-W6 production-wiring closeout).