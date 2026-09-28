# Wf-248-C — Per-pocket Vina config adapter (TODO-125)

**Status**: in-progress
**Date**: 2026-09-28
**Owner**: Hugo Lee + ultracode
**Parent**: TODO-125 / `mol-metal-anticancer` / `mol-metal-cfm-training` skill alignment

---

## §0 Why this workflow exists

TODO-125 §0 documented that the per-pocket ``vina_high_affinity`` rate
spread (1h36 = 0.95 vs 3eml = 0.05) reflects a real distribution
mismatch AND that per-pocket docking configuration (exhaustiveness +
n_poses) could also affect pose quality.

The existing project stack used a SINGLE global
``DockingConfig(n_poses=5, exhaustiveness=8)`` for all 5 (1h36, 3eml,
3ny8, 4aaw, 4rlu) pockets. This workflow introduces
:class:`PerPocketVinaConfig` — a tiny adapter that maps each pocket
to its own exhaustiveness + n_poses values:

| Pocket | exhaustiveness | n_poses | Label       | Rationale              |
|--------|----------------|---------|-------------|------------------------|
| 1h36   | 16             | 10      | small-tight | MMP13 Zn-coord, fine sampling needed |
| 3eml   | 12             | 8       | mid-tight   | DHFR folate pocket     |
| 3ny8   | 10             | 6       | mid-loose   | β2 adrenergic receptor |
| 4aaw   | 10             | 6       | mid-loose   | Carbonic anhydrase II  |
| 4rlu   | 8              | 5       | large-loose | GST large pocket       |

This is the Wf-248-C deliverable: a small, focused adapter that
PROVES per-pocket DRS tuning is wired into the project, even before
any exhaustiveness-vs-affinity calibration study.

---

## §1 Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `molmetal/adapters/per_pocket_vina_config.py` | ~250 | `PerPocketVinaConfig` adapter + `PocketVinaConfig` dataclass |
| `molmetal/scripts/dock_per_pocket_vina.py` | ~340 | CLI smoke runner (5 pockets × 5 smiles × 3 seeds = 75 cells) |
| `molmetal/tests/test_per_pocket_vina.py` | ~330 | 21 unit tests (all PASS) |
| `molmetal/reports/wf248c_per_pocket_vina/` | this | Stage 2 docs + per-pocket JSONs + summary |

---

## §2 Acceptance gates (FLAT)

| Gate | Result | Evidence |
|------|--------|----------|
| G1_vina_adapter_located      | True   | `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` (NOTE: task said `molmetal/adapters/vina_adapter.py` — actual path is `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`; see §6 honest-framing). |
| G2_per_pocket_config         | True   | `PerPocketVinaConfig.DEFAULT_POCKET_MAP` covers all 5 pockets. |
| G3_adapter_class             | True   | `PerPocketVinaConfig` + `PocketVinaConfig` dataclass in `molmetal/adapters/per_pocket_vina_config.py`. |
| G4_cli_script                | True   | `molmetal/scripts/dock_per_pocket_vina.py` (340 LOC). |
| G5_test_added                | True   | `molmetal/tests/test_per_pocket_vina.py` — 21 tests PASS. |
| G6_smoke_75poses             | True   | `smoke_summary.json` `metadata.n_cells == 75` (5 pockets × 5 smiles × 3 seeds). |
| G7_per_pocket_exhaustiveness | True   | `smoke_summary.json.per_pocket_exhaustiveness == {"1h36": 16, "3eml": 12, "3ny8": 10, "4aaw": 10, "4rlu": 8}` |
| G8_report_written            | True   | `molmetal/reports/wf248c_per_pocket_vina/README.md` (this file). |

---

## §3 Test results

```
$ pytest molmetal/tests/test_per_pocket_vina.py -v
======================== 21 passed, 1 warning in 0.32s =========================
```

Test classes:
- `TestPocketVinaConfig` (5 tests) — dataclass defaults, validation.
- `TestPerPocketVinaConfig` (10 tests) — adapter lookup, overrides,
  fallback, `to_docking_config`, exhaustiveness/n_poses methods.
- `TestDockPerPocketVinaCLI` (3 tests) — full CLI smoke + 75-cell
  smoke + help.
- `TestPerPocketWiring` (2 tests) — adapter wires into `DockingConfig`.

---

## §4 Smoke results (75 cells)

CLI invocation (hermetic — no Vina binary required):

```
molmetal/.venvs/mol-metal-venv/bin/python \
    molmetal/scripts/dock_per_pocket_vina.py \
    --pockets 1h36 3eml 3ny8 4aaw 4rlu \
    --smiles CCO 'c1ccccc1' 'CC(=O)Oc1ccccc1C(=O)O' \
             'Nc1ncnc2[nH]cnc12' 'O=C(O)c1ccccc1' \
    --seeds 0 1 2 \
    --out-dir molmetal/reports/wf248c_per_pocket_vina
```

CLI output (last line):

```
[wf248c] DONE pockets=5 smiles=5 seeds=3 cells=75 real_vina=0 stub_fallback=75
per_pocket_exhaustiveness={'1h36': 16, '3eml': 12, '3ny8': 10, '4aaw': 10, '4rlu': 8}
```

Per-pocket JSONs written (one per pocket):
- `molmetal/reports/wf248c_per_pocket_vina/1h36/vina_per_pocket.json`
- `molmetal/reports/wf248c_per_pocket_vina/3eml/vina_per_pocket.json`
- `molmetal/reports/wf248c_per_pocket_vina/3ny8/vina_per_pocket.json`
- `molmetal/reports/wf248c_per_pocket_vina/4aaw/vina_per_pocket.json`
- `molmetal/reports/wf248c_per_pocket_vina/4rlu/vina_per_pocket.json`

Combined CSV: `molmetal/reports/wf248c_per_pocket_vina/per_pocket_vina.csv`
(76 lines = 1 header + 75 data rows).

---

## §5 Per-pocket exhaustiveness map (gate G7 evidence)

```json
{
  "1h36": 16,
  "3eml": 12,
  "3ny8": 10,
  "4aaw": 10,
  "4rlu": 8
}
```

This is exactly the spread the Wf-248-C spec requested:
- 1h36 (small tight pocket) → 16 (max)
- 4rlu (large loose pocket) → 8 (baseline)
- The other three pockets fall on a smooth interpolation.

---

## §6 Honest framing

1. **Path mismatch**: The task description said
   ``molmetal/adapters/vina_adapter.py``. The actual file is at
   ``molmetal/molmetal_lam/sbdd_env/vina_adapter.py``. The adapter
   module `PerPocketVinaConfig` itself was created at
   ``molmetal/adapters/per_pocket_vina_config.py`` (the path the
   task asked for). The mismatch is the task description vs the
   project layout — the adapter DOES depend on
   ``molmetal.ports.DockingConfig`` (the canonical dataclass that
   ``molmetal_lam/sbdd_env/vina_adapter.py`` consumes).

2. **STUB_FALLBACK only**: This machine has no Vina binary
   (``vina``/``qvina02``/``smina`` on PATH), so the smoke runner
   correctly records every cell as ``STUB_FALLBACK`` with
   ``vina_score_kcal_mol=None``. The per-pocket config IS honoured
   (exhaustiveness + n_poses appear in every per-cell record) —
   only the kcal/mol emission is stubbed. No kcal/mol value is
   fabricated. A real-Vina smoke run on a host with the binary is
   a follow-up.

3. **Per-pocket values are heuristic**: The exhaustiveness + n_poses
   values are based on pocket-size heuristics (small → more sampling,
   large → baseline). They are NOT calibrated against an external
   benchmark. A future workflow (Wf-?) should bench
   exhaustiveness-vs-affinity per pocket (n=500 cohort) to validate
   the heuristic.

4. **Wf-179 P2 reference**: The task description says
   ``Smoke test on 5 pockets (Wf-179 P2 75 docked poses)``. The
   Wf-179 P2 summary.json on disk shows 65 actual poses
   (3ny8 has 1 seed instead of 3). The smoke matrix is
   5 × 5 × 3 = 75 cells, matching the EXPECTED reference count.

5. **Future work**:
   - Bench per-pocket exhaustiveness-vs-affinity on a real-Vina
     host (n=500 cohort per pocket).
   - Wire `PerPocketVinaConfig` into the Wf-126 P3 / Wf-142 P1
     `bench_vina_per_pocket_closure.py` runner so the per-pocket
     settings propagate end-to-end.
   - Promote pocket_box `center`/`size` overrides from None to
     real values (per-pocket PDBQT receptor centre).

---

## §7 Cross-references

- TODO-125 (`TODO/completed/125_vina_per_pocket_strengthen.md`)
  — original per-pocket Vina strengthen plan.
- `molmetal/scripts/bench_vina_per_pocket_closure.py` (Wf-126 P3)
  — sister script using single global exhaustiveness; future
  follow-up can swap to `PerPocketVinaConfig.to_docking_config(...)`.
- `molmetal/reports/wf179_bd9_m15_rmsd_p2/summary.json` —
  Wf-179 P2 reference (65-75 expected poses, 5 pockets × 3 seeds
  × 5 poses).
- `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` — actual Vina
  adapter module that consumes `DockingConfig`.
- `molmetal/ports/__init__.py` — `DockingConfig` dataclass.

---

## §8 Output JSON schema (`smoke_summary.json`)

```json
{
  "metadata": {
    "workflow": "Wf-248-C",
    "task": "Per-pocket Vina config adapter (TODO-125)",
    "pockets": ["1h36", "3eml", "3ny8", "4aaw", "4rlu"],
    "seeds": [0, 1, 2],
    "smiles": [...],
    "n_cells": 75
  },
  "vina_binary_status": {
    "status": "REAL_VINA_AVAILABLE" | "MISSING_DEP",
    "bin": ...,
    "version": ...,
    "reason": ...
  },
  "per_pocket_config_map": {...},
  "per_pocket_exhaustiveness": {"1h36": 16, "3eml": 12, ...},
  "per_pocket_n_poses": {"1h36": 10, "3eml": 8, ...},
  "n_real_vina_total": ...,
  "n_stub_fallback_total": ...,
  "score_summary": {...},
  "per_pocket_aggregate": {...},
  "honest_framing": {...}
}
```