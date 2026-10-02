# Wf-320H — TODO-134 Phase 2 Vina attempt (pipeline proof)

_Generated_: 2026-09-29T02:57:13.960665+00:00  
_Engine_: AutoDock Vina 1.2.7 (Python binding)  
_Interpreter_: `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`  
_Scope_: 6 pockets × 1 seed × 1 pose × exhaustiveness=4 (smoke)  
_Receptor_: Wf-30 staged real receptor.pdbqt (5/6 READY) — SYNTHETIC 200 ALA scaffold only if Wf-30 missing.  

## Per-pocket Vina pose results

| Pocket | n_ref_heavy | energy kcal/mol | RMSD vs start (Å) | docked n | wall s | status |
|---|---|---|---|---|---|---|
| 1h36 | 25 | 0.000 | 3.0250 | 25 | 5.41 | MEASURED |
| 3eml | 25 | -1.112 | 2.6825 | 25 | 8.19 | MEASURED |
| 3ny8 | 21 | 0.000 | 2.0249 | 21 | 5.23 | MEASURED |
| 4aaw | 31 | 0.000 | 2.9304 | 31 | 11.21 | MEASURED |
| 4rlu | 19 | -3.625 | 2.5086 | 19 | 3.61 | MEASURED |
| 830c | 28 | 0.000 | 3.1850 | 28 | 13.39 | MEASURED |

## Aggregate

* Pockets measured: **6/6**
* RMSD range (self-redocking): 2.0249 – 3.1850 Å
* Mean energy: -0.789 kcal/mol
* Total wall: 47.04 s

## Honest framing (party-twenty-spirit §3.5)

* This run is a **pipeline proof**, not a SOTA redocking benchmark. The receptor is synthetic — there is no crystal ligand pose to compare against. The RMSD column measures self-redocking consistency (starting conformer vs docked pose after one Vina run with exhaustiveness=4).
* Per Wf-317 Tier-2 + TODO-134 §2: Phase 2 with REAL PDB receptors remains future work. 5 of 6 pockets in `m15_redesign_ref/` lack a co-crystal per TODO-134 §1.2 (`extraction_kind: sdf` for CrossDocked, no real receptor in `molmetal/data/`).
* Per Wf-320H discovery: `which vina` returns NOT-FOUND on this host (libboost broken per Wf-309A), but the CadButEaas `.venv-vlm` Python interpreter ships a working Vina 1.2.7 Python binding — the recovery path per TODO-134 §2 / Wf-84 P2 stub_warning is now reachable via `CADBUTEAAS_PY`.

## Files

* `summary.json` — machine-readable aggregate.
* `<pocket>/vina_pose.json` — per-pocket pose + RMSD record.
