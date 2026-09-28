# Wf-63 Phase 3 — m8 PoseBusters production re-run (script4 retry)

**Date (UTC)**: 2026-09-26T08:48:22.376837+00:00
**Configuration**: pb_mode=mol (chemistry only, 14 checks), n_cands=5, seeds=1, use_real_mcts=False
**Real PoseBusters binary installed**: True
**Target**: m8 PB pass_rate ≥ 0.70
**Verdict**: **BELOW_TARGET**

## Aggregate

| Metric | Value |
|---|---:|
| Pockets requested | 10 |
| Cells completed | 10 |
| PB-eligible cells (≥1 candidate passed PB) | 10 |
| PB attempted (sum across cells) | 50 |
| PB pass total (sum across cells) | 0 |
| m8 PB pass rate (macro, per-pocket) | 0.000 |
| m8 PB pass rate (micro, per-molecule) | 0.000 |
| Wall seconds total | 2.144 |

## Per-pocket table

| pocket | n_cands | n_eligible | n_pass | m8 rate | pb_status | wall_s |
|---|---:|---:|---:|---:|---|---:|
| 830c | 5 | 5 | 0 | 0.000 | ok | 0.232 |
| 1h36 | 5 | 5 | 0 | 0.000 | ok | 0.213 |
| 3eml | 5 | 5 | 0 | 0.000 | ok | 0.213 |
| 3ny8 | 5 | 5 | 0 | 0.000 | ok | 0.212 |
| 4aaw | 5 | 5 | 0 | 0.000 | ok | 0.212 |
| 4rlu | 5 | 5 | 0 | 0.000 | ok | 0.212 |
| 1g9v | 5 | 5 | 0 | 0.000 | ok | 0.212 |
| 2hnx | 5 | 5 | 0 | 0.000 | ok | 0.213 |
| 4rn0 | 5 | 5 | 0 | 0.000 | ok | 0.212 |
| 1fmc | 5 | 5 | 0 | 0.000 | ok | 0.212 |

## Status counts

- `ok`: 10

## Honest framing

* **Real PB binary**: True. Every cell ran the real PoseBusters ``mol``-mode adapter (14 chemistry checks).
* **Vina availability**: not required for this phase (PB ``mol`` mode needs only RDKit + 3D embedding).  Wf-63 Phase 1 SHIP the Vina install status; Phase 3 re-runs only the m8 portion of script4.
* **Candidate source**: fixture list from ``molmetal/scripts/_demo_5_metaldrugs.txt`` (8 known Pt/Ru/Ir/Au/Pd drugs).
* **Target gate**: m8 ≥ 0.70.  Macro per-pocket = 0.000; micro per-molecule = 0.000.

## Reference

* Buttenschoen, Morris & Deane, *PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences*, Chem. Sci. 2024, 15, 3130-3139. doi:10.1039/D3SC04185A
* Wf-47 Phase 4 partial ship: `molmetal/reports/wf47_real_runs/script_pb_run_pb_production/` (EMPTY).
* Wf-63 Phase 1 (Vina install status): see commit ``e59a378``.

