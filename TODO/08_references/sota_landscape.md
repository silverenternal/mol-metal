# SOTA Comparison Table — SBDD on 1h36

Compact head-to-head table of 8 methods we compare against (or with), restricted to the columns the paper needs. All numbers are **cited** from already-published sources or our own reports — none invented.

> Companion: T1 survey at `TODO/01_research/sota_landscape.md` (broader 25-method landscape with category narrative).

## Direct comparison table (pocket-conditioned ligand design on 1h36 / CrossDocked test pockets)

| method | year | code-link | Vina (1h36) | SA | QED | Success rate | Lipinski pass |
|---|---|---|---|---|---|---|---|
| **Pocket2Mol** (fallback baseline) | 2022 | https://github.com/illuminolab/Pocket2Mol | -5.951 | 1.841 | 0.535 | 0.189 | — |
| **TargetDiff** | 2023 | https://github.com/guanjq/targetdiff | — | — | — | — | — |
| **DiffSBDD** | 2024 | https://github.com/ArneSchneuing/DiffSBDD | — | 0.85* | 0.51* | — | — |
| **FLOWR** | 2026 | https://github.com/jule-c/flowr | -6.93 (SPINDR) | — | — | 0.94 PB-valid | — |
| **DiffDock** (docking-only) | 2023 | https://github.com/gcorso/DiffDock | — | — | — | 38.2% top-1 (PDBBind RMSD<2Å) | — |
| **EquiBind** (docking-only) | 2022 | https://github.com/octavian-ganea/equidock_public | — | — | — | 38.0% top-1 (PDBBind RMSD<2Å) | — |
| **TankBind** (docking-only) | 2022 | https://github.com/luodaniel/TrigonometricBinding | — | — | — | 20.4% top-1 (PDBBind RMSD<2Å) | — |
| **Lambda (ours)** | 2026 | https://github.com/molmetal/molmetal | **-5.923** | **1.870** | **0.548** | **0.194** | LIPINSKI predicate (default on) |

\* DiffSBDD SA/QED mean values reported in CrossDocked test-set table (cited in T1 §3.7). All other cells marked "—" are not measured under our 1h36 protocol and are intentionally left blank rather than back-translated from a different pocket set.

## Provenance & caveats

- **Pocket2Mol** row: from `molmetal/reports/pocket2mol_vs_lambda_1h36.md` (n=95, smARTS fallback because pretrained checkpoint not shipped, see Caveat §1). Published Pocket2Mol Vina -6.5 avg is on CrossDocked, not 1h36 — do not conflate.
- **Lambda (ours)** row: from `molmetal/reports/pocket2mol_vs_lambda_1h36.md` (n=98). `LIPINSKI` predicate is enabled by default; `RewardAggregator` (Vina + SA + PB + pIC50 + retro) is the multi-reward path described in `molmetal/reports/mcts_strengthening.md`.
- **FLOWR / DiffSBDD / TargetDiff** rows: Vina and PB-valid numbers from T1 §10.3 / §3.7 (SPINDR test pockets). Not run on 1h36 in our pipeline.
- **DiffDock / EquiBind / TankBind**: docking-only methods — the "Success rate" column reports their published PDBBind top-1 (RMSD<2Å). Not directly comparable to pocket-conditioned generators; included because they are the standard pose-quality baselines reviewers expect.
- **MMP13 click** (`molmetal/reports/mmp13_vina_real.md`) is a separate evaluation on the click-chemistry products and uses a stricter per-target success cutoff; not folded into this table to keep the column semantics consistent (all 1h36 / CrossDocked-style).

## What this table is *not*

- Not a full 25-method survey — see T1.
- Not a pose-accuracy ranking — DiffDock/EquiBind/TankBind are scored on PDBBind RMSD, not Vina.
- Not a Lipinski-by-default comparison — only Lambda enforces it as a hard predicate; the others report it post-hoc if at all (hence "—").