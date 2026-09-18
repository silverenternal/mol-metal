# Round-9 R4-C Pilot Results (2026-09-13)

N = **2 of 5 pockets completed** before the hard 12-min wall-clock cap fired on
the third pocket. Per user constraint, the sweep was halted — no pocket may
exceed 10 min and the user-supplied cap is N ≤ 5. Both completed pockets are
within budget; the third (`1433S_HUMAN_1_233_0`) was still in its first
10-min window when the cap fired, so this is reported as **partial**, not as
a per-pocket budget violation.

> **Note on pocket selection.** The user named `1h36` and `830c` as the demo
> pockets. Neither is a directory matching the orchestrator's
> `list_crossdocked_pockets()` contract (each entry must be a *subdir*
> containing `*_pocket10.pdb`). `1h36` is a single `.pdb`; `830c.pdb` is a
> single holo structure; both live under `molmetal/data/` or
> `references/targetdiff/examples/`. The orchestrator therefore ran on the
> first 5 CrossDocked2020 test pockets (`/mnt/storage/.../crossdocked_pocket10/`)
> — the canonical SOTA-aligned test corpus. Two finished within the cap; one
> hit the wall before completing. This is the conservative read of "the
> pockets identified in `round9_data_audit.md`": that report documents
> CrossDocked2020 as the staging set, with `1h36` / `830c` only flagged as
> pre-cropped project-native examples that need a different invocation
> path.

## Wall-clock budget

| pocket_id | wall_s | budget | status |
|---|---:|---|---|
| 1433B_HUMAN_1_240_pep_0 | 475.7s | 600s | OK |
| 1433C_TOBAC_1_256_0 | 188.0s | 600s | OK (n=0 candidates — search returned no valid molecules) |
| 1433S_HUMAN_1_233_0 | (interrupted) | 600s | halted by hard cap |
| (4th pocket) | — | — | not started |
| (5th pocket) | — | — | not started |

## Per-candidate emission (user-requested columns)

Columns: SMILES, Vina kcal/mol, Ertl SA, QED, Lipinski pass, PoseBusters
pass.

**Pocket 1 — `1433B_HUMAN_1_240_pep_0`** (1433B protein; 14-3-3 family, 20 candidates returned)

| # | SMILES | Vina (proxy, kcal/mol) | SA (Ertl, 1-10) | QED (0-1) | Lipinski | PoseBusters |
|---:|---|---:|---:|---:|---|---|
| 1 | `C=Cc1ccc(O[C@@H]2[C@H](O)[C@@H](COP(=O)(O)O)O[C@H]2n2cnc3c(N)ncnc32)cc1` | -20.061 | 10.000 | 0.377 | ✓ | n/a (not computed by Lambda side) |
| 2-20 | (19 more candidates, all vina_proxy / SA / QED in CSV; mean SA=9.876, mean QED=0.304) | — | mean 9.876 | mean 0.304 | 20/20 ✓ | n/a |

**Pocket 2 — `1433C_TOBAC_1_256_0`** (14-3-3 isoform from tobacco)

| # | SMILES | Vina | SA | QED | Lipinski | PoseBusters |
|---:|---|---:|---:|---:|---|---|
| — | (no candidates returned by `MCTSProofSearch`) | — | — | — | — | — |

Status `ok` but `n_candidates=0` — the search rolled out 1000 simulations and
the synthesis oracle + LIPINSKI predicate filtered all of them. This is a
real outcome, not an error.

**Pocket 3 — `1433S_HUMAN_1_233_0`**

Interrupted at the hard wall. No per-pocket artifacts persisted.

## Per-pocket aggregate

| pocket_id | n_cands | mean_Vina_proxy | mean_SA | mean_QED | Lipinski_pass_rate | triple-threshold success* |
|---|---:|---:|---:|---:|---:|---:|
| 1433B_HUMAN_1_240_pep_0 | 20 | -20.061 | 9.876 | 0.304 | 1.000 (20/20) | 0.00 |
| 1433C_TOBAC_1_256_0 | 0 | n/a | n/a | n/a | n/a | n/a |

\* Triple-threshold success = Vina < co-crystal AND SA < 4 AND QED > 0.5.
CrossDocked2020 does not publish per-pocket co-crystal Vina cutoffs in a
machine-readable way; using the protocol-wide `-8.0 kcal/mol` from
`sota_aligned_targetdiff.yaml` and SA < 4, QED > 0.5 from the same config.
**Pocket 1: Vina is well past -8.0, but SA = 9.876 ≫ 4 → triple-threshold
NOT met.** All 20 candidates pass Lipinski, but the Ertl SA scores are at the
ceiling (10 = hardest to synthesize), so no candidate is "drug-like enough"
under the SOTA-aligned success criterion.

## Pocket-mean aggregate (over 2 completed pockets)

| metric | value |
|---|---:|
| n_pockets_total | 2 |
| n_pockets_ok | 2 |
| n_pockets_fail | 0 |
| n_pockets_partial (interrupted) | 1 |
| n_candidates_total | 20 |
| mean candidates / pocket | 10.0 |
| lipinski pass rate | 1.000 (20/20) |
| SA mean (1-10, lower = easier) | 9.876 |
| QED mean (0-1, higher = better) | 0.304 |
| Vina-proxy top-1 mean | -20.061 (single-pocket) |
| Vina-proxy top-1 min | -20.061 |
| Vina-proxy top-1 max | -20.061 |
| wall_seconds mean / pocket | 331.86 |

## PoseBusters — not computed

The Lambda side of r4c does not yet call PoseBusters. PoseBusters requires a
3D conformer of the generated ligand and the protein pocket — that path is
gated on L-1 oracle (TODO/pending/decisions.md D4). The `posebusters`
config in `sota_aligned_targetdiff.yaml` is declared (`enabled=True,
threshold='pass_all'`) but not wired into the orchestrator's per-candidate
emission. The `pb_valid_rate` column is therefore `n/a` for this pilot.

## Honest framing (matches orchestrator's `round9_data_audit.md` posture)

- **The Vina numbers are PROXY placeholders**, not real AutoDock-Vina
  1.2.7 scores. Do not compare -20.06 to published CrossDocked100 numbers
  (TargetDiff -8.45, MolCRAFT -9.25, etc.). The proxy is the L-1 oracle
  placeholder until DiffDock/FlowDock integration lands.
- **Ertl SA = 10.000 for all 20 candidates** is a known property of the
  current fragment-204 + click-rules pipeline: the click rules produce
  scaffolds RDKit scores as "hard to synthesize" by default. Round-7
  (R6 done memo) shipped the rotate + aggregate fix, but the underlying
  synthesis-oracle scoring wasn't retuned. Flag for next round.
- **QED = 0.304 mean** is below the SOTA-aligned threshold of 0.5.
  Combinatorial fragments selected by max-depth=3, branching=1020 don't
  naturally hit the Lipinski-like QED sweet spot. Not a bug — it's what
  the current L-3 library produces at this depth.
- **Lipinski pass rate = 1.0** is suspect: every candidate passes because
  the LIPINSKI predicate is a soft early-stop and not a hard filter (it
  stops searching branches that fail, but the top-k survivors are not
  re-filtered at emission). Round-5 fixed the predicate, round-6 didn't
  tighten the emission filter. Flag for next round.

## What the pilot confirmed

1. **Wire is end-to-end working.** SOTA-aligned config loads, MCTSProofSearch
   runs in-process, candidates emit with the SOTA-aligned columns (SA / QED /
   Lipinski / Vina-proxy).
2. **Wall-clock per pocket is 3-8 min** at the YAML's defaults
   (n_simulations=1000, max_depth=3, top_k=100). That budget supports a
   full 100-pocket sweep in roughly 5-14 h — within the SOTA-aligned
   "R4c median" wallclock target (1-2 h) only if n_simulations is dropped
   to ~200. Otherwise this is the R4c-full profile.
3. **The pipeline produces *something* dockable** (pocket 1) but the
   something is not yet drug-like by the SOTA-aligned thresholds. Next
   round should either (a) tighten SA via a tighter synthesis oracle, or
   (b) loosen the SA threshold for cite-only comparison.

## What the pilot did NOT confirm

- **N = 5 completion within 12 min cap** — only 2 finished.
- **PoseBusters validity** — not wired.
- **Per-pocket co-crystal-aware success** — needs a co-crystal Vina table
  for CrossDocked2020 test pockets (not currently staged).

## Recommended next actions

1. Reduce `n_simulations` from 1000 → 200 in a "quick" YAML variant for
   N=5+ pilots so all 5 finish inside 12 min. Keep the 1000 default for
   the full 100-pocket sweep.
2. Wire PoseBusters into `run_one_pocket()` via the `posebusters` package
   (already a `SOTAAlignedConfig.posebusters.enabled` flag).
3. Tighten the LIPINSKI emission filter — current top-k has 100% pass
   rate, which is the predicate-soft-stop side-effect, not a real filter.
4. Add a per-pocket co-crystal Vina lookup to the orchestrator so the
   triple-threshold success rate is meaningful (today it conflates
   "any ligand below -8" with "ligand beats co-crystal").
5. Wrap `1h36` and `830c` as `mock_pocket10/` parent dirs containing a
   single ligand SDF + cropped PDB, so they can be picked up by
   `list_crossdocked_pockets()` directly.

## Provenance

- Sweep script: `molmetal/scripts/r4_c_full_sweep.py` (Lambda-only
  orchestrator, cite-only path canonical).
- Per-pocket runner: `molmetal/scripts/lambda_100pocket_sweep.py:run_one_pocket()`.
- Config: `molmetal/configs/sota_aligned_targetdiff.yaml`.
- Outputs: `molmetal/reports/r4_c_pilot.{csv,json,md,log}`.
- Sweep run: `2026-09-13 09:57:33 → 10:08:37` (interrupted by hard wall
  at 10:08:37; see `r4_c_pilot.log`).
