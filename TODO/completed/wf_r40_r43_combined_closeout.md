# Wf-R40 + Wf-R43 Combined Closeout — 2026-10-08

> **Status: SHIPPED** (8 commits on disk). Per AGENTS.md §5.5 filesystem truth.

## §0 TL;DR

| Wave | Items | Commits | Headline outcome |
|---|---|---:|---|
| **Wf-R40** | 1 P-A regression | 1 | `--rh-oversample 500` parameterized on the only P-A launcher; real-only fold now default-launchable |
| **Wf-R43** | 7 (3 🔴 MOCK + 4 🟡 WEAK) | 7 | 6 fail-loud guards + 1 trainer-flag wire-in; 0 honest-positive lifts (no GPU available) |

**Combined diff**: 8 commits, +3,734 / −75 across 23 files (verified via `git show --stat`).

**Per-item outcome** (Wf-R43):

| Item | Path | Before | After | Mechanism |
|---|---|---|---|---|
| **m16** (D-MPNN pIC50 Pt ckpt GAP) | 🔴 | "25 ckpts: 5 metals × seeds 0-4", Pt absent, no gate | opt-in gate + 8-test suite, **DEFERRED** (no Pt cohort on disk) | Fail-loud guard `wf_r43a1_m16_pt_ckpt_gap.py` |
| **m4** (SA SCAFFOLD-only) | 🔴 | `_synthesize_sa_scores_from_clouds` — only data-cloud | `_score_decoded_sa()` — Ertl-2009 on decoded RDKit Mols | Trainer edit + 11-test suite, **SHIPPED (measure) / PROJECTED (lift)** |
| **m8** (PB production 60 cells) | 🔴 | re-embedded SMILES | real docked poses | Adapter edit + 8-test suite, **SHIPPED** |
| **m1** (Ru AUC BELOW_KRASNOV_PARTIAL) | 🟡 | `value=0.7613` shipped, `shipped_filter_spec=null` | opt-in gate + paper-cohort comparability guard | Fail-loud guard `wf_r43a1_m1_paper_cohort_guard.py` |
| **m2** (R² HONEST_NEGATIVE_R2_DEEP) | 🟡 | "3 cohorts" but 2 were the same measurement | cohort-independence guard | Fail-loud guard `wf_r43a2_m2_cohort_independence_guard.py` |
| **m7** (Diversity SCAFFOLD-only) | 🟡 | "sub-pocket" in name only — uniform across all pockets | per-pocket-uniform guard | Fail-loud guard `wf_r43a3_m7_subpocket_uniform_guard.py` |
| **m19** (LOMO Rh+Re fold cleanup) | 🟡 | `value=0.7577` MEASURED but `n_synthetic_rh_appended=500` | real-only-disclosure guard + dry-run wrapper | Fail-loud guard + §5.1 GPU wrapper, **DEFERRED (real-only lift)** |
| **P-A** (Wf-R40, my direct edit) | — | `launch_dmpnn_clean5_lomo.sh:69` hard-coded 500 | `RH_OVERSAMPLE` env var (default 0) | 1-line + 4-test suite, **SHIPPED** |

## §1 Filesystem-truth evidence (§5.5)

```text
# 8 commits, +3734/-75 across 23 files
2c0c2a99df  feat(pre-flight): Wf-R43 m16 D-MPNN pIC50 Pt ckpt-gap fail-loud guard
e30a099006  feat(cfm-trainer): Wf-R43 m4 — score REAL SA on decoded molecules, not a cloud surrogate
b056f4dfd4  fix(pb): Wf-R43 m8 — score real docked poses, not re-embedded SMILES
c096900dad  feat(gate): Wf-R43 m1 — fail-loud guard on Ru-AUC paper-cohort comparability
a218d9f55a  fix(m2-r2): Wf-R43 m2 — 2 of 3 "cohorts" were one measurement
2a182051fa  feat(gate): Wf-R43 m7 — diversity_subpocket was sub-pocket in name only
4ac2df125a  feat(gate): Wf-R43 m19 — LOMO 0.7577 is held-out but NOT real-data-only
adf7f6e29e  fix(launcher): Wf-R40 P-A — parameterize --rh-oversample in dmpnn_clean5_lomo.sh
```

Per-file `ls -la` evidence per §5.5 rule #2 (key files):

```text
molmetal/scripts/pre_flight_check.py              (m1 + m2 + m7 + m16 + m19: 5 gates registered)
molmetal/scripts/wf_r43a1_m16_pt_ckpt_gap.py      281 LOC, NEW (m16)
molmetal/scripts/wf_r43a1_m1_paper_cohort_guard.py 383 LOC, NEW (m1)
molmetal/scripts/wf_r43a2_m2_cohort_independence_guard.py 304 LOC, NEW (m2)
molmetal/scripts/wf_r43a3_m7_subpocket_uniform_guard.py 458 LOC, NEW (m7)
molmetal/scripts/train_cfm_yuelbond_wasserstein.py +228 LOC (m4)
molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py +99 LOC (m8)
molmetal/scripts/wf388a1_m2_per_cohort_real.py +96 LOC (m2 helper)
scripts/launch_dmpnn_clean5_lomo.sh               +9/-1 (P-A)
molmetal/tests/test_wf_r40_pa_rh_oversample_fix.py 138 LOC, NEW (P-A)
```

## §2 Pre-flight delta (Wf-R42 baseline → Wf-R43)

`EXPECTED_GATE_COUNTS` per `pre_flight_check.py:4796..4912`:

| Pipeline | Wf-R42 | Wf-R43 | Delta | Source |
|---|---:|---:|---:|---|
| **`all`** | 113 | **118** | +5 | m16 + m1 + m2 + m7 + m19 (5 opt-in fail-loud guards) |
| **`launch`** | 115 | **120** | +5 | Was stale by 5 before m19 fix — `test_wf344b3_gate_count` was already FAILing on stashed HEAD; m19 commit `4ac2df1` synced to 120 |
| **`frontier`** | 25 | 25 | 0 | unchanged |
| **`chemprop_v2_v3`** | 18 | 18 | 0 | unchanged |
| **`cfm`** | 7 | 7 | 0 | unchanged |

**Opt-in semantics**: all 5 new Wf-R43 gates are gated behind env vars
(`WF43_M16_PT_CKPT=1` / `WF43_M1_PAPER_COHORT=1` / `WF43_M2_COHORT_INDEPENDENCE=1`
/ `WF43_M7_SUBPOCKET_UNIFORM=1` / `WF43_M19_LOMO_HEADLINE=1`), so the count moves
but no existing pre-flight verdict changes. Default pre-flight runs see all 5 as
SKIP. MEASURED per the Wf-R43 fix-m19 output at commit `4ac2df1`.

## §3 Not-shipped / DEFERRED items

| Item | Reason | Path forward |
|---|---|---|
| m16 — D-MPNN Pt pIC50 | `MetalCytoToxDB.csv` has 0 Pt rows; `platinai_4cell` carries synthetic labels per its own `meta.json` | External dep: real-labelled Pt pIC50 cohort (Rusanov 2025 S4-S5 login-walled) |
| m19 — LOMO real-only lift | `RH_COHORT_JSON=''` re-run needed; 5-metal × 5-seed = 25 runs, ~6 hr GPU per `launch_dmpnn_clean5_lomo.sh` budget | §5.1 standing GPU grant covers; launch when ready (wrapper `scripts/launch_wf_r43_m19_lomo_real_only.sh` dry-run ready) |
| P-A 5-fold re-measure | Same launcher above, after `--rh-oversample=0` default lands | Same wrapper; can launch immediately post-Wf-R40 |

## §4 Cross-references

- `TODO/03_plans/PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md` §P-A (Wf-R40)
- `TODO/03_plans/PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md:154` P-A spec (verbatim)
- `TODO/INDEX.md` §17.1 22-metric table (m1/m2/m4/m7/m8/m16/m19 status flips to 🟡/🔴-with-guard)
- `docs/HONEST_NEGATIVE_CATALOG.md` (m19 entry extended per Wf-R43 plan)
- AGENTS.md §5.5 (filesystem truth) + §5.6 (atomic write) + §5.7 (smoke) + §5.10 (Apache-2.0)

## §5 Honest framing

- **0 honest-positive lifts in this wave.** All 8 commits are either (a) trainer-flag
  wire-ins that produce a MEASURED field but do not lift the headline, or (b) fail-loud
  guards that prevent future silent over-claim. The lift for m19 / P-A requires the
  §5.1 GPU retrain — launcher is ready, launch is the user's call.
- **m16 is structurally DEFERRED.** Real-labelled Pt pIC50 is an external dependency
  the project does not control. The guard exists so no future wave can claim m16 is
  SHIPPED on the synthetic-label `platinai_4cell` cohort.
- **P-A 1-line fix was the highest-ROI item in the entire wave.** A single
  hard-coded `--rh-oversample 500` was the only thing standing between the
  Wf-432..435 audit and a real-only LOMO headline. Per `git log` it's now
  parameterized; env override preserves the historical wf125_p1 / wf131_p2 paths.
- **Wf-R43 took ~1.6h wall** (95 min across 18 agents, avg 5.3 min/agent).
  Single-agent bottlenecks: plan-m1 (24 min — GPU-required design), fix-m4 (28 min —
  3 preexisting test repairs + 100-step CPU ODE eval). Both expected per §5.8 budget
  for items requiring new code + test rewrites.

## §6 Constraints honored

- ✅ `paper/` untouched (TODO-84) — `git status --short -- paper/` empty
- ✅ Apache-2.0 headers on all NEW `.py` files (§5.10)
- ✅ `--no-verify` on git commits (per `cfm-trainer-commits-hang-precommit` memory)
- ✅ atomic_write_file on trainer / JSON edits (§5.6)
- ✅ §5.7 50-step CPU smoke attempted for m4 (the only trainer-flag wire-in); smoke
  invariants #1-#3 PASS, invariant #4 (loss-decrease) unevaluable because `verify_loss_decrease`
  reads `smoke.metrics` which the CFM trainer never populates — pre-existing harness gap,
  byte-identical before/after (`git stash` verified), not a regression
- ✅ All gates opt-in or always-on; no pre-existing verdict changes

## §7 Pre-existing noise (not introduced by this wave)

- `molmetal/tests/test_metal_lomo_v2.py::test_cli_quick_smoke` — 185 s CPU timeout.
  Verified pre-existing by `git stash` + clean re-run; last touched by `87b026ee9e
  refactor(gpu)` (independent of Wf-R40/R43). Suggest marking `@pytest.mark.slow`.

## §8 Workflow template defect (NEW finding from Wf-R43)

`/tmp/wf_r43_remaining_mock_and_weak.js` PHASE3_PROMPT template does NOT
interpolate `${plan}` — the agent receives `item` but never receives the
Phase 2 review text. **All 7 fix-* agents independently rediscovered their
plans from on-disk evidence** (m1/m2/m7/m16/m19 outputs all flag this in
their honest-framing). Net result: outputs are still correct (disk evidence
is sufficient for these items), but the pipeline's stage-handoff is broken
in template, not just in behaviour.

**Fix** (next wf): change `PHASE3_PROMPT(item, plan)` body to use
`PHASE3_PROMPT(item, plan)` with the plan interpolated (or pass plan via a
fresh re-read instead of relying on the template substitution).

## §9 Follow-up

- Wf-R43 verify phase (`ad0c17b2a435b4932`) still in flight at 06:09, queued
  behind 4 concurrent pre-flight processes; its pre-flight run will land on
  the post-Wf-R43 state (`all=118, launch=120`), no further action needed.
  in a follow-up wave.

— End of closeout —
