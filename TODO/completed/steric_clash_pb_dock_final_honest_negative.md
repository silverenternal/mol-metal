# steric_clash_pb_dock — FINAL honest-negative close-out

**Date**: 2026-09-29
**Workflow**: Wf-318G (steric_clash_pb_dock FINAL close-out per AGENTS.md §5.4)
**Status**: ✅ **CLOSED — NEGATIVE_RESULT_HONEST**
**Owner**: Hugo Lee (per user directive relayed through harness)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline)

---

## §0 TL;DR

**Status**: `NEGATIVE_RESULT_HONEST`. The steric_clash_pb_dock PoseBusters
'dock'-mode metric is **wired but NOT measured on the 830c pocket in this
snapshot** — closed as honest-negative for Tier-2 scope per AGENTS.md §5.4.

- **MEASURED on disk**: `metrics/by_metric/steric_clash_pb_dock.json` carries
  `status: DEFERRED` since 2026-09-18; `metrics/by_metric/steric_clash_pb_dock_1h36.json`
  carries `MEASURED_SMOKE m8 pass rate = 1.0000 (15/15)` per Wf-315D 2026-09-29
  cross-reference (RDKit 2026.03.6 + PoseBusters 0.6.5 verified available in
  `/home/hugo/codes/CadButEaas/.venv-vlm/`).
- **Status flag**: This Wf-318G pass adds the canonical `HONEST_NEGATIVE=true`
  + `honest_negative_reason` + `honest_framing_note` + `verdict: FAIL` schema
  for the **830c pocket specifically** (per Wf-311F §7 candidate #4 "staging").
  The 1h36 pocket carries its own `MEASURED_SMOKE` tag separately and is NOT
  affected by this close-out.
- **What is preserved**: the `PoseBustersAdapter.dock(receptor_pdb, docked_pdbqt)`
  wired-in path is SHIPPED; when RDKit + posebusters are available AND receptor
  PDB + docked poses are on disk, the recompute is one function call away.

## §1 MEASURED state — what is on disk

### §1.1 830c pocket JSON (source verbatim — `metrics/by_metric/steric_clash_pb_dock.json`)

| Field | Value |
|---|---|
| `metric` | `steric_clash_pb_dock` |
| `description` | `Steric clash: PoseBusters 'dock' mode per-pose clash check between docked ligand and receptor (TODO-22 metric #16).` |
| `unit` | `pass-rate (fraction in [0, 1])` |
| `source` | `PoseBusters 'dock' mode per Buttenschoen, Morris & Deane, Chem. Sci. 2024, 15, 3130-3139, doi:10.1039/D3SC04185A — wraps RDKit + clash-distance checks against the receptor PDB.` |
| `value.rate` | `null` |
| `value.n_passed` | `null` |
| `value.n_total` | `100` |
| `value.criteria` | `PoseBusters 'dock' mode — chemistry + protein-ligand clash checks per docked pose` |
| `pocket.name` | `830c` |
| `pocket.receptor_pdb` | `/mnt/disk1/Hugo_Lee/mol-metal/molmetal/data/mmp13_real/830c.pdb` |
| `pocket.engine` | `AutoDock Vina 1.2.7` |
| `pocket.exhaustiveness` | `8` |
| `pocket.n_poses` | `5` |
| `status` | `DEFERRED` |
| `deferred_reason` | `deferred_posebusters_unavailable_AND_830c_docked_poses_missing` |

### §1.2 Wf-315D cross-reference (2026-09-29, 830c remains DEFERRED)

Per `metrics/by_metric/steric_clash_pb_dock.json` `notes` field:

> "Wf-315D cross-reference (2026-09-29): RDKit 2026.03.6 + PoseBusters 0.6.5
> verified available in /home/hugo/codes/CadButEaas/.venv-vlm/. 1h36 pocket has
> MEASURED_SMOKE m8 pass rate = 1.0000 (15/15) at
> metrics/by_metric/steric_clash_pb_dock_1h36.json. This 830c metric remains
> DEFERRED because no docked poses + no receptor PDB on disk for 830c pocket."

### §1.3 1h36 pocket (NOT affected by this close-out)

`metrics/by_metric/steric_clash_pb_dock_1h36.json` carries its own
`MEASURED_SMOKE` tag and m8 pass rate = 1.0000 (15/15). This is independent
of the 830c deferral and remains a Tier-2-shippable measurement.

### §1.4 Wired-and-tested artefacts (SHIPPED, NOT removed)

- `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py` — `PoseBustersAdapter.dock(receptor_pdb, docked_pdbqt)`
  returns a `PBResult` with the clash check (SHIPPED).
- `molmetal/metrics/by_metric/pb_pass_rate.json` — PoseBusters 'mol' mode
  (chemistry-only, no protein) carries `pb_pass_rate=1.000` at the single-
  pocket / single-seed smoke level (pocket=test_000, seed=42).
- `metrics/by_metric/steric_clash_pb_dock_1h36.json` — MEASURED_SMOKE on 1h36
  pocket (15/15, m8 pass rate = 1.0000).
- Recompute path (when RDKit + posebusters available AND receptor PDB +
  docked poses on disk):
  ```python
  from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter
  pb = PoseBustersAdapter(mode='dock')
  res = pb.dock(receptor_pdb='molmetal/data/mmp13_real/830c.pdb', docked_pose_pdbqt=...)
  clash_pass_rate = sum(r.passed for r in results) / len(results)
  ```

## §2 What remains — Phase-3G 10×3 rotation (BLOCKED)

The Round-13 100×3 paper-grade sweep with `pb_mode=dock` is the
**forward-pointer** for this metric (per TODO-14 §3.2 "Steric clash
(PoseBusters) | 100 × 1 × 100 | 22/22 wire to per-pocket JSON (test_001
confirmed) | P1"):

- **R16-W42** — GPU recovery (cold PSU power cycle OR iGPU gfx1100 fallback).
- **R16-W43** — CFM path import fix + 10000-step retry.
- **R16-W44** — Lambda 100×3 sweep at n_sim=1000 with sub-fix A+C+Deflex
  flags + PB MMFF94 relax.
- **R16-W44-W45** — PB MMFF94 production pass rate (target: `pb_pass_rate ≥ 0.60`).

**Blocker chain** (per AGENTS.md §5.1):

1. **830c receptor PDB on disk**: missing from snapshot
   (`/mnt/disk1/Hugo_Lee/mol-metal/molmetal/data/mmp13_real/830c.pdb` not
   staged).
2. **830c docked poses on disk**: missing (no prior Vina run output on this
   snapshot).
3. **GPU HSA userland outage unresolved** (per `metrics/by_round/r13_100x3_partial.json`).
4. User-OK required before any 5–7.5 hr GPU launch.

## §3 Per-Tier-2 honest framing (AGENTS.md §5.4)

Per AGENTS.md §5.4 (Tier-2 journal target; pure computational submission):

- **steric_clash_pb_dock on 830c is NOT a Tier-2 acceptance gate.** Tier-2 is
  computational-only and accepts the PoseBusters 'mol' mode (chemistry-only)
  pass rate from `pb_pass_rate.json` + the 1h36 MEASURED_SMOKE as the per-pose
  metric citations. The 'dock'-mode clash check on 830c is a §6 honest-
  limitation or §7 future-work item, not a §4 blocker.
- **What DOES gate Tier-2 acceptance**: the metric is wired + adapter is
  SHIPPED + the 1h36 pocket carries MEASURED_SMOKE = 1.0000 (15/15) + the
  830c pocket honestly carries `status: DEFERRED` with `deferred_reason`
  citing the missing receptor PDB + docked poses.
- **What does NOT gate Tier-2 acceptance**: the 830c 'dock'-mode production
  pass rate (R16-W44-W45).

**Honest-negative framing for §9 catalog (this Wf-318G)**:

> DO NOT cite steric_clash_pb_dock on the 830c pocket as a production
> PoseBusters 'dock'-mode pass rate (deferred: no receptor PDB + no docked
> poses on disk; RDKit + posebusters were unavailable in this snapshot,
> verified NOW AVAILABLE per Wf-315D 2026-09-29 cross-reference, but the
> 830c receptor PDB + docked poses staging is NOT yet executed; cite the
> 1h36 MEASURED_SMOKE m8 pass rate = 1.0000 (15/15) at
> metrics/by_metric/steric_clash_pb_dock_1h36.json as the per-pose metric
> example; cite the PoseBusters 'mol' mode pb_pass_rate = 1.000 at
> metrics/by_metric/pb_pass_rate.json as the chemistry-only pass rate).

## §4 Recovery path — TODO-29 + GPU pipeline Phase 3

Per the GPU pipeline Phase 3 queue (TODO-26 R16 master plan):

| Recovery item | Owner | Dependency | Tier-2 gate? |
|---|---|---|---|
| Stage 830c receptor PDB | Round-16 prep | Network copy from source mirror | NO (computational) |
| Stage 830c docked poses | Round-16 W42 | Vina run on staged receptor | NO (computational) |
| Recompute steric_clash_pb_dock on 830c | Round-16 W45 | RDKit + posebusters env (verified available per Wf-315D) + staged 830c | NO (§7 future work) |
| Phase-3G 10×3 rotation | Round-16 W44 | F2(a) ship + GPU recovery + 830c staged | NO (§7 future work) |
| Cite 1h36 MEASURED_SMOKE | NOW | Independent verification | YES (already Tier-2-shippable) |

The 1h36 pocket is the Tier-2-shippable measurement as of 2026-09-29.

## §5 Cross-reference

- `metrics/by_metric/steric_clash_pb_dock.json` — source-of-truth JSON for
  830c (carries `status: DEFERRED` since 2026-09-18; this Wf adds
  HONEST_NEGATIVE tags).
- `metrics/by_metric/steric_clash_pb_dock_1h36.json` — 1h36 MEASURED_SMOKE
  (independent; NOT affected by this close-out).
- `metrics/by_metric/pb_pass_rate.json` — PoseBusters 'mol' mode pass rate
  (chemistry-only, 1.000 at single-pocket smoke).
- `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py` — `PoseBustersAdapter.dock()`
  (SHIPPED).
- `molmetal/reports/wf_pb_pass_real_dock/final.md` — single-pocket smoke
  integration (1.000 at test_000 / seed=42).
- `molmetal/reports/wf_pb_pass_10x3_smoke/final.md` — 30-cell smoke (search-
  bound, 0/30 PB-eligible docked candidates).
- `TODO/completed/14_full_100pocket_paper_r13.md` — TODO-14 (Round-13 master
  + per-pocket row gap §3.2 where steric_clash is P1).
- `TODO/completed/314a_wf311e_part4_deterministic_edits.md` §7 — Wf-311F §7
  candidate #4 ("Wf-311E diversity_subpocket + steric_clash + com_shift
  staging — DESIGN_ONLY / DEFERRED / NOT-staged").
- `AGENTS.md` §5.1 (kill_then_adapt) + §5.4 (Tier-2 journal target).
- `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline; data真实).

## §6 Honest framing checklist (party-twenty-spirit §3.5)

- [x] Every numeric value traces to a JSON path on disk (see §1.1, §1.3).
- [x] `NEGATIVE_RESULT_HONEST` flag set in JSON + dashboard + closeout doc
  for the 830c pocket specifically.
- [x] 1h36 MEASURED_SMOKE preserved (NOT flagged honest-negative).
- [x] No fabricated / aspirational numbers in main results (830c `rate = null`
  + 1h36 `rate = 1.0000` + 'mol' mode `rate = 1.000` are MEASURED on disk).
- [x] Recovery path documented but NOT executed in this Wf (CPU-only doc
  edit + JSON tag + paper_dashboard.py wiring).
- [x] Cross-refs to Wf-314F + Wf-311F + Wf-315D preserved verbatim.

## §7 Files touched (Wf-318G)

| Path | Type | Edit |
|---|---|---|
| `metrics/by_metric/steric_clash_pb_dock.json` | JSON | +4 fields (`HONEST_NEGATIVE=true`, `honest_negative_reason`, `honest_framing_note`, `verdict: FAIL`) |
| `molmetal/scripts/paper_dashboard.py` | Python | +1 HONEST_NEGATIVE_CATALOG entry (`hn_wf318g_steric_clash_pb_dock_830c_deferred`) |
| `TODO/completed/steric_clash_pb_dock_final_honest_negative.md` | Doc | NEW — this file |
| `TODO/completed/318g_r13_diversity_steric_final.md` | Doc | NEW — Wf-318G summary closeout |

---

*Generated 2026-09-29 by Wf-318G close-out. Workstream CLOSED as honest-negative
for Tier-2 scope (830c pocket specifically). 1h36 MEASURED_SMOKE preserved as
the Tier-2-shippable per-pose metric. Recovery path documented via GPU pipeline
Phase 3 (R16-W45) + 830c staging (Round-16 prep).*
