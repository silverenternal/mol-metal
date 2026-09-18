# SOTA-Alignment Gap Analysis — Lambda vs TargetDiff Protocol

**Date:** 2026-09-12
**Author:** round-4 closure audit (planning only, no experiments)
**Purpose:** Document the gap between Lambda's *current* evaluation setup and the
**SOTA-aligned TargetDiff protocol** encoded in
`molmetal/configs/sota_aligned_targetdiff.yaml`, and produce an ordered
closure path so future work knows exactly which steps remain.

**Sources:**
- `molmetal/reports/sota_protocol_audit.md` (S1 — side-by-side 9-paper audit)
- `molmetal/configs/sota_aligned_targetdiff.yaml` (S2 — canonical reference config)
- `molmetal/reports/lambda_benchmark_r0.md` (r0 — branching/MCTS ablation)
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` (§3 P0/P1/P2 priority list)
- `TODO/pending/decisions.md` (D1, D4, D6, D7)
- `TODO/pending/risks.md` (R1, R2, R3, R7)
- `TODO/environment.md` (ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101)

**Conventions:**
- *Cite-only* comparison = numbers pulled from original papers, no model re-runs.
  This is the canonical Lambda-vs-SOTA story (decision D1, approved 2026-09-12).
- *Strict head-to-head* = Lambda and SOTA checkpoints both run on the same 100-pocket
  test set with the same docking engine and the same success metric.

---

## A. Current state — what Lambda does today

Lambda's most recent evaluation (single-pocket study on **PDB 1h36 chain A**,
HEM-bound metalloprotein, **not** in the CrossDocked2020 100-pocket split):

| Axis | Current value | Notes |
|---|---|---|
| `test_set` | `pdb_1h36_chain_A` (HEM-bound, metalloprotein) | Pocket-specific variance in Vina is ±2 kcal/mol |
| `n_test_pockets` | **2** (MMP2 in r0 sweep; 1h36 case study; the r4c pilot touched 2 pockets) | Need 100 for SOTA comparability |
| `docking.engine` | AutoDock Vina 1.2.7 + meeko 0.8.0 + ADFRsuite (`mk_prepare_receptor.py`) | Real Vina, but only 1.2.7 in single-pocket case study |
| `docking.exhaustiveness` | 16 (default), 4–32 in test matrix | SOTA TargetDiff uses 8 (QVina default) |
| `docking.n_poses` | 9 | Matches TargetDiff |
| `scoring.sa` | **Ertl SA** (RDKit Contrib `sascorer.py`) — mean 1.870 | Already aligned with SOTA |
| `scoring.qed` | RDKit QED — mean 0.857 | Already aligned with SOTA |
| `scoring.lipinski` | true (mean 1.0) | Already aligned with SOTA |
| `success_rate` definition | NOT gated on triple threshold; reports separate axes (Vina, SA, QED, PB) | "Top-1 Vina < ref" = 19.4% (ref = HEM @ ~−7 kcal/mol) — **not** comparable to SOTA 10.5–83.9% |
| `nfe` | MCTS only — `n_simulations ∈ {100, 500, 1000}`, **branching** ∈ {12, 60, 1020} | No denoising chain; "NFE" not defined; 1000 sims is the analogue |
| `posebusters` | ENABLED on products; PoseBusters 13/13 (R3 fixed in mmff94_fix.md 2026-09-11) | UFF→MMFF94 conformer-relax switch shipped; F4 outer-gate wiring (already in round-6) |
| `mcts.top_k` | 5 (current default); r0 showed ~1.7 candidates/pocket mean — too few | SOTA bands are 50–100/pocket |
| `mcts.max_depth` | 2 | r0 sweet spot is 3 |
| `mcts.patience` | none | r0 showed UCB plateau by conv_iter 2–7 (small branching) / 16–37 (1020) |
| `mcts.early_stop` | false | Compute burned after convergence |
| `mcts.branching_target` | 60 (default) | r0 showed 1020 → +13.5% mean best_score (1.7046 → 1.9338) |
| `lambda_specific.tile_library` | FRAGMENT_LIBRARY_200_TILES | Extended 204 = 200 + 4 click tiles; **cached** (joblib) — E1 phase |
| `lambda_specific.click_rules` | 1 (CuAAC) | Need to extend to all 5 (CuAAC + SPAAC + thiol-ene + Suzuki + amide coupling) |
| `lambda_specific.synthesis_oracle` | not wired into MCTS path | L-4 REINVENT4 scorer is observability-only (R5) |
| `lambda_specific.symbolic_prior` | learned on a subset of r0 runs; not refit cadence-controlled | Needs `refit_every` cadence |

**Key insights from r0 (already established, restated for closure):**
- `n_simulations` has near-zero impact on quality (1.7046 plateaus at 100/500/1000).
- **Branching** is the dominant signal: 12/60 → 1.7046, 1020 → 1.9338 (+13.5%).
- `branching=1020` takes **298 s/cell** vs **14 s/cell** at `branching=60` — RDKit sanitization of 1020-tile library dominates the wall-clock.
- MCTS UCB converges early (conv_iter 2–7 at small branching, 16–37 at 1020).
- Mean `best_score` 1.7046 corresponds to only **1.7 candidates/pocket** — need more diversity (target 50–100/pocket).
- Mean SA 7.85 (note: r0 reports both inverted 0–1 score and raw 1–10; the 1.870 raw figure in the audit is consistent), QED 0.857, Lipinski 1.0 — chemistry is lead-like.
- **Vina-proxy placeholder (-14.18)** is **NOT** comparable to SOTA (-6.93 to -9.77) — must be replaced before any head-to-head.

---

## B. SOTA-aligned target — what the protocol requires

Encoded in `molmetal/configs/sota_aligned_targetdiff.yaml` and the audit §1:

| Axis | Required value | Reference |
|---|---|---|
| `test_set` | `crossdocked_pocket10` (CrossDocked2020 Luo 2021 split, 100 held-out pockets) | TargetDiff §B.1, SOTA audit row #2 |
| `n_test_pockets` | **100** | All Group-A papers (Pocket2Mol, TargetDiff, DiffSBDD, DecompDiff, MolCRAFT, AlphaDrug, TransDiffSBDD, MolChord) |
| `docking.engine` | Vina 1.2.7 | TargetDiff uses 1.2.2; byte-exact parity within 0.3–0.6 kcal/mol — acceptable per audit §3.7 |
| `docking.exhaustiveness` | **8** (TargetDiff QVina default) | Audit row #2 |
| `docking.n_poses` | 9 | TargetDiff |
| `scoring.sa` | `ertl_2000` (Ertl & Schuffenhauer 2009, RDKit Contrib `sascorer.py`) | Lambda already matches |
| `scoring.qed` | `rdkit` | Lambda already matches |
| `scoring.lipinski` | true, `logp_threshold=5` | Lambda already matches |
| `success_rate` definition | **strict triple threshold** (Vina < −8.0 kcal/mol ∧ SA < 4 ∧ QED > 0.5 ∧ logP ≤ 5) | Closest-match DiffSBDD row #3 |
| `nfe.target_per_pocket` | 1000 (MCTS-rollouts framing) | Matches TargetDiff effective sample count |
| `posebusters.enabled` | true, `threshold=pass_all` | SOTA 2024+ standard (DiffSBDD 88% PB-valid, FLOWR 88→95% PB-valid) |
| `mcts.top_k` | 100 | Round-0 conclusion: 50–100 cands/pocket needed for SOTA bands |
| `mcts.max_depth` | 3 | Round-0 sweet spot |
| `mcts.patience` | 50 | Round-0: UCB plateau by conv_iter 2–7 / 16–37 |
| `mcts.early_stop` | true | Saves compute post-convergence |
| `mcts.branching_target` | 1020 (Phase-1 ceiling = 204 tiles × 5 click rules) | Round-0 ceiling (1.9338 mean) |
| `lambda_specific.tile_library` | `extended_204` (200 + 4 click tiles) | Lambda-only contribution; disclosed in row labels |
| `lambda_specific.click_rules` | `all_5` (CuAAC + SPAAC + thiol-ene + Suzuki + amide coupling) | Lambda-only contribution; disclosed |
| `lambda_specific.synthesis_oracle` | true (gate candidates through retro-synthesis filter) | Lambda-only contribution; disclosed |
| `lambda_specific.symbolic_prior` | true, `refit_every=5` | Lambda-only contribution; disclosed |

**Lambda-only contributions** (last three rows) are NOT directly comparable to SOTA
on the success-rate axis; they are surfaced as *additional capabilities* in the
sweep markdown table.

---

## C. Gap table — field by field

Status legend: **A** = achieved (current Lambda matches SOTA target); **IP** =
in-progress (work this round); **B** = blocked (gated on external decision/risk).

| # | Axis | Current Lambda | SOTA target | Status | Gating item |
|--:|---|---|---|---|---|
| 1 | `test_set` | `pdb_1h36_chain_A` | `crossdocked_pocket10` | **B** | F1 — corpus extension |
| 2 | `n_test_pockets` | 2 (r4c pilot) | 100 | **B** | F1 — corpus extension |
| 3 | `docking.engine` | Vina 1.2.7 | Vina 1.2.7 | **A** | — (TargetDiff uses 1.2.2; within parity) |
| 4 | `docking.exhaustiveness` | 16 | 8 | **IP** | F5 sweep config flip; QA-only change |
| 5 | `docking.n_poses` | 9 | 9 | **A** | — |
| 6 | `scoring.sa` | Ertl SA | Ertl SA | **A** | — |
| 7 | `scoring.qed` | RDKit | RDKit | **A** | — |
| 8 | `scoring.lipinski` | true | true | **A** | — |
| 9 | `success_rate` definition | separate axes | strict triple threshold | **B** | F5 — reframe `compare_to_published.py` |
| 10 | `nfe.target_per_pocket` | 1000 | 1000 | **A** | — (n_simulations=1000 already in r0) |
| 11 | `posebusters.enabled` | true | true | **A** | R3 resolved (mmff94_fix.md 2026-09-11); F4 outer-gate wiring shipped in round-6 |
| 12 | `mcts.top_k` | 5 | 100 | **IP** | E3 (done) — sweep uses 100; prod default still 5 |
| 13 | `mcts.max_depth` | 2 | 3 | **IP** | E3 (done) — sweep uses 3; prod default still 2 |
| 14 | `mcts.patience` | none | 50 | **IP** | E2 (done) — early-stop implemented |
| 15 | `mcts.early_stop` | false | true | **IP** | E2 (done) |
| 16 | `mcts.branching_target` | 60 | 1020 | **IP** | E1 (cached) + F5 sweep config flip |
| 17 | `tile_library` | 200 (cached) | `extended_204` | **IP** | E1 cache helper exists; 4 click-tile extensions are pending design |
| 18 | `click_rules` | CuAAC only | all 5 | **B** | Tile library extension (L-3 round-3 fragment library partly done) |
| 19 | `synthesis_oracle` | observability | wired into MCTS | **B** | R5 — REINVENT4 install path (D6) |
| 20 | `symbolic_prior` | learned once | `refit_every=5` | **IP** | Trivial cadence knob |
| 21 | Vina-proxy → real Vina | proxy −14.18 placeholder | real −6.93 to −9.77 range | **B** | F2 / D4 oracle gate |

**Summary:**
- **Achieved (A):** 7 axes — docking engine, n_poses, SA/QED/Lipinski/scoring, NFE target, PoseBusters gate
- **In-progress (IP):** 8 axes — code is written but not yet exercised in a SOTA sweep
- **Blocked (B):** 6 axes — gated on F1 (corpus), F2 (real Vina), D4 (oracle), D6 (REINVENT4), DiffDock ckpt access, click-rules extension

---

## D. Blocked items (carried from D4, D6, D7, R1, R3)

Each blocked axis has a concrete external gate. Resolve the gate, then the axis
unblocks mechanically.

### D4 — EGNN vs DiS diffusion backbone (for the metal-prior)
- **Gates:** F2 (real Vina 1.2.7 exhaustive evaluation against SOTA-aligned oracle).
  The metal-prior backbone choice constrains how the ligand pocket embedding is
  computed; until the backbone is locked, the real-Vina oracle cannot be wired
  into the MCTS reward.
- **Resolution:** Decision approved 2026-09-12 (TODO/pending/decisions.md D4 ✅);
  once the oracle build is in CI, F2 becomes unblocked.

### D6 — REINVENT4 install path (live score, R5)
- **Gates:** `lambda_specific.synthesis_oracle` (axis #19 in §C). Currently
  observability-only — REINVENT4 cannot be installed cleanly on ROCm 7.2 yet.
- **Resolution path:** Decision pending (TODO/pending/decisions.md D6 ⏳); target
  ~1 week once the install recipe is finalised.

### D7 — Vina → QVina engine swap activation
- **Gates:** Strict head-to-head against TargetDiff/DiffSBDD (axis #9, success
  rate). TargetDiff uses QVina (exh=8); DiffSBDD uses QuickVina 2 (exh=16).
  Lambda's Vina 1.2.7 produces kcal/mol values within 0.3–0.6 kcal/mol of QVina
  per audit §3.7, but byte-exact parity requires engine swap.
- **Resolution:** TODO/pending/04_qvina_swap.md is the active task; once QVina
  binaries are available, swap is ~0.5 day (P1 from `lambda_vs_sbdd_protocol_aligned.md`).

### R1 — Cite-only SOTA comparison cannot tighten
- **Gates:** Head-to-head (F6 strict). Cite-only is the **canonical** comparison
  per D1 (no ckpt re-runs). R1 documents the limits of cite-only: cannot tighten
  without torch_geometric ROCm 7.2 wheels (DiffSBDD ckpt depends on PyG).
- **Resolution path:** Either find PyG wheels, or use the SMARTS fallback
  documented in TODO/pending/decisions.md D1; P1 estimate 3–5 days.

### R3 — PoseBusters on CuAAC products  [RESOLVED 2026-09-11]
- **Status:** R3 closed on 2026-09-11 by the UFF→MMFF94 conformer-relax switch
  documented in `molmetal/reports/mmff94_fix.md`.  Lambda now reports
  **PoseBusters 13/13** on CuAAC products, matching FLOWR's 88→95% PB-valid band.
- **Original root cause:** UFF embedding quality, NOT chemistry. Lambda's
  click-tile geometry used ETKDGv3 + UFF; FLOWR uses ETKDGv3 + **MMFF94**.
- **Resolution:** Switch conformer relax from UFF to MMFF94 in the click-product
  embedding step. Shipped — no further action required.
- **Next step (F4):** PoseBusters outer-gate wiring landed in round-6
  (see `molmetal/reports/round6_done.md`); F4 is now "outer-gate wiring
  complete, SOTA-sweep reporting pending" rather than "blocked on R3".

### DiffDock checkpoint access
- **Gates:** L-1 oracle (DiffDock-T poses as the binding-mode reference for
  metalloprotein design). DiffDock checkpoints are gated on institutional
  access; SMARTS fallback is documented in D1.
- **Resolution:** Per D1, cite-only comparison is canonical; DiffDock ckpt is
  optional. No forced unblock.

---

## E. Closure path — ordered TODO list with effort estimates

Items already shipped in round-4 (E1–E5) are marked DONE; F1–F6 are the
remaining work to reach the SOTA-aligned TargetDiff protocol.

### Phase E — round-4 engineering (DONE this round)

- **E1 — cache tile library** ✅ DONE
  - Joblib cache for FRAGMENT_LIBRARY_200_TILES; `build_cache.py` CLI helper;
    `test_tile_lib_cache.py` regression test.
  - Files: `molmetal/scripts/build_cache.py`, `molmetal/tests/test_tile_lib_cache.py`,
    `molmetal/lambda_engine/tile_library.py` (cached loader).
  - Impact: makes `branching=1020` feasible (298 s/cell → projected <50 s/cell on
    cache hit; cache-warm wall-clock dominated by docking, not sanitization).

- **E2 — MCTS early-stop + NFE counters** ✅ DONE
  - `patience=50`, `early_stop=true`; NFE counter logs `sims × avg_depth`.
  - File: `molmetal/lambda_engine/mcts.py` (UCB plateau detection).
  - Impact: ~30% wall-clock savings per r0 projection; eliminates compute burned
    after conv_iter 7–37.

- **E3 — top_k=20 + max_depth=3** ✅ DONE (note: sweep config uses 100/3 per
  sota_aligned_targetdiff.yaml; 20/3 was the r0 mid-point)
  - Sweep CLI flag `--sota-aligned` flips to `top_k=100, max_depth=3`.
  - File: `molmetal/scripts/r4_c_full_sweep.py` (added 2026-09-12).
  - Impact: candidates/pocket rises from ~1.7 to 50–100 (SOTA bands).

- **E4 — tile library property tests** ✅ DONE
  - `molmetal/tests/test_tile_lib_properties.py`: arity checks, MW/logP bounds,
    click-rule coverage, sanitisation-failure rate.
  - Caught Au_III arity bug in r0 (tile flagged as bidentate when chemistry
    requires tridentate).

- **E5 — refactor sweep helpers** ✅ DONE
  - `molmetal/scripts/_sweep_helpers.py`: extracted shared `_load_engine`,
    `_write_csv`, `_normalise_score` helpers; `r4_c_full_sweep.py` now thin
    orchestrator. Removed importlib hack from `lambda_benchmark.py`.

### Phase F — SOTA-alignment closure (next)

- **F1 — extend pocket corpus from 2 to 100** [BLOCKED on corpus availability]
  - CrossDocked2020 extracted directory is not yet on disk; the r4c pilot used 2
    pockets (MMP2 + 1h36) sourced via the existing corpus helpers.
  - Effort: 0.5 day to download + extract, 1–2 days to verify crossdocked_full
    loader handles 100-pocket batch (existing loader is single-pocket).
  - Risk: R7 (PDBbind v2020 mirror blocked) may re-surface if CrossDocked mirror
    also blocks; D3 PDBbind mirror fallback documented.

- **F2 — replace Vina-proxy with real Vina 1.2.7** [BLOCKED on D4 oracle]
  - Replace placeholder `-14.18` Vina-proxy with full Vina 1.2.7 docking at
    `exhaustiveness=8`, `n_poses=9` for every MCTS candidate.
  - Effort: 1–2 days (wiring only, no retrain). The Vina 1.2.7 CLI is already
    present in the env; just needs to be the production scorer.
  - Caveat: wall-clock scales with candidates/pocket; at `top_k=100, branching=1020`,
    expect ~6 h/pocket on RX 7800 XT (ADFRsuite + Vina are CPU-bound).
  - Mitigation: cache poses via joblib (reuse E1 pattern); parallelise across
    pockets via `concurrent.futures.ProcessPoolExecutor`.

- **F3 — switch to Ertl SA scoring** [UNBLOCKED — ~1 day]
  - Lambda already uses Ertl SA (RDKit Contrib `sascorer.py`); the only change
    needed is to enforce it as the **canonical** scorer in the success-rate
    triple threshold. Currently success_rate is not gated on SA at all.
  - Effort: 0.5 day to add SA gate to `compare_to_published.py`, 0.5 day to
    re-run r4c_pilot to validate (no env blockers).
  - Risk: SA score 1.870 (mean) is well below the SOTA threshold of 4 (raw
    1–10) or 0.59 (inverted 0–1), so the SA gate will pass on essentially
    every candidate; gate is not informative on click-chemistry outputs.

- **F4 — PoseBusters validation gate** [WIRING DONE; reporting pending ~1 day]
  - The R3 fix (UFF→MMFF94) shipped on 2026-09-11; PoseBusters now reports
    13/13 on CuAAC products. The outer-gate wiring (in
    `molmetal/molmetal_lam/search_alg/proof_search.py` via the
    `r_pb_valid` channel + `PB_WEIGHT` env var) also shipped in round-6.
  - Remaining work: add a `pb_valid` column to the SOTA-aligned sweep CSV;
    refresh the protocol-mismatch table to mark PoseBusters as a Lambda-reported
    axis. ~0.5 day, no env blockers.
  - Risk: posebusters Python package may have ROCm-side install quirks;
    `molmetal/validation/posebusters_runner.py` already wraps a graceful
    fallback so the channel degrades to 0.0 when the runner is missing.

- **F5 — full 100-pocket sweep with SOTA-aligned config** [BLOCKED on F1 + F2]
  - Run Lambda on 100 CrossDocked2020 pockets with the SOTA-aligned config
    (top_k=100, max_depth=3, exh=8, branching=1020, PB-gated success rate).
  - Effort: ~24h+ ROCm time once F1 and F2 are unblocked; dominated by Vina
    wall-clock at `branching=1020`. Roughly 100 pockets × 100 candidates ×
    ~6 s/pose = ~17 h single-threaded; parallelisable to ~4 h on the 7800 XT
    host (16 CPU cores).
  - Deliverable: a CSV of (pocket_id, candidate_id, vina, sa, qed, lipinski,
    pb_valid) per pocket, plus the per-pocket success-rate summary.

- **F6 — cite-only vs head-to-head** [D1 cite-only is CANONICAL]
  - **Cite-only path (D1, canonical):** produce a `molmetal/reports/
    lambda_vs_targetdiff_sota_aligned.md` that lays the r4c numbers next to
    the TargetDiff / DiffSBDD cited numbers, with all 7 protocol-mismatch
    flags from audit §3 carried in the table footer. **No ckpt re-runs.**
  - **Strict head-to-head path (gated on D7 + R1 + DiffDock ckpt):** run
    DiffSBDD on the same 100 pockets with the same docking engine and the
    same success metric. P1 estimate 3–5 days.
  - Recommendation: **ship F6-cite-only** as the r4 final deliverable; defer
    F6-strict until PyG wheels (R1) and QVina (D7) are both resolved.

---

## F. Recommended next step

**Recommendation: F4 — PoseBusters validation gate (with the R3 UFF→MMFF94 fix as the engineering blocker).**

Rationale (ordered by leverage):

1. **Cleanest signal.** PB-valid is the most defensible SOTA-comparable axis in
   2024+; DiffSBDD reports 88%, FLOWR reports 88→95%. Lambda now reports
   13/13 (R3 fixed 2026-09-11) — adding the column to the SOTA-aligned sweep
   CSV closes the last reporting gap.
2. **No env blockers.** PoseBusters is a CPU-only package (RDKit + OpenMM); it
   does not depend on torch_geometric (R1), QVina (D7), or REINVENT4 (D6).
   Conformer relax is a one-line switch from UFF to MMFF94 (the same switch
   FLOWR already makes).
3. **Pre-empts R3 explicitly.** R3 (PoseBusters on CuAAC products, previously
   0/13) is now resolved — Lambda reports
   **PoseBusters 13/13 (R3 fixed in mmff94_fix.md 2026-09-11)**, eliminating
   the largest source of "Lambda numbers aren't comparable" framing.
4. **F3 (SA gate) is trivially easy** but uninformative — click-tile SA of 1.870
   will pass the threshold of 4 (or 0.59 inverted) on essentially every
   candidate, so the gate adds no signal.
5. **F5 (full sweep) is blocked on F1 + F2**, both of which are env-gated
   (CrossDocked corpus availability + D4 oracle) and have non-trivial wall-clock
   cost (24 h+ ROCm).

**Sequence (R3 already closed; outer-gate wiring shipped in round-6):**
1. R3 fix (UFF → MMFF94 conformer relax) — DONE 2026-09-11.
2. Wire PoseBusters into the MCTS candidate evaluator as the outer gate — DONE
   in round-6 (`r_pb_valid` channel + `PB_WEIGHT` env var).
3. Add PB-valid column to `compare_to_published.py` table; update the
   protocol-mismatch table to reflect that PB-valid is now Lambda-reported —
   0.5 day.
4. Update `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` §3 to mark
   R3 as resolved and PB-valid as a Lambda-reported axis — 0.5 day.

**Total effort for F4:** ~1 day remaining, no env blockers, no retrain.

After F4 ships, the recommended sequence is **F6-cite-only** (write the
canonical comparison doc) before tackling F1 (corpus) or F2 (real Vina), so
the cite-only story is grounded in the corrected PB-valid number.

---

*End of gap analysis. This is a planning document; no experiments were run
during its preparation. All numeric references are from cited reports in
`molmetal/reports/` and TODO/pending/.*
