# WF-Honest-Negatives — R12-R17 Consolidation

**Date:** 2026-09-25
**Workflow ID:** wf_honest_negatives
**Scope:** Consolidate every MEASURED honest-negative result from Rounds
12-17 into a single canonical list. Close-out of TODO-28 (R12 reframe)
extended with the new R17 honest negatives from workflows A (TODO-33 P1+P2),
E (F1/F2/F4 attempted fix), F (Deflex follow-up), G (RCA), I (100×3 sweep
shim interaction), and B (L-R17-B2 SMARTS expansion).
**Status:** ✅ DONE — 5 honest negatives consolidated + 3 paper-section
mappings written + 30-line §6 Limitations paragraph appended.

---

## 1. TL;DR

| # | Honest negative | Workflow | MEASURED signal | Fix shipped? | Paper section |
|---|---|---|---|---|---|
| 1 | `decode_ratio=0` on Blackwell pre-fix | E + G | 0/192 across 3 seeds × 4 probes | YES (workflow A) | §3.5 + §6 NEW |
| 2 | shim TypeError blocks 100×3 sweep | I | 0/300 paper-grade cells on disk | YES (workflow A) | §3.5 + §6 NEW |
| 3 | 84-rule matrix alone does not lift diversity | F | `n_distinct` lifts 1→20 within-cell but `div_tan=0.137` basket-internal, `metal_pairs=2` FAILS TODO-25 §3 gate | PARTIAL (F2(a) 1→7 rules; 84-rule matrix QUEUED) | §4.5 + §6 NEW |
| 4 | F5 linear+hinge vs Deflexpressor ASTSR | L-R17-B3 | F5 in-sample R²=1.0, LOO R²=-0.0635; ASTSR R²=1.0000 on 30-cell hold-out | YES (ASTSR ships default) | §6 + F5 paragraph |
| 5 | 1K-step smoke `decode_ratio=1.000` may be lucky | R16/R17 probe budgets | `mean_atoms=8.0` every step (unconditional collapse) | PARTIAL (3-seed retry GPU-blocked) | §6 NEW |

---

## 2. Detailed negatives

### Negative #1 — decode_ratio=0 on Blackwell pre-fix (workflow E)

- **Source reports:** `molmetal/reports/wf_todo33_p1_done.md` (workflow E
  3-seed verify) + `molmetal/reports/wf_rca_decode0.md` (workflow G RCA).
- **MEASURED:** `decode_ratio = 0.000` at every probe step (250/500/1000/2000)
  on Blackwell (RT PRO 6000). F1 (FP32), F2 (LR warmup 200), and F4
  (logits clamp ±50 + 20-atom vocab) each individually lift `atom_loss`
  from `Inf` to finite bounded values, but `decode_ratio` stays 0.
- **Root cause (orthogonal to F1/F2/F4):** the ODE-solver shim
  (`molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:152`)
  hands the upstream `FlowMatchingSampler` a `ModelWrapper`, which the
  upstream's `isinstance(model, VelocityNet)` check rejects at
  `.sample()`-time, not at `__init__`-time. The shim's
  `_UPSTREAM_AVAILABLE` probe uses a `_ProbeModel` that constructs OK,
  so the probe falsely reports the upstream is usable.
- **Fix (workflow A, `wf_todo33_p2_shim_fix/final.md`):** a 1-line
  defensive `try/except TypeError` around `_sampler.sample()` plus 5
  companion shim edits:
  1. `_reference_loader.py` flat-clone auto-detect.
  2. `_reference/__init__.py` hybrid `__path__` (local + upstream clone).
  3. `_reference/path/__init__.py` qualified import for `interpolation`.
  4. `_reference/path/scheduler/__init__.py` qualified import for `SamplerConfig`.
  5. `_reference/utils/__init__.py` ModelWrapper keyword-time fallback
     (`model(x,t)` → `model(x,t=t)` → `model(x,t=t,**extras)` →
     `model(x,t,**extras)`).
- **Verified result:** 3 seeds × 4 probes × 16 samples = 192/192 decode
  at `decode_ratio = 1.000 ± 0.000`, `disconnected = 0/48`.
- **Honest framing:** the symptom looked like an architecture /
  loss-curve problem for 4 days. The actual cause was a sample-time
  wrapper-typecheck race that no amount of training-time engineering
  could fix. Engineering lesson: always defensive-try/except around
  upstream `isinstance` checks whose enforcement is deferred to
  `.sample()`.

### Negative #2 — shim TypeError blocks the 100×3 paper-grade sweep (workflow I)

- **Source reports:** `molmetal/reports/wf_round13_100x3/final.md` +
  `molmetal/reports/wf_round13_100x3/aggregate.json`.
- **MEASURED:** the Round-13 100-pocket × 3-seed paper-grade sweep
  (`paper-grade-300cells`) returns **0 final aggregate cells on disk**.
  Path A (Lambda) killed (CPU contention from parallel sweeps); Path B
  (PoseBusters) 30/30 search-bound (`pb_pass_rate=null` by
  construction); Path C (CFM) BLOCKED (`r10_cfg_real_crossdocked.py`
  import error + dGPU HSA userland outage).
- **Root cause:** Negative #1's shim TypeError poisoned every probe in
  the sweep; compounded with the dGPU firmware SMU hang and the import
  error, every path needed for a 300-cell paper-grade aggregate failed.
- **Honest framing:** the 0/300 cell count is **not** an algorithm
  failure — every individual subfix (F1, F2, F4, shim) was verified on
  a 30-cell panel. The 100×3 sweep's failure mode is environmental
  (dGPU outage + shim race + CPU contention), and must be reported as
  such rather than as an algorithm regression.

### Negative #3 — 84-rule matrix alone does not lift diversity (workflow F)

- **Source reports:** `molmetal/reports/wf_deflex_followup/phase1_f5_adapter.md`
  + `molmetal/reports/wf_lr17_b2_done.md` §3.
- **MEASURED:** extending F2(a) `MetalLigandExchange` SMARTS from
  1 → 7 rules and wiring `auto-pt-strict` to use the 3-rule CuAAC +
  SPAAC + Suzuki subset **lifts `n_distinct` 1 → 20 within-cell**
  (singleton barrier broken on all 30 cells of the LR17-B2 pilot).
  However:
  - The *same* 20 SMILES appear in every one of the 10 pockets
    (`overlap test_000 vs test_009 = 20/20`).
  - `div_tan` lifts to 0.137 (basket-internal diversity).
  - `distinct_metal_ligand_pairs` stays at **2** — **fails** the
    TODO-25 §3 ≥3-pair gate.
- **Root cause (3 layered):**
  1. MCTS shares the same `[Pt]C#C` root across all pockets because
     `--metal-seed cisplatin` overrides `--metal-seed-from-pocket`
     (Phase-2C helper ships but is not enabled).
  2. `auto-pt-strict` keeps only 3 click rules and gates out the
     F2(a) `MetalLigandExchange` / `AquaExchange` SMARTS that would
     diversify the metal-coordination sphere.
  3. MLE / AE rules emit **single** product configurations
     (collapsed `[Cl][Pt][Cl]`-class), so the chemistry-layer
     branching is narrower than the rule count suggests.
- **Honest framing:** rule count alone does **not** lift diversity. The
  7-rule F2(a) subset is necessary but not sufficient; the 84-rule
  matrix (12 metals × 7 ligands) + 4 NEW SMARTS (transmetalation,
  Cl-OH swap, ligand-shell-MMA, Pt/Pd transmetalation) is QUEUED in
  L-R17-B2 and must be merged before the diversity gap to TargetDiff
  can be honestly closed. The diversity gap of `-0.7535` (Mol-Metal
  0.1065 vs TargetDiff 0.860) is **not closed** at this round.

### Negative #4 — F5 linear+hinge vs Deflexpressor ASTSR (workflow L-R17-B3)

- **Source reports:** `molmetal/reports/wf_deflex_followup/phase1_f5_adapter.md`
  + `molmetal/reports/wf_deflex_astsr/final.md` (L-R17-B3 SHIPPED).
- **MEASURED:**
  - F5 closed form `R_F5(sa_norm) = 2.5836 - 2.5149 * sa_norm`
    achieves in-sample `R² = 1.0` but leave-one-out
    `R² = -0.0635` (close to random).
  - ASTSR (LamVar / LamAbs / LamApp hypothesis class, Schmidt-Lipson
    2009 evolutionary search, depth=4, pop=64, gen=50) achieves
    `R² = 1.0000` on the same 30-cell hold-out, recovering the
    reward target `add(add(metal_compliance_rate, sa_mean_norm),
    diversity_tanimoto)` exactly.
- **Honest framing:**
  - F5 is **interpretive** (literal regression on the Round-12
    cohort, not validated to lift downstream metrics).
  - ASTSR is **interpretive + structurally richer** (full λ-calculus
    hypothesis class), but its **superiority over F5 on a downstream
    oracle (Vina, PB pass-rate, pIC50) is unproven** at this round.
  - The ASTSR's wall-time cost (0.41 s on a 33-cell fit) is
    negligible compared to the rest of the coupling pipeline.
  - Both F5 and ASTSR are honest priors at this round; neither is
    yet a validated downstream-lift reward prior.

### Negative #5 — 1K-step smoke decode_ratio=1.000 may be lucky for very small batches

- **Source reports:** `molmetal/reports/wf_todo33_p2_shim_fix/final.md`
  (TODO-33 P2 verify) + `molmetal/reports/wf_r16_yuelbond_2000_probe/final.md`
  (R16 probe) + `molmetal/reports/wf_r17_10k_retrain_done/final.md`
  (R17 retrain).
- **MEASURED:** the 1K-step / 2000-step / 10000-step smokes on
  `n_train=32-500`, `batch=4`, `h=128`, `l=3` all return
  `decode_ratio = 1.000` at every probe — but
  `mean_atoms = 8.0` *every step*, indicating **unconditional-mode
  collapse** (the decoder reproduces the training-set mean atom count
  and never pocket-conditions the output).
- **Three honest caveats** from `wf_r17_10k_retrain_done/final.md`:
  1. **single-seed partial** — only seed=42 started; cross-seed
     stability not measured; cannot rule out that
     `decode_ratio=1.0` at step 2000 is a single-seed artefact.
  2. **`mean_atoms = 8.0` every step is unconditional mode**
     (`pocket=None`); pocket-conditioned run is the lever to break
     the 8.0 ceiling.
  3. **GPU firmware SMU hang** — cannot retry until cold power cycle;
     3-seed stability retry is deferred to R17+.
- **Honest framing:** `decode_ratio = 1.000` at the probe budget is a
  necessary-but-not-sufficient gate — it confirms the ODE path runs
  and the wrapper race is fixed, but it does NOT confirm the model has
  learned pocket-conditioned placement. Reviewers should be told that
  explicitly. The 1.000 lift off the 0.000 floor is real, but it may
  be partially attributable to the small training set (n=32-500)
  collapsing to a single chemotype.

---

## 3. Paper-section mapping

| # | Honest negative | Paper section | Status |
|---|---|---|---|
| 1 | `decode_ratio=0` on Blackwell pre-fix | §3.5 Implementation Details + §6 NEW | ✅ DONE |
| 2 | shim TypeError blocks 100×3 sweep | §3.5 + §6 NEW | ✅ DONE |
| 3 | 84-rule matrix alone does not lift diversity | §4.5 Honest Negative Framing + §6 NEW | ✅ DONE |
| 4 | F5 linear+hinge vs Deflexpressor ASTSR | §6 F5-vs-ASTR paragraph (extended) | ✅ DONE |
| 5 | 1K-step smoke `decode_ratio=1.000` unconditional collapse | §6 NEW | ✅ DONE |

---

## 4. Files shipped

| File | Status | LOC |
|---|---|---|
| `TODO/completed/38_round12_honest_negatives.md` | NEW | ~150 |
| `paper/sections/06_limitations.tex` | APPEND +30 lines + F5 paragraph extension | +30 + ~10 |
| `molmetal/reports/wf_honest_negatives/final.md` | NEW (this file) | ~250 |

---

## 5. Recommended follow-ups

- Add the §6 paragraph to the next pdflatex compile cycle (4-pass,
  0 unresolved refs).
- Cross-reference `wf_todo33_p2_shim_fix/final.md` from §3.5 Deflex
  (currently §3.5 mentions only
  `wf_r15_round_re_runs/r12_deflex_allon_10x3`).
- Add a §4.6 ablation row comparing F5 vs ASTSR on the 30-cell hold-out.
- Promote `TODO-38` to the INDEX master close-out entry next to
  `TODO-28`.

---

End of consolidation — 5 negatives, 3 paper sections, 30-line §6 paragraph.