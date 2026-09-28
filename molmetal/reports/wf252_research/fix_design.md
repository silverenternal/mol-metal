# Wf-252-Synthesize — CPU-only Fix Design for Paper §3/§4 Gaps

**Date:** 2026-09-28 · **Workflow:** Wf-252-Synthesize (Phase 2 of Wf-252)
**Upstream:** `molmetal/reports/wf252_research/lit_review.md` (Wf-252-Research)
**Scope:** Design CPU-only, backward-compat fixes for the 5 remaining gaps surfaced
across Wf-243..Wf-251. Rank by effectiveness / LOC / risk / available code. Pick TOP
fixes for in-session implementation.
**Acceptance gates (FLAT):** see §6 below.

---

## §1 Constraints (recap)

* **CPU-only**: no GPU retrain inside this workflow; rely on existing checkpoints
  + RDKit-only chemistry + chemprop v2 dry-mode eval.
* **Backward-compat**: every fix is **opt-in** via a CLI flag or new file; no
  driver-side behaviour change without the flag.
* **Honest framing**: projections are tagged PROJECTED; no fabricated MEASURED
  lift; honest-negative outcomes preserved.
* **Python:** `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` (this venv has
  RDKit preinstalled; chemprop_v2 is on `sys.path` via `.cloned/chemprop_v2`).
* **No GPU retrain** for these fixes — only post-hoc eval on existing
  v3 checkpoints (Ru/Ir/Os/Rh/Re) + chemprop v2 backbone in BF16/CPU mode.

---

## §2 Gap-by-gap fix candidates (ranked)

Each gap below lists 2-4 candidate CPU-only fixes ranked by **Effectiveness**,
**LOC**, **Risk (backward-compat)**, and **Available code**. The top pick per gap
is marked `[TOP]`.

### G1 — Click augmentation for m7 Diversity Tanimoto

**Root cause:** m7 = 0.107 (vs TargetDiff 0.860 cite-only; gap −0.753). The
existing 4-fix bundle lifted Lambda-search diversity; chemprop v2 candidate
side has no analogous augmentation lever wired into the v3 driver.

| # | Candidate | Effectiveness | LOC | Risk | Available code |
|---|-----------|---------------|-----|------|----------------|
| 1.1 | **Wire `ClickAugmentedChempropGenerator` into v3 driver as opt-in `--use-click-aug` (use_click_prob=0.5)** [TOP] | HIGH (existing code already lifts) | ~30 (1 hook + 1 wrapper import + 1 CLI flag + 1 eval helper) | LOW (opt-in, default off) | `molmetal/models/chemprop_v2_click_augmentation.py` (807 LOC, tested) |
| 1.2 | Add CLICHEMA (Park 2025) CGR-mode RDKit reaction applier to chemprop v2 input pipeline | MEDIUM (new code) | ~120 (new file + tests + driver hook) | MEDIUM (CGR encoding adds a new feature type) | None — would need new module |
| 1.3 | ClickGen RL/MCTS inner loop (Wang 2024 Nat Commun) | HIGH but GPU-only | ~250 | HIGH (GPU dependency) | None |

**Pick [TOP] = 1.1.** The wrapper already exists, is tested, and the candidate-
generation lever is the same lever that lifted Lambda-search m7 by +28.3% on
the Wf_lift_ev3 run.

### G2 — QED metallodrug calibration (m6 = 0.6123)

**Root cause:** RDKit QED (Bickerton 2012) calibrated on oral-drug-like
molecules; under-rates Pt/Pd/Au coordination centers per Casini & Pöthig 2024
(ACS Cent Sci) + Rauf 2024 (Front Drug Discov).

| # | Candidate | Effectiveness | LOC | Risk | Available code |
|---|-----------|---------------|-----|------|----------------|
| 2.1 | **Add `m6_kappa` (Shape Kappa Index = mean(Kappa1, Kappa2, Kappa3) normalised to [0,1])** [TOP] | MEDIUM (descriptor; needs MetalCytoToxDB calibration) | ~60 (1 metric function + 1 JSON emitter + tests) | LOW (new metric, additive) | `molmetal/data/rdkit_2d_extended.py` already exposes Kappa1/2/3 |
| 2.2 | Add ChemAxon `Shape Kappa` re-implementation from PMC11387870 | MEDIUM (validated vs Pt/Au dataset) | ~140 (full paper replication) | MEDIUM | None |
| 2.3 | Lipinski + Veber recalibration with metal-coord penalty | LOW (paper-only) | ~80 | MEDIUM | None |

**Pick [TOP] = 2.1.** Lowest cost; Kappa1/2/3 already in the descriptor
registry; emits side-by-side with QED so paper §3 can show both. Honest framing:
RDKit's `Descriptors.Kappa1/2/3` is the standard topological implementation; the
PMC11387870 paper's "Shape Kappa Index" is a normalised aggregate, which is
trivial to compute from RDKit's outputs.

### G3 — m11 per-metal Wasserstein pIC50 emission check

**Root cause:** Whether m11 (per-metal cytotoxicity stratification) is wired into
`metrics/by_metric/`. Wf-124 P3 ran 5-metal × Wasserstein loss but did not emit a
per-metal pIC50 JSON.

| # | Candidate | Effectiveness | LOC | Risk | Available code |
|---|-----------|---------------|-----|------|----------------|
| 3.1 | **Plumb m11 from v3 checkpoint JSONs into `metrics/by_metric/per_metal_pic50_wasserstein.json`** [TOP] | HIGH (closes TODO-130 G3) | ~25 (1 emitter script) | LOW (post-hoc read of existing checkpoints) | `molmetal/reports/wf200_chemprop_v2_v3/{metal}/per_seed_metrics_seed{N}.json` (already produced) |
| 3.2 | Re-train with multi-task D-MPNN per-metal head + Wasserstein (Heid 2024) | HIGH but GPU | ~300 (driver) | HIGH (GPU) | Existing `train_chemprop_v2_v3.py` |
| 3.3 | Wasserstein-distance multi-task pIC50 benchmark replication (Brahmavar 2024) | MEDIUM but GPU | ~250 | HIGH (GPU) | None |

**Pick [TOP] = 3.1.** Pure plumbing; reads per-seed JSON files that already exist
and emits a single m11 JSON in the standard format. ZERO model training.

### G4 — Per-lever ablation CPU-only pilot (TODO-182)

**Root cause:** TODO-182 calls for full 4-config × 5-metal × 3-seed = 60 GPU
runs (~7.5 hr). Wf-229 ablation launcher exists but uses chemprop v2 v3 (CPU
fallback ok).

| # | Candidate | Effectiveness | LOC | Risk | Available code |
|---|-----------|---------------|-----|------|----------------|
| 4.1 | **Extend `wf229_ablation_launcher.py` with a `--quick` 1-metal × 3-seed × 4-config CPU pilot (12 runs)** [TOP] | MEDIUM (attribution evidence) | ~40 (add quick mode + report writer) | LOW (script-arg driven) | `molmetal/scripts/wf229_ablation_launcher.py` (already exists) |
| 4.2 | Full 4-config × 5-metal × 3-seed GPU run | HIGH (paper-grade) | ~0 (already written) | HIGH (7.5 hr GPU) | Existing scripts |
| 4.3 | QM9 CPU baseline (Brahmavar 2024) | LOW (transfer weak) | ~120 | MEDIUM | None |

**Pick [TOP] = 4.1.** Reuses existing launcher; 12 runs on CPU = ~hours, not
days. Provides attribution evidence to defend the design choice in paper §3.

### G5 — TODO-130 m17 DrugOOD honest-negative re-eval

**Root cause:** Wf-174/185 reported chemprop v2 v3 DrugOOD AUC = 0.71/0.67/0.68
(assay/scaffold/size) vs iMoLD negative reference. Need v3 driver re-run on
post-6-fixes weight snapshot.

| # | Candidate | Effectiveness | LOC | Risk | Available code |
|---|-----------|---------------|-----|------|----------------|
| 5.1 | **Post-hoc eval of existing v3 checkpoints on DrugOOD scaffold split (`--seeds 3 --split scaffold`)** [TOP] | HIGH (closes honest-negative loop) | ~50 (1 eval script + 1 JSON emitter) | LOW (read-only on checkpoints) | `molmetal/scripts/wf185_chemprop_v2_drugood_full.py` (existing driver) |
| 5.2 | CSRL (Liu 2025 BIB) re-train as SOTA comparison | HIGH but GPU | ~300 | HIGH (GPU) | None |
| 5.3 | GOODHSE (Piao 2024 CVPR) re-train | HIGH but GPU | ~300 | HIGH (GPU) | None |

**Pick [TOP] = 5.1.** Post-hoc metric extraction on existing v3 checkpoints is
the only CPU-feasible action; CSRL/GOODHSE numbers would be cite-only.

---

## §3 TOP fixes (final pick)

The 5 picks (one per gap) form a coherent "CPU-only paper §3/§4 closure" bundle:

| # | Fix ID | What | File(s) | LOC est | Risk |
|---|--------|------|---------|---------|------|
| 1 | **F1-ClickAug-V3Wire** | Wire `ClickAugmentedChempropGenerator` into `train_chemprop_v2_v3.py` as opt-in `--use-click-aug`; emit post-aug diversity JSON | `molmetal/scripts/train_chemprop_v2_v3.py` (hook), `molmetal/reports/wf252_fix/click_aug_eval.py` (new) | 30 | LOW |
| 2 | **F2-M6Kappa** | Add `m6_kappa` Shape Kappa Index metric + side-by-side JSON with QED | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (extend), `metrics/by_metric/m6_kappa.json` (new) | 60 | LOW |
| 3 | **F3-M11Emit** | Plumb m11 (per-metal Wasserstein pIC50) from existing per-seed JSONs into `metrics/by_metric/per_metal_pic50_wasserstein.json` | `molmetal/reports/wf252_fix/m11_emit.py` (new) | 25 | LOW |
| 4 | **F4-AblationCPU** | Add `--quick` 1-metal × 3-seed × 4-config CPU pilot to `wf229_ablation_launcher.py` | `molmetal/scripts/wf229_ablation_launcher.py` (extend), `molmetal/reports/wf229_ablation/quick_pilot/` (new outputs) | 40 | LOW |
| 5 | **F5-DrugOOD-Reeval** | Post-hoc eval of existing v3 checkpoints on DrugOOD `scaffold` split × 3 seeds; emit `metrics/by_metric/drugood_v3_scaffold.json` | `molmetal/reports/wf252_fix/drugood_reeval.py` (new) | 50 | LOW |

**Total LOC:** 30 + 60 + 25 + 40 + 50 = **205 LOC** (including tests).

---

## §4 Per-fix design (detailed)

### F1 — ClickAug-V3Wire (`--use-click-aug` in v3 driver)

**Backwards-compat gate (G1):** default behaviour unchanged. Without the flag,
v3 driver emits the same SMILES as before. The flag is consumed only at the
candidate-generation step (`--use-click-aug` triggers a wrapper around the
candidate sampler).

**Mechanism:**
1. Add `from molmetal.models.chemprop_v2_click_augmentation import ClickAugmentedChempropGenerator` to the imports block (try/except for backward-compat).
2. Add `args.use_click_aug: bool = False` to `argparse`.
3. In the candidate-generation call site (if present), wrap the base sampler
   in `ClickAugmentedChempropGenerator(base_sampler, use_click_prob=0.5,
   rng=random.Random(args.seed))`.
4. After generation, emit
   `molmetal/reports/wf200_chemprop_v2_v3/click_aug_eval.json` containing:
   - `n_base`, `n_augmented`
   - `diversity_lift_scaffolds`
   - `diversity_lift_tanimoto`
   - `per_rule_fire_count`
   - `wall_time_s`

**Why CPU:** RDKit-only, no torch, no chemprop forward pass in the augmentation.

**Honest framing:** This is the same lever that lifted Lambda-search m7 by
+28.3% on Wf_lift_ev3. PROJECTED chemprop v2 lift: m7 → ~0.18-0.22 (vs current
0.107). Cannot reach TargetDiff 0.860 (cite-only); the gap remains but narrows.

### F2 — M6Kappa (Shape Kappa Index metric)

**Backwards-compat gate (G2):** additive metric. QED computation unchanged; the
new metric is opt-in via `--emit-m6-kappa`.

**Mechanism:**
1. Add `def m6_kappa_from_smiles(smi: str) -> float | None:` to
   `molmetal/molmetal_lam/sbdd_env/metrics_v2.py`:
   - Read `mol = Chem.MolFromSmiles(smi)`
   - Compute `k1 = Descriptors.Kappa1(mol)` / `k2 = Descriptors.Kappa2(mol)` /
     `k3 = Descriptors.Kappa3(mol)`
   - Normalise to [0,1] via min/max clipping from a reference distribution
     (RDKit's default normalisation in `rdMolDescriptors.CalcKappa` already
     returns kappa-shape-index values; clip to `[0, 2.5]` then rescale).
   - Return `mean(normalised_k1, normalised_k2, normalised_k3)`.
2. Add `m6_kappa_mean(smiles_list)` aggregator following the existing
   `metric_*_mean` convention.
3. Emit `metrics/by_metric/m6_kappa.json` from the Wf-61 close-out path (same
   449 metallo-drug SMILES as `qed.json`).
4. Add 1 test: `tests/test_m6_kappa.py` — assert (a) RDKit round-trip on
   cisplatin/carboplatin/oxaliplatin SMILES, (b) returned value in [0,1], (c)
   the 449-SMILES aggregate is finite.

**Why CPU:** RDKit only. No GPU.

**Honest framing:** This is a *normalised topological index*, not a learned
metallodrug score. Validate against MetalCytoToxDB labelled pIC50 in a follow-up
workstream (not in this fix). PROJECTED m6_kappa on the same 449 SMILES: ~0.55
(numerical estimate based on RDKit default ranges for organometallics; pending
MEASURED re-computation).

### F3 — M11Emit (per-metal Wasserstein pIC50 plumbing)

**Backwards-compat gate (G3):** read-only on existing checkpoints; new emitter
script only.

**Mechanism:**
1. Read all per-seed JSONs from
   `molmetal/reports/wf200_chemprop_v2_v3/{metal}/per_seed_metrics_seed{N}.json`.
2. Aggregate per-metal test-AUC mean + std + per-metal Wasserstein loss final
   value.
3. Emit `metrics/by_metric/per_metal_pic50_wasserstein.json` matching the
   standard `metrics/by_metric/{name}.json` schema:
   ```json
   {
     "metric": "per_metal_pic50_wasserstein",
     "description": "Per-metal test-AUC + Wasserstein pIC50 loss from Wf-124 P3 / Wf-200 v3 checkpoints",
     "per_metal": {
       "Ru": {"test_auc_mean": ..., "test_auc_std": ..., "w2_final_mean": ..., "n_runs": N},
       "Ir": {...}, "Os": {...}, "Rh": {...}, "Re": {...}
     },
     "status": "MEASURED",
     "timestamp": "2026-09-28"
   }
   ```

**Why CPU:** JSON parsing only.

**Honest framing:** If the per-seed JSONs do not exist for some (metal, seed),
the corresponding entry is `null` with `n_runs=0`. PROJECTED: 5 metals × 3 seeds
= 15 entries IF Wf-124 P3 ran to completion (it did for Ru seed=0 per
`training_summary.json`; the rest pending).

### F4 — AblationCPU (1-metal × 3-seed × 4-config pilot)

**Backwards-compat gate (G4):** new `--quick` flag on the existing launcher;
default behaviour unchanged.

**Mechanism:**
1. Add `--quick` arg to `wf229_ablation_launcher.py`:
   - Default `--quick`: 1 metal (Ru), 3 seeds (0,1,2), 4 configs (K-only /
     D-only / F-only / K+D+F), 5 epochs each → 12 runs.
   - Paper-grade default (no `--quick`): 5 metals × 3 seeds × 8 configs = 120
     runs (kept for GPU retrain).
2. In `--quick` mode, set `--epochs 5` and `--device cpu`; skip BF16; set
   `--max-rows 5000` to bound per-run time.
3. Aggregate per-config table + per-lever attribution (delta vs baseline).
4. Emit `molmetal/reports/wf229_ablation/quick_pilot/ablation_summary.json`
   with per-lever delta + PROJECTED vs MEASURED labels.

**Why CPU:** backbone already supports CPU mode; `--quick` keeps the per-run
SMILES-featurise step within ~5-7 min on multi-core CPU.

**Honest framing:** This is a **pilot** for attribution evidence, NOT a
paper-grade result. PROJECTED per-lever attribution (delta test-AUC):
- K-only: +0.005 (uncertainty weighting helps marginal metals)
- D-only: +0.012 (DANN adversarial head helps OOD)
- F-only: +0.018 (4-domain concat dataset)
- K+D+F: +0.049 (matches Wf-124 P3 anchor)
These are PROJECTED values, not MEASURED — actual numbers depend on the
Wf-124 P3 weight snapshot used for the pilot.

### F5 — DrugOOD-Reeval (post-hoc v3 eval on scaffold split)

**Backwards-compat gate (G5):** read-only on existing v3 checkpoints + new
eval script; no driver changes.

**Mechanism:**
1. New script `molmetal/reports/wf252_fix/drugood_reeval.py`:
   - Load DrugOOD `lbap_refined_ic50_scaffold` (266K SMILES, ~95K train).
   - Load existing v3 checkpoints from
     `molmetal/checkpoints/wf200_chemprop_v2_v3/{metal}_seed{N}.pt`.
   - Run forward pass on DrugOOD val/test split through chemprop v2 backbone
     + per-metal head (CPU mode, BF16 OFF).
   - Compute ROC-AUC + AUPR per (metal, seed).
2. Run with `--seeds 3 --split scaffold --metal {Ru,Ir,Os,Rh,Re}` (15 runs).
3. Emit `metrics/by_metric/drugood_v3_scaffold.json`:
   ```json
   {
     "metric": "drugood_v3_scaffold",
     "description": "Post-hoc chemprop v2 v3 eval on DrugOOD scaffold split × 5 metals × 3 seeds",
     "per_metal": {
       "Ru": {"auc_mean": ..., "auc_std": ..., "n_seeds": 3},
       ...
     },
     "baseline_imold": {"assay": -0.0074, "scaffold": -0.0177, "size": -0.0035},
     "status": "MEASURED",
     "honest_framing": "Post-hoc metric extraction on v3 checkpoints; v3 weights were NOT re-trained on DrugOOD (no leakage); scaffold split is the most generalisable OOD lever."
   }
   ```

**Why CPU:** small forward-pass on val/test split, ~95K val × 5 metals × 3
seeds = 1.4M samples. Per-sample time at chemprop v2 backbone hidden=64/2
(CPU mode) ~2 ms → ~45 min total. Fits in CPU budget.

**Honest framing:** PROJECTED AUCs vs iMoLD cite-only baseline:
- Ru: ~0.65-0.70 (above iMoLD negative −0.0177)
- Ir/Os/Rh/Re: ~0.55-0.65 (smaller training set, noisier)
- Aggregate: ~0.62 mean
These are PROJECTED from the chemprop v2 backbone capacity; actual numbers
depend on the checkpoint convergence state.

---

## §5 Honest framing (for paper §6 Limitations)

1. **No magic CPU bullet exists for G3/G5** beyond plumbing / post-hoc eval.
   Model weights are GPU-trained; only metric extraction can run CPU.
2. **F1 (click-aug wire-in) is the most paper-grade CPU contribution** —
   the lever directly improves m7 Diversity Tanimoto, which is the largest
   negative gap in TODO-130.
3. **F4 (CPU ablation pilot) provides attribution evidence** for paper §3
   algorithm description; the paper-grade 5-metal × 3-seed version still needs
   GPU retrain.
4. **None of these fixes displace R20/R21 ship work** — they are paper §3/§4
   supplementary material, not core ship blockers.
5. **F2 (m6_kappa) is a normalised topological index**, NOT a learned
   metallodrug score. Treat it as a side-by-side metric with QED in paper §3;
   do not claim it as a substitute for QED without validation against
   MetalCytoToxDB labelled pIC50 (deferred to follow-up workstream).
6. **F3 (m11 emission) closes the gap on TODO-130 G3** but is plumbing only;
   the underlying per-metal Wasserstein pIC50 training is Wf-124 P3 (already
   done for Ru seed=0; rest pending).

---

## §6 Acceptance gates (FLAT)

| Gate | Value | Rationale |
|------|-------|-----------|
| G1_top_fixes_count | 5 | One fix per gap (G1, G2, G3, G4, G5) |
| G2_top_fixes_described | 5 | Each fix has a §4 detailed design |
| G3_total_loc_estimate | 205 | 30 + 60 + 25 + 40 + 50 LOC (incl. tests) |
| G4_fix_design_path | `/home/hugo/codes/mol-metal/molmetal/reports/wf252_research/fix_design.md` | This file |
| G5_m7_diversity_projection | 0.20 | PROJECTED m7 lift via F1 click-aug; current MEASURED 0.107 → PROJECTED 0.18-0.22 range; cite-only TargetDiff 0.860 unreachable on CPU |
| G6_m6_qed_projection | 0.55 | PROJECTED m6_kappa on 449 metallo-drugs; not validated against MetalCytoToxDB labelled pIC50 (deferred); QED 0.6123 unchanged |

**honest_framing_note:**
F1 (click-aug wire-in) is the strongest paper-grade CPU contribution — the
existing `ClickAugmentedChempropGenerator` is fully tested and the lever
mechanism is the same one that lifted Lambda-search m7 by +28.3% on
Wf_lift_ev3. F2 (m6_kappa) is a normalised topological index, NOT a learned
metallodrug score — paper §3 should show it side-by-side with QED. F3, F5 are
pure plumbing/post-hoc eval and close TODO-130 G3 + G5 honestly. F4 provides
attribution evidence but is a pilot only; full paper-grade ablation needs GPU.
None of these displace R20/R21 ship work.

---

## §7 Implementation order (suggested)

1. **F3 (m11 emit)** — pure plumbing, lowest risk, 25 LOC, no model changes.
   Unblocks TODO-130 G3 immediately.
2. **F1 (click-aug wire-in)** — opt-in CLI flag on v3 driver, 30 LOC, uses
   existing tested module. Most paper-grade contribution.
3. **F2 (m6_kappa)** — additive metric, 60 LOC, low risk. Side-by-side with
   QED for paper §3.
4. **F4 (ablation CPU pilot)** — extends existing launcher, 40 LOC. Provides
   attribution evidence.
5. **F5 (DrugOOD reeval)** — 50 LOC post-hoc eval script. Closes honest-
   negative loop.

Order chosen by **risk-adjusted-effectiveness**: lowest-risk first (plumbing),
then additive metrics, then opt-in driver hooks, then paper-grade eval.

---

## §8 Cross-references

* Wf-252-Research: `molmetal/reports/wf252_research/lit_review.md` (this file
  synthesises from that review).
* Existing click-aug code: `molmetal/models/chemprop_v2_click_augmentation.py`
  (807 LOC, tested in `tests/test_chemprop_v2_click_augmentation.py`).
* Existing ablation launcher: `molmetal/scripts/wf229_ablation_launcher.py`
  (187 LOC).
* Existing v3 driver: `molmetal/scripts/train_chemprop_v2_v3.py` (685 LOC).
* Existing Kappa descriptors: `molmetal/data/rdkit_2d_extended.py:175-176,
  253-254` (Kappa1/2/3 already in the descriptor registry).
* TODO-130 G3/G5 reference: TODO/completed/130_wf130_p1_catherine_30.md §3.4
  (TODO-130 G3 + G5 explicit).
* TODO-182 (G4): TODO/pending/182_wf124_p3_per_innovation_ablation.md §3
  (ablation matrix).
