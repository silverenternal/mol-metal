# TODO 183 — Wf-123 P3 + Wf-124 P3 Gap Routing Plan: Top 5 Next Ultracodes

**Date**: 2026-09-27 (post-Wf-123 P3 + Wf-124 P3 ship, party-twenty-spirit §3 + §R19 self-check applied)
**Owner**: Hugo Lee + ultracode agent
**Status**: 📐 PLANNING — ranks the 6 already-filed gap TODOs (177-182) into a 5-ultracode execution queue + identifies whether any GPU-default fix is needed first
**Parent**: `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1) + `TODO/pending/178_wf123_p3_metalcytotox_csv_ondisk.md` (G2) + `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (G3) + `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4) + `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5) + `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6)
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 5-seed × 50-epoch protocol + §7 MEASURED-vs-PROJECTED tag discipline + `mol-metal-anticancer` Tier-1 SOTA comparison

---

## §1 Why this plan exists (user request + harness task)

The user said: "刚才结束的两个工作流有什么没做好的东西就先在todo目录里规划一下好了" (For things that weren't done well in the two workflows that just ended, plan them in the todo directory).

The harness task asked for: top 5 next ultracodes ranked by (a) ROI = gap closure per effort, (b) dependencies, (c) whether GPU-default fix is needed first.

This file is the single TODO entry that converts the 6 filed gap TODOs into a **single ranked queue** with explicit blockers, ETA, and GPU-default gate.

---

## §2 Status of the 6 filed gap TODOs (ground truth 2026-09-27)

| ID | Title | File | Wall (est) | GPU? | Status |
|---|---|---|---:|---|---|
| **G2** | metalcytotox CSV on disk | `pending/178_*.md` | 0.5-1 day | ❌ NO | 🟢 **PARTIAL** — CSV exists on disk at `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (26801 rows, 5 metals: Ru 19135 / Ir 4546 / Os 1118 / Rh 1134 / Re 868); schema is `smiles, metal, scaffold_id, split, activity` (5 cols — **MISSING `pic50`, `cell_line`, `source`**); smoke test currently SKIPs (graceful) instead of FAILs — needs FAIL-on-missing + audit JSON |
| **G4** | 14/15 (metal,seed) PENDING | `pending/180_*.md` | 1.8 hr wall + 0.5 day overhead | ✅ RTX 5090 GPU-1 | 🔴 **OPEN** — only Ru seed=0 complete (0.7613 +0.049 lift); 14 PENDING need run |
| **G5** | 50-epoch full sweep | `pending/181_*.md` | 5 hr wall + 0.5 day overhead | ✅ RTX 5090 GPU-1 | 🔴 **OPEN** — current 30-epoch undertraining; W₂ trajectory still decreasing at epoch 30 (final 0.3906) |
| **G6** | Per-innovation ablation | `pending/182_*.md` | 7.5 hr wall + 1 day overhead | ✅ RTX 5090 GPU-1 | 🔴 **OPEN** — 4 configs × 5 metals × 3 seeds = 60 runs needed for attribution |
| **G3** | W₂ alignment on PlatinAI 4-cell | `pending/179_*.md` | 4-5 hr wall + 1 day overhead | ✅ RTX 5090 | 🟡 **OPEN** — biggest single-metric gap (L5 R²=+0.163 → 0.85 = −0.687) |
| **G1** | M1+M2 downstream metric lift | `pending/177_*.md` | 6 weeks (staggered across 6 consumers) | ✅ Blackwell + RTX 5090 dual | 🟡 **OPEN** — composite of G3/G4/G5/G6 consumer-side lifts |

**Total wall-time if all 6 closed**: ~12 days GPU + 1 week CPU hygiene (G2)

---

## §3 GPU-default fix gate (party-twenty-spirit §3.5)

Per INDEX.md + TODO-148 + R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §1.4 — **standing "kill then adapt" directive**: all GPU work > 1 hr DEFERRED pending explicit fresh user OK.

The Wf-124 P3 Ru seed=0 run (7.5 min on RTX 5090) is the only GPU work post-directive. It is BELOW the 1-hr threshold and was authorized. The 4 upcoming GPU ultracodes (G4 / G5 / G6 / G3) ALL exceed 1 hr:
- G4: 1.8 hr wall (within 2-hr subagent budget but close)
- G5: 5 hr wall — REQUIRES explicit user OK
- G6: 7.5 hr wall — REQUIRES explicit user OK
- G3: 4-5 hr wall — REQUIRES explicit user OK

**GPU-default fix needed first? NO** — RTX 5090 + Blackwell environment is already proven (Wf-124 P3 shipped on it). The directive is **process**, not technical. Each wave needs fresh user OK per standing rule. There is NO environment bug to fix.

---

## §4 Top 5 next ultracodes (ranked)

Ranking criterion (per harness task):
1. **ROI** = gap closure per effort (gap_size / wall_time)
2. **Dependencies** = must this run after another?
3. **GPU-default gate** = does this need fresh user OK?

### Rank 1 — G2: CSV hygiene FAIL-not-SKIP fix (CPU, 0.5-1 day)

- **Why rank 1**: ZERO GPU risk (CPU-only); the gap is small (CSV already exists, just schema-extension + smoke test flip + audit JSON); UNBLOCKS the honest cross-metal guarantee for §4; cheapest ROI per hour.
- **Scope**: 1) extend schema (add `pic50`, `cell_line`, `source` cols); 2) flip `test_metalcytotox_csv_integration_smoke` to FAIL not SKIP; 3) write `molmetal/reports/wf123_p3_integration/metalcytotox_csv_audit.json`; 4) update `molmetal/scripts/cytotox_filter.py` if needed.
- **ETA**: 0.5-1 day wall (CPU-only, ~4-6 hr engineering + 1-2 hr audit)
- **Expected gains**: §6 limitations no longer needs to say "cross-metal transfer UNVERIFIED"; enables downstream G4/G5/G6 to cite real cross-metal CSV path.
- **Deps**: none (independent)
- **Blockers**: none — pure CPU hygiene
- **MEASURED label**: PROJECTED @ Wf-125 (post-fix) — audit JSON will be MEASURED

### Rank 2 — G4: 14/15 (metal,seed) PENDING (GPU, 1.8 hr + 0.5 day)

- **Why rank 2**: HIGHEST ROI per GPU-hour — wall-time is 1.8 hr (below the 2-hr subagent harness cap, so single-session launchable); directly addresses m1 Ru AUC + 4 other metals (Ir/Os/Rh/Re); needed for §4 honest per-metal 3-seed aggregate; WITHOUT this, paper §4 cannot honestly cite m1 mean ± std.
- **Scope**: Run 14 PENDING (metal, seed) pairs at `--epochs 30 --alpha 0.5 --wass-weight 0.05 --use-l2-guard --device cuda:1`; regenerate `training_summary.json` with per-metal 3-seed mean ± std; verify W₂ decay monotonic per (metal, seed).
- **ETA**: 1.8 hr wall + 0.5 day overhead (run + audit + summary) = ~1.5 days total
- **Expected gains**: m1 per-metal 3-seed MEASURED (or honest-negative if < 0.86); enables §4 Table 2 row 1 honest report; foundational for G5 (50-epoch is an extension of G4).
- **Deps**: G2 (CSV hygiene must be closed first so audit is clean)
- **Blockers**: subagent harness 1.8 hr is borderline — needs fresh user OK per standing rule
- **MEASURED label**: MEASURED @ Wf-124 P3.5 — per (metal, seed) per-seed mean ± std

### Rank 3 — G5: 50-epoch full sweep (GPU, 5 hr + 0.5 day)

- **Why rank 3**: HIGH ROI for closure of undertraining concern — W₂ trajectory still decreasing at epoch 30 (final 0.3906); 50-epoch run is the proper scientific measurement per `mol-metal-gpu-execution` §6 standard protocol; without it, the 30-epoch numbers are MEASURED but possibly underfit.
- **Scope**: Re-run all 15 (metal, seed) pairs at `--epochs 50`; write `molmetal/reports/wf124_p3_50ep/{metal}/per_seed_metrics_seed{seed}.json`; 30ep vs 50ep comparison report; verify W₂ monotonic over 50 epochs; verify L2 guard grad-norm bounded.
- **ETA**: 5 hr wall + 0.5 day overhead = ~2 days total
- **Expected gains**: PROJECTED +0.01-0.03 absolute test_auc lift over 30-epoch (or plateau honest-negative); if 5-metal × 3-seed mean ≥ 0.86 → paper §4 LEAD-claim candidate for Krasnov 2026 + 5pp headroom; else §6 honest-negative.
- **Deps**: G4 (must run 30-epoch G4 sweep first to populate the baseline); trainer needs `--epochs 50` to be supported (currently default 30 per Wf-124 P3).
- **Blockers**: 5 hr wall EXCEEDS the 1-hr "kill then adapt" directive → explicit fresh user OK required; cannot run within a single subagent session — must be launched as background task with explicit OK.
- **MEASURED label**: MEASURED @ Wf-124 P3.6 — 50-epoch per (metal, seed)

### Rank 4 — G6: Per-innovation ablation (GPU, 7.5 hr + 1 day)

- **Why rank 4**: MEDIUM-HIGH ROI for paper §3 algorithm description — current Wf-124 P3 paper §3 MUST say "aggregate lift not per-lever attribution" because all 3 innovations ran simultaneously; 60 ablation runs would unlock per-lever attribution (multi-task 60-80% / W₂ 10-30% / L2 guard ~5% PROJECTED); important for §3 honesty, NOT for §4 headline.
- **Scope**: 4 configs × 5 metals × 3 seeds = 60 runs (multi-task only / W₂ only / L2 guard only / full recipe); ablation flags need to be added to trainer (`--no-multi-task`, `--no-wass`, `--no-l2-guard`, `--single-metal`); attribution analysis report.
- **ETA**: 7.5 hr wall + 1 day overhead (run + audit + ablation analysis + report) = ~3 days total
- **Expected gains**: per-lever attribution (multi-task %, W₂ %, L2 guard %, interaction %); paper §3 algorithm description can cite the per-lever attribution; if W₂ contribution < 20%, down-weight W₂ in future strengthening plans.
- **Deps**: G4 + G5 (the "full recipe" baseline is the G4/G5 number); ablation flags need engineering.
- **Blockers**: 7.5 hr wall EXCEEDS 1-hr directive + exceeds single subagent session; needs fresh user OK; ablation may run in 2 batches (2 × 3.75 hr) but each batch still > 1 hr.
- **MEASURED label**: MEASURED @ Wf-124 P3.7 — per (config, metal, seed) per-seed

### Rank 5 — G3: W₂ alignment on PlatinAI 4-cell (GPU, 4-5 hr + 1 day)

- **Why rank 5**: HIGHEST absolute gap (−0.687 L5 R²) but LOWEST ROI per GPU-hour because of the **12-d signal-floor caveat** — per `TODO/completed/127_l5_hilbert_diagnostic_closeout.md`, the 12-d RDKit 2D feature ceiling is 0.40-0.60 honest projection without F5_RDKIT_2D expansion to 32-d or 64-d; W₂ alone may not lift R² ≥ 0.85; better to do F5 expansion first OR in parallel.
- **Scope**: Wire `wasserstein_pic50_loss` into `molmetal/scripts/train_hilbert_astsr_end_to_end.py` with `--wass-weight 0.05` and `--ref-distribution platinai_4cell` flags; 5-seed × 50-epoch protocol; per-cell breakdown; W₂ decay curve verification.
- **ETA**: 4-5 hr wall + 1 day overhead (trainer modification + 5-seed run + per-cell audit + report) = ~2.5 days total
- **Expected gains**: PROJECTED L5 R² 0.40-0.60 (honest ceiling, NOT 0.85) per Wf-127 §Cause 2; if < 0.40 → §6 limitations; escalation path is F5_RDKIT_2D 32-d feature expansion.
- **Deps**: G2 (CSV hygiene); F5_RDKIT_2D expansion may be needed in parallel for ceiling lift.
- **Blockers**: 4-5 hr wall EXCEEDS 1-hr directive → fresh user OK required; trainer modification is non-trivial (need to wire `wasserstein_pic50_loss` into the joint loss without breaking the Wf-37 P1 bug-fix invariant — i.e., the bias-column + cosine LR + grad clip + L2 ridge + pairwise features must remain).
- **MEASURED label**: MEASURED @ Wf-127 P3 — per-seed per-cell R²

---

## §5 Composite / batch option: batch G4 + G5 + G6 into single 2-week wave

Per `mol-metal-gpu-execution` §6 5-seed × 50-epoch protocol: the G4 (1.8 hr) + G5 (5 hr) + G6 (7.5 hr) together = 14.3 hr wall, which is the natural R20-P2 GPU retrain block.

**Batch benefit**: per-lever attribution (G6) needs the 50-epoch baseline (G5) which needs the 30-epoch baseline (G4) — running them in sequence with consistent seed saves per-run overhead (data prep / model loading).

**Batch cost**: 14.3 hr wall EXCEEDS single subagent session; requires explicit fresh user OK + background launch + nightly audit.

**Recommendation**: if user OKs, batch G4+G5+G6 into a 2-week Wf-124 P3.5-3.7 wave with daily audit checkpoints; if user declines, run sequentially per rank.

---

## §6 G1 (consumer-side lift) — DEFERRED until G3/G4/G5/G6 ship

G1 (TODO-177) is a **composite** of G3/G4/G5/G6 consumer-side lifts. Per `party-twenty-spirit` §3.5 honest framing, G1 does NOT need to be re-spawned; once G3/G4/G5/G6 ship, the consumer-side lift table updates naturally:
- TODO-120 (m1 Ru AUC): lifted via G4 + G5
- TODO-122 (L5 Hilbert R²): lifted via G3 (with ceiling caveat)
- TODO-119 (m14 decode_ratio): lifted via G6 (closure bias contribution)
- TODO-121 (m19 LOMO Rh): lifted via G6 (W₂ transfer contribution)
- TODO-123 (m15 RMSD): lifted via G6 (W₂ pose contribution)
- TODO-125 (L12 OOD per-pocket): deferred to Wf-130 P3 (Catherine ≥ 25 metals × per-pocket Vina)

G1 will be marked SHIP in the parent TODO-126 once all 6 consumer lifts update with MEASURED numbers.

---

## §7 Honest framing per party-twenty-spirit §3.5

- **MEASURED**: Wf-123 P3 SHIP (25/25 M1+M2 integration tests PASS); Wf-124 P3 SHIP (Ru seed=0 test_auc=0.7613, +0.049 vs 0.7123 baseline; 18 trainer tests PASS).
- **PROJECTED @ R20-P2**: G2 (CSV hygiene) + G4 (14/15 PENDING) + G5 (50-epoch full sweep) + G6 (per-innovation ablation) + G3 (W₂ PlatinAI alignment) — all 5 are PROJECTED @ R20-P2 with explicit MEASURED-vs-PROJECTED labels.
- **Smaller-metal caveat** (per Wf-124 P3 §5): Rh/Os/Re < 1.2k rows → noisier per-seed AUC; per-seed std likely ±0.10 or more; report unfiltered, no best-seed selection.
- **W₂ regularization may slow early convergence** (per Wf-124 P3 §5): JKO-style CFL gate satisfied; predicted pIC50 forced to track reference before encoder has learned anything.
- **30-epoch undertraining concern** (per TODO-181 G5): W₂ still decreasing at epoch 30; 50-epoch is the proper scientific measurement.
- **Per-lever attribution honesty** (per TODO-182 G6): current Wf-124 P3 paper §3 MUST say "aggregate lift" not "per-lever attribution"; projected attribution (multi-task 60-80% / W₂ 10-30% / L2 guard ~5%) is a hypothesis, not measured.
- **No best-seed selection** (per `mol-metal-gpu-execution` §7): if 3-seed mean is below gate, the per-seed std IS the answer; do not pick the highest seed and call it MEASURED.
- **Standing GPU directive** (per INDEX.md §0 + TODO-148 + R20 plan §1.4): all GPU work > 1 hr DEFERRED pending explicit fresh user OK; subagent harness budget 3600s (1 hr); each of G3/G4/G5/G6 needs fresh user OK.
- **No fabrication**: do NOT cite 50-epoch numbers / per-lever attribution / L5 R² ≥ 0.85 as MEASURED until the corresponding gap TODO actually closes with 5-seed × 50-epoch protocol.

---

## §8 Cross-references

- `TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md` (G1) — composite, defers to G3/G4/G5/G6
- `TODO/pending/178_wf123_p3_metalcytotox_csv_ondisk.md` (G2) — Rank 1 (CPU hygiene)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (G3) — Rank 5 (L5 R² lift)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4) — Rank 2 (14/15 PENDING)
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5) — Rank 3 (50-epoch full sweep)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6) — Rank 4 (per-innovation ablation)
- `TODO/completed/172_comprehensive_lift_plan.md` (R21 master plan)
- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` (12-week 6-phase retrain plan)
- `mol-metal-gpu-execution` §6 5-seed × 50-epoch protocol
- `mol-metal-gpu-execution` §7 MEASURED vs PROJECTED tag discipline
- `party-twenty-spirit` §3.5 honest framing (0 容忍 不掩盖)

---

## §9 Top 5 ultracode summary (for harness JSON output)

| Rank | ID | Title | ETA (days) | GPU? | Expected gain | Deps | Fresh user OK needed? |
|---|---|---|---:|---|---|---|---|
| 1 | G2 | metalcytotox CSV hygiene | 0.5-1 | ❌ NO | §4 cross-metal CSV honest cite; §6 unblocks | none | ❌ NO (CPU) |
| 2 | G4 | 14/15 (metal,seed) PENDING | 1.5 | ✅ RTX 5090 | m1 per-metal 3-seed MEASURED | G2 | ⚠️ YES (1.8 hr > 1 hr directive) |
| 3 | G5 | 50-epoch full sweep | 2 | ✅ RTX 5090 | PROJECTED +0.01-0.03 lift OR plateau honest-neg | G4 | ⚠️ YES (5 hr > 1 hr directive) |
| 4 | G6 | Per-innovation ablation | 3 | ✅ RTX 5090 | per-lever attribution (multi-task / W₂ / L2 guard) | G4, G5 | ⚠️ YES (7.5 hr > 1 hr directive) |
| 5 | G3 | W₂ PlatinAI 4-cell alignment | 2.5 | ✅ RTX 5090 | L5 R² 0.40-0.60 PROJECTED (NOT 0.85 due to 12-d ceiling) | G2 | ⚠️ YES (4-5 hr > 1 hr directive) |

**GPU-default fix needed first? NO** — RTX 5090 environment already proven via Wf-124 P3. Each wave just needs fresh user OK per standing "kill then adapt" directive.

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: 5 ranked ultracodes span ~10 days wall-time (sequential) or ~14.3 hr wall + ~5 days overhead (batched). All 5 are PROJECTED @ R20-P2 with explicit MEASURED-vs-PROJECTED labels. The 5 ranked ultracodes cover the 6 filed gap TODOs (177-182); G1 (TODO-177) is composite and updates naturally once G3/G4/G5/G6 ship. The standing GPU directive applies to G3/G4/G5/G6 — each needs explicit fresh user OK before launch. No fabrication of MEASURED numbers until each gap TODO actually closes with the proper 5-seed × 50-epoch protocol.
