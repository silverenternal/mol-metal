# WF-Data-Gap-Analysis — High-level summary

**Workflow:** WF-Data-Gap-Analysis
**Created:** 2026-09-14
**Status:** COMPLETE (pure-analysis workflow; no code modifications to `molmetal/`)
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Stack:** uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101

---

## Goal (one paragraph)

Download one Q1 SBDD paper as benchmark (TargetDiff or DiffSBDD), inventory
all tables/figures/statistics it reports, then compare to our current data
inventory, write a comprehensive gap-analysis TODO with prioritised ship
targets. **Honest framing mandatory:** cite-only SOTA is context, never a
one-sample significance-test null; Lambda arm is measured at N≤10 pockets
until Round-13 ships; CFM arm is decoder-bound (0/96 docked today).

---

## What we shipped (high level)

1. **Benchmark paper selected + inventoried:** TargetDiff (arXiv:2303.03543
   v1, ICLR 2023 OpenReview kJqXEPXMsE0; cloned repo `142f1eb`, Jul 13
   2023). 100-pocket × 100-mol × 1000-DDPM-step protocol = **10 M NFE
   total**.
2. **Per-metric inventory:** 25 metrics catalogued (7 primary Table 1 + 10
   extended follow-up + 8 out-of-band anticancer). Headline numbers: Vina
   Dock −7.80 / −7.91 kcal/mol (Avg/Med); High Affinity 58.1 %; strict
   Triple-threshold success rate 10.5 %; CoM shift 1.45 Å.
3. **Mol-Metal current inventory:** ~250 measured cells across 4 unique
   pockets (1h36, 830c, test_001, test_005 modeled); Lambda arm N=10×3
   (3000 mols); CFM arm **decoder-bound (0/96)**. Measures 12/25 metrics.
4. **Per-metric gap table:** 25-row table at
   `gap_analysis_targetdiff.md` showing TargetDiff scale vs Mol-Metal
   current vs gap type vs ship target vs priority (P0/P1/P2/P3).
5. **Prioritised closure plan:** `TODO/pending/22_data_gap_alignment_plan.md`
   with 5 ship targets (Round-12 1–2 h, Round-13 6 h, cite-only 1 h,
   metal ablation +30 min, wet-lab out-of-scope). Total ~8 h GPU + 3 d
   serial write-up.
6. **Cross-references wired into TODO-13 §scope and TODO-14 §3:** explicit
   gap to close by Round-12 (9 P0 metrics) and Round-13 (17 metrics total).
7. **Cite-only SOTA column already ship** at `wf_3_citeonly_sota.tex`
   (9 SOTA rows + 2 Lambda rows, 7 protocol-mismatch flags M1–M7).

---

## File index

### Authoritative gap-analysis documents

| File | Purpose | Status |
|---|---|---|
| `molmetal/reports/wf_data_gap_analysis.md` | **THIS FILE** — high-level summary + file index | shipped |
| `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md` | Full 25-row per-metric gap table (TargetDiff ↔ Mol-Metal) | shipped |
| `molmetal/reports/provenance_targetdiff.md` | TargetDiff inventory (100-pocket × 100-mol protocol) | shipped |
| `TODO/pending/22_data_gap_alignment_plan.md` | Prioritised closure plan (P0/P1/P2/P3, time, resources, risks, sequence diagram) | shipped |

### Cross-referenced authoritative inputs

| File | Purpose | Status |
|---|---|---|
| `molmetal/reports/sota_alignment_gap_analysis.md` | Lambda-vs-TargetDiff protocol audit (M1–M7 mismatch flags) | shipped |
| `molmetal/reports/wf_3_citeonly_sota.md` | Cite-only SOTA workflow narrative | shipped |
| `molmetal/reports/wf_3_citeonly_sota.tex` | Cite-only SOTA LaTeX fragment (paper-ready) | shipped |
| `molmetal/reports/wf_3_citeonly_sota.csv` | Cite-only SOTA CSV (machine-readable) | shipped |
| `molmetal/reports/wf_3_citeonly_sota_consistency.md` | Provenance audit (`all_have_provenance = TRUE` for 9/9 SOTA rows) | shipped |
| `molmetal/reports/wf_lambda1c_pilot_v3/final.md` | Lambda arm MEASURED pilot (N=5×3 with `--metal-seed cisplatin` + ablation) | shipped |
| `molmetal/reports/round11_engine_parity_n50.md` | Vina ↔ QVina parity r=0.9983 (N=50) | shipped |
| `molmetal/reports/round10_e2e_pt_cfg_vina.md` | CFM arm decoder-bound diagnostic (0/96) | shipped |
| `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` | Cite-only SOTA comparison (round-9 baseline) | shipped |

### TODO files updated by this workflow

| File | Update |
|---|---|
| `TODO/completion_audit_2026-09-13.md` | Appended §"WF-Data-Gap-Analysis — Q1 SBDD benchmark data gap (2026-09-14)" with headline numbers + ship target summary |
| `TODO/pending/13_top_journal_pilot_r12.md` | Appended §scope (gap-closure mandate + 9 P0 metrics + 4 P1 patches + 5 deferred) |
| `TODO/pending/14_full_100pocket_paper_r13.md` | Appended §3 (gap cross-reference + 12 closed + 8 deferred + risk inheritance) |
| `TODO/README.md` | Updated index to include TODO-22 (and TODO-13/14/15/17/18/19/20/21) |

### Harness / config files referenced (NOT modified)

| File | Purpose |
|---|---|
| `molmetal/scripts/r4_lambda_only_run.py` | Round-12/13 harness (already extended with `--seeds` + `--metal-seed`) |
| `molmetal/scripts/r4_c_full_sweep.py` | Already-cite protocol exists |
| `molmetal/configs/sota_aligned_targetdiff.yaml` | Canonical reference config |
| `paper/sections/04_evaluation.tex` | Round-12/13 §4 + §4.5 integration target |
| `paper/sections/05_ablation.tex` | Round-12/13 §5 ablation target |
| `paper/sections/07_future_work.tex` | Wet-lab validation future-work declaration |

---

## Headline gap analysis (one-page version)

### TargetDiff inventory (Q1 SBDD benchmark)

| Axis | Count |
|---|---:|
| n_tables (main) | **3** |
| n_figures (main) | **5** |
| n_metrics (primary Table 1) | **7** |
| n_metrics (extended follow-up) | ~10 |
| n_pockets | **100** (held-out test) |
| n_samples_per_pocket | **100** |
| NFE total | **10 M** (100 pockets × 100k) |
| Docking engine | QVina (exh=8) |
| Atom-vocab-restricted? | yes (discrete categories) |
| Metal-aware? | **no** (metal-agnostic) |

### Mol-Metal current inventory (2026-09-14)

| Axis | Count |
|---|---:|
| n_cells (measured runs, total) | **~250** |
| n_pockets (measured, total) | **4 unique** (1h36, 830c, test_001, test_005 modeled) |
| n_seeds (per pocket, Lambda arm) | **3** (42, 0, 1234) |
| n_metrics (computed per cell) | **9** |
| n_metrics (paper-grade, TargetDiff-comparable) | **4** |
| Docking engines | Vina 1.2.7 + QuickVina 2.1 (parity r=0.9983) |

### Gap closure at each round

| Round | Metrics ship-ready | Wall | Notes |
|---|---:|---|---|
| Round-12 (N=10×3) | **9 P0** | 1–2 h GPU | Vina Score, High Affinity, Triple/Relaxed Success Rate, Validity, QED, SA, Lipinski, logP, Recon cascade |
| Round-13 (N=100×3) | **17** | 6 h GPU | All Round-12 + Vina Dock, Diversity (Morgan-ECFP4), Steric clash, Vina Min |
| Cite-only SOTA (already ship) | **context column** for all 7 primary | 1 h (done) | `wf_3_citeonly_sota.tex` |
| Metal ablation (Round-12 ± cisplatin) | Lambda-vs-Lambda | +30 min | Doubles cells 30 → 60 |
| Wet-lab validation | **0** (out of scope) | 0 | §7 future-work declaration |

### What does NOT close (deferred to supplementary or post-R13)

| # | Metric | Reason |
|---|---|---|
| 12 | JSD bond-distance histogram (8 bond types) | Lambda has no raw 3D by design |
| 13 | Rigid-fragment RMSD post-MMFF | 2-day module |
| 15 | CoM shift vs reference | 1-day module |
| 17 | Strain energy (PoseCheck) | 2-day module |
| 18 | Interaction fingerprint (PLI, PoseCheck) | 3-day wire |
| 20 | mol_stable / atm_stable | CFM arm blocked on TODO-21 |
| 23–25 | pIC50, TPSA/RotB, cytotox | Supplementary S2 only |

---

## Honest framing reminders (must preserve)

1. **Cite-only SOTA is context, NEVER a one-sample significance-test null.**
   Per `TODO/pending/19_user_decisions.md` §cite-only.
2. **Lambda arm is measured at N≤10 pockets** until Round-13 ships; do not
   claim "100-pocket measured" before Round-13 closes.
3. **CFM arm is decoder-bound (0/96 docked today).** Geometric metrics
   (mol/atm stability, rigid-fragment RMSD) depend on TODO-21 retrain.
4. **Wet-lab validation is out of sandbox scope.** Declare §7 future-work;
   cite TODO-18 pIC50 + TODO-15 anticancer as computational surrogates.
5. **QVina-GPU scaling beyond N=50 is unverified** (parity r=0.9983 at N=50
   is strong evidence but not a 100×3 guarantee). Round-12 validates the
   protocol first; Round-13 scales only after.

---

## Decisions referenced

| Decision | Document | Status |
|---|---|---|
| D1 — Cite-only SOTA path (no re-runs of competitors) | `TODO/pending/decisions.md` D1 | approved |
| D2 — Lambda-only + cite-only fallback if CFM decoder remains bound | `TODO/pending/decisions.md` D2, memory `rx7800xt_strategy.md` | approved |
| D4 — L-1 oracle (DiffDock) for measured Vina | `TODO/pending/decisions.md` D4 | approved (alternative to L-4 REINVENT4) |
| D6 — REINVENT4 multiproperty bridge | `TODO/pending/19_user_decisions.md` D6 | env-blocked; TODO-05 |
| D7 — Vina ↔ QVina both in headline table | `TODO/pending/19_user_decisions.md` D7 | approved; parity N=50 r=0.9983 |
| Cite-only SOTA footnote pattern | `TODO/pending/19_user_decisions.md` §cite-only | approved |
| MW-range flag (300–700 Da) for metal complexes | `TODO/pending/19_user_decisions.md` MW-range | approved |

---

## Memory cross-references

- `~/codes/try_triton_on_rocm/.claude/memory/`: `sota_papers_r4.md` (PlatinAI,
  MetalCytoToxDB, TransDiffSBDD, DrugOOD metrics), `rx7800xt_strategy.md`
  (D2 cite-only path), `molmetal_state_2026_09_12.md` (TODO roadmap), and
  earlier round summaries (Round-4 through Round-8).

---

## Next action items (post-WF-Data-Gap-Analysis)

1. **WF-3 Round-12 N=10×3 scientific pilot** (TaskList #355, ~1–2 h GPU) —
   closes 9 P0 metrics. Cite-only SOTA column attached per pocket row.
2. **WF-5 Round-13 100×3 sweep** (TaskList #357, ~6 h GPU, depends on
   Round-12) — closes 17 metrics total. Wall-clock budget ≤6 h via QVina-GPU.
3. **WF-6 paper draft** (TaskList #358, ~3 d serial) — §4 evaluation,
   §4.5 100-pocket headline, §5 ablation, §7 future-work.
4. **TODO-21 CFM retrain** (independent, optional) — if user authorises,
   geometric column ships; otherwise Lambda-only + cite-only is the
   canonical Q1 path.

---

**Workflow complete.** All deliverables shipped. No code modifications to
`molmetal/`. Cite-only SOTA column already integrated in
`paper/sections/04_evaluation.tex` (TaskList #462 verified). Round-12
and Round-13 ship targets are unblocked and ready for user authorisation.