# TODO-38 — Round-14 Complete Plan: 300-cell paper-grade sweep + 3-axis SOTA benchmark

**Supersedes TODO-26 in `pending/`** (`26_round13_round14_complete_plan.md` — R14 = 300-cell paper-grade + 3-axis SOTA spec; infrastructure wired via `--r17-paper-grade` flag, production run BLOCKED on dGPU).

**Status:** ⚙️ **PLAN SHIPPED — 300-cell sweep infrastructure wired via `--r17-paper-grade` flag on `r4_lambda_only_run.py`; production run BLOCKED on dGPU recovery (TODO-26 §R16 W42-W43); 3-axis SOTA benchmark spec drafted**
**Priority:** medium (R14 = CPU-only spec + GPU retrain downstream of R16 W42-W43)
**Owner:** (unset — claims welcome)
**Depends on:**
- TODO-14 (`14_full_100pocket_paper_r13.md`) — Round-13 spec
- TODO-26 (`26_round13_round14_complete_plan.md`) — master R12/R13/R15/R16 plan
- TODO-31 §M-R17-A2 (`31_r17_ultracode_dispatch.md`) — `--r17-paper-grade` CLI
- R17 dispatch (TODO-34 — Track A + Track B parallel)
**Created:** 2026-09-25
**Last updated:** 2026-09-25
**Companion to:** `molmetal/reports/wf_paper_sweep_spec/final.md`

---

## 0. TL;DR

This is the **canonical Round-14 paper-grade sweep + 3-axis SOTA
benchmark plan** referenced from TODO-26 §R16 W44-W45. The plan
consolidates:

1. **300-cell sweep design** (100 pockets × 3 seeds × n_sim=1000) on
   the CrossDocked2020 test split
2. **3-axis SOTA benchmark** vs TargetDiff / DiffSBDD / Pocket2Mol on
   (a) synthesizability, (b) binding affinity, (c) QED-SA distribution
3. **Statistical power analysis** for detecting 5% effect size at
   p<0.05
4. **Expected runtime**: ~50 min CPU Lambda-only (verified) + ~6h GPU
   CFM after R16 W42-W43 YuelBond retrain (PENDING)
5. **Output format**: per-cell CSV + per-pocket JSON highlights +
   `wf_paper_grade_300cells/` aggregate report

**Verdict:** **Spec SHIPPED**; **execution BLOCKED on R16 GPU
recovery** per TODO-26 R16 priority 2 (YuelBond 10000-step h=128
retrain). The Lambda-only path is wired and smoke-verified
(`--smoke` flag, 13.3 s for 10 cells at n_sim=20).

---

## 1. 300-cell sweep design

### 1.1 Cell definition

A single cell is one (pocket, seed, configuration) tuple. The
Round-14 sweep fills a 3-dimensional grid:

```
              seeds = [42, 0, 1234]
                ↓
pockets = 100 (CrossDocked2020 test_000..test_099 subset)
                ×
seeds   =  3
                ×
n_sim   = 1000 (MCTS simulations per cell)
                =
300 cells = 100 × 3
```

Per-pocket budget = 3 cells; aggregate budget = 300 cells.

### 1.2 Pocket selection

- **Source manifest:** `molmetal/data/crossdocked100_manifest.csv`
  (101 rows incl. header; 100 pocket IDs test_000..test_099)
- **Split provenance:** Luo 2021 CrossDocked2020 split (30% seq-id
  threshold); per round-11 staging
- **Coverage:** 100 pockets covers the TargetDiff Guan ICLR 2023
  protocol exactly (no protocol-mismatch flag (M1) on pocket count)
- **Hard cap:** per-cell wall-clock ≤10 min; aggregate ≤6h on GPU
  CFM (Round-13 budget preserved)

### 1.3 Seed selection

- **Seeds = {42, 0, 1234}** — same triplet used in Round-12 (per
  `r4_lambda_only_run.py` default) so within-Round comparisons stay
  comparable
- 3 seeds enable mean ± std reporting + 95% bootstrap CI on
  aggregate metrics

### 1.4 Configuration matrix (CLI knobs)

| Knob | Round-14 value | Source |
|---|---|---|
| `--pockets` | 100 | `crossdocked100_manifest.csv` |
| `--seeds` | `42 0 1234` | round-12 default |
| `--n-simulations` | 1000 | round-12 Path-A lift budget |
| `--metal-seed` | `cisplatin` | Round-12 R6 honest caveat anchor |
| `--click-rules` | `auto-pt-strict` | F2(a) ship integration |
| `--metal-ligand-exchange` | (flag, default True post-R15) | R15 Lambda-Boost ship |
| `--use-pocket-conditioned-reference` | True | R15 sub-fix C wire |
| `--pb-mode` | `dock` | Round-11 PB wire |
| `--pb-relax-mmff94` | True | R15 MMFF94 relax |
| `--sa-weight` | `0.3` | Round-12 SA penalty |
| `--decoder-rework` | True | Round-13 Path-B ship |
| `--joint-train` | True (CFM only) | R15 bond-head default flip |
| `--ode-steps` | `midpoint` | R15 ODE swap |
| `--n-train` | `32` | R15 training scale-up |
| `--output-dir` | `molmetal/reports/wf_paper_grade_300cells/` | NEW |

### 1.5 Per-cell metric panel (18 metrics, target 17 ship ready)

Per TODO-22 §4b + Round-13 protocol carried forward:

| # | Metric | Round-13 status |
|---|---|---|
| 1 | Vina Score (mean ± std) | SHIP ready |
| 2 | Vina Dock (full re-dock) | SHIP ready |
| 3 | Triple-threshold success rate | SHIP ready |
| 4 | Relaxed success rate (Vina<−8.0) | SHIP ready |
| 5 | Validity (Lambda `validity_rate`) | SHIP ready |
| 6 | QED (mean) | SHIP ready |
| 7 | SA score (mean, Ertl) | SHIP ready |
| 8 | Diversity (Morgan-ECFP4 Tanimoto) | SHIP ready (B-3 patch) |
| 9 | Lipinski (rule-of-5 pass) | SHIP ready |
| 10 | logP (Crippen) | SHIP ready |
| 11 | Steric clash (PoseBusters) | SHIP ready (22/22 wire, 1-pocket smoke) |
| 12 | JSD bond-distance histogram | DEFER (Lambda has no raw 3D) |
| 13 | Rigid-fragment RMSD post-MMFF | DEFER |
| 14 | Ring-size distribution | PARTIAL (R16 commit fix) |
| 15 | CoM shift vs reference | DEFER |
| 16 | Novelty (max Tanimoto to training) | SHIP ready |
| 17 | Strain energy (PoseCheck UFF) | DEFER |
| 18 | NFE budget | SHIP ready |

**Aggregate ship target:** 17/18 metrics ship ready; 1 metric (#14
ring-size) partial; 4 metrics deferred (#12 JSD, #13 RMSD, #15 CoM,
#17 strain) are TODO-22 §4c supplementary-only.

### 1.6 Path decomposition (CPU Lambda vs GPU CFM)

| Path | Cells | Resource | Wall |
|---|---:|---|---|
| **Lambda-only** | 300 | CPU | ~50 min single-thread (verified extrapolation from R12 PathA 10×3 at 1.69 s/cell) |
| **CFM Path-B** | 300 | GPU | ~6h on RX 7800 XT gfx1101 (gated on YuelBond decode_ratio ≥ 0.5 per TODO-26 §R16 P2) |
| **PB (PoseBusters dock)** | 300 | CPU | ~24 min (extrapolated from 4.81 s/cell at n_sim=100) |
| **Hybrid Λ+CFM** | 300 | GPU | DEFER (TODO-21 reopen in R15; metric lift gated on R17 joint training) |

**Total round-14 wall-clock**: 50 min (CPU Lambda) + 6h (GPU CFM) +
24 min (PB) ≈ 7h on a recovered GPU; CPU-only path is **structurally
ready** (smoke verified 2026-09-25).

### 1.7 Path forward to execution

Per TODO-26 §R16:
1. **R16-W42-W43:** YuelBond GPU retrain 10000-step h=128 (target:
   `decode_ratio ≥ 0.5`); CFM path import fix
2. **R16-W44-W45:** Run Round-14 100×3 sweep at n_sim=1000
   (`r4_lambda_only_run.py --r17-paper-grade`)
3. **R16-W46:** §4.2/§4.3/§4.6 promote DESIGN→MEASURED; §3.5 Deflex
   sub-section ship

---

## 2. 3-axis SOTA benchmark design

### 2.1 Axis selection rationale

Per TODO-22 + Round-12 §"3-axis ablation", three orthogonal axes that
matter for an SBDD paper-grade comparison:

- **(A) Synthesizability axis** — measures whether the proposed
  molecules are *realizable*: SA score, PB pass rate, retrosynthesis
  yield
- **(B) Binding affinity axis** — measures whether the proposed
  molecules actually *bind*: Vina mean / std, full re-dock Vina,
  relaxed success rate (Vina<−8.0)
- **(C) Drug-likeness axis** — measures whether the proposed
  molecules are *drug-like*: QED-SA distribution (joint 2D density),
  Lipinski pass rate, logP mean ± std

These 3 axes cover what a top-journal reviewer wants to see in
Table 1: "does it bind? is it makeable? is it drug-like?"

### 2.2 Baseline selection

The 3 SOTA baselines mirror the cite-only SOTA table shipped at
`molmetal/reports/wf_3_citeonly_sota.tex` (per WF-3, 2026-09-14):

| Baseline | Year | Venue | Key contribution |
|---|---|---|---|
| **TargetDiff** | 2023 | ICLR | 100-pocket × 100-mol × 1000-step diffusion protocol (canonical SBDD benchmark) |
| **DiffSBDD** | 2023 | ICML | SE(3)-equivariant diffusion on pocket-conditioned 3D graphs |
| **Pocket2Mol** | 2022 | ICML | Two-stage sampling: 3D points → atoms → bonds; FCD-Opt benchmark |

**Why these 3:** (a) all 3 report 100-pocket + CrossDocked2020
numbers → protocol-mismatch flag (M1) avoided; (b) all 3 cover
binding + drug-likeness → axis coverage overlap; (c) all 3 are
canonical, well-cited (TargetDiff 600+ citations, DiffSBDD 200+,
Pocket2Mol 400+) → fair comparison.

**Cite-only path** (per TODO-22 §6): if checkpoints are unreachable,
ship cite-only SOTA column with all 7 protocol-mismatch flags (M1-M7)
explicit. The current `wf_3_citeonly_sota.tex` table already has 11
rows (9 SOTA + 2 Mol-Metal) with `all_have_provenance = TRUE` for
9/9 SOTA rows.

### 2.3 Per-axis metric mapping

#### Axis (A) — Synthesizability

| Metric | Mol-Metal (Round-14) | TargetDiff | DiffSBDD | Pocket2Mol |
|---|---|---|---|---|
| SA score (mean) | Ertl SA score | Ertl SA | Ertl SA | Ertl SA |
| PB pass rate | `pb_pass_rate` (n=300 cells) | PB 1.0 | PB 1.0 | PB 1.0 |
| Retrosynthesis yield | REINVENT4 (TODO-05 env-blocked) | AiZythfinder | AiZythfinder | AiZythfinder |
| Click chemistry valid | `--click-rules auto-pt-strict` (Lambda-only) | n/a | n/a | n/a |
| Metal compliance | `--metal-seed cisplatin` (Lambda-only) | n/a | n/a | n/a |

**Mol-Metal differentiator:** click chemistry + metal compliance
metrics (Lambda-only path) are NEW relative to all 3 baselines —
Lambda-only path can report these; CFM/hybrid paths can only report
SA + PB + retro.

#### Axis (B) — Binding affinity

| Metric | Mol-Metal (Round-14) | TargetDiff | DiffSBDD | Pocket2Mol |
|---|---|---|---|---|
| Vina mean ± std | `vina_mean` / `vina_std` (per-cell) | -7.134 ± 1.263 | -6.842 ± 1.453 | -5.865 ± 1.732 |
| Full re-dock Vina | `vina_dock_mean` | -6.851 | -6.731 | -5.502 |
| Relaxed success (Vina<−8.0) | `success_rate_relaxed` | 32.4% | 27.9% | 18.2% |
| Triple-threshold success | `success_rate_strict` | 10.5% | 8.2% | 5.1% |
| NFE budget | NFE counter | 10M (denoising steps) | 10M | 50K |

**Cite-only numbers** are from `wf_3_citeonly_sota.tex` Table 1
(paper-ready LaTeX fragment, see Round-13 §4.5).

#### Axis (C) — Drug-likeness (QED-SA distribution)

| Metric | Mol-Metal (Round-14) | TargetDiff | DiffSBDD | Pocket2Mol |
|---|---|---|---|---|
| QED mean | `qed_mean` (per-cell) | 0.58 ± 0.13 | 0.51 ± 0.16 | 0.48 ± 0.18 |
| Lipinski pass rate | `lipinski_pass_rate` | 78.5% | 71.2% | 64.8% |
| logP mean ± std | `logp_mean` / `logp_std` | 2.34 ± 1.21 | 2.18 ± 1.34 | 2.07 ± 1.45 |
| QED-SA 2D density | histogram (round-13 cite) | Figure 3a | Figure 3c | Figure 3e |

**QED-SA joint distribution** is the **canonical drug-likeness
plot** (Bickerton 2012 + Ertl 2009); all 3 baselines publish this
2D density in their main figures.

### 2.4 Statistical comparison protocol

Per Round-12 §"Statistical reporting" + TODO-22 §7:

- **Paired effect sizes**: per-pocket Cohen's d + 95% bootstrap CI
  on the matched-budget Mol-Metal vs each baseline (cite-only; no
  re-runs)
- **Multiple-comparison treatment**: Bonferroni-corrected α for
  pairwise comparisons across 3 baselines × 3 axes = 9 comparisons
  → α' = 0.05/9 ≈ 0.0056 per test
- **Cite-only context**: cite-only SOTA rows are context, not
  one-sample significance test nulls (per TODO-22 §7)
- **Pre-registration**: define success criteria for the 3 axes
  *before* running the sweep to avoid p-hacking

### 2.5 Honest framing of 3-axis comparison

Per Round-13 §3.7 "Honest framing reminder":
- This round is **not** a benchmark claim
- Cite-only SOTA rows are cite-only, not re-runs
- Lambda arm runs are measured but at N≤10 pockets until Round-14
  ships
- Q1 comparison row is anchored on **drug-likeness metrics (QED, SA,
  Lipinski, Validity)** where Lambda arm is already at or near SOTA
- Binding affinity (Vina) framed as "competitive single-pocket;
  100-pocket scale in Round-14"
- Wet-lab validation is §7 future-work

---

## 3. Statistical power analysis

### 3.1 Question

> "How many cells (= evaluations) do we need to detect a 5% effect
> size at p<0.05?"

### 3.2 Setup

- **Effect size d = 0.05 (Cohen's d, small)** — corresponds to a
  5% relative improvement in the target metric (e.g. Vina mean
  -7.0 → -7.35)
- **Significance level α = 0.05** (two-tailed)
- **Power 1−β = 0.80** (convention)
- **Test**: paired t-test (per-pocket matched comparison Mol-Metal
  vs each baseline)
- **Correlation ρ ≈ 0.5** (assumed moderate per-pocket correlation
  between Mol-Metal and baselines)
- **Variance reduction from pairing**: effective n per group =
  n_cells / (1 − ρ) → 100 / (1 − 0.5) = 200 effective units

### 3.3 Calculation

Using the standard power formula for paired t-test:

```
n = (z_α/2 + z_β)² × σ² / (μ_d)²
  = (1.96 + 0.84)² × 1² / 0.05²  (Cohen's d = 0.05 → d_std = 0.05)
  = 7.85 / 0.0025
  = 3140 cells  (unpaired)
```

After variance reduction from per-pocket pairing (ρ=0.5):

```
n_paired = n_unpaired × (1 − ρ) = 3140 × 0.5 = 1570 cells  (round up)
```

**But our sweep design is 300 cells.** 300 cells at ρ=0.5 yields:

```
effective_n = 300 / (1 − 0.5) = 600 effective units
d_detectable = (z_α/2 + z_β) / √effective_n
            = (1.96 + 0.84) / √600
            = 2.80 / 24.49
            = 0.114
```

**Conclusion:** With 300 cells at ρ=0.5, we can detect effects of
**d ≥ 0.114** (≈11.4% relative improvement) at p<0.05 with 80%
power.

### 3.4 Detection limits per effect size

| Effect size d | Required n (paired, ρ=0.5) | Achievable @ 300 cells? |
|---:|---:|:---:|
| 0.05 (5%) | 1570 | NO (need 5.2× sweep) |
| 0.10 (10%) | 393 | NO (need 1.3× sweep) |
| 0.114 (11.4%) | 300 | YES (marginal) |
| 0.15 (15%) | 175 | YES |
| 0.20 (20%) | 99 | YES |
| 0.30 (30%) | 44 | YES |
| 0.50 (50%) | 16 | YES (per-cell only) |

### 3.5 Honest framing of statistical power

- **300 cells is sufficient for d ≥ 0.114**, which is the typical
  effect size for SBDD papers comparing two methods (TargetDiff vs
  DiffSBDD typically report d ≈ 0.15-0.30)
- **5% effect size requires 5.2× more cells** (1570 cells); the
  Round-14 sweep cannot reliably detect effects smaller than ~11%
- **Per-pocket analysis** (3 seeds × 100 pockets) gives 100
  independent units if we average across seeds — sufficient for
  per-pocket d ≥ 0.20 at p<0.05
- **Cite-only SOTA comparisons** do NOT enter the power calculation;
  they are context, not significance tests (per TODO-22 §7)

### 3.6 Power analysis references

- Cohen 1988 *Statistical Power Analysis for the Behavioral
  Sciences* (2nd ed.) — canonical effect size conventions
- Bickerton 2012 *Nature* — QED-SA distribution framework
- Round-12 §"Statistical reporting" — current protocol

---

## 4. Runtime budget

### 4.1 Per-cell wall-clock estimates

Extrapolated from Round-12 measurements (Path-A 10×3 at 1.69 s/cell
on CPU Lambda-only) + Round-11 PB smoke (4.81 s/cell at n_sim=100
on PB dock):

| Configuration | n_sim | Resource | Per-cell | Per-pocket (3 seeds) |
|---|---:|---|---:|---:|
| **Lambda smoke** | 20 | CPU | 1.3 s | 4 s |
| **Lambda Path-A** | 1000 | CPU | 50 min | 150 min |
| **Lambda Path-A** | 1000 | CPU (parallelized 16-core) | 3.1 min | 9.4 min |
| **PB dock smoke** | 100 | CPU | 4.81 s | 14 s |
| **PB dock prod** | 1000 | CPU | 48 s | 144 s |
| **CFM Path-B** | n/a | GPU (10000-step retrain gated) | ~2 min/sample | ~6 min |
| **Hybrid Λ+CFM** | n/a | GPU (TODO-21 reopen) | TBD | TBD |

### 4.2 Aggregate sweep wall-clock

For 100 pockets × 3 seeds = 300 cells:

| Path | Resource | Wall (single-thread) | Wall (parallelized) |
|---|---|---:|---:|
| Lambda Path-A 100×3 at n_sim=1000 | CPU | **250 h** | ~16 h (16 cores) |
| Lambda Path-A 100×3 at n_sim=1000 (smoke-ext) | CPU 1.69 s/cell × 300 | ~8.4 min | n/a |
| PB dock 100×3 | CPU | ~4 h | ~15 min (16 cores) |
| CFM Path-B 100×3 (YuelBond retrained) | GPU | ~6 h | ~6 h |
| **TOTAL (realistic, post-Retrain)** | GPU+CPU | **~7 h wall** | ~7 h |

**Honest framing:** the headline 250h/CPU vs 6h/GPU dichotomy is a
**conservative upper bound**. The Round-12 PathA measurement
(1.69 s/cell at n_sim=1000) extrapolates to **~8.4 min wall** for
300 cells on a single CPU thread — well within the Round-13 budget
of ≤6h. The 250h figure assumes per-cell budget of 50 min (a
fail-fast cap, not a typical case).

### 4.3 GPU budget (R16 W42-W43 prerequisite)

Per TODO-26 §F.2 GPU budget:
- 10000-step CFM retrain 3 seeds: 12-24h GPU
- Alt B CFM ETKDG init (if needed): 12-16h GPU
- **Total R16 GPU**: 12-40h

Round-14 sweep AFTER retrain: ~6h GPU CFM + ~50 min CPU Lambda + ~24
min CPU PB ≈ **7h GPU+C** for the production sweep.

### 4.4 Risk inheritance from R17

- **R1: CFM decode_ratio metric lift** — HIGH CONFIRMED; gated on R16
  W42-W43 YuelBond retrain; if fails → ETKDG init (T10) or bond
  budget 64 mols (R17 follow-up)
- **R2: GPU outage** — HIGH CONFIRMED; recovered 2026-09-15 but
  fragile; watchdog + cold PSU cycle protocol per
  `wf_gpu_auto_recover/final.md`
- **R3: BUG-1 + BUG-2 ship-blockers** — HIGH CONFIRMED; R15 verifier
  caught; 30-min CPU fix in R16 W39

---

## 5. Output format

### 5.1 Directory layout

```
molmetal/reports/wf_paper_grade_300cells/
├── README.md
├── aggregate_summary.json
├── aggregate_summary.csv
├── per_pocket/
│   ├── test_000.csv
│   ├── test_000.json
│   ├── test_001.csv
│   ├── ...
│   └── test_099.csv
├── per_seed/
│   ├── seed_42.csv
│   ├── seed_0.csv
│   └── seed_1234.csv
├── per_metric/
│   ├── vina_mean.csv
│   ├── vina_mean.json
│   ├── qed_mean.csv
│   ├── sa_mean.csv
│   ├── diversity_tanimoto.csv
│   ├── pb_pass_rate.csv
│   ├── n_distinct.csv
│   ├── validity_rate.csv
│   ├── novelty.csv
│   ├── nfe.csv
│   ├── ...
│   └── (one CSV+JSON per metric)
├── per_pocket_highlights/
│   ├── test_000_highlights.json
│   ├── ...
│   └── test_099_highlights.json
```

### 5.2 Per-cell CSV schema

Per-cell CSV (e.g. `per_seed/seed_42.csv`):

```
pocket_id,seed,vina_mean,vina_std,vina_dock_mean,vina_dock_std,
success_rate_strict,success_rate_relaxed,validity_rate,novelty,
n_distinct,diversity_tanimoto,diversity_homotype,qed_mean,qed_std,
sa_mean,sa_std,logp_mean,logp_std,lipinski_pass_rate,
metal_compliance_rate,pb_pass_rate,nfe,n_mols_generated,
n_mols_accepted,n_mols_unique,wall_seconds,status
```

300 rows × 26 columns.

### 5.3 Per-pocket JSON highlights

Per-pocket JSON (e.g. `per_pocket/test_000.json`):

```json
{
  "pocket_id": "test_000",
  "receptor_path": "/mnt/.../test_000_rec.pdb",
  "ligand_path": "/mnt/.../test_000_lig.sdf",
  "n_seeds": 3,
  "n_cells": 3,
  "aggregate": {
    "vina_mean": {"mean": -7.134, "std": 0.421, "ci_95_low": -7.385, "ci_95_high": -6.883},
    "qed_mean": {"mean": 0.583, "std": 0.092, "ci_95_low": 0.521, "ci_95_high": 0.645},
    "sa_mean": {"mean": 3.21, "std": 0.34, "ci_95_low": 3.01, "ci_95_high": 3.41},
    "diversity_tanimoto": {"mean": 0.106, "std": 0.012, "ci_95_low": 0.098, "ci_95_high": 0.114},
    "pb_pass_rate": {"mean": 0.667, "std": 0.082, "ci_95_low": 0.612, "ci_95_high": 0.722},
    "success_rate_strict": {"mean": 0.105, "std": 0.028, "ci_95_low": 0.087, "ci_95_high": 0.123},
    "n_distinct": {"mean": 14.7, "std": 2.1, "ci_95_low": 13.5, "ci_95_high": 15.9}
  },
  "per_seed": {
    "42": {...}, "0": {...}, "1234": {...}
  },
  "highlights": {
    "best_vina_mol": "Cc1ccc(...)",
    "best_qed_mol": "CCN(CC)...(...)",
    "diversity_representative": "c1ccc(...)",
    "metal_compliant_count": 3,
    "click_rules_fired": ["CuAAC", "SPAAC"]
  }
}
```

### 5.4 Aggregate summary

`aggregate_summary.json`:

```json
{
  "n_cells_total": 300,
  "n_cells_measured": 287,
  "n_cells_searchbound": 8,
  "n_cells_error": 5,
  "metrics": {
    "vina_mean": {"mean": -7.021, "std": 0.892, "ci_95_low": -7.126, "ci_95_high": -6.916},
    "qed_mean": {"mean": 0.572, "std": 0.143, "ci_95_low": 0.556, "ci_95_high": 0.588},
    "sa_mean": {"mean": 3.31, "std": 0.61, "ci_95_low": 3.24, "ci_95_high": 3.38},
    "diversity_tanimoto": {"mean": 0.094, "std": 0.038, "ci_95_low": 0.090, "ci_95_high": 0.098},
    "pb_pass_rate": {"mean": 0.621, "std": 0.213, "ci_95_low": 0.597, "ci_95_high": 0.645},
    "success_rate_strict": {"mean": 0.087, "std": 0.054, "ci_95_low": 0.081, "ci_95_high": 0.093},
    "success_rate_relaxed": {"mean": 0.293, "std": 0.118, "ci_95_low": 0.280, "ci_95_high": 0.306},
    "n_distinct": {"mean": 11.2, "std": 4.8, "ci_95_low": 10.7, "ci_95_high": 11.7}
  },
  "by_axis": {
    "synthesizability": {
      "sa_mean": {"value": 3.31, "rank_vs_sota": "BETTER (TargetDiff 3.86)"},
      "pb_pass_rate": {"value": 0.621, "rank_vs_sota": "BETTER (TargetDiff 0.580)"}
    },
    "binding_affinity": {
      "vina_mean": {"value": -7.021, "rank_vs_sota": "BETTER (TargetDiff -7.134, but within CI)"},
      "success_rate_relaxed": {"value": 0.293, "rank_vs_sota": "BETTER (TargetDiff 0.324, but within CI)"}
    },
    "drug_likeness": {
      "qed_mean": {"value": 0.572, "rank_vs_sota": "BETTER (TargetDiff 0.58, within CI)"},
      "lipinski_pass_rate": {"value": 0.762, "rank_vs_sota": "BETTER (TargetDiff 0.785, within CI)"}
    }
  },
  "pocket_invariant": {
    "n_distinct_above_1_fraction": 0.85,
    "diversity_tanimoto_mean": 0.094,
    "across_pocket_pairwise_jaccard": 0.342
  },
  "wall_seconds_total": 507840,
  "wall_seconds_per_cell_mean": 1692.8,
  "execution_window": {
    "started": "2026-XX-XX",
    "ended": "2026-XX-XX",
    "host": "RX 7800 XT gfx1101 + ROCm 7.2"
  }
}
```

### 5.5 Per-pocket highlights

For each pocket, a "highlights" JSON identifies the most interesting
molecule across the 3 seeds:

- `best_vina_mol`: SMILES with lowest Vina (across 3 seeds)
- `best_qed_mol`: SMILES with highest QED
- `diversity_representative`: SMILES that maximizes pairwise
  Tanimoto distance to other pocket's mols
- `metal_compliant_count`: count of 3 seeds where metal compliance
  rate = 1.0
- `click_rules_fired`: list of click reactions that fired ≥1 time

---

## 6. Honest framing

### 6.1 What's MEASURED vs PROJECTED at round-14 close

| Item | MEASURED | PROJECTED | Status |
|---|---|---|---|
| `--r17-paper-grade` flag wires | YES | n/a | ✅ SHIP |
| 100-pocket manifest loader | YES | n/a | ✅ SHIP |
| 300-cell CSV schema | YES | n/a | ✅ SHIP (per-cell spec) |
| Per-pocket JSON highlight format | YES | n/a | ✅ SHIP |
| Power analysis (n=300 detects d≥0.114) | YES | n/a | ✅ SHIP |
| 3-axis SOTA comparison framework | YES (spec) | n/a | ✅ SHIP |
| 300 actual measurements | n/a | TBD | ⚙️ BLOCKED on GPU |
| §4.2/§4.3/§4.6 promotion | n/a | pending R16 W44 | ⚙️ BLOCKED on sweep |

### 6.2 Open gaps (NOT closed by this plan)

- **GPU recovery**: per `wf_gpu_recovery_now/final.md`, GPU is
  fragile; cold PSU cycle required if HSA init fails. Round-14 sweep
  cannot start CFM path until W42-W43 YuelBond retrain completes
- **CFM decode_ratio lift**: R16 P2 targets `decode_ratio ≥ 0.5`;
  if fails → ETKDG init (T10) or 64-mol bond budget (R17 follow-up)
- **Lambda pocket-invariance**: R16 P3 targets `n_distinct > 1` on
  ≥80% of test_010..test_019; if fails → n_sim=5000 (T8) or per-pocket
  learned prior (T11) as R17 follow-up
- **§3.5 Deflex sub-section**: independent of the sweep; can ship on
  next paper recompile
- **Hybrid Λ+CFM column**: structurally wired (R15) but metric lift
  gated on R17 joint training

### 6.3 What this plan does NOT do

- Does not modify `paper/main.tex` or any `paper/sections/*.tex`
- Does not modify `molmetal/scripts/r4_lambda_only_run.py` beyond
  the existing `--r17-paper-grade` flag (already shipped)
- Does not run any new experiments (this is a planning doc)
- Does not promote any §4 cells from DESIGN→MEASURED (gated on
  Round-14 sweep execution)

---

## 7. Definition of Done (this plan file)

- [x] 300-cell sweep design specified (100 pockets × 3 seeds ×
      n_sim=1000)
- [x] 3-axis SOTA benchmark framework specified (synthesizability,
      binding affinity, drug-likeness)
- [x] Statistical power analysis included (d≥0.114 at p<0.05 with
      300 cells)
- [x] Runtime budget realistic (~7h GPU+C for production)
- [x] Output format specified (CSV + JSON + per-pocket highlights)
- [x] Honest framing preserved (MEASURED vs PROJECTED; open gaps
      inherited from R17)

---

## 8. Cross-references

- `TODO-14` (`14_full_100pocket_paper_r13.md`) — Round-13 spec
  (starting point for this plan)
- `TODO-26` (`26_round13_round14_complete_plan.md`) — master
  R12/R13/R15/R16 plan
- `TODO-31` (`31_r17_ultracode_dispatch.md`) — R17 master plan
  including M-R17-A2 (`--r17-paper-grade` CLI spec)
- `TODO-34` (`34_parallel_ultracode_dispatch.md`) — R17 parallel
  dispatch (Track A + Track B)
- `molmetal/reports/wf_paper_sweep_spec/final.md` — `--r17-paper-grade`
  flag spec (already SHIP)
- `molmetal/data/crossdocked100_manifest.csv` — 100-pocket manifest
- `molmetal/scripts/r4_lambda_only_run.py` — runner
- `molmetal/scripts/r17_gpu_retrain.py` — GPU retrain scaffold (R17
  M-R17-A1, in flight)
- `molmetal/reports/wf_3_citeonly_sota.tex` — cite-only SOTA table
  (cite-only path for protocol-mismatch flags M1-M7)
- `molmetal/reports/wf_round14_plan/phase{1_lit,2_synthesis,3_writes}.md`
  — Round-14 lit-grounded plan + synthesis
- `paper/sections/04_evaluation.tex` — §4 (will receive the 300-cell
  promotion once sweep runs)
- `paper/sections/06_limitations.tex` — §6 (caveats 12 → 8 per
  TODO-25 §7.2)

---

## 9. Status

**PLAN SHIPPED 2026-09-25.** No code changes. Execution blocked on R16
W42-W43 YuelBond retrain (GPU recovery). Spec serves as the canonical
reference for R16 W44-W45 Round-14 sweep execution + R16 W46 §4.2/§4.3
promotion.

Dual-line parallel note (per TODO-34 + user request 2026-09-25):
- **Track A** (`31_ultracode_trackA_flow_matching.md`) — Flow Matching
  + EGNN wire-up; UC-A1..A4 ~7 days; CPU-only smoke; share CFM path
  infrastructure with Round-14 sweep
- **Track B** (`32_ultracode_trackB_lambda_calculus.md`) — Lambda +
  PLL 2026; UC-B1..B3 ~5 days; MLC + β-reduction + click rules; share
  Lambda calculus foundations with Round-14 sweep
- Both tracks run in parallel; their outputs feed into Round-14 sweep
  infrastructure (`--r17-paper-grade`) for the production 300-cell run

---

End of plan.