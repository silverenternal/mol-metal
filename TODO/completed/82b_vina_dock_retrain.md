# TODO-82b — Wf-30 Phase B Vina Dock retrain synthesis

**Date**: 2026-09-26
**Workflow**: Wf-30 (Phase 1 staging + Phase 3 sweep + Phase 4 lift analysis)
**Status**: ⚠️ **MEASURED_BUT_STUB_DOMINATED** — 250 evaluations completed, m3 median `-9.6574 kcal/mol`, but lift vs baseline is **+1.3e-08 kcal/mol** (≈ 0; floating-point noise from stub scorer).
**Parent**: `TODO/pending/82_metric_gap_filling_plan.md` §2 Phase B + `TODO/completed/82_metric_gap_filling.md` §2.2 + `metrics/by_metric/vina_kcal_per_mol.json`
**Honest verdict** (party-twenty-spirit §3.5): the headline ship gate (≤ -8.5 kcal/mol) is **technically met** at -9.6574 kcal/mol, but the pass is a **property of the deterministic stub floor**, not of either method. No quantitative lift can be claimed from this sweep. Real-Vina path on the metal cohort is blocked by `meeko 0.8.0` rejecting Pt / Au / Ru atoms. Stub result is **honest-negative** for Wf-30's stated claim; this is **R20 P1** territory.

---

## §1 TL;DR — 10 pockets × 5 candidates × 5 seeds = 250 dockings

| Item | Value |
|---|---|
| Sweep size | 10 pockets × 5 candidates × 5 seeds × 2 methods (se3sb + baseline) = **500 evaluations** |
| m3 Vina Dock median (se3sb) | **-9.6574 kcal/mol** (`metrics/by_metric/vina_kcal_per_mol.json`, status `MEASURED_BUT_STUB`) |
| m3 Vina Dock median (baseline) | **-9.6574 kcal/mol** (same floor, identical per-pocket best to ~1e-7 kcal/mol) |
| Lift (se3sb − baseline) | **+1.2963e-08 kcal/mol** (≈ 0; floating-point noise) |
| Scorer | `stub` (deterministic ±2.0 kcal/mol jitter on init-centroid distance; `use_real_vina=false`) |
| Real Vina path | **0/250 metal-cohort dockings succeeded** — meeko 0.8.0 rejects Pt/Au/Ru (`UFFTYPER: Unrecognized atom type: Pt+2 (0)`) |
| Wall-time | ≤ 2.1 s per 250-eval sweep (CPU-only, parameter-free stub velocity field) |
| Honest framing | **Stub-floor artifact, not a method-level result** → honest-negative → R20 P1 |

**One-sentence verdict**: 250 dockings executed end-to-end; the se3sb and baseline methods produce **identical** per-pocket best kcal/mol to floating-point noise, which is consistent with the parameter-free stub velocity field offering no signal at this exhaustiveness level. The ship gate pass is non-discriminating. **No quantitative method-level lift is claimed.**

---

## §2 Per-deliverable

### §2.1 Wf-30 Phase 1 — Pocket receptor staging (SHIP)

**Script**: `molmetal/scripts/check_10_pockets.py` → `mk_prepare_receptor.py` (meko CLI, MGLTools-derived)
**Output**: `molmetal/reports/wf30_pocket_prep/{pocket}/receptor.pdbqt` for 10/10 pockets
**Status**: **10/10 READY** (830c, 1h36, 3eml, 3ny8, 4aaw, 4rlu, 1g9v, 2hnx, 4rn0, 1fmc)
**Wall-time**: ~25 s total (4.1 s for 830c, 1.7–2.2 s per CrossDocked cut)
**Per-pocket provenance**: SHA-256 + bytes + source path in `pocket_status.json` (120 lines log in `pocket_prep_log.md`)
**Special handling**: `4rlu` requires `--set_template A:11=HIE,...,D:58=HIE` (12 histidines; meeko cannot resolve HIE/HID/HIP tie)

| Pocket | Size (B) | Source | Status |
|---|---:|---|---|
| `830c` | 344560 | mmp13_real | READY |
| `1h36` | 55680 | CrossDocked SQHC | READY |
| `3eml` | 32880 | CrossDocked AA2AR | READY |
| `3ny8` | 44240 | CrossDocked ADRB2 | READY |
| `4aaw` | 38880 | CrossDocked GLMU | READY |
| `4rlu` | 511280 | pdbbindpp-2020 (HCC) | READY (HIE-forced) |
| `1g9v` | 33520 | CrossDocked HBA | READY |
| `2hnx` | 42320 | CrossDocked FABP4 | READY |
| `4rn0` | 22880 | CrossDocked HDAC8 (Zn2+) | READY |
| `1fmc` | 38160 | CrossDocked HDHA | READY |

### §2.2 Wf-30 Phase 3 — Dock script + sweep (SHIP, but stub path)

**Script**: `molmetal/scripts/dock_10pockets_se3sb.py` (~998 LOC)
**New script (this wf)**: yes — adds SE(3) Schrödinger-Bridge initial pose generator + 10-pocket driver
**Summarizer**: `molmetal/scripts/wf30_dock_summarize.py` (~433 LOC)
**Verify**: `molmetal/scripts/verify_dock_10pockets.py` (CPU smoke, 1 pocket × 1 cand × 2 seeds)
**Sweep output**: `molmetal/reports/wf30_dock/{se3sb,baseline}/metrics.json` (250 raw evaluations each)
**Aggregate**: `best_m3_summary.json`, `per_pocket_metrics.json`, `comparison.md`, `m3_lift_analysis.json`

### §2.3 Wf-30 GPU 1 run — actual dispatch

**GPU**: not used (CPU-only; the SE(3) SB integrator is parameter-free, no torch gradient needed)
**Wall-time**: 2.1 s per 250-eval sweep (real Vina would have been 4–12 h on metal cohort with NaN returns)
**Decision**: real-Vina path was **skipped on the metal cohort** because meeko rejects Pt/Au/Ru (would produce `n_docked=0`). A 50-eval real-Vina smoke on the **organic cohort** (CCO/benzene/toluene/TEA/benzamide) at 830c **did** dock 50/50 in ~1.3 s/docking → proves the adapter works end-to-end. See `molmetal/reports/wf30_dock/real_vina_organic/`.

### §2.4 Wf-30 Phase 4 — m3 lift analysis (SHIP, honest-negative)

**Output**: `molmetal/reports/wf30_dock/m3_lift_summary.md` (100 lines), `m3_lift_analysis.json`
**Method**: median of 10 per-pocket best (most negative) kcal/mol values across the 5-candidate × 5-seed sweep.
**Headline numbers**: se3sb -9.6574 vs baseline -9.6574, lift +1.3e-08 kcal/mol. Per-pocket deltas in §4 below.

---

## §3 MEASURED vs PROJECTED

### §3.1 MEASURED — what is on disk

| Item | Value | Source |
|---|---:|---|
| m3 Vina Dock median (se3sb, stub) | **-9.6574 kcal/mol** | `molmetal/reports/wf30_dock/best_m3_summary.json` |
| m3 Vina Dock median (baseline, stub) | **-9.6574 kcal/mol** | same |
| n_evaluations per method | 250 (10 × 5 × 5) | `se3sb/metrics.json` + `baseline/metrics.json` |
| n_pockets | 10 | `pocket_status.json` |
| Scorer | `stub` | `per_pocket_metrics.json` field |
| Status label | `MEASURED_BUT_STUB` | `metrics/by_metric/vina_kcal_per_mol.json` |

### §3.2 PROJECTED — what we wanted to measure

| Comparison | Target / Reference | Naive delta | Caveat |
|---|---:|---:|---|
| R17 baseline | -1.2885 kcal/mol (`metrics/by_metric/vina_score.json`, real Vina, 830c, 100 CuAAC click products) | -8.37 kcal/mol | **NOT comparable**: stub scorer vs real Vina; 10 pockets vs 830c; metal-drugs vs CuAAC |
| SOTA (TransDiffSBDD, CrossDocked) | -9.37 kcal/mol (Guan 2024) | +0.29 kcal/mol | stub formally meets it but inside stub determinism — not a real-method pass |
| Ship gate (TODO-82 §6) | ≤ -8.5 kcal/mol | -1.16 kcal/mol | stub-pass, but pass is non-discriminating (both methods produce same value) |
| Real-Vina organic cohort smoke | -3.915 kcal/mol mean (830c, n=25, real Vina, exhaustiveness=4) | n/a | proves adapter plumbing works; not the headline number |

### §3.3 Honest framing — what the comparison actually means

The "PROJECTED -8.5 kcal/mol" lift was a TODO-82 Phase-B target assuming real Vina on a discrimination-capable candidate pool. With the **stub scorer** + **parameter-free stub velocity field** + **meeko-blocked metal cohort**, no lift is measurable:

- The stub scorer returns `-9.66 kcal/mol` as its deterministic floor (uniform jitter lower bound); both methods bottom out at the same value.
- The W2.1 SE(3) twist is zero during training; the trajectory carries no per-atom velocity signal.
- Real-Vina exhaustiveness=4 makes the SE(3) SB initial pose less relevant (search is thorough).

**Conclusion**: the 250-eval sweep is *executed* and *on disk*, but it is **not** a real-method result. It is a **stub-scorer property**.

---

## §4 se3sb vs baseline lift — actual numbers (per pocket)

| Pocket | se3sb best | baseline best | Δ (se3sb − baseline) |
|---|---:|---:|---:|
| 1fmc | -9.657403 | -9.657404 | +8.9035e-08 |
| 1g9v | -9.657403 | -9.657403 | +1.6318e-08 |
| 1h36 | -9.657404 | -9.657404 | +8.0356e-08 |
| 2hnx | -9.657403 | -9.657403 | +4.9593e-08 |
| 3eml | -9.657403 | -9.657403 | +7.9658e-08 |
| 3ny8 | -9.657404 | -9.657404 | +7.6646e-11 |
| 4aaw | -9.657404 | -9.657404 | +1.8333e-07 |
| 4rlu | -9.657404 | -9.657404 | -3.8840e-08 |
| 4rn0 | -9.657403 | -9.657403 | -5.8236e-08 |
| 830c | -9.657404 | -9.657404 | +5.4221e-10 |
| **Median** | **-9.657404** | **-9.657404** | **+1.2963e-08** |

**Interpretation**: every per-pocket best agrees to ~1e-7 to 1e-11 kcal/mol. Mean ± std = -9.657 ± 0.000 kcal/mol for both methods. Win/tie/loss = 0/10/0. **The se3sb trajectory offers no quantitative signal at this exhaustiveness level.**

The real-Vina organic cohort smoke (830c, 50 evals, 5 organic molecules × 10 seeds ≈ 25/method) shows the same pattern: se3sb mean = baseline mean = -3.915 kcal/mol, min = -5.023 kcal/mol. See `molmetal/reports/wf30_dock/notes.md` §"Honest-negative finding".

---

## §5 Honest framing — what counts as MEASURED for Wf-30

Per `party-twenty-spirit` §3.5 (0 容忍 + 不掩盖):

| Claim | Status | Evidence |
|---|---|---|
| 250 dockings executed | ✅ MEASURED | `se3sb/metrics.json` (250 records), `baseline/metrics.json` (250 records) |
| 10 pockets × 5 seeds × 5 candidates sweep completed | ✅ MEASURED | `pocket_status.json` + `best_m3_summary.json` |
| Receptor staging 10/10 READY | ✅ MEASURED | `pocket_prep_log.md`, SHA-256 per pocket |
| m3 Vina Dock median = -9.6574 kcal/mol | ✅ MEASURED (but stub scorer) | `metrics/by_metric/vina_kcal_per_mol.json` |
| m3 lift se3sb − baseline = -8.5 or more | ❌ **NOT MEASURED** — stub-floor artifact | per-pocket best agree to 1e-8 kcal/mol |
| Real Vina on metal cohort | ❌ **NOT EXECUTED** — meeko blocker | `notes.md` §"Why we didn't run real Vina on the full 250-eval metal-cohort sweep" |
| SE(3) SB initial-pose gives real-Vina lift | ❌ **honest-negative** — adapter works, trajectory gives no signal | organic-cohort smoke 830c se3sb = baseline |

### §5.1 Why this is honest-negative (not a ship gate pass)

The TODO-82 §6 acceptance criterion was "m3 Vina Dock median: -1.2885 → **-8.5 以上** (PROJECTED) 或 honest-neg". The stub scorer produced **-9.6574**, which technically meets the gate. **But**:

- The -9.6574 value is the **stub floor**, not a method result. Both se3sb and baseline hit it identically.
- A pass that doesn't discriminate methods is **not** a real-method pass; it cannot be cited as evidence of algorithm lift.
- Per party-twenty-spirit §3.5, **no fabricated MEASURED lift is reported**. The stub result is recorded truthfully with `scorer="stub"` and `use_real_vina=false` flags; this report labels the result honest-negative.

### §5.2 What would be required to claim a real-method lift (R20 P1)

1. **Train W2.1 EGNN velocity field on the demo cohort** so the SE(3) SB trajectory has per-atom velocity signal (not the parameter-free stub).
2. **Re-run at Vina exhaustiveness ≤ 2** so the initial pose actually matters.
3. **Extend meeko (or use PlatinAI / MolSimplify)** to handle Pt / Au / Ru so real Vina can run on the demo metal cohort.
4. **Or**: switch the headline candidate set to the organic cohort (CCO / benzene / toluene / TEA / benzamide) where meeko already works — 250 evals ≈ 25 min on RTX 5090.

---

## §6 Unmodified commitments (Wf-30 scope discipline)

Per the Wf-30 task brief, only two code paths are modified:

| Path | Modification | Lines | Justification |
|---|---|---|---|
| `molmetal/scripts/train_se3sb.py` | 1-line bug fix (Wf-29 closure of `args.se3_use_sinkhorn` typo carried forward from Wf-23) | 1 | TODO-82 Phase A blocker; Wf-29 confirmed already fixed at `:613` during Wf-23 |
| `molmetal/scripts/dock_10pockets_se3sb.py` | **NEW** (~998 LOC) | 998 | Wf-30 Phase 3 sweep driver — SE(3) SB pose generator + 10-pocket dispatch + stub scorer + per-(pocket, method, seed) JSON writer |

**Everything else is unchanged**: no modifications to `train_ru_5metal_se3sb.py`, `train_cisplatin_resistance.py`, `train_multiph.py`, `train_affinity_regression.py`, `train_metal_lomo.py`, or any of the R18 / R17 module code. Wf-30 deliberately stays narrow to the m3 Vina Dock metric — no scope creep into m1 / m2 / m7 / m16 / m19 (those are Wf-31 / Wf-32 territory).

**What was NOT modified**:
- No R18 module code (T1 SE(3) SB / T2 transport / T3 multi-PH / T4 Hilbert).
- No metric JSONs other than `vina_kcal_per_mol.json` (the canonical m3 file).
- No paper sections (Wf-33 territory).
- No skill code (Wf-34 territory; `mol-metal-gpu-execution` registered separately).

---

## §7 Cross-reference

### §7.1 Parent plan

- **`TODO/pending/82_metric_gap_filling_plan.md`** §2 Phase B (Wf-30) — this file is the execution evidence for that plan section.
- **`TODO/completed/82_metric_gap_filling.md`** §2.2 (Wf-30 high-level summary) + §5 R20 P1 (where G1 m3 lands in R20).
- **`TODO/03_plans/WEAK_TO_STRONG_PLAN.md`** §1.2 G1 m3 — the metric ledger this synthesis updates.
- **`TODO/03_plans/R19_PLAN.md`** §6-12 — R19 GPU training schedule (Wf-30 is one of the execution workflows).

### §7.2 Sibling workflows

| Workflow | Scope | Status | Output |
|---|---|---|---|
| **Wf-29** | fix train_se3sb.py + 4-script smoke | partial (4 scripts SHIP, 0 GPU runs) | `molmetal/reports/wf29_*/` (empty) |
| **Wf-30** (this) | m3 Vina Dock + 10 pockets | partial (Phase 1 SHIP, Phase 3 SHIP-stub, Phase 4 honest-neg) | `molmetal/reports/wf30_*/` |
| **Wf-31** | m1 Ru AUC + m2 Pt A2780cis + m7 Diversity | partial (3 scripts SHIP, 0 GPU runs) | R20 P2 / P3 / P4 |
| **Wf-32** | m16 Affinity + m19 LOMO + m8 PB-Valid | partial (2 scripts SHIP, 0 GPU runs) | R20 P5 / P6 / P7 |
| **Wf-33** | paper v3 (replace PROJECTED with MEASURED) | NOT ship (no MEASURED lifts to insert; main.pdf still Wf-26 v2, 179 pages) | R20 |
| **Wf-34** | NEW skill `mol-metal-gpu-execution` + final synthesis | partial (skill SHIP 302 LOC, final synthesis = this file's parent) | `molmetal/.claude/skills/mol-metal-gpu-execution/` |

### §7.3 Downstream artifacts

- **`metrics/by_metric/vina_kcal_per_mol.json`** — canonical m3 metric file, updated 2026-09-26 with `MEASURED_BUT_STUB` entry (status label = honest-negative, lift=+1.3e-08, n=250).
- **`molmetal/reports/wf30_pocket_prep/`** — 10/10 READY receptors (~25 s total wall-time).
- **`molmetal/reports/wf30_dock/`** — `se3sb/`, `baseline/`, `best_m3_summary.json`, `comparison.md`, `m3_lift_analysis.json`, `notes.md`, `real_vina_attempt/`, `real_vina_organic/`.
- **`TODO/completed/82_metric_gap_filling.md`** — final TODO-82 synthesis (this file is a Phase-B-focused companion).
- **`TODO/INDEX.md` §13.6** — final results table now lists Wf-30 as "10/10 READY + 250-evals × 4 methods stub ship, m3 lift = 0, honest-negative → R20 P1".

### §7.4 Honest-verdict summary

Wf-30 Phase B **delivered**:
- ✅ 10/10 CrossDocked pocket receptor PDBQT staging.
- ✅ `dock_10pockets_se3sb.py` (~998 LOC) — sweep driver.
- ✅ `verify_dock_10pockets.py` — CPU smoke.
- ✅ `wf30_dock_summarize.py` (~433 LOC) — summarizer.
- ✅ 250-eval × 2-method sweep on disk.
- ✅ Phase 4 lift analysis (honest-negative, stub-floor dominated).
- ✅ Real-Vina smoke on organic cohort (50/50 docked, adapter plumbing validated).

Wf-30 Phase B **did NOT deliver**:
- ❌ A real-Vina lift measurement on the metal cohort (meeko blocker).
- ❌ A method-level lift (parameter-free stub velocity field gives no signal).
- ❌ The PROJECTED -8.5 kcal/mol lift as a real-method result (only as a stub-floor artifact).

**R20 P1 is open**: G1 m3 Vina Dock lift remains WEAK. Recommended R20 path = train W2.1 EGNN velocity field + extend meeko for Pt/Au/Ru + reduce Vina exhaustiveness, OR switch headline candidate set to organic cohort and re-run with `--use-real-vina`.

---

## §8 File index

Pocket staging: `molmetal/reports/wf30_pocket_prep/{10 pockets}/receptor.pdbqt`, `pocket_status.json`, `pocket_prep_log.md`. Scripts: `molmetal/scripts/dock_10pockets_se3sb.py` (NEW, 998 LOC), `verify_dock_10pockets.py` (NEW), `wf30_dock_summarize.py` (NEW, 433 LOC). Sweep output: `molmetal/reports/wf30_dock/{se3sb,baseline}/metrics.json` (250 each), `best_m3_summary.json`, `per_pocket_metrics.json`, `comparison.md`, `m3_lift_summary.md`, `m3_lift_analysis.json`, `notes.md`, `real_vina_organic/`. Metric: `metrics/by_metric/vina_kcal_per_mol.json` (canonical m3, updated).

---

**Date**: 2026-09-26 · **Verdict**: ⚠️ MEASURED_BUT_STUB_DOMINATED — honest-negative, R20 P1

## 2026-09-27 update

-9.6574 was the legacy stub-floor artifact; Wf-121 P3 ship real Vina path now active per TODO-113 honest reframe.
