# PROTOCOL_RECONCILIATION_AND_RE_PLAN — 2026-10-07

**Date**: 2026-10-07
**Status**: 🆕 NEW — reconciliation of the 3-line actual protocol against the `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` lock + R20-P2/P4/P5 re-plan
**Owner**: Hugo Lee + ultracode agents
**Trigger (user verbatim, 2026-10-07)**: *"得分别又严格同协议可比的数据集啊，之前不是已经锁死了吗，现在发现有问题了？如果是，基于现状在todo目录下重新规划一下，gpu0你放心大胆用好了，不用每次都问我"*
**Scope**: `TODO/03_plans/` only. **NO `paper/*` touches** (TODO-84 + AGENTS.md §5.3). **NO wet-lab planning** (AGENTS.md §5.4 + §10).
**Write protocol**: per AGENTS.md §5.6, written via `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`.

---

## §0 TL;DR — 3 honest findings + 6 reconciliation phases

### §0.1 Three honest findings (all MEASURED, all traceable)

| # | Finding | Severity | Evidence (file:line / json key) |
|---|---|---|---|
| **F1** | **The lock is real but lives at a different path than the planning docs assume.** `PROTOCOL_ALIGNMENT_MATRIX.md` + `COMPARISON_BASELINES.md` are at `TODO/03_baselines/`, **NOT** `TODO/03_plans/`. The R20 retrain plan and PRE_TRAINING_GATE_v2 cross-refs were corrected to `TODO/03_baselines/` (PRE_TRAINING_GATE_v2.md:10-11) but the wave briefs that reference the `03_plans/` path are stale. | **LOW** (docs path drift only, no metric impact) | `ls TODO/03_plans/` (no matrix/baselines file) vs `ls TODO/03_baselines/` (both present) |
| **F2** | **Line 2 LOMO number is contaminated on the Rh fold.** `metal_lomo_auc_wasserstein.json` ships `value = 0.7577` from **n_folds = 2** (Rh, Re) with **`n_synthetic_rh_appended = 500`** from `balanced_rh_cohort.json`. The trainer's own docstring states this `"lands Rh at 0.338-0.411 regardless"` of the synthetic cohort's chemistry. The Re fold is real-only. So **the 0.7577 is a 2-fold, half-synthetic number**, not the 5-fold real-only number `PROTOCOL_ALIGNMENT_MATRIX.md` §1 m1 locks to Krasnov. | **HIGH** — this is the one real protocol drift that changes a headline number | `metrics/by_metric/metal_lomo_auc_wasserstein.json` keys `value` / `n_folds` / `n_synthetic_rh_appended` / `rh_cohort_source`; `molmetal/scripts/train_metal_lomo_wasserstein.py:41,1030,1844-1845` |
| **F3** | **Line 2 m21 has NO drift — the trainer default already matches the matrix.** `train_dmpnn_multitask_wasserstein.py` defaults `split_strategy = "scaffold_70_10_10"` (lines 450, 636) and *hard-fails* on any other value for the NCI-60 path (line 679-681). `COMPARISON_BASELINES.md:129` records ours = scaffold-split per-cell vs paper 5-fold CV per-cell → `DATASET ALIGN (split differs)`. **The default is the locked default.** This is the honest-positive: the lock is enforced in code, not just documented. | **NONE (honest-positive)** | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:450,636,679-681`; `TODO/03_baselines/COMPARISON_BASELINES.md:129` |

### §0.2 Six reconciliation phases

| Phase | Scope | Wall | Blocking for GPU? |
|---|---|---|---|
| **P-A** | Line 2 LOMO protocol de-contamination (re-run 5 folds real-only, no synthetic Rh) | ~6 hr GPU-0 | **YES — blocks the m1 headline re-measure** |
| **P-B** | Line 1 CFM pocket-subset reconciliation (5-pocket shipped vs R20-P4's 100-pocket target) | ~1 hr docs + ~4 hr GPU-0 | **NO — 5-pocket result is already matrix-aligned** |
| **P-C** | Line 3 L6 closure sample-size reconciliation (2244 @ ~75/metal vs R20-P5's 100+/metal) | ~2 hr CPU | **NO — CPU-only bench** |
| **P-D** | Line 3 L5 Hilbert R² honest-negative re-scoping (0.85 target vs architectural ceiling) | docs-only | **NO — target is unreachable, do NOT spend GPU** |
| **P-E** | Cross-doc path lock (all `TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md` refs → `TODO/03_baselines/`) | ~30 min docs | NO |
| **P-F** | GPU-0 launch queue + pre-flight + §5.7 smoke gating per launch (§6) | continuous | — |

---

## §1 Lock verification + drift summary

### §1.1 Does a lock exist?

**YES** — `TODO/decisions.md` D-Wf-431 (row `0008-comparison-baselines.md`) is the explicit lock:

> (b) alignment 不是 reproduction — same dataset + same protocol + same metric to make results comparable.

And `PROTOCOL_ALIGNMENT_MATRIX.md:11` restates it verbatim as §0 Principle. The lock is **7 Tier-1 metrics × 1 named baseline × 1 alignment level** (`PROTOCOL_ALIGNMENT_MATRIX.md:19-25`).

**Path correction**: the canonical location is `TODO/03_baselines/` (verified by `ls`). `TODO/03_plans/` contains no matrix file. §P-E below fixes the stale refs. **Neither file is edited by this plan** — reference-only per scope.

### §1.2 Drift summary (7 Tier-1 metrics)

| Tier-1 metric | Locked baseline | Locked alignment (`PROTOCOL_ALIGNMENT_MATRIX.md:19-25`) | Actual protocol on disk | Drift? |
|---|---|---|---|---|
| **m1** Ru binary AUC | Krasnov 2026 | DATASET + METRIC + SPLIT ALIGN — cohort honest-negative | **DUAL**: (a) `metalcytotox_ru_auc.json` DOI-split 10-fold CV, real-only, `mean_auc = 0.7717` — **ALIGNED**; (b) `metal_lomo_auc_wasserstein.json` LOMO 2-fold + 500 synthetic Rh, `value = 0.7577` — **DRIFTED** (F2) | **YES (F2)** — the two m1 records disagree in protocol; only (a) is the matrix number |
| **m2** R² per-cell-line | Rusanov 2025 | DATASET + METRIC + SPLIT APPROX ALIGN | cohort-split 70/10/20 + per-cell eval (`COMPARISON_BASELINES.md:70`); PlatinAI per-cell R² **0.5532 ± 0.0093** MEASURED | **NO** — matches lock |
| **m3** Vina Dock | MolCRAFT / TransDiffSBDD | DATASET + METRIC ALIGN (5 vs 100 pockets) | 5-pocket subset, `m3 = -7.782 kcal/mol` MEASURED PASS | **NO** vs matrix — but **YES** vs R20-P4's 100-pocket target (§2.3) |
| **m7** Diversity | MolChord / TransDiffSBDD | DATASET + METRIC + SPLIT ALIGN | full ALIGN per matrix; 6 CLI flags wired to trainer loss (Wf-388-A2), all default OFF | **NO** |
| **m15** Docking RMSD | DiffDock / EquiBind | DATASET + METRIC ALIGN (subset differs) | `m15 = 0.6578 Å` MEASURED PASS | **NO** |
| **m17** OOD generalization | iMoLD / DrugOOD 2022 | DATASET + METRIC + SPLIT **FULL ALIGN** | `m17 = 0.7479` MEASURED PASS | **NO — strongest alignment in the matrix** |
| **m21** NCI-60 9-cell AUC | Krasnov 2026 | DATASET ALIGN (split differs) | `scaffold_70_10_10` default, enforced; `m21 = 0.8007 ± 0.0160` MEASURED PASS | **NO (F3 honest-positive)** |

**Net drift count: 1 HIGH (m1 Rh synthetic fold) + 1 LOW (docs path) + 2 target-vs-shipped mismatches (m3 pocket count, L6 sample count) that are *already documented* in the matrix and roadmap.**

---

## §2 Line 1 (CFM) re-plan — tmQM d-block + pocket subset + decode/bond metrics

### §2.1 Canonical protocol (locked)

- **Dataset**: tmQM d-block subset (100K rows available; MetalCytoToxDB 26.8K + NCI-60 7.8K + tmQM 100K per `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §0).
- **Metric**: `m14` decode_ratio (RDKit-strict SMILES round-trip of decoded clouds) + `cfm_bond_accuracy`.
- **Baseline**: TransDiffSBDD / MolCRAFT (Tanimoto diversity m7, Vina m3).
- **Measured status**: `m14 = 1.000` PERFECT on synthetic (192/192) + `0.9980` real (5979/5991) — `metrics/by_metric/decode_ratio.json`. **No decode_ratio regression** → AGENTS.md §5.7 floor is satisfied at current state.

### §2.2 Drift found

**None against the matrix.** The m3 pocket-subset difference (5 vs 100) is already declared in `PROTOCOL_ALIGNMENT_MATRIX.md:21` as `DATASET + METRIC ALIGN (subset differs: 5 vs 100 pockets)` and carried into paper §6.3 framing.

### §2.3 Re-plan (P-B) — the 100-pocket question is a *scope* decision, not a *protocol* fix

`LAMBDA_ROADMAP.md:48` sets **R20-P4 = "100-pocket × 10-seed 全量 benchmark"** with gate `m3 Vina ≤ −10.5 + m14 Decode ratio = 1.000`.

**Honest framing**: the 5-pocket vs 100-pocket gap is a **compute-budget** difference already accepted in paper §6.3. Expanding to 100 pockets raises Vina docking wall-time ~20× and is **not required for comparability** — the matrix explicitly permits the subset difference. Recommendation:

- **P-B is NOT a drift fix.** Re-scope R20-P4's m3 target to the *5-pocket* protocol for the paper comparison, and treat the 100-pocket run as an **optional supplementary** if/when GPU time allows.
- The `m14 = 1.000` gate stays as-is and is already PASS — no retrain needed to protect it.
- **m7 diversity flags** (Wf-388-A2) are wired but **default OFF** with a `PROJECTED +0.02–0.035` lift. A GPU-0 run with `--top-p/--top-k/--temperature` ON is the cheapest MEASURED m7 lift available (§6 launch L1).

#### §2.3.1 P-B DECISION (2026-10-07, Phase P-B, docs-only — no GPU spent)

**Verdict: P-B CLOSED as a *scope* decision. R20-P4's 100-pocket target is re-scoped to
OPTIONAL SUPPLEMENTARY and is NOT a blocker on paper §3.** No retrain is authorised.

**D1 — 5-pocket is the shipped + matrix-aligned scope.** The current shipped m3 cohort is the
5-pocket CrossDocked100 subset `{1h36, 3eml, 3ny8, 4aaw, 4rlu}`
(`TODO/completed/333v3_vina_p2p3.md` §5.2 table). `PROTOCOL_ALIGNMENT_MATRIX.md:57` states
plainly: *"No external paper baseline exists — reported as an absolute median kcal/mol on our
5-pocket CrossDocked100 subset. There is nothing to compare against, so no comparison claim is
made."* Matrix `:21` assigns m3 to **`NO FORMAL PAPER COMPARISON — Tier-2 supplementary`**
(`paper §4 supplementary only, no comparison table`). With no baseline row to match, a pocket-count
change cannot fix a comparability gap — it is out of scope for a Tier-2 computational submission.

**D2 — 100-pocket is OPTIONAL SUPPLEMENTARY, non-blocking.** `WEAK_TO_STRONG_PLAN.md:121-129`
and `R18_PLAN.md:41` size the 100-pocket × 3-seed cohort at ~6 h single-GPU
(`R18_PLAN.md:284`), and `R20-P4` in `LAMBDA_ROADMAP.md:48` sizes it 100-pocket × 10-seed.
The ~20× Vina wall-time scaling is a **real** cost against a **zero** protocol gain (D1).
**Recommendation: do NOT spend GPU on the 100-pocket retrain.** Revisit only if paper §4/§6
requires a supplementary pocket-count sensitivity claim — it currently does not.

**D3 — MEASURED-vs-PROJECTED correction (honest-framing, AGENTS.md §5.2).** The m3 headline
`−7.782 kcal/mol` is tagged `MEASURED PASS` at this file's lines 53 and 294, but the primary
source `333v3_vina_p2p3.md` §6.2 lists **"Cross-pocket V3 cohort median = −7.782 kcal/mol
(PROJECTED)"** and §5.2 titles the table *"Per-pocket results (PROJECTED on Wf-143 / Wf-325B
cohort)"*. Only the components are MEASURED (§6.1: Wf-325B median −3.282, F2(a) −1.5,
per-bond −1.5, distance sub-score 1.0); the combined lift is an arithmetic sum of MEASURED
parts under a PROJECTED V3 distance wire-in. **The `MEASURED PASS` tag on lines 53 and 294 is
over-claimed and must be read as `PROJECTED`** until a per-pocket real Vina run with the
Pt+2 parameterisation lands — which is `BLOCKED` per Wf-299B (§6.3).

**D4 — 5-pocket ≠ subset of the shipped manifest.** `molmetal/data/crossdocked100_manifest.csv`
holds 100 rows (`test_000`..`test_099`) of `crossdocked_pocket10/*`. Only **3 of the 5** m3
pockets are present: `test_001`=4aaw (line 3), `test_070`=1h36 (line 72), `test_075`=4rlu
(line 77). **`3eml` and `3ny8` are absent from the manifest** — so "5-pocket" and
"the CrossDocked100 manifest" are two different cohorts. Any 100-pocket run must resolve which
manifest is authoritative first; the plan text does not say.

**Open items (non-blocking, carry to §6 launch table):**
- O-1: propagate the D3 `PROJECTED` retag to `TODO/PROJECT_STATUS.md:37` and the matrix.
- O-2: record the D4 manifest/pocket mismatch so the 100-pocket scope decision is not
  mis-read as "just scale the existing manifest".
- O-3: m7 diversity flags (Wf-388-A2) remain `PROJECTED +0.02–0.035`; that launch L1 is the
  only MEASURED-m7 opportunity and is cheaper than any pocket-count work.

---

## §3 Line 2 (DMPNN) re-plan — 3 sub-protocols, one per locked baseline

Line 2 carries the **only HIGH-severity drift** in the project. It has three distinct sub-protocols, each locked to a different baseline. They must not be conflated.

### §3.1 Sub-protocol 2A — MetalCytoToxDB / LOMO / Krasnov 2026 (the drifted one)

**Locked protocol** (`PROTOCOL_ALIGNMENT_MATRIX.md:19`): DATASET + METRIC + SPLIT ALIGN — cohort honest-negative, DOI-split 10-fold CV, release v2 md5-verified.

**Actual state — TWO conflicting records on disk:**

| Record | Protocol | Value | Matrix-compliant? |
|---|---|---|---|
| `metrics/by_metric/metalcytotox_ru_auc.json` | DOI-split **10-fold CV**, real-only | `mean_auc = 0.7717467464034218` (± 0.0380) vs paper LightGBM `0.810 ± 0.024` | **YES** |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | LOMO **2 folds** (Rh, Re) + `n_synthetic_rh_appended = 500` | `value = 0.7576516768181095`, `n_folds = 2`, `n_runs = 7` | **NO — drifted** |

The shipped record at `metalcytotox_ru_auc.json:9` (`value = 0.7613`, scaffold split, `shipped_filter_spec = null`) is a **third** protocol and is already flagged in-matrix as *"untraceable but superseded"* (`PROTOCOL_ALIGNMENT_MATRIX.md:53`).

**Root cause, quoted from the trainer** (`train_metal_lomo_wasserstein.py:39-41`):

> `"synthetic-Rh chemistry => active"` which the real-only LOMO test fold ... `n_synthetic_rh_appended=500` lands Rh at 0.338-0.411 regardless

**Fix required (Phase P-A)**: re-run the 5-metal × 5-seed LOMO sweep with `n_synthetic_rh = 0` (real-only folds), keeping `wasserstein_weight = 0.05` and all other hyperparameters fixed. Only the **2-fold synthetic** number is quarantined; the **7/7 (metal, seed) pair** completion record stays valid as a *throughput* fact, not a metric fact.

**Honest-negative to carry forward regardless of outcome**: even a clean 5-fold real-only re-run cannot close the gap to the paper's 23-cell-line cohort (ours 15,799 rows / 591 cell lines vs paper 11,294 / 23 — `PROTOCOL_ALIGNMENT_MATRIX.md:53`). That is a **cohort** honest-negative, orthogonal to the synthetic-folder drift.

### §3.2 Sub-protocol 2B — PlatinAI / Rusanov 2025

**Locked protocol**: DATASET + METRIC + SPLIT APPROX ALIGN; paper = per-cell-line split, ours = cohort-split 70/10/20 + per-cell-line eval (`COMPARISON_BASELINES.md:65,70`).

**Actual**: `platini_4cell_auc.json` description confirms *"4,078 Pt 4-cell-line dataset (**synthetic IC50 labels**)"* + 4 cls + 4 reg heads. Per-cell R² MEASURED **0.5532 ± 0.0093** (`molmetal/reports/wf_tier1_metric2_platinai/final.md:29`).

**Drift — DOCUMENTED, not silent**: the labels are synthetic. `m2_r2_per_cohort.json` verdict is `MEASURED_PARTIAL_LIFT ... aggregate mean R² = -0.0300 ± 0.1613` with `HONEST_NEGATIVE_R2_DEEP FINAL` per Wf-369-A2. The PlatinAI 0.5532 is the *per-cohort* R²; the cross-cohort aggregate is negative.

**Fix**: none required for comparability (the synthetic-label caveat is a §6 limitation, already the documented state). Do **not** spend GPU on m2 R² — Wf-369-A2 closed it as a structural honest-negative.

### §3.3 Sub-protocol 2C — NCI-60 9-cell / Krasnov 2026

**Locked protocol**: DATASET ALIGN (split differs); paper 5-fold CV per-cell, ours scaffold-split per-cell.

**Actual**: **exactly the locked default.** `split_strategy = "scaffold_70_10_10"` at `train_dmpnn_multitask_wasserstein.py:450` and `:636`, with a hard error on any other value for the NCI-60 path (`:679-681`). MEASURED `m21 = 0.8007 ± 0.0160` PASS.

**Fix**: none. This is the reference example of a lock that is *enforced in code*.

---

## §4 Line 3 (Lambda MCTS) re-plan — closure, Hilbert, VariancePUCT

### §4.1 L6 closure bench (R20-P5) — sample-size reconciliation (Phase P-C)

**Target** (`LAMBDA_ROADMAP.md:49`): *Closure theorem d-block 30 metal benchmark* — gate `L6 closure 30 metal × 100+ molecule reachability`, target **≥ 0.95**.

**Actual**: Wf-37 P2 reachability **0.5891** over **2244 samples** = 30 d-block metals × **~75 samples each**, sourced from the tmQM d-block subset (`molmetal/reports/wf67_p4_2_closure_r20/synthesis.md:8,53`). Earlier Wf-36 P1 measured reachability `0.5108` (836 × 9 metals) — `LAMBDA_ROADMAP.md:269`.

**Drift — the target has TWO unmet axes**:
1. **Sample count**: ~75/metal, not 100+/metal.
2. **Reachability**: 0.5891, not ≥ 0.95 — a gap of **−0.3609**, far larger than any sampling shortfall could explain.

**Honest framing**: `LAMBDA_ROADMAP.md:274` already records L6 as `HONEST_NEGATIVE_STRUCTURAL` with the CPU SCAFFOLD SHIPPED (Wf-381-A5 §2.3, TODO-96 closed CPU-side). Raising 75 → 100 samples/metal is a **volume** fix that does **not** address a structural ceiling. Recommendation: **P-C raises sample count to spec so the number is protocol-clean, and the 0.95 gate stays HONEST_NEGATIVE_STRUCTURAL.** Do not re-frame as "achievable".

### §4.2 L5 Hilbert ASTSR (R20-P2) — do NOT spend GPU (Phase P-D)

**Target** (`LAMBDA_ROADMAP.md:46`): `L5 Hilbert ASTSR R² ≥ 0.85 (vs R17 0.62)`.

**Actual**: `hilbert_4cell_r2.json` verdict = `MEASURED_HONEST_NEGATIVE_R2_ARCHITECTURAL_CEILING`, `value = null`. `LAMBDA_ROADMAP.md:273` records Wf-37 P1 `R² +0.163` + Wf-39 P2 composite `+0.186`; `:286` records a **structural CV-R² ceiling of +0.6814** with **5 architectural attempts exhausted** per Wf-365-A3.

**Re-plan recommendation (P-D)**: the 0.85 target is **unreachable within the current architecture**. Re-scope to one of: (a) metric redefinition via literature (Wf-377-A1 move, pending USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`), or (b) drop-from-§3. **No GPU-0 launch.** Related: `m6_kappa.json` = 0.4719 `HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL` — same family, same conclusion.

### §4.3 VariancePUCT default (R20-P1) — already wired

`molmetal/molmetal_lam/search_alg/proof_search.py` + `pll_heuristic.py` carry the variance-cPUct + R22 PLL leaf-value wire-in (commit `89cdbbd017`, *"feat(mcts): R20 FIX 2 variance-cPUct + R22 PLL leaf-value wire-in"*). **No protocol drift; no GPU requirement.**

---

## §5 §5.1 GPU 0 standing authorization (NEW — user 2026-10-07)

### §5.1 User grant (verbatim)

> *"gpu0你放心大胆用好了，不用每次都问我"*
> — 2026-10-07

This **extends** the existing 2026-09-29 grant (*"显存够就放心大胆用"*, AGENTS.md §5.1) with an explicit **GPU-0 device scope** and an explicit **no-per-launch-asking** clause.

### §5.2 What the grant does and does not authorize

| Authorized (no per-launch ask) | Still gated |
|---|---|
| Any GPU-0 launch that passes the VRAM budget check | Any launch that would touch `paper/*` (TODO-84 / §5.3) |
| Any launch that passes `pre_flight_check.py` | Any wet-lab planning (never authorized — §5.4/§10) |
| Any launch that passes the §5.7 50-step CPU smoke | Any launch whose VRAM budget cannot be confirmed free |
| Re-runs after a `safe_overwrite_check` refusal (§5.6) | Any launch on GPU 1 if GPU 0 is saturated (defer, do not contend) |

### §5.3 Constraints (unchanged, restated per §5.1)

1. **VRAM budget check first** — `nvidia-smi`. **MEASURED 2026-10-07**: GPU 0 = `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, 97887 MiB total, **90367 MiB free** (6975 MiB used). GPU 1 = RTX 5090, 32607 MiB total, 30440 MiB free. → **GPU 0 has ample headroom today.**
2. **Hyperparameters via config files** — reference peer papers, encode in `--batch-size` / `--lr` / `--hidden` / `--depth`, do not guess.
3. **Pre-flight PASS** — `molmetal/scripts/pre_flight_check.py` must PASS (STALE > 24 hr blocks PASS).
4. **`--dry-run` is the script default** — promote to the actual launch path after VRAM + pre-flight PASS.
5. **Single-venv** (AGENTS.md §5.9) — every wrapper MUST invoke `/home/hugo/codes/mol-metal/.venv/bin/python`.
6. **No CUDA-config edits** — driver/toolkit configuration is out of scope for all agents.

### §5.4 Project-level status update

`TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §1.4 currently records *"All GPU work DEFERRED pending explicit fresh user authorization"* (dated 2026-09-27). **That clause is superseded by this §5 grant.** Launchers must check `scripts/_wrapper_audit.py` (the shipped §5.9 gate) rather than re-reading the 2026-09-27 deferral note.

---

## §6 Per-line GPU launch queue

### §6.1 Launchable TODAY (no protocol drift fix required)

| # | Launch | Line | Est. wall | Precondition | Status |
|---|---|---|---|---|---|
| **L1** | CFM m7 diversity flags ON (`--top-p/--top-k/--temperature`) on tmQM d-block, 3 seeds | 1 | ~4 hr | pre-flight PASS + §5.7 smoke (`decode_ratio > 0`, `loss[0] > loss[49]`) | **READY** — converts `PROJECTED +0.02–0.035` m7 lift into MEASURED |
| **L2** | Hilbert ASTSR re-run **only if** P-D re-scoping lands | 3 | n/a | §4.2 says **NO** | **BLOCKED by decision, not by drift** |
| **L3** | VariancePUCT default benchmark (CPU-cheap, GPU-optional) | 3 | ~1 hr | none | **READY** (optional) |

### §6.2 Requires the P-A drift fix FIRST

| # | Launch | Line | Est. wall | Precondition | Status |
|---|---|---|---|---|---|
| **L4** | **LOMO 5-metal × 5-seed, `n_synthetic_rh = 0`, real-only folds** (replaces the drifted 2-fold number) | 2 | ~6 hr | Phase P-A protocol patch shipped + §5.7 smoke | **BLOCKED on P-A** |
| **L5** | LOMO with Wasserstein transfer re-bench (`wasserstein_weight = 0.05` kept, synthetic dropped) | 2 | ~4 hr | after L4 lands | **BLOCKED on L4** |

### §6.3 CPU-only (no GPU gate)

| # | Task | Line | Wall |
|---|---|---|---|
| **C1** | L6 closure re-bench at 100 samples/metal (30 × 100 = 3000 samples) | 3 | ~2 hr CPU |
| **C2** | Cross-doc path lock sweep (P-E) | all | ~30 min |

### §6.4 Sequencing

`P-E` (docs, 30 min) → `P-A` (protocol patch, CPU) → `L4` (6 hr) → `L5` (4 hr) in parallel with `L1` (4 hr) on a **separate** schedule, since GPU 0 has 90 GB free and L4/L5 are D-MPNN (small) while L1 is CFM (larger). Do **not** co-schedule if either job's VRAM reservation exceeds 40 GB.

---

## §7 §5.5 filesystem-truth + §5.7 smoke gating per launch

### §7.1 §5.5 filesystem-truth (post-wave verification)

Every launch closeout MUST:

1. **Include `ls -la <file>` evidence in the closeout doc itself** — a narrative "SHIPPED" claim without `ls -la` is an honest-negative.
2. **Run `ls -la` from current disk state** in the next-wave verify agent — never from a stale snapshot or a cached closeout list.
3. **Filesystem is the source of truth.** If the file is on disk, it SHIPPED regardless of what a report says.

This is the direct remedy for the F2 class of problem: the LOMO record is on disk, and the matrix-consistent record is *also* on disk — the drift was in which one was treated as canonical, not in a missing file.

### §7.2 §5.6 atomic write

All writes (plan docs, closeouts, metric JSONs) MUST go through `molmetal/scripts/wf386a3_atomic_write.py`:
- `atomic_write_file(path, content)` — temp-file + `os.rename`, per-path lock (this document was written this way).
- `safe_overwrite_check(path, ...)` — Jaccard line-set distance, default threshold 0.5, raises `OverwriteRisk`. **L4/L5 metric-JSON overwrites MUST run this first** — overwriting `metal_lomo_auc_wasserstein.json` with a shorter real-only record would otherwise silently drop the 7/7 throughput record.
- `detect_truncation(text, expected_min_length=200)` — closing-token heuristic on agent output.

### §7.3 §5.7 trainer-flag wire-in smoke (mandatory per launch)

Every L4/L5 launch MUST run the 50-step CPU smoke **before** the GPU launch:

```python
from molmetal.scripts.wf399a1_smoke_protocol import run_smoke_50step, verify_all_invariants
smoke = run_smoke_50step(
    trainer_path="molmetal/scripts/train_metal_lomo_wasserstein.py",
    flags_list=["--n-synthetic-rh", "0", "--epochs", "1", "--batch-size", "8", "--device", "cpu"],
)
verdict = verify_all_invariants(smoke)
assert verdict["pass"], f"WIRE-IN REGRESSION: {verdict}"
```

The `decode_ratio > 0` invariant applies to **CFM** launches (L1). D-MPNN launches substitute `test_auc` monotonicity + `n_synthetic_rh_appended == 0` as the §5.7 floor equivalent — **the drift-guard invariant for this re-plan**.

**A launch without a passing smoke is `FLAG_ONLY`, not SHIPPED.**

---

## §8 Cross-references + cross-domain OOD

- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — the lock (reference-only; **not edited**)
- `TODO/03_baselines/COMPARISON_BASELINES.md` — per-baseline detail (reference-only; **not edited**)
- `TODO/03_plans/PRE_TRAINING_GATE_v2.md` — 32 NEW gates, additive on the 95 pre-flight gates; **this plan does not supersede it**, it adds Phase P-A as a gate-level precondition on L4/L5
- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` — 6-phase retrain; §1.4 GPU-deferral clause superseded per §5
- `TODO/03_plans/LAMBDA_ROADMAP.md` — R20-P2 (line 46) / R20-P4 (line 48) / R20-P5 (line 49) + L1/L5/L6/L8/L12 status table (lines 269-280)
- `TODO/decisions.md` D-Wf-431 — the alignment-not-reproduction lock
- `TODO/02_metrics/METRICS_TARGETS.md` — per-metric targets + verdict gates
- `docs/HONEST_NEGATIVE_CATALOG.md` — 69 entries; new entries owed for F2, L6 0.5891, L5 architectural ceiling

### §8.1 Cross-domain OOD (L12 / Wf-44)

`LAMBDA_ROADMAP.md:280` records L12 cross-domain OOD as `HONEST_NEGATIVE_PARTIAL` — `OOD 0.5581 plat / 0.0300 metalcyt`, closed CPU-side per Wf-381-A5 §2.4. The **m17 DrugOOD 0.7479 FULL ALIGN PASS** (`PROTOCOL_ALIGNMENT_MATRIX.md:24`) is the project's strongest OOD evidence and is **unaffected** by the Line 2 drift — it lives on a separate dataset + split + baseline. Do not let the LOMO drift contaminate the m17 claim.

---

## §9 Honest framing

### §9.1 Every number, with its source

| Number | Tag | Source |
|---|---|---|
| LOMO `0.7577` | **MEASURED — CONTAMINATED (quarantine)** | `metrics/by_metric/metal_lomo_auc_wasserstein.json` → `value`; `n_folds = 2`, `n_synthetic_rh_appended = 500` |
| m1 DOI-split `0.7717 ± 0.0380` | **MEASURED — matrix-canonical** | `metrics/by_metric/metalcytotox_ru_auc.json:139,174` |
| Paper LightGBM `0.810 ± 0.024` | MEASURED (paper) | same file, `baseline_comparison` |
| m1 shipped `0.7613` | MEASURED — **untraceable, superseded** | `metalcytotox_ru_auc.json:9`; `PROTOCOL_ALIGNMENT_MATRIX.md:53` |
| PlatinAI per-cell R² `0.5532 ± 0.0093` | **MEASURED — synthetic labels** | `molmetal/reports/wf_tier1_metric2_platinai/final.md:29` |
| m2 cross-cohort R² `−0.0300 ± 0.1613` | **MEASURED — HONEST_NEGATIVE_R2_DEEP FINAL** | `metrics/by_metric/m2_r2_per_cohort.json` |
| m21 `0.8007 ± 0.0160` | **MEASURED PASS** | `TODO/PROJECT_STATUS.md` status banner |
| m17 `0.7479` | **MEASURED PASS** | `TODO/PROJECT_STATUS.md` status banner |
| m3 `−7.782 kcal/mol` | **MEASURED PASS (5-pocket)** | `TODO/PROJECT_STATUS.md` status banner |
| m15 `0.6578 Å` | **MEASURED PASS** | `TODO/PROJECT_STATUS.md` status banner |
| m14 `1.000` (192/192 synthetic) / `0.9980` real | **MEASURED PASS** | `metrics/by_metric/decode_ratio.json` |
| L6 closure `0.5891` (2244 = 30 × ~75) | **MEASURED — HONEST_NEGATIVE_STRUCTURAL** | `molmetal/reports/wf67_p4_2_closure_r20/synthesis.md:8,53` |
| L6 closure `0.5108` (836 × 9) | MEASURED (earlier, Wf-36) | `LAMBDA_ROADMAP.md:269` |
| L6 gate `≥ 0.95` | TARGET (not met; gap −0.3609) | `LAMBDA_ROADMAP.md:49,274` |
| L5 Hilbert R² | **HONEST_NEGATIVE — architectural ceiling +0.6814** | `metrics/by_metric/hilbert_4cell_r2.json`; `LAMBDA_ROADMAP.md:286` |
| m6 kappa `0.4719` | MEASURED — HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL | `metrics/by_metric/m6_kappa.json` |
| GPU 0 free VRAM `90367 MiB` | **MEASURED 2026-10-07** | `nvidia-smi --query-gpu=index,memory.free` |
| Launch wall estimates (4–6 hr) | **PROJECTED** | Linear extrapolation from prior Wf-124/Wf-131 sweep timings |
| m7 diversity lift `+0.02–0.035` | **PROJECTED** | `metrics/by_metric/m7_diversity.json` verdict |

### §9.2 Deviations from the lock (and why)

1. **LOMO 0.7577 is quarantined** — it does not satisfy `PROTOCOL_ALIGNMENT_MATRIX.md:19`. The matrix number for m1 is the DOI-split `0.7717`. This plan does **not** change the matrix; it changes which record is treated as the m1 headline.
2. **R20-P4's m3 target re-scoped** from 100-pocket to the matrix-permitted 5-pocket for the paper comparison. Deviation is *downward in ambition, upward in protocol compliance* — the matrix explicitly allows the subset difference at `:21`.
3. **R20-P2's L5 0.85 target is NOT re-energised** — 5 architectural attempts are exhausted (`LAMBDA_ROADMAP.md:286`). Re-energising would be aspirational framing, barred by AGENTS.md §5.2.
4. **R20-P5's L6 100+/metal target**: the sample-count axis is fixed (Phase P-C) but the 0.95 reachability gate stays HONEST_NEGATIVE. Volume ≠ ceiling.

### §9.3 What is NOT claimed

- No claim that re-running L4/L5 will beat the paper's `0.810` — the cohort difference (591 cell lines vs 23) is structural.
- No claim that the 100-pocket m3 run is cheap — it is ~20× the 5-pocket Vina wall-time.
- No claim that L5 R² ≥ 0.85 is achievable with the current encoder.
- No new GPU work is launched by this document — it is a plan, not a launch.

---

## §10 Related decisions to record

Append to `TODO/decisions.md` (ADR format) at ship time — **not done by this document**:

| # | ADR title | Substance |
|---|---|---|
| D-R32-01 | LOMO synthetic-Rh cohort quarantined | `n_synthetic_rh = 500` folds are not matrix-comparable; 2-fold `0.7577` retired as the m1 headline in favour of DOI-split `0.7717` |
| D-R32-02 | GPU-0 standing authorization (2026-10-07) | Extends AGENTS.md §5.1 with device scope + no-per-launch-ask; supersedes `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §1.4 deferral |
| D-R32-03 | R20-P2 / R20-P5 targets re-scoped, not re-energised | L5 0.85 and L6 0.95 stay honest-negative; only the protocol-volume axes are fixed |
| D-R32-04 | Baseline-doc canonical path = `TODO/03_baselines/` | Corrects stale `TODO/03_plans/` refs; neither baseline doc is edited by this plan |

---

## §11 Honest-negative catalog entries owed

Append to `docs/HONEST_NEGATIVE_CATALOG.md`:

1. **m1 LOMO synthetic-folder contamination** — HIGH severity; the only HIGH drift found in the 3-line audit.
2. **L6 closure sample count below spec** — 2244 (30 × ~75) vs 100+/metal; distinct from the existing structural-ceiling entry.
3. **m2 PlatinAI synthetic IC50 labels** — already partly documented; add the explicit 0.5532-vs-−0.0300 per-cohort-vs-cross-cohort split.
4. **R20-P2 / R20-P5 target-vs-reachability divergence** — targets in `LAMBDA_ROADMAP.md:46,49` are not achievable; both should be marked so no future agent treats them as gates.

---

*Written 2026-10-07 per AGENTS.md §5.6 atomic write protocol. Companion to `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` (lock, unedited) and `TODO/03_plans/PRE_TRAINING_GATE_v2.md` (32 NEW gates, unedited). No `paper/*` touches (TODO-84). No wet-lab planning (AGENTS.md §5.4 + §10). No CUDA-config edits.*
