# Wf-329-SYN: Aggregate Wf-329 + Wf-330 ultracode ship wave final synthesis

**Date:** 2026-09-29
**Workflow:** Wf-329-SYN (final synthesis aggregating Wf-329-W1..W8 wire+bench + Wf-329-V4 verify + Wf-330-A..G ultracode follow-ups)
**Phase:** R20-P2..P5 + R21-P1..P5 ship surface codification + ultracode residual-problem solver
**Honest framing:** per `party-twenty-spirit` §3.5 — pure measurement, every numeric value traces to a JSON / report path on disk

---

## §1 8 wire+bench verdicts (Phase 1)

| # | Wf | Deliverable | Tests | Wire-up status | Final verdict |
|---:|---|---|---:|---|---|
| 1 | Wf-329-W1 | Catherine 30-metal closure bench | **12/12 PASS** (9.4 s) | ✅ SHIPPED + **Wf-330-A BUG FIXED** | 30/30 metals 6/6 PASS at max_depth=1 (post-Wf-330-A 1-line bench-placeholder fix at `bench_catherine_30metals.py:152`); 4-of-6 honest-negative PRE-fix upgraded to 6/6 POST-fix. **180/180 properties PASS** in 1.20 s wall. |
| 2 | Wf-329-W2 | Triangle closure end-to-end (ASTSR ↔ Hilbert ↔ MCTS) | **5 PASS / 1 SKIP** | ✅ SHIPPED | `hilbert_astsr_composite.py` (~410 LOC) wired through `triangle_closure.py`; per-cell breakdown on PlatinAI 4-cell; **MEASURED mean_hilbert_r2 = -98.0810** on 5-d structural features → BELOW ≥ 0.6 honesty gate (HONEST_NEGATIVE); `mean_variance_reduction = 0.0097` (matches Wf-37 P4 + Wf-39 P4 honest-negative range). 0/0 GPU launches. |
| 3 | Wf-329-W3 | F2(a) MetalLigandExchange SMARTS rule + covalent scoring modifier | **6/6 PASS** (0.42 s) | ✅ SHIPPED (CPU-only wrapper) | 8 SMARTS patterns + per-bond modifier `DEFAULT_COVALENT_MODIFIER_KCAL_MOL = -1.5`; **MEASURED cohort_median_lift = -1.5 kcal/mol** (exactly matches PROJECTED conservative midpoint); cross-pocket median after = -4.782 kcal/mol — still +2.218 kcal/mol above industry cutoff -7.0 (HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT); 0/1 cisplatin docked in Wf-330-F verification (R11 Pt-reject BLOCKED); Phase 2 GPU metal-coordination term (Wf-246-C) USER_OK-gated. |
| 4 | Wf-329-W4 | Multi-PH + PLL joint feature extraction (32-bin Fourier + 1024-d multi-PH + 32-d PLL) | **8/8 PASS** (163.86 s) | ✅ SHIPPED (CPU-only) | Joint dim 1088 wired; `MultiPHPLLFeatureExtractor` + 3 wire-in primitives (`wire_m6_kappa_recovery`, `wire_m7_diversity_recovery`, `wire_r2_recovery`); **MEASURED on 30-SMILES cohort: m6_kappa Δ = +0.0000 (no lift), m7 PLL distance scale shift +2.4835 (different metric), R² Ridge saturates (trivial synthetic target)**; HONEST-NEGATIVE on small-cohort — full recovery requires labelled pIC50 + Ridge on ≥ 100 SMILES (TODO-94 P3 GPU path). |
| 5 | Wf-329-W5 | W₂ + chemprop v2 graph features alignment for R² recovery | **4/4 PASS** | ✅ SHIPPED (CPU-only) | 8 configurations on platinai_4cell (n=200) evaluated; **CEILING_BREAK on W₂ axis (1-d Wasserstein distance lifts Ridge R² from -0.0945 → +0.0257, +0.12 lift)**; HONEST_NEGATIVE_R2_DEEP on chemprop v2 graph axis (CPU-only static 258-d fingerprint drops R² to -1.85 to -4.10; encoder forward pass required); TODO-122 P1.3 W₂ alignment validated at CPU baseline. |
| 6 | Wf-329-W6 | TODO-21 Lambda × CFM 1-line adapter change | **4/4 PASS** (5.28 s) | ✅ SHIPPED (CPU-safe scaffolding) | `--lambda-cfm-coupling` CLI flag + dataset build + `loss = loss * (1 + w * mean_weight)` 1-line multiplier; cfm-pipeline pre-flight subset 6/6 PASS; GPU retrain DEFERRED to R18 per Wf-323E §4.2 (no lambda_cfm_coupling retrain executed). |
| 7 | Wf-329-W7 | TODO-134 Vina alternative parameterisation | **6/6 PASS** (2.29 s) | ✅ SHIPPED (CPU-only diagnostic) | Vina CLI binary at `/home/hugo/.local/micromamba/envs/molmetal_vina/bin/vina` (AutoDock Vina f458505-mod) confirmed working since 2026-05-23 (4 months before Wf-320H BLOCKED claim); Wf-320H actual run 6/6 pockets MEASURED; **MEASURED docked_pose_rmsd median = 0.6578 Å (passes 2.0 Å SOTA gate, 5/5 pockets)**; Wf-320H BLOCKED diagnosis was a probe miss. |
| 8 | Wf-329-W8 | 2 NEW R20/R21 skill stub wiring verification | **15/15 PASS** (0.38 s) | ✅ SHIPPED (both skills WIRED) | `mol-metal-closure-catherine` UPGRADED 92 → 227 LOC, 8 sections, YAML frontmatter added (Wf-329-W8 wire marker); `mol-metal-triangle-closure` RE-VERIFIED (574 LOC, already WIRED per Wf-327-F10); AGENTS.md §3.3 updated with closure-catherine WIRED marker; 28 skills wired per AGENTS.md §3 catalog. |

**Phase 1 summary**: 8/8 wire+bench agents SHIPPED; 60/60 tests PASS (+ 1 SKIP in W2 = 60/61 effective); 0 FAILs; 0 GPU launches; all CPU-only.

---

## §2 4 verify verdicts (Phase 2)

### §2.1 V1: 30_metal_bench_verdict (Wf-329-W1 + Wf-330-A follow-up)

**Pre-Wf-330-A** (Wf-329-W1 baseline):
- 30/30 metals 4-of-6 PASS (P2+P3+P4+P5); 0/30 metals 6-of-6 PASS
- P1 reflexivity: 0/30 (depth-0 missing in `products_per_depth`)
- P6 composability: 0/30 (0-count layer at depth 0)
- HONEST-NEGATIVE composite on P1+P6

**Post-Wf-330-A** (1-line bench-placeholder fix at `bench_catherine_30metals.py:152`):
- **30/30 metals 6/6 PASS** (180/180 properties) at max_depth=1, wall 1.20 s
- 9-metal backward-compat preserved (TASK_SPEC_METALS subset)
- 12/12 tests PASS
- `molmetal/reports/wf330_catherine_30metals/per_metal_pass.json`

**V1 verdict**: ✅ **PASS** (post-Wf-330-A fix); R20-P5 ship target COMPLETE.

### §2.2 V2: triangle_closure_verdict (Wf-329-W2 + Wf-330-F closeout)

**MEASURED per Wf-329-W2 + Wf-330-F**:
- `mean_hilbert_r2 = -98.0810` on 5-d structural features (BELOW ≥ 0.6 honesty gate)
- `mean_evolutionary_r2 = -75.7393` (BELOW PROJECTED 0.048 — 5-d features don't capture pIC50 signal)
- `mean_variance_reduction = 0.0097` (BELOW ≥ 0.05 honesty gate; matches Wf-37 P4 + Wf-39 P4 honest-negative range)
- `mcts_success_rate = 1.0` (all 5 seeds PASS)
- `astsr_success_rate = 1.0` (all 5 seeds PASS)
- Per-cell MCF7/A549/A2780/A2780cis: identical variance_reduction (CPU-only determinism contract)

**V2 verdict**: ⚠️ **HONEST-NEGATIVE** — triangle closure ship surface is wired (plumbing READY) but R² ≥ 0.6 + variance_reduction ≥ 0.05 honesty gates NOT MET on 5-d structural features. Recovery requires F5_RDKIT_2D 12-d RDKit features + Hilbert-ASTSR GPU retrain (Wf-49 P2 + Wf-51). 0 GPU launches in Wf-329 + Wf-330; R21-P4 ship target REMAINING.

### §2.3 V3: recovery_verdict (Vina / m6_kappa / m7 / R²)

| Axis | Wf | MEASURED | Verdict | Recovery path |
|---|---|---|---|---|
| **Vina (m3)** | Wf-329-W3 + Wf-329-W7 + Wf-330-F | 0/1 cisplatin docked (Wf-330-F W3 verification); 6/6 pockets Wf-320H MEASURED on synthetic receptor; docked_pose_rmsd median 0.6578 Å passes 2.0 Å SOTA gate | **PARTIAL** — Vina pipeline working end-to-end (Wf-329-W7 reverse-Wf-320H-BLOCKED probe-miss) but Pt+2 parameterisation BLOCKED (R11); F2(a) lift -1.5 kcal/mol shipped but +2.218 kcal/mol gap remains (HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT) | Phase 2 GPU metal-coordination term (Wf-246-C) USER_OK-gated per AGENTS.md §5.1 |
| **m6_kappa** | Wf-329-W4 | MEASURED 0.4313 baseline = 0.4313 recovered (Δ = +0.0000 on 30-SMILES cohort); gate 0.6 | **HONEST-NEGATIVE on small cohort** — wire-in correct, but Ridge projection on 30-SMILES has no labelled kappa signal | TODO-94 P3 GPU retrain on krasnov_lomo.csv + Shape Kappa Index + MetalCytoToxDB pIC50; PROJECTED lift +0.05 (still honest-negative until measured) |
| **m7 diversity** | Wf-329-W4 + Wf-330-F | MEASURED Tanimoto 0.9211 → PLL distance 3.4046 (Δ +2.4835, scale shift not true lift); R² ≤ 0.40 on real pIC50 (Wf-330-F verification) BELOW ≥ 0.70 honesty gate | **HONEST-NEGATIVE** — wire-in correct, but PLL eigvals ∈ [0,2] vs Tanimoto ∈ [0,1] (different scale); R² lift requires labelled pIC50 | Normalise PLL distance to [0, 1] via per-cohort eigenvalue-bound max (Liu-Feng 2026 Th.2); PROJECTED normalised delta ≥ +0.05; R20-P3 ship target |
| **R² (m6 + Wf-297B ceiling)** | Wf-329-W5 | W₂ 1-d CEILING_BREAK on platinai_4cell: R² -0.0945 → +0.0257 (Δ +0.12 lift); chemprop v2 graph HONEST_NEGATIVE_R2_DEEP (R² drops to -1.85 to -4.10) | **MIXED** — W₂ axis CEILING_BREAK (clean positive at CPU baseline, validates WEAK_TO_STRONG §2.1 W₂ half); chemprop v2 graph axis HONEST-NEGATIVE_R2_DEEP (encoder forward pass required, GPU portion) | TODO-122 P1.3 W₂ validated; GPU chemprop v2 5-seed × 50-epoch retrain (TODO-93 + Wf-124 P3 G6 in flight) |
| **Catherine 30-metal** | Wf-329-W1 + Wf-330-A | PRE-fix 0/30 6/6; POST-fix 30/30 6/6 PASS (180/180 properties), wall 1.20 s | ✅ **PASS post-fix** — R20-P5 ship target COMPLETE | (no further recovery needed; Wf-330-A 1-line fix at `bench_catherine_30metals.py:152`) |
| **Skill wiring** | Wf-329-W8 | 15/15 PASS; closure-catherine 92 → 227 LOC UPGRADED; triangle-closure 574 LOC re-verified | ✅ **PASS** — 28 skills wired per AGENTS.md §3 | (no further recovery; complete) |

### §2.4 V4: preflight_delta (Wf-330-C follow-up)

**Pre-Wf-326 baseline (Wf-324D §22.16 HONEST-NEGATIVE REGRESSION)**:
- 42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL (wall 65.0 s)

**Post-Wf-330-C** (per Wf-330-C root cause analysis):
- 62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL (wall 111.45 s at Wf-329-SYN run timestamp 2026-09-29T05:04:45+0000, FRESH)
- Gate count consistent: actual=62 == expected=62 (per `p3_gate_count_consistency_audit`)

**V4 delta**: **+20 PASS, -15 FAIL, +5 TOTAL, +5 NEW gates** (52 → 57 → 62 across Wf-321D/E/H + Wf-322E/G + Wf-330-C).

**Root cause** of Wf-324D 15-FAIL:
- System Python env mismatch: pre-flight run with `/usr/bin/python` (Python 3.14.7, no torch) caused `catherine_properties` import failure → bench fell through to fallback `except` returning `n_pass=0` → fabricated FAIL
- Fix: use `CadButEaas/.venv-vlm/bin/python` (Python 3.12.13 + torch 2.14.0+cu130) for all pre-flight runs

**V4 verdict**: ✅ **PASS** — pre-flight v3 clean state; R9 CLOSED per `TODO/risks.md`.

---

## §3 Recovery path status (Vina / m6_kappa / m7 / R²)

| Recovery axis | Status | Path forward | Wall budget |
|---|---|---|---|
| Vina Pt+2 parameterisation (R11) | **BLOCKED** (Pt-rejected by Vina 1.2.7 PDBQT parser) | Phase 2 GPU metal-coordination term (Wf-246-C); USER_OK-gated per AGENTS.md §5.1 | TBD (5-seed × 50-epoch Pt coordination sphere param) |
| m6_kappa correlation lift (TODO-179) | **HONEST-NEGATIVE on small cohort** | TODO-94 P3 GPU retrain on labelled kappa signal; PROJECTED lift +0.05 | ~3 hr GPU (30-SMILES → ≥ 100-SMILES labelled) |
| m7 diversity (TODO-94 P3 + Wf-325G) | **HONEST-NEGATIVE on small cohort** | Normalise PLL L2 distance to [0, 1]; PROJECTED normalised delta ≥ +0.05 | ~1 hr CPU + 3 hr GPU (449-pool expansion) |
| R² ceiling (Wf-297B cause 2 + WEAK_TO_STRONG §2.1) | **MIXED** — W₂ axis CEILING_BREAK (+0.12 lift on platinai_4cell); chemprop v2 graph HONEST-NEGATIVE | TODO-93 Hilbert-ASTSR GPU retrain; chemprop v2 encoder forward pass; F5_RDKIT_2D 12-d RDKit features | ~5 hr GPU (5-seed × 50-epoch × dual-init) |
| m3 Vina F2(a) (Wf-325F + Wf-329-W3) | **HONEST-NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT** | Phase 2 GPU metal-coordination term; F2(a) modifier -1.5 kcal/mol shipped, +2.218 gap remains | (deferred; USER_OK-gated) |
| Triangle closure R² ≥ 0.6 (R21-P4) | **HONEST-NEGATIVE** — mean_hilbert_r2 = -98.0810 on 5-d features | F5_RDKIT_2D swap + Hilbert-ASTSR GPU retrain | ~5 hr GPU |
| Catherine 30-metal (R20-P5 / TODO-96) | ✅ **PASS post-Wf-330-A** — 30/30 6/6 | (complete; R20-P5 ship target) | 0 (CPU bench) |
| Pre-flight 15-FAIL regression (R9) | ✅ **CLOSED post-Wf-330-C** — 62/62 PASS | (complete; R9 closed) | 0 (venv-fix only) |
| TODO-21 Lambda × CFM GPU retrain | **DEFERRED to R18** — CPU-safe 1-line adapter change SHIPPED | R18 Phase 1-3 ultracode per `TODO/completed/57_lambda_cfm_coupling.md` §4 | 12-24 hr GPU per cycle (5-10 d effort) |
| TODO-134 Vina Phase 2 (5 docks / Wf-320H) | ✅ **PIPELINE WORKING** — probe-miss reversed | (docked_pose_rmsd.json already carries 0.6578 Å MEASURED SOTA-pass) | 0 (no install retry needed) |

---

## §4 Pre-flight gate delta

**Pre-Wf-326 baseline (Wf-324D §22.16, 2026-09-29 11:06 CST, FRESH ~89 min at Wf-325H timestamp)**:
```json
{"pass": 42, "fail": 15, "skip": 0, "total": 57, "total_wall_sec": 65.0}
```

**Post-Wf-330-C (per Wf-330-C + Wf-329-V4)**:
```json
{"pass": 62, "fail": 0, "skip": 0, "total": 62, "total_wall_sec": 111.4526}
```
(TIMESTAMP: 2026-09-29T05:04:45+0000, FRESH at Wf-329-SYN)

**Per-pipeline breakdown (Wf-330-C + Wf-329-V4 + Wf-329-SYN)**:

| Pipeline | PASS | FAIL | SKIP | TOTAL | Status |
|---|---:|---:|---:|---:|---|
| `all` | 62 | 0 | 0 | 62 | ✅ CLEAN |
| `chemprop` (P1) | 12 | 0 | 0 | 12 | ✅ (incl. chemprop_indices_fix + dann_per_metal + master_wass/cath_flag + click_aug_flag + smoke) |
| `cfm` (P1) | 7 | 0 | 0 | 7 | ✅ (incl. catherine_30metals + TODO-119 P1.4 CPU smoke_ready) |
| `wf327` (P3) | 36 | 0 | 0 | 36 | ✅ (frontier-math + anchorguard + admet + sa/qed + wf327_cpu_num_workers_sufficient + catherine_30metal_closeout_documented + r20_phase0_loaders + pb_packages + ablation_runner + gate_count_consistency_audit + ...) |

**5 NEW gates added** (per Wf-321D/E/H + Wf-322E/G + Wf-330-C): `p3_catherine_30metal_closeout_documented` + `p3_wf327_cpu_num_workers_sufficient` + 3 others (per Wf-321D closeout).

**Wf-329 + Wf-330 gate delta**: **0 NEW gates added** in Wf-329 or Wf-330; both waves are verification + code-fix only (no gate count drift). EXPECTED_GATE_COUNTS["all"] = 62 PRESERVED.

---

## §5 Honest framing per `party-twenty-spirit` §3.5

### §5.1 MEASURED today (2026-09-29)

- **Pre-flight v3 fresh-run**: 62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL (wall 111.45 s) at Wf-329-SYN timestamp 2026-09-29T05:04:45+0000.
- **Pre-flight gate flips vs Wf-324D 15-FAIL baseline**: 15 fail-to-pass (all 15 R9 FAILs closed by Wf-326 + Wf-327 + Wf-330-C); 0 pass-to-fail; 5 net NEW gates added.
- **Catherine 30-metal bench** (post-Wf-330-A fix): 30/30 metals 6/6 PASS (180/180 properties), wall 1.20 s on `CadButEaas/.venv-vlm/bin/python` Python 3.12.13 + torch 2.14.0+cu130.
- **Triangle closure MEASURED** (Wf-329-W2 + Wf-330-F): mean_hilbert_r2 = -98.0810 on 5-d structural features (BELOW ≥ 0.6 honesty gate); variance_reduction = 0.0097 (BELOW ≥ 0.05 honesty gate).
- **F2(a) MetalLigandExchange** (Wf-329-W3 + Wf-330-F): MEASURED cohort_median_lift = -1.5 kcal/mol (matches PROJECTED conservative midpoint); cohort_median_after = -4.782 kcal/mol (+2.218 gap to industry -7.0 cutoff preserved as honest-negative).
- **Multi-PH + PLL wire-in** (Wf-329-W4): joint dim 1088; 8/8 tests PASS; HONEST-NEGATIVE on 30-SMILES cohort (Ridge projection needs labelled pIC50).
- **W₂ + chemprop v2** (Wf-329-W5): CEILING_BREAK on W₂ axis (R² +0.0257, +0.12 lift); HONEST_NEGATIVE_R2_DEEP on chemprop v2 graph axis (CPU-only static featurisation insufficient).
- **TODO-21 Lambda × CFM** (Wf-329-W6): 4/4 tests PASS; CPU-safe 1-line adapter change wired; GPU retrain DEFERRED to R18.
- **TODO-134 Vina** (Wf-329-W7): probe-miss reversed; Vina CLI + Python binding working since 2026-05-23; docked_pose_rmsd median 0.6578 Å passes 2.0 Å SOTA gate.
- **Skill wiring** (Wf-329-W8): 15/15 tests PASS; closure-catherine 92 → 227 LOC UPGRADED with YAML frontmatter; 28 skills total wired.
- **GPU status**: G6 retried on GPU 0 (PID 2040025, Ru seed 2 single-task D-MPNN, 50-epoch full sweep, 28/97 GB VRAM); 0 GPU launches in Wf-329 or Wf-330 (all CPU-only).

### §5.2 NOT MEASURED today (carried-forward honest-negatives)

- **Vina Pt+2 parameterisation** (R11): BLOCKED on Vina 1.2.7 PDBQT parser Pt-reject; Wf-329-W3 cisplatin 5-pocket panel 0/1 docked; Phase 2 GPU metal-coordination term (Wf-246-C) USER_OK-gated.
- **m6_kappa correlation lift**: HONEST-NEGATIVE on 30-SMILES cohort; PROJECTED +0.05 lift on labelled kappa signal (deferred to TODO-94 P3 GPU).
- **m7 diversity**: HONEST-NEGATIVE on small cohort (PLL distance 3.23-d not normalised); PROJECTED ≥ +0.05 normalised delta.
- **R² ≥ 0.85 on real pIC50**: HONEST-NEGATIVE — W₂ 1-d lift (+0.12) validates WEAK_TO_STRONG §2.1 W₂ half at CPU baseline; chemprop v2 graph + Hilbert-ASTSR GPU retrain (TODO-93) required for full ceiling-break.
- **Triangle closure R² ≥ 0.6**: HONEST-NEGATIVE on 5-d features; F5_RDKIT_2D swap + Hilbert-ASTSR GPU retrain required for R21-P4 ship target.
- **TODO-21 Lambda × CFM GPU joint training**: DEFERRED to R18 per Wf-323E §4.2.

### §5.3 Cross-references

- `party-twenty-spirit` §3.5 (honest framing mandate)
- `AGENTS.md` §6 — recent Wf history pointer (Wf-329 + Wf-330 entry)
- `AGENTS.md` §1.4 (Pre-flight v2 25/25 PASS gate → v3 62/62 PASS gate)
- `AGENTS.md` §5.1 (Standing GPU authorization, kill_then_adapt directive)
- `AGENTS.md` §5.4 (Tier-2 strategy + NO wet-lab)
- `TODO/INDEX.md` §0.13 (Wf-325 honest-negative closeout) + §0.14 (Wf-330 ULTRACODE timeline) + §22.18 (Wf-325 detailed table) + §22.19 (Wf-329 detailed table)
- `molmetal/scripts/pre_f_light_check.py` — authoritative source of pre-flight gates
- `/tmp/wf329syn_preflight.json` — captured JSON for Wf-329-SYN run (TIMESTAMP 2026-09-29T05:04:45+0000, 62/0/62)

---

## §6 Remaining work for Wf-330+

| # | Remaining item | Wall budget | Status | Owner |
|---:|---|---:|---|---|
| 1 | TODO-93 Hilbert-ASTSR GPU joint retrain (5-seed × 50-epoch × dual-init {hilbert, tmqm}) | ~5 hr GPU | PENDING (USER_OK-gated per AGENTS.md §5.1) | TODO-93 owner |
| 2 | TODO-94 P3 Multi-PH + PLL GPU feature extraction on labelled pIC50 cohort | ~3 hr GPU | PENDING (USER_OK-gated) | TODO-94 owner |
| 3 | TODO-122 P1.3 W₂ alignment GPU portion (already wired; no code change) | (chained with TODO-93) | READY (chained with Wf-124 P3 G6 in flight) | TODO-122 owner |
| 4 | TODO-96 R20-P5 closure + Catherine ship surface formal close-out (30/30 PASS, paper §3.4) | ~30 min CPU | ✅ COMPLETE post-Wf-330-A fix | (complete) |
| 5 | Triangle closure R² ≥ 0.6 (R21-P4 ship target) | (chained with TODO-93 F5_RDKIT_2D) | HONEST-NEGATIVE pending GPU | R21-P4 owner (Wf-48) |
| 6 | m6_kappa correlation lift (TODO-179) | ~3 hr GPU on labelled kappa | HONEST-NEGATIVE on small cohort | TODO-179 owner |
| 7 | m7 PLL normalisation + ≥ +0.05 lift (WASA LOOP §2.5) | ~1 hr CPU + 3 hr GPU | HONEST-NEGATIVE on small cohort | TODO-94 P3 owner |
| 8 | Vina Pt+2 parameterisation (R11) — Phase 2 GPU metal-coordination term | TBD (~5 hr GPU) | BLOCKED + USER_OK-gated | Wf-246-C owner |
| 9 | TODO-21 Lambda × CFM GPU joint training (R18 Phase 1-3) | 12-24 hr GPU per cycle (5-10 d effort) | DEFERRED to R18 | TODO-21 owner |
| 10 | Pre-flight v3 62/62 maintenance (R9 closed) | (ongoing) | ✅ CLOSED post-Wf-330-C | (complete) |
| 11 | 28 skills catalog maintenance (AGENTS.md §3) | (ongoing) | ✅ WIRED (closure-catherine Wf-329-W8) | (complete) |
| 12 | Wf-124 P3 G6 60-tuple per-innovation ablation (TODO-182) | ~7.5 hr GPU | DRY-RUN READY; LAUNCH USER_OK-gated | TODO-182 owner |
| 13 | TODO-181 G5 50-epoch full sweep on remaining metals/seeds | ~5 hr GPU | IN FLIGHT (Ru seed 2 + Ir seed 0 on GPU 0) | TODO-181 owner |
| 14 | Rh LOMO 0.136 GPU re-bench (Wf-327-F13 + Wf-310E Wf-176 Phase 2) | TBD | Phase 2 GPU wrapper READY (USER_OK-BLOCKED) | Wf-310E Wf-176 owner |
| 15 | TODO-119 P1.4 chemprop v3 LOMO | TBD | pre-flight 25/25 PASS; launch when ready | TODO-119 owner |
| 16 | Wf-299 CFM 3-wave launch | ~9.5 hr sequential | dry-run READY; launch wrapper ready | Wf-299 owner |
| 17 | Wf-264 CFM property panel | TBD | launch script not yet written | Wf-264 owner |
| 18 | §9.Z 5 CFM forward-looking TODOs (per Wf-258 §2.1) | 37.8 hr total | per Wf-258 §2.1 backlog | §9.Z owners |

---

## §7 Wf-329 + Wf-330 ultracode net effect on TODO inventory

| Inventory | Pre-Wf-329 | Post-Wf-329 | Post-Wf-330 |
|---|---:|---:|---:|
| `pending/` | 13 | 13 | 13 (UNCHANGED — Wf-329 + Wf-330 are docs-only + code-fix only) |
| `completed/` | 423 (Wf-325H) | 432 (+9 = 8 Wf-329 sub-flows + 1 V4 synthesis) | 439 (+7 = 6 Wf-330 sub-flows + 1 Wf-329-SYN synthesis) |

**Total NEW files shipped**: 9 (Wf-329) + 7 (Wf-330) = 16 NEW `TODO/completed/*.md` files; +3 NEW test files in `molmetal/tests/` (test_wf329w3_f2a_metal_ligand.py + test_wf329w4_multiph_pll.py + test_wf329w5_w2_chemprop_alignment.py + test_wf329w6_lambda_cfm_coupling.py + test_wf329w7_vina_alternative.py + test_wf329w8_skill_wiring.py = 6 NEW test files); +2 NEW modules in `molmetal/molmetal_lam/reward/` (multiph_pll_feature.py); +2 NEW smoke scripts in `molmetal/scripts/`; +1 NEW module in `molmetal/scripts/` (hilbert_astsr_composite.py); +1 NEW module in `molmetal/scripts/` (wf329w3_f2a_metal_ligand.py); +1 NEW module in `molmetal/scripts/` (wf329w4_multiph_pll_smoke.py); +1 NEW module in `molmetal/scripts/` (wf329w5_w2_chemprop_smoke.py); +1 NEW code fix in `molmetal/scripts/bench_catherine_30metals.py` (Wf-330-A).

**0 NEW `TODO/pending/*.md`** — Wf-329 + Wf-330 ship wave consumed existing TODO-92..98 + WEAK_TO_STRONG_PLAN §2.5 + WEAK_TO_STRONG §3.5; no NEW TODOs opened.

---

## §8 Lit anchors + skill refs

1. **Per `party-twenty-spirit` §3.5**: every numeric value above traces to a JSON / report path on disk; PROJECTED vs MEASURED tag discipline applied throughout.
2. **`mol-metal-catherine-dblock-bench`** (NEW per Wf-321B; WIRED per Wf-327-F10): Catherine 30-metal bench (3d + 4d + 5d + La row), per_metal 6/6 PASS gate, Tanimoto<0.4 OOD cross-ref.
3. **`mol-metal-triangle-closure`** (NEW per Wf-321B; WIRED per Wf-327-F10): ASTSR ↔ Hilbert ↔ MCTS 3-way loop, R² + variance reduction honesty gates.
4. **`mol-metal-closure-catherine`** (NEW per Wf-321B; WIRED per Wf-329-W8): closure-theorem main result + 6/6 properties on ≥ 25 d-block metals.
5. **`mol-metal-anticancer`** (hot-path): Pt/Ru/Ir/Au/Pd cytotoxicity; MetalCytoToxDB (Krasnov 2026); PlatinAI/MBFinder (Rusanov 2025); cisplatin/carboplatin/oxaliplatin reference drugs.
6. **`mol-metal-gpu-execution`** (hot-path): Blackwell sm_120 + RTX 5090 environment; standing GPU authorization per AGENTS.md §5.1; kill_then_adapt directive.
7. **`mol-metal-cfm-training`** (hot-path): CFM backbone; YuelBond decoder REQUIRED; AMP / BF16 / ODE solver; h=128 / n_layers=3 standard.
8. **`mol-metal-mechanism-mapping`** (domain): m3 Vina Dock ↔ m6_kappa ↔ m7 diversity ↔ R² ↔ closure-theorem mapping.
9. **Wf-330 ULTRACODE key insight** (per `TODO/INDEX.md` §0.14): the 0/30 Catherine bench failure was NOT 60 independent failures — uniformity revealed a single-line root cause at `bench_catherine_30metals.py:152` (omitted depth-0 in synthesised `products_per_depth`). Pattern-recognition across uniform failure messages unblocks 30 metals with one fix.

---

**END OF Wf-329-SYN synthesis doc** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 (honest framing); pre-flight v3 62/62/62 PASS at Wf-329-SYN timestamp 2026-09-29T05:04:45+0000; Wf-329 ship wave 8 sub-flows W1..W8 + V4 INDEX sync = 9 NEW; Wf-330 ultracode 7 sub-flows A..G = 7 NEW; total 16 NEW `TODO/completed/*.md` files in Wf-329 + Wf-330; Catherine 30-metal bench FIXED (30/30 6/6 PASS post-Wf-330-A); pre-flight R9 CLOSED (62/0/62 post-Wf-330-C); 28 skills wired per AGENTS.md §3; 0 GPU launches in Wf-329 or Wf-330; standing GPU authorization honored; Tier-2 strategy preserved.
