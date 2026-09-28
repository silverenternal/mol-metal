# TODO-30 Pitfall Tier-2 — CPU Skeleton Synthesis

**Date**: 2026-09-26
**Workflow**: Wf-7 (Pitfall Tier-2) — TODO-30 Rank 6 / 8 / 14 (P2.3 + P1.3 + P6.3)
**Parent plan**: `TODO/pending/30_pitfall_reinforce_plan.md` (updated)
**Companions**:
- `molmetal/reports/wf_t30_tier2_cpu_done/MASTER.md` (Tier-2 round 1: P6.1+P6.2 / P5.3 / P2.2)
- `molmetal/reports/wf_pitfall_tier2/p63_mmseqs2_followup.md` (P6.3 MMseqs2 stub follow-up)
- `TODO/completed/59_cfm_m3_m5_frontier.md` (Wf-6 sister ship)
- `TODO/completed/41_r17_final_closeout.md` (R17 baseline anchor)

**Status**: Tier-2 Rank 6 P2.3 + Rank 8 P1.3 **shipped (CPU-only)**; Rank 14 P6.3 **partial (Tanimoto OK, MMseqs2 DEFER)**; Rank 7-13 still DEFER.

---

## 1. TL;DR — P2.3 Layered gen + P1.3 Buyable gate + P6.3 OOD protocol

Three TODO/30 Tier-2 CPU-only pitfalls closed in this round (Wf-7):

| # | Deliverable | Path | LOC | Tests | Status |
|---|---|---|---|---|---|
| **P2.3** | Layered gen (scaffold → attach → decorate) | `molmetal/molmetal_lam/lam_chem/layered_generation.py` | **924** | 25 (`test_layered_generation.py`, 542 LOC) | **shipped** (CPU skeleton) |
| **P1.3** | Buyable-reactant gate (AiZynth-as-gate) | `molmetal/molmetal_lam/lam_chem/buyable_gate.py` | **640** | 28 (`test_buyable_gate.py`, 421 LOC) | **shipped** (CPU skeleton) |
| **P6.3** | OOD protocol (Tanimoto<0.4 + MMseqs2 stub) | `molmetal/molmetal_lam/lam_chem/ood_protocol.py` | **481** | 24 (`test_ood_protocol.py`, 376 LOC) | **partial** (MMseqs2 DEFER) |
| **Total** | | **3 NEW modules + 3 test files** | **2 045** | **77 pass / 77 collected** | |

All three import OK (`uv run python -c "from molmetal.molmetal_lam.lam_chem.{layered_generation,buyable_gate,ood_protocol} import *"` returns 3 modules). All **77/77 tests pass** in 0.40s on CPU (RDKit + dataclasses, no torch/CFM). 3 smoke scripts run 0/100 fail.

**Honest framing**: All three are **CPU skeletons** — the API surface + orchestration contract land, but no learned prior is wired (P2.3), no live AiZynthFinder backend (P1.3 falls back to SMARTS), and no MMseqs2 binary (P6.3 pocket axis raises `NotImplementedError`). Real wall-time / diversity / synthesizability lift is **PROJECTED** to R18 GPU eval (deferred per `wf_gpu_recovery_now`).

---

## 2. Per-deliverable (file path + LOC + test count + status)

### 2.1 P2.3 — LayeredGenerationPipeline (Rank 6)

| Surface | Path | LOC | Tests | Status |
|---|---|---|---|---|
| `ScaffoldStage` (Bemis-Murcko) + `AttachStage` (typed R-groups) + `DecorateStage` (stereo) + `LayeredGenerationPipeline` orchestrator | `molmetal/molmetal_lam/lam_chem/layered_generation.py` | **924** | 25 (`test_layered_generation.py`, 542 LOC) | **shipped** |
| Smoke harness | `molmetal/scripts/verify_layered_generation.py` | 206 | — | **shipped** (PASS: 6/6 seeds, 100% validity, 3 distinct) |

**Lit anchors**: Bemis & Murcko 1996 *J. Med. Chem.* 39(15), 2887-2893; RDKit `AssignStereochemistry`; MLC `closure.py` cis/trans NF.

**API contract**: `LayeredGenerationPipeline(seed_smiles).run() → LayeredResult(smiles, scaffold, attachments, valid)`. Each stage independently testable; stage failure short-circuits so orchestrator records failure mode. Pure-RDKit, no torch at runtime.

**Constraint honoured**: `decoder_rework.py` + `lambda_combinators.py` + `batched_rdkit.py` **未修改** — stage classes do not import legacy decoder. Pipeline co-exists as alternative single-pass path.

### 2.2 P1.3 — BuyableReactantGate (Rank 8)

| Surface | Path | LOC | Tests | Status |
|---|---|---|---|---|
| `BuyableReactantGate` (thread-parallel AiZynth oracle) + `GenerateWithGate` (generator wrapper) + `GateMetrics` (pass/total) + `build_buyable_gate` (factory) | `molmetal/molmetal_lam/lam_chem/buyable_gate.py` | **640** | 28 (`test_buyable_gate.py`, 421 LOC) | **shipped** |
| Smoke harness | `molmetal/scripts/verify_buyable_gate.py` | 150 | — | **shipped** (n_tested=100, n_passed=32, pass_rate=0.32 SMARTS fallback) |

**Architecture**: AiZynth-as-gate re-uses `molmetal_lam.sbdd_env.aizynth_adapter.AiZynthAdapter` (R17-ship) as buyability oracle. When `aizynthfinder` wheel + config absent, transparent fallback to `_smarts_fallback` (triazole / imine / cyclohexene / aromatic). Lazy adapter import bypasses heavy `sbdd_env` `__init__`.

**Constraint honoured**: `aizynth_worker.py` / `aizynth_torch_worker.py` / `cisplatin_resistance.py` / `pt_metal_ligand_exchange.py` **未修改**.

### 2.3 P6.3 — OOD protocol (Rank 14, PARTIAL)

| Surface | Path | LOC | Tests | Status |
|---|---|---|---|---|
| `TanimotoOODProtocol` (ECFP4 Morgan-2048, threshold 0.4) + `MMseqs2OODProtocol` (stub) + `CombinedOODProtocol` (AND-combined) + `compute_tanimoto` scalar helper | `molmetal/molmetal_lam/lam_chem/ood_protocol.py` | **481** | 24 (`test_ood_protocol.py`, 376 LOC) | **partial** (Tanimoto axis OK; MMseqs2 axis DEFER) |

**Lit anchors**: Peng 2022 Pocket2Mol (Tanimoto<0.4 ligand-side); TargetDiff 2023 (same convention); Soeding 2017 MMseqs2 (30%-seq-id pocket-side, NOT shipped).

**Why partial**: `mmseqs` binary not in conda env (`which mmseqs` empty; no `pyMMseqs` in `pyproject.toml`). `MMseqs2OODProtocol.is_ood()` and `max_seq_identity()` raise `NotImplementedError` so callers must explicitly opt in. `CombinedOODProtocol(mmseqs2=None)` is the CPU-only safe default (Tanimoto only). Follow-up install steps in `wf_pitfall_tier2/p63_mmseqs2_followup.md`.

---

## 3. MEASURED vs PROJECTED

### 3.1 MEASURED (this round, Wf-7)

| Metric | Measured | Evidence |
|---|---|---|
| 3 modules import OK | **YES** | `python -c "from molmetal.molmetal_lam.lam_chem.{layered_generation,buyable_gate,ood_protocol} import *"` returns 0 errors |
| P2.3 + P1.3 + P6.3 tests | **77 / 77 pass** | `uv run pytest molmetal/tests/test_{layered_generation,buyable_gate,ood_protocol}.py` → 77 passed in 0.40s |
| P2.3 smoke (`verify_layered_generation.py`) | **PASS** | scaffolds=6, valid=6, distinct=3, pass_rate=100% |
| P1.3 smoke (`verify_buyable_gate.py`, 100 mols) | **PASS** (engine=smarts_fallback) | n_tested=100, n_passed=32, pass_rate=0.32, n_errors=0 |
| P6.3 Tanimoto axis | **PASS** | 24/24 tests; cisplatin vs train NOT OOD; empty train = OOD |
| GPU retrain claimed | **NO** | Pure CPU (RDKit + dataclasses + json + argparse). Zero torch/CFM touched. |

### 3.2 PROJECTED (deferred, R18 GPU eval)

| Metric | Target | Gate |
|---|---|---|
| P2.3 layered gen wall-time vs single-pass | **≤ 1.0×** at 1K candidates (3-stage overhead ≤ single-pass) | R18 Phase 2 GPU eval |
| P2.3 layered gen diversity lift | **+5-10%** distinct Murcko scaffolds vs single-pass | R18 Phase 2 GPU eval |
| P1.3 buyability gate pass-rate | **≥ 60%** at 1K candidates with live AiZynthFinder (vs 32% SMARTS fallback measured) | R18 GPU eval + `conda install -c bioconda aizynthfinder` |
| P1.3 false-positive rate (true OOD rejected) | **< 5%** vs AiZynthFinder oracle | R18 GPU eval |
| P6.3 pocket-axis OOD claim | **MMseqs2 30%-seq-id** between train / test pockets | R13 / R14 follow-up: `conda install -c bioconda mmseqs2` |
| P6.3 dual-axis combined | **Tanimoto<0.4 AND MMseqs2<30%** → strict dual-axis OOD | R18 GPU eval after MMseqs2 binary install |

---

## 4. Pitfall status update (TODO-30 progress)

| Tier | Rank | ID | Pitfall | Status | When |
|---|---|---|---|---|---|
| **Tier-1** | 1 | P3.3 | Pareto ranker wire-in | **DONE** | R17 2026-09-17 (`wf_t30_pareto_wire`) |
| **Tier-1** | 2 | P4.2 | Patent / known-Pt similarity axis | **DONE** | R17 2026-09-17 (`wf_t30_patent_axis`) |
| **Tier-1** | 3 | P1.4 | Known-Pt regression tests | **DONE** | R17 2026-09-17 (`wf_t30_known_pt_regression`) |
| **Tier-1** | 4 | P5.2 | Wet-lab plumbing Tier 1 | **DONE** | R17 2026-09-17 (`wf_t30_wetlab_plumbing`) |
| **Tier-1** | 5 | P2.5 | Pharmacophore wire-in | **DONE** | R17 2026-09-17 (`wf_t30_pharmacophore_wire`) |
| **Tier-1** | 7 | P2.2 | Always-on 3D default flip | **DONE** | R17 2026-09-17 (`wf_t30_p22_default_3d`) |
| **Tier-1** | 9 | P5.3 | Rule registry public API | **DONE** | R17 2026-09-17 (`wf_t30_p53_rule_registry`) |
| **Tier-1** | 14a | P6.1+P6.2 | Reproducibility (scaffold split + per-run hygiene) | **DONE** | R17 2026-09-17 (`wf_t30_p62_scaffold_seed`) |
| **Tier-2** | **6** | **P2.3** | **Layered gen (scaffold→attach→decorate)** | **DONE ✅ NEW** | **Wf-7 2026-09-26 (this round)** |
| **Tier-2** | **8** | **P1.3** | **Buyable-reactant gate (AiZynth-as-gate)** | **DONE ✅ NEW** | **Wf-7 2026-09-26 (this round)** |
| **Tier-2** | **14** | **P6.3** | **OOD protocol (Tanimoto<0.4 + MMseqs2 30%)** | **PARTIAL (Tanimoto OK, MMseqs2 DEFER)** | **Wf-7 2026-09-26 (this round)** |
| **Tier-2** | 10 | P1.2 | FG-compat / leave-group veto | **DEFER** | needs property tests first |
| **Tier-2** | 11 | P1.1 | Non-click med-chem rules | **DEFER** | Round-15+ scope (expansion not bug fix) |
| **Tier-2** | 12 | P2.4 | Skeleton jump / fragment-swap | **DEFER** | after P2.3; high invalid-structure risk |
| **Tier-2** | 13 | P2.5 Phases C-E | MD relaxation (OpenMM `relax_mol`) | **DEFER** | needs ROCm OpenMM validation |
| **Tier-2** | 14 | P6.3 pocket-axis | MMseqs2 30%-seq-id | **DEFER** | binary not installed (partial ship in this round) |
| **Tier-2** | 15 | P5.2 Tier 3 | Wet-lab outreach (3 collaborators) | **DEFER** | needs chemistry facility |
| **Tier-2** | 16 | P4.1 | Generalisation 100×3 sweep | **DEFER** | GPU-blocked per `wf_gpu_recovery_now` 2026-09-15 |

**Summary**: Tier-1 fully done (8/8). Tier-2 partial: 2 done + 1 partial (this round); 5 still DEFER (Ranks 10, 11, 12, 13, 14-pocket, 15, 16).

---

## 5. 不修改承诺 (no-modify commitments honoured)

Per Wf-7 spec, the following files were **NOT modified** in this round:

- `molmetal/molmetal_lam/lam_chem/decoder_rework.py` — R17 Path-B single-pass decoder preserved. P2.3 layered pipeline co-exists as alternative.
- `molmetal/molmetal_lam/lam_chem/lambda_combinators.py` — 42-entry `PRIMITIVE_ATOMS` preserved (LambdaASTSR regression watch per `mol-metal-lambda-calculus` skill).
- `molmetal/molmetal_lam/lam_chem/batched_rdkit.py` — parallel RDKit wrapper preserved.
- `molmetal/molmetal_lam/sbdd_env/aizynth_worker.py` + `aizynth_torch_worker.py` — gate re-uses `AiZynthAdapter`, never imports worker scripts.
- `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` + `cisplatin_resistance.py` — not touched by P1.3 (only used for SMARTS fallback reference).

**Edit invariant**: 3 NEW modules + 3 NEW test files + 2 NEW smoke scripts only. Zero edits to existing chemistry code paths.

---

## 6. Cross-reference: skills

- **mol-metal-lambda-calculus** (`/home/hugo/codes/mol-metal/.claude/skills/mol-metal-lambda-calculus/SKILL.md`) — §R18 ship reference cites **Wf-7 (Pitfall Tier-2)** as a regression-watch trigger: any change to `lambda_astsr.py` or `coupling_adapter.py` must re-verify the +0.05 ablation gate on PlatinAI 1K 70/30 hold-out (`molmetal/reports/wf_coupling_ablation/final.md`). P2.3 layered pipeline preserves MLC `closure.py` cis/trans NF distinction (`AssignStereochemistry` pairs with closure theorem).
- **mol-metal-click-chemistry** (`/home/hugo/codes/mol-metal/.claude/skills/mol-metal-click-chemistry/SKILL.md`) — P1.3 buyable gate SMARTS fallback covers click motifs (triazole / imine / cyclohexene / aromatic) for conservative synthesis pre-filtering. Click-rule compatibility matrix (`COMPAT_MATRIX` in `pt_click_compat.py`) not modified; P1.3 reads from it but never alters it.

---

## 7. Verify

```bash
uv run python -c "from molmetal.molmetal_lam.lam_chem.layered_generation import LayeredGenerationPipeline; from molmetal.molmetal_lam.lam_chem.buyable_gate import BuyableReactantGate; from molmetal.molmetal_lam.lam_chem.ood_protocol import TanimotoOODProtocol; print('3 modules import OK')"
uv run pytest molmetal/tests/test_layered_generation.py molmetal/tests/test_buyable_gate.py molmetal/tests/test_ood_protocol.py -v
uv run python molmetal/scripts/verify_layered_generation.py --smoke
uv run python molmetal/scripts/verify_buyable_gate.py --smoke --n-test 100
```

All CPU-only, <1s end-to-end, no GPU / CFM / aizynthfinder dependency at test time.
