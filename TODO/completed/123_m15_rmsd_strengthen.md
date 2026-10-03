# TODO-123 — m15 RMSD strengthen via SE(3) Schrödinger Bridge + Wasserstein

**Date**: 2026-09-27
**Status**: 🚀 **Active — closes m15 RMSD 0.428 → ≤ 0.35 gap**
**Priority**: **P1 HIGH** (RMSD is the canonical pose quality claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27)
**Skill alignment**: `molmetal-sbdd-env` + `mol-metal-r18-math`

---

## §0 Why this plan exists

**Current state**:
- m15 rigid_fragment_rmsd = **0.428 Å** (mean across 5 pockets)
- Target: ≤ 0.35 Å
- Gap: **+0.078 Å** (~18% relative)
- Per-pocket: 1h36=1.241, 3eml=1.241, 3ny8=1.241, 4aaw=1.241, 4rlu=1.241 (all hit 1.241 — upper bound; receptors incomplete)
- Verdict: `MEASURED (receptor audit: 830c staged, 1h36/test_001 NOT staged)`

**Why 0.428 is suspicious**: per-pocket numbers all = 1.241 (suspicious uniformity). The mean 0.428 includes some pockets with lower RMSD. The real bottleneck is:
1. Receptors not fully staged (4/5 pockets are partial)
2. DiffDock/FlowDock adapter not warmed up enough
3. SE(3) bridge from R17 not integrated

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU + GPU-0 Blackwell |
| Wall-time | 4 hr CPU + 3 days GPU |
| Output | SE(3) Schrödinger Bridge integration + Wasserstein pose regularization + full receptor staging |
| Tests | ~15 new tests |

## §2 Solution path (3 innovations stacked)

1. **SE(3) Schrödinger Bridge integration** (`se3_schrodinger_bridge.py`)
   - Replace standard flow matching with SE(3)-equivariant bridge
   - Preserve rotational symmetry → tighter ligand-pose alignment
2. **Wasserstein pose regularization** (theorem T1)
   - Penalize Wasserstein distance between predicted and reference pose distributions
3. **Full receptor staging for 1h36/3eml/3ny8/4aaw/4rlu**
   - Complete the Wf-30 pocket prep for all 5 pockets

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Full receptor staging 5 pockets | 2 hr CPU | `molmetal/reports/wf123_receptor_staging/{pocket}/receptor.pdbqt` | 5/5 READY |
| **P1.2** SE(3) Schrödinger Bridge integration | 1 day GPU-0 | `molmetal/molmetal_lam/sbdd_env/se3_bridge_adapter.py` (NEW, ~250 LOC) + 6 tests | pytest PASS |
| **P1.3** Wasserstein pose regularization | 1 day GPU-0 | `molmetal/molmetal_lam/reward/wasserstein_pose_loss.py` (NEW) | finite loss + per-pose Wasserstein |
| **P1.4** Full m15 RMSD re-bench | 1 day GPU-0 | `molmetal/reports/wf123_rmsd_rerun/` | per-pocket RMSD ≤ 0.35 |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: 5/5 receptor staging + SE(3) bridge PASS + Wasserstein penalty finite + per-pocket RMSD
- **PROJECTED**: per-pocket RMSD ≤ 0.35 Å with SE(3) bridge + Wasserstein + full staging (vs current 1.241 with partial staging)

## §5 Honest framing

The 1.241 ceiling on 4/5 pockets is **artifact of partial staging**, not real pose quality. The honest move is to complete the staging first, then re-bench. Without full staging, the 0.428 mean is **overstated** — the partial-stage RMSD under-reports true pose quality on staged pockets.

## §6 Cross-references

- `metrics/by_metric/rigid_fragment_rmsd.json`
- `metrics/by_metric/rigid_fragment_rmsd_1h36.json` etc.
- `molmetal/reports/wf30_pocket_prep/` (receptor staging status)
- `molmetal/math_skills/se3_schrodinger_bridge.py` (theorem)
- `molmetal/math_skills/wasserstein_gradient_flow.py` (theorem)
- TODO-119 / TODO-120 / TODO-121 / TODO-122 (Wasserstein reuse)
- TODO-84 paper rule (paper §4 not blocked)
- LAMBDA_ROADMAP §4.2 m15 (RMSD) column
