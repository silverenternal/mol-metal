# TODO-119 — Decode_ratio strengthen via YuelBond decoder + Wasserstein gradient flow

**Date**: 2026-09-27
**Status**: 🚀 **Active — fixes the biggest honest-negative (decode_ratio 0/8 with 4-fix bundle stacked)**
**Priority**: **P0 CRITICAL** (m14 decode_ratio is the canonical CFM validity claim — currently fails)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (T5 Multi-PH) + `mol-metal-cfm-training`

---

## §0 Why this plan exists

**Current state** (decode_ratio.json, 2026-09-27):
- 4-fix bundle (Path A with decoder_rework + h=64 + midpoint solver + joint_train) lifted n_distinct 1 → 20 ✅
- BUT same 4-fix bundle lifted decode_ratio from baseline 0.0 to **still 0.0** (0/8) ❌
- Path B bond-bearing decoder = 1.0 pre-RDKit-strict; Path B RDKit-strict = 0.5
- Verdict: `MEASURED_NEGATIVE_RESULT — CFM as currently architected does not lift decode_ratio above 0 on synthetic or trained-CFM coordinates`

**Why**: 4-fix bundle repaired the **diversity attractor** (the multi-attractor singleton trap) but did NOT fix the **EGNN velocity field architecture** that produces off-manifold samples (overvalent Pt atoms in synthetic clouds → RDKit sanitization fails).

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU + GPU-0 Blackwell (full retrain) |
| Wall-time | 4 hr CPU + 2-3 days GPU |
| Output | YuelBond decoder swap + Wasserstein regularization + new CFM training |
| Tests | ~25 new tests |

## §2 Solution path (3 innovations)

1. **YuelBond decoder swap** (available but not wired per `wf_cfm_rescue/phase1_yuelbond.md`)
   - Replace default bond decoder with YuelBond module that respects valence constraints at the atom-graph level
   - `use_yuelbond=True` flag, deferred to follow-up retrain
2. **Wasserstein gradient flow regularization** (theorem `wasserstein_gradient_flow.py`)
   - Add W₂ gradient flow penalty to CFM velocity field training
   - Force the learned flow to stay on the RDKit-valid manifold
3. **Catherine closure-theorem-bounded generation**
   - Replace unconstrained atom-bond generation with closure-theorem-bounded generation (5-click productive space)
   - Every generated atom-bond sequence must satisfy Catherine 6/6 closure properties

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Wire YuelBond decoder swap | 4 hr CPU | `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` (NEW, ~250 LOC) + 8 tests | pytest PASS; smoke 100 samples |
| **P1.2** Wasserstein gradient flow regularization | 1 day CPU | `molmetal/molmetal_lam/lam_chem/cfm_wasserstein_loss.py` (NEW, ~200 LOC) + 6 tests | pytest PASS |
| **P1.3** CFM full retrain with YuelBond + Wasserstein + Closure | 2 days GPU-0 | `molmetal/reports/wf119_cfm_retrain/` (NEW) | decode_ratio RDKit-strict ≥ 0.5 (Path B) → ≥ 0.8 (with new fixes) |
| **P1.4** Compare against baselines | 1 day CPU | `synthesis.md` | honest framing of new vs Path A/B |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: YuelBond swap test PASS + Wasserstein loss finite + per-epoch decode_ratio
- **PROJECTED**: decode_ratio RDKit-strict ≥ 0.8 with YuelBond + Wasserstein + Closure (vs current 0.5 Path B)

## §5 Honest framing

This is the **biggest honest-negative** in the system. 4-fix bundle was supposed to fix CFM, but it only fixed diversity. The architecture lever (YuelBond decoder) and the regularization lever (Wasserstein) are unproven on this code path. **There is a real chance P1.3 still produces 0.0**. The fix is honest reporting of which path actually moves the needle.

## §6 Cross-references

- `metrics/by_metric/decode_ratio.json` (current state)
- `molmetal/molmetal_lam/lam_chem/cfm_decoder_rework.py` (current decoder)
- `molmetal/math_skills/wasserstein_gradient_flow.py` (theorem)
- `molmetal/math_skills/catherine_properties.py` (closure theorem)
- `wf_cfm_rescue/phase1_yuelbond.md` (the available-but-not-wired path)
- TODO-22 metric #14 (CFM decode — what this fixes)
- TODO-84 paper rule (paper §4 not blocked)
- LAMBDA_ROADMAP §4.2 m14 (CFM decode) column