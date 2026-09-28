# TODO-99 — R20 Skill ecosystem (8 NEW skills: 4 R20 + 4 R21)

**Date**: 2026-09-26
**Status**: 🚀 **Active — queued for R20-P6 + R21-P6 (CPU docs, ~4 weeks)**
**Priority**: **MEDIUM** (skill ecosystem expansion — supports all 7 R20-P + 4 R21-P plan items)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 + §2.4 + `TODO/completed/83j_lambda_closeout.md` §5.4
**Lambda gap(s)**: all 12 gaps (skill-level support)
**Skill alignment**: 8 NEW skills across `mol-metal-lambda-calculus` + `mol-metal-r18-math` + `mol-metal-mechanism-mapping` + `mol-metal-anticancer` + `mol-metal-click-chemistry`

---

## §0 Why this plan exists

Current skill ecosystem: 5 R17 (`lambda-calculus`, `cfm-training`, `click-chemistry`, `anticancer`, `mechanism-mapping`) + 3 R18 (`r18-math`, `paper-writing`, `party-twenty-spirit`) + 1 R19 (`gpu-execution`, NEW per Wf-34) = **9 skills total**. **R20 ships 4 NEW skills** + **R21 ships 4 NEW skills** to fully cover the 12 Lambda gaps + 8 R20 plan items. Each new skill cross-references the R17/R18/R19 existing skill matrix.

## §1 Scope — 8 NEW skills

| # | Skill | Plan item | Scope | Wall |
|---:|---|---|---|---|
| **S1** | `mol-metal-variance-puct` | R20-P1 (TODO-92) | VariancePUCT scoring rule + lazy RDKit import + L2 vuln_a fix | 2 wk CPU |
| **S2** | `mol-metal-hilbert-astsr-e2e` | R20-P2 (TODO-93) | Hilbert-ASTSR end-to-end retrain + R² ≥ 0.85 + L5/L7/L11 fix | 2 wk CPU |
| **S3** | `mol-metal-multiph-pll` | R20-P3 (TODO-94) | Multi-PH Fourier 32-bin + PLL signature 128-d + L7 fix | 2 wk CPU |
| **S4** | `mol-metal-3way-integration` | R20-P4 (TODO-95) | Deflexpressor × Multi-PH × ASTSR 3-way + 100-pocket × 10-seed + L10 fix | 2 wk CPU |
| **S5** | `mol-metal-closure-catherine` | R20-P5 (TODO-96) | Closure theorem + Catherine 6 properties + L1/L6/L12 fix | 2 wk CPU |
| **S6** | `mol-metal-wetlab-protocol` | R20-P6 (TODO-97) | Wet-lab IC50 validation protocol + m18/m19/m20 cooperation | 1 wk CPU |
| **S7** | `mol-metal-triangle-closure` | R21-P4 | ASTSR × Hilbert × MCTS triangle closure on real data + L8 fix | 2 wk CPU |
| **S8** | `mol-metal-mcts-replay-learner` | R21-P2 | AlphaGo-style MCTS replay learner GPU trainer + L4 fix | 2 wk CPU |

## §2 Cross-skill matrix (R20+R21)

```
        S1   S2   S3   S4   S5   S6   S7   S8
R17-LC  ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓
R17-CFM ✓    ✓    ✓    ✓    -    ✓    ✓    -
R17-CC  -    ✓    -    ✓    ✓    ✓    ✓    -
R17-AN  -    ✓    ✓    ✓    -    ✓    -    -
R17-MM  ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓
R18-MTH ✓    ✓    ✓    ✓    ✓    -    ✓    ✓
R18-PAP -    ✓    -    ✓    ✓    ✓    ✓    -
R18-PTS ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓
R19-GPU ✓    ✓    ✓    ✓    -    -    ✓    ✓
```

## §3 Phases (per LAMBDA_ROADMAP §1.5 + §2.4)

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **R20-P1-S1** `mol-metal-variance-puct` | 2 wk CPU | `.claude/skills/mol-metal-variance-puct/SKILL.md` (≥ 250 LOC) | SKILL.md PASS review + 3 test scripts |
| **R20-P2-S2** `mol-metal-hilbert-astsr-e2e` | 2 wk CPU | `.claude/skills/mol-metal-hilbert-astsr-e2e/SKILL.md` | same |
| **R20-P3-S3** `mol-metal-multiph-pll` | 2 wk CPU | `.claude/skills/mol-metal-multiph-pll/SKILL.md` | same |
| **R20-P4-S4** `mol-metal-3way-integration` | 2 wk CPU | `.claude/skills/mol-metal-3way-integration/SKILL.md` | same |
| **R20-P5-S5** `mol-metal-closure-catherine` | 2 wk CPU | `.claude/skills/mol-metal-closure-catherine/SKILL.md` | same |
| **R20-P6-S6** `mol-metal-wetlab-protocol` | 1 wk CPU | `.claude/skills/mol-metal-wetlab-protocol/SKILL.md` | same |
| **R21-P2-S8** `mol-metal-mcts-replay-learner` | 2 wk CPU | `.claude/skills/mol-metal-mcts-replay-learner/SKILL.md` | same |
| **R21-P4-S7** `mol-metal-triangle-closure` | 2 wk CPU | `.claude/skills/mol-metal-triangle-closure/SKILL.md` | same |
| **W99 close-out** | 1 wk CPU | `TODO/completed/99_r20_skill_ecosystem.md` | INDEX §17 cross-ref |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = 8 NEW skills shipped + SKILL.md ≥ 250 LOC each + 3 test scripts each
- PROJECTED = all 12 Lambda gaps have a dedicated skill by R21-P6

## §5 Cross-references

- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 (R20 skill ecosystem) + §2.4 (R21 skill ecosystem)
- `TODO/completed/83j_lambda_closeout.md` §5.4 (8 NEW R20+R21 skills)
- `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` → S1
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` → S2
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` → S3
- `TODO/pending/95_r20_p4_3way_100pocket.md` → S4
- `TODO/pending/96_r20_p5_closure_catherine.md` → S5
- `TODO/pending/97_r20_p6_wetlab_ic50.md` → S6
- `TODO/pending/98_wf44_closure_ood_phase2.md` → S5 (shared)
- `TODO/INDEX.md` §17 (5-category audit cross-ref)