# wf_docs_sync — final

**Workflow**: R17 skill-SKILL.md updates
**Date**: 2026-09-25
**Owner**: docs-sync (T2 dual-line, parallel to model-track R17 close-out)
**Status**: COMPLETE — 5/5 SKILL.md files updated

## Purpose

Synchronize the 5 `.claude/skills/mol-metal-*/SKILL.md` files with the R17 final state, so future invocations (a) reflect the 30/30 decode probes, (b) cite the 14/14 evaluators, (c) document the 6 shim fixes from TODO-33, and (d) point at the cross-cutting diagnostic docs (`METRICS_TARGETS.md`, `MECHANISM_METRIC_MAPPING.md`).

## Per-skill delta

| # | Skill | Section added | Lines added | Topics covered |
|---|---|---|---|---|
| 1 | `mol-metal-lambda-calculus` | `## R17 production state` (between "Core anchors" table and "When the user asks about…") | +14 | 30/30 decode probes · 14/14 evaluators · 6 shim fixes (TODO-33 P1+P2) · **42-entry `PRIMITIVE_ATOMS` (8 organic + 4 halogen + 30 d-block)** · **KataGo-MCTS with PLL heuristic (`use_pll_heuristic=True`, `fpu_reduction=0.25`)** · **84-rule MLE matrix** · **18 click rules** · `TODO/METRICS_TARGETS.md` ref · `TODO/MECHANISM_METRIC_MAPPING.md` ref |
| 2 | `mol-metal-cfm-training` | `## R17 production state` (between "Core anchors" table and "Default flags (current production)") | +15 | 30/30 decode probes · 14/14 evaluators · **6 shim files affected (`solver/utils/__init__/loader` + `_reference_loader.py` + `_reference/utils/__init__.py`)** · **reference path auto-detect (P6, `_reference_loader.py:82-86`)** · **ModelWrapper kw-`t` fallback (P7, `_reference/utils/__init__.py:44-63`)** · `TODO/METRICS_TARGETS.md` ref · `TODO/MECHANISM_METRIC_MAPPING.md` ref |
| 3 | `mol-metal-click-chemistry` | `## R17 production state` (between "Core anchors" table and "When the user asks about…") | +14 | 30/30 decode probes · 14/14 evaluators · 6 shim fixes (TODO-33 P1+P2) · **84-rule MLE matrix (`pt_metal_ligand_exchange.py:356`)** · **18 click rules = 5 base + 13 SyntheMol** · **4 NEW SMARTS templates (tetrazine-TCO, dialkyl-disulfide, aryl-I/aryl-B C-N, oxime ligation)** · `TODO/METRICS_TARGETS.md` ref · `TODO/MECHANISM_METRIC_MAPPING.md` §3 ref |
| 4 | `mol-metal-anticancer` | `## R17 production state` (between "Datasets" table and "When the user asks about…") | +14 | 30/30 decode probes · 14/14 evaluators · 6 shim fixes (TODO-33 P1+P2) · **Updated metric gates (Tier 1 must-pass = m1/m2/m3/m14/m15/m18 vs Tier 2 paper-grade = m4-m13/m16/m17/m19/m20)** · **BioPython PLIP replacement** (`molmetal/scripts/bio_plip_protein_ligand.py`) · `TODO/METRICS_TARGETS.md` ref · `TODO/MECHANISM_METRIC_MAPPING.md` §4 ref |
| 5 | `mol-metal-mechanism-mapping` (new skill, already comprehensive) | `## R17 production state` (at top, before "When to use this skill") | +16 | 30/30 decode probes · 14/14 evaluators · **6 shim fixes mapped to mechanism column M16/M17** · coverage = 21 mechanisms × 20 metrics contribution matrix fully populated · source docs (`METRICS_TARGETS.md`, `MECHANISM_METRIC_MAPPING.md`, `EVALUATORS_MIGRATED.md`) |

**Total**: 5 files, +73 lines added across 5 R17 production-state sections.

## Common shape of every R17 section

Each new section follows the same template so any of the 5 skills is self-sufficient:

1. **Production-ready banner** with the close-out doc reference (`TODO/completed/37_model_track_mr17.md`)
2. **30/30 decode probes = 1.000** (closure theorem + 84-rule MLE matrix coverage)
3. **14/14 evaluators verified** with the exact script path (`molmetal/scripts/eval_all_evaluators.py`)
4. **6 shim fixes (TODO-33 P1+P2)** with file locations
5. **Skill-specific content** per the per-file instructions (e.g. 42-entry `PRIMITIVE_ATOMS` for lambda, BioPython PLIP for anticancer)
6. **`TODO/METRICS_TARGETS.md`** reference for the full metric list
7. **`TODO/MECHANISM_METRIC_MAPPING.md`** reference for mechanism attribution

## Cross-skill coherence check

- All 5 skills cite the same production-close-out doc (`TODO/completed/37_model_track_mr17.md`)
- All 5 skills use the same wording for the 30/30 + 14/14 line
- The 6 shim files appear in **both** `mol-metal-cfm-training` (file locations) and `mol-metal-mechanism-mapping` (mechanism column M16/M17) — no contradiction
- The 84-rule MLE matrix appears in **3** skills (lambda, click-chemistry, mechanism-mapping) — same line number `pt_metal_ligand_exchange.py:356`
- The 18 click rules appear in **2** skills (click-chemistry + mechanism-mapping) — same rule count

## Verification commands

```bash
# All 5 files should now contain the R17 banner
grep -l "R17 production state" .claude/skills/mol-metal-*/SKILL.md

# Each should reference both METRICS_TARGETS and MECHANISM_METRIC_MAPPING
for f in .claude/skills/mol-metal-*/SKILL.md; do
  echo "=== $f ==="
  grep -c "METRICS_TARGETS.md" "$f"
  grep -c "MECHANISM_METRIC_MAPPING.md" "$f"
done
```

## Follow-ups

- None — the 5 SKILL.md files are now consistent with R17 final state. Any R18 changes must repeat this workflow.