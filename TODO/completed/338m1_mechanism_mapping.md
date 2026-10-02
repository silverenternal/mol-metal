# Wf-338-M1 — Mechanism mapping skill wire + per-metal reachability matrix

**Date (UTC)**: 2026-09-29
**Workflow**: Wf-338-M1 (mechanism mapping sub-phase)
**Type**: Documentation-only wire + test module + skill SKILL.md extension

## Summary

Wire the `mol-metal-mechanism-mapping` skill (per AGENTS.md §3.2) and add the
**Wf-330-C4 per-metal reachability matrix** anchor that ties this skill to the
L1 / L6 / L12 closure-theorem benchmarks. CPU-only docs + tests; no GPU
retrain; no closures benchmarks re-run.

## What shipped

1. **Skill fully wired** — verified `.claude/skills/mol-metal-mechanism-mapping/SKILL.md`
   already had the R17 + R18 + R19 + R20 + Wf-45 L9-L12 sections (~555 lines).
   No STUB. Skill is production-ready per R17 close-out.

2. **Per-metal reachability matrix section** — appended to `SKILL.md` after
   the Wf-45 L9-L12 section. Pins the 4 Wf-330-C4 anchors + the per-metal
   diagnostic flow:

   | L | Reachability | Description | Source |
   |---|---:|---|---|
   | L1 | **0.5108** | closure verifier (Wf-36 P1) | `molmetal/reports/wf36_closure_theorem/closure_bench.json` |
   | L6 | **0.5891** | closure d-block 30 metal (Wf-37 P2) | `molmetal/reports/wf37_closure_dblock/per_metal_bench.json` |
   | L12 PlatinAI 5K | **0.5581** | closure OOD (Wf-41 P2) | `molmetal/reports/wf41_closure_ood/ood_closure.json` |
   | L12 MetalCytoTox 5-metal | **0.0300** | closure OOD honest-negative | same as L12 PlatinAI |

3. **Per-metal rows** — added 9 precious-metal rows for L6 (Pt / Ru / Ir / Rh
   / Re / Os / Au / Pd / Ag) + 7 PlatinAI 5K rows for L12 (Pt / Pd / Ru / Rh
   / Ir / Os / Au) + 5 MetalCytoTox 5-metal rows for L12 (Ru / Ir / Rh / Os
   / Re). All reachability rates cross-validated against `per_metal_bench.md`
   + `ood_closure.md` on disk.

4. **SKILL.md cross-reference cleanup** — updated stale references from
   `TODO/MECHANISM_METRIC_MAPPING.md` → `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`,
   and from `TODO/pending/82_metric_gap_filling_plan.md` →
   `TODO/completed/82_metric_gap_filling.md` (and similar for 83 / 91). All
   cross-references now resolve to real on-disk paths.

5. **Test module** — `molmetal/tests/test_wf338m1_mechanism_mapping.py` with
   **30 tests** covering:
   - Test 1 (6 tests): SKILL.md has required sections + frontmatter + R17 state
     + when-to-use table + R18 wave tables + per-metal reachability section
   - Test 2 (3 tests): cross-references resolve + bench anchors present + R20-P5
     stretch-gate honest framing
   - Test 3 (4 tests): activation keywords + symptom-action table + 5-step
     diagnostic procedure + 3-command cheatsheet
   - Test 4 (12 tests): per-metal reachability matrix with 4 anchor rates +
     per-metal rows for L6 (9 metals) + L12 PlatinAI (7 metals) all
     cross-validated against bench JSON/MD on disk
   - Test 5 (3 tests): diagnostic chain integrity (MECHANISM_METRIC_MAPPING.md +
     METRICS_TARGETS.md + EVALUATORS_MIGRATED.md exist)

## Test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wf338m1_mechanism_mapping.py -v
============================== 30 passed in 0.20s ==============================
```

## Honest framing (per AGENTS.md §1.5 / party-twenty-spirit §3.5)

- All 4 reachability anchors (L1 / L6 / L12 PlatinAI / L12 MetalCytoTox) are
  **MEASURED @ Wf-36 / Wf-37 / Wf-41 ship** (JSON + MD on disk); the Wf-338-M1
  work is **CPU-only documentation wire** (no retrain, no re-benchmark).
- L12 MetalCytoTox 5-metal = **0.0300 is honest-negative** (ligand-only SMILES
  coverage gap, NOT a bug to chase). Tagged accordingly in the SKILL.md.
- R20-P5 stretch gate ≥ 0.95 is **35.9 pp short of L6 = 0.5891 baseline**;
  honest per `LAMBDA_ROADMAP.md` §1.4 is 70-80% realistic, NOT 95%.
- Per-periodic-block L6 split: 3d = 0.6682 > 4d = 0.5813 > 5d = 0.4918. The
  5d_lanthanide 0.4918 is honest-negative precondition for R20-P5 (some 5d
  metals like Tc, Pm, lanthanide closure rate < 50%).
- SKILL.md cleanup: 3 stale `TODO/pending/8X_*` paths updated to
  `TODO/completed/8X_*` paths (those TODOs have shipped since the original
  SKILL.md was written). No semantic drift; just path rotation to match the
  current on-disk state.

## Files touched

| Path | Change |
|---|---|
| `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` | Appended "Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)" section (~110 lines); updated 6 stale path references |
| `molmetal/tests/test_wf338m1_mechanism_mapping.py` | NEW: 30 tests covering SKILL.md wire + per-metal matrix |
| `TODO/completed/338m1_mechanism_mapping.md` | NEW: this file |

## Cross-reference

| Anchor | Path | Status |
|---|---|---|
| Wf-330-C4 per-metal reachability matrix | `TODO/completed/330_c4_per_metal_reachability.md` | ship (this Wf) |
| Wf-36 P1 closure verifier (L1 = 0.5108) | `molmetal/reports/wf36_closure_theorem/closure_bench.md` | MEASURED 2026-09-26 |
| Wf-37 P2 closure d-block (L6 = 0.5891) | `molmetal/reports/wf37_closure_dblock/per_metal_bench.md` | MEASURED 2026-09-26 |
| Wf-41 P2 closure OOD (L12 PlatinAI = 0.5581) | `molmetal/reports/wf41_closure_ood/ood_closure.md` | MEASURED 2026-09-26 |
| Wf-41 P2 closure OOD MetalCytoTox = 0.0300 | same as L12 PlatinAI | MEASURED honest-negative |
| Wf-45 TODO-83 close-out (L1-L8 ship) | `TODO/completed/83_lambda_algorithm_enhancement_plan.md` | shipped 2026-09-26 |
| Wf-45 TODO-91 close-out (L9-L12 ship) | `TODO/completed/91_wf43_phase4_integration.md` | shipped 2026-09-26 |

## Self-check

```bash
# Verify all 30 tests pass
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf338m1_mechanism_mapping.py -v

# Verify SKILL.md cross-references resolve
grep -c "TODO/02_metrics/" .claude/skills/mol-metal-mechanism-mapping/SKILL.md

# Verify per-metal matrix anchors
grep -E "0\.5108|0\.5891|0\.5581|0\.0300" \
  .claude/skills/mol-metal-mechanism-mapping/SKILL.md | head -10
```

## Status

**SHIPPED** — 2026-09-29 (Wf-338-M1).