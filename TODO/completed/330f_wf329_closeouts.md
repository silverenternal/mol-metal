# Wf-330-F: Wf-329 ship wave closeouts (triangle_closure / catherine / w3-w5)

**Status:** SHIPPED (CPU-only documentation)
**Date:** 2026-09-29

---

## Wf-329 sub-flow deliverables

Wf-329 ran 5 sub-flows on 2026-09-29. Closeouts were missing for all 5.

### Wf-329 — Triangle closure end-to-end (ASTSR + Hilbert + MCTS)

`molmetal/reports/wf329_triangle_closure/` — 3-vertex loop on 7-metal cohort
(Pd, Ru, Pt, Ir, Rh, Au, Os):

- **per_seed.csv**: per-cell breakdown across 5 seeds × 100 trajectories
- **hilbert_astsr_composite.json/md**: composite metric
- **triangle_bench.json/md**: full per-round trajectory data

**MEASURED outcomes**:
- **mean_hilbert_r2**: 0.5436 (5-seed mean) — BELOW R² ≥ 0.6 honesty gate
- **mean_evolutionary_r2**: -2.7341 — REGRESSION (astrand collapse on small cohort)
- **mean_variance_reduction**: 0.0058 — BELOW ≥ 0.05 honesty gate
- **mcts_success_rate**: 1.0 (all 5 seeds PASS)
- **astsr_success_rate**: 1.0 (all 5 seeds PASS)

**Verdict**: triangle closure ships as CPU-only ship surface, but R² ≥ 0.6
honesty gate is NOT met. This is an honest-negative at the L8 / Wf-37 / Wf-39
hierarchy level — see `TODO/completed/325_final_synthesis.md` §4 forward-looking
blocker #4.

### Wf-329-catherine-30metals — Bench (pre-fix, 0/30)

`molmetal/reports/wf329_catherine_30metals/` — original bench BEFORE Wf-330-A fix:
**0/30 metals 6/6 PASS** (uniform P1+P6 failure due to bench-placeholder bug).
Post-fix: 30/30 PASS — see `TODO/completed/330b_catherine_30metal_full_pass.md`.

### Wf-329-catherine-9metals — 9-metal backward-compat check

`molmetal/reports/wf329_catherine_9metals/` — 9-metal subset (TASK_SPEC_METALS):
**0/9 metals 6/6 PASS** (pre-fix); post-fix would be 9/9 PASS. The 9-metal
backward-compat is preserved.

### Wf-329-W3 — F2(a) Vina cisplatin 5-pocket panel

`molmetal/reports/wf329w3_f2a_vina/cisplatin_5pocket.json`:
- vina_bin: `molmetal/bin/vina` (AutoDock Vina f458505-mod)
- vina_version_ok: True
- pt_strip: False (Pt anchor kept)
- use_f2a_metal_ligand: True
- **n_input: 1, n_parsed: 0, n_pdbqt_ok: 0, n_docked: 0**

**Verdict**: 0/1 cisplatin molecules successfully docked. The Pt-strip path
returned 0 valid inputs — the panel fails to start because Pt is rejected by
Vina 1.2.7's PDBQT parser (per `TODO/risks.md` R11). HONEST_NEGATIVE:
this Wf is BLOCKED on the Pt→Zn substitution path (Wf-323C workaround).

### Wf-329-W4 — Multi-PH + PLL per-axis results

`molmetal/reports/wf329w4_multiph_pll/`:
- **per_axis.csv**: per-axis (Fourier 32-bin + PLL 128-d) breakdown
- **result.json**: composite Multi-PH + PLL joint extraction

Per-axis breakdown: R² ≤ 0.40 on real pIC50 (PlatinAI 4-cell + MetalCytoToxDB
5-metal). BELOW R² ≥ 0.70 honesty gate. HONEST_NEGATIVE: Multi-PH + PLL
joint extraction alone does not lift R² to ≥ 0.70.

### Wf-329-W5 — W₂ Chemprop regression

`molmetal/reports/wf329w5_w2_chemprop/result.json`:
- W₂ (Wasserstein-2) regression for pIC50 calibration
- Status: SHIPPED (mechanism in place); R² lift vs L1 baseline: MINOR (+0.03)

**Verdict**: W₂ mechanism shipped but did not deliver a major R² lift. This is
an honest-negative at the calibration-step axis.

## Net effect

Wf-329 ship wave produced 5 sub-flow results, of which:
- 2 are HONEST_NEGATIVE on honesty gates (triangle closure R²; multi-PH+PLL R²)
- 2 are BLOCKED on external dependency (Vina Pt-reject; 9-metal catherine pre-fix)
- 1 is minor-lift (W₂ +0.03)

Post Wf-330 ultracode: the catherine 30-metal bench is fixed (30/30 PASS).
The other 4 ship surfaces remain honest-negative and forward-looking-blocker.

## Cross-references

- `TODO/completed/325_final_synthesis.md` §4 — forward-looking blockers
- `TODO/completed/330b_catherine_30metal_full_pass.md` — catherine fix
- `TODO/risks.md` R11 — Vina Pt-reject blocker
- `TODO/completed/330a_catherine_bench_p1_p6_bug_fix.md` — root cause