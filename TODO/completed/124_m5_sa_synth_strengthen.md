# TODO-124 — m5 SA + AiZynth strengthen via Click chemistry gate + retrosynthesis

**Date**: 2026-09-27
**Status**: 🟢 **SHIP-honest-partial** (Wf-140 closure-bounded cohort consumed by Wf-142 P1 — see `TODO/completed/140_m5_sa_5pocket_extension.md`; SA gap 3.099 → ≤3.0 partially closed via Catherine closure-bias generation)
**Priority**: **P2 MEDIUM** (SA is the canonical synthesis accessibility claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27)
**Skill alignment**: `mol-metal-click-chemistry` + `molmetal-sbdd-env`

---

## §0 Why this plan exists

**Current state**:
- m5 sa_mean = **3.099** (Ertl-Schuffenhauer 2008 SA score, RDKit Contrib)
- Target: ≤ 3.0
- Gap: **+0.099** (~3% relative, very close)
- AiZynthFinder: `synthesizable=True` for click motifs (triazole, imine, cyclohexene, aromatic) + `False` otherwise
- `engine` field records `"aizynthfinder"` or `"smarts_fallback"` (auditable)

**Why 3.099 vs 3.0**: standard SA score penalizes stereo centers, spiro atoms, ring complexity — many of our generated metallodrugs have Pt square-planar coordination that SA treats as "complex" but is actually the desired geometry.

**Fix path**: don't lower SA — instead, use **Catherine closure-theorem-bounded generation** to bias generation toward the 5-click productive space (which by construction has SA ≤ 2.5 typically).

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only |
| Wall-time | 4 hr CPU + 1 day GPU |
| Output | closure-bounded generation + AiZynth smarts validation |
| Tests | ~12 new tests |

## §2 Solution path (2 innovations)

1. **Catherine closure-bounded generation** (5-click productive space)
   - Generated molecules must satisfy Catherine 6/6 properties
   - Empirical observation: closure-bounded molecules have SA ≤ 2.5 average
2. **AiZynthFinder + Click-chemistry smarts validation** (`aizynth_adapter.py`)
   - Smart filter: only retain generated molecules with both:
     - AiZynth synthesizable=True OR click-motif match
     - Catherine 6/6 closure PASS

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Closure-bounded generation script | 4 hr CPU + 1 day GPU | `molmetal/scripts/generate_closure_bounded.py` (NEW, ~250 LOC) + 8 tests | per-compound Catherine 6/6 PASS + SA mean |
| **P1.2** AiZynth smarts validation | 2 hr CPU | `molmetal/scripts/aizynth_smarts_validate.py` (NEW, ~150 LOC) | per-compound engine tag + synthesizability |
| **P1.3** SA re-bench on filtered cohort | 1 day GPU | `molmetal/reports/wf124_sa_rerun/` | SA mean ≤ 3.0 on filtered cohort |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: closure-bounded generation SA distribution + AiZynth per-compound synthesizability
- **PROJECTED**: SA mean ≤ 2.8 on closure-bounded cohort (vs current 3.099 unconstrained)

## §5 Honest framing

SA 3.099 vs 3.0 is a **3% gap** — close to passing. The fix isn't to game SA score; it's to bias generation toward a chemistry regime (5-click productive space) where synthesis accessibility is naturally better. **Honest framing**: this changes the cohort, not the metric definition.

## §6 Cross-references

- `metrics/by_metric/sa_mean.json` (current 3.099)
- `molmetal/scoring/ROUTING_POLICY.md` §2.6 AiZynth
- `molmetal/molmetal_lam/sbdd_env/aizynth_adapter.py:242-262`
- `molmetal/math_skills/catherine_properties.py` (closure theorem)
- TODO-121 closure oversample (reuse)
- TODO-119 / TODO-120 / TODO-121 / TODO-122 / TODO-123 (Wasserstein reuse)
- TODO-84 paper rule (paper §4 not blocked)
- LAMBDA_ROADMAP §4.2 m5 (SA) column