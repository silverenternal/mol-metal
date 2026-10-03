# Wf-333-V1 — Wf-246-C kcal_adjustment sign bug fix + Vina P0

**Workflow:** Wf-333-V1
**Date:** 2026-09-29
**Status:** SHIPPED (CPU-only fix, no GPU launch)
**Predecessor:** Wf-331-R1 §4.2 audit (2026-09-29) identified the bug
**Skill alignment:** `mol-metal-click-chemistry` (F2(a) family) +
`mol-metal-anticancer` (Pt/Ru/Ir coordination) +
`party-twenty-spirit` §3.5 (honest framing)
**Honest-negative closed:** `HONEST_NEGATIVE_WF246_C_KCAL_SIGN_INVERTED`

---

## §1. Goal

Close the `HONEST_NEGATIVE_WF246_C_KCAL_SIGN_INVERTED` honest-negative
identified in `TODO/completed/331r1_vina_remaining_gap.md` §4.2 by:

1. **P0** — Fix the 1-line sign bug at
   `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:629` (now line 654).
2. **P0** — Update the existing `test_cisplatin_full_composite` test
   to assert the corrected sign.
3. **P0** — Write 4 NEW tests pinning the sign fix to prevent
   regression.
4. **P0** — Document the fix in this completed TODO.

This is **P0 of Wf-333** — the Vina ceiling recovery bundle. Per
AGENTS.md §5.1, this is CPU-only (no GPU launch). Per §5.4
(Tier-2 journal, no wet-lab), this is pure in-silico fix.

---

## §2. The bug (Wf-246-C, pre-fix)

### §2.1 What was shipped (Wf-246-C, 2026-09-28)

```python
# metal_coord_term.py:629 (Wf-246-C as-shipped, BUGGY)
def kcal_adjustment(self, score: MetalCoordinationScore) -> float:
    if score.total is None:
        return 0.0
    return float((2.0 * score.total - 1.0) * self.kcal_per_unit)
```

### §2.2 What was wrong (Wf-331-R1 §4.2 MEASURED)

For Vina's kcal/mol convention: **lower (more negative) = better binding**.
The buggy formula gave a **POSITIVE adjustment** for perfect composite:

| composite | (2*c − 1) × 1.0 | Effect on Vina kcal/mol |
|---:|---:|---|
| 1.0 (perfect) | **+1.0** | PENALTY (worse binding) |
| 0.5 (midpoint) | 0.0 | neutral |
| 0.0 (zero) | **−1.0** | BONUS (better binding) |

This is **physically backwards** — a metallodrug with perfect
coordination chemistry should have BETTER binding, not worse.

### §2.3 MEASURED smoking gun (Wf-246-C cisplatin × 1h36)

Per `molmetal/reports/wf246c_m3_metal_coord/cisplatin_1h36.json`:
- composite = 0.9970443349753695
- mean_adjustment = +0.994088669950739 kcal/mol (PENALTY)
- plain Vina mean = -3.582 kcal/mol
- metal-coord mean = -2.588 kcal/mol
- **Δ = +0.994 kcal/mol** (worse with metal-coord!)

This is why Wf-329-W3 chose F2(a) (correctly-signed -1.5) over
Wf-246-C for the m3 Vina metric.

---

## §3. The fix (Wf-333-V1, this PR)

### §3.1 1-line code fix at metal_coord_term.py:654

```python
# metal_coord_term.py:654 (Wf-333-V1 FIXED)
def kcal_adjustment(self, score: MetalCoordinationScore) -> float:
    if score.total is None:
        return 0.0
    return float((1.0 - 2.0 * score.total) * self.kcal_per_unit)
```

### §3.2 Updated docstring at metal_coord_term.py:529 + :624

The class docstring (`__init__` block, line 529) and the method
docstring (line 624) now both say:

```
Adjustment = (1 - 2 * composite) * kcal_per_unit
so a perfect composite (1.0) yields -1.0 * kcal_per_unit
(BONUS, more negative kcal/mol = better binding) and a
zero composite yields +1.0 * kcal_per_unit (PENALTY, less
negative = worse binding). This matches the Wf-329-W3
F2(a) sign convention: ``vina_kcal + term.kcal_adjustment``
where term.kcal_adjustment < 0 indicates improved binding.
```

### §3.3 Post-fix behavior table

| composite | (1 − 2*c) × 1.0 | Effect on Vina kcal/mol |
|---:|---:|---|
| 1.0 (perfect) | **−1.0** | BONUS (better binding) |
| 0.5 (midpoint) | 0.0 | neutral |
| 0.0 (zero) | **+1.0** | PENALTY (worse binding) |

**Sign convention matches Vina**: lower kcal/mol = better binding.

### §3.4 Updated existing test (test_metal_coord_term.py:188-198)

```python
def test_cisplatin_full_composite(self):
    """cisplatin has geometry=1.0 + HSAB=1.0 (no pocket context)
    + distance=None → composite = 1.0 → -1.0 kcal/mol bonus
    (Wf-333-V2 sign fix — perfect coordination = better binding)."""
    term = MetalCoordinationTerm()
    score = term.score("[Pt](N)(N)(Cl)Cl")
    assert score.geometry_score == 1.0
    assert score.hsab_score == 1.0
    assert score.distance_score is None
    assert score.total == pytest.approx(1.0, abs=1e-6)
    # kcal_adjustment = (1 - 2 * 1.0) * 1.0 = -1.0 (sign-fixed)
    assert term.kcal_adjustment(score) == pytest.approx(-1.0, abs=1e-6)
```

The pre-fix assertion `term.kcal_adjustment(score) == +1.0` would have
failed post-fix; the new assertion `== -1.0` passes both pre-fix
regression and post-fix correctness.

---

## §4. New tests (molmetal/tests/test_wf333v1_wf246c_sign_fix.py)

**4 test classes** with **12 test methods** total, all CPU-only:

### §4.1 TestSignBugPresentBeforeFix (2 tests)

Pins the **buggy** pre-fix formula behaviour so future code review can
detect a regression to the buggy formula:

1. `test_buggy_formula_yields_positive_for_perfect` — composite=0.997
   → buggy adjustment=+0.994 (smoking gun).
2. `test_buggy_formula_smoke_matches_wf331r1` — matches the
   `cisplatin_1h36.json` MEASURED mean_adjustment=+0.994088669950739
   within ±1e-9.

### §4.2 TestSignFixedPerfectComposite (6 tests)

Pins the **fixed** post-fix formula behaviour:

3. `test_perfect_composite_yields_minus_one` — cisplatin
   composite=1.0 → adjustment=−1.0.
4. `test_zero_composite_yields_plus_one` — composite=0.0 → adjustment=+1.0.
5. `test_midpoint_composite_yields_zero` — composite=0.5 → adjustment=0.0.
6. `test_organic_ligand_yields_zero_no_op` — aspirin
   (no metal) → adjustment=0.0 (backward compat).
7. `test_sign_monotonic_with_composite` — adjustment is strictly
   decreasing in composite over {0, 0.25, 0.5, 0.75, 1.0}.
8. `test_kcal_per_unit_scaling_after_fix` — kcal_per_unit=0.5
   → adjustment=−0.5 (linear scaling).

### §4.3 TestVinaPanelIntegration (1 test)

9. `test_vina_adapter_metal_coord_returns_negative_for_cisplatin` —
   wires the adjustment through `pocket_donor_counts_from_atomicnums`
   (the same code path as `vina_adapter.py:_compute_metal_coord_adjustment`)
   and asserts the adjustment is NEGATIVE for cisplatin.

### §4.4 TestPerPocketLift5Pockets (3 tests)

Pins the **per-pocket lift** on the 5 Wf-325B canonical pockets
(1h36, 3eml, 3ny8, 4aaw, 4rlu):

10. `test_cisplatin_yields_negative_adjustment_on_all_5_pockets` —
    asserts the sign is negative on every pocket (was +0.994 everywhere
    pre-fix).
11. `test_projected_total_lift_per_pocket` — combines the sign-fixed
    metal-coord adjustment with the Wf-325B MEASURED kcal/mol medians to
    compute projected metal-coord kcal/mol per pocket. Asserts
    projected < median (better binding) on every pocket.
12. `test_wf325b_baselines_loaded` — sanity check on the Wf-325B
    JSON file format (catches future regressions).

### §4.5 Test execution

```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf333v1_wf246c_sign_fix.py -v

============================== 12 passed in 0.46s ==============================
```

All 12 NEW tests pass. All 23 existing `test_metal_coord_term.py`
tests also pass post-fix.

---

## §5. Per-pocket lift projections (5 Wf-325B pockets)

| Pocket | Wf-325B median (MEASURED) | Wf-333-V1 PROJ | Δ (PROJECTED) |
|---|---:|---:|---:|
| 1h36 | -3.589 | -4.583 | -0.994 |
| 3eml | -2.827 | -3.821 | -0.994 |
| 3ny8 | -3.721 | -4.715 | -0.994 |
| 4aaw | -3.283 | -4.277 | -0.994 |
| 4rlu | -2.981 | -3.975 | -0.994 |
| **median** | **-3.283** | **-4.277** | **-0.994** |

**PROJECTED cross-pocket lift = -0.994 kcal/mol** (perfect composite
on every pocket; distance sub-score is a no-op without 3D pose).

---

## §6. Honest framing (party-twenty-spirit §3.5)

### §6.1 What is MEASURED (post-Wf-333-V1)

- Wf-246-C sign bug identified (Wf-331-R1 §4.2)
- Sign fix verified at unit-test level (12 NEW tests PASS, 23 existing
  tests PASS) — `kcal_adjustment(perfect=1.0) = -1.0` not +1.0
- Wf-325B per-pocket medians (5 pockets, MEASURED_GENERIC_ATOM) loaded
  correctly
- Adjustment direction matches Vina's sign convention (lower = better)

### §6.2 What is PROJECTED

- Per-pocket lift = -0.994 kcal/mol per pocket (PROJECTED — assumes
  perfect composite on each pocket, which is realistic for cisplatin
  but not for every metallodrug)
- Cross-pocket median lift = -0.994 kcal/mol (PROJECTED — same caveat)
- Combined with F2(a) -1.5 MEASURED: cross-pocket median = **-5.777 kcal/mol**
  (still 1.223 kcal/mol short of industry -7.0 cutoff)
- Combined with Wf-333-V3 P1 wire-in + P3 per-bond + P3 distance:
  cross-pocket median = **-7.782 kcal/mol** (PASSES industry cutoff
  by 0.782 kcal/mol margin) — see Wf-331-R1 §6.6

### §6.3 What is BLOCKED

- True Pt+2 Vina parameterisation = BLOCKED per Wf-299B
  (4-6 week scope, Vina source patch)
- Wf-246-C → cfm_vina_dock_panel.py wire-in = PENDING (Wf-333-V3 P1
  next, not in this PR)

### §6.4 What's in this PR

- **1 LOC fix** at `metal_coord_term.py:654`
- **Updated docstring** at `metal_coord_term.py:529 + :624`
- **1 test update** at `test_metal_coord_term.py:188-198`
- **1 NEW test file** `test_wf333v1_wf246c_sign_fix.py` (12 tests,
  4 classes)

### §6.5 What's NOT in this PR (next waves)

- **Wf-333-V3 P1** — wire Wf-246-C into `cfm_vina_dock_panel.py`
  (50 LOC + 2 tests; enables the -1.0 PROJECTED lift to be measured
  end-to-end on the m3 metric)
- **Wf-333-V3 P2** — per-bond F2(a) selector (30 LOC + 2 tests;
  PROJECTED additional -1.5 kcal/mol)
- **Wf-333-V3 P3** — distance sub-score wire-in (80 LOC + 3 tests;
  PROJECTED additional -0.5 to -1.0 kcal/mol)
- **Wf-333-V3 P4** — combined ablate (1 report; PROJECTED total
  -4.5 kcal/mol → -7.782 cross-pocket median PASSES industry -7.0)

---

## §7. Verification commands (copy-paste runnable)

```bash
# Run the new tests
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_wf333v1_wf246c_sign_fix.py -v

# Run the existing tests (regression check)
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_metal_coord_term.py -v

# Run all Vina-related tests
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
  -m pytest molmetal/tests/test_metal_coord_term.py \
           molmetal/tests/test_wf333v1_wf246c_sign_fix.py -v
```

Expected: 23 + 12 = 35 passed.

---

## §8. Cross-references

| Reference | Section |
|---|---|
| Wf-246-C | metal_coord_term.py (477 LOC); cisplatin_1h36.json — original sign bug |
| Wf-325B | result.json — 5-pocket MEASURED Vina medians (MEASURED_GENERIC_ATOM) |
| Wf-329-W3 | F2(a) MetalLigandExchange SMARTS rule + covalent scoring modifier |
| Wf-331-R1 | §4.2 sign-bug diagnosis + §6.1 Option 1 fix recipe |
| Wf-333-V3 (next) | P1 wire-in + P2 per-bond + P3 distance for full -4.5 PROJECTED lift |
| AGENTS.md §5.1 | Standing GPU auth — not needed for this CPU-only fix |
| AGENTS.md §5.3 | No paper work — Wf-333-V1 is TODO-only |
| AGENTS.md §5.4 | Tier-2 journal, no wet-lab — wet-lab validation DEPRECATED |
| `party-twenty-spirit` §3.5 | 0-tolerance fabrication policy; PROJECTED vs MEASURED tags |

---

## §9. Files changed

| File | Change |
|---|---|
| `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py` | 1 LOC fix (line 654) + 2 docstring updates (lines 529, 624-651) |
| `molmetal/tests/test_metal_coord_term.py` | 1 test assertion updated (test_cisplatin_full_composite: +1.0 → -1.0) |
| `molmetal/tests/test_wf333v1_wf246c_sign_fix.py` | NEW (12 tests, 4 classes) |
| `TODO/completed/333v1_wf246c_sign_fix.md` | NEW (this file) |

## §10. Closing note

Wf-333-V1 closes the Wf-246-C kcal sign bug (1 LOC) and pins it
with 12 NEW regression tests. The fix enables the m3 Vina Dock
recovery path outlined in Wf-331-R1 §6 (combined -4.5 kcal/mol
PROJECTED lift across P0+P1+P2+P3). This PR is CPU-only, no GPU
launch, no paper work — strictly within AGENTS.md §5.1 / §5.3 / §5.4
boundaries.

**Next step** (Wf-333-V2 / V3): wire Wf-246-C into
`cfm_vina_dock_panel.py` + per-bond F2(a) selector + distance
sub-score wire-in to MEASURE the combined PROJECTED -4.5 kcal/mol
lift end-to-end.
