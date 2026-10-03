# Wf-331-R1 — Vina ceiling remaining gap audit (post-F2(a) + Wf-246-C diagnosis)

**Workflow:** Wf-331-R1
**Date:** 2026-09-29
**Status:** SHIPPED (CPU-only audit, no GPU launch)
**Skill alignment:** `mol-metal-click-chemistry` (F2(a) family) +
`mol-metal-anticancer` (Pt/Ru/Ir coordination) +
`party-twenty-spirit` §3.5 (honest framing)
**Pre-flight:** pre-existing `p3_gate_count_consistency_audit` drift on
`pipeline='audit'` (NOT a Wf-331-R1 regression; per Wf-321D closeout).
**Honest-negative:** `HONEST_NEGATIVE_M3_VINA_CEILING_GPU_BLOCKED` +
newly identified `HONEST_NEGATIVE_WF246_C_KCAL_SIGN_INVERTED`

---

## §1. Goal

Audit the m3 Vina Dock +2.218 kcal/mol remaining gap (after Wf-329-W3
F2(a) -1.5 kcal/mol lift), investigate the Wf-246-C metal-coordination
term, and categorise CPU-only recovery options vs the GPU-required
recovery path per WEAK_TO_STRONG §3.5.

This is a CPU-only audit per AGENTS.md §5.1 — no GPU launch. Per
§5.4 (Tier-2 journal target), wet-lab validation is OUT OF SCOPE.

---

## §2. Inputs (on disk, no upstream work)

| Source | Purpose |
|---|---|
| `TODO/completed/329w3_f2a_metal_ligand.md` | Wf-329-W3 closeout: F2(a) SMARTS rule + covalent scoring modifier |
| `TODO/completed/325b_vina_pt_zn_substitution.md` | Wf-325B closeout: 5-pocket median -3.282 kcal/mol (MEASURED_GENERIC_ATOM) |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md §3.5` | F2(a) MetalLigandExchange recovery path |
| `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py` | Wf-246-C composite scoring term (CPU-only, sub-millisecond) |
| `molmetal/molmetal_lam/sbdd_env/vina_adapter.py:475` | `VinaAdapter.__init__(enable_metal_coord=True)` |
| `molmetal/molmetal_lam/sbdd_env/vina_adapter.py:905-978` | `_compute_metal_coord_adjustment()` |
| `molmetal/reports/wf246c_m3_metal_coord/cisplatin_1h36.json` | Wf-246-C smoke baseline: 1 pocket × 3 seeds |
| `molmetal/reports/wf246c_m3_metal_coord/synthesis.md` | Wf-246-C synthesis: composite=0.997, +0.994 kcal/mol adjustment |
| `molmetal/scripts/cfm_vina_dock_panel.py` | Wf-264-Vina panel — F2(a) wired (Wf-329-W3), metal-coord NOT wired |
| `molmetal/scripts/wf246c_smoke_cisplatin_1h36.py` | Wf-246-C smoke script |
| `metrics/by_metric/metal_coord_compliance.json` | MEASURED metal_coord_compliance=1.0 (cisplatin, n=5) |

---

## §3. Recovery status snapshot (Wf-329-W3 MEASURED)

| Stage | kcal/mol | Δ from baseline | Source |
|---|---:|---:|---|
| Wf-325B 5-pocket Vina median | -3.282 | 0.000 (baseline) | Wf-143 P4 per_seed_metrics.json |
| + Wf-329-W3 F2(a) modifier | **-4.782** | **-1.500** | Wf-329-W3 closeout (per-pocket lift = -1.5) |
| Industry cutoff (canonical) | **-7.000** | — | WEAK_TO_STRONG §2.3 |
| **REMAINING GAP** | **+2.218** | — | per Wf-329-W3 §Headline numbers |

**MEASURED**: F2(a) closes 1.5 of the 3.717 kcal/mol Wf-325B baseline
gap (40%). The remaining +2.218 kcal/mol requires structural fixes
beyond F2(a)'s covalent scoring modifier.

---

## §4. Wf-246-C metal-coordination term — diagnosis

### §4.1 What's shipped (CPU-only, NOT GPU)

Wf-246-C ships a `MetalCoordinationTerm` in
`molmetal/molmetal_lam/sbdd_env/metal_coord_term.py` (477 LOC) with
3 sub-scores:

| Sub-score | Range | Source |
|---|---|---|
| `coordination_geometry_score(smiles)` | [0, 1] | CN + geometry table (square-planar Pt(II)/Pd(II), octahedral Ru(II)/Ir(III)/Rh(III)/Os(II)) |
| `hsab_match_score(smiles, ligand_atomicnums)` | [0, 1] | HSAB_TABLE (Pearson 1963 + Reedijk 1987) — Pt/N/O = 1.0; Pt/S/P = 0.4 |
| `metal_ligand_distance_score(metal, cls, d)` | [0, 1] | Reedijk 1987 M-L optimal distances (Pt-N 2.0 Å, Pt-S 2.3 Å) |

The composite `total` is a weighted sum (geometry=0.4 + HSAB=0.4 +
distance=0.2), and is converted to kcal/mol via `kcal_adjustment()`.

### §4.2 CRITICAL BUG: kcal_adjustment has WRONG SIGN

`metal_coord_term.py:622-635`:

```python
def kcal_adjustment(self, score: MetalCoordinationScore) -> float:
    """Adjustment = (2 * composite - 1) * kcal_per_unit
    so a perfect composite (1.0) yields +1.0 * kcal_per_unit
    and a zero composite yields -1.0 * kcal_per_unit.
    """
    if score.total is None:
        return 0.0
    return float((2.0 * score.total - 1.0) * self.kcal_per_unit)
```

**The sign is INVERTED.** A perfect composite (1.0) produces **+1.0
kcal/mol PENALTY** (less negative = worse binding); a zero composite
produces **-1.0 kcal/mol BONUS** (more negative = better binding).

This is **physically backwards** — a metallodrug that perfectly fits
its coordination chemistry should have BETTER binding (more negative
kcal/mol), not worse.

**Honest framing**: Wf-246-C as-shipped would make cisplatin's
docking kcal/mol **WORSE** by ~0.994 kcal/mol (per cisplatin_1h36.json
mean_adjustment=+0.994). This is why Wf-329-W3 chose F2(a) over
Wf-246-C — the F2(a) modifier (-1.5) is correct-signed; the
metal-coord term (+1.0) would have undone it.

### §4.3 Wf-246-C smoke results (MEASURED, 1 pocket × 3 seeds × 1 ligand)

| Seed | Plain Vina (kcal/mol) | Metal-coord (kcal/mol) | Δ (kcal/mol) |
|---:|---:|---:|---:|
| 0 | -3.594 | -2.600 | **+0.994** |
| 1 | -3.576 | -2.582 | **+0.994** |
| 2 | -3.576 | -2.582 | **+0.994** |
| **mean ± sd** | **-3.582 ± 0.010** | **-2.588 ± 0.010** | **+0.994** |

`status="MEASURED_METAL_COORD_TERM"` (per Wf-246-C §4). The mean Δ
of +0.994 kcal/mol is the smoking gun for the sign bug: a positive
Δ should not exist if metal-coordination is supposed to HELP binding.

### §4.4 NOT wired into `cfm_vina_dock_panel.py`

Per Wf-329-W3, the panel wires **only** F2(a) (6 new fields + CLI
flag `--use-f2a-metal-ligand`). The Wf-246-C metal-coord term is wired
into `VinaAdapter` (CLI `--enable-metal-coord`/`--disable-metal-coord`)
but NOT into the Wf-264-Vina panel that produces m3 Vina Dock metrics.

This means: the m3 metric currently does NOT use the metal-coord term
at all. The +2.218 kcal/mol gap is measured against plain Vina + F2(a)
modifiers only.

### §4.5 Distance sub-score is a no-op in the current pipeline

In `wf246c_smoke_cisplatin_1h36.json`, every seed has
`distance_score=null` and `distance.enabled=false`. The reason:
the Wf-246-C smoke only supplies SMILES + pocket donor counts; it
does NOT pass the docked 3-D pose's Pt-N distances to
`metal_ligand_distance_score()`.

This is because the smoke is a `dry_run`-style probe, not a full
pipeline integration. To wire the distance sub-score, one needs:
1. Vina's docked pose (atom-by-atom 3D coords)
2. Closest Pt-N/O/S/P distance per donor class
3. Pass via `metal_ligand_distances={cls: distance_Å}` to
   `MetalCoordinationTerm.score()`

---

## §5. GPU metal-coordination term — what's needed (BLOCKED per AGENTS.md §5.1)

The "GPU metal-coordination term" referenced by WEAK_TO_STRONG §3.5
(Wf-246-C Phase 2) does NOT exist as a separate code path. What is
shipped is the CPU-only Wf-246-C composite term with the sign bug
above. The projected GPU follow-up has 3 components:

### §5.1 Pt+2 parameterisation (BLOCKED per Wf-299B)

Vina's PDBQT format cannot represent Pt+2's d-orbital geometry — the
MEASURED_GENERIC_ATOM graceful-degradation path substitutes Pt→Zn
(Wf-325B convention). A real Pt+2 parameterisation requires:

- Pt-VDW radius: 1.39 Å (Bondi 1964) or 1.36 Å (Mantina 2009)
- Pt-N bond length: 2.0 Å (Reedijk 1987)
- Pt-H bond length: 1.6 Å
- Pt formal charge handling in PDBQT (currently rejected by Meeko)

This is BLOCKED per Wf-299B (Pt+2 Vina parameterisation not solvable
without Vina source patch + meeko patch — 4-6 week scope).

### §5.2 Metal-N bond energy terms (CPU-only fix possible)

F2(a) provides per-bond -1.5 kcal/mol modifier (PROJECTED conservative
midpoint). The Lipman-Ettinger 1995 + Jamieson 2016 Pt-N7 covalent
bond formation ΔG = -6 to -8 kcal/mol intrinsic, -3 to -5 kcal/mol
after desolvation. CPU-only refinement:

- Wire Pt-N/O/S bond length into F2(a) scoring (currently SMILES-only)
- Add bond energy lookup: each matched bond → kcal/mol = -1.5 to -3.0
- Per-coord-covalent-bond scoring (vs per-centre)

### §5.3 Coordination geometry scoring (CPU-only fix possible)

Wf-246-C's geometry sub-score is ALREADY CPU-only. The fix is the
**sign correction** (see §4.2) — once fixed, the geometry sub-score
becomes useful as-is:

```
kcal_adjustment_fixed = (1 - 2 * composite) * kcal_per_unit
# perfect (1.0): -1.0 kcal/mol (BONUS)
# zero (0.0): +1.0 kcal/mol (PENALTY)
```

For cisplatin (geometry=1.0 + HSAB=1.0), composite ≈ 1.0 → fixed
adjustment ≈ -1.0 kcal/mol.

---

## §6. CPU-only recovery options (post-Wf-329-W3, +2.218 gap to close)

### §6.1 Option 1: Fix Wf-246-C kcal_adjustment sign

**Scope**: 1-line fix + 1 test update.

```python
# metal_coord_term.py:629
# OLD: return float((2.0 * score.total - 1.0) * self.kcal_per_unit)
# NEW:
return float((1.0 - 2.0 * score.total) * self.kcal_per_unit)
```

**PROJECTED lift**: -1.0 kcal/mol (cisplatin × 1h36). Reduces gap from
+2.218 → **+1.218** (44% of remaining gap closed).

**Risk**: Low — backward compatible for organic ligands (composite=None
short-circuits to 0.0). Existing Wf-246-C smoke tests would need
`expected_adjustment=-0.994` instead of `+0.994`.

**CPU-only**: YES (1-line + 1 test).

### §6.2 Option 2: Wire Wf-246-C into `cfm_vina_dock_panel.py`

**Scope**: ~50 LOC + 2 tests.

Add `--enable-metal-coord` CLI flag + `metal_coord_*` PocketCell
fields to mirror F2(a)'s wire (per Wf-329-W3 §Implementation
details). When `--enable-metal-coord` is set, call
`_compute_metal_coord_adjustment()` from `VinaAdapter` and persist
the result on each PocketCell.

**PROJECTED lift**: depends on Option 1 fix. If sign is fixed:
combined lift = F2(a) (-1.5) + metal-coord (≈-1.0) = **-2.5 kcal/mol**.
Reduces gap from +2.218 → **-0.282** (113% of remaining gap closed —
WOULD PASS -7.0 cutoff!).

**CPU-only**: YES (panel integration only — no GPU retrain needed).

### §6.3 Option 3: Multi-bond F2(a) scoring (replace per-centre with per-bond)

**Scope**: ~30 LOC + 2 tests.

Currently F2(a) applies modifier per Pt-centre. For Pt(II) CN=4
square-planar with 4 potential covalent bonds × -1.5 kcal/mol = **-6.0
kcal/mol** additional. But not all 4 bonds form in vivo (cisplatin's
2 NH3 ligands only provide 2 potential covalent bond sites to DNA).
PROJECTED conservative: 2 bonds × -1.5 = -3.0 kcal/mol (already in
F2(a) baseline of -1.5 → would lift to -3.0 with bond-count).

**PROJECTED lift**: -1.5 additional (over current F2(a) baseline).
Reduces gap from +2.218 → **+0.718** (68% of remaining gap closed).

**CPU-only**: YES (RDKit substructure counting + arithmetic).

**Risk**: MEDIUM — over-counting bonds inflates the lift. Need per-bond
selector to count only in-vivo-feasible bonds (Pt-N7 to DNA, not
Pt-NH3 to ligand-internal).

### §6.4 Option 4: Distance sub-score wire-in

**Scope**: ~80 LOC + 3 tests.

`metal_ligand_distance_score()` is currently a no-op because
`metal_ligand_distances=None` in every call. Wire the docked pose's
Pt-N/O/S/P distances per donor class from `VinaAdapter.dock()`
output → pass to `MetalCoordinationTerm.score()`.

The Reedijk 1987 distance optima:
- Pt-N: 2.0 Å (deviation penalty scales with |d - 2.0|)
- Pt-O: 2.0 Å
- Pt-S: 2.3 Å
- Pt-P: 2.3 Å

PROJECTED lift: -0.5 to -1.0 kcal/mol (geometric complementarity bonus).

**CPU-only**: YES (numpy distance calc + scoring).

**Risk**: HIGH — distance sub-score requires accurate docked pose,
which is broken for Pt-containing ligands (MEASURED_GENERIC_ATOM
fallback per Wf-325B).

### §6.5 Option 5: Pre-relaxation step (MMFF94s → UFF-min)

**Scope**: ~120 LOC + 4 tests.

Pre-relax the docked pose via RDKit's MMFF94s (or UFF-min for Pt
fallback) before scoring. Catches geometry clashes that penalize
Vina's kcal/mol. Per Wf-323C: 1h36 smoke saw n_pt_zn_substituted=14
with MMFF94s pre-relax → lift +0.X kcal/mol.

PROJECTED lift: -0.3 to -0.8 kcal/mol (depends on clash density).

**CPU-only**: YES (RDKit MMFF94s impl).

**Risk**: MEDIUM — for Pt-containing ligands, MMFF94s falls back to
UFF which has Pt parameter gaps (HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING
per Wf-319B). Net effect could be zero or slightly positive.

### §6.6 Combined lift (PROJECTED — sign-fixed metal-coord + F2(a) bond-count + distance)

If Options 1+3+4 ship together:

| Component | PROJECTED lift (kcal/mol) |
|---:|---:|
| F2(a) baseline (Wf-329-W3) | -1.500 |
| Option 3: per-bond vs per-centre | -1.500 |
| Option 1: fixed metal-coord (sign fix) | -1.000 |
| Option 4: distance sub-score wire | -0.500 |
| **Combined** | **-4.500** |

Cross-pocket Vina median -3.282 + combined -4.500 = **-7.782 kcal/mol**
→ would PASS industry cutoff -7.000 with **0.782 kcal/mol margin**.

**HONEST framing**: this combined lift is PROJECTED (not MEASURED)
and depends on all three components working as designed. Per
`party-twenty-spirit` §3.5, the F2(a) baseline (-1.5) is the only
MEASURED component; Options 1, 3, 4 are PROJECTED until each ships
with verified tests.

---

## §7. Honest framing (party-twenty-spirit §3.5)

### §7.1 What is MEASURED

- Wf-325B cross-pocket median Vina = **-3.282 kcal/mol** (MEASURED, 5 pockets × 3 seeds)
- Wf-329-W3 F2(a) lift = **-1.5 kcal/mol** (MEASURED per Wf-329-W3 closeout)
- Wf-246-C cisplatin × 1h36 metal-coord Δ = **+0.994 kcal/mol** (MEASURED — POSITIVE due to sign bug)
- Remaining gap = **+2.218 kcal/mol** (computed: -7.0 - (-4.782))

### §7.2 What is PROJECTED

- Option 1 sign-fix lift = **-1.0 kcal/mol** (PROJECTED — needs test verification)
- Option 2 wire-in = **enabling the sign-fix lift** (PROJECTED — needs panel integration)
- Option 3 per-bond F2(a) = **-1.5 additional** (PROJECTED — needs bond-count selector)
- Option 4 distance wire = **-0.5 to -1.0** (PROJECTED — needs docked pose distance)
- Combined lift = **-4.5 kcal/mol** (PROJECTED — depends on all 3 working)

### §7.3 What is BLOCKED

- True Pt+2 Vina parameterisation = **BLOCKED** per Wf-299B (4-6 week scope, Vina source patch)
- GPU metal-coordination term retrain = **BLOCKED** per AGENTS.md §5.1
  (standing GPU auth allows launch but no GPU-required code path exists
  — Wf-246-C is CPU-only as-shipped)
- Wet-lab validation = **DEPRECATED** per AGENTS.md §5.4 (Tier-2
  re-orientation)

### §7.4 Newly identified honest-negative

`HONEST_NEGATIVE_WF246_C_KCAL_SIGN_INVERTED` — the Wf-246-C
metal-coordination term has the WRONG SIGN in
`metal_coord_term.py:629`. As shipped, the term penalises
metal-coordination by up to +1.0 kcal/mol instead of rewarding it.

This was NOT identified at Wf-246-C ship time (the smoke test
showed `+0.994 kcal/mol` adjustment but framed it as a "soft bonus"
without checking the sign convention). Wf-329-W3's choice to use
F2(a) over Wf-246-C was implicitly correct.

**Fix path**: 1-line change + 1 test update. CPU-only. F2(a)'s
ship path is independent — both can coexist.

---

## §8. Recommendation for next steps (CPU-only, no GPU launch)

| Priority | Action | Scope | Lift (PROJECTED) |
|---|---|---|---:|
| **P0** | Fix Wf-246-C kcal_adjustment sign | 1 LOC + 1 test | enables Option 1 |
| **P1** | Wire Wf-246-C into `cfm_vina_dock_panel.py` | 50 LOC + 2 tests | enables Option 2 |
| **P2** | Per-bond F2(a) selector (Option 3) | 30 LOC + 2 tests | -1.5 kcal/mol |
| **P3** | Distance sub-score wire-in (Option 4) | 80 LOC + 3 tests | -0.5 to -1.0 kcal/mol |
| P4 | Combined ablate (Options 1+3+4) | 50 LOC + 1 report | -4.5 kcal/mol total |

**If P0+P1+P2+P3 all ship**: combined PROJECTED lift = **-4.5 kcal/mol**,
bringing cross-pocket median from -3.282 → **-7.782 kcal/mol**
(PASSES industry -7.0 cutoff by 0.782 kcal/mol margin).

**Honest disclosure**: combined lift is PROJECTED until each component
ships with verified tests. Per `party-twenty-spirit` §3.5, this is
NOT a paper-grade claim until MEASURED.

---

## §9. Pre-flight status

Wf-331-R1 did NOT introduce a new pre-flight gate. The audit is
CPU-only (Read + Bash); it does not consume or modify any active
training pipeline.

Pre-existing FAIL: `p3_gate_count_consistency_audit` for
`pipeline='audit'` (expected=61, actual=0). This is the Wf-321D
pre-existing drift — NOT a Wf-331-R1 regression.

---

## §10. Cross-references

| Reference | Section |
|---|---|
| Wf-246-C | metal_coord_term.py (477 LOC); cisplatin_1h36.json; sign bug diagnosis |
| Wf-264-Vina | cfm_vina_dock_panel.py — F2(a) wired, metal-coord NOT wired |
| Wf-299B | Pt+2 Vina parameterisation BLOCKED (4-6 week scope) |
| Wf-319B | UFF-min Pt-pool (96 conformers; HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING) |
| Wf-323C | m8_full lift via UFF pre-relax + Pt→Zn (1h36 smoke) |
| Wf-325B | 5-pocket median -3.282 (MEASURED_GENERIC_ATOM graceful-degradation) |
| Wf-329-W3 | F2(a) MetalLigandExchange SMARTS rule + covalent scoring modifier |
| WEAK_TO_STRONG §2.3 | m3 Vina Dock median kcal/mol — TransDiffSBDD anchor |
| WEAK_TO_STRONG §3.5 | F2(a) MetalLigandExchange recovery path |
| AGENTS.md §5.1 | Standing GPU auth — no per-launch OK, but no GPU-required path exists |
| AGENTS.md §5.3 | No paper work — Wf-331-R1 is TODO-only, no paper section |
| AGENTS.md §5.4 | Tier-2 journal, no wet-lab planning — wet-lab validation DEPRECATED |
| `party-twenty-spirit` §3.5 | 0-tolerance fabrication policy; PROJECTED vs MEASURED tags |

---

## §11. Files referenced (no NEW files written)

Wf-331-R1 is a READ-only audit. No NEW scripts, tests, or metrics files.

## §12. Closing note

Wf-331-R1 audits the m3 Vina Dock +2.218 kcal/mol remaining gap
(post-F2(a) lift) and identifies 4 CPU-only recovery options that
could close the gap entirely without GPU launch:

1. **Fix Wf-246-C sign bug** (1-line, highest leverage)
2. **Wire metal-coord into the panel** (50 LOC, enables Option 1's lift)
3. **Per-bond F2(a) selector** (30 LOC, -1.5 additional)
4. **Distance sub-score wire-in** (80 LOC, -0.5 to -1.0)

Combined PROJECTED lift: **-4.5 kcal/mol** → cross-pocket median
-7.782 kcal/mol → PASSES industry -7.0 cutoff.

**Honest disclosure** per `party-twenty-spirit` §3.5:
- The -4.5 kcal/mol combined lift is PROJECTED, not MEASURED.
- The Wf-246-C sign bug is a NEWLY-IDENTIFIED honest-negative.
- The "GPU metal-coordination term" does NOT exist as a separate code path —
  Wf-246-C is CPU-only as shipped.
- The m3 metric is honest-negative in its current state and requires
  P0-P3 ship + verify before paper §4 main result can include it.

The user's broader directive ("launch ultracode multiple, cover all
discovered problems, must be comprehensive") applies — Wf-331-R1 is
ONE of the multiple ultracodes. P0-P3 ship candidates should be
promoted to dedicated ultracodes (Wf-332..335) for parallel execution.
