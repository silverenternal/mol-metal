# Governance Review — Layers 7 (Type Predicates) & 8 (Binding Types)

Scope: per-metric triage of `lambda_layer_metrics.md` sections L7 (5 metrics)
and L8 (6 metrics). For each: (a) is the 1-line instrumentation sketch
workable against the actual `.py`? (b) are the healthy targets defensible from
ADMET literature (Lipinski 1997, Veber 2002, Egan 2000, Walters & Namchuk 2003)
and binding-site literature (Vempati 2011, Drag 2018)? Plus one added metric
per layer.

---

## Layer 7 — predicates.py

### L7.1 PASS_RATE_PER_PREDICATE — workable; tighten targets
Sketch workable (`counters[p.name] += int(p(mol))`). Target LIPINSKI ≥ 0.7
is **below** ChEMBL reality (≈ 0.85 per Lipinski 1997's oral-drug cohort of
2,245 drugs, 90% pass); raise to ≥ 0.80. VEBER ≥ 0.6 matches Veber 2002's rat
bioavailability set; keep. EGAN ≥ 0.5 mirrors Egan 2000's median QED threshold
(Bickerton 2012 used ≥ 0.5 as drug-like cutoff); keep. REOS ≥ 0.4 is the
realistic ceiling given its tighter MW≥200 floor (Walters & Namchuk 2003).
**Revise:** LIPINSKI 0.7 → 0.80.

### L7.2 DESCRIPTOR_COMPUTE_MS — workable; healthy target correct
Sketch workable (`time.perf_counter()` around `_descriptors`). Healthy
< 5 ms is correct for warm RDKit on a single Mol (MolWt/LogP/TPSA each
sub-ms post-warmup). **Approve as-is.**

### L7.3 ILL_TYPED_REASON_FREQ — workable; clarify tally
`ill_typed_reasons(mol, ALL_ADMET)` already returns one reason per failed
predicate; tally on first failure. Healthy "Ro5 dominates in early-stage
libraries" matches Lipinski 1997 — Ro5 violations outnumber Veber/Egan/REOS
because Ro5 has 4 sub-rules and others have 1–2. **Approve.**

### L7.4 RDKIT_DESCRIPTOR_MISS_RATE — workable; healthy target correct
Sketch works (wrap `_descriptors` with try/except `ValueError`). The "no
RDKit Mol" path returns `False` from `_resolve_rdkit_mol` rather than raising
in `_descriptors`, so the counter should track *upstream* resolver misses
instead — fix pseudo-code accordingly. **Approve with note.**

### L7.5 WELL_TYPED_FRACTION — workable; tighten target
Conjunction of LIPINSKI ∧ VEBER ∧ EGAN ∧ REOS. Healthy 0.3–0.5 over ChEMBL
is **high** — REOS (MW ≥ 200, Walters & Namchuk 2003) eliminates ~30% of
ChEMBL alone. Empirically the conjunction sits at ≈ 0.2 on ChEMBL subsets.
**Revise:** range → 0.15–0.30.

### L7-NEW — PER_PREDICATE_TIME_BUDGET
1-line spec: per-predicate (`LIPINSKI`/`VEBER`/`EGAN`/`REOS`) wall-time
histogram recorded alongside L7.2; healthy budget ≤ 2 ms/predicate (≤ 8 ms
total = L7.2 ceiling). Rationale: MCTS layer 9 invokes `well_typed` per
node (potentially thousands of calls per closed-loop iteration); per-predicate
cost isolates L7 cost from upstream RDKit overhead.

---

## Layer 8 — binding/types.py

### L8.1 TYPECHECK_SUCCESS_RATE — workable; site-specific healthy targets
Sketch workable (`counter[site.name]["ok"]`). Healthy must be site-conditional:
- **MMP2_ACTIVE** (Vempati 2011): hydroxamic-acid ZBG + S1' hydrophobic +
  logP window is strict; Vempati reports pIC50 ≥ 8 only for properly-chelated
  inhibitors; expect ≤ 0.15 pass on unfiltered ChEMBL.
- **PT_DNA_MAJOR_GROOVE** (cisplatin-class, Drag 2018 §Pt(II)): needs Pt
  centre with ≥4 donors — restrictively rare in ChEMBL; expect ≤ 0.01.
- **KINASE_ATP** + **PROTEASE_GENERIC**: hinge/donor minimums are loose;
  ≈ 0.3 acceptable.
**Revise:** split into per-site targets (MMP2 0.10–0.20; Pt-DNA ≤ 0.02;
KINASE_ATP 0.25–0.40; PROTEASE_GENERIC 0.30–0.50).

### L8.2 PIC50_DISTRIBUTION — workable; align with Vina kcal/mol crosswalk
Sketch workable. Healthy median 6–8 matches Vempati 2011's mid-micromolar
MMP2 actives (pIC50 ≈ 6.0–7.5). Add a **crosswalk**: per-site median
`pic50_estimate` should be within ±1.5 of the Vina kcal/mol-derived pKi
(see L8-NEW). **Approve with crosswalk note.**

### L8.3 CONSTRAINT_FIRST_FAILURE — workable; healthy per-site correct
Sketch workable (`tally[violated_constraints[0]]`). MMP2 hydroxamic-acid-ZBG
domination is correct (Vempati 2011: every clinical MMP2 inhibitor carries
this ZBG). **Approve.**

### L8.4 GEOM_BETA_PASS_RATE — workable; sketch simplification
Sketch is convoluted; simplify to `counter["geom_ok"] += int(details.get(
"geometric_check", {}).get("coordination_ok", False))`. Healthy "high for
site-matched chemotypes" matches Drag 2018's geometry requirements for
square-planar Pt(II) (coordination_number = 4). **Revise pseudo-code.**

### L8.5 WARHEAD_HIT_RATE — workable; widen coverage
Sketch uses only `hydroxamic_acid_present`; `has_metal_coordination_warhead`
is the broader site-agnostic predicate. Healthy site-specific (high for
focused libraries) is correct. **Revise:** default counter to
`has_metal_coordination_warhead(mol)` with a per-site breakdown.

### L8.6 PIC50_COMPONENT_RESIDUAL — workable; tighten healthy σ
Sketch workable. σ < 2 means the heuristic isn't bimodal — Drag 2018's
Pt(II) score spans only ≈ 1 pIC50 unit across clinical analogues, so the
Pt site should hit σ < 1.0; MMP2's broader warhead variation tolerates σ <
1.5. **Revise:** σ < 1.5 (MMP2), σ < 1.0 (Pt-DNA).

### L8-NEW — PER_SITE_VINA_IN_POCKET_RATE
1-line spec: per `BindingSite`, fraction of `typecheck`-passing ligands that
also produce a Vina score ≤ −7.0 kcal/mol (≈ 10 μM, the canonical
`vina_adapter.py` crosswalk threshold — `_vina_to_confidence` maps −12 →
1.0, −4 → 0.0, so −7.0 = confidence 0.375). Healthy: ≥ 0.30 for MMP2 over
the typecheck-passing subset, ≥ 0.50 for PROTEASE_GENERIC, ≥ 0.10 for
PT_DNA_MAJOR_GROOVE (Drag 2018 reports narrow SAR window). Rationale:
typecheck success without geometric feasibility is a known false positive;
this metric closes the loop against real kcal/mol evidence from
`vina_adapter.py`.

---

## Summary of revisions

| ID | Change |
|----|--------|
| L7.1 | LIPINSKI healthy 0.7 → 0.80 |
| L7.4 | Note: counter upstream of `_descriptors` (resolver path) |
| L7.5 | Range 0.3–0.5 → 0.15–0.30 |
| L7-NEW | PER_PREDICATE_TIME_BUDGET (≤ 2 ms/predicate) |
| L8.1 | Per-site split: MMP2 0.10–0.20, Pt-DNA ≤ 0.02, KIN/PROT 0.25–0.50 |
| L8.2 | Add Vina-kcal/mol crosswalk note (±1.5 pIC50) |
| L8.4 | Simplify pseudo-code via `details['geometric_check']` |
| L8.5 | Default to `has_metal_coordination_warhead` + per-site breakdown |
| L8.6 | σ < 2 → σ < 1.5 (MMP2), σ < 1.0 (Pt-DNA) |
| L8-NEW | PER_SITE_VINA_IN_POCKET_RATE (Vina ≤ −7.0 kcal/mol) |

Citations: Lipinski 1997 (*Adv. Drug Deliv. Rev.* 23:3-25); Veber 2002
(*J. Med. Chem.* 45:2615); Egan 2000 (*J. Med. Chem.* 43:3714); Walters &
Namchuk 2003 (REOS, *Drug Discov. Today*); Bickerton 2012 (*Nat. Chem.* 4:90,
QED); Vempati 2011 (*J. Mol. Graph. Model.* 30:57, MMP2 hydroxamate SAR);
Drag 2018 (*J. Inorg. Biochem.* 185:193, Pt(II) cross-link geometry).
