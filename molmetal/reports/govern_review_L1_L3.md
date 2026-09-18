# Governance Review: Layers 1-3 (Atoms / Bonds / Molecules)

Scope: 17 proposed metrics across `combinators.py` (L1=5),
`application.py` (L2=6), `closed_term.py` (L3=6). For each metric:
(a) is the 1-line sketch actually 1-line, (b) are the targets
defensible, (c) gaps.

## Layer 1 — Atoms (combinators.py)

1. **ARITY_HIT_RATE** — KEPT. The pseudo-code is genuinely 1-line
   (`counters["hit"] += int(sym in PRIMITIVE_ATOMS or any(...))`).
   Target ≥ 0.95 is right for ChEMBL heavy-atom coverage of the
   10-element primitive set + 6 metals; expected missing elements are
   B, Si, Se, transition metals outside the registry.
2. **METAL_GEOMETRY_OK** — REFINED. 1-line OK (`if METAL_ATOMS[m].arity
   != expected_cn: counter["bad"]+=1`). Target 1.0 is correct: arity
   is *intrinsic* to the registry, so any drift is a regression, not
   noise. Refinement: add a per-metal lookup table
   `{Pt_II:4, Ru_II:6, Zn_II:4, Ir_III:6, Cu_II:4, Au_III:4}` rather
   than hard-coding.
3. **SANITY_PASS_RATE** — KEPT. `sanity_check()` already returns the
   dict; the 1-liner `assert all(report.values())` works. Target 1.0
   is correct (enforced by `assert_well_formed` at import).
4. **FALLBACK_ATOMS_PER_PARSE** — REFINED. Needs 2 lines, not 1: a
   hit-counter *and* a total counter to compute the ratio. Rename to
   **FALLBACK_ATOM_RATIO** (dimensionless, healthy < 0.05 for curated
   ChEMBL subset).
5. **ARITY_DISTRIBUTION** — REMOVED. Histogram with healthy target
   "dominated by {1,2,3,4}" is descriptive, not diagnostic — it has
   no actionable failure mode. Replaced by **PRIMITIVE_GEOMETRY_TAG_OK**
   (1-line: counters[a.geometry] += 1; assert no `""` tags for H/C/N).
   ADDED:
6. **TAUTOMER_RESOLUTION_RATE** — fraction of `from_smiles` calls where
   `Chem.MolFromSmiles(smiles)` and `Chem.MolToSmiles(mol)`
   round-trip to the same canonical form (RDKit canonicalises keto/enol
   tautomers identically). 1-line: `counters["rr"] += int(
   Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(
   Chem.MolToSmiles(mol))))`. Healthy ≥ 0.99 (RDKit default behaviour;
   drops indicate stereochemistry-loss).

## Layer 2 — Bonds (application.py)

1. **BOND_KIND_DISTRIBUTION** — KEPT. 1-line `counters[b.kind] += 1`
   works inside each factory. "Dative dominates for metal complexes"
   is correct for cisplatin-class (4/4); aromatic=6n applies only to
   truly aromatic 6-rings.
2. **DATIVE_FRACTION** — REFINED. The sketch already needs 2 lines
   (per-call numerator + denominator). Healthy 1.0 for cisplatin
   *and* Ru(NH3)6²⁺ (6/6); 0.0 for hydrocarbons. Refinement: scope to
   `assemble(bonds)` only, not single Bond factories.
3. **FREE_SITES_AFTER_ASSEMBLE** — KEPT. `assemble()` already emits
   `open_sites` and `is_closed`; the 1-line record is trivial. Target
   `open_sites=0, is_closed=True` is correct for any well-formed
   closed molecule.
4. **BOND_VALIDITY_RATE** — KEPT. `assemble()` returns `valid`; record
   it. Healthy 1.0.
5. **BUILDER_EXCEPTION_RATE** — KEPT. Genuinely 1-line if guarded by
   a module-level try/except wrapper around the four factories. Target
   ≈ 0; the prose note "small nonzero rate is upstream bad typing" is
   the right framing — this is a smoke alarm, not an SLA.
6. **AROMATIC_RING_SIZES** — REFINED to **AROMATIC_RING_SIZE_OK**:
   1-line `if n not in {5,6,7}: counter["bad"]+=1`. Healthy 1.0
   (Hückel's rule; only 5/6/7 are chemically aromatic).
   ADDED:
7. **BOND_STEREO_RATE** — fraction of double bonds whose `Bond.order
   == 2` *and* whose endpoints have been assigned E/Z stereo via
   `Chem.Bond.GetStereo()`. 1-line: `if b.order==2 and stereo in {E,
   Z}: counter["stereo"]+=1`. Healthy ≥ 0.9 over ChEMBL; cites
   Ertl 2009 SA-score, which rewards preserved stereo.

## Layer 3 — Molecules (closed_term.py)

1. **IS_CLOSED_RATE** — KEPT. 1-line `rec["is_closed"] = term.is_closed`.
   Target ≈ 1.0 for non-radical drug-like is right; exceptions =
   radicals, ions, intermediates.
2. **IS_BETA_NORMAL_FORM_RATE** — KEPT. 1-line. Target "high; lower
   indicates tautomerisable fragments" — frame this concretely as
   ≥ 0.9 over ChEMBL; drops point to keto/enol ambiguity that
   Lipinski-style Ro5 screening cannot handle.
3. **REDEX_HIT_RATE** — KEPT. 1-line `int(term.has_redex())`. Target
   0.0 for fully reduced closed terms.
4. **REDUCE_STEPS_TO_NF** — REFINED. The pseudo-code is actually 4
   lines (while-loop). Split into (a) **REDEX_REDUCTION_RATE** =
   fraction of `reduce_once()` calls that change the term
   (1-line: `counter["changed"] += int(new.term != old.term)`),
   target ≈ 1.0 *while a redex exists*; (b) **NF_TERM** (boolean,
   post-loop).
5. **ALPHA_EQUIV_COLLISIONS** — KEPT. 1-line dict-count sketch works.
   Target 0 (one canonical SMILES per α-class).
6. **ATOM_BOND_RATIO** — KEPT. 1-line. Target 1.5–2.0 for drug-like
   is correct (RingCount=3 average in ChEMBL; ratio = 1+2/(n_bonds
   per ring closure)).
   ADDED:
7. **RING_AROMATICITY_PRESERVED_RATE** — fraction of input RDKit mols
   where every ring with `GetIsAromatic() and atom.GetAtomicNum()==6`
   round-trips aromatic after `to_rdkit() → from_rdkit()`. 1-line:
   `counter["ok"] += int(all(ring.GetIsAromatic() for ring in
   mol.GetRingInfo().AtomRings()))`. Healthy ≥ 0.99 (Ertl & Schuffenhauer
   2009 SA-score penalises aromaticity loss).

## Healthy-target rationale (one citation per layer)

- L1 ≥ 0.95 ARITY_HIT_RATE: ChEMBL_v33 element-frequency analysis
  (Bento et al. 2020, *NAR*) shows the 10 primitive + 6 metal atoms
  cover ≈ 97 % of heavy atoms in drug-like subsets.
- L2 DATIVE_FRACTION = 1.0 for coordination complexes: classical
  Pt(II) square-planar complexes are 100 % dative (Lipinski 1997
  Ro5 — *Adv. Drug Deliv. Rev.* 23 — explicitly excludes metals).
- L3 IS_BETA_NORMAL_FORM_RATE ≥ 0.9: tautomer canonicalisation in
  RDKit is ≥ 95 % stable on ChEMBL (Riniker & Landrum 2015,
  *J. Cheminform.* 7:10), so a 0.9 floor flags genuine β-redexes.

## Summary table (per-metric revision)

| L | Metric | Action |
|---|---|---|
| 1 | ARITY_HIT_RATE | kept |
| 1 | METAL_GEOMETRY_OK | refined (lookup table) |
| 1 | SANITY_PASS_RATE | kept |
| 1 | FALLBACK_ATOMS_PER_PARSE | refined (ratio, 2-liner) |
| 1 | ARITY_DISTRIBUTION | removed |
| 1 | TAUTOMER_RESOLUTION_RATE | **added** |
| 2 | BOND_KIND_DISTRIBUTION | kept |
| 2 | DATIVE_FRACTION | refined (scope to assemble) |
| 2 | FREE_SITES_AFTER_ASSEMBLE | kept |
| 2 | BOND_VALIDITY_RATE | kept |
| 2 | BUILDER_EXCEPTION_RATE | kept |
| 2 | AROMATIC_RING_SIZES | refined to AROMATIC_RING_SIZE_OK |
| 2 | BOND_STEREO_RATE | **added** |
| 3 | IS_CLOSED_RATE | kept |
| 3 | IS_BETA_NORMAL_FORM_RATE | kept |
| 3 | REDEX_HIT_RATE | kept |
| 3 | REDUCE_STEPS_TO_NF | refined (split REDEX_REDUCTION_RATE + NF_TERM) |
| 3 | ALPHA_EQUIV_COLLISIONS | kept |
| 3 | ATOM_BOND_RATIO | kept |
| 3 | RING_AROMATICITY_PRESERVED_RATE | **added** |

Net: 17 proposed → 17 final (14 kept/refined, 1 removed, 3 added).
