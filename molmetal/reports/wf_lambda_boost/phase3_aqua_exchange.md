# WF-Lambda-Boost Phase 3 — AquaExchange SMARTS extension + pKa context

**Date:** 2026-09-16
**Owner:** Lambda boost agent
**Phase:** 3 of 4
**Goal:** Extend AquaExchange from 1 SMARTS pattern to **2 patterns**
(first + second aquation of cisplatin per Reedijk 1987 kinetics) +
ship an `AquaContext` dataclass that propagates pKa1 / ionic strength /
temperature to the search layer for logging.  Lit anchors:
Reedijk 1987 (Pt(II) aquation kinetics).

---

## 1. Honest framing (read first)

* **What was attempted:** ship 2 SMARTS patterns (first + second
  aquation) in the NEW `pt_metal_ligand_exchange.py` module + an
  `AquaContext` dataclass carrying the aquation kinetic context +
  wire it into the existing `AquaExchange` rule class via
  `available_smarts()` + `aqua_context()` accessor methods.
* **What was measured:** 2/2 SMARTS patterns present + pKa1 = 6.5
  propagates for first aquation + pKa1 > 6.5 for second + unknown
  pattern names fall back to first-aquation context + AquaExchange
  rule exposes `aqua_context()` accessor.
* **What is NOT measured:** `AquaContext` is documentation-only —
  the search layer reads it for logging, NOT for branching.  The
  aquation reaction SMARTS are unconditional: when the rule fires,
  the products form regardless of pKa / Cl- concentration.
* **What was NOT changed:** the rule's `_reduce` method is
  unchanged.

---

## 2. The 2 SMARTS patterns

| pattern_name | description | pKa1 | SMARTS |
| --- | --- | --- | --- |
| `Pt_Cl_H2O_first` | first aquation — Pt-Cl + H2O → Pt-OH + HCl (Reedijk 1987, pKa1 ~ 6.5, t_1/2 ~ 2 h at 37 °C in 4 mM Cl-) | 6.5 | `[Pt:1]([Cl:2])[*:3].[OH2:4]>>[Pt:1]([OH:4])[*:3].[Cl:2]` |
| `Pt_OHCl_H2O_second` | second aquation — Pt-OH(Cl) + H2O → Pt-(OH)2 + Cl (Reedijk 1987, the diaqua complex that binds DNA-N7-guanine) | 7.2 | `[Pt:1]([OH:2])([Cl:3])[*:4].[OH2:5]>>[Pt:1]([OH:2])([OH:5])[*:4].[Cl:3]` |

The first aquation is the canonical cisplatin activation step:
water displaces a chloride on the metal centre to give a cationic
aqua complex (pKa1 ~ 6.5).

The second aquation is the species that binds DNA-N7-guanine
(the actual cytotoxic event) — once the first Cl has left, the
second Cl is even more labile (trans effect from the H2O).

---

## 3. AquaContext dataclass

```python
@dataclass(frozen=True)
class AquaContext:
    pka1: float = 6.5              # first aquation pKa
    ionic_strength_M: float = 0.10  # NaCl concentration approximation
    temperature_K: float = 310.0    # physiological default (37 °C)
    notes: str = "first aquation (Reedijk 1987 cisplatin canonical)"
```

**Honest framing:** this is a *documentation dataclass*.  The search
layer reads it for logging only — it does NOT branch on these values.
The aquation kinetics (pKa, rate constants) are the
`molmetal_lam.reactions.rate_predictor`'s job, not this module's.

---

## 4. Files changed

| File | Edit | Lines |
|------|------|-------|
| `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` | NEW: 2 AquaExchange SMARTS + `AquaContext` dataclass + accessors | ~80 (this phase) |
| `molmetal/molmetal_lam/reactions/beta_reductions.py` | Import + add `available_smarts()` + `aqua_context()` to `AquaExchange` class | ~25 |
| `molmetal/tests/test_wf_lambda_boost.py` | NEW: 6 tests for `TestAquaExchangeSMARTS` | ~85 |

---

## 5. Test coverage (6 tests)

| Test | Asserts |
|------|---------|
| `test_two_aqua_exchange_patterns` | `len(patterns) == 2` |
| `test_first_aquation_pka_propagates` | `get_aqua_context("Pt_Cl_H2O_first").pka1 == 6.5` |
| `test_second_aquation_diaqua_context` | pKa > 6.5 + notes mention "diaqua" or "DNA" |
| `test_unknown_pattern_falls_back_to_first` | Unknown name → first-aquation context (defensive) |
| `test_aqua_exchange_patterns_have_atom_maps` | Each SMARTS has `:1` + `:2` + `>>` + `[OH2]` |
| `test_aqua_context_via_rule_method` | `AquaExchange.aqua_context("Pt_Cl_H2O_first").pka1 == 6.5` |

---

## 6. Honest limitations + follow-ups

* **AquaContext is documentation-only.**  When the search layer
  wants true aquation kinetics (pKa-dependent product distribution),
  it must consume the `rate_predictor` module (Phase 6 TODO-29
  follow-up).  This module ships the *context* (pKa, temperature)
  but not the rate law.
* **Pattern 2 (`Pt_OHCl_H2O_second`) is structurally more complex**
  than pattern 1 — the SMARTS matches a Pt atom with TWO leaving
  groups (OH + Cl) and one donor (OH2).  RDKit sanitisation
  failures are tolerated (the `_L4_COUNTERS` instrumentation still
  fires), but the canonical product (the diaqua complex) is rarely
  observed in production runs.  Phase 6 follow-up: validate pattern 2
  on a hand-built [Pt(NH3)2(OH)(Cl)] educt.
* **pKa1 = 6.5 is for cisplatin canonical.**  Transplatin has
  pKa1 ~ 4.5 (faster aquation).  Carboplatin pKa1 ~ 5.5 (the
  cyclobutane-dicarboxylate chelate leaves instead of Cl).  Future
  work: add per-scaffold pKa1 overrides.
