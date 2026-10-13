# BLK-C3: `--decode-chemistry-validity` flag on cfm_property_panel (2026-10-09)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` BLK-C3; CPU-only code change, no GPU.
**AGENTS.md compliance**: §5.4 (no wet-lab) + §5.5 (filesystem-truth) +
§5.6 (atomic_write_file) + §5.7 (smoke gate for trainer flag wire-ins;
not applicable here — this is a property-panel flag, not a trainer flag)
+ §5.8 (direct Edit/Write preferred over wf) + §5.9 (single-venv
.venv/bin/python).

## Problem (per `todo.json` BLK-C3)

m14 CFM `decode_ratio` of 0.998 (RDKit parse) is misleading: it counts
SMILES that RDKit can read as a string, NOT SMILES that are chemically
valid. A SMILES can pass `Chem.MolFromSmiles` and still have
hypervalent atoms, anti-aromatic 4n+2-on-4n rings, or open-shell
radicals. For CFM-generated transition-metal complexes (the Wf-124 P2
+ Wf-299 + Wf-fix3lines cohort), the model routinely emits Pt species
with non-standard valences — `Chem.MolFromSmiles` happily returns a
mol for these, so the existing `parsed_ratio=0.998` looks healthy
while the actual chemistry is broken.

m4 (click-reaction success) measures a different thing (per the
existing root_cause) and runs at 0.0022, but m14 was implicitly being
used as a chemistry gate. Adding `--decode-chemistry-validity` makes
the distinction explicit: parsed_ratio is string-level, chemistry-
validity is bond/atom-level.

## Fix design

Add 5 sub-checks beyond `Chem.MolFromSmiles`:

| Sub-check | Catches |
|---|---|
| `parse_ok` | caller already passed a non-None mol (caller invariant) |
| `sanitize_ok` | `Chem.SanitizeMol(mol)` does not raise (anti-aromatic, kekulization, valence implicit-h) |
| `valence_ok` | every atom's `GetTotalValence()` is in `GetAllowedValences()` (hypervalent C, 5-valent N, [Pt] with non-standard valence) |
| `ring_ok` | every ring is size ≥ 3 (catches 2-membered rings; sanitize usually catches these first, kept as a separate gate for per-check breakdown) |
| `no_radical_ok` | every atom has `GetNumRadicalElectrons() == 0` |

`all_pass` is the AND of all 5. `chemistry_validity_pass_rate = all_pass / n_input`.

Any exception inside a sub-check is treated as a failure (False)
rather than raising, so the panel always returns a verdict even on
malformed inputs (this is the same fail-soft pattern as
`compute_one` for the SA/QED path).

The flag is **off by default** — existing runs that do not pass
`--decode-chemistry-validity` get the original `parsed_ratio`-only
output (bit-for-bit unchanged shape for the legacy fields).

## §5.5 filesystem-truth evidence

```
7388fd4e81d492d442026f8a37e2799f  molmetal/scripts/cfm_property_panel.py
483 lines (was 329 → +154 lines for the new helper + arg + plumbing)
```

New symbol locations (verified by `grep -n`):
- L192-241: `def _validate_molecule(mol) -> dict[str, bool]`
- L263-296: `def compute_one(smiles, sascorer, *, include_validity: bool = False)`
- L347-403: `def build_panel(smiles_list, *, include_validity: bool = False)`
- L456: `panel = build_panel(smiles_list, include_validity=args.decode_chemistry_validity)`
- L466: extra `print(...)` line for the new aggregate

`--help` shows the new flag:
```
--decode-chemistry-validity
                    BLK-C3 NEW: emit per-molecule chemistry-validity flags
                    ...
```

## §5.7 smoke gate (not applicable — flag is on a property panel, not a trainer)

Per AGENTS.md §5.7, the 50-step CPU smoke is required for *trainer*
flag wire-ins. `cfm_property_panel.py` is a post-hoc analyzer, not a
trainer, so the loss-monotonicity check does not apply. The wire-in
was smoke-tested with the existing 5-ep CPU smoke CSV
(`molmetal/reports/cfm_smoke_wire_5ep/generated_smiles.csv`):

| Run | parsed_ratio | chemistry_validity_pass_rate | valence_ok | sanitize_ok | ring_ok | no_radical_ok |
|---|---:|---:|---:|---:|---:|---:|
| legacy (no flag) | 1.000 | (not emitted) | — | — | — | — |
| + `--decode-chemistry-validity` | 1.000 | 0.000 | 0/69 | 69/69 | 69/69 | 69/69 |

The honesty-revealing output: **69/69 SMILES parse AND sanitize, but
0/69 pass valence check** (the [Pt] coordination in every generated
molecule is not in `Pt.GetAllowedValences()`). This is the
honest-negative BLK-C3 was designed to surface — the legacy m14
0.998 hides this fact.

> **Audit update (2026-10-09)**: the original closeout quoted 15/15
> from a partial subset; the live audit re-run of the panel on the
> full `cfm_smoke_wire_5ep/generated_smiles.csv` shows 69/69 parse +
> 0/69 valence. The headline finding (m14 0.998 is misleading) is
> unchanged; only the denominator differs.

## Honest-negative framing (per §5.2)

- `chemistry_validity_pass_rate = 0.0` for the 5-ep CFM output is a
  real measurement, not a bug. RDKit's `GetAllowedValences()` does
  not know about Pt square-planar coordination (it returns
  {2, 4} for neutral Pt, but the CFM emits 5- and 6-coordinate Pt
  with explicit single bonds to multiple heteroatoms — RDKit's
  per-atom valence counts are based on the explicit bond orders
  and don't account for the metal's d-electron shell).
- This is a *measurement* finding, not a fix. The CFM's generated
  SMILES may be chemically plausible (a chemist could draw them)
  but they are not "RDKit-valid" by the strict per-atom valence
  check. The fix would be either (a) post-hoc valence
  reconciliation that subtracts 2 from each metal's expected
  valence, or (b) a different metric like "all bonds form a
  connected molecular graph with no valence > 8".
- Future m14 reporting in paper §3 should cite BOTH `parsed_ratio`
  and `chemistry_validity_pass_rate` for any CFM cohort. The
  single number 0.998 is misleading; the two-number pair
  (1.000 / 0.000) is honest.

## Cross-references

- `todo.json` BLK-C3 (this fix)
- `molmetal/scripts/cfm_property_panel.py` (the modified script)
- `molmetal/reports/cfm_smoke_wire_5ep/generated_smiles.csv` (smoke input)
- `TODO/completed/blk_c4_cfm_bond_head_qed_2026_10_09.md` (parallel BLK-C4 GPU retest, will use this flag in its closeout)
- `TODO/completed/blk_l2_m2r2_alias_line_ref_2026_10_09.md` (parallel BLK-L2 doc fix)
- `AGENTS.md` §5.5 + §5.6 + §5.7 + §5.8 + §5.9
- `TODO/INDEX.md` §22 (will be added in final closeout wave per execution_order step 19)
