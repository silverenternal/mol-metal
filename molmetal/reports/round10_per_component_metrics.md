# Round-10 per-component sanity metrics report (axis F)

Date: 2026-09-13

Harness:
- File: molmetal/molmetal_lam/tests/test_round10_metrics_harness.py
- Runner: `uv run pytest -q molmetal/molmetal_lam/tests/test_round10_metrics_harness.py --tb=short -s`
- Result: 6 passed in 4.84s
- Constraints respected: spot-check only, no full regression, no sweep, single pocket micro-bench deferred.

## Observed values (all 6 metrics present and passing)

| # | Metric                          | Observed                                     | Threshold          |
|---|---------------------------------|----------------------------------------------|--------------------|
| 1 | ARITY_HIT_RATE                  | 0.998 (1565/1568)                            | >= 0.95            |
| 2 | METAL_GEOMETRY_OK               | 1.000 {Pt_II:4, Ru_II:6, Zn_II:4, Ir_III:6}   | exact match        |
| 3 | DATIVE_FRACTION                 | 1.000 (20 coordination bonds)                | == 1.0             |
| 4 | BOND_KIND_DISTRIBUTION          | {covalent:1, dative:1, aromatic:3, hydrogen:1}| all > 0            |
| 5 | CLICK_RULE_COVERAGE             | 5/5; rules.py=present                        | == 5               |
| 6 | TILE_POOL_SIZE                  | 220 (target 220 or 204)                      | in {200,204,220}   |

## Interpretation

- ARITY_HIT_RATE 0.998 (1565/1568) — primitive/metal atoms recovered from SMILES are arity-correct essentially all the time. The remaining 3 misses are the long-tail fragments that fall outside the PRIMITIVE_ATOMS / METAL_ATOMS dictionary (same behaviour as round-7/8 spot-check).
- METAL_GEOMETRY_OK 1.000 — all four canonical metals hold the expected coordination number (Pt_II=4 square-planar, Ru_II=6 octahedral, Zn_II=4 tetrahedral, Ir_III=6 octahedral). The Round-3 Au_III arity property-test caught bug does not regress.
- DATIVE_FRACTION 1.000 — every bond minted via Bond.dative against a metal atom lands in DATIVE (not COVALENT), confirming the FreeSiteLedger distinguishes dative donation from covalent sharing.
- BOND_KIND_DISTRIBUTION shows all four Bond kinds (COVALENT, DATIVE, AROMATIC, HYDROGEN) mint at least once and in non-degenerate counts. Aromatic naturally returns 3 because `Bond.aromatic` takes a cycle; counts remain consistent with prior rounds.
- CLICK_RULE_COVERAGE 5/5 — all of CuAAC, SPAAC, thiol-ene, Suzuki, amide coupling are exposed (case-insensitively) by molmetal_lam.lam_chem.rules.
- TILE_POOL_SIZE 220 — matches the round-7 well-formed fragment library size exactly (220), unchanged by round-8/9 work.

## Status

- All 6 metric assertions implemented as pytest tests AND print observed values (run with `-s`).
- pytest returns 6 passed / 0 failed / 0 skipped; no metric required skipping.
- No edits needed — file was already complete from the codex:rescue agent (task ad3955ab0d97ef577).
- No experiments run. Pure algorithmic / harness work this axis.