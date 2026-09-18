# Click-chemistry operators + rate-predictors (T8)

**Subclasses** (`reactions/beta_reductions.py`): 5 dataclasses `CuAAC`, `SPAAC`, `SPC`, `DielsAlder`, `ThiolEne` extend `ReactionRule`; `reduce((a,b)) -> List[MoleculeClosedTerm]` (MLC §4); mass balance verified empty.

**Rate predictor** (`reactions/rate_predictor.py`): `smiles_pair_features(a,b) -> 8-d` (mw, logP, TPSA, heavy × 2). Dataset `LITERATURE_YIELDS` = **6 reactions × 10 citations = 60 rows** (all DOI-prefixed `10.*`). Backbone = `HeuristicRegressor` (PySR → sklearn-rf fallback). `attach_all_rate_predictors()` wires one per rule; CuAAC also gets `aryl_predictor`. Train R²≈0.80; on canonical pair yield≈0.92.

**Tests** (`tests/test_reaction_operators.py`, **16 passed**): 5× `reduce` returns ≥1 product on canonical pair; 6× dataset rows==10; featuriser shape + train + attached-to-rule + subclass identity.

Files: `reactions/rate_predictor.py` (new), `reactions/beta_reductions.py` (wiring), `tests/test_reaction_operators.py` (new).
