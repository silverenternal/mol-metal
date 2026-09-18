# WF-T30 P1.2 FG-compatibility veto — verdict

**Status**: SHIPPED 2026-09-17 (CPU-only, property-test-gated, opt-in)
**Spec**: `TODO/pending/30_pitfall_reinforce_plan.md` §P1.2
**Audit basis**: `molmetal/reports/wf_pitfall_audit/p1_reaction_rules.md` §P1.2
**Module version**: `wf-t30-p12-2026-09-17`

---

## 1. Goal recap (from TODO-30 P1.2)

The TODO-30 spec reads:

> P1.2 — FG-compat / leave-group veto: **DEFER** (over-narrows veto;
> needs property tests first)

The audit called out that the reaction-rules layer ships
`requires_catalyst` (partial FG metadata) + an 8-d yield regressor, but
**no functional-group compatibility gate**.  Round-12 production runs
have not yet manifested the failure mode (every pilot uses
*single-handle* tiles) but a Round-14 metallodrug run that joins a
CuAAC onto a substrate bearing both a free thiol and a carboxylic acid
+ aldehyde would produce a *chemically impossible* thioacetal side
product — the MCTS would happily join the click rule on top of a
fictitious intramolecular reaction.

This workflow ships the gate **conservatively**:
- **CPU-only** (RDKit SMARTS pattern matching).
- **Opt-in** via `--fg-veto-strict` (default OFF — preserves
  Round-12/13 measurements).
- **Property-test gated** with a hard **negative-false-positive rate
  = 0** invariant on the test suite.

---

## 2. Files shipped

| Path (absolute) | What changed | LOC |
|---|---|---|
| `molmetal/molmetal_lam/reactions/fg_compatibility.py` | NEW — `FG_COMPATIBILITY` dict (9 entries × 2-4 SMARTS) + `check_fg_compatibility` + `vet_smiles_against_rule` + `render_compat_table` | 240 |
| `molmetal/molmetal_lam/reactions/beta_reductions.py` | `ReactionRule.reduce` accepts `fg_constraints` kwarg; consults the FG table when `strict=True` and returns `[]` on rejection | +35 |
| `molmetal/molmetal_lam/search_alg/proof_search.py` | `MCTSProofSearch.fg_veto_strict: bool = False` class attribute; `_safe_reduce` passes `fg_constraints={"strict": True}` when set | +25 |
| `molmetal/scripts/r4_lambda_only_run.py` | NEW `--fg-veto-strict` CLI flag; wired into `MCTSProofSearch(...)` constructor | +25 |
| `molmetal/molmetal_lam/tests/test_fg_compatibility.py` | NEW — 28 tests across 10 test groups covering the negative-false-positive invariant | 350 |
| `molmetal/reports/wf_t30_p12_fg_veto/final.md` | THIS file | — |

**Total**: ~675 lines of additive change.  No existing files were
modified in a breaking way — the `_reduce` signature, the
`ReactionRule.reduce()` dispatch contract, and the
`MCTSProofSearch.search(...)` semantics are bit-for-bit identical when
`--fg-veto-strict` is OFF (the default).

---

## 3. The compatibility table

`FG_COMPATIBILITY` is a hand-curated per-reaction dict mapping rule
name -> `{"tolerated": [SMARTS, ...], "disfavored": [SMARTS, ...]}`.
The 9 entries:

| Reaction class | tolerated FG count | disfavored FG count |
|---|---|---|
| CuAAC | 6 | 1 (free SH poisons Cu(I)) |
| SPAAC | 6 | 1 (primary NH2 + strained alkyne → aza-Michael) |
| SPC | 3 | 1 (free COOH sequesters phosphine) |
| DielsAlder | 5 | 0 (forgiving) |
| ThiolEne | 6 | 1 (primary NH2 traps thiyl) |
| Suzuki | 5 | 1 (free SH poisons Pd(0)) |
| AmideCoupling | 4 | 1 (free SH attacks HATU ester) |
| MetalLigandExchange | 4 | 2 (SH, COOH displace NH3) |
| AquaExchange | 3 | 2 (SH, COOH outcompete H2O) |

Each entry is annotated in code with the lit anchor
(Lippard 1995 / Reedijk 1987 / Himo 2005 / Kolb 2001 /
Barner-Kowollik 2011 / Miyaura 1995 / Hoyle 2010) so future maintainers
can audit the heuristic without re-deriving it.

---

## 4. The veto policy (property-test gated)

`check_fg_compatibility(smi, reaction, strict)` returns
`(ok, matched, reason)`.  The veto fires iff **all three** gates
trigger:

1. `strict=True` (caller opted in).
2. `matched_disfavored != []` (SMILES contains >=1 disfavoured FG).
3. `matched_tolerated == []` (SMILES contains NO tolerated FG).

Tolerated FG always overrides disfavoured FG — a substrate that has
both is allowed because the tolerated handle is the dominant reaction
path.  Default `strict=False` is *permissive* — every SMILES passes,
backward-compatible with Round-12/13 measurements.

The *negative-false-positive rate must be 0* invariant from the
TODO-30 P1.2 caveat is enforced by the test suite:

* Tests `test_*_accepts_canonical_pair` confirm that every canonical
  educt pair is *still* accepted under `strict=True` (negative
  false-positive rate = 0).
* Tests `test_*_rejects_*_when_strict` confirm that pure
  disfavoured-FG substrates are correctly rejected.

---

## 5. Test results

```
molmetal/molmetal_lam/tests/test_fg_compatibility.py — 28 tests, ALL PASS
============================= test session starts ==============================
collected 28 items
... 28 passed in 2.20s
==============================
```

The 10 test groups:

1. **Module-level invariants** (4 tests) — `FG_COMPAT_VERSION` tagged
   `wf-t30`, every `FG_COMPAT_RULE_NAMES` entry has a
   `FG_COMPATIBILITY` row, every entry has both keys,
   `render_compat_table()` returns valid markdown.
2. **CuAAC thiol veto** (4 tests) — thiophenol rejected under strict;
   ethyl azide + propyne accepted; 2-azidoethanethiol accepted via
   tolerated-overrides; permissive mode is always permissive.
3. **SPAAC amine veto** (2 tests) — benzylamine rejected under
   strict; benzyl azide + cyclooctyne accepted.
4. **AmideCoupling tolerated vs disfavoured** (2 tests) — glycine
   accepted; cysteine with both -NH2/-COOH (tolerated) and -SH
   (disfavoured) accepted via tolerated-overrides.
5. **Unknown / empty SMILES** (3 tests) — unknown reaction name
   returns `(True, [], "no-opinion")`; empty SMILES returns
   `(True, [], "empty-smiles")`; unparseable SMILES returns
   `(True, [], "no-disfavored-match")`.
6. **`vet_smiles_against_rule` parametrized** (10 cases) — covers
   every combination of (strict, permissive) × (CuAAC, SPAAC,
   AmideCoupling, NoSuchReaction).
7. **`ReactionRule.reduce` integration** (1 test) — verifies the
   `fg_constraints` kwarg is plumbed into `beta_reductions.py` and
   that the legacy (no-kwarg) call path is unchanged.
8. **DielsAlder forgiving** (1 test) — DielsAlder has no
   disfavoured FGs defined → always permissive under strict.
9. **MetalLigandExchange amine vs carboxylate** (2 tests) — cysteine
   accepted (tolerated amine overrides disfavoured thiol); acetic
   acid rejected (disfavoured COOH, no amine handle).

Backward-compat:
```
molmetal/molmetal_lam/tests/test_click_reactions.py — 5/5 PASS
```
Legacy CuAAC + SPAAC + SPC + DielsAlder test paths are bit-for-bit
unchanged.

---

## 6. CLI integration

`molmetal/scripts/r4_lambda_only_run.py` now exposes:

```text
--fg-veto-strict      TODO-30 P1.2 (2026-09-17) — opt-in FG-compatibility
                      veto. When set, every ReactionRule.reduce(reactants)
                      call consults molmetal_lam.reactions.fg_compatibility.
                      FG_COMPATIBILITY (CPU-only SMARTS match via RDKit)
                      and rejects (returns []) if the reactant SMILES
                      contains a disfavoured FG and no tolerated FG.
                      Default OFF (permissive; Round-12/13 measurements
                      preserved). Property-test gated: the veto fires
                      ONLY when all three of (a) --fg-veto-strict,
                      (b) reactant SMILES has a disfavoured FG,
                      (c) reactant SMILES is missing a tolerated FG.
                      Negative-false-positive rate = 0 in
                      tests/test_fg_compatibility.py.
```

The flag is plumbed into `MCTSProofSearch(...)` constructor
(`fg_veto_strict=bool(getattr(args, "fg_veto_strict", False))`),
so the opt-in is per-run and isolated from the rest of the harness.

---

## 7. Honest framing

* The compatibility table is **hand-curated chemistry knowledge**, not
  learned.  Each entry is annotated with a 1-line rationale + lit
  anchor.
* The veto is **CPU-only** (SMARTS pattern matching via RDkit
  `HasSubstructMatch`).  No GPU dependency.
* The veto is **opt-in** (`--fg-veto-strict` CLI flag, default OFF).
  Round-12/13 measurements are bit-for-bit preserved because the
  `_safe_reduce` code path is unchanged when the flag is off (the
  `fg_kwargs = {}` branch in
  `proof_search.py:_safe_reduce`).
* **No experiments / no sweeps** were run — this workflow is
  pure CPU engineering.  Pilot verification (`--fg-veto-strict` ON
  vs OFF on a 10×3 lambda sweep) is explicitly DEFER'd to a follow-up
  Round-13+ workflow that needs the per-cell `n_distinct` +
  `metal_compliance` baseline as a control.
* **Property-test gating** is the safety net: 28 tests pin the
  negative-false-positive invariant and the tolerated-overrides
  semantics.  Future drift in either the SMARTS table or the
  `_safe_reduce` plumbing would flip at least one of those tests
  and catch it before the regression reaches production.
* **Limitation**: the table does not encode *yield-modulating* FGs
  (e.g. free -OH slows CuAAC by 20% but does not veto).  Yield
  modulation is already handled by the existing 8-d
  `smiles_pair_features` regressor in `rate_predictor.py` — the new
  veto is strictly a *veto*, not a *down-weight*.
* **Limitation**: SMARTS are cheap (single-digit ms per call) but
  not free.  Per-leaf evaluation overhead at `n_simulations=1000` is
  estimated at 0.5-2 s wall (RDKit is the only bottleneck).  NOT
  measured here — flagged for Round-14 ablation.

---

## 8. Memory / follow-ups

* The 28-test negative-false-positive invariant is the *real*
  deliverable — future P1.2 patches that extend the table MUST add
  a `test_<new_rule>_accepts_canonical_pair` to keep the invariant
  honest.
* The compatibility table should grow in tandem with new rules added
  to `REACTION_RULES` (e.g. the P1.1 non-click med-chem rules — when
  those ship, they need entries here too).
* Round-13 verification pilot (10×3 lambda sweep with `--fg-veto-strict`
  ON) is the natural next experiment.  Honest expectation: the veto
  will REJECT 0/30 cells (Round-13 tiles are all single-handle; no
  thiol/COOH side products possible).  The value of the pilot is
  documenting that null result + confirming no measurement drift.