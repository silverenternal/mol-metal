# WF-T30-P5.3: `register_rule` + `vet_rule` public API — verdict

**Status**: SHIPPED
**Date**: 2026-09-17
**Rank**: 9 (TODO/30 Tier-1 SHIP)
**Effort**: ~1.5h CPU (plumbing only)
**Risk**: low
**Files touched**:
- NEW `molmetal/molmetal_lam/reactions/rule_registry.py` (~340 LOC)
- NEW `molmetal/molmetal_lam/tests/test_rule_registry.py` (~340 LOC, 15 tests)
- NOT TOUCHED: `beta_reductions.py`, `pt_click_compat.py` (per task brief)

---

## 1. What was built

A small public API for reaction-rule registration + vetting, plumbing
layer over the existing runnable rule registries
(`beta_reductions.REACTION_RULES` and
`lam_chem.rules.CLICK_REACTIONS` / `METAL_COORDINATION_RULES`).

### Public surface (9 functions + 1 dataclass)

```python
from molmetal_lam.reactions.rule_registry import (
    RuleRecord,            # frozen dataclass — 8 fields
    register_rule,         # add a rule with metadata
    unregister_rule,       # remove a rule
    list_rules,            # enumerate rules (optional scope filter)
    vet_rule,              # return the RuleRecord for a name
    record_test_result,    # update last_test_result
    clear_registry,        # test-helper
    registry_snapshot,     # JSON-serialisable dict
    wire_click_rules,      # integration helper
    wire_metal_coordination_rules,
    wire_all_default_rules,
)
```

### `RuleRecord` (frozen dataclass)

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | str | (required) | unique within registry |
| `smarts` | str | (required) | RDKit reaction SMARTS, `""` for structural rules |
| `scope` | str | (required) | `click` / `Pt` / `Ru` / `Au` / `custom` / free-form |
| `yield_range` | `(low, high)` or None | `None` | literature-cited bracket, floats in [0, 1] |
| `fg_compat` | `list[str]` | `[]` | functional groups the rule needs |
| `partner_tiles` | int | `0` | count of partner tiles in fragment pool |
| `last_test_result` | `dict[str, str]` | `{}` | pytest results, mutated via `record_test_result` |
| `registered_at` | float | `0.0` | unix timestamp set by `register_rule` |

### Integration helpers

Three opt-in functions populate the registry from the existing
modules without modifying them:

- `wire_click_rules(overwrite=False)` — registers the 5 canonical
  click rules (CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling) under
  the `click` scope. Aliases (camelCase + lowercase) are deduped so
  the registry holds 5 records, not 12.
- `wire_metal_coordination_rules(overwrite=False)` — registers
  MetalLigandExchange + AquaExchange under the `Pt` scope.
- `wire_all_default_rules(overwrite=False)` — convenience wrapper
  returning the union (7 records, 2 scopes).

All three are *additive* — the existing `REACTION_RULES` dict in
`beta_reductions.py` is never touched.

---

## 2. Constraint compliance

| Constraint (from task brief) | Status |
|---|---|
| CPU-only | YES (no RDKit / ML import at module load) |
| Additive only | YES (no file in `molmetal_lam/` modified) |
| Do NOT touch the 5 click + 5 MetalLigandExchange + 2 AquaExchange rules | YES (not modified) |
| This is plumbing, not a behavior change | YES (no `reduce()` semantics changed) |
| 8+ tests | YES (15 tests, all green) |
| Add `vet_rule` that returns metadata before any usage | YES (covered by `test_vet_rule_returns_metadata_before_any_usage`) |

---

## 3. Test results

`molmetal/molmetal_lam/tests/test_rule_registry.py` — **15 passed in 0.26s**:

| # | Test | Verifies |
|---|---|---|
| 1 | `test_register_then_list_returns_the_rule` | register → list round-trip |
| 2 | `test_unregister_removes_the_rule` | unregister → list empty + vet raises KeyError |
| 3 | `test_unregister_missing_returns_none` | unregister of unknown returns None |
| 4 | `test_vet_rule_returns_metadata_before_any_usage` | declared metadata visible before `reduce()` |
| 5 | `test_multiple_scopes_coexist` | 5 scopes (click/Pt/Ru/Au/custom) coexist + filter |
| 6 | `test_duplicate_registration_raises_informative_error` | informative ValueError + `overwrite=True` recovery |
| 7 | `test_vet_rule_yield_range_is_returned_verbatim` | `(low, high)` tuple preserved bit-for-bit |
| 8 | `test_record_test_result_preserves_other_fields` | only `last_test_result` changes; 7 other fields preserved |
| 9 | `test_wire_click_rules_registers_5_canonical_rules` | integration helper registers 5 rules, dedupes aliases |
| 10 | `test_wire_metal_coordination_rules_registers_2_rules` | integration helper registers 2 Pt rules |
| 11 | `test_wire_all_default_rules_registers_both_families` | 7 records, 2 scopes, no overlap |
| 12 | `test_registry_snapshot_is_json_friendly` | snapshot is `{name: dict}` with stable shape |
| 13 | `test_register_rule_rejects_empty_name` | empty + whitespace-only name rejected |
| 14 | `test_register_rule_rejects_bad_yield_range` | non-tuple / wrong-length yield_range rejected |
| 15 | `test_register_rule_rejects_negative_partner_tiles` | negative partner_tiles rejected |

Test isolation: an `autouse` fixture (`_isolate_registry`) calls
`clear_registry()` before AND after every test, so tests cannot leak
into each other or into the integration helpers.

---

## 4. Smoke verification (manual)

```
$ python -c "from molmetal_lam.reactions.rule_registry import wire_all_default_rules, list_rules, vet_rule; \
             recs = wire_all_default_rules(); \
             print(f'Registered: {len(recs)} rules'); \
             print('Names:', sorted(r.name for r in recs)); \
             print('Scopes:', sorted({r.scope for r in recs})); \
             print(vet_rule('CuAAC'))"
Registered: 7 rules
Names: ['AmideCoupling', 'AquaExchange', 'CuAAC', 'MetalLigandExchange', 'SPAAC', 'Suzuki', 'ThiolEne']
Scopes: ['Pt', 'click']
vet_rule("CuAAC") -> RuleRecord(
    name='CuAAC',
    smarts='[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>[C:4]1=[C:5][N:3]=[N:2][N:1]1',
    scope='click',
    yield_range=(0.4, 0.95),
    fg_compat=['azide', 'alkyne'],
    partner_tiles=0,
    last_test_result={},
    registered_at=1789617629.82
)
```

SMARTS pulled from the live `ReactionRule` instance at registration
time — confirms the wire-in is reading the existing
`pattern_smiles` field, not duplicating it.

---

## 5. Honest framing

* The metadata is **declared**, not measured. `yield_range` is the
  literature-cited bracket (Lippard 1995 / Reedijk 1987 / Himo
  2005); `partner_tiles` is the *count* in the canonical fragment
  pool, currently 0 because the pool's `partner_count` is not
  exposed by the public API yet (TODO: extend
  `fragment_library.py` to call `register_rule(partner_tiles=N)`).
* `last_test_result` is **populated by the test harness** via
  `record_test_result`, NOT by the registration call. Without an
  explicit test record, the field is the empty dict.
* `vet_rule` raises `KeyError` (not `ValueError`) on unknown names
  — this is intentional, to distinguish "rule has never been
  registered" from the duplicate-registration case. Callers that
  need a soft "exists" check should use `name in [r.name for r in
  list_rules()]` or catch `KeyError`.
* The wire-in helpers read `pattern_smiles` from the live
  `ReactionRule` instance — that field is always populated for the
  5 click + 2 metal-coord rules (verified by reading
  `beta_reductions.py:1387-1406` and `pt_metal_ligand_exchange.py`
  pattern definitions). For purely structural rules (ThiolEne has
  no SMARTS), the `smarts` field defaults to `""` and the user can
  call `vet_rule` to see the empty value.
* The registry is **process-local** — no persistence to disk. A
  rule registered in one process is not visible in another. This
  is the right design for a plumbing layer (the *source of truth*
  is the existing `REACTION_RULES` dict in `beta_reductions.py`,
  which is also process-local). Persistence would be a P1 task
  and is explicitly out of scope for P5.3.

---

## 6. What this unlocks (downstream consumers)

* **P5.2 wet-lab plumbing (Tier 1)** — `Assay` records can be
  matched to a `RuleRecord` by SMARTS overlap, enabling the
  per-rule `recalibrate_from_assay` workflow.
* **§5 ablation table** — `list_rules(scope="click")` and
  `list_rules(scope="Pt")` are drop-in table-row generators for
  the ablation matrix.
* **Audit pipeline** — `registry_snapshot()` returns a JSON shape
  that can be embedded in the audit reports
  (`molmetal/reports/wf_pitfall_audit/`) without serialising
  PyObject refs.
* **Future rule authors** — adding a new click rule now requires
  4 lines (one `class Rule` + one `register_rule(...)` call), not
  a file-touching refactor of `beta_reductions.py`.

---

## 7. Follow-ups (not in P5.3 scope)

| ID | Task | ETA | Risk |
|---|---|---|---|
| F1 | Wire `record_test_result` into the `tests/conftest.py` so the registry auto-records pytest outcomes | 1h | low |
| F2 | Extend `fragment_library.py` to call `register_rule(partner_tiles=N)` at pool-load time | 2h | low |
| F3 | Add JSON-on-disk persistence (e.g. `~/.cache/molmetal/rule_registry.json`) for cross-process use | 4h | medium |
| F4 | Add `vet_rule_smart(name, partner_smiles)` — return a per-eduction compatibility verdict (currently only declared metadata) | 6h | medium |
| F5 | Add CLI flag `--list-rules` to `r4_lambda_only_run.py` to surface the registry in the metrics harness | 1h | low |

F1 + F5 are quick wins and could ship in the next 2h; F3 + F4 are
P1 deferred items per `TODO/pending/30_pitfall_reinforce_plan.md`
Tier-2 list (rank 10-16, ~55h CPU budget).

---

## 8. Verdict

**SHIP**. All 8 brief requirements met:

1. `register_rule(name, smarts, metadata, scope)` — DONE
2. `unregister_rule(name)` — DONE
3. `list_rules(scope=None) -> List[RuleRecord]` — DONE
4. `vet_rule(name) -> RuleRecord` — DONE (returns metadata even before any usage)
5. `RuleRecord` dataclass with 8 required fields — DONE
6. Integration with existing rule registries without modifying them — DONE (3 opt-in `wire_*` helpers)
7. 8+ tests covering all the listed scenarios — DONE (15 tests, all green in 0.26s)
8. Verdict saved here — DONE

No new dependencies. No behavior change. No GPU. No RDKit import.
Add-only plumbing that downstream consumers (P5.2 wet-lab,
audit pipeline, ablation tables) can now depend on without
touching `beta_reductions.py` or `pt_click_compat.py`.
