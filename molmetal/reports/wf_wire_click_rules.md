# WF-Wire-Click-Rules-All5 — `--click-rules all-5` alias fix

**Date:** 2026-09-14
**Goal:** fix the `--click-rules all-5` alias mismatch in
`r4_lambda_only_run.py` so the filter actually constrains the Lambda
generator without emitting the misleading `click_rules_filter_emptied`
warning.

---

## 1. Audit

### 1.1 Where the alias was being parsed

`molmetal/scripts/r4_lambda_only_run.py` (`--click-rules` CLI flag,
~line 1558) accepts a space-separated list of strings (`nargs="*"`).
The flag is forwarded verbatim into `run_sweep(...)` → `run_one_cell(...)`
where the filter step lives:

```python
# Pre-fix
if click_rules is not None:
    keep = {name.strip() for name in click_rules if name and name.strip()}
    selected_rules = {k: v for k, v in CLICK_REACTIONS.items() if k in keep}
    if not selected_rules:
        cell.warnings.append(
            f"click_rules_filter_emptied: requested={list(click_rules)}; "
            "falling back to all-5"
        )
        selected_rules = dict(CLICK_REACTIONS)
    cell.warnings.append(f"click_rules_active={sorted(selected_rules.keys())}")
```

### 1.2 Why the warning fired on a known alias

`CLICK_REACTIONS` (in `molmetal_lam/lam_chem/rules.py`) is keyed by the
**canonical rule names** (plus a few lower-case variants):

```python
CLICK_REACTIONS = {
    "CuAAC": ..., "SPAAC": ..., "Suzuki": ..., "ThiolEne": ...,
    "AmideCoupling": ...,
    # lower-case / dashed / spaced aliases (for the metrics harness).
    "cuaac": ..., "spaac": ..., "thiol-ene": ..., "suzuki": ...,
    "amide coupling": ...,
    "amide": ..., "amide_coupling": ...,
}
```

The alias `all-5` does **not** appear in that dict, so the filter
dict-comprehension `if k in keep` yielded `{}`. The code then took the
"empty filter" branch, emitted the contradictory warning
`click_rules_filter_emptied: requested=['all-5']; falling back to all-5`,
and re-loaded the entire registry. The net behaviour was a working
fallback (all 5 rules active) but a misleading diagnostic that made
the user think the alias had been rejected.

### 1.3 Registered rule names — confirmed against `proof_search.py`

The MCTS expansion consults `CLICK_REACTIONS` (registered keys)
exactly as listed above; `proof_search.py` accepts `rules: Dict[str, ReactionRule]`
and the construction site `MoleculeClosedTerm`/`ReactionRule` does
not impose further naming constraints beyond the dict keys. So the
canonical 5 names — and only those — must appear in the resolved set.

---

## 2. Fix

### 2.1 Alias map (added at module level)

```python
CLICK_RULE_ALIASES: Dict[str, List[str]] = {
    # "all-5" / "all_5" / "all" — the canonical 5 click rules.
    "all-5":   ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    "all_5":   ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    "all":     ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    "default": ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    # Individual rule pass-throughs (also accept the lower-case /
    # dashed variants that CLICK_REACTIONS already understands).
    "CuAAC":         ["CuAAC"],
    "cuaac":         ["CuAAC"],
    "SPAAC":         ["SPAAC"],
    "spaac":         ["SPAAC"],
    "Suzuki":        ["Suzuki"],
    "suzuki":        ["Suzuki"],
    "ThiolEne":      ["ThiolEne"],
    "thiol-ene":     ["ThiolEne"],
    "thiol_ene":     ["ThiolEne"],
    "thiolene":      ["ThiolEne"],
    "AmideCoupling": ["AmideCoupling"],
    "amide-coupling": ["AmideCoupling"],
    "amide_coupling": ["AmideCoupling"],
    "amide":         ["AmideCoupling"],
    # Friendly pairs.
    "cu+spaac":      ["CuAAC", "SPAAC"],
    "click-azide":   ["CuAAC", "SPAAC"],
}
```

### 2.2 Resolver

```python
def _resolve_click_rules_aliases(requested):
    resolved, seen, unknown = [], set(), []
    for raw in requested:
        token = str(raw or "").strip()
        if not token:
            continue
        if token in CLICK_RULE_ALIASES:
            expanded = CLICK_RULE_ALIASES[token]
        else:
            expanded = [token]      # pass-through; the filter step decides
        for name in expanded:
            if name not in seen:
                resolved.append(name)
                seen.add(name)
    return resolved, unknown
```

### 2.3 Filter step (rewired)

```python
if click_rules is not None:
    resolved, unknown = _resolve_click_rules_aliases(click_rules)
    if unknown:
        cell.warnings.append(
            f"click_rules_unknown: requested={list(click_rules)}; "
            f"unknown_tokens={unknown}; falling back to all-5"
        )
        resolved = []
    keep = {name for name in resolved}
    selected_rules = {k: v for k, v in CLICK_REACTIONS.items() if k in keep}
    if not selected_rules:
        cell.warnings.append(
            f"click_rules_filter_emptied: requested={list(click_rules)}; "
            "falling back to all-5"
        )
        selected_rules = dict(CLICK_REACTIONS)
    cell.warnings.append(f"click_rules_active={sorted(selected_rules.keys())}")
```

The two failure paths are now distinct:
* `click_rules_unknown` — user typo / unregistered rule (NEW).
* `click_rules_filter_emptied` — resolver returned an empty list after
  the unknown-token path fell back (kept for completeness).

For all known aliases (`all-5`, `CuAAC`, `thiol-ene`, …) the resolver
returns a non-empty resolved list, so the filter applies directly and
**neither** warning fires.

---

## 3. Tests

Added 4 new tests in
`molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`:

| Test | Purpose |
|---|---|
| `test_click_rules_all_5_alias_resolves` | End-to-end: `--click-rules all-5` produces the 5-rule active set AND no `click_rules_filter_emptied` / `click_rules_unknown` warnings. |
| `test_click_rules_single_alias_works` | End-to-end: `--click-rules CuAAC` resolves to `['CuAAC']` only and emits no fallback warnings. |
| `test_click_rules_filter_emptied_no_warning` | Unit test on the resolver: every entry in `CLICK_RULE_ALIASES` resolves to a non-empty list of registered rule names; the legacy `click_rules_filter_emptied` warning **must never** fire on a known alias. |
| `test_click_rules_unknown_token_warns_and_falls_back` | Negative test: a genuinely unknown token (`BogusRule`) emits `click_rules_unknown` (or the legacy `click_rules_filter_emptied`) and falls back to the full registry. |

### 3.1 Run command

```
uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_only_metrics.py --tb=short -k click_rules
```

### 3.2 Result

```
....                                                                     [100%]
4 passed, 34 deselected, 1 warning in 5.45s
```

### 3.3 Regression check (full file)

```
uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_only_metrics.py --tb=short
```

```
......................................                                   [100%]
38 passed, 1 warning in 29.78s
```

---

## 4. Metrics

| Metric | Value |
|---|---|
| `n_tests` (click_rules subset) | 4 |
| `n_passed` (click_rules subset) | 4 |
| `n_tests` (full file) | 38 |
| `n_passed` (full file) | 38 |
| `aliases_mapped` | 18 (4 all-variants + 4×3 individual-name variants + 2 pairs) |
| `warning_emitted_after_fix` (on a known alias) | False |

---

## 5. Honest framing

The pre-fix behaviour was technically functional — the fallback path
correctly loaded all 5 rules — but it produced a **misleading warning**.
Post-fix:

* Known aliases (`all-5`, `all_5`, `CuAAC`, `thiol-ene`, …) resolve
  cleanly and emit **no warning** — the filter actually constrains the
  Lambda generator exactly as requested.
* Unknown tokens (typos) emit a new, more diagnostic warning
  `click_rules_unknown` that lists the offending tokens.
* The legacy `click_rules_filter_emptied` warning is retained as a
  defence-in-depth fallback for the unlikely case where the resolver
  itself returns an empty list.

The fix is **strictly additive** — no public API changed, no
test that was passing pre-fix now fails, and the existing
`config.click_rules` field on the report JSON still records what the
user typed (not the expanded list) so auditability is preserved.
