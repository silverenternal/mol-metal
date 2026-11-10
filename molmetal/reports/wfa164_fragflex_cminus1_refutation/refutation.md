# Wf-A164 — adversarial refutation of branch D (`wfa142_fragflex_cminus1_probe.py`)

**VERDICT: NOT REFUTED.** All load-bearing claims reproduced independently on disk; 0 GPU.

| # | Check | Result |
|---|---|---|
| 1 | Claims: literals gone, replacements computed | PASS (4 nits) |
| 2 | Regression test bites / negative control fails pre-repair | PASS |
| 3 | Guard not vacuous | PASS (main guard) / **FAIL** (`cohort_detail_recheck` cannot fire) |
| 4 | Nothing broke on the default pool | PASS |
| 5 | Scope discipline | PASS |

## What I re-measured (independently)
- AST walk: **582** string literals, **50** with a digit — identical to the census denominator. The 163 numeric constants (outside the census's scope) carry no pool number (`17` = cohort name, `120` = CLI default, `20` = sha chunk).
- Renderings re-derived from the real pool: `epoch 17 rows of the pool (704 rows)` / `every row of every epoch (12,669 rows)` / `... at most 60 rows per epoch, all 18 epochs (1,062 rows)`.
- Pool: sha256 `2d7b7c44…fec7384` matches the artifact; 12,669 records / 18 epochs / 704 rows in epoch 17 by `csv.DictReader`.
- `pytest` (own test + pre-existing): **25 passed** in 21.05 s, CPU-only.
- Guard: an unbacked token (424242) injected into each of the 5 narrative paths is REJECTED in every case.
- Negative control proven to bite: with the guard disabled, the old literal returns `rc=0, status=PASS` → the branch's `assert rc == 1` test FAILS pre-repair.

## Surviving concerns (non-blocking)
1. **`cohort_detail_recheck` is tautological** — `payload["cohort_detail"]` and the recheck both call the same pure `derive_cohort_detail`, so it can never disagree; the branch's own negative-control asserts `matches is True` under the monkeypatched literal. Its rule text is false for the case it claims to cover.
2. **"Backed" ≠ "correct"** — a wrong-but-backed claim passes (`'spanning 3 epochs'` on an 18-epoch pool → guard PASS; `'(12,669 rows)'` for a 704-row cohort → guard PASS), and small integers are spuriously backed by the pool path / `method.gpu` / the artifact name (0,1,4,5,10,20,142,2026 all accepted as "backed").
3. **Disclosure**: "0 of 367 shared leaves changed" holds only after excluding `wall_seconds*` / `generated_utc` (7 leaves changed in a raw diff; the exclusion is declared in `compare_default_pool.py` L52-54 but not in the prose report).
4. **Verifiability**: the script is untracked, so there is no diff and no pre-repair snapshot on disk (a concurrent branch kept one); the pre-repair literals are corroborated only by the two published `wfa142_fragflex_cminus1/*.json` artifacts.

Machine-readable detail: `refutation.json` (same directory).
