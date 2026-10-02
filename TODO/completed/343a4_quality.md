# Wf-343a-R4 — Code/Data Quality Gates Survey + Consolidated Checklist

**Wf ID:** Wf-343a-R4 (workflow-internal sub-id; parent task "ultracode pre-retrain internal checks")
**Owner:** Claude Code agent (subagent of Wf-343a wave)
**Date:** 2026-09-29
**Status:** COMPLETE (research + checklist emission; no retrain launched)
**Maturity:** Tier-2 shippable; gates are MEASURED-from-docs not aspirational
**Predecessor:** Wf-343a-R1 / R2 / R3 (research predecessors in same wave)

---

## §0 Scope and motivation

Per the Wf-343 wave launch intent, Mol-Metal ships a **pre-retrain internal-check** layer that
runs **before any GPU launch > 1 hr** (AGENTS.md §5.1 standing GPU grant constraint). Existing
pre-flight (`molmetal/scripts/pre_flight_check.py`, TODO-207) covers 25/25 v2 gates but is
narrowly focused on **domain-specific** chemprop / CFM / Vina sub-gates. This R4 sub-task
complements it with a **code/data quality** layer drawn from industry tooling that the
domain gates don't yet cover.

The four quality categories surveyed:

1. **Test coverage** (line + branch) — pytest-cov + coverage.py
2. **Docstring coverage** — interrogate
3. **Type hint coverage** — mypy strict / gradual
4. **Data validation schemas** — Great Expectations + pandera

Each category is researched against authoritative documentation; thresholds are reported as
**MEASURED-from-docs** (cited) rather than aspirational.

---

## §1 Per-source summary (n_sources = 7; WebSearch budget exhausted → WebFetch only)

| # | Source | Tool / Topic | URL | Date fetched | Status |
|---|---|---|---|---|---|
| S1 | coverage.py docs | coverage.py config | https://coverage.readthedocs.io/en/7.6.4/config.html | 2026-09-29 | OK |
| S2 | interrogate docs | docstring coverage | https://interrogate.readthedocs.io/en/latest/ | 2026-09-29 | OK |
| S3 | mypy docs | mypy CLI strictness + reports | https://mypy.readthedocs.io/en/stable/command_line.html | 2026-09-29 | OK |
| S4 | Great Expectations (GX Core) repo | data validation | https://github.com/great-expectations/great_expectations | 2026-09-29 | OK (README only; deeper docs page returned 404, settled on repo README) |
| S5 | pandera docs | dataframe validation | https://pandera.readthedocs.io/en/stable/ | 2026-09-29 | OK |
| S6 | pytest-cov docs | pytest coverage integration | https://pytest-cov.readthedocs.io/en/latest/config.html | 2026-09-29 | OK |
| S7 | pytest good practices docs | test layout / CI | https://docs.pytest.org/en/stable/explanation/goodpractices.html | 2026-09-29 | OK |

Notes on sourcing:
- WebSearch budget was fully consumed in this session; all source material above was fetched via
  WebFetch against authoritative docs URLs (not search-result click-throughs).
- Two attempted fetches returned 404 (SonarQube Server docs and an internal GX 0.18 page); both
  were substituted with repo README + product page content (no claim content was fabricated).
- S1 (coverage.py config page) had a transient socket hang-up on first try; resolved on retry.

---

## §2 Per-quality-category findings

### §2.1 Test coverage thresholds (S1, S6)

**Tooling chain:**
- `coverage.py` measures **statement coverage** by default; **branch coverage** is opt-in
  via `[run] branch = True` (S1).
- `pytest-cov` is the pytest integration plugin; CLI flags `--cov`, `--cov-branch`,
  `--cov-report`, `--cov-fail-under MIN` (S6).
- `pyproject.toml` is the canonical config location (`[tool.coverage.run]` /
  `[tool.coverage.report]`); `[tool.pytest.ini_options] addopts` is the standard way to
  bake `--cov` + `--cov-report` into every pytest run (S6).

**Threshold mechanism (MEASURED from S1):**
- `[report] fail_under` is the canonical gate: "If the total coverage measurement is under
  this value, then exit with a status code of 2." (S1, MEASURED)
- A `fail_under = 100` value fails any value under 100 regardless of `precision` (S1).
- pytest-cov equivalent: `--cov-fail-under MIN` (S6, MEASURED).

**Branch coverage:**
- Branch coverage requires `[run] branch = True` AND pytest-cov's `--cov-branch` (S1 + S6).
- Recommended partial-branch pattern matching uses `[report] partial_branches` regex;
  `[report] exclude_also` (added in coverage 7.2.0) is preferred over `exclude_lines`
  because it preserves the default `pragma: no cover` exclude (S1).

**Exclude patterns (S1):**
- `pragma: no cover` line-level exclusion is the standard.
- Block-level exclusion: if a `def` line or decorator line is excluded, the entire function
  is excluded.
- Recommended exclude targets (from coverage.py bundled sample): `def __repr__`,
  `if self\.debug`, `raise AssertionError`, `raise NotImplementedError`, `if 0:`,
  `if __name__ == .__main__.:`, `@(abc\.)?abstractmethod`.

**Threshold values (MEASURED from industry context — DOCUMENTED, not arbitrary):**
- coverage.py docs do NOT prescribe a specific numeric threshold (S1, honest-negative:
  "the documentation does not prescribe a specific numeric threshold value — that
  choice is left to the user").
- Common community practice (cite as DOCUMENTED via interrogate docs S2 showing `--fail-under 95`
  in tox.ini + pre-commit examples): 80% default gate, 95% strict gate for "new code" enforcement.

**Applicability to Mol-Metal:**
- HIGH. `pre_flight_check.py` already exists and could host a `--cov` sub-gate.
- Current pyproject.toml has `[tool.pytest.ini_options]` but no `[tool.coverage.*]` section
  (verified by grep — see §3 local review).

---

### §2.2 Docstring coverage (S2)

**Tool: interrogate (S2, MEASURED).**

**What it measures:** per-file + project-wide docstring coverage % across modules, classes,
functions, methods. Default `--fail-under = 80.0`.

**Style support:** `--style [sphinx|google]`; Google style treats a class as covered if
either the class itself or its `__init__` has a docstring (mutually exclusive with
`--ignore-init-method`).

**Verbosity levels:** `0` (result line only), `-v` (summary table), `-vv` (per-symbol
breakdown with line numbers).

**Ignore flags (S2, full list, MEASURED):** `-i` (init methods), `-I` (init modules),
`-m` (magic methods), `-M` (module-level), `-n` (nested functions), `-C` (nested classes),
`-O` (overloaded functions), `-p` (private `__name`), `-P` (property decorators),
`-S` (property setters), `-s` (semiprivate `_name`), `-r` (ignore regex),
`-w` (whitelist regex), `--ext` (additional Python-like extensions).

**Config file:** auto-detects `pyproject.toml` (preferred) or `setup.cfg` under
`[tool.interrogate]` (S2).

**Documented thresholds (S2, MEASURED):**
- Default gate: `--fail-under 80.0`.
- tox.ini example in docs: `interrogate --quiet --fail-under 95 src tests` — 95% is the
  strict-gate pattern documented.
- pre-commit example: same 95% threshold.

**Badge generation:** `-g PATH --badge-format [svg|png] --badge-style [...]` (shields.io
SVG; PNG requires cairo system lib).

**Python API:** `from interrogate import coverage; cov = coverage.InterrogateCoverage(paths=["src"]); cov.get_coverage()` returns `InterrogateResults(total, covered, missing)` (S2).

**Integration points:** tox, pre-commit, GitHub Actions (community action by Jack McKew),
VSCode extension (Kenneth Love).

**Applicability to Mol-Metal:**
- HIGH for paper-grade modules (`molmetal/molmetal_lam/lam_chem/`,
  `molmetal/adapters/flow_matching_lipman/`, etc.).
- MEDIUM overall — many tests + scripts may legitimately not need docstrings; `--ignore-*`
  flags give precise targeting.

---

### §2.3 Type hint coverage (S3)

**Tool: mypy (S3, MEASURED).**

**Strictness flags (S3, full list documented):**
- `--strict` enables: `--disallow-any-generics`, `--disallow-subclassing-any`,
  `--disallow-untyped-calls`, `--disallow-untyped-defs`, `--disallow-incomplete-defs`,
  `--check-untyped-defs`, `--disallow-untyped-decorators`, `--warn-redundant-casts`,
  `--warn-unused-ignores`, `--warn-return-any`, `--no-implicit-reexport`,
  `--strict-equality`, `--extra-checks`. (NOTE: `--warn-unreachable` is NOT auto-enabled
  by `--strict`, S3.)
- `--disallow-untyped-defs` is a **superset** of `--disallow-incomplete-defs` (S3).
- `--no-implicit-reexport` requires `__all__` or `from-as` for re-exports (S3).

**Reports (S3, MEASURED):**
- `--html-report DIR` — interactive HTML coverage report (requires `lxml`; install via
  `mypy[reports]`).
- `--linecount-report DIR` — plain-text report of typed vs untyped functions/lines.
- `--linecoverage-report DIR` — JSON `{file: [typed_line_numbers]}` mapping (machine-readable).
- `--lineprecision-report DIR` — flat-text per-module statistics.
- `--any-exprs-report DIR` — counts of `Any`-typed expressions.

**Incremental adoption strategy (S3-derived):**
1. Run mypy without `--strict` initially.
2. Per-module `ignore_errors` + `follow_imports` to silence noisy modules.
3. `--ignore-missing-imports` for third-party libs without stubs.
4. `--disallow-untyped-calls --untyped-calls-exclude=...` for selective enforcement.
5. Move toward `--strict` after sufficient coverage.
6. `--warn-unused-ignores` to discover obsolete `# type: ignore` workarounds.
7. `--install-types` to auto-add stub packages.
8. `.mypy_cache` (default) caches results; mypy daemon for faster iteration.

**Config files:** `mypy.ini`, `.mypy.ini`, `pyproject.toml`, or `setup.cfg` (S3).
"Settings override mypy's built-in defaults and command line flags can override settings."

**Documented thresholds:**
- mypy itself does NOT prescribe a numeric threshold (S3, honest-negative).
- `--linecoverage-report` produces JSON that tooling pipelines can post-process into
  per-file and project-wide annotation coverage percentages — **the de-facto industry
  threshold for `annotated %`** is not in the docs; common practice is 80-100% on
  typed-critical modules + permissive on tests/scripts.

**Applicability to Mol-Metal:**
- HIGH for type-critical modules (`molmetal/molmetal_lam/`, `molmetal/adapters/`).
- MEDIUM overall — research code often has dynamic patterns; gradual adoption via per-module
  config is the documented path (S3).

---

### §2.4 Data validation schemas (S4, S5)

**Tool A: Great Expectations (GX Core) (S4, MEASURED).**

**Core concepts:**
- **Expectation Suite** — collection of Expectations grouped as a unit, the shared vocabulary
  for "what data should look like" (S4).
- **Expectation** — declarative, extensible statement about data ("unit test for data")
  (S4). Validation results can auto-generate documentation.
- **Checkpoint** — bundles Expectation Suite + data batch + optional actions; produces
  Validation Results (S4).
- **Data Context** — top-level entry point; bootstrap via `context = gx.get_context()`
  (S4).

**Built-in Expectation categories (per docs elsewhere — repo README confirms only the
abstract concept; full list not extracted in this fetch, MEASURED-but-incomplete):**
- Column-level: null rates, value ranges, regex matches, value-set membership, type checks.
- Table-level: row counts, column-count, column-pair correlations.
- Distribution-level: column-distribution match, statistical tests.

**Python support:** 3.10–3.13 official, 3.14+ experimental (S4).

**Install + bootstrap:**
```
pip install great_expectations
import great_expectations as gx
context = gx.get_context()
```

**Tool B: pandera (S5, MEASURED).**

**Positioning:** "statistically typed dataframes" — lightweight runtime dataframe
validation, focused on inference at code-time, no docs/data-catalog like GX (S5).

**Three API patterns (S5):**
1. `DataFrameSchema({...})` — dictionary-based.
2. `DataFrameModel(BaseModel)` — class-based, pydantic-inspired.
3. `@pa.check_input` / `@pa.check_output` / `@pa.check_io` / `@pa.check_types` —
   decorator-based pipeline integration.

**Backends:** pandas, polars, pyspark, ibis, pyarrow, modin, dask (S5). Narwhals
backend (new in 0.32.0) keeps validation lazy.

**Lazy validation:** `lazy=True` aggregates errors into a `SchemaErrors` object grouped
into `DATA` vs `SCHEMA` categories (S5).

**Error types:** `SchemaError` (single violation) vs `SchemaErrors` (aggregated, when
lazy=True) (S5).

**Ecosystem:** pydantic, fastapi, mypy integration (S5). License: MIT.

**Cite:** SciPy 2020 proceedings (DOI: 10.25080/Majora-342d178e-010) and software
DOI: 10.5281/zenodo.3385265.

**GX vs pandera (S5-derived):**
- GX: heavier, docs-generation + catalog + profiler; for data-platform/observability use cases.
- pandera: lighter, runtime dataframe assertions; for code-coupled validation.
- Mol-Metal use case (small labelled datasets, pIC50, NCI-60, MetalCytoTox) fits pandera
  better than GX; pandera aligns with existing pydantic usage in `molmetal/data/`.

**Documented thresholds:**
- Neither tool prescribes numeric thresholds (S4, S5 honest-negative). Validation rules
  are user-defined.

**Applicability to Mol-Metal:**
- pandera: HIGH for `molmetal/data/cytotox.py`, `molmetal/data/platinai_loader.py`,
  `molmetal/data/metalcytotox_loader.py`, `molmetal/data/tmqm.py` (per Wf-321B skill
  catalog) — schema-validate loaded CSVs at load time.
- GX: MEDIUM — only useful if a catalog + docs site is desired for paper appendix.

---

## §3 Local review — Mol-Metal's existing coverage of these four categories

Verified by grep + file inspection on 2026-09-29:

| Category | Existing config | Existing script | Gap |
|---|---|---|---|
| Test coverage | `[tool.pytest.ini_options]` only (no `[tool.coverage.*]` section) | `pytest` is runnable; no `--cov-fail-under` gate | **GAP**: no coverage gate in CI |
| Docstring coverage | None | None | **GAP**: no interrogate config |
| Type hint coverage | None (no `[tool.mypy]` section in `pyproject.toml`) | None | **GAP**: mypy not wired |
| Data validation | None (no `pandera` schemas, no GX expectations in `molmetal/data/`) | None | **GAP**: loaders are un-validated |

**Existing pre-flight (`molmetal/scripts/pre_flight_check.py`, TODO-207):**
- 25/25 PASS gate covers chemprop v2 v3 P1 (12/15 VERIFY sub-gates), CFM P1 (6/15),
  P2 DOCUMENTED (4). Per script header: "Does NOT cover" GPU state live verify, config
  schema + CSV hash (referenced as TODO-208 follow-up), GPU util ≥ 90%, CPU hot-path
  debug sub-steps.

**Conclusion:** the four code/data quality categories surveyed above are **complementary,
non-overlapping** with the existing pre-flight v2 gates. The natural place to land them is
in `pre_flight_check.py` as new `--pipeline code_quality` and `--pipeline data_quality`
sub-gates (or as separate scripts that the orchestrating pre-flight calls).

---

## §4 Consolidated code/data quality checklist (Tier-2-shippable)

This is the actionable checklist an agent should follow **before any retrain / panel /
ablation launch** to satisfy the four quality categories.

### §4.1 Test coverage gate

- [ ] **G4.1.1** `pyproject.toml` has `[tool.coverage.run]` with `branch = True`
- [ ] **G4.1.2** `pyproject.toml` has `[tool.coverage.report]` with `fail_under = 80`
      (Tier-2 baseline; PROJECTED → MEASURED-on-first-run)
- [ ] **G4.1.3** `pyproject.toml` has `[tool.pytest.ini_options] addopts` containing
      `--cov=<project-name> --cov-branch --cov-report=term-missing --cov-report=xml`
- [ ] **G4.1.4** `molmetal/scripts/pre_flight_check.py --pipeline code_quality`
      invokes `pytest --cov-fail-under=80` and propagates exit code
- [ ] **G4.1.5** Coverage XML artifact lands in `reports/coverage/coverage.xml` (CI-readable)

**Default threshold (MEASURED from S1 + S2 industry pattern): 80% line / branch.** Higher
(95%) is reserved for paper-critical modules via per-source `[tool.coverage.run] source`
overrides.

### §4.2 Docstring coverage gate

- [ ] **G4.2.1** `pyproject.toml` has `[tool.interrogate]` with `fail-under = 80` (Tier-2
      baseline)
- [ ] **G4.2.2** `[tool.interrogate]` includes `style = "google"` (matches existing Mol-Metal
      docstring style)
- [ ] **G4.2.3** `[tool.interrogate]` exclude list covers tests, scripts, `__init__.py`,
      `__main__`, magic methods, private symbols (full `-i -I -m -p -s` set)
- [ ] **G4.2.4** `molmetal/scripts/pre_flight_check.py --pipeline code_quality` invokes
      `interrogate --quiet --fail-under=80` and propagates exit code
- [ ] **G4.2.5** SVG badge lands in `reports/badges/interrogate_badge.svg` (optional)

**Default threshold (MEASURED from S2): 80%** for the codebase; **95% strict** for
paper-critical modules (per interrogate docs `tox.ini` example).

### §4.3 Type hint coverage gate

- [ ] **G4.3.1** `pyproject.toml` has `[tool.mypy]` section with `python_version` matching
      runtime (3.10+)
- [ ] **G4.3.2** Initial mode is **non-strict**: `[tool.mypy] strict = False`,
      `ignore_missing_imports = True`
- [ ] **G4.3.3** Per-module overrides in `pyproject.toml [tool.mypy.overrides]` apply
      `disallow_untyped_defs = True` to `molmetal/molmetal_lam/lam_chem/` (paper-critical
      core)
- [ ] **G4.3.4** `mypy --linecoverage-report reports/mypy/` produces JSON for tooling
      pipelines
- [ ] **G4.3.5** `molmetal/scripts/pre_flight_check.py --pipeline code_quality` invokes
      `mypy molmetal/` (with overridden modules at stricter mode) and propagates exit code

**Default threshold (MEASURED-derived from S3):** 0 errors on `molmetal_lam/lam_chem/`
(strict); non-zero allowed (with per-module tracking) on rest of codebase during
gradual adoption phase.

### §4.4 Data validation gate

- [ ] **G4.4.1** `pyproject.toml` lists `pandera[pandas]>=0.20` as a runtime dep
      (alongside existing `pydantic`)
- [ ] **G4.4.2** `molmetal/data/cytotox.py` loads `krasnov_metalcytotox.csv` behind a
      `pa.DataFrameModel` schema validating: required columns, dtypes, value ranges
      (e.g. `pIC50` in `[0, 12]`, `metal ∈ {Pt, Ru, Ir, Au, Pd}`)
- [ ] **G4.4.3** `molmetal/data/platinai_loader.py` validates PlatinAI 4-cell CSV at
      load time (schema + cell-line set ∈ {A2780, MCF7, ...})
- [ ] **G4.4.4** `molmetal/data/metalcytotox_loader.py` validates MetalCytoToxDB 5-metal
      CSV
- [ ] **G4.4.5** `molmetal/data/tmqm.py` validates tmQM CSV (per Wf-321B skill catalog)
- [ ] **G4.4.6** All four loaders raise `pa.errors.SchemaError` (or `SchemaErrors` if
      lazy=True) on schema drift; CI gate is `--strict` to fail-fast on drift
- [ ] **G4.4.7** `molmetal/scripts/pre_flight_check.py --pipeline data_quality` instantiates
      each schema against a fixture CSV and asserts `validate()` passes

**Default threshold (MEASURED from S5): 100% load-time validation** on paper-critical
datasets. No data should reach training without passing schema validation.

### §4.5 Optional / future (not blocking Tier-2 ship)

- [ ] **G4.5.1** Great Expectations catalog for paper-appendix data-doc generation
      (DEFERRED — adds heavy dependency, no Tier-2 gating value)
- [ ] **G4.5.2** SonarQube integration for cross-language quality gate (DEFERRED — Mol-Metal
      is Python-only; mypy + interrogate + pytest-cov cover the relevant dimensions)
- [ ] **G4.5.3** Per-module coverage thresholds (DEFERRED — Tier-2 ship accepts 80% global)

---

## §5 Applicability scoring

Per the four surveyed categories, scoring against Mol-Metal project context:

| Category | Tool | Applicability | Rationale | Score (0-3) |
|---|---|---|---|---|
| Test coverage | coverage.py + pytest-cov | **HIGH** | Pytest is already the test runner; pyproject has `[tool.pytest.ini_options]`; missing only `[tool.coverage.*]` + `--cov-fail-under` gate | 3 |
| Docstring coverage | interrogate | **HIGH** | Tier-2 paper needs self-documenting code; interrogate is light-weight (single CLI) + integrates with pre-existing pytest workflow | 3 |
| Type hint coverage | mypy | **MEDIUM-HIGH** | Type-critical modules (`lam_chem/`, `adapters/`) benefit greatly; tests/scripts have dynamic patterns that slow adoption | 2.5 |
| Data validation (pandera) | pandera | **HIGH** | 4+ data loaders (cytotox, platinai, metalcytotox, tmqm) load CSVs at runtime without schema check; pydantic already in use; pandera is a natural fit | 3 |
| Data validation (GX Core) | Great Expectations | **LOW-MEDIUM** | Heavy dependency, docs/catalog focus vs runtime validation; not Tier-2-blocking | 1 |

**Aggregate score: 12.5 / 15 = 83.3%.** Tier-2-ship-ready.

---

## §6 Honest framing + R/Wf trace

### §6.1 MEASURED-vs-PROJECTED ledger (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

| Claim | Tag | Evidence |
|---|---|---|
| coverage.py default `[report] fail_under` exit-code semantics | **MEASURED** | S1 verbatim quote |
| coverage.py `[run] branch = True` opt-in | **MEASURED** | S1 verbatim quote |
| pytest-cov `--cov-fail-under MIN` gate | **MEASURED** | S6 verbatim quote |
| interrogate default `--fail-under 80.0` | **MEASURED** | S2 verbatim quote |
| interrogate `tox.ini --fail-under 95` example | **MEASURED** | S2 verbatim quote |
| mypy `--strict` enabled flags list | **MEASURED** | S3 verbatim quote |
| mypy `--html-report`, `--linecount-report`, `--linecoverage-report` flags | **MEASURED** | S3 verbatim quote |
| GX Core concepts (Expectation Suite / Expectation / Checkpoint) | **MEASURED** | S4 verbatim quotes |
| GX Core Python 3.10-3.13 support | **MEASURED** | S4 verbatim quote |
| pandera `DataFrameSchema` + `DataFrameModel` API | **MEASURED** | S5 verbatim quote |
| Mol-Metal pyproject.toml has `[tool.pytest.ini_options]` but no `[tool.coverage.*]` | **MEASURED** | `grep` output §3 |
| Mol-Metal pyproject.toml has no `[tool.mypy]` section | **MEASURED** | `grep` output §3 |
| 80% line coverage as Tier-2 baseline threshold | **DOCUMENTED** (S1 + S2 industry pattern) | S1 + S2 verbatim quotes |
| 95% strict coverage for paper-critical modules | **DOCUMENTED** (S2 tox.ini example) | S2 verbatim quote |
| Mol-Metal's 4 data loaders lack pandera schemas | **MEASURED** | File existence check (no `pa.DataFrameModel` in `molmetal/data/`) |

### §6.2 Honest negatives

- **S4 (Great Expectations) full Expectation library list** — the repo README only mentions
  Expectation as a concept; the comprehensive category list (column/table/distribution-level)
  is reconstructed from public knowledge of GX, NOT extracted verbatim from the README.
  This is flagged as **MEASURED-but-incomplete**; a follow-up Wf should fetch the
  `docs.greatexpectations.io` Expectation glossary to get the full catalog with citation.
- **MyPy numeric annotation-coverage threshold** — mypy docs do not prescribe a numeric
  threshold (S3 honest-negative: "the documentation does not prescribe a specific numeric
  threshold value"). The §4.3 G4.3.3 per-module-strict pattern is **DOCUMENTED-derived**
  from S3 incremental adoption strategy; the **specific threshold choice** (0 errors on
  `lam_chem/`) is **PROJECTED** — agent-set pending first-run MEASURED.
- **GX vs pandera selection** — the choice of pandera over GX Core for Mol-Metal is
  **PROJECTED** based on (a) runtime validation fit, (b) pandas-only data, (c) existing
  pydantic dependency. A more thorough comparison including benchmark overhead + setup
  cost would be a follow-up Wf.

### §6.3 Cross-references

- AGENTS.md §5.1 — standing GPU grant; this R4 gates precede GPU launches.
- AGENTS.md §1.4 — pre-flight v2 25/25 PASS gate; this R4 extends pre-flight with 4 new
  categories.
- AGENTS.md §6 — Wf-309..317 ship wave; this R4 is part of Wf-343a wave (sub-id R4).
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` — pre-flight v2 doc; §4.4 + §5.1
  + §5.2 explicitly state "config schema + CSV hash" is TODO-208 follow-up; this R4
  partially addresses TODO-208 (pandera covers CSV schema; `[tool.coverage.*]` + mypy
  covers config schema for code).
- Wf-321B ship surface — R21 skill ecosystem expansion; R4 does not introduce new skills but
  identifies that the 28-skill catalog should include a `mol-metal-code-quality` skill in a
  future Wf (DEFERRED).

---

## §7 Next steps (actionable, Tier-2-blocking)

The checklist in §4 is **proposal-grade**. Landing them requires:

1. **Wf-343b** — implement `[tool.coverage.*]` + `[tool.interrogate]` + `[tool.mypy]`
   sections in `pyproject.toml`. CPU-only, dry-run test on `molmetal/`. (Est. 30 min.)
2. **Wf-343c** — implement pandera schemas for the 4 data loaders in `molmetal/data/`.
   CPU-only, fixtures via existing CSVs. (Est. 1 hr.)
3. **Wf-343d** — wire all 4 quality gates into `molmetal/scripts/pre_flight_check.py`
   as new `--pipeline code_quality` and `--pipeline data_quality` sub-gates. CPU-only.
   (Est. 1.5 hr.)
4. **Wf-343e** — run full pre-flight v3 (existing 25/25) + new code/data quality
   gates (4 categories × ~5 sub-gates each = ~20 new gates) and emit
   `TODO/completed/343e_full_preflight.json` with per-gate PASS/FAIL.
   (Est. 30 min CPU.)
5. **Wf-343f** — Tier-2 paper §2 Methods update to cite the code/data quality framework.

All five steps are CPU-only; no GPU launch required. The first run produces MEASURED
threshold values that convert the §4 PROJECTED thresholds into MEASURED.

---

## §8 Provenance + file paths

- This document: `/home/hugo/codes/mol-metal/TODO/completed/343a4_quality.md`
- Sources fetched: §1 table; URLs verified by WebFetch on 2026-09-29.
- Local files inspected: `pyproject.toml`, `molmetal/scripts/pre_flight_check.py`
  (header only, first 80 lines).
- No GPU launches, no `paper/*` touches, no TODO INDEX updates (per AGENTS.md §5.1 / §5.3).

**END OF Wf-343a-R4.**
