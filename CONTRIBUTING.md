# Contributing to Mol-Metal

Thank you for your interest in contributing to Mol-Metal — a hybrid molecular
λ-calculus + flow-matching framework for precious-metal drug design. This
document is the single entry point for new contributors: how to set up a
development environment, how to pick a task, the coding conventions we
follow, and how to submit a pull request.

If you only have five minutes, read the [Quick start](#1-quick-start) section.
If you plan to send a PR, please read the entire document — sections
[3 Coding conventions](#3-coding-conventions), [5 Testing](#5-testing), and
[6 Submitting PRs](#6-submitting-prs) are mandatory.

---

## 1. Quick start

The repository ships a single-file developer entry point (`Makefile`) plus a
canonical-environment bootstrap script (`env.sh`) that pins all contributors
to the same Python interpreter — the repo-local
`/home/hugo/codes/mol-metal/.venv` (AGENTS.md §5.9 single-venv rule: torch
2.14.1 + cu130, rdkit, posebusters, meeko, vina, etc.).

```bash
# 1. Clone
git clone https://github.com/Hugo-Lee/mol-metal.git
cd mol-metal

# 2. Bootstrap the canonical venv (idempotent — wires UV_PROJECT_ENVIRONMENT)
source env.sh

# 3. Sanity-check the environment
make setup            # verifies the canonical .venv interpreter is reachable

# 4. Run the full evaluation harness (all 14 evaluators on Blackwell)
make eval

# 5. Run the fast unit-test suite (pytest molmetal/tests/ -q)
make test
```

If `make eval` and `make test` both pass on your machine, your environment is
correct and you can start contributing.

Optional auxiliary targets:

| Target        | What it does                                                           |
| ------------- | ---------------------------------------------------------------------- |
| `make pitfall`| Runs `wf_pitfall_guard/check_all.py` — CI gates that block bad merges  |
| `make smoke`  | 200-step CFM smoke retrain (metallo_drug_smoke_retrain, hidden_dim=64) |
| `make reproduce` | Full R17 reproduction script (`wf_reproduce_R17.sh`)              |
| `make coverage` | `pytest --cov=molmetal --cov-report=html`                            |
| `make lint`   | Optional flake8 on `molmetal/` (skipped silently if flake8 missing)    |
| `make paper`  | 4-pass `pdflatex` + `bibtex` rebuild of `paper/main.pdf`               |
| `make clean`  | Removes `logs/`, `htmlcov/`, `.coverage*` (keeps `.venv` + reports)    |

For the full target list, run `make help`.

---

## 2. Development workflow

We follow a five-step workflow that mirrors the project's own
`TODO/EXECUTION_ORDER.md` and `TODO/project_workflow.md` documents:

1. **Pick a task.** Open `TODO/EXECUTION_ORDER.md` and choose the
   highest-priority open task under "Phase A — Tier-1 AUC fix", "Phase B —
   R18 mathematical upgrade", or whichever phase is currently active. Each
   task has a stable identifier (`TODO-XX` or `TODO/completed/XX_*.md`).
   Read the linked plan file (`TODO/03_plans/...`) for the *why* and the
   *how*.

2. **Read the docs.** Before editing any source file, read the relevant
   `SKILL.md` (lambda-calculus / cfm-training / click-chemistry /
   anticancer / mechanism-mapping) under `.claude/skills/`, plus the
   mechanism × metric mapping
   (`TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`). The skills encode the
   domain knowledge — the lambda-calculus skill defines the typed β-NF
   rules, the click-chemistry skill defines the five CuAAC/SPAAC/ThiolEne/
   Suzuki/AmideCoupling reductions, and so on.

3. **Make changes.** Edit code under `molmetal/` following the conventions
   in [§3](#3-coding-conventions). Every new module ships with a docstring
   header that names the mechanism it implements and the metric it gates.
   Keep diffs focused — one task per PR.

4. **Update docs.** If you added a new mechanism, update
   `MECHANISM_METRIC_MAPPING.md`. If you changed a CLI flag, update the
   relevant `SKILL.md`. If your change is user-visible, add a one-line
   bullet under the next-version header of `CHANGELOG.md`.

5. **Open a PR.** Follow [§6 Submitting PRs](#6-submitting-prs). Reference
   the `TODO-XX` task ID in the PR title and description; CI must be green
   before a reviewer will look at it.

---

## 3. Coding conventions

The conventions below are enforced by `pre-commit` (`.pre-commit-config.yaml`)
where possible and by code review where not.

- **Python 3.12.** `pyproject.toml` pins `requires-python = "==3.12.*"`. Use
  3.12-native syntax freely: PEP 695 type parameters, `match` statements,
  `type X = ...` aliases, structured `except*`, etc.
- **Type hints are mandatory on every public function and method.** Internal
  helpers should also be typed. Use `from __future__ import annotations`
  only when you need forward references under `if TYPE_CHECKING:`.
- **Line length ≤ 120 characters.** Enforced by `pre-commit`/`flake8`
  (`--max-line-length=120`, `--ignore=E203,W503,E402`). Long strings may be
  wrapped with implicit line continuation; do not break after a binary
  operator.
- **Absolute imports only.** No relative imports (`from .foo import bar` is
  forbidden inside `molmetal/`). This makes `grep`-based refactors
  tractable and keeps the import graph explicit. The single exception is
  `molmetal/__init__.py`, which may use relative imports to break circular
  module-init order.
- **Test file mirroring.** For every production file
  `molmetal/<pkg>/foo/bar.py` there should be a corresponding test file
  `molmetal/tests/foo/test_bar.py` (or, for cross-package units,
  `molmetal/tests/test_bar.py`). Unit tests live in `molmetal/tests/`;
  integration tests live under `molmetal/tests/integration/`.
- **No `print()` in production code.** Use the `logging` module — `logger =
  logging.getLogger(__name__)`. `print()` is permitted in CLI scripts under
  `molmetal/scripts/` and in notebooks.
- **Domain modules document their mechanism.** Every file under
  `molmetal/`, `molmetal/molmetal_lam/`, `molmetal/flow_matching/`, and
  `molmetal/triton_kernels/` must open with a one-line comment naming the
  mechanism (`# Mechanism: lambda-calculus / CuAAC / SE(3)-equivariant
  flow / ...`) so the mechanism × metric mapping stays greppable.
- **Configuration via `molmetal/configs/`.** Hard-coded magic numbers in
  scripts are bugs. Add a YAML entry under `molmetal/configs/` and load it
  via `utils.config.load_config(...)`.

---

## 4. Documentation

Mol-Metal treats documentation as a first-class artifact: every code change
that affects behavior must come with a doc update in the same PR.

- **Skills.** The five domain skills under `.claude/skills/` (`lambda-
  calculus`, `cfm-training`, `click-chemistry`, `anticancer`,
  `mechanism-mapping`) are the authoritative reference for how each
  mechanism works. If you change a typed reduction rule, a click rule, or
  a flow-matching step, update the matching `SKILL.md`. Skills live next
  to the source so reviewers can diff them in the same PR.
- **Mechanism × Metric mapping.** `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  is the cross-reference table that tells reviewers which metric a
  mechanism gates. Adding a new mechanism (e.g. a sixth click reaction)
  requires a new row; fixing a metric requires updating the existing row's
  *Status* column.
- **Changelog.** Add a bullet under the next unreleased version header of
  `molmetal/CHANGELOG.md` (create the header if missing — semver
  `Unreleased` is fine). One bullet per user-visible change. Format:
  `- <description> (<PR number or TODO-XX>)`.
- **Inline docstrings.** Google-style docstrings on every public class and
  function. The first line is an imperative-mood summary; subsequent lines
  describe arguments, return value, side effects, and metric gates.

---

## 5. Testing

Every PR must include tests. The bar depends on the kind of change:

- **New feature** → at least **one unit test** (covers the happy path and
  the most obvious failure mode) **plus one integration test** (runs the
  feature end-to-end against a small fixture, e.g. a 5-molecule tmQM
  subset).
- **Bug fix** → at least **one regression test** that fails on `master`
  before the fix and passes after it. The regression test must target the
  exact failure mode; do not rely on existing tests to cover it.
- **Refactor with no behavior change** → the existing test suite must pass
  unchanged. If it does not, the refactor is not behavior-preserving and
  must be split into a behavior-change PR and a pure-refactor PR.

Conventions:

- Unit tests use `pytest`. Mark slow tests with `@pytest.mark.slow`; the
  default `make test` target skips them.
- Integration tests live under `molmetal/tests/integration/` and may
  download fixtures; they are gated by the `integration` marker.
- Property-based tests (using `hypothesis`) are encouraged for any
  module that ingests user-controlled data (SMILES, YAML, JSON configs).
- Test names follow `test_<unit>_<scenario>_<expected>()` (e.g.
  `test_lambda_terminal_rejects_unbound_metal_returns_empty`).

Before opening a PR, run **all three** of:

```bash
make test           # fast unit tests
make pitfall        # wf_pitfall_guard CI gates
make eval           # full evaluator harness (slow — only if you changed an evaluator)
```

---

## 6. Submitting PRs

### Branch naming

`<type>/<scope>-<short-description>` where `<type>` is one of
`feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`. Examples:

- `feat/lambda-b3-deflexpressor-coupling`
- `fix/r17-blackwell-loader-paths`
- `docs/contributing-and-changelog`

### Commit format

We use [Conventional Commits](https://www.conventionalcommits.org/). The
commit subject is `<type>(<scope>): <imperative summary>`; the body
explains *why* and references the TODO task ID. Example:

```
feat(lambda): couple B3 deflexpressor into the lambda terminal

Adds the SE(3)-invariant deflexion tensor to the typed β-NF reduction
so the MCTS prior can veto terminal terms with unbound metals.

Refs TODO-35 (lambda B2/B3 deflexpressor)
Refs TODO-36 (lambda B1 coupling deflex)
```

### PR description

Every PR must include:

1. **TODO reference** — the `TODO-XX` task ID in the title (e.g.
   `[TODO-35] couple B3 deflexpressor`). The body must quote the relevant
   task file path (`TODO/pending/35_*.md` or
   `TODO/completed/35_*.md`).
2. **What & why** — one paragraph summarizing the change and the metric it
   improves (or the bug it fixes).
3. **Test plan** — the commands you ran locally and their results. CI will
   re-run them.
4. **Doc updates** — the list of `SKILL.md` / `MECHANISM_METRIC_MAPPING.md`
   / `CHANGELOG.md` files you touched.

### CI requirements

A PR is mergeable only when **all** of the following are green:

- `make test` — fast unit tests
- `make pitfall` — `wf_pitfall_guard` CI gates
- `pre-commit run --all-files` — trailing whitespace, EOF newline,
  YAML/JSON validation, flake8 (max-line-length=120)
- `make eval` — only required if the PR changed an evaluator under
  `molmetal/eval/`, a metric under `molmetal/metrics/`, or an adapter
  under `molmetal/adapters/`

A PR that touches `paper/main.tex` or `paper/refs.bib` additionally needs
`make paper` to succeed locally (CI does not run LaTeX builds to keep the
runner fast — reviewers may ask you to paste the last 30 lines of
`/tmp/pdflatex_pass4.log`).

### Review SLA

Maintainers aim for a first response within **two business days**. Be
patient; we are a small team and the metric landscape is dense. If your PR
stalls, ping the reviewer on the TODO thread and reference the task ID.

---

## 7. Code of conduct & questions

By participating in this project you agree to keep discussions technical,
respectful, and on-topic. For questions that do not fit the bug-report or
feature-request shape, open a GitHub Discussion referencing the relevant
`TODO-XX` task — do not file an issue.

Welcome aboard — the precious-metal chemistry is hard, the rewards are
real, and every contribution is one row closer to a reproducible precious-
metal AUC we can defend in review.

---

## 8. Reviewer guide for first-time PR reviewers

First-time reviewers in this repo should walk through this checklist
before approving a PR. Most checks take five minutes and prevent a
re-review cycle.

1. **Read the linked task first.** The PR description must reference a
   `TODO-XX` task ID. Open the linked plan under `TODO/03_plans/` (or
   `TODO/pending/`) and confirm the diff only touches the scope of that
   task. Anything outside the plan is a follow-up PR.
2. **Check tests.** Every PR must include at least one new or modified
   test (see [§5 Testing](#5-testing)). Confirm the test runs locally
   via `make test` and that it would have failed on `master` before the
   fix landed for bug-fix PRs.
3. **Confirm docs were updated.** New mechanism → row in
   `MECHANISM_METRIC_MAPPING.md`. New CLI flag → matching `SKILL.md`.
   User-visible change → bullet in `CHANGELOG.md`. A PR that updates
   no docs is rarely mergeable.
4. **Sanity-check the imports and types.** Run `python -c "import <pkg>"`
   on every module touched. Confirm `from __future__ import annotations`
   is used only where forward references require it. Public functions
   must have type hints on arguments and the return value.
5. **Skim the pitfalls.** Confirm `make pitfall` is green — all seven
   R12-R17 guards must stay green. A single regression is enough to
   block.
6. **Look for accidental scope creep.** PR title and body should match
   the diff. Bundled reformatting, unrelated refactors, and "while I'm
   here" cleanups belong in separate PRs.

Reviewers are encouraged to leave inline comments, but please mark
blocking issues explicitly (`nit:` / `blocking:` / `question:`) so the
author knows what to address before re-review.

---

## 9. Ablation experiment template

Use this template for any ablation reported in the paper, in
`TODO/pending/*_ablation.md`, or in a PR description. Fill in all five
fields; missing fields turn the ablation into a vibes-based claim.

```yaml
ablation_id: abl_<short_slug>         # e.g. abl_no_pll_heuristic
hypothesis:                           # 1-2 sentences, falsifiable
  Removing <mechanism> should <direction> <metric> by ≥ <delta>
  because <mechanism-based reasoning>.
variable:                             # the single knob you turn
  <one config flag or one feature, e.g. --use-pll-heuristic>
control:                              # baseline configuration
  <commit SHA or config SHA, flag set to default / on>
measurement:                          # which metric(s) and seeds
  - metric: <mID from METRICS_TARGETS>
    seeds: [0, 1, 2]
expected_delta:                       # pre-registered expectation
  direction: <up | down | none>
  magnitude: <numeric, with units>
  confidence: <95% CI minimum>
```

The pre-registered `expected_delta` is what makes an ablation a
falsifiable experiment rather than post-hoc storytelling. If the
measured delta lands in the opposite direction, file the result
honestly and explain the gap.

---

## 10. Adding a new metric to METRICS_TARGETS.md

When you add a new metric (e.g. a Tier-3 DrugOOD AUC, a brand-new
evaluator output), update `TODO/02_metrics/METRICS_TARGETS.md` using
this five-step checklist:

1. **Anchor paper.** Pick a single anchor paper that other readers can
   compare against. Prefer Tier-1 same-domain (precious-metal
   anticancer) over Tier-2 SBDD anchors.
2. **Set direction and verdict gates.** Record `higher` or `lower`,
   then set four verdict bands: regression / match / beat /
   significant-beat. Each band needs a numeric threshold.
3. **Pick a reproducible test protocol.** Provide a single bash
   command the reader can paste to reproduce your number, with the same
   seed set you used to set the targets.
4. **Add the metric ID to `MECHANISM_METRIC_MAPPING.md`.** A new metric
   with no mechanism mapping is ungoverned; every metric must list
   which mechanisms can move it.
5. **Cite it in `paper/refs.bib`.** If the anchor paper is not yet in
   the bibliography, add it under the next free handle.

A PR that adds an evaluator without step 1-5 above will be sent back
for "metric governance" before review.

---

## 11. Adding a new mechanism to MECHANISM_METRIC_MAPPING.md

Adding a mechanism (a new reduction rule, a new reward head, a new
vocabulary, etc.) is a six-step governance flow:

1. **Allocate a mechanism ID.** Read
   `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §1 to find the next
   free `Mxx`. Do not reuse a deprecated ID.
2. **Write the SKILL.md update.** If the mechanism is new domain
   knowledge (lambda rule, click reaction, EGNN block), update the
   matching skill under `.claude/skills/`. Cite at least one primary
   source in `paper/refs.bib`.
3. **Land the source change.** Implement in `molmetal/`, add a
   `# Mechanism:` header in the file, and add at least one unit test
   under `molmetal/tests/`.
4. **Map mechanisms to metrics.** In
   `MECHANISM_METRIC_MAPPING.md` §3, mark the cells (`✓` / `○` / `✗`)
   for every existing metric that the new mechanism can move. If it
   cannot move any existing metric, justify that or add a new metric.
5. **Register the regression guard.** If the mechanism is known-fragile,
   add a test to `molmetal/tests/test_pitfall_guards.py` so
   `make pitfall` keeps it honest. Pitfalls live by R-number
   (`R12-Pitfall` etc.).
6. **Changelog + paper update.** Add a bullet under the next-version
   header of `CHANGELOG.md` and (if the paper is in scope) a one-line
   note in the most relevant `paper/sections/*.tex`.

A PR that lands mechanism #N without the §3 mapping row will fail
review; reviewers will request step 4 explicitly.

---

## 12. Code review checklist

Every PR should pass these ten checks before the reviewer approves.
Treat them as a quick self-review — most are stylometric, not design.

- [ ] **Tests** — new code has tests, modified code has updated tests
- [ ] **Docs** — matching `SKILL.md` / `MECHANISM_METRIC_MAPPING.md` /
      `CHANGELOG.md` bullet exists for user-visible changes
- [ ] **Type hints** — every public function/method annotated; internal
      helpers annotated unless trivially obvious
- [ ] **Imports** — absolute only (no `from .foo import bar` outside
      `molmetal/__init__.py`)
- [ ] **No `print()`** in production code (CLI scripts and notebooks are
      permitted)
- [ ] **Line length ≤ 120** — enforced by flake8 via pre-commit
- [ ] **Mechanism header** — domain files open with `# Mechanism: ...`
      on the first non-shebang line
- [ ] **Magic numbers moved to config** — hard-coded literals in
      scripts must live under `molmetal/configs/`
- [ ] **No new TODOs in code** — open a `TODO-XX` task instead of
      leaving a dangling comment in production code
- [ ] **Pitfalls still green** — `make pitfall` reports 7/7 PASS

A reviewer may skip items for trivial PRs (typo fixes, doc-only edits);
otherwise please address them before requesting review.

---

## 13. Common pitfalls (R12-R17)

The seven canonical pitfalls in `wf_pitfall_guard/check_all.py` recur
often in PRs. Citing each by ID so you can grep them later:

- **P1 — Out-of-vocab atoms.** Always apply `vocab_mask` and default to
  the 20-element vocab (H, B, C, N, O, F, Si, P, S, Cl, Sc, Se, Br, Y,
  Zr, I, Hf, Pt, …). Symptom: random SMILES, `IndexError` in atom
  lookup.
- **P2 — SynFlowNet shim `TypeError`.** Wrap per-rule reductions in
  `try/except` so a single failed rule does not kill the whole MCTS
  batch. Symptom: batch stops after the first bad rule.
- **P3 — `MetalLigandExchange` 84-rule matrix + cisplatin seed.** The
  default `--metal-seed` is now `cisplatin`; do not silently fall back
  to random seeds — that breaks the 30/30 probes = 1.000 invariant.
- **P4 — BF16 atom logits clamp ±50 + FP32 AMP fallback.** Skip the
  clamp and you get overflow into NaN mid-step; keep the AMP fallback
  so bf16 failures degrade gracefully to fp32.
- **P5 — `r4_lambda_only_run` `SAFETY_MAX` ≥ 10 000 floor.** The floor
  is required so a runaway `n_simulations` cannot OOM the host. Do not
  lower it without R-number sign-off.
- **P6 — Reference clone location auto-detect.** The reference adapter
  may live at the parent or inside the clone root depending on the
  upstream revision. Detect both at import time (`_reference` vs
  `_reference_loader`); a hard-coded path fails on the alternate
  layout.
- **P7 — `ModelWrapper` keyword-only `t` fallback.** Some wrappers are
  called as `model(x, t)`, others as `model(x, t=t)`. Use
  `functools.partial` (or accept both) so both call patterns work.

New pitfalls discovered post-R17 should be added both here and in
`molmetal/tests/test_pitfall_guards.py`; the R18+ entries will keep
growing.