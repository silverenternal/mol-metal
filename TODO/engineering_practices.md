# Engineering Practices

Canonical doc for the engineering practices followed by this project. Use this
as the contract every PR, report, and adapter must satisfy. Anything that
contradicts this file is a bug in the new code, not a bug here.

Related: `TODO/README.md`, `TODO/AUDIT_RESEARCH_GRADE.md`,
`TODO/07_risks/open_questions.md`, `TODO/13_lambda_clickchem/plan.md`.

---

## 1. Architecture: hexagonal ports & adapters

The molmetal package is organised as a hexagonal (ports & adapters)
architecture. Code outside the core domain depends only on **ports**, never on
adapter classes.

- **Ports live in `molmetal/ports/`** (and its sub-package `ports/`):
  - `molmetal/ports/__init__.py` — `MoleculeGenerator`, `DockingEngine`,
    `PropertyPredictor`, `ScoringFunction`, `DesignLoop` (plus the
    `GenerationConfig`, `DockingConfig` frozen dataclasses).
  - `molmetal/ports/generators.py` — `MetalLigandGenerator`,
    `MetalLigandConfig` (metal-complex specialisation).
  - Additional ports (per inventory): `Predictor`, `Validator`,
    `RetrosynthesisChecker`. Each port file defines `Protocol` classes and
    config dataclasses; nothing more.
- **Adapters live in `molmetal/adapters/`**:
  - `flow_matching_lipman/` — Lipman 2023 flow-matching generator.
  - `egnn_predictor.py`, `egnn_rocm.py` — EGNN-based property predictors.
  - `rdkit_predictor.py` — RDKit-based predictor (QED, SA, etc.).
  - `diffdock.py`, `equibind.py` — docking engines (skeleton).
  - `mock.py` — deterministic stand-ins used in tests.
- **Domain types in `molmetal/domain/`** (`Complex`, `Molecule`, `Pocket`).
  Adapters convert from domain types to library types and back.

### Rules

1. Every port is a `typing.Protocol` (PEP 544). Adapters are **not** required
   to inherit; duck-typing via `@runtime_checkable` is encouraged so
   `isinstance(obj, MoleculeGenerator)` works in tests.
2. Every config object is a `frozen=True` dataclass. Adapter methods receive
   the config object and never mutate it.
3. The orchestrator (`molmetal/orchestration/`) depends only on ports. It
   never imports a concrete adapter.
4. External plug-in libraries (vina + meeko, posebusters, aizynthfinder,
   RDKit Contrib SA_Score, RDKit MMFF94, Ertl) are wired in via thin
   adapters — never imported directly from the domain layer.
5. New adapter? Add a file under `molmetal/adapters/`, implement the relevant
   port, and register it in `molmetal/adapters/__init__.py` only if
   something iterates adapters dynamically.

---

## 2. Testing discipline

Layout lives in `molmetal/tests/`. Currently 40+ test modules. Rules:

1. **One test module per port + per adapter.** Naming: `test_<port>.py`
   exercises the Protocol contract using the mock adapter;
   `test_<adapter>.py` exercises the concrete adapter.
2. **Integration tests** (`test_e2e.py`, `test_design_loop.py`,
   `test_clone_integration_adapters.py`) drive the full
   generate → dock → score → refine loop on a fixed pocket.
3. **Smoke tests** for every Triton kernel: a tiny synthetic input compared
   to a PyTorch reference within numerical tolerance (default `atol=1e-5`).
4. **Fixtures in `conftest.py`** (project root and `molmetal/tests/`):
   deterministic seeds, a 3-pocket fixture loaded from
   `molmetal/data/test_pockets/`, and shared molecule fixtures. Do not
   redefine fixtures inside individual test modules.
5. **Negative results are tested too.** When a known bug is documented
   (e.g. `r1_cross_attn_v3_fix.md`), add a regression test that fails on
   the buggy path and passes on the fix.
6. **CI gate**: `uv run pytest -q` must pass before any PR merges. Use
   `uv sync --frozen` to install before running.

---

## 3. Validation pipeline

A generated molecule set is **only "complete"** when the following checks
have all run, in this order, with thresholds documented in the report.

| Step | Tool | Threshold (default) | Purpose |
|---|---|---|---|
| 1 | **Ertl SA** (RDKit Contrib `SA_Score`) | `SA < 5` for ≥80% of mols | Synthetic accessibility |
| 2 | **Lipinski** (RDKit) | ≤1 violation | Oral bioavailability |
| 3 | **Veber / Egan / REOS** (custom in `molmetal/adapters/rdkit_predictor.py` + `molmetal/data/`) | pass = 0 hard violations | Rotatable bonds, TPSA, reactive groups |
| 4 | **QED** (RDKit, Bickerton) | `QED > 0.5` for ≥70% | Drug-likeness |
| 5 | **PoseBusters** (PB-Valid) | ≥80% pass the 20 checks | Chemical/geometric validity |
| 6 | **AiZynthFinder 4.0** | ≥70% routes solvable on stock | Retrosynthesis feasibility |
| 7 | **AutoDock Vina** (vina + meeko) | mean Vina Dock < reference | Binding affinity |

Order matters: cheap filters first (SA, Lipinski), expensive ones last
(PoseBusters, Vina). The `generate_validation` pipeline lives in
`molmetal/scripts/` and emits `molmetal/reports/generate_validation.md`.

---

## 4. Split discipline

**Never random-split for paper claims.** The reference is
`molmetal/data/splits.py` (which defines `LigandDeduplicatedSplitter` at
line 385). We always report **four** splits:

1. **Random** — only for sanity checks / ablations, never for headline
   numbers.
2. **Scaffold** — Bemis–Murcko scaffold split (RDKit `MurckoScaffold`).
3. **Temporal** — by date of PDB deposition (e.g. CrossDocked pre-2020
   train, PoseBusters post-2021 test).
4. **Ligand-dedup** — `LigandDeduplicatedSplitter` is the reference;
   drops near-duplicate ligands (Morgan FP, Tanimoto ≥ 0.9) from the
   train side to prevent memorisation.

Plus a 30%-sequence-identity filter (MMseqs2) on the pocket side is
mandatory before any pocket-conditioned split.

---

## 5. Baseline comparison methodology

For every reported number, distinguish:

- **Published-number rows**: cite the paper (Pocket2Mol = −7.07,
  TargetDiff = −8.45, DiffSBDD = −7.62). These are *not* measured by us
  head-to-head and live in `molmetal/reports/lambda_vs_sbdd_paper_numbers.md`.
- **Measured-head-to-head rows**: re-run the official checkpoint on a
  shared pocket set, `n ≥ 100` generated molecules per pocket, identical
  Vina setup. These are the rows that appear in our headline tables.

When both rows exist, the measured row wins for our claims; the published
row anchors context. The DecompDiff paper is the most-cited SBDD
comparison sheet and is the de-facto reference table (used by BoKDiff,
READ, VoxBind). If we cite one baseline number, we cite all of them.

Live comparison artifacts: `molmetal/reports/honest_baseline_summary.md`,
`molmetal/reports/lambda_vs_sbdd_baselines.md`,
`molmetal/reports/lambda_vs_sbdd_protocol_aligned.md`.

---

## 6. Paper-grade reporting

Every report file in `molmetal/reports/` (and every TODO report) follows
this template:

```
# <Title>
## Goal     — one paragraph, what we set out to show
## Outcome  — pass / fail / mixed, with hard numbers
## Caveats  — what could be wrong, what was not measured
## Reproduce — exact command(s) and config hash
```

Additional rules:

- **Ablation tables** are mandatory for any non-trivial change. Format:
  rows = ablated component, columns = metric, footer = base config hash.
- **Seed management**: every training script writes a seed JSON
  (`*_seed.json` or `*_seeds.json`). Tables report mean ± std over ≥3
  seeds unless the report says otherwise.
- **Statistical significance**: use paired bootstrap (≥1000 resamples) when
  comparing two rows; report 95 % CI.
- **Negative results are documented in the same table as positives** (see
  section 9). Hiding them in an appendix is forbidden.

---

## 7. Triton kernel patterns

Triton kernels live in `triton_kernels/` (`equivariant_ops.py`,
`fused_norm.py`, `ode_solver.py`). Rules:

1. **Autotune configs are per-kernel.** A canonical RDNA grid:

   ```python
   configs = [
       triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},
                     num_warps=4, num_stages=2),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},
                     num_warps=8, num_stages=3),
       triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},
                     num_warps=8, num_stages=2),
   ]
   ```

2. **`waves_per_eu` is CDNA-only** (MI300/MI355). For gfx1101 it is
   ignored. The only knobs are `num_warps ∈ {4, 8, 16}` and `num_stages`.
3. **Wave size = 64** for gfx1101 (RDNA3, RX 7800 XT).
4. **Scatter/gather via `tl.atomic_add`**. Reference pattern: gather
   edge indices with `tl.load(idx_ptr + offsets)`, then in-kernel
   `tl.atomic_add(out_ptr + dst_offsets, msg)`. See
   `triton_kernels/equivariant_ops.py`.
5. **Correctness vs reference torch**: every kernel has a
   `test_<kernel>.py` in `molmetal/tests/` that compares against an
   `torch.scatter_add_` / `torch.gather` reference. Default tolerance
   `atol=1e-5`, `rtol=1e-5`. Float16 kernels compare in FP32.
6. **Report achieved FLOPs vs peak** in the kernel docstring and in any
   performance report. gfx1101 peak FP32 ≈ 35 TFLOPS, FP16 ≈ 70 TFLOPS.
   Use `rocprof --stats` or `torch.profiler`.
7. **Async transfers**: `tensor.pin_memory()` + `.to(device,
   non_blocking=True)` to overlap H2D/D2H with kernel launches.
8. **Hand-written Triton for**: EGNN edge aggregation, radius-neighbour
   gather, scatter-add reductions, FP16→FP32 accumulators.
   `torch.compile(mode="max-autotune")` is for outer training steps only.

---

## 8. uv + project hygiene

Dependency manager is **uv**. Never `pip install`. The quartet
`uv add` / `uv remove` / `uv lock` / `uv sync --frozen` keeps
`pyproject.toml` and `uv.lock` atomic.

1. **`.python-version` is pinned** (currently `3.12`). Do not edit by
   hand outside `uv python pin`.
2. **`requires-python = ">=3.12,<3.13"`** in `pyproject.toml`.
3. **Dev/test deps live in `[dependency-groups]`** (PEP 735), not in
   `[project.optional-dependencies]`.
4. **Workspace mode** for multi-package repos: `libs/*` + `apps/*` are
   workspace members; one `uv.lock` for the whole tree.
5. **Non-PyPI wheels** (ROCm PyTorch, flash-attn) go through
   `[[tool.uv.index]]` entries. Pin via `--index-strategy unsafe-best-match`.
6. **`uv run <cmd>`** for everything; never `python -m` against the
   system interpreter.
7. **CI uses `uv sync --frozen`**; pre-commit uses `uv lock --check` to
   fail-fast on drift.
8. **Never mix pip-installed packages with `uv sync`** — they get blown
   away on the next sync and silently corrupt reproducibility.

---

## 9. Negative results are first-class

Every experiment that does not beat the baseline, or that regresses a
prior metric, gets its own report under `molmetal/reports/`. Examples:

- `r1_cross_attn_v3_fix.md` — cross-attention v3 *worsened* pIC50; fix
  and the negative result are both documented.
- `paper_negative_results_section.md` — central index of all
  negative / null findings.
- PoseBusters `0/13` CuAAC products — recorded in
  `molmetal/reports/h1_sa_score_ertl.md` and the retrosynthesis report.
- V4 stability σ regression — `molmetal/reports/metal_hybrid_v4_stability_v2.md`.

Rules:

1. A negative result is **never deleted**. If the fix later succeeds, the
   negative report stays as a historical record.
2. Negative reports include: what we tried, what we expected, what we
   saw, and the closest positive baseline we beat (if any).
3. Negative results are listed in the index of the report they belong to
   (e.g. `TODO/13_lambda_clickchem/plan.md`).

---

## 10. Reproducibility checklist

Every reported number must be reproducible from a clean checkout:

- [ ] `uv.lock` is committed and matches the report's command.
- [ ] `.python-version` is committed.
- [ ] `.venv/` and `.cache/` are in `.gitignore`.
- [ ] All seeds are logged in `*_seed.json` next to the report.
- [ ] Data hashes (`md5` / `sha256`) are recorded next to any dataset
      version (CrossDocked2020, PoseBusters, AiZynth stock).
- [ ] Checkpoint state-dict is saved (`*.pt` or `*.safetensors`) and
      referenced by the report.
- [ ] The report includes the exact `uv run` command(s) and any
      environment variables (e.g. `CUDA_VISIBLE_DEVICES`,
      `HIP_VISIBLE_DEVICES`).
- [ ] For Triton kernels: the autotune `Config` chosen is logged in
      `*_autotune.json` so the choice is reproducible.

---

## Cross-references

- `TODO/environment.md` — runtime / driver versions.
- `TODO/05_compute/` — GPU compute budget.
- `TODO/06_milestones/` — what is planned and what shipped.
- `TODO/09_interfaces/abstract_interfaces.md` — companion to section 1.
- `molmetal/reports/rocm_throughput_measured.md` — kernel benchmark log.
- `molmetal/reports/paper_outline.md` — paper structure (sections 6, 9).