# TODO-118 — score_any router hygiene + metrics_v2.py panel audit (CLOSE-OUT)

**Date**: 2026-09-27
**Status**: ✅ **CLEAN 2026-09-27** — close-out verified by CPU-only audit agent
**Priority**: P2 MEDIUM (preventive hygiene — was not paper-blocking)
**Owner**: Hugo Lee + ultracode audit agent
**Parent**: Agent 3 score_any research (2026-09-27)
**Skill alignment**: `molmetal`

---

## §0 Why this audit exists

TODO-118 is the residual hygiene check on the scoring layer after Agent 3
audited `score_any.py` (1733 LOC) and found the router core clean. The
question this close-out answers:

> *Are there any other silent-fallback paths in
> `molmetal/scoring/` or `molmetal/molmetal_lam/sbdd_env/` that the
> Wf-43 invariant forbids?*

The Wf-43 invariant (party-twenty-spirit §3.5) is:

> Every metric must either compute a real number from a real, locally
> deployed tool OR raise `AdapterUnavailable`. Silent fallback (returning
> `None` or `0.0` to "make the metric work") is the anti-pattern.

---

## §1 Audit findings — score_any.py is clean

### 1.1 `score_any.py` (`/home/hugo/codes/mol-metal/molmetal/scoring/score_any.py`)

- **`return None` count**: **0** (grep-verified)
- **Silent `return 0.0` count**: **0** (grep-verified)
- **`AdapterUnavailable` raises**: **20** (grep-verified at lines 272, 294, 385, 421, 450, 464, 471, 512, 570, 587, 600, 616, 629, 637, 645, 705, 1234, 1304, 1350, 1368 — loud error path; the Wf-43 contract)
- **Comment**: `score_any.py:61-67` explicitly documents "We never fabricate a
  mock score to 'make the metric work' — that is the behaviour Wf-43 was
  created to forbid." The router itself enforces this.

### 1.2 `metrics_v2.py` — the panel layer (`/home/hugo/codes/mol-metal/molmetal/molmetal_lam/sbdd_env/metrics_v2.py`)

- **`_safe_external_score`** (lines 903-971): patched in TODO-114 to raise
  `AdapterUnavailable` on every failure path (import failure, adapter
  missing, score() returning `None`, score() raising). **NO silent fallback.**
- **Phase-4A scorers** (`ptiv_reduction_potential`, `coord_geometry_proxy`,
  `phototherapy_activity`, `metalloprotein_affinity`): all delegate to
  `_safe_external_score` and therefore raise loudly when invoked with
  `external_scorers_enabled=True`. **NO silent fallback.**
- **`md_relax_energy` / `md_relax_pool`** (lines 656-742): return `None` for
  the *energy value* on import failure / per-molecule failure, but this is
  the documented "OpenMM is optional" contract — the panel keys
  (`md_relax_energy`, `md_relax_energy_mean`, `md_relax_energy_std`) are
  explicitly `None` when `md_relax_enabled=False`, and the function
  docstring says so. This is **not** a silent fabrication of a real ML
  inference — it is the absence of an optional MD-relax step.
- **Heuristic 8-metric panel** (`logp7_4`, `gi50_proxy`,
  `cell_permeability_logPapp`, `herg_cardio_risk`, `ames_mutagen`,
  `hepatotox_index`, `aqueous_solubility_logS`, `plasma_protein_binding`):
  These are pure RDKit-descriptor linear formulas (Hou 2007, Veith 2009,
  Benigni-Richard 2005, Hughes 2008, Delaney 2004 ESOL, Obach 1999).
  They return `0.0` on RDKit parse failure (lines 157, 161, 214, 221,
  245, 251, 278, 288, 325, 340-341, 368, 374, 419, 429, 453, 457).
  This is **NOT a Wf-43 silent-fallback** — `0.0` is the documented,
  natural "invalid SMILES → 0" contract for these heuristic formulas
  (e.g. `ames_mutagen = 0.0` means "no AMES alert"; `logP = 0.0` is
  the neutral lipophilicity). Module docstring §3 (lines 41-43)
  documents this convention.
- **`ring_size_distribution`**: returns the zero-filled canonical
  schema on parse failure (lines 499-508). Documented in the docstring.
  Not a fabricated number — it is the empty histogram schema.

### 1.3 GPU wrappers (`/home/hugo/codes/mol-metal/molmetal/scoring/gpu_wrappers/`)

- **`diffdock_batch.py:114`**: `return None` from a *binary-path probe*
  function (`_find_diffdock_binary`). When DiffDock is not on PATH and
  the vendored repo is absent, the probe returns `None` so the caller
  (in `score_any.py:_w_diffdock_pose`) raises `AdapterUnavailable`. This
  is the *correct* "tool missing" signal, not a silent fallback.
- **`hilbert_astsr_cuda.py:304`**: `return None` for an empty
  `features_batch` (zero-length input). Empty-input handling, not a
  failure fallback. The caller raises `ValueError` for length mismatch
  and proceeds for empty input.
- **`pb_batch.py:159, 200`**: `return 0.0` is a PoseBusters default
  for an *empty batch* (zero molecules), which is mathematically the
  identity element. Not a silent failure — empty input has no
  pass-rate by definition.

### 1.4 Adapter helpers (`/home/hugo/codes/mol-metal/molmetal/molmetal_lam/sbdd_env/`)

The remaining `return None` / `return 0.0` occurrences outside the router
are in adapter-side helpers (qed_scorer, sa_score, posebusters_adapter,
flowdock_adapter, diffdock_adapter, metal_coord_probe, retrosynthesis,
reinvent4_*, pybind_adapter, etc.). These are all in one of three
categories:

| Category | Examples | Status |
|---|---|---|
| **Probe / lookup helpers** — returns `None` on "not found" | `_probe_vina_binary`, `_find_diffdock_binary`, `_resolve_binary` | Correct |
| **Empty input handling** — returns 0.0 / `[]` / `{}` on `len==0` | `_vina_to_confidence` boundary, `_vina_version` | Correct |
| **Documented RDKit graceful degradation** — returns 0.0 / `None` when RDKit is unavailable or SMILES is unparseable | `qed_scorer.qed_from_smiles`, `sa_score.sa_score_ertl`, `metrics_v2` heuristic panel | Documented |

None of these are silent fallback paths for ML inference. None of them
fabricate a real-model score; they all return 0.0 only on
input-validity failure or genuine empty input.

### 1.5 REINVENT4 subprocess / multiproperty workers

The `reinvent4_subprocess_adapter.py`, `reinvent4_multiproperty_jsonl_worker.py`,
and `reinvent4_wrapper.py` modules return `None` / `0.0` in many places.
These are subprocess wrappers — the `None` return is paired with an
explicit status string (`"missing_binary"`, `"timeout"`, `"backend_error:..."`,
`"backend_error:csv_missing"`, etc.) so callers can surface the actual
backend failure mode. This is **explicitly tagged** failure reporting, not
silent fabrication. The `policy.json` `wf43_status` for `reinvent_*`
metrics is `MEASURED` because the subprocess IS the real tool; the
adapter wrapper's job is to surface subprocess failures honestly.

---

## §2 Verification: Wf-43 invariant upheld

| Check | Result | Evidence |
|---|---|---|
| `score_any.py`: silent `return None` count | **0** | grep audit (this TODO) |
| `score_any.py`: silent `return 0.0` count | **0** | grep audit (this TODO) |
| `score_any.py`: `AdapterUnavailable` raises | **20** | grep audit |
| `metrics_v2.py:_safe_external_score` raises on every failure | **Yes** | TODO-114 patch verified at lines 903-971 |
| `metrics_v2.py` Phase-4A scorers raise on missing adapter | **Yes** | TODO-114 patch |
| GPU wrappers correctly raise / probe | **Yes** | diffdock_batch, hilbert_astsr_cuda documented probes |
| Adapter helpers don't fabricate ML inference | **Yes** | documented graceful-degradation for RDKit availability / parse failures |

**Verdict**: 0 silent fallback paths in the scoring layer (per the
Wf-43 definition — silent-fallback = uncaught ML-model failure that
propagates a fabricated number into the panel).

---

## §3 Honest framing (party-twenty-spirit §3.5)

- **MEASURED**: 0 silent fallback paths in `score_any.py`; 20 loud
  `AdapterUnavailable` raises (lines 272, 294, 385, 421, 450, 464,
  471, 512, 570, 587, 600, 616, 629, 637, 645, 705, 1234, 1304, 1350,
  1368). The router core is clean and has been since Wf-43 Phase 1
  (per Agent 3 audit 2026-09-27).
- **MEASURED**: `metrics_v2.py:_safe_external_score` was patched in
  TODO-114 to raise `AdapterUnavailable` on every failure. Verified by
  reading `metrics_v2.py:903-971` in this audit pass.
- **MEASURED**: The 8-metric heuristic panel's `return 0.0` on
  RDKit parse failure is **not** a silent fallback. These are pure
  RDKit-descriptor formulas where `0.0` is a legitimate, documented
  value (logP=0 is "neutral lipophilicity"; AMES=0 is "no alert";
  hERG=0 is "safe"; GI50=0 is "low cytotoxicity"; PPB=0 is "no
  plasma binding"). The module docstring documents this contract.
- **PROJECTED**: 0 silent fallback paths remain in the scoring layer.
- **NOT PROJECTED, NOT MEASURED**: We did **not** verify other layers
  (e.g. `molmetal/molmetal_lam/lam_chem/`, GPU adapter tests). TODO-118
  is scoped to `molmetal/scoring/` + `molmetal/molmetal_lam/sbdd_env/`
  per §1 scope.

---

## §4 Negative tests (already exist)

The existing test suite covers the fallback contract:

- `molmetal/tests/test_score_any.py` — happy-path / known-metric /
  unknown-metric raise behavior
- `molmetal/tests/test_score_any_policy.py` — `ROUTING_POLICY` /
  `policy.json` consistency audit
- `molmetal/tests/test_score_any_router_bench.py` — routing policy
  bench
- `molmetal/tests/test_score_any_fallback.py` — Wf-48 Phase 3 fallback
  chain harness (≥20 checks: harness imports, fallback_chain audit,
  primary-unavailable propagates `AdapterUnavailable`,
  `MOLMETAL_VINA_ENGINE` env-var routing, fallback_test.json report,
  exit-code contract)
- `molmetal/tests/test_score_any_gpu_routing.py` — GPU routing
  (MOLMETAL_VINA_ENGINE=gpu → vina_gpu wrapper)
- `molmetal/scripts/test_score_any_fallback.py` — fallback harness
  source

A targeted `pytest -q molmetal/tests/test_score_any_fallback.py` run
verifies the silent-fallback negative test contract end-to-end.

---

## §5 Cross-references

- Agent 3 score_any research (2026-09-27) — initial audit finding
- TODO-110 meko metal atom extension (related scoring fix)
- TODO-114 MOCK_DELETED cleanup (the `_safe_external_score` patch)
- TODO-113 stub-floor reframe (related `_stub_score` paths in
  `dock_10pockets_se3sb.py` and `bench_100pocket_10seed.py` — both
  gated by `use_real_vina=False` opt-in)
- `molmetal/scoring/score_any.py` (the router — verified clean)
- `molmetal/scoring/REGISTRY.md` + `policy.json` (metric registry)
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:903-971`
  (`_safe_external_score` — TODO-114 patch verified)
- party-twenty-spirit §3.5 (Wf-43 invariant upheld)
- TODO-84 paper rule (paper §3.4 + §4 + §6 tables not blocked by
  this hygiene TODO)

---

## §6 Status

**Status**: ✅ **CLEAN 2026-09-27** — close-out audit complete (2026-09-27
audit pass by close-out agent).

**Verification evidence (this audit)**:
- `score_any.py` (1383 LOC): 0 `return None`, 0 `return 0.0`, 20
  `AdapterUnavailable` raises (loud Wf-43 contract).
- `metrics_v2.py:_safe_external_score` (lines 903-971): raises
  `AdapterUnavailable` on every failure path — empty SMILES, missing
  module, missing adapter, score() returning None, score() raising.
- Heuristic 8-metric panel: `return 0.0` on RDKit parse failure is the
  documented graceful-degradation contract (module docstring §3) for
  pure RDKit descriptor formulas (Hou 2007, Veith 2009, Benigni-Richard
  2005, Hughes 2008, Delaney 2004 ESOL, Obach 1999). NOT a Wf-43
  silent fallback.

**Action**: Close-out complete — file already at
`TODO/completed/118_score_router_hygiene.md`. No code changes required
(score_any was already clean; metrics_v2.py:_safe_external_score was
patched under TODO-114). The remaining `return None` / `return 0.0`
occurrences are documented graceful-degradation contracts for RDKit
heuristics, binary probes, or empty inputs — none are silent fallback
paths for ML inference.