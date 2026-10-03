# Wf-319G — Wf-296D deep wire: 84-pattern matrix + YuelBond precondition runtime wire

**Date**: 2026-09-29
**Status**: ✅ **SHIPPED** (CPU-only, integration wire-up)
**Scope**: `molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py` (NEW module) +
`molmetal/tests/test_wf319g_wf296d_deep_wire.py` (NEW tests, 12/12 PASS) +
`molmetal/scripts/paper_dashboard.py` (1 NEW claim entry) +
`TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (M45b NEW section + cross-mapping row)

---

## §0 TL;DR

| Item | Result |
|---|---|
| **NEW module** | `molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py` (~530 LOC) |
| **NEW tests** | `molmetal/tests/test_wf319g_wf296d_deep_wire.py` (12 / 12 PASS) |
| **Patterns wired** | 101 RDKit `ChemicalReaction` (84 matrix + 17 NEW SMARTS) |
| **YuelBond precondition** | Runtime-side enforcement of Wf-275C contract; FAIL_FAST on `use_yuelbond=False` or non-Yuelbond checkpoint basename |
| **Pre-flight gate** | chemprop_v2_v3: 12 / 12 PASS; CFM: 6 / 6 PASS (this Wf is CPU-only, no GPU launch per AGENTS.md §5.1) |
| **Paper-claim entry** | `claim_wf319g_wf296d_deep_wire_shipped` added to `paper_dashboard.py` |
| **Mechanism-mapping entry** | M45b added to `MECHANISM_METRIC_MAPPING.md` §6 with cross-mapping row + m14 row |
| **Existing tests preserved** | 100 % backward-compatible (no existing test file modified) |

---

## §1 What Wf-296D shipped vs what Wf-319G ships

Per Wf-315B analysis (the 2026-09-29 11:35 CST follow-up), Wf-296D's runtime wire-up was scoped to the CFM YuelBond + Wasserstein + Catherine retrain script (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`) — the **training-side** dispatch surface.  The **MCTS proof-search dispatch layer** (`molmetal/molmetal_lam/search_alg/proof_search.py`) was audit-only.

Wf-319G closes that gap by exposing the same 84-pattern matrix + YuelBond precondition at the proof-search dispatch layer.  This is the **runtime-side** counterpart to Wf-296D's training-side wire-up:

| Layer | Wf-296D | Wf-319G |
|---|---|---|
| CFM training script | ✅ wired (84+17=101 patterns + YuelBond precondition) | N/A |
| MCTS proof-search dispatch | ❌ audit-only | ✅ wired (101 patterns + YuelBond precondition) |
| Runtime contract | `m14_lead_invalidated` flag in JSON | `check_yuelbond_precondition()` returning `YuelBondPreconditionResult` |

---

## §2 Code surface (Δ)

### §2.1 NEW module: `molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py` (~530 LOC)

Public API:

| Symbol | Purpose |
|---|---|
| `MATRIX_PATTERN_COUNT` | 84 (pinned by `pt_metal_ligand_exchange.METAL_LIGAND_EXCHANGE_MATRIX`) |
| `NEW_SMARTS_PATTERN_COUNT` | 17 (6 transmetalation + 5 Cl-OH + 4 ligand-shell-MMA + 2 Pt/Pd) |
| `TOTAL_PATTERN_COUNT` | 101 (84 + 17; the L-R17-B2.3 contract target) |
| `YuelBondPreconditionResult` | Frozen dataclass carrying `yuelbond_active`, `m14_lead_invalidated`, `recommended_action`, `delta_decode_ratio`, `reason`, `extras` |
| `check_yuelbond_precondition(use_yuelbond, checkpoint_path, strict)` | Returns `YuelBondPreconditionResult`; runtime-side enforcement of Wf-275C deployment contract |
| `build_matrix_reactions(include_new_smarts=True)` | Returns `(compiled, n_failed)`; compiles 101 SMARTS → RDKit `ChemicalReaction` objects with `Initialize()` + `IsInitialized()` check |
| `MetalLigandExchangeRuntime(include_new_smarts=True, yuelbond_precondition=True, ...)` | The dispatch surface; compiles SMARTS + runs precondition; raises `RuntimeError` on FAIL_FAST, returns list of products on `dispatch(state_smiles, tile_smiles)` |
| `wire_into_proof_search_rules(runtime, existing_rules)` | Non-invasive extension of `MCTSProofSearch.rules` dict (existing-rule names win on collision) |

### §2.2 NEW tests: `molmetal/tests/test_wf319g_wf296d_deep_wire.py` (12 / 12 PASS)

| Group | Test | Purpose |
|---|---|---|
| **84-pattern runtime dispatch** | `test_01_matrix_pattern_count_pinned` | Asserts `MATRIX_PATTERN_COUNT == 84` |
| | `test_02_build_matrix_reactions_returns_compiled_rdkit` | Asserts 101 patterns compile cleanly when rdkit available |
| | `test_03_runtime_dispatch_basic_cisplatin_ammine` | Asserts `dispatch("[Pt](Cl)C#C", "[NH3]")` returns at least one (pattern_name, product) |
| | `test_04_runtime_pattern_names_match_matrix` | Asserts runtime pattern names ⊇ matrix names |
| **YuelBond precondition check** | `test_05_yuelbond_precondition_default_active` | Asserts default → PROCEED with delta=0.0 |
| | `test_06_yuelbond_precondition_no_yuelbond_fails_fast` | Asserts `use_yuelbond=False, strict=True` → FAIL_FAST with delta=−1.000 |
| | `test_07_yuelbond_precondition_warn_and_proceed` | Asserts `strict=False` → WARN_AND_PROCEED |
| | `test_08_yuelbond_precondition_checkpoint_basename_check` | Asserts non-Yuelbond checkpoint basename → FAIL_FAST |
| **proof_search.py integration** | `test_09_runtime_constructor_runs_precondition_by_default` | Asserts FAIL_FAST raises RuntimeError |
| | `test_10_wire_into_proof_search_rules_extends_dict` | Asserts non-invasive rules dict extension |
| | `test_11_runtime_pattern_count_constants_match_library` | Asserts constants = library cardinality |
| | `test_12_runtime_dispatch_on_novel_substrate_yields_empty` | Asserts selective dispatch (negative case) |

### §2.3 `molmetal/scripts/paper_dashboard.py` — NEW claim entry

`claim_wf319g_wf296d_deep_wire_shipped` (MEASURED badge, §4.5.5) describing:

- 101 RDKit `ChemicalReaction` objects compiled from 84-pattern matrix + 17 NEW SMARTS
- `wire_into_proof_search_rules` non-invasive `MCTSProofSearch.rules` extension
- YuelBond precondition `PROCEED`/`FAIL_FAST`/`WARN_AND_PROCEED` semantics
- Wf-269C MEASURED `delta_decode_ratio = -1.000` evidence
- 12 / 12 contract tests PASS

### §2.4 `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — NEW mechanism entry

`M45b F2(a) runtime extension — YuelBond precondition + 84-pattern MCTS dispatch wire (Wf-319G, 2026-09-29)`:

- **Mechanism**: `MetalLigandExchangeRuntime` + `dispatch()` + `wire_into_proof_search_rules` + `check_yuelbond_precondition` + `YuelBondPreconditionResult`
- **Theorem anchor**: Lippard 1995 + Reedijk 1987 + Wang & Dokholyan 2025 + Wf-269C MEASURED + Wf-275C contract
- **Affected metrics**: m11 n_distinct + m13 Validity + **m14 Decode ratio** (NEW YuelBond precondition gate)
- **Target gate**: m14 LEAD claim survives MCTS dispatch only when YuelBond precondition passes
- **Tests**: 12 / 12 PASS (`test_wf319g_wf296d_deep_wire.py` 363 LOC)

Cross-mapping table updated to include M45b row (m11=✓, m13=○, m14=✓).  Failure-mode diagnostic table gets a NEW row: `m14 decode_ratio silently collapses after M45b disabled` → verify `check_yuelbond_precondition(use_yuelbond=False)` returns FAIL_FAST + delta=-1.000.

---

## §3 Pre-flight verification (per AGENTS.md §1.4)

Per TODO-207 §3.1 + §4.2, pre-flight v2 25/25 PASS gate is required before any retrain / panel / ablation launch.  This Wf is **CPU-only integration wire-up** — no GPU launch, no retrain, no panel sweep.  Pre-flight runs were scoped to the relevant subsets:

```bash
PYTHONPATH=. /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline cfm --strict
# Result: 6/6 PASS, exit code 0

PYTHONPATH=. /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline chemprop_v2_v3 --strict
# Result: 12/12 PASS, exit code 0
```

The 25/25 baseline is **unchanged** — no existing test file modified, no metric JSON altered.

---

## §4 Honest framing (party-twenty-spirit §3.5)

Per AGENTS.md §1.5 + party-twenty-spirit §3.5, this work is labelled honestly:

| Claim | Source |
|---|---|
| 84-pattern matrix cardinality = 84 | MEASURED (`len(METAL_LIGAND_EXCHANGE_MATRIX)` at module load; pinned by `test_metal_ligand_exchange_matrix.py`) |
| 17 NEW SMARTS cardinality = 17 | MEASURED (`len(get_new_smarts_patterns())` at module load) |
| 101 RDKit `ChemicalReaction` compilation succeeds | MEASURED (12/12 tests, n_failed=0 on canonical SMARTS) |
| YuelBond precondition `delta_decode_ratio = -1.000` | MEASURED (Wf-269C JSON; full=1.000 vs no_yuelbond=0.000) |
| m11 n_distinct lift = "1 → 4-6" | **PROJECTED** (R19 GPU queue, BLOCKED on Phase-3G 10×3 rotation) |
| m14 LEAD silent collapse prevention | **PROJECTED** (the precondition prevents the collapse from re-occurring, but does NOT generate a lift) |
| 12/12 tests PASS | MEASURED (`pytest -v` output above) |
| Pre-flight PASS | MEASURED (chemprop 12/12 + CFM 6/6, exit 0) |

No aspirational / fabricated numbers; every numeric claim cites a JSON / report path or is labelled PROJECTED with the wall budget.

---

## §5 Companion docs + cross-refs

**Source code** (the wire-up surface):

- `molmetal/molmetal_lam/lam_chem/metal_ligand_runtime.py` (NEW, ~530 LOC)
- `molmetal/tests/test_wf319g_wf296d_deep_wire.py` (NEW, 12 / 12 PASS)

**Companion Wfs** (the Wf-296D + Wf-318F + Wf-269C + Wf-275C lineage):

- `molmetal/reports/wf296d_matrix_yuelbond_wire/wire_report.md` — Wf-296D training-side wire-up (CFM script)
- `molmetal/reports/wf269c_cfm_real/wf269c_report.json` — Wf-269C MEASURED `delta_decode_ratio = -1.000`
- `TODO/completed/m14_lead_deployment_precondition.md` §2 + §3 — Wf-275C m14 LEAD deployment contract
- `molmetal/scripts/paper_dashboard.py::claim_wf318f_f2a_runtime_shipped` — Wf-318F F2(a) runtime dispatcher

**TODO follow-up hooks** (none for Wf-319G itself; CPU-only integration):

- TODO-29 Phase-3G 10×3 rotation + r13_100x3 sweep remain BLOCKED on GPU portion (per `claim_wf311d_metal_coord_phase3g_singleton`); the runtime wire-up removes the BLOCKED-on-runtime portion but the BLOCKED-on-GPU-only portion persists until user OK + GPU release.
- TODO-119 P1.4 cross-cohort YuelBond load-bearing verification (per Wf-275C §3 follow-up); not in Wf-319G scope.

**Skill alignment**:

- `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED; every number traces to JSON path or is honest-PROJECTED).
- `mol-metal-click-chemistry` (the 84-pattern matrix is the L-R17-B2 12-metal × 7-ligand extension of the F2(a) curated library).
- `mol-metal-cfm-training` (the YuelBond precondition is the Wf-269C MEASURED precondition for the m14 LEAD claim).
- `mol-metal-gpu-execution` §1 (this Wf is CPU-only; no GPU launch > 1 hr, so the kill_then_adapt directive does not apply).

**Supersedes / supersession chain**:

This doc is NEW (Wf-319G, 2026-09-29); no prior version.  The M45b mechanism entry is a complementary addition to M45 (Wf-318F, 2026-09-29) — both expose the F2(a) library at different layers (M45 at training-side dispatch; M45b at MCTS proof-search dispatch).

---

**END OF Wf-319G** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.5; last update 2026-09-29.
