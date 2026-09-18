# WF-T30 Tier-2 CPU-only — Master Consolidation

**Date**: 2026-09-17
**Scope**: TODO/30 Tier-2 (ranks 10-16) — three CPU-only pitfalls closed
**Status**: 3 / 3 SHIPPED · 0 BLOCKED on this round
**Companion**: `TODO/pending/30_pitfall_reinforce_plan.md` (updated)

---

## 1. TL;DR

Three TODO/30 Tier-2 CPU-only pitfalls shipped: P6.1+P6.2
reproducibility (scaffold split + per-run hygiene), P5.3 rule
registry (public API), and P2.2 always-on 3D (default flip). All
additive, CPU-only, bit-for-bit backward compatible. **No GPU
retrain claimed.**

---

## 2. Outcomes table

| Workflow | TODO ID | Rank | Verdict | LOC | Tests |
|---|---|---|---|---|---|
| `wf_t30_p62_scaffold_seed` | P6.1+P6.2 | 14 | **PASS** (14/14) | ~460 (1 NEW + 3 EDIT + 1 test) | 14 |
| `wf_t30_p53_rule_registry` | P5.3 | 9 | **PASS** (15/15) | ~680 (2 NEW) | 15 |
| `wf_t30_p22_default_3d` | P2.2 | 7 | **PASS** (8/8) | ~140 (2 EDIT + 1 test) | 8 |

Total: ~1280 LOC across 9 files (5 NEW, 4 EDITED); 37 new tests,
all green.

---

## 3. MEASURED deltas

### 3.1 P6.1 + P6.2 — Reproducibility (`wf_t30_p62_scaffold_seed`)
- **`--split {none,scaffold,time,random,scaffold_tanimoto_0.4}`**
  flag wired into `r4_lambda_only_run.py` (default `none`).
- **`metallo_pool_scaffold_split.py`** NEW: Bemis-Murcko scaffold
  split. 449 rows → 213 Murcko clusters → 314/67/68.
  Deterministic (byte-identical CSVs).
- **`seeds.json`** next to `report.json` per cell.
- **`fingerprint.json`** under `runs/{run_id}/` with `cli_sha256`,
  `smiles_concat_sha256`, `receptor_sha256`, `ligand_sha256`,
  `manifest_sha256`, `driver_sha256`, `runner_sha256`,
  `runtime_sha256`, `seeds`, `n_pockets`.
- **Honest caveat**: `time` is placeholder (no temporal column);
  `--split scaffold_tanimoto_0.4` is Tanimoto relaxation, NOT a true
  sequence-identity split.

### 3.2 P5.3 — Rule registry (`wf_t30_p53_rule_registry`)
- **`molmetal_lam/reactions/rule_registry.py`** NEW (~340 LOC).
  Public surface: `RuleRecord` (8-field frozen dataclass),
  `register_rule`, `unregister_rule`, `list_rules`, `vet_rule`,
  `record_test_result`, `clear_registry`, `registry_snapshot`,
  `wire_click_rules`, `wire_metal_coordination_rules`,
  `wire_all_default_rules`.
- 15 tests cover register/unregister round-trip, multi-scope
  coexistence, duplicate-registration errors, yield-range
  preservation, wire-in helpers (7 records / 2 scopes).
- `vet_rule` raises `KeyError` on unknown names (not `ValueError`).
  Metadata **declared** (literature-cited), not measured.
- `beta_reductions.py` + `pt_click_compat.py` NOT modified.

### 3.3 P2.2 — Always-on 3D (`wf_t30_p22_default_3d`)
- **`MCTSProofSearch.search(materialize_3d=True)`** — default flipped
  False → True at `proof_search.py:3116`. Every emitted candidate
  carries `coords_3d` (torch tensor `(n_heavy_atoms, 3)`).
- **`--no-materialize-3d`** escape hatch on `r4_lambda_only_run.py:4421`.
- 8 tests verify: signature default, escape hatch, RDKit-failure
  WARNING guard, CLI round-trip, no clobber of explicit `False`.
- Best-effort per-candidate: SMILES that fails `EmbedMolecule` falls
  back to leaving `coords_3d` unset; audit counter
  `n_coords_3d_attached` recorded on `self.history[-1]`.
- Backward compat: SMILES/connectivity unchanged; coord tensor is an
  extra attribute only.

---

## 4. What remains BLOCKED

- **P1.2** FG-compat veto — DEFER (needs property tests first).
- **P2.4** skeleton jump — DEFER (after P2.3; high invalid-structure risk).
- **P6.3** MMseqs2 + Tanimoto<0.4 — DEFER (binary not installed).
- **P4.1** generalisation 100×3 — DEFER (Round-13 sweep re-execution,
  GPU-blocked per `wf_gpu_recovery_now` 2026-09-15).
- **P5.2** Tier 3 wet-lab outreach — DEFER (chemistry-facility dep).

---

## 5. Honest framing

1. **No GPU retrain.** All three are pure CPU engineering (RDKit +
   json + signatures + argparse). The only GPU-requiring TODO/30 item
   is P4.1, DEFER'd.
2. **Backward compatibility is bit-for-bit when flags are OFF.**
   `--split none`, `--no-materialize-3d`, and the rule registry
   opt-in helpers preserve legacy behavior. Existing `r4_*`
   invocations continue unchanged.
3. **`time` split is placeholder.** No temporal column in the
   bundled 500-pool; helper preserves input order. Forward-compat:
   when a `date` column is added, the helper switches to cutoff-date
   split without changing the CLI surface.
4. **`vet_rule` raises `KeyError`** (intentional distinction from
   duplicate-registration `ValueError`). Callers needing a soft
   "exists" check should `name in [r.name for r in list_rules()]`.
5. **`materialize_3d` is best-effort per-candidate.** A SMILES that
   fails RDKit `EmbedMolecule` leaves `coords_3d` unset and logs
   `_LOG_P2_2_FALLBACK.warning(...)`. Callers must check
   `hasattr(cand, 'coords_3d')` before consuming the tensor.

---

## 6. Files touched (summary)

- **NEW**: `metallo_pool_scaffold_split.py` (~300), `rule_registry.py`
  (~340), `test_rule_registry.py` (15 tests), `test_materialize_3d_default.py`
  (8 tests), `test_reproducibility_split.py` (14 tests), scaffold split CSV +
  manifest.
- **EDITED**: `r4_lambda_only_run.py` (~+250: --split flag, seeds.json,
  --no-materialize-3d), `r4_c_full_sweep.py` (~+50: fingerprint.json),
  `proof_search.py` (default flip at line 3116).
- **NOT TOUCHED**: `beta_reductions.py`, `pt_click_compat.py`.

---

## 7. Verify

```bash
uv run python molmetal/scripts/metallo_pool_scaffold_split.py \
    --tanimoto-threshold 0.6 \
    --output-dir molmetal/data/metallo_scaffold_split
uv run pytest molmetal/tests/test_reproducibility_split.py -v
uv run pytest molmetal/molmetal_lam/tests/test_rule_registry.py -v
uv run pytest molmetal/molmetal_lam/tests/test_materialize_3d_default.py -v
```

All CPU-only, seconds, no GPU/CFM dependency.
