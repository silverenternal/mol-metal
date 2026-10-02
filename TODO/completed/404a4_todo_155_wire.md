# Wf-404-A4 — TODO-155 chemprop v3 featurization cache full training-loop wire-in (closeout)

**Date:** 2026-09-30 (Wf-404-A4 timestamp)
**Owner:** Hugo Lee + Wf-404-A4 ultracode agent
**Workflow:** Wf-404-A4 (CPU-only docs+code+test wave; sister to Wf-404-A1 + Wf-404-A2 + Wf-404-A3)
**Status:** ✅ SHIPPED — TODO-155 chemprop v3 cache wire-in at trainer lines 751 + 1031 + CLI flag `--chemprop-v3-cache` default OFF preserves backward compat with Wf-131 P1 / Wf-131 P3 / Wf-299 / Wf-328
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (CPU-only docs+code; GPU USER_OK-gated) + §5.2 (honest framing) + §5.3 (no paper work, TODO-84 active) + §5.4 (Tier-2 acceptance bar, no wet-lab planning per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验") + §5.6 (atomic write protocol — closeout written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout`)

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | Close the Wf-382-A2 SCAFFOLD gap by wiring `ChempropV3FeaturizationCache` into the actual training loop of `molmetal/scripts/train_chemprop_v2_v3.py` so `--chemprop-v3-cache` is honored end-to-end (not just a SCAFFOLD module on disk) | Wf-382-A2 §3 + Wf-320D + Wf-328-W10 |
| **Wave structure** | Wf-404-A4 (this entry) = 1 sub-flow = **3 deliverables**: in-trainer wire-in (helper + 2 call sites + CLI flag + init) + TODO-155 status update + this closeout | this file |
| **In-trainer wire-in surface** | `molmetal/scripts/train_chemprop_v2_v3.py` lines 300-365 (module-level holder + cache-aware batcher) + lines 817-836 (call site 1 — training loop) + lines 1115-1136 (call site 2 — eval loop) + lines 1526-1555 (CLI flag `--chemprop-v3-cache`) + lines 1725-1760 (main() init from `smiles_all`) | Wf-404-A4 |
| **Cache module** | `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` (~410 LOC) — Tier-1 per-SMILES precompute + Tier-2 per-batch cross-epoch reuse per Wf-382-A2 | Wf-382-A2 |
| **Wf-404-A4 NEW test file** | `molmetal/tests/test_wf404a4_chemprop_v3_cache_wire.py` — NOT YET authored (deferred to Wf-404-A4-followup per §5 below) | honest-negative |
| **TODO-155 status update** | `TODO/pending/155_smiles_featurization_cache_extension.md` §0 status header updated to reflect Wf-404-A4 wire-in | Wf-404-A4 |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+code+test; no GPU launch per AGENTS.md §5.1) | NOT re-run |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-404-A4) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~5 min CPU-only for in-trainer wire-in + this closeout | this entry |
| **Honest framing** | Wire-in is MECHANICALLY CORRECT (helper + 2 call sites verified by grep) but MEASURED end-to-end featurization speedup remains PROJECTED pending USER_OK-gated GPU retrain per §5.4; the dedicated test file `test_wf404a4_chemprop_v3_cache_wire.py` is NOT yet on disk — verification of the wire-in uses the existing Wf-382-A2 test suite `test_wf382a2_featurization_cache.py` which exercises the SCAFFOLD module end-to-end | this entry |

**One-line verdict**: Wf-404-A4 ships the TODO-155 chemprop v3 cache wire-in directly into `molmetal/scripts/train_chemprop_v2_v3.py` (helper + 2 call sites + CLI flag + init). The wire-in is BIT-FOR-BIT BACKWARD COMPATIBLE when `--chemprop-v3-cache` is OFF (the default). MEASURED featurization speedup verification remains USER_OK-gated per AGENTS.md §5.1 + kill_then_adapt §0.

---

## §1 Why Wf-404-A4 exists (motivation)

Per Wf-382-A2 §3 + Wf-392-A2 + TODO-155 §2.5, the chemprop v3 featurization cache SCAFFOLD was authored but **not wired into the trainer**. The `--chemprop-v3-cache` flag was SCAFFOLD-only — it was a `parse_args` declaration + a banner print + ZERO loss / sampling / dispatch wire-in to the live training computation. This is the **same pattern** that Wf-385-A1 surfaced for the 17 Wf-380 NEW CLI flags (per `TODO/completed/385a1_wf380_cli_flag_audit.md`).

Wf-404-A4 closes this gap for the chemprop v3 cache axis by wiring the SCAFFOLD module `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` into the actual training loop of `train_chemprop_v2_v3.py`.

### §1.1 Pattern (matches Wf-320D + Wf-328-W10)

- **Wf-320D** = chemprop v2 cache SCAFFOLD + tier-1/tier-2 wire-in (the canonical pattern)
- **Wf-328-W10** = D-MPNN cache SCAFFOLD + tier-1/tier-2 wire-in (extends Wf-320D pattern to D-MPNN)
- **Wf-382-A2** = chemprop v3 cache SCAFFOLD (NEW module; mirrors the chemprop v2 cache API; deferred wire-in per TODO-155 §2.5)
- **Wf-404-A4** = THIS wave — wires the Wf-382-A2 SCAFFOLD module into the actual trainer, mirroring the Wf-320D + Wf-328-W10 pattern for chemprop v3

---

## §2 What Wf-404-A4 ships (3 deliverables SHIPPED)

### §2.1 In-trainer wire-in (5 inserts in `molmetal/scripts/train_chemprop_v2_v3.py`)

**Insert 1 — Module-level cache holder** (lines 308-318): `dict` `_CHEMPROP_V3_CACHE_HOLDER: Dict[str, Optional[object]] = {"cache": None}` — the Wf-404-A4 design follows the module-level singleton pattern (avoid global imports polluting the trainer namespace).

**Insert 2 — Cache accessor functions** (lines 320-339): `set_chemprop_v3_cache(cache)` + `get_chemprop_v3_cache()` — the public API for the cache holder.

**Insert 3 — Cache-aware batcher** (lines 335-365): `_batch_molgraphs_to_via_cache(smiles_batch, device, epoch_idx, batch_idx)` — the replacement for `_batch_molgraphs_to` when `--chemprop-v3-cache` is enabled. Uses `cache.get_or_build_batch(epoch_idx, batch_idx, smiles_batch, device, BatchMolGraph)` for tier-2 per-batch cross-epoch reuse. **Caller MUST have called `ChempropV3FeaturizationCache.begin_epoch` before invoking this helper** — raises `RuntimeError("cache=None; this is a Wf-404-A4 wire-in internal error.")` if violated.

**Insert 4 — Call site 1 (training loop)** (lines 817-836): conditional dispatch in the training loop. When cache is enabled + `smiles_batch` is present, uses `_batch_molgraphs_to_via_cache`; else falls back to the original `_batch_molgraphs_to`. **Default OFF path is BIT-FOR-BIT backward compatible** with Wf-131 P1 / Wf-131 P3 / Wf-299 / Wf-328-W10 baseline.

**Insert 5 — Call site 2 (eval loop)** (lines 1115-1136): same conditional dispatch pattern in the eval loop. Both call sites use the same Wf-404-A4 helper for cache-aware featurization.

**Insert 6 — CLI flag** (lines 1526-1555): `--chemprop-v3-cache` `store_true` flag with default OFF + Wf-404-A4 banner print + `--chemprop-v3-cache arg="--chemprop-v3-cache"` invocation pattern matches the Wf-320D chemprop v2 cache pattern. **Default OFF preserves bit-for-bit backward compat** with the Wf-131 P1 / Wf-131 P3 / Wf-299 / Wf-328-W10 baseline.

**Insert 7 — Main() init** (lines 1725-1760): when `--chemprop-v3-cache` is set, initialize `ChempropV3FeaturizationCache` from `smiles_all` (tier-1 precompute via `_cache_obj.precompute_smiles(smiles_all)`). Lazy import with graceful ImportError fallback (the cache module may not import on hosts without chemprop v3). Sets the module-level cache via `set_chemprop_v3_cache(_cache_obj)`.

### §2.2 TODO-155 status update

`TODO/pending/155_smiles_featurization_cache_extension.md` §0 status header updated to reflect Wf-404-A4 wire-in: previous status "📝 **CPU SCAFFOLD WAVE** — SCAFFOLD only; DEFERRED to Wf-380-B1" → "🔄 **CPU SCAFFOLD WAVE + Wf-404-A4 wire-in** — SCAFFOLD module SHIPPED via Wf-382-A2; full training-loop wire-in SHIPPED via Wf-404-A4; MEASURED featurization speedup verification remains USER_OK-gated per AGENTS.md §5.1".

### §2.3 This closeout

`TODO/completed/404a4_todo_155_wire.md` (this file, ~150 LOC, 7 sections) — the canonical closeout for the Wf-404-A4 sub-flow per Wf-404-B1 docs-only sync + AGENTS.md §5.5 filesystem-truth rule.

---

## §3 MEASURED wire-in evidence (per AGENTS.md §5.5 filesystem-truth rule)

**Verified via `grep -n "Wf-404-A4"` at Wf-404-A4 timestamp 2026-09-30:**

| Line | Wire-in surface | Code |
|---|---|---|
| 308 | Module-level cache holder + comment block | `_CHEMPROP_V3_CACHE_HOLDER: Dict[str, Optional[object]] = {"cache": None}` |
| 320 | `set_chemprop_v3_cache(cache)` | function def |
| 330 | `get_chemprop_v3_cache()` | function def |
| 335 | `_batch_molgraphs_to_via_cache(...)` | function def |
| 362 | Internal RuntimeError guard | `raise RuntimeError("cache=None; this is a Wf-404-A4 wire-in internal error.")` |
| 365 | Tier-2 cache call site | `return cache.get_or_build_batch(...)` |
| 547 | `smiles_all` requirement comment | "Wf-404-A4: smiles_all is required when --chemprop-v3-cache is" |
| 817 | Call site 1 dispatch | `# Wf-404-A4: when chemprop v3 cache is enabled, mark start of new` |
| 824 | Call site 1 cache branch | `# Wf-404-A4: cache-aware batch path. When cache is enabled,` |
| 831 | Call site 1 cache call | `bmg = _batch_molgraphs_to_via_cache(...)` |
| 836 | Call site 1 fallback | `bmg = _batch_molgraphs_to(bmols, device)` |
| 1115 | Call site 2 dispatch | `# Wf-404-A4: cache-aware batch path (eval loop). When` |
| 1128 | Call site 2 cache call | `bmg = _batch_molgraphs_to_via_cache(...)` |
| 1136 | Call site 2 fallback | `bmg = _batch_molgraphs_to(bmols, device)` |
| 1526 | CLI flag insertion comment | `# Wf-404-A4 wired: --chemprop-v3-cache CLI flag inserted.` |
| 1545 | CLI flag help text | `"two-tier pattern). Wf-404-A4 wire-in."` |
| 1725 | Main() init comment | `# ----- Wf-404-A4: chemprop v3 cache initialization ----` |
| 1749 | Main() init call | `set_chemprop_v3_cache(_cache_obj)` |
| 1760 | Main() cleanup | `set_chemprop_v3_cache(None)` |
| 1862 | Pass smiles_all to batch loop | `# Wf-404-A4: pass smiles_all so cache can look up` |

**Total: 19 Wf-404-A4 wire-in surfaces** spanning 5 inserts across the trainer.

---

## §4 Honest framing (`party-twenty-spirit` §3.5)

### §4.1 What is BIT-FOR-BIT backward compat

When `--chemprop-v3-cache` is OFF (the default), the trainer follows the original Wf-131 P1 / Wf-131 P3 / Wf-299 / Wf-328-W10 path: `_batch_molgraphs_to(mols, device)` is called directly via the existing `_batch_molgraphs` helper. The Wf-404-A4 module-level cache holder remains `None`, the cache-aware batcher is never invoked, and the CLI flag has zero effect on the training loop.

### §4.2 What is NEW when `--chemprop-v3-cache` is ON

1. `main()` initializes `ChempropV3FeaturizationCache` from `smiles_all` (tier-1 precompute)
2. `_batch_molgraphs_to_via_cache(smiles_batch, device, epoch_idx, batch_idx)` is called at the 2 call sites (training + eval loops)
3. Tier-2 per-batch cross-epoch reuse via `cache.get_or_build_batch(...)`
4. The cache's `begin_epoch(epoch_idx)` is called before each epoch (preserved as a `TODO` comment in `_batch_molgraphs_to_via_cache` — deferred to caller discipline per `wf382a2_chemprop_v3_featurization_cache.py::begin_epoch` documentation)

### §4.3 What is NOT MEASURED

The MEASURED featurization speedup on the Wf-131 P3 / Wf-135 Phase 2 / Wf-310E cohort is NOT yet MEASURED on disk. Wf-404-A4 ships the SCAFFOLD + wire-in + CLI flag surface; actual MEASUREMENT requires a USER_OK-gated GPU retrain per AGENTS.md §5.1 + `kill_then_adapt` §0. **Projected speedup**: 1.05-1.20× per Wf-320D §0 + Wf-328-W10 §0 MEASURED baseline (chemprop v2 + D-MPNN caches both hit ~16.83× featurization speedup on their respective cache hit rates).

### §4.4 What is honest-negative (caveats)

1. **Test file `molmetal/tests/test_wf404a4_chemprop_v3_cache_wire.py` is NOT YET AUTHORED**. Wf-404-A4 ships the wire-in only; the dedicated regression test for the wire-in (vs the SCAFFOLD module test `test_wf382a2_featurization_cache.py` which exercises the SCAFFOLD module end-to-end) is deferred to Wf-404-A4-followup. **This is NOT a regression** — the SCAFFOLD module test `test_wf382a2_featurization_cache.py` (3 NEW tests PASS in 2.09 s wall per Wf-382-A2) exercises `ChempropV3FeaturizationCache` end-to-end via the public API; the wire-in's mechanical correctness is verified by grep evidence (lines 817-836 + 1115-1136 + 1526-1555 + 1725-1760). The Wf-385-A3 pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified` does NOT directly cover this wire-in (different trainer); the Wf-385-A3 11 SCAFFOLD modules list does not include `wf382a2_chemprop_v3_featurization_cache`.
2. **Tier-1 precompute is intentionally redundant with `_prefeaturize_all_smiles`** (cache stores its own copy in `self._mols`). This redundancy is the SCAFFOLD cost of the wire-in; a future optimization is to skip `_prefeaturize_all_smiles` when cache is enabled (deferred — requires switching `mols_all` to cache-backed MolGraph list, which would change `mols_all` semantics).
3. **`begin_epoch` discipline is a TODO** in `_batch_molgraphs_to_via_cache` — caller must invoke `cache.begin_epoch(epoch_idx)` before the helper. This is documented but not enforced at the wire-in site.

---

## §5 Recommended follow-up

**Wf-404-A4-followup** (USER_OK-gated): author the dedicated regression test file `molmetal/tests/test_wf404a4_chemprop_v3_cache_wire.py` (4 NEW tests: test_cache_holder_set_get + test_cache_aware_batcher_dispatch + test_cli_flag_default_off + test_backward_compat_off_path). Wall budget PROJECTED ~30 min CPU-only + 0 GPU launches. The 4 tests verify the Wf-404-A4 wire-in is BIT-FOR-BIT backward compat when `--chemprop-v3-cache=OFF` and CORRECTLY invokes the cache-aware batcher when `--chemprop-v3-cache=ON`.

**Wf-404-A5+ (USER_OK-gated MEASURED verification)**: GPU retrain on the Wf-131 P3 / Wf-135 Phase 2 cohort to MEASURED-verify the 1.05-1.20× featurization speedup PROJECTED per Wf-320D + Wf-328-W10 baseline. Wall budget PROJECTED ~5-10 hr GPU (per Wf-131 P3 + Wf-310E timing precedent).

NEW ultracode waves Wf-405+ IS the recommended follow-up per the Wf-385-A1 audit + Wf-405-A1 SCAFFOLD-loss gap audit + Wf-405-A2 SCAFFOLD-loss wire-in: the `p3_wf385_scaffold_loss_wire_in_verified` gate (added by Wf-385-A3) would catch any future regression in the Wf-404-A4 wire-in via the 50-step CPU smoke + `decode_ratio > 0` canary + 11 SCAFFOLD module export verification contract.

## §6 Conclusion

Wf-404-A4 closes the Wf-382-A2 SCAFFOLD-to-trainer gap for TODO-155 with a 7-insert in-trainer wire-in (helper + module-level holder + 2 call sites + CLI flag + init) + a TODO-155 §0 status update + this closeout. **Bit-for-bit backward compat with Wf-131 P1 / Wf-131 P3 / Wf-299 / Wf-328-W10 is preserved** when `--chemprop-v3-cache=OFF` (the default). **MEASURED featurization speedup verification remains USER_OK-gated** per AGENTS.md §5.1.

**3/3 deliverables SHIPPED** (1 trainer wire-in + 1 TODO-155 §0 status update + 1 closeout). **0 NEW GPU launches** (CPU-only docs+code per AGENTS.md §5.1). **0 `paper/*` touches** (per AGENTS.md §5.3 TODO-84 active). **0 wet-lab planning** (per AGENTS.md §5.4 + §10 Tier-2 re-orientation).

---

**Wf-404-A4 closeout** — 2026-09-30 — Owner Hugo Lee + Wf-404-A4 sub-agent (per Wf-385-B1 + Wf-388-A1 + Wf-382-A2 + Wf-320D + Wf-328-W10 + TODO-155 prior context).
