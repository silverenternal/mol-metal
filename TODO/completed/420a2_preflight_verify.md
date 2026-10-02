# Wf-420-A2 — pytest + pre-flight fresh-run verify AFTER Wf-420-A1 fix

**Date:** 2026-10-01
**Wave:** Wf-420 (ultracode TRUE UNBLOCK via fixing the actual root cause of `p2_tests_pass` FAIL per Wf-419-A1 §0)
**Pre-flight baseline:** TODO/completed/207_unified_pre_flight_checklist_v2.md
**Atomic write protocol:** AGENTS.md §5.6 (Wf-386-A3)

---

## §0 TL;DR

**HONEST-NEGATIVE — pre-flight cascade NOT fully closed at Wf-420-A2 timestamp.**

| Metric | Task brief expectation | MEASURED Wf-420-A2 timestamp | Status |
|---|---|---|---|
| `p2_tests_pass` FAIL closed via Wf-420-A1 9 SMARTS fix | 0 FAIL | **2 FAIL** (p2_tests_pass + cascade child) | **NOT closed** — wavelength conflict |
| Pre-flight v3 fresh-run total | 117 TOTAL | 117 TOTAL | OK |
| Pre-flight v3 fresh-run PASS | ~113 | **112** | within noise |
| Pre-flight v3 fresh-run SKIPs preserved | ~3 BY-DESIGN | **3** BY-DESIGN | OK |
| Pre-flight v3 fresh-run FAIL | 0 | **2** (`p2_tests_pass` + cascade child) | NOT closed |
| Gate count consistency | actual=117 == expected=117 | **PASS** `actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` | OK |
| Wf-299 launcher `--wave A --dry-run` exit code | 0 | **0** | OK |
| Wf-299 launcher pre-flight FAIL block | none | none | OK |

**WAVELENGTH CONFLICT ROOT-CAUSE (per `party-twenty-spirit §3.5` honest framing)**: Wf-420-A1 closeout claimed the 9 missing SMARTS symbols fix would close `p2_tests_pass`. The Wf-420-A1 closeout narrative is HONEST about closing the SPECIFIC `CL_OH_SWAP_SMARTS` ImportError on `molmetal/tests/test_f2a_metal_ligand_exchange.py:62` line — that local fix is real. However, the pre-flight `p2_tests_pass` gate runs the FULL pytest suite (701 passed + 1 failed + 12 skipped). The single failure is **NOT** in `test_f2a_metal_ligand_exchange.py` — it is a SEPARATE ImportError at `molmetal/tests/test_joint_3stage_mcts.py:95`:

```
ImportError: cannot import name 'EXPLORATION_STRATEGIES' from 'molmetal.molmetal_lam.search_alg.proof_search'
```

This is a different module + different symbol + different file. Wf-420-A1 did NOT address this. Closure path = Wf-421-A1 ~5 LOC fix at `molmetal/molmetal_lam/search_alg/proof_search.py` adding `EXPLORATION_STRATEGIES` constant + `__all__` export list; or alternatively rename the import in `test_joint_3stage_mcts.py:95` to use the actual symbol exposed by `proof_search.py`.

**VERDICT**: Wf-420-A1 fix is PARTIAL — it closed ONE OF TWO independent ImportErrors; the remaining one keeps `p2_tests_pass` FAIL. Wave 420 cannot be marked as having closed the pre-flight cascade.

0 GPU launches performed. 0 `paper/*` touches. 0 wet-lab planning.

---

## §1 Pytest results (3 specified test files — task brief step 1)

Command: `python -m pytest molmetal/tests/test_f2a_metal_ligand_exchange.py molmetal/tests/test_no_kendall_no_dann_flags.py molmetal/tests/test_pitfall_guards.py -v`

### §1.1 Summary

| File | Pass | Fail | Skip | Notes |
|---|---|---|---|---|
| `test_f2a_metal_ligand_exchange.py` | 73 | **65** | 1 | REAL_TO_LAMBDA rules missing from `REACTION_RULES` registry |
| `test_no_kendall_no_dann_flags.py` | 8 | 0 | 0 | clean |
| `test_pitfall_guards.py` | 3 | **7** | 0 | real molecule scaffold construction errors |
| **Total** | 84 | **72** | 1 | **Wave brief expects 0 FAIL — actual 72 FAIL** |

### §1.2 Root-cause analysis (NOT closed by Wf-420-A1)

The 65 FAILs in `test_f2a_metal_ligand_exchange.py` are due to a path bug in `molmetal/molmetal_lam/reactions/beta_reductions.py:1473`:

```
RuntimeWarning: [beta_reductions] SyntheMol registration failed:
[Errno 2] No such file or directory:
'/home/hugo/codes/try_triton_on_rocm/molmetal/references/SyntheMol/synthemol/utils.py'.
REACTION_RULES will contain only the 5 click rules.
```

The SyntheMol submodule path is hardcoded to a different repo (`/home/hugo/codes/try_triton_on_rocm/...`) which does not exist on this host. As a result, `REACTION_RULES` contains only the 5 click-chemistry rules, but `REAL_TO_LAMBDA` (the parametrized fixture) references 16+ SyntheMol rule names like `amide_coupling`, `ester_hydrolysis`, `boc_cleavage`, etc. The test fails because `REACTION_RULES[rule_name]` raises `KeyError` for every SyntheMol rule.

Closure path: ~10 LOC fix at `molmetal/molmetal_lam/reactions/beta_reductions.py:1473` to compute the SyntheMol path relative to `__file__` rather than hardcoding it.

### §1.3 Wf-420-A1 fix status (the SMARTS ImportError on `test_f2a_metal_ligand_exchange.py:62`)

Wf-420-A1 closed the SPECIFIC line-62 ImportError on `CL_OH_SWAP_SMARTS` by adding 9 missing SMARTS symbols + accessor functions to `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`. The local fixture `test_register_metal_ligand_exchange_rules_returns_101` now passes (84 → 101 patterns). However, this fix did NOT address the broader `test_f2a_metal_ligand_exchange.py` failure pattern (REAL_TO_LAMBDA rule-name mismatch from the beta_reductions path bug).

The 1 FAIL the pre-flight pytest actually sees is a DIFFERENT file: `test_joint_3stage_mcts.py:95` failing to import `EXPLORATION_STRATEGIES`. This failure pre-existed Wf-420-A1 (it is unrelated to the SMARTS fix) but Wf-420-A1 did not detect it because Wf-420-A1 ran a subset of pytest (just the 3 test files in step 1) without checking the FULL pytest suite that the pre-flight runs.

---

## §2 Fresh-run results table (task brief step 3)

Source: `molmetal/reports/wf420a2_preflight_freshrun.txt` (1056 lines, atomic-written via `atomic_write_file` per AGENTS.md §5.6) — MEASURED at 2026-09-30T16:12:12+0000 UTC.

| Pipeline | PASS | FAIL | SKIP | TOTAL | Wall (s) |
|---|---:|---:|---:|---:|---:|
| all | 112 | **2** | 3 | 117 | 193.60 |

### §2.1 FAIL detail

| Gate | Status | Evidence |
|---|---|---|
| `p2_tests_pass` | **FAIL** | `pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 11.04s` |
| `p3_wf343c2_per_wave_runtime_check` | **FAIL** | cascade child of `p2_tests_pass` per Wf-367-A2 `--strict` removal semantics |

### §2.2 SKIPs preserved (BY-DESIGN, 3 total)

| Gate | Reason |
|---|---|
| `p3_wf343c4_yuelbond_decoder_env` | env gap (no YuelBond decoder binary on host) |
| `p3_wf343c4_wave_budget_remaining_nonneg` | -183.04h overrun per Wf-262 §7.1 |
| `data_2_1_6_bindingdb_external_honest_negative` | external BindingDB 1.07M dependency |

### §2.3 Gate count consistency audit

**`p3_gate_count_consistency_audit` PASS** — `gate count consistent for pipeline='all': actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` wall_sec=0.0002.

---

## §3 Wf-299 launcher dry-run verification (task brief step 5)

Command: `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A --dry-run`

**Result**: exit code 0; no pre-flight FAIL block.

Key dry-run output:

```
[00:13:35]   trainer: /home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py
[00:13:35]   flags:   --epochs 3 --seed 0 --pocket-id pocket_a --use-ot-cfm --stable-vm
                       --so3-average --log-normal-t --triton-ot --shortcut-distill
                       --consistency-weight 1.0 --top-p 0.9 --top-k 50 --temperature 1.0
                       --mmd-weight 0.1 --repulsion-weight 0.05 --per-cohort-features
                       --scaffold-conditioning --sa-penalty 0.0 --sa-target 2.8
                       --no-yuelbond --no-wasserstein --out-dir .../wave_a/catherine_only
[00:13:35] DRY-RUN: would launch Wave A
[OK] Wave A DRY-RUN logged
[OK] Wave A all 4 ablation configs complete
[OK] Wf-299C launch script done — wave=A
```

The launcher accepts `--wave A` per the Wf-408-A3 alias mapping fix and surfaces all 16 NEW trainer flags transparently per the Wf-415-A1 wrapper fix. USER_OK=1 honored.

---

## §4 Honest framing + forward-looking

**WAVELENGTH CONFLICT NOTE** (per AGENTS.md §5.5 filesystem-truth rule): this closeout records the on-disk reality, NOT the harness-claimed state. The task brief asserted "expect all 0 FAIL" + "Verify: 0 FAIL (`p2_tests_pass` closed)" — this is INCORRECT. Wf-420-A1 closed a LOCAL ImportError on `test_f2a_metal_ligand_exchange.py:62` (the SMARTS symbols), but the pre-flight `p2_tests_pass` gate collects the FULL pytest suite and still sees 1 FAILURE — a DIFFERENT ImportError at `test_joint_3stage_mcts.py:95`.

**Closure path for Wf-421+** (1 CPU-only sub-flow recommended):

1. **Wf-421-A1** — fix the `EXPLORATION_STRATEGIES` ImportError at `molmetal/tests/test_joint_3stage_mcts.py:95` (~5 LOC CPU-only). Either:
   - Add `EXPLORATION_STRATEGIES` constant to `molmetal/molmetal_lam/search_alg/proof_search.py` with appropriate value
   - OR rename the import in `test_joint_3stage_mcts.py` to use an actually-exposed symbol
2. **Wf-421-A2** — fix the SyntheMol path bug at `molmetal/molmetal_lam/reactions/beta_reductions.py:1473` (~10 LOC CPU-only). Either:
   - Use `Path(__file__).resolve().parents[N] / 'references/SyntheMol/synthemol/utils.py'` relative path
   - OR mark the `test_f2a_metal_ligand_exchange.py` SyntheMol tests as `@pytest.mark.skip(reason="SyntheMol submodule not installed")` to align with the env gap honest-negative convention
3. **Wf-421-A3** — pre-flight v3 fresh-run verify (~3 min wall CPU-only)
4. **Wf-421-A4** — Wf-299 GPU launch on cuda:0 USER_OK-gated (~5 hr GPU)

**Combined wall**: ~20 min CPU-only for Wf-421-A1 + A2 + A3; then USER_OK-gated ~5 hr GPU for Wf-421-A4.

**Pre-flight cascade baseline preserved at Wf-420-A2 timestamp 2026-09-30T16:12:12+0000 UTC** = **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** wall 193.60 s. Drift from Wf-419-A2 baseline 112/2/3/117: **NO CHANGE** — the Wf-420-A1 SMARTS fix did NOT actually lift any pre-flight gate; the 2 FAILs (`p2_tests_pass` + cascade child) are PRESERVED.

EXPECTED_GATE_COUNTS["all"] **117** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-420-A2 — verification-only sub-flow).

**MEASURED-OUTCOME TABLE** (3 lines):

| Verification step | Expected | MEASURED | Verdict |
|---|---|---|---|
| Pytest on 3 specified files | 0 FAIL | 72 FAIL | NOT closed |
| Pre-flight v3 fresh-run | 0 FAIL / ~113 PASS / 3 SKIP / 117 TOTAL | **2 FAIL / 112 PASS / 3 SKIP / 117 TOTAL** | NOT closed |
| Gate count consistency | actual=117 == expected=117 | **PASS** | OK |
| Wf-299 launcher dry-run | exit 0 + no pre-flight FAIL | **exit 0 + no pre-flight FAIL** | OK |

**Standing directives re-confirmed**: AGENTS.md §1 + §5.1 (standing GPU authorization honored — 0 actual GPU launches) + §5.2 (honest framing — the 2 FAILs are recorded rather than retried or hidden) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §10) + §5.5 (post-wave on-disk verification rule — strictly observed; this closeout records the actual disk state, not the harness-claimed state) + §5.6 (atomic write protocol — strictly observed via `atomic_write_file` for `wf420a2_preflight_freshrun.txt` write) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule — Wf-299 dry-run emitted 16 NEW flags transparently) + §10 (wetlab deprecation).
