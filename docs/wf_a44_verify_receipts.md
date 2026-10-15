# Wf-A44 Verify-Receipts — 15 M-CODE-FIX candidates + J1 fix on-disk audit

**Wave:** Wf-A44 (2026-10-15)
**Auditor:** closeout agent (post-design-defaults execution)
**Source brief:** `[scan-fresh]` triage + `[design-defaults]` brief input

Per AGENTS.md §5.5 filesystem-truth rule, every "SHIPPED" claim in this audit
back-traces to a file-on-disk + a `grep` / `pytest` verification below. All
receipts are MEASURED on 2026-10-15 against master HEAD `0eaa8896`.

---

## # 15 MEDIUM candidates — receipts

| §id | file:line | on-disk receipt | verification |
|-----|-----------|-----------------|--------------|
| M1 §6.69a | `molmetal/eval/scoring/stereo_postprocess.py:32-37` | `_ALLOWED_STEREO_MODES = frozenset({"noop", "remove_atom"})` present | `grep -n "_ALLOWED_STEREO_MODES" molmetal/eval/scoring/stereo_postprocess.py` returns L32; collapse at Wf-A43 commit `b3482fc9` |
| M2 §6.69b | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:110-178 + 220/285/364/513/572` | `_has_metal(mol)` calls at 5 ADMET proxies (logp7_4/gi50_proxy/herg_cardio_risk/aqueous_solubility_logS/plasma_protein_binding) | `grep -n "_has_metal" molmetal/molmetal_lam/sbdd_env/metrics_v2.py` returns 6 hits (line 110 def + 5 call sites) |
| M3 §6.69c | `molmetal/scripts/wf_a40_pivot_long_to_wide.py:1-205` | NEW script, 6639 bytes, mtime 2026-10-15 16:36 | `ls -la molmetal/scripts/wf_a40_pivot_long_to_wide.py` → 6639 B; per §6.69c body: `python molmetal/scripts/wf_a40_pivot_long_to_wide.py --self-test` exit 0 |
| M4 §6.69d | `molmetal/loss/distillation_losses.py:34` | `shortcut_distill_term.weight: float = 0.05` + inner-block fix | `grep -n "0.05" molmetal/loss/distillation_losses.py` returns L34 |
| M5 §6.69e | `molmetal/loss/aggregation_losses.py:135` | `soft_valence_term.weight: float = 0.05` | `grep -n "0.05" molmetal/loss/aggregation_losses.py` returns L135 |
| M6 §6.69f | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:8261 (CLI) + 3291 (fn)` | `fragment_loss_weight=0.05` default | `grep -n "fragment_loss_weight=0.05" molmetal/scripts/train_cfm_yuelbond_wasserstein.py` returns L3291 |
| M7 §6.69g | `molmetal/tests/test_bond_aromatic_lo_invariance.py` | NEW pytest, 5819 B | `pytest molmetal/tests/test_bond_aromatic_lo_invariance.py` → 3/3 PASS |
| M8 §6.69h | `molmetal/loss/regularization_losses.py:85` + `train_cfm_yuelbond_wasserstein.py:7918 (CLI) + 3235 (fn)` | `entropy_term.weight=0.1` default | `grep -n "entropy_term_weight=0.1" molmetal/scripts/train_cfm_yuelbond_wasserstein.py` returns L3235 |
| M9 §6.69i | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:3141` | `consistency_term_weight: float = 0.05` | `grep -n "consistency_term_weight: float" molmetal/scripts/train_cfm_yuelbond_wasserstein.py` returns L3141 |
| M10 §6.69j | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:8271 (CLI) + 3296 (fn)` | `molecular_property_target_weight=0.05` default | `grep -n "molecular_property_target_weight=0.05" molmetal/scripts/train_cfm_yuelbond_wasserstein.py` returns L3296 |
| M11 §6.69k | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:513-530` | `_has_metal(mol)` logS amend at L513 | `sed -n '511,531p' molmetal/molmetal_lam/sbdd_env/metrics_v2.py` shows the wire-in |
| M12 §6.68a | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:800` | `DEFAULT_EARLY_STOP_PATIENCE = 10` | `grep -n "DEFAULT_EARLY_STOP_PATIENCE = 10" molmetal/scripts/train_cfm_yuelbond_wasserstein.py` returns L800 |
| M13 §6.66c | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:tmqm_min_per_metal` | dest wired + plumbed through load_dataset | `grep -n "tmqm_min_per_metal" molmetal/scripts/train_cfm_yuelbond_wasserstein.py` returns 4 hits (L7714 dest + L9716 read + L9725 + L9736 plumb) |
| M14 §6.66d | `molmetal/data/tmqm.py:78-110` | TRANSITION_METALS = 30 metals | `grep -n "TRANSITION_METALS" molmetal/data/tmqm.py` returns 3 hits (L78 + L110 + L534) |
| M15 §6.66f | `molmetal/scripts/fpocket_wrapper.py:doc` + `molmetal/tests/test_fpocket_path.py` | PATH doc + test SHIPPED | `ls -la molmetal/scripts/fpocket_wrapper.py molmetal/tests/test_fpocket_path.py` |

## # M2 PARTIAL residual — §6.71a/b receipts

| §id | file:line | on-disk receipt | verification |
|-----|-----------|-----------------|--------------|
| §6.71a | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:567-573` | `plasma_protein_binding` metal-aware amend +5 LOC | `pytest -k ppb_cisplatin` → `val ≈ 0.228 ±0.01` (Δ=+0.008 Pt amend) |
| §6.71b | `molmetal/tests/test_metrics_v2.py:215-247` | `test_ppb_cisplatin_has_metal_amend` + `test_ppb_organic_no_amend` (2 NEW tests) | `pytest -k ppb -v` → 4/4 PASS (2 pre-existing + 2 NEW) |

## # 19 JUDGEMENT-CALL receipts — §6.71 ledger

All 19 J-doc ledger rows recorded verbatim in `docs/HONEST_NEGATIVE_CATALOG.md`
§6.71 ledger table. Per the §6.71 entry, 18 are SHIPPED-DECISION (locked on disk
via prior commits) + 1 is DEFERRED-TO-Wf-A45 (J19 PoC pilot on 3 skills before
28-skill fan-out).

## # J19 PoC — 3-skill pilot receipts

| §id | file | on-disk receipt | verification |
|-----|------|-----------------|--------------|
| J19-pilot | `molmetal/tests/test_skill_r18_math_examples.py` (NEW, 7473 B, mtime 2026-10-15 17:22) | 15 tests | `pytest test_skill_r18_math_examples.py -v` → 15/15 PASS |
| J19-pilot | `molmetal/tests/test_skill_pareto_multiobj_examples.py` (NEW, 7925 B, mtime 2026-10-15 17:24) | 15 tests | `pytest test_skill_pareto_multiobj_examples.py -v` → 15/15 PASS |
| J19-pilot | `molmetal/tests/test_skill_neural_ode_kinetics_examples.py` (NEW, 7305 B, mtime 2026-10-15 17:23) | 10 tests | `pytest test_skill_neural_ode_kinetics_examples.py -v` → 6 PASS + 4 SKIP (4 PROJECTED-boundary correctly skipped) |

## # §I.2 assemble_term regression — §6.71f receipts

| §id | file | on-disk receipt | verification |
|-----|------|-----------------|--------------|
| §6.71f | `molmetal/tests/test_assemble_term_arithmetic.py` (NEW, 3 tests) | arithmetic_mean no-underflow + product-formulation-documented-underflow + hinge_tau default | `pytest test_assemble_term_arithmetic.py -v` → 3/3 PASS |

## # J1 test fix + stereo docstring — §6.71e receipts

| §id | file:line | on-disk receipt | verification |
|-----|-----------|-----------------|--------------|
| §6.71e J1 fix | `molmetal/tests/test_sascorer_fork_divergence.py:403` | `for vacuous in ("noop",):` (post-Wf-A43 §6.69a collapse alignment) | `pytest test_sascorer_fork_divergence.py::test_opt1_direction_reverses_under_the_canonical_scorer` → PASS (was FAIL pre-fix) |
| §6.71e docstring | `molmetal/eval/scoring/stereo_postprocess.py:49-56` | cross-ref to §I.1 + §6.66e JUDGEMENT-CALL added | `grep -n "§6.66e" molmetal/eval/scoring/stereo_postprocess.py` returns L54 |

## # 13 trainer-flag default flips — §6.71c receipts

All 13 flips have `# Wf-A44 — default ON` marker comments in
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`:

| flag | line | old → new |
|------|------|-----------|
| `edm_term_weight` | 3143 | 0.0 → 0.05 |
| `shortcut_distill_term_weight` | 3144 | 0.0 → 0.05 |
| `fm_ode_term_weight` | 3145 | 0.0 → 0.05 |
| `energy_conservation_weight` | 3146 | 0.0 → 0.05 |
| `sboo3_term_weight` | 3147 | 0.0 → 0.05 |
| `chem_loss_weight` | 3283 | 0.0 → 0.05 |
| `qed_loss_weight` | 3284 | 0.0 → 0.05 |
| `molecule_loss_weight` | 3285 | 0.0 → 0.05 |
| `stereo_loss_weight` | 3286 | 0.0 → 0.05 |
| `--chem-loss-weight` | 8187 | 0.0 → 0.05 |
| `--qed-loss-weight` | 8234 | 0.0 → 0.05 |
| `--molecule-loss-weight` | 8261 | 0.0 → 0.05 |
| `--stereo-loss-weight` | 8287 | 0.0 → 0.05 |

Verification: `git grep -c "Wf-A44" molmetal/scripts/train_cfm_yuelbond_wasserstein.py` → 13 hits.

## # 20 yaml amendments — §6.71d receipts

3 real config flips (random → scaffold, C13 fix):
- `configs/cfm_fullcorpus_2026_10_10.yaml` (real flip)
- `configs/cfm_fullcorpus_b64_2026_10_11.yaml` (real flip)
- `configs/cfm_fullcorpus_smiles_persist.yaml` (real flip)

17 docs-only LEGACY OVERRIDE comments (no value change, just comment amendment):
- All 17 ablation/experiment yamls (BLK-C5 / bond head / entropy reg / FM ab /
  FM control / FM distort / frag assemble / frag control / gen collapse FM /
  objective-terms / ODE link probe / pad groups / PCGrad OFF / PCGrad ON /
  per-innovation ablation / real-decode lift / TMQM all6)

Verification: `git grep -l "tmqm_split_kind: scaffold" configs/` → 3 hits; `git grep -l "LEGACY OVERRIDE" configs/` → 17 hits.

---

**Total SHIPPED-on-disk fixes (Wf-A44 + prior waves this audit covers):**
15 MEDIUM + 1 §6.71a PARTIAL + 2 §6.71b regression tests + 1 §6.71f regression test + 13 §6.71c trainer flips + 20 §6.71d yaml amendments + 1 §6.71e J1 test fix + 1 §6.71e docstring + 3 §6.71g J19 PoC files + 19 J-doc ledger rows = 76 LOC/audit items, all on disk at 2026-10-15.

**Pre-flight:** PRESERVED at Wf-A43 baseline (110 PASS / 3 FAIL / 10 SKIP / 124 total) per AGENTS.md §1.4 escape hatch + memory `wfa33-v2-precommit-timeout.md`. 0 NEW pre-flight gates introduced. 0 GPU launches per 2026-10-15 user directive verbatim *"gpu先别动"*. 0 paper/* touches per §5.3 + TODO-84. 0 wet-lab per §5.4 + §10.
