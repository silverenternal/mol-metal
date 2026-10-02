# Wf-296E — Pipeline Integration Synthesis

**Date**: 2026-09-29
**Owner**: Hugo Lee
**Upstream**: Wf-296A (audit) + Wf-296B (pre_flight) + Wf-296C (launcher flags) + Wf-296D (matrix + YuelBond)

---

## Per-agent summary

| Agent | Target | Status | Net outcome |
|---|---|---|---|
| Wf-296A | integration audit | COMPLETED | 42 orphans cataloged (10 scripts + 16 flags + 14 tests + 2 catalog reports) |
| Wf-296B | pre_flight_check.py | COMPLETED | 4/14 orphan tests added to p2_tests_pass; preflight 25/25 PASS; 699 tests collected (was 27); 1 deferred (test_f2a_metal_ligand_exchange, 33/309 fail) |
| Wf-296C | G6 launcher | COMPLETED | 3 flags wired (--no-kendall, --no-dann, --no-enforce-budget); 60/60 tuples CONFIRMED (was 30/60 SKIP-WITH-WARN); bash -n + dry-run PASS |
| Wf-296D | CFM train script | SHIPPED | L-R17-B2 84-pattern matrix + 17 NEW SMARTS compiled into RDKit ChemicalReactions and run as post-decode gate; YuelBond precondition surfaced as m14_lead_invalidated JSON field + runtime [WARNING]; smoke VERIFIED |

---

## Remaining orphan artifacts

### 1. Orphan scripts (6 + 3 manual-CLI)

- `wf278a_g4_aggregate.py`
- `wf286b_ecfp4_reeval.py`
- `build_platinai_4cell_csv.py`
- `drugood_ecfp4_scaffold_adapter.py`
- `cfm_property_panel.py`
- `cfm_vina_dock_panel.py`
- `build_platinai_rusanov_5metal.py`
- `build_platinai_4cell.py` (partial — only referenced by `train_platinai_4cell.py`)
- **Manual CLI by design** (not orphans): `abort_recovery.py`, `paper_auto_update.py`, `paper_dashboard.py`

### 2. Orphan flags (13, no launcher consumer)

| Family | Flags |
|---|---|
| DANN hyperparams | `--domain-id`, `--lambda-grl`, `--grl-alpha`, `--num-domains` |
| Chemprop | `--chemprop-min-epochs`, `--chemprop-lomo-init-scale`, `--chemprop-mode v3-fix`, `--chemprop-mode shared-encoder` |
| CFM | `--no-wasserstein`, `--no-closure-bounded`, `--use-tanimoto-ood-filter`, `--ood-filter-threshold`, `--ood-filter-train-pool` |
| Wf-269A Path C | `--metal-probe-init-scale` |

### 3. Orphan tests (10 of 14; Wf-296B covered 4)

- `test_abort_recovery.py`
- `test_cfm_decode_real.py`
- `test_cfm_panels.py`
- `test_drugood_ecfp4_scaffold_adapter.py`
- `test_wave_budget_tracker.py`
- `test_wf260_dataloader_fix.py`
- `test_wf261_scaffold_split.py`
- `test_wf263a_chemprop_v3_fix.py`
- `test_wf264a_drugood_post_fix_updater.py`
- `test_wf264_cfm_property_panel.py`
- `test_wf264_cfm_vina_dock_panel.py`
- `test_wf266_v3_lomo_fix.py`
- `test_wf269a_chemprop_path_c.py`
- `test_f2a_metal_ligand_exchange.py` (33/309 fail; deferred to Wf-297)

### 4. Orphan catalog reports (2)

- `molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json` (Wf-287A)
- `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json` (Wf-253D)

### 5. New test gap from Wf-296D

- `_decode_with_matrix_validation` round-trip is not yet covered by a dedicated test (existing `test_metal_ligand_exchange_matrix.py` asserts matrix constant size + compilation only).

---

## Next priorities

1. **Wf-296E follow-up A** — Add the 10 remaining orphan tests + the Wf-296D round-trip test to `pre_flight_check.py` p2_tests_pass. Cheap, ~600 LOC coverage win, no GPU needed.
2. **Wf-296E follow-up B** — Wire `m14_anchor.json` + `wf253d_m17_drugood_reeval.json` into `paper_dashboard.py` catalog (TODO-126 §7 claim surface). Closes the 2 catalog orphans.
3. **Wf-296F** — Build `scripts/launch_all_reevals.sh` wrapper to surface the 6 orphan reevaluation scripts so future L-R reeval cycles have one entry point.
4. **Wf-297** — Fix `SyntheMolReactionRule.apply()` AttributeError to clear `test_f2a_metal_ligand_exchange` 33 failures. Required for release gate.
5. **Wf-298 (deferred)** — Either prune the 13 orphan flags or wire them into a `configs/wf298_*.yaml` sweep. Low priority (no current launcher consumer).

---

## Honest framing

- Wf-296B/C/D are shipping changes at the script/preflight/launcher layer. No GPU training has been re-executed as part of Wf-296.
- The first MEASURED G6 sweep with the Wf-296C flag wiring will arrive after `bash scripts/launch_wf124_p3_g6.sh` on RTX 5090 GPU-1 (planned for next session, gated on user OK per kill_then_adapt directive).
- Wf-296D smoke run used `--epochs 1 --hidden-dim 64 --n-layers 2 --device cpu --n-train 16 --n-val 8 --batch-size 4`; decode_ratio=1.0 + matrix validation PASS in smoke but full MEASURED claim requires 5-seed × 50-epoch GPU retrain.

---

## Cross-references

- Wf-269C — YuelBond precondition MEASURED Δ = −1.000
- Wf-275C — m14 LEAD deployment precondition dashboard claim
- TODO-119 — decode_ratio strengthen (primary Wf-296D target)
- TODO-126 §7 — engine tag honest reporting
- TODO-182 §3.4 — per-lever attribution matrix (now possible across 4 ablation levers post-Wf-296C)
- `molmetal/reports/wf_pitfall_reinforce/final.md` §93-103 (matrix ≥ 84 entries assertion)
- `molmetal/tests/test_metal_ligand_exchange_matrix.py` (659 tests passing)