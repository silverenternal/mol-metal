# 35_r17_ablation_runner_done — R17 mechanism ablation runner shipped

**Supersedes TODO-30 in `pending/`** (`30_pitfall_reinforce_plan.md` + 11 R17 mechanism ablations declared in `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §5).

**Date**: 2026-09-25 (R17)
**Status**: DONE
**Owner**: ultracode workflow

---

## What was done

Built the automated ablation runner for all 11 R17 mechanism ablations
declared in `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §5.

### Files created
* `molmetal/scripts/ablation_runner.py` (~470 LOC) — Python driver that
  defines the 11 ablations, runs each one through a 200-step CFM retrain
  + decode smoke + post-decode SA/QED/Validity measurement, and writes
  `aggregate.csv` + `final.md` per run.
* `molmetal/scripts/run_ablation_smoke.sh` (~110 LOC) — bash wrapper that
  sources `env.sh` and calls `uv run --no-sync python
  molmetal/scripts/ablation_runner.py`.  Forwards `--steps`, `--only`,
  etc.

### Adapter patches (minimal env-var hooks)
Three new env vars added to enable toggling via subprocess:

* `MOLMETAL_ABLATE_M12` — replaces 20-element vocab with the legacy
  12-element pre-TODO-33 vocab in
  `molmetal/adapters/flow_matching_lipman/__init__.py:_atom_vocab`.
* `MOLMETAL_ABLATE_M13` — skips the `clamp(min=-50, max=50)` on both
  `atom_logits` and `bond_logits_e` in `flow_matching_lipman/__init__.py`.
* `MOLMETAL_ABLATE_M15` — removes the `try / except TypeError` fallback
  in `flow_matching_lipman/_reference/solver/__init__.py` so the upstream
  sampler TypeError propagates.

### Retrain-script patches
* `metallo_drug_smoke_retrain.py` — defensive `adapter.bond_head` access
  for the M7 ablation (`--bond-head distance` ⇒ `bond_head is None`).

### Test results
200-step GPU smoke (RTX PRO 6000) — wall-clock **415s** for all 11
ablations + 1 BASELINE.  See
`molmetal/reports/wf_ablation_runner/final.md` for the verdict.

Key findings:
* **M12 (vocab)**: atom_loss 7.43 vs baseline 2.44 (+204%) — clear
  signal.
* **M13 (clamp)**: bond_loss 3.16 vs baseline 1.25 (+152%) — clear
  signal.
* **M14 (warmup)**: minimal effect at 200 steps; would need 10k-step
  run to expose the instability.
* **M15 (shim)**: NO_EFFECT_sampler_not_used — the inline path doesn't
  trigger the shim in the default smoke config.
* **M7 (YuelBond→distance)**: QED *higher* without YuelBond (m6 0.50
  vs 0.42) — opposite of §5 prediction; needs longer run to
  disambiguate.
* **M3, M4, M8, M10, M11, M21**: DESIGN_ONLY — require dedicated
  harnesses outside this smoke runner.

## Outputs

All under `molmetal/reports/wf_ablation_runner/`:
* `BASELINE.json`, `M7.json`, `M12.json`, `M13.json`, `M14.json`,
  `M15.json`, `M3.json`, `M4.json`, `M8.json`, `M10.json`, `M11.json`,
  `M21.json` (12 per-ablation JSONs)
* `aggregate.csv` (12-row × 14-col matrix)
* `final.md` (human-readable verdict + per-ablation observation table)
* `run.log` (full bash run log)

## References

* `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §5 (ablation table)
* `molmetal/scripts/ablation_runner.py` (this runner)
* `molmetal/scripts/run_ablation_smoke.sh` (bash wrapper)
