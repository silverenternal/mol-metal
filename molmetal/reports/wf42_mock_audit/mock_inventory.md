# Wf-42 Phase 1 — Mock / Stub / Placeholder audit

Generated UTC: 2026-09-26T05:41:59+00:00

## TL;DR

* **Total findings**: 129
* **Critical**: 8
* **Major**: 23
* **Minor**: 98

Counts by file kind:

| File kind | Findings |
|---|---:|
| `metric_json` | 6 |
| `report` | 96 |
| `script` | 27 |

Counts by pattern kind:

| Kind | Count |
|---|---:|
| `stub_module_name` | 50 |
| `protocol_stub_adapter` | 21 |
| `stub_identifier` | 13 |
| `use_real_vina_false` | 10 |
| `placeholder_text` | 9 |
| `stub_class` | 6 |
| `stub_scorer` | 6 |
| `placeholder` | 5 |
| `MEASURED_BUT_STUB` | 3 |
| `stub_bridge` | 2 |
| `stub_floor` | 2 |
| `mock_chemistry` | 1 |
| `stub_fn` | 1 |

Counts by metric:

| Metric | Count |
|---|---:|
| `n/a (module name)` | 50 |
| `external_sota_adapter` | 21 |
| `m3_vina_dock` | 21 |
| `unknown` | 20 |
| `n/a (paper template)` | 14 |
| `coupling_bridge_rmse` | 2 |
| `mcts_trajectory` | 1 |

---

## Critical findings (must fix to ship)

### `vina_kcal_per_mol.json:37` — `stub_scorer` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/metrics/by_metric/vina_kcal_per_mol.json`

```
L36:      "n_pockets": 10,
L37:      "scorer": "stub",
L38:      "use_real_vina": false,
```

**Remediation**: Real Vina path is required; route through VinaDockingAdapter after fixing the metal-PDBQT issue (meeko vs Pt/Au/Ru).

### `vina_kcal_per_mol.json:39` — `MEASURED_BUT_STUB` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/metrics/by_metric/vina_kcal_per_mol.json`

```
L38:      "use_real_vina": false,
L39:      "status": "MEASURED_BUT_STUB",
L40:      "date_utc": "2026-09-26",
```

**Remediation**: Replace the stub scorer with the real Vina / real pIC50 / real SA scorer and rerun the affected workflow phase.

### `vina_kcal_per_mol.json:58` — `MEASURED_BUT_STUB` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/metrics/by_metric/vina_kcal_per_mol.json`

```
L57:  },
L58:  "verdict": "MEASURED_BUT_STUB_DOMINATED — wf30_phase4 m3 median formally passes the ship gate (≤ -8.5 kcal/mol) at -9.6574 kcal/mol, but both methods produce identical per-pocket best to floating-point noise (lift ~1e-8 kcal/mol). This is a stub-scorer property, not a method-level result. The one real-Vina measurement (wf_pb_pass_real_dock, -6.929 kcal/mol single point on 830c) is the only meaningful data point; gap to TargetDiff CrossDocked lit -8.45 remains ~1.5 kcal/mol. Cannot claim Vina lift at paper scale. Path (b) decoder rework + meeko extension for Pt/Au/Ru + n_simulations>=1000 needed for paper-grade comparison."
L59:}
```

**Remediation**: Replace the stub scorer with the real Vina / real pIC50 / real SA scorer and rerun the affected workflow phase.

### `best_m3_summary.json:4` — `stub_scorer` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/best_m3_summary.json`

```
L3:  "generated_utc": "2026-09-26T04:47:25+00:00",
L4:  "scorer": "stub",
L5:  "use_real_vina": false,
```

**Remediation**: Real Vina path is required; route through VinaDockingAdapter after fixing the metal-PDBQT issue (meeko vs Pt/Au/Ru).

### `m3_lift_analysis.json:5` — `stub_scorer` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/m3_lift_analysis.json`

```
L4:  "unit": "kcal/mol",
L5:  "scorer": "stub",
L6:  "use_real_vina": false,
```

**Remediation**: Real Vina path is required; route through VinaDockingAdapter after fixing the metal-PDBQT issue (meeko vs Pt/Au/Ru).

### `m3_lift_analysis.json:95` — `MEASURED_BUT_STUB` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/m3_lift_analysis.json`

```
L94:  },
L95:  "verdict": "MEASURED_BUT_STUB \u2014 both methods yielded identical per-pocket best to floating-point noise; no quantitative lift. Ship gate (\u2264 -8.5 kcal/mol) is technically met by both methods because the stub floor is -9.66 kcal/mol, but this is a property of the stub, not of the methods. Reported honestly as a Phase-3 negative.",
L96:  "sources": {
```

**Remediation**: Replace the stub scorer with the real Vina / real pIC50 / real SA scorer and rerun the affected workflow phase.

### `per_pocket_metrics.json:512` — `stub_scorer` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/per_pocket_metrics.json`

```
L511:  "use_real_vina": false,
L512:  "scorer": "stub"
L513:}
```

**Remediation**: Real Vina path is required; route through VinaDockingAdapter after fixing the metal-PDBQT issue (meeko vs Pt/Au/Ru).

### `wf30_dock_summarize.py:177` — `stub_scorer` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/wf30_dock_summarize.py`

```
L176:        ),
L177:        "scorer": "stub" if not (
L178:            se3sb_metrics.get("use_real_vina", False)
```

**Remediation**: Real Vina path is required; route through VinaDockingAdapter after fixing the metal-PDBQT issue (meeko vs Pt/Au/Ru).

---

## Major findings

### `vina_kcal_per_mol.json:38` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/metrics/by_metric/vina_kcal_per_mol.json`

```
L37:      "scorer": "stub",
L38:      "use_real_vina": false,
L39:      "status": "MEASURED_BUT_STUB",
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `vina_kcal_per_mol.json:41` — `stub_floor` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/metrics/by_metric/vina_kcal_per_mol.json`

```
L40:      "date_utc": "2026-09-26",
L41:      "note": "Phase 4 m3 median: se3sb and baseline agree to floating-point noise (~1e-8 kcal/mol lift); stub scorer's deterministic ±2.0 kcal/mol jitter bottoms out at the same -9.66 kcal/mol across all 10 pockets. No quantitative method lift can be claimed from this sweep. Ship gate (≤ -8.5 kcal/mol) is technically met by both methods because the stub floor is -9.66 kcal/mol, but this is a property of the stub, not of the methods. See molmetal/reports/wf30_dock/m3_lift_summary.md and notes.md for the honest-negative rationale. Real-Vina path blocked by meeko rejecting Pt/Au/Ru atoms in the demo metal-drug cohort.",
L42:      "report_json": "molmetal/reports/wf30_dock/m3_lift_analysis.json",
```

**Remediation**: Deterministic stub property — replace with real scorer output.

### `wf30_dock_verify.json:6` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/metrics/by_metric/wf30_dock_verify.json`

```
L5:  "candidate": "CCO",
L6:  "use_real_vina": false,
L7:  "per_evaluation": [
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `metrics.json:49` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/baseline/metrics.json`

```
L48:  "n_steps": 8,
L49:  "use_real_vina": false,
L50:  "pocket_dir": "/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep",
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `summary.json:100` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/baseline/summary.json`

```
L99:  "win_tie_loss_vs_baseline": {},
L100:  "use_real_vina": false
L101:}
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `best_m3_summary.json:5` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/best_m3_summary.json`

```
L4:  "scorer": "stub",
L5:  "use_real_vina": false,
L6:  "pockets": [
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `m3_lift_analysis.json:6` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/m3_lift_analysis.json`

```
L5:  "scorer": "stub",
L6:  "use_real_vina": false,
L7:  "n_pockets": 10,
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `m3_lift_summary.md:10` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/m3_lift_summary.md`

```
L9:- **Lift (se3sb − baseline)**: **+1.2963e-08 kcal/mol** (≈ 0; floating-point noise)
L10:- **Stub scorer?** Yes (`scorer=stub`, `use_real_vina=false`). The Phase-3 sweep on the demo metal-drug cohort could not exercise real Vina because meeko rejects Pt/Au/Ru atoms during PDBQT conversion. See `notes.md` for the honest-negative rationale.
L11:- **Ship gate (≤ -8.5 kcal/mol)**: stub-pass (se3sb median -9.6574 < -8.5), but this is a property of the stub floor, not a real-method result. **Reported as honest negative**: no quantitative lift can be claimed.
```

**Remediation**: Real Vina path is required.

### `m3_lift_summary.md:11` — `stub_floor` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/m3_lift_summary.md`

```
L10:- **Stub scorer?** Yes (`scorer=stub`, `use_real_vina=false`). The Phase-3 sweep on the demo metal-drug cohort could not exercise real Vina because meeko rejects Pt/Au/Ru atoms during PDBQT conversion. See `notes.md` for the honest-negative rationale.
L11:- **Ship gate (≤ -8.5 kcal/mol)**: stub-pass (se3sb median -9.6574 < -8.5), but this is a property of the stub floor, not a real-method result. **Reported as honest negative**: no quantitative lift can be claimed.
L12:
```

**Remediation**: Deterministic stub property — replace with real scorer output.

### `notes.md:102` — `stub_scorer` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/notes.md`

```
L101:value (the -2.0 lower bound) for every pocket. The summary JSON is
L102:written truthfully; the `scorer="stub"` field in
L103:`per_pocket_metrics.json` and `best_m3_summary.json` makes the limitation
```

**Remediation**: Replace the stub scorer with the real backend and rerun.

### `per_pocket_metrics.json:511` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/per_pocket_metrics.json`

```
L510:  },
L511:  "use_real_vina": false,
L512:  "scorer": "stub"
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `metrics.json:49` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/se3sb/metrics.json`

```
L48:  "n_steps": 8,
L49:  "use_real_vina": false,
L50:  "pocket_dir": "/home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep",
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `summary.json:100` — `use_real_vina_false` (m3_vina_dock)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf30_dock/se3sb/summary.json`

```
L99:  "win_tie_loss_vs_baseline": {},
L100:  "use_real_vina": false
L101:}
```

**Remediation**: Real Vina path is required (after meeko-vs-metal fix or organic-cohort re-pivot).

### `phase3_integration.md:141` — `stub_class` (unknown)

File: `/home/hugo/codes/mol-metal/molmetal/reports/wf_deflex_lambda_combinators/phase3_integration.md`

```
L140:
L141:class MockRule:
L142:    def __init__(self, name, applies, score, apply_fn):
```

**Remediation**: Stub class — replace with the real backend implementation.

### `bench_coupling_ic50.py:812` — `stub_bridge` (coupling_bridge_rmse)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/bench_coupling_ic50.py`

```
L811:            f"[wf39_coupling] bridge NPZ not found: {npz_path} "
L812:            f"(stub bridge will be used)",
L813:            file=sys.stderr,
```

**Remediation**: Stub bridge — load the trained checkpoint at models/coupling_adapter_platinai5k.npz (or analogous).

### `bench_lambda_cfm_e2e.py:514` — `stub_bridge` (coupling_bridge_rmse)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/bench_lambda_cfm_e2e.py`

```
L513:    # the forward pipeline.  We seed the Python RNG here so any
L514:    # downstream RNG-driven sampler (e.g. stub bridge weights) sees
L515:    # a stable stream.
```

**Remediation**: Stub bridge — load the trained checkpoint at models/coupling_adapter_platinai5k.npz (or analogous).

### `benchmark_r18_end_to_end.py:516` — `stub_class` (unknown)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/benchmark_r18_end_to_end.py`

```
L515:
L516:        class _StubVNet(nn.Module):
L517:            def __init__(self) -> None:
```

**Remediation**: Stub class — replace with the real backend implementation.

### `benchmark_r18_platinai.py:302` — `stub_class` (unknown)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/benchmark_r18_platinai.py`

```
L301:        # small wrapper that accepts and ignores the extra kwargs.
L302:        class _StubVNet(nn.Module):
L303:            def __init__(self) -> None:
```

**Remediation**: Stub class — replace with the real backend implementation.

### `dock_10pockets_se3sb.py:506` — `stub_fn` (unknown)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/dock_10pockets_se3sb.py`

```
L505:
L506:def _stub_score(
L507:    init_coords: torch.Tensor,
```

**Remediation**: Stub function — replace with the real backend (Vina / cross_dock / pIC50 model / SA / QED).

### `verify_fg_compat_veto.py:55` — `stub_class` (unknown)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_fg_compat_veto.py`

```
L54:
L55:class _MockReducer:
L56:    """Minimal reducer exposing ``reduce(input) -> str``.
```

**Remediation**: Stub class — replace with the real backend implementation.

### `verify_mcts_replay.py:37` — `stub_class` (unknown)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_mcts_replay.py`

```
L36:
L37:    class _StubMol:
L38:        def __init__(self, smiles):
```

**Remediation**: Stub class — replace with the real backend implementation.

### `verify_mcts_replay.py:59` — `stub_class` (unknown)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_mcts_replay.py`

```
L58:
L59:    class _StubSearch:
L60:        history = [{
```

**Remediation**: Stub class — replace with the real backend implementation.

### `verify_variance_mcts_proof_search.py:11` — `mock_chemistry` (mcts_trajectory)

File: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_variance_mcts_proof_search.py`

```
L10:
L11:The script runs 100 trajectories on the same mock chemistry
L12:(identity concatenation rule, 4 tile SMILES, scorer ∝ n_atoms)
```

**Remediation**: Mock chemistry — verify scripts are unit-level by design; flag as major so the bench/benchmark report line is visible.

---

## Minor findings (informational)

| File:Line | Kind | Metric | Match |
|---|---|---|---|
| `clone_integration_adapters.md:36` | `stub_identifier` | `unknown` | `- `is_stub()` — `True` iff no real FlowDock import` |
| `clone_integration_adapters.md:42` | `stub_identifier` | `unknown` | `- `name`, `setup`, `is_stub`` |
| `clone_integration_adapters.md:57` | `stub_identifier` | `unknown` | `- `is_stub()` — always True` |
| `clone_integration_adapters.md:101` | `stub_identifier` | `unknown` | `\| FlowDockReferenceAdapter \| `DockingEngine` \| Protocol stub: `setup/is_stub/dock/get_metadata`,模块级 `is_flowdock_avai` |
| `clone_integration_adapters.md:102` | `protocol_stub_adapter` | `external_sota_adapter` | `\| FLOWRReferenceAdapter \| `MoleculeGenerator` \| Protocol stub: `generate/train_step/get_metadata`,模块级 `is_flowr_avail` |
| `clone_integration_adapters.md:103` | `protocol_stub_adapter` | `external_sota_adapter` | `\| Pybind11StubAdapter \| `PybindBindingProtocol` \| Protocol stub: `forward(*args, **kwargs) -> Any`,`call_log` 记录调用;返回` |
| `clone_integration_adapters.md:104` | `protocol_stub_adapter` | `external_sota_adapter` | `\| REINVENT4Adapter \| `ScoringFunction` \| Protocol stub: `score -> List[ScoredCandidate]`(ranked 0..N-1),`discover_rei` |
| `diffsbdd_targetdiff_1h36.md:48` | `protocol_stub_adapter` | `external_sota_adapter` | `Investigation + adapter stub + two reports: ~25 min.` |
| `diffsbdd_targetdiff_no_ckpt.md:53` | `protocol_stub_adapter` | `external_sota_adapter` | `- `molmetal/molmetal_lam/sbdd_env/targetdiff_adapter.py` — a *deferred* / stub adapter that documents the exact interfac` |
| `h5_lambda_honest_framing.md:77` | `protocol_stub_adapter` | `external_sota_adapter` | `Our `Vina (kcal/mol)` column for Lambda is `NOT REPORTED`. We do not run AutoDock Vina in this environment for every Lam` |
| `h5_lambda_honest_framing.md:175` | `protocol_stub_adapter` | `external_sota_adapter` | `As noted in §2.4, our Vina column is `NOT REPORTED`. The Vina adapter stub at `molmetal/molmetal_lam/sbdd_env/vina_adapt` |
| `h5_lambda_honest_framing.md:219` | `protocol_stub_adapter` | `external_sota_adapter` | `4. **No measured Vina.** AutoDock Vina has not been run on Lambda candidates in this paper. The Vina adapter stub is at ` |
| `h5_lambda_honest_framing.md:265` | `protocol_stub_adapter` | `external_sota_adapter` | `* `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` — Vina adapter stub.` |
| `r4_c_pilot.md:27` | `placeholder_text` | `n/a (paper template)` | `*Vina proxy is a placeholder until L-1 DiffDock/FlowDock binary is installed.*` |
| `r4_c_pilot_1k.md:27` | `placeholder_text` | `n/a (paper template)` | `*Vina proxy is a placeholder until L-1 DiffDock/FlowDock binary is installed.*` |
| `r4_c_test.md:27` | `placeholder_text` | `n/a (paper template)` | `*Vina proxy is a placeholder until L-1 DiffDock/FlowDock binary is installed.*` |
| `round9_final_report.md:175` | `placeholder_text` | `n/a (paper template)` | `\| 7 \| Docking engine differences (Vina 1.2.7 vs QVina/QVina2) \| **OPEN**. Vina adapter accepts `--engine {auto, vina,` |
| `round9_r4c_pilot_results.md:108` | `placeholder_text` | `n/a (paper template)` | `  placeholder until DiffDock/FlowDock integration lands.` |
| `agent_d_model.md:196` | `protocol_stub_adapter` | `external_sota_adapter` | `**Missing / skeleton ports not used by r4_c**: real DiffDock, real EquiBind (Phase-2 deferred), EGNN pIC50 predictor (st` |
| `wf_wire_clone_scoring_audit.md:29` | `stub_identifier` | `unknown` | `\| 15 \| `reinvent4_adapter.py` \| `REINVENT4Adapter.score` \| **STUB** (always returns RDKit-fallback scores; `is_stub(` |
| `phase3j_warm_start.md:87` | `placeholder_text` | `n/a (paper template)` | `are zeros (the placeholder for the learned MLP head).  Following` |
| `final.md:131` | `protocol_stub_adapter` | `external_sota_adapter` | `\| 9 \| `test_learned_policy_prior_with_real_adapter_emits_coupling_bias` \| **regression sentinel**: integration throug` |
| `final.md:204` | `protocol_stub_adapter` | `external_sota_adapter` | `  adapter stub verified: `_coupling_bias` is no longer `None`` |
| `results.json:39` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub",` |
| `results.json:1869` | `stub_module_name` | `n/a (module name)` | `    "lam_chem/md_relaxation_stub": {` |
| `wf_data_gap_targetdiff_paper.md:205` | `protocol_stub_adapter` | `external_sota_adapter` | `- `molmetal/reports/diffsbdd_targetdiff_1h36.md` (2026-09-11): placeholder log + adapter stub (`targetdiff_adapter.py`).` |
| `inventory.json:127` | `stub_module_name` | `n/a (module name)` | `      "module": "molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py",` |
| `inventory.json:582` | `stub_module_name` | `n/a (module name)` | `      "path": "molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py",` |
| `inventory.md:138` | `stub_module_name` | `n/a (module name)` | `\| 820 \| `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` \| R18 W3 mechanism \|` |
| `inventory.md:259` | `stub_module_name` | `n/a (module name)` | `\| **M37** \| MD relaxation stub \| `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` \| 820 \| 24 \|` |
| `import_check.json:43` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub",` |
| `import_check.json:89` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": true,` |
| `import_check.json:307` | `stub_module_name` | `n/a (module name)` | `      "module": "molmetal.molmetal_lam.lam_chem.md_relaxation_stub",` |
| `import_check.json:308` | `stub_module_name` | `n/a (module name)` | `      "logical_id": "lam_chem/md_relaxation_stub",` |
| `import_check.json:311` | `stub_module_name` | `n/a (module name)` | `      "file": "/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py",` |
| `paper_inventory.md:29` | `placeholder` | `n/a (paper template)` | `\| 3 \| `sections/03_method.tex` \| §3 Method (placeholder assembled by WF-Lambda-3) \|` |
| `test_inventory.json:212` | `stub_module_name` | `n/a (module name)` | `        "test_md_relaxation_stub.py",` |
| `test_inventory.json:1647` | `stub_module_name` | `n/a (module name)` | `      "file": "molmetal/tests/test_md_relaxation_stub.py",` |
| `test_inventory.json:1648` | `stub_module_name` | `n/a (module name)` | `      "stem": "md_relaxation_stub",` |
| `test_inventory.md:182` | `stub_module_name` | `n/a (module name)` | `\| `molmetal/tests/test_md_relaxation_stub.py` \| R18 ship \| r18_lam_chem \| 24 \| 0 \| 0 \| 0 \| 0 \|` |
| `phase2_adapter.md:32` | `protocol_stub_adapter` | `external_sota_adapter` | `  Falls back to a deterministic stub adapter when the .npz is` |
| `phase2_adapter.md:53` | `protocol_stub_adapter` | `external_sota_adapter` | `* The stub adapter's ``embed_pocket(None, pocket_name="X")``` |
| `phase2_adapter.md:60` | `stub_identifier` | `unknown` | `  ``(64,)``, every entry finite, ``is_stub=False`` when a real` |
| `results.json:50` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub",` |
| `results.json:99` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 236.74405299243517,` |
| `results.json:133` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 249.98295650584623,` |
| `results.json:167` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 262.84927799133584,` |
| `results.json:201` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 313.4015559917316,` |
| `results.json:235` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 217.7996820246335,` |
| `results.json:271` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 0.6921987533569336,` |
| `results.json:305` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 0.09807586669921875,` |
| `results.json:339` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 0.10111045837402344,` |
| `results.json:373` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 0.08716106414794922,` |
| `results.json:407` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 0.08804893493652344,` |
| `results.json:443` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 10,` |
| `results.json:477` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 10,` |
| `results.json:511` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 10,` |
| `results.json:545` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 10,` |
| `results.json:579` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": 10,` |
| `results.json:615` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": "ok",` |
| `results.json:649` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": "ok",` |
| `results.json:683` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": "ok",` |
| `results.json:717` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": "ok",` |
| `results.json:751` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub": "ok",` |
| `final.md:95` | `protocol_stub_adapter` | `external_sota_adapter` | `\| §3.4 SE(3)-Equivariant CFM \| §3.6 CFM-stub + §3.7 bond-prior \| Pre-existing deferred stub + Path B soft prior \|` |
| `final.md:112` | `placeholder_text` | `n/a (paper template)` | `The 5-metal multi-task model has heads for Ru / Ir / Rh / Re / Os. The Stage 1.4 recipe reports `test_auc_per_metal` (on` |
| `final.md:227` | `placeholder_text` | `n/a (paper template)` | `If Stage 2.4 still produces A2780cis < 0.65, the BLOCK verdict persists and Stage 2.6 ablation is required (§6). The pap` |
| `final_phase2_s36.md:1` | `protocol_stub_adapter` | `external_sota_adapter` | `WF-Pivot-A Phase 2 §3.6→§7 relocate SHIPPED 2026-09-16: §3 method's "Secondary generator (CFM geometric)" sub-section co` |
| `f1_integrate.md:73` | `protocol_stub_adapter` | `external_sota_adapter` | `   - `sec:method:secondary-cfm` (the §3.6 stub) and/or `sec:future:cfm-deferred` (§7.2)` |
| `f1_verify.md:60` | `protocol_stub_adapter` | `external_sota_adapter` | `3. **`\label{sec:method:secondary-cfm}` is a self-ref anchor only**. §3.6 is now a 30-line stub whose sole job is to hos` |
| `f2_platinai_s4.md:61` | `placeholder` | `n/a (paper template)` | `\| `§4.7 metallodrug panel` in `04_evaluation.tex` \| MISSING (now SHIPPED as §4.6.1 placeholder by this workflow) \|` |
| `f2_platinai_s4.md:106` | `placeholder` | `n/a (paper template)` | `4. **No 5-seed data cell** is populated. The seed set `{42, 0, 1234, 7, 99}` is documented in the placeholder text but n` |
| `f3_crossref.md:155` | `placeholder` | `n/a (paper template)` | `The F2 placeholder text at line 1815 says (paraphrasing from `04_evaluation.tex:1815`):` |
| `results.json:30` | `stub_module_name` | `n/a (module name)` | `    "lam_chem/md_relaxation_stub": 228.04783697938547,` |
| `results.json:64` | `stub_module_name` | `n/a (module name)` | `    "lam_chem/md_relaxation_stub": 0.691554069519043,` |
| `results.json:98` | `stub_module_name` | `n/a (module name)` | `    "lam_chem/md_relaxation_stub": 10,` |
| `results.json:132` | `stub_module_name` | `n/a (module name)` | `    "lam_chem/md_relaxation_stub": "ok",` |
| `results.json:172` | `stub_module_name` | `n/a (module name)` | `      "lam_chem/md_relaxation_stub",` |
| `phase2_implement.md:159` | `placeholder_text` | `n/a (paper template)` | `                    pass  # no-op placeholder for the post-expand hook` |
| `final.md:61` | `protocol_stub_adapter` | `external_sota_adapter` | `  end-to-end with stub adapter).` |
| `final.md:124` | `protocol_stub_adapter` | `external_sota_adapter` | `\| Coupling adapter stub returns constant embedding \| `_stub_adapter()` is deterministic + documented; honest zero cont` |
| `benchmark_r18_end_to_end.py:433` | `stub_module_name` | `n/a (module name)` | `    from molmetal.molmetal_lam.lam_chem.md_relaxation_stub import (` |
| `benchmark_r18_metalcytotox.py:172` | `stub_module_name` | `n/a (module name)` | `        "lam_chem/md_relaxation_stub",` |
| `benchmark_r18_platinai.py:365` | `stub_module_name` | `n/a (module name)` | `    from molmetal.molmetal_lam.lam_chem.md_relaxation_stub import (` |
| `benchmark_r18_platinai.py:488` | `stub_module_name` | `n/a (module name)` | `    runners["lam_chem/md_relaxation_stub"] = _run_md` |
| `benchmark_r18_platinai.py:672` | `stub_module_name` | `n/a (module name)` | `        "lam_chem/md_relaxation_stub",` |
| `dock_10pockets_se3sb.py:307` | `stub_identifier` | `unknown` | `    class _Stub(nn.Module):` |
| `dock_10pockets_se3sb.py:312` | `stub_identifier` | `unknown` | `    net = _Stub()` |
| `dock_10pockets_se3sb.py:485` | `stub_identifier` | `unknown` | `    Returns ``(adapter, is_stub)`` — when the adapter can't be` |
| `dock_10pockets_se3sb.py:666` | `stub_identifier` | `unknown` | `    vina_adapter, is_stub = (` |
| `dock_10pockets_se3sb.py:670` | `stub_identifier` | `unknown` | `    if use_real_vina and is_stub:` |
| `dock_10pockets_se3sb.py:699` | `stub_identifier` | `unknown` | `                        use_real_vina=bool(use_real_vina and not is_stub),` |
| `verify_md_relaxation.py:5` | `stub_module_name` | `n/a (module name)` | `:mod:`molmetal.molmetal_lam.lam_chem.md_relaxation_stub` (per the` |
| `verify_md_relaxation.py:39` | `stub_module_name` | `n/a (module name)` | `from molmetal.molmetal_lam.lam_chem.md_relaxation_stub import (  # noqa: E402` |
| `verify_paper_build.py:178` | `placeholder` | `n/a (paper template)` | `        "sections/03_method.tex": "§3 Method (placeholder assembled by WF-Lambda-3)",` |
| `verify_r18_all_imports.py:164` | `stub_module_name` | `n/a (module name)` | `    ("lam_chem/md_relaxation_stub",` |
| `verify_r18_all_imports.py:165` | `stub_module_name` | `n/a (module name)` | `     "molmetal.molmetal_lam.lam_chem.md_relaxation_stub",` |
| `verify_variance_mcts.py:44` | `stub_identifier` | `unknown` | `    class _Stub:` |

---

## Scope

* Scripts scanned: 63
* Reports scanned: 939
* Metric JSONs scanned: 113

## Interpretation

The Wf-30 Phase-3 m3 Vina Dock sweep (`scorer="stub"`, `use_real_vina=false`) is the **only** critical finding that ships into the paper §4 table. Every other critical / major finding is either an honest-negative pre-recorded verdict (e.g. `MEASURED_BUT_STUB` in `m3_lift_analysis.json:95`) or an internal fallback that the script correctly falls back on when the real backend is unavailable.
