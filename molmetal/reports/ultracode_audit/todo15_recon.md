# TODO-15 / TODO-16 Recon — anticancer metric suite + DNA fragment docking

**Author:** ultracode recon agent
**Date:** 2026-09-14
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Python:** uv-managed 3.12; ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64

---

## 1. Spec sources

| TODO | Path | Status |
|------|------|--------|
| TODO-15 | `/home/hugo/codes/try_triton_on_rocm/TODO/pending/15_anticancer_metric_suite_r11b.md` | Local completion update 2026-09-13: "RewardAggregator canonicalizes object states, registers logP/TPSA/rotatable-bond/hERG channels, and exposes composite_score through reward and metrics output." |
| TODO-16 | **MISSING** | No `16_*.md` in `TODO/pending/`. The slot is open. The reconstruction target must be inferred from §6 of `molmetal/reports/anticancer_vs_general_metrics_survey.md` (DNA binding Kb proxy / docking score vs a 1,2-d(GpG) fragment — Jamieson & Lippard, Chem Rev 99:2467, 1999). |

---

## 2. State-of-the-suite (one-paragraph)

TODO-15 is **partially implemented, not finished**: `molmetal/molmetal_lam/priors/anticancer_metric_suite.py` (`AnticancerMetricSuite`) exists and exposes `logp_anticancer`, `tpsa_iv`, `rotatable_bonds`, `herg_proxy`, `composite_score`, `_raw_descriptors`, `descriptor_report`, `adjusted_flags`, `molecular_weight`, `l5_metrics`; it already drops the Lipinski MW≤500 hard requirement (replaces it with the descriptive 300–700 Da `MW_metal_adjusted` band, while keeping `MW_lipinski` available for cross-comparison). The five core channels are registered into `RewardAggregator` via `register_anticancer_channels()` (`molmetal/molmetal_lam/search_alg/proof_search.py:716`) and aggregated inside `__call__` (lines 901-930) with matching `w_logp_anticancer / w_tpsa_iv / w_rotatable_bonds / w_herg_proxy / w_anticancer_composite` weights. The composite_score already folds in the metal-specific `aquation_rate_proxy` and the GSH flag (`gs_resistance_flag` via `gsh_resistance_flag`/`gs_resistance_flag` getattr — `metal_hydration.py:117`). **What's missing**: the §6 buckets list a DNA Kb proxy and an explicit metal-coordination score (square-planar Pt(II)=4, octahedral Ru/III,Ir/III=6) — neither is surfaced as a top-level `AnticancerMetricSuite` channel. The DNA-fragment oracle (TODO-16) does not exist anywhere on disk; the only DNA-aware artefact is the **declarative** `PT_DNA_MAJOR_GROOVE` `BindingSite` in `molmetal/molmetal_lam/binding/types.py:1003` plus a one-paragraph mention in `molmetal_lam/SMOKE_TEST_PLAN.md`. `molmetal_lam/tile_lib/library.py` has a fragment *tile* pool for the Lambda proof search but it is unrelated to DNA.

## 3. File / path inventory

### Existing (canonical)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/anticancer_metric_suite.py` — 138 L, the §15 implementation. Bands: logP 2–5, TPSA 60–150, RotB <10 strict (reward keeps ≤10 = 1.0 then linear 10–15), hERG proxy (Fermini/Coyne), composite_score w=(0.20,0.15,0.10,0.20,0.20,0.15) over {logP_ant, tpsa_iv, rotb, herg, aquation_proxy, gsh_flag}. `descriptor_report` returns raw logP/TPSA/RotB/MW + flags: `descriptor_valid`, `logP_in_range`, `TPSA_in_range`, `RotB_in_range`, `MW_lipinski`, `MW_metal_adjusted`. Empty/wildcard/unparseable rows are all-NaN with `descriptor_valid=False`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py` — 867 L. `MetalGeometryPrior` supports SQUARE_PLANAR/TETRAHEDRAL/OCTAHEDRAL/TRIGONAL_BIPYRAMIDAL; `DEFAULT_METAL_GEOMETRY = {78: SP, 46: SP, 29: TD, 30: TD, 26: OCT, 44: OCT, 77: OCT, 45: OCT, 25: TBP}` (Pt(II), Pd(II), Cu(I), Zn(II), Fe(II), Ru(II/III), Ir(III), Rh(III), Mn(0)). Coord numbers via `GEOMETRY_COORDINATION_NUMBER` (4/4/6/5). Has `apply_prior`/`prior_loss`/Bondi vdW; `enabled` flag for prior-on/off (round-10).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_hydration.py` — 196 L. `MetalHydrationAnalyzer` exposes `aquation_rate_proxy`, `kinetic_inertness_score`, `gs_resistance_flag`, `reduction_potential_class`, `trans_effect_indicator`. `DEFAULT_PRIORS = {"Pt":{2:0.72,4:0.25}, "Ru":{2:0.22,3:0.38}, "Ir":{3:0.18}}`. Encodes chelation, aromatic-N, pi-acceptor, S-donor, Cl count, d-electron count heuristics.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/__init__.py` — re-exports `SquarePlanarPtII`, `MetalGeometryPrior`, `MetalHydrationAnalyzer`, `AnticancerMetricSuite`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py:566-945` — `RewardAggregator` dataclass. Channels wired today: `r_vina, r_sa, r_qed, r_vina_proxy, r_posebusters, r_pb_valid, r_pic50, r_retro, r_reinvent4, r_synth, r_admet, r_logp_anticancer, r_tpsa_iv, r_rotatable_bonds, r_herg_proxy, r_anticancer_composite`. `aggregate()` dict path does **not** route anticancer channels (lines 827-839) — only the 11 non-anticancer channels. `metrics()` returns anticancer channel callables when set (lines 939-945).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/scripts/pocket2mol_vs_lambda_1h36.py:386-429` — `_aggregate_metrics` calls `anti.descriptor_report` and `anti.composite_score` per row; writes `anticancer_composite_mean / n_valid / mw_lipinski_rate / rotb_in_range_rate / mw_metal_adjusted_rate / success_rate_docked / mean_vina` etc.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/binding/types.py:1003` — `PT_DNA_MAJOR_GROOVE: BindingSite` (declarative type, no oracle/scoring). The bound geometry/predicate lives there.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/SMOKE_TEST_PLAN.md:128-148` — conceptual mention of a `DNA_Pt_crosslink` `BindingType` for cisplatin, but no oracle module.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_anticancer_metric_suite.py` — 143 L, 8 test fns + 6 parametrize blocks. Covers invalid SMILES, composite unit-interval, l5 counters, channel registration, MW flag boundaries (300/500/700 Da), strict RotB (<10), NaN/inf descriptors, single-parse invariant, `_aggregate_metrics` integration. All green as of round-8.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/anticancer_vs_general_metrics_survey.md` — survey document; §6 is the canonical spec for what TODO-15/16 must satisfy.

### Missing (must be created)
- `molmetal/molmetal_lam/metrics/__init__.py` — NEW (parallel to `priors/`). Will host the metric reporting layer separately from the metal priors (descriptor-vs-reward split per the survey).
- `molmetal/molmetal_lam/metrics/anticancer_metric_suite.py` — NEW. Same surface as `priors/anticancer_metric_suite.py` but with the *additional* buckets: metal-coordination score (CN lookup using `GEOMETRY_COORDINATION_NUMBER` + `DEFAULT_METAL_GEOMETRY`), DNA Kb proxy (heuristic), GSH flag exposed as a first-class channel (currently buried inside `composite_score` only).
- `molmetal/molmetal_lam/metrics/dna_fragment_oracle.py` — NEW. TODO-16 oracle stub. Should declare a `DNAFragmentOracle` returning a 1,2-d(GpG)-style DNA-fragment heuristic score (count of guanine-N7 motifs + aromatic N + d-electron geometry match). **Pure heuristic — no docking binary needed for v0.** Wired to `PT_DNA_MAJOR_GROOVE` so the existing `binding_site`-bonus path picks it up.
- `molmetal/molmetal_lam/tests/test_dna_fragment_oracle.py` — NEW. pytest covering invalid SMILES, Pt(II) cisplatin-like ligand, Ru(III) prodrug, non-metal ligand → 0.0, descriptor-parity with `descriptor_report`.
- `molmetal/molmetal_lam/tests/test_anticancer_metric_suite_extended.py` — NEW. Additional pytest for the metal-coordination + DNA-Kb channels.
- `molmetal/scripts/evaluate_anticancer_metrics.py` — NEW. Demo pocket runner: take 1h36 (or any staged pocket), compute `AnticancerMetricSuiteExtended` per top-k ligand, dump JSON + markdown table.
- `molmetal/reports/todo15_anticancer_metric_suite.md` — NEW. Implementation report + 1 demo pocket output.

### Untouched
- `molmetal/molmetal_lam/priors/anticancer_metric_suite.py` — **DO NOT modify** (per existing todo rule that `RewardAggregator` keeps canonicalizing off the prior copy).
- All `TODO/pending/*` files — **DO NOT modify**.
- `molmetal/molmetal_lam/search_alg/proof_search.py` — `RewardAggregator` is **already wired**; no change required for TODO-15 demo. The aggregator's `aggregate()` dict-path *should* gain anticancer keys in a follow-up if a future caller wants a flat-channel prior; out of scope for this round.

---

## 4. Concrete spec for what to wire next

The next codex/code-agent task should:

1. **Create** `molmetal/molmetal_lam/metrics/__init__.py` that re-exports `AnticancerMetricSuiteExtended` and `DNAFragmentOracle`.
2. **Create** `molmetal/molmetal_lam/metrics/anticancer_metric_suite.py` (class `AnticancerMetricSuiteExtended`) — wrap the existing `AnticancerMetricSuite` and add:
   - `metal_coordination_score(smiles)` → 1.0 when the detected metal's coord number matches `GEOMETRY_COORDINATION_NUMBER[DEFAULT_METAL_GEOMETRY[Z]]`; otherwise linear penalty. Default metal `Pt(II)`.
   - `dna_kb_proxy(smiles)` → 0.0–1.0 heuristic based on (a) presence of Pt/Ru/Ir centre, (b) two or more leaving groups (Cl/H₂O-style), (c) at least one aromatic N donor (guanine N7 mimic). Square-planar Pt(II)→1.0; octahedral Ru(III)/Ir(III) → 0.7; no metal → 0.0.
   - `gsh_evasion_flag(smiles)` → forwards `metal_hydration.gs_resistance_flag`.
   - `composite_score_extended(smiles)` → re-weights `composite_score` to fold in the new channels at 0.10 (metal-coord), 0.10 (DNA Kb), 0.05 (GSH explicit), so the sum stays in [0,1].
   - `descriptor_report_extended(smiles)` → merges the existing `descriptor_report` dict with `{metal, oxidation_state, coordination_score, dna_kb_proxy, gsh_evasion_flag, composite_extended}`.
3. **Create** `molmetal/molmetal_lam/metrics/dna_fragment_oracle.py` (class `DNAFragmentOracle`):
   - Constructor: `metal="Pt"`, `oxidation_state=2`, `pH=7.4`, `dna_motif="GpG"`.
   - `score(smiles) -> float` returning the `dna_kb_proxy` heuristic above, with deterministic NaN/inf fallback = 0.0.
   - `report(smiles) -> dict` returning the raw feature dict (`has_pt`, `n_leaving_groups`, `n_aromatic_n_donors`, `coordination_number`, `geometry_match`) for explainability.
   - `binding_site() -> BindingSite` returning `PT_DNA_MAJOR_GROOVE` (re-exported from `molmetal_lam.binding.types`) so callers can compose with `RewardAggregator(binds_target=True, bonus_binder=0.5)`.
4. **Write tests**:
   - `tests/test_anticancer_metric_suite_extended.py` — covers invalid SMILES, cisplatin-like Pt(II) complex hits metal-coord + DNA Kb = 1.0, methanol hits all 0.0, NaN propagation, descriptor_report_extended has all expected keys.
   - `tests/test_dna_fragment_oracle.py` — same suite scoped to the oracle; also asserts `binding_site().name == "Pt_DNA_major_groove"`.
5. **Create** `molmetal/scripts/evaluate_anticancer_metrics.py` — demo runner that:
   - Loads the staged pocket from `molmetal/data/...` (default: `1h36`, fallback to `crossdocked_first10_resolved`).
   - Generates `N=20` ligands via the existing `proof_search.MCTSProofSearch` (or static canned list if MCTS weights are unset).
   - Calls `AnticancerMetricSuiteExtended` + `DNAFragmentOracle` per ligand.
   - Writes JSON to `molmetal/reports/todo15_anticancer_metric_suite_demo.json` and a markdown table to stdout.
6. **Create** `molmetal/reports/todo15_anticancer_metric_suite.md` — implementation report: scope, design, citations from §6, demo pocket output, next-step recommendations (e.g. wire `aggregate()` to include anticancer channel keys, expose `bonus_dna_kb` flag in `RewardAggregator`).
7. **Run** `uv run pytest molmetal/molmetal_lam/tests/test_anticancer_metric_suite_extended.py molmetal/molmetal_lam/tests/test_dna_fragment_oracle.py -v` and confirm green.
8. **Run** `uv run python molmetal/scripts/evaluate_anticancer_metrics.py --pocket 1h36 --n 20` to materialise the demo output referenced in §6.

### Out of scope (deferred)
- True 1,2-d(GpG) docking via Vina/QuickVina against a DNA PDB fragment (would require staging the DNA PDB and a docking backend against a non-protein target — follow-up round).
- `RewardAggregator.aggregate()` dict-path routing for anticancer channels (TODO-15 spec already calls it "wired"; current canonical path is `__call__`, which is what `proof_search.heuristic` ultimately uses).
- Updates to `TODO/pending/13_top_journal_pilot_r12.md` and `TODO/pending/14_full_100pocket_paper_r13.md` to point at the new demo output — codex subagent #305/#306 already cover this.

### Risks / known limits
- The metal-coordination channel is a *hard* match against `DEFAULT_METAL_GEOMETRY`; atypical coordination (e.g. 5-coordinate Pt(II) in some pincer variants) will be penalised. Acceptable for v0; document as heuristic in the implementation report.
- `dna_kb_proxy` is explicitly **not** a measured Kb — it's a structural heuristic per the survey's caveat. The docstring must say so.
- `binding_site()` in `DNAFragmentOracle` re-exports `PT_DNA_MAJOR_GROOVE`, which is already used by the Lambda `typecheck` pipeline; no risk of duplicate binding-site definition.
