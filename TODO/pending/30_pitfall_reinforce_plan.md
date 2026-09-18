# TODO-30 — Pitfall Reinforce Plan

**Status**: TIER-1 SHIPPED 2026-09-17 (5/5 Rank 1-5 closed); Tier-2 still DEFER (Rank 6-16)
**Master**: `molmetal/reports/wf_t30_tier1_done/MASTER.md`
**Companion**: `molmetal/reports/wf_pitfall_audit/MASTER.md`
**Source audits**: `molmetal/reports/wf_pitfall_audit/p{1..6}_*.md`
**(22 pitfalls across 6 layers; 2 AVOIDED, 11 PARTIAL, 9 OPEN)**

---

## Priority order (ranked by EV)

EV = (paper-strengthening value) × (implementation affordability).
See MASTER.md §"Top 5 OPEN pitfalls" for rationale.

| Rank | ID | Pitfall | ETA | Risk | Verdict |
|---|---|---|---|---|---|
| 1 | **P3.3** | Wire `pareto.py` (440 LOC unused) into candidate ranker | 2h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_pareto_wire`) |
| 2 | **P4.2** | Patent / known-Pt-drug similarity axis | 4-6h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_patent_axis`) |
| 3 | **P1.4** | Known-Pt regression tests (cisplatin/carboplatin/oxaliplatin) | 8h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_known_pt_regression`) |
| 4 | **P5.2** | Wet-lab feedback plumbing (`r_wetlab` + `recalibrate_from_assay()`) Tier 1 | 6h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_wetlab_plumbing`) |
| 5 | **P2.5** | Wire `r_pharmacophore` into RewardAggregator (Phase A) | 2h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_pharmacophore_wire`) |
| 6 | **P2.3** | Layered generation (scaffold→attach→decorate phases) | 12h | medium | **SHIP** |
| 7 | **P2.2** | Always-on 3D/pocket (`materialize_3d=True` default flip) | 9.5h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_p22_default_3d`) |
| 8 | **P1.3** | Buyable-reactant gate (AiZynth-as-gate) | 12h | medium-high | **SHIP** |
| 9 | **P5.3** | `register_rule` + `vet_rule` public API | 11h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_p53_rule_registry`) |
| 10 | **P1.2** | FG-compat / leave-group veto in `_reduce` | 9h | medium | **DEFER** (over-narrows veto; needs property tests first) |
| 11 | **P1.1** | Non-click med-chem rules (peptide / Boc / reductive amination) | 10h | low | **DEFER** (Round-15+ scope; expansion not bug fix) |
| 12 | **P2.4** | Skeleton jump / fragment-swap rule | 15h | medium-high | **DEFER** (sequence after P2.3; high invalid-structure risk) |
| 13 | **P2.5** | MD relaxation (OpenMM `relax_mol`) Phases C-E | 14h | medium | **DEFER** (needs ROCm OpenMM validation) |
| 14 | **P6.3** | Tanimoto<0.4 + MMseqs2 30%-seq-id (OOD protocol) | 6h | medium | **DEFER** (MMseqs2 binary not installed) |
| 14a | **P6.1+P6.2** | Reproducibility (scaffold split + per-run hygiene) | ~3h | low | **SHIP [x] DONE 2026-09-17** (`wf_t30_p62_scaffold_seed`) |
| 15 | **P5.2** | Wet-lab outreach (3 collaborators) Tier 3 | 1h | n/a | **DEFER** (needs chemistry facility) |
| 16 | **P4.1** | Generalisation sample size (100×3 sweep) | 8h | medium | **DEFER** (R-13 re-execution; GPU-blocked per `wf_gpu_recovery_now`) |

**Total Tier-1 ship (rank 1-9)**: ~64h CPU ≈ 1.6 dev-weeks.
**Total Tier-2 defer (rank 10-16)**: ~55h CPU ≈ 1.4 dev-weeks.

---

## Per-pitfall concrete patch (SHIP items)

### Rank 1 — P3.3 wire pareto.py (2h, low risk)
**Blocker**: none (all code exists; just unused).
**Patch**: replace `candidates.sort(key=lambda p: p[0], reverse=True)`
at `proof_search.py:3559-3593` with `rank_population(population,
weights=...)` from `molmetal_lam.search_alg.pareto`. Add
`--postprocess-pareto` CLI flag to `r4_lambda_only_run.py:1487-1596`
argparser (default OFF). Cite `pareto.py` in paper §4.7.
**Files**: `proof_search.py:3559-3593`, `r4_lambda_only_run.py`.
**ETA**: 2h + 30 min pytest.

### Rank 2 — P4.2 patent axis (4-6h, low risk)
**Blocker**: none (morgan_fingerprints exists at `data_diversity.py:64-89`).
**Patch**:
1. `data/known_pt_drugs.csv` — 5-7 canonical SMILES (cisplatin,
   carboplatin, oxaliplatin, nedaplatin, satraplatin, picoplatin,
   heptaplatin).
2. `metric_max_sim_known_pt(smiles) -> float` in `r4_lambda_only_run.py`.
3. `patent_risk_mean` + `scaffold_in_known_pt_drugs: bool` columns.
4. Bemis-Murcko scaffold via `training_set_novelty.py:46`.
5. orange/red/green risk-flag column in Table 1 caption.

**Files**: `r4_lambda_only_run.py:881-940`, `training_set_novelty.py`.
**ETA**: 4-6h.

### Rank 3 — P1.4 known-Pt regression tests (8h, low risk)
**Blocker**: none (pure test addition).
**Patch**:
1. `test_pt_metal_ligand_exchange_known_complexes`: cisplatin + NH3
   → mono-ammine; + NH3 → diammine; carboplatin + NH3 → CBDCA ring
   opening.
2. `test_pt_aqua_exchange_first_second_aquation` (both patterns at
   `pt_metal_ligand_exchange.py:120-136`).
3. 35-cell `pt_click_compat` smoke verifying `COMPAT_MATRIX` cell
   verdicts.
4. Cisplatin vs satraplain coordination sanity on `metal_geometry`.

**Files**: NEW `tests/test_pt_metal_ligand_exchange.py`,
`tests/test_pt_click_compat_smoke.py`.
**ETA**: 8h.

### Rank 4 — P5.2 wet-lab plumbing Tier 1 (6h, low risk)
**Blocker**: none (no wet-lab execution needed).
**Patch**:
1. `Assay` dataclass in NEW `wetlab_protocol.py`:
   `cell_line`, `target_protein`, `outcome_metric`, `outcome_value`,
   `outcome_unit`, `smiles`, `assay_id`, `collaborator`, `date`.
   CSV `load_assays` / `append_assay` helpers.
2. `r_wetlab` channel in `RewardAggregator`
   (`proof_search.py:614-706`): mean outcome_value for assayed
   SMILES, 0.0 if not assayed. Weight `w_wetlab=0.0` default.
3. `RewardAggregator.recalibrate_from_assay()` method — 10 lines:
   fit 2-d (mw, logP) regressor, blend 50/50 with existing oracle.
4. `--recalibrate-from-assay path/to/assays.csv` CLI flag.

**Files**: NEW `wetlab_protocol.py`, `proof_search.py:614-706`,
`r4_lambda_only_run.py`.
**ETA**: 6h.

### Rank 5 — P2.5 pharmacophore wire-in (2h, low risk)
**Blocker**: none (`pharmacophore_filter.py:445` exists).
**Patch**:
1. `r_pharmacophore: Optional[Callable]` + `w_pharmacophore` weight
   in `RewardAggregator` (mirrors `r_posebusters`).
2. Default wraps `pharmacophore_filter.pass_pharmacophore(smi)`.
3. `--pharmacophore-filter on` CLI flag in both `r4_*` scripts.
4. 3 unit tests.

**Files**: `proof_search.py:614-706`, `r4_lambda_only_run.py`,
`r4_c_full_sweep.py`.
**ETA**: 2h.

### Rank 6-9 (P2.3 / P2.2 / P1.3 / P5.3) — see MASTER.md §"Recommended sequencing"
Detailed patch plans in P2 + P1 + P5 audit reports.
Total Tier-1 ship budget: 64h CPU ≈ 1.6 dev-weeks.

---

## Categorisation

**SHIP (1-week CPU budget, 64h total)** — rank 1-9. All CPU-only, no
GPU dependency, no new external services. Recommended order:
P3.3 → P4.2 → P1.4 → P5.2 → P2.5 → P2.3 → P2.2 → P1.3 → P5.3.

**DEFER (GPU, wet-lab, or new deps)** — rank 10-16:
- P1.2 (FG-compat) — needs property tests to bound false-positive rate
- P1.1 (non-click med-chem) — Round-15+ expansion scope
- P2.4 (skeleton jump) — needs P2.3 first; high invalid-structure risk
- P2.5 Phases C-E (MD relax) — needs ROCm OpenMM validation
- P6.3 (MMseqs2 + Tanimoto<0.4) — needs MMseqs2 binary install
- P5.2 Tier 3 (wet-lab outreach) — out of scope
- P4.1 (generalisation n) — depends on Round-13 sweep re-execution;
  GPU-blocked per `wf_gpu_recovery_now` 2026-09-15

---

## Reference evidence

Per-pitfall evidence (file:line + upstream audit section) is in the
master audit (`MASTER.md`) and the per-layer audits (`p{1..6}_*.md`).
Cite-only honest framing preserved across all 6 layers per
`WF-Pivot-Followup MASTER` and `wf_pitfall_audit_done_2026-09-17`
memory entry.

**No GPU retrain claimed.** All Tier-1 SHIP items are CPU-only
engineering. The only GPU-requiring item is P4.1 (generalisation),
which is DEFER'd to a future Round-13 re-execution.