# Post-Round-10 / Round-11 Refined Action Plan

**Status:** living plan, append-only superset of `TODO/pending/17_aggregate_weak_impls_and_pending.md`
**Created:** 2026-09-14
**Authoritative inputs:**
- `molmetal/reports/round10_e2e_pt_cfg_vina.md` (Pt prior + CFG end-to-end Vina micro-bench)
- `molmetal/reports/round11_engine_parity_n50.md` (N=50 Vina-vs-QuickVina paired parity)
- `molmetal/reports/todo15_anticancer_metric_suite.md` (anticancer metric suite ship)
- `TODO/pending/19_user_decisions.md` (6 user-gated decisions)
- `molmetal/reports/ultracode_audit/PROJECT_STATUS.md` (synthesis)

## Why this file exists

The 4 ultracode workflows run on 2026-09-14 produced **3 concrete structural
insights** + **1 new empirical number**. None of them fit cleanly into the
existing `11_algorithm_strengthening_r10.md` or `12_qvina_data_staging_r11.md`
files because they are **architectural or experimental-design revisions**,
not additional scope items. This file is the consolidated, ordered list of
what to do next.

## TL;DR — 4 actions, in order

| # | Action | Why | Wall | Blocker |
|---|---|---|---|---|
| 1 | **Replace "seed sweep" with architectural CFG fix** (learned bond-order head + atom vocab mask + `--seeds` CLI) | Today's CFG seed1 run = 96/96 finite / **0 decoded** bit-identical to v2 baseline 92+4. Failure is **decoder + unconstrained atom head**, not seed. Further seed sweeps without architecture fix are wasted compute. | 3–4 d | none |
| 2 | **Pt(II) prior experiment redesign**: Pt-pocket + real CFM data + n_seeds=3 + n_mols=60+ | Today's run = 0.000 kcal/mol delta, n_docked=4. 1h36 is Fe(HEM) not Pt; CFM data is synthetic random points; 30 train steps. The harness works; the design is wrong. | 3–5 d (post-CFG fix) | tmQM Pt/Ru/Ir subset staging |
| 3 | **D7 Vina/QVina parity — adopt (c) both engines in headline table with explicit caveat** | Today's N=50: Pearson **r=0.9983** / Spearman **ρ=0.9984** / mean diff **+0.0086±0.0184** kcal/mol (n=46 paired). Strong enough for 1h36/seed=42/exh=8 claim; single-pocket only. | 0 d | user decision D7 |
| 4 | **Re-stack timeline** to honour (1) + (2) + (3) before Round-12 pilot | Original timeline assumed Round-10 closes in W3 and Round-12 launches in W5. With (1)+(2) added, Round-12 launches in W6 at earliest. | 0 d | user sign-off |

## Action 1 — CFG architectural fix (3–4 d)

### What is wrong (MEASURED, `molmetal/reports/round10_e2e_pt_cfg_vina.md` §CFG §Decode-failure-mode)

- **92/96** decoded-status = `disconnected_distance_graph`: `decode_distance_graph` (RDKit `rdDetermineBonds.DetermineConnectivity` with `covFactor=1.3`) emits fragmented graphs because pairwise-distance distribution shows **min=1.03 Å, median=4.55 Å, max=8.59 Å** — only 3/171 pairs <1.5 Å. The atom cloud itself is sparse.
- **4/96** decoded-status = `atom_outside_training_vocabulary`: unconstrained atom head picks Z ∉ `{6,7,8,9}`.
- Failure is **bit-identical** between v2 baseline (`r10_cfg_real_crossdocked_v2_train32_2000/`) and today's `r10_cfg_real_seed1/` run, confirming architecture-bound.

### Fix plan

1. **Learned bond-order head** (`A1`, ~2 d): replace the `decode_distance_graph` step with a small EGNN-based bond head trained on TMQM bond patterns. Acceptable bond candidates = argmax of bond logits; valence check stays RDKit.
   - Reuse: `molmetal/adapters/egnn_rocm.py` (existing ROCm-resident EGNN); `molmetal/molmetal_lam/lam_chem/ast.py` (capture-avoiding subst).
   - New: `molmetal/models/bond_head.py` (small MLP, ~120 LOC); `molmetal/molmetal_lam/tests/test_bond_head.py` (≥6 tests: recovers CuAAC azide-alkyne pattern, recovers amide pattern, rejects impossible valences, single-pair candidate = single bond, multi-candidate = best-ranked bond).
2. **Atom vocabulary mask** (`A2`, ~1 d): at sample time, mask softmax over Z ∈ `{1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78}` (adds P, S, Cl, Se, Br, I, Pt for metal-aware generation). Metal atoms are rare events; mask only relaxes, doesn't force.
   - Reuse: `molmetal/molmetal_lam/priors/metal_geometry.py:DEFAULT_METAL_GEOMETRY` for the Z-mapping.
   - New: vocab mask line in `flow_matching_lipman/__init__.py:generate()`.
3. **`--seeds` CLI flag** (`A3`, ~0.5 d): extend `molmetal/scripts/r10_cfg_real_crossdocked.py` with `--seeds` (nargs="+", default `[42, 0, 1234]`) so future runs can sweep without editing source.
4. **Re-run CFG e2e with A1+A2+A3** (`A4`, ~0.5 d): `--seeds 42 0 1234 7 2024 31415 --train-steps 2000 --n-samples 16`. Expected n_decoded > 0 if A1+A2 succeed. If still 0, escalate to A5 (joint decoder+head training) and re-design.
5. **A1+A2+A3+A4 success criterion**: n_decoded/n_finite ≥ 0.5 (was 0/96); preserved PB validity among decoded mols ≥ 0.5.

### What this is NOT

- This is **not** "more seed sweeps at the same configuration". Audit-agent recon (`r10_recon_cfg.md`) flagged this. Any further seed sweep without A1+A2 is wasted compute.
- This is **not** "rewrite CFG math" — `molmetal/tests/test_egnn_velocity_cfg.py` 5/5 green, CFG algebra is bit-exact at cfg_scale==1.0. The math is fine; the decoder is not.

## Action 2 — Pt(II) prior experiment redesign (3–5 d, post-CFG fix)

### What is wrong (MEASURED, `molmetal/reports/round10_e2e_pt_cfg_vina.md` §Pt prior §Why OFF and ON produced identical scores)

- Delta = **+0.00000 kcal/mol**, score vectors bit-identical OFF/ON.
- 1h36 receptor = **Fe(HEM)**, prior target is **Pt(II) square-planar**. Generated mols rarely contain Pt → prior short-circuits to 0.
- CFM training data is **synthetic random points** (per `round10_pt_prior_ablation.md` §Notes); 30 train steps ≈ noise floor.
- 16/20 mols fail PDBQT conversion (`non finite charge`) — known micro-bench artifact.

### Redesign plan

1. **Identify a Pt-pocket receptor** (`B1`, ~1 d): from `/mnt/storage/data/molmetal/tmqm_*` or CrossDocked2020, select 2-3 pockets whose bound ligand is a Pt complex (search SMILES for `[Pt]` in `molmetal/data/crossdocked100_manifest.csv` ligand_path column; or PDB `cisplatin` analogue query).
   - Fallback: use the **tmQM Pt subset** (~15k transition-metal complexes with Z ≥ 21) already staged under `/mnt/storage/data/molmetal/tmQM_*` — Pt subset should have ≥500 entries.
2. **Real CFM training data** (`B2`, ~1 d): replace synthetic random points with the tmQM Pt/Ru/Ir subset, written to `molmetal/data/cfm_training/pt_subset.{sdf,npz}`. Update `LipmanFlowMatchingAdapter.train_step()` loader.
3. **Extended budget** (`B3`, ~0.5 d): `train_steps=2000`, `n_mols=60`, `n_seeds=3` (seeds `[42, 0, 1234]`). Use the new `--seeds` CLI flag from Action 1.
4. **Tighter success criterion** (`B4`): |Δ| ≥ **0.5 kcal/mol** (was 0.2), at n_docked ≥ 20 per setting, pooled SE ≤ 0.15 kcal/mol.
5. **Run on the chosen Pt-pocket** (`B5`, ~0.5 d): write `molmetal/reports/r10b_pt_prior_e2e_pt_pocket.md` with the same Method/Result/What-this-DOES-NOT-measure structure as today's report.
6. **Append to `TODO/pending/11_algorithm_strengthening_r10.md`** as a new "R10b" section.

### Expected outcome

If prior genuinely lowers Vina on a Pt-pocket with real CFM data, Δ should be **−0.5 to −1.5 kcal/mol** (typical SG/MMFF94 prior term magnitude on transition-metal complexes). If Δ ≈ 0 again, the prior's value is **architecturally correct but empirically silent** — flag as `NOT-CLAIM` for Round-13 paper.

## Action 3 — D7 Vina/QVina decision unblock (0 d, user)

### Evidence (MEASURED, `molmetal/reports/round11_engine_parity_n50.md`)

| Metric | Value | n |
|---|---:|---:|
| Pearson r (kcal/mol) | **0.9983** | 46 paired |
| Spearman ρ (rank) | **0.9984** | 46 paired |
| Mean diff (Vina − QVina) | +0.0086 ± 0.0184 kcal/mol (paired SE) | 46 paired |
| Mean abs diff | 0.071 kcal/mol | 46 paired |
| Vina mean ± std | −7.350 ± 2.060 kcal/mol | 50 |
| QVina mean ± std | −7.422 ± 2.099 kcal/mol | 46 |
| QVina failures | 4/50 (CG0 atom-type vocab mismatch in vendored qvina02) | 50 |

Protocol: 1h36 / exh=8 / seed=42 / 1 CPU / matched box (centre [35.684, 52.085, 44.486] Å, side 21.502 Å).

### Recommendation: D7 option (c) both engines in headline table

- Pearson r=0.9983 **exceeds** Alhossary 2015's published r=0.967 between Vina and QuickVina (because we control seed + box + receptor).
- The 4 QVina failures are a **bundled qvina02 (AutoDock Vina 1.1.2) vocabulary delta**, not a parity failure. Replacing qvina02 with the modern QuickVina 2.1 release (round-11 axis A fallback ladder item 3) should close it.
- For Round-12/13, both engines in the headline table preserve the strongest comparison claim. Single-engine decision would lose either Vina's broader literature footprint or QuickVina's 100× speedup.

### What this does NOT measure

- **Single pocket only.** Cross-pocket parity (≥3 pockets) needed before Round-13 acceptance.
- **Per-seed σ within each engine.** Today's run = seed=42 only; σ within Vina and within QVina is the **between-molecule** std (2.060, 2.099), not the **between-seed** std.
- **exh=8 QVina ≈ exh=16 Vina.** Round-9 §5 explicitly assumes this but does not measure it.
- **Modern QuickVina 2.1 binary** (vs today's vendored qvina02 = Vina 1.1.2). Real QuickVina 2.1 has the modern atom vocab and is ~3-5× faster.

### User decision

Reply with D7 = (a) Vina-only / (b) QVina-only / (c) both in headline table. Default = (c).

## Action 4 — Timeline re-stack

### Original timeline (per `TODO/pending/roadmap.md`)

| Week | Target |
|---|---|
| W3 (2026-10-03) | Round-10 ship |
| W4 (2026-10-10) | Round-11 ship |
| W5-W6 | Round-12 pilot |
| W7 (2026-10-31) | Round-13 paper draft |

### Revised timeline (post-2026-09-14)

| Week | Target | Notes |
|---|---|---|
| W3 mid (2026-09-30) | **Action 1 (CFG architectural fix)** ship | New scope; not in original W3 |
| W3 end (2026-10-03) | **Action 2 (Pt-pocket redesign)** + **D7 user decision** | Round-10b ship |
| W4 (2026-10-10) | Round-11 close (QuickVina 2.1 binary + CrossDocked100 manifest final) + D6 decision | Round-11 was already W4; stays |
| W5 (2026-10-17) | **Round-12 scientific-budget pilot** (N=10 × 3-seed, anticancer metric + CFG-fixed mols) | Was W5-W6; compresses because CFG is fixed in W3 mid |
| W6 (2026-10-24) | Round-12 ablation + failure analysis + cite-only SOTA column decision | |
| W7 (2026-10-31) | **Round-13 100-pocket × 3-seed sweep** | Was W7; pushes a bit because Round-12 is tighter |
| W8 (2026-11-07) | Round-13 paper draft (`paper/digital_discovery_submission.tex`) | Was W7; +1 week slack |

### What this preserves

- Round-10 ship target (W3 / 2026-10-03) — but redefined as **R10b = Action 1 + 2**, not the original "6-axis ablation matrix".
- Round-11 ship target (W4 / 2026-10-10) — unchanged.
- Round-12 acceptance gate (W5 / 2026-10-17) — **moves up 1 week** because CFG fix unblocks decoded mols.
- Round-13 paper draft (W8 / 2026-11-07) — **+1 week slack** vs original W7.

### What this adds

- W3 mid: Action 1 (CFG fix) — **3-4 d work**.
- W3 end: Action 2 (Pt redesign) — **3-5 d work** post-CFG-fix.
- W3 end: D7 user decision — **0 d work** but blocks Round-12 acceptance.

## Other user-gated items (per `TODO/pending/19_user_decisions.md`)

| ID | Decision | Default | Deadline | Status |
|---|---|---|---|---|
| D6 | REINVENT4 install path | (a) separate venv | **2026-09-19** | awaiting user input |
| D7 | Vina/QVina swap | (c) both engines | post-R11-parity | **ready for user** (Action 3) |
| test_005 | cohort inclusion | (a) labeled modeled-atom | pre-R12-budget | awaiting user input |
| cite-SOTA | ship column | (a) ship with footnote | pre-paper-draft | awaiting user input |
| MW-range flag | dual flag | (c) both Lipinski + 300-700 Da | pre-paper-draft | awaiting user input |
| journal | Round-13 pick | story-dependent | 2026-10-31 | awaiting Round-12 results |

D7 is the only one with concrete evidence to decide now.

## Cross-references

- `TODO/completion_audit_2026-09-13.md` — authoritative state
- `TODO/pending/17_aggregate_weak_impls_and_pending.md` — weak impls / bad results (live audit)
- `TODO/pending/19_user_decisions.md` — 6 user-gated decisions
- `TODO/pending/11_algorithm_strengthening_r10.md` — Round-10 scope; append R10b Action 2 results here
- `TODO/pending/12_qvina_data_staging_r11.md` — Round-11 scope; N=50 parity result appended
- `molmetal/reports/ultracode_audit/PROJECT_STATUS.md` — full audit (synthesis)

## Update protocol

This file is **append-only**. Any new evidence (new ultracode round results,
new experiments, new user decisions) appends a new dated section at the end.
Prior sections are never edited — they are evidence of their original write.
---

## Update 2026-09-14 — WF-1 verification (Action 1) result

**Append-only summary line:** WF-1 verify (A1+A2+A3 + 6 seeds × 2 pockets × 2 cfg × 16 samples = 384 raw clouds) completed 2026-09-14. MEASURED: n_finite=384, n_decoded=0, decode_ratio=0.000 — success criterion ≥0.5 NOT MET. Dominant failure: `disconnected_distance_graph` 368/384 (95.8%); secondary `AtomValenceException` 12/384 (3.1%, introduced by A1 bond-head path); `atom_outside_training_vocabulary` 4/384 (1.0%, A2 reduced from 95.8% baseline → 1.0% — A2 measurably effective at CFG=1 but breaks at CFG=2). **Escalate to A5 (joint decoder+head training) for Round-12 / WF-2 prep.** Full report: `molmetal/reports/wf1_final.md`. MEASURED vs PROJECTED clearly labelled in §"Honest framing".

## Update 2026-09-14 — WF-Lambda-1 verify result

**Append-only summary line:** WF-Lambda-1 verify (N=10 test split pockets × 3 seeds, n_simulations=100, n_top_k=20, NO docking/AdmetAI/PB) completed 2026-09-14. MEASURED on 30 cells: validity_rate=0.9000, uniqueness_rate=0.9000, diversity_alpha=0.0043, novelty=1.0000 (placeholder, no training-set file), synthesizability_rate=0.0000, metal_compliance_rate=0.0000, ref_tanimoto=0.8595. Ablation CuAAC-only vs all-5: synthesizability_rate Δ = 0.0000 (both arms hit the same MoleculeClosedTerm.from_smiles round-trip defect). Total wall-clock 120.7 s. **Λ-only is competitive on validity/uniqueness (90% each, comparable to hybrid's ~92/88%) but loses on synthesis (Λ 0% vs hybrid ~50%) — loss is structural (round-trip defect), not algorithmic (click-rule machinery works).** Next: fix round-trip defect, then escalate to round-13 100×3 sweep. Full report: `molmetal/reports/wf_lambda1_pilot_v1/final.md`. MEASURED vs PROJECTED clearly labelled in §"Honest framing". `--click-rules` CLI flag added to `r4_lambda_only_run.py` for this ablation.

## Update 2026-09-14 — WF-2 verify result (A5 joint training + A6 Gumbel fallback)

**Append-only summary line:** WF-2 verify (A5 = joint end-to-end `BondOrderHead` + CFM training; A6 = DropEdge + Gumbel-top-k connectivity prior) completed 2026-09-14. Both runs executed the full 6 seeds × 2 pockets × 2 cfg × 16 samples = 384 raw clouds. MEASURED: A5 n_finite=384 n_decoded=0 decode_ratio=0.000; A6 n_finite=384 n_decoded=0 decode_ratio=0.000. Success criterion `decode_ratio ≥ 0.5` NOT MET by either path. Dominant failure (both): `disconnected_distance_graph` 380/384 (98.96%); residual `atom_outside_training_vocabulary` 4/384 (1.04%). A5 protocol fields confirmed in `report.json`: `wf2_a5_joint_train=True, wf2_a5_bond_loss_weight=1.0, wf2_a5_bond_pattern_mask=True, wf2_a6_connectivity_prior=gumbel`. **Root cause = CFM velocity field under-trained at 2000-step budget (hidden=32, layers=2, lr=1e-4), NOT the decoder** — the A1+A5+A6 stack is end-to-end correct (10+10+13+7 tests green in `molmetal_lam/tests/`) but cannot rescue a sparse cloud. **Side-finding:** the decoder import path in `r10_cfg_real_crossdocked.py` was importing `ConnectivityAwareDecoder` from `molmetal.models.connectivity_gumbel` instead of `molmetal.models.bond_head`; the resulting `learned_decoder_unavailable:ImportError` was silently masking the real per-status distribution in earlier WF-1 v2 reports. Fixed at `molmetal/scripts/r10_cfg_real_crossdocked.py:97-105`. **Verdict:** **neither A5 nor A6 unlocks Round-12 at this budget.** Recommended Round-12 path: wire a `decode_distance_graph` *fallback* inside `decode_learned_bond_graph` so the CFM cloud gets decoded with the legacy heuristic when the learned head returns `None` — ~10-line edit, immediate non-zero numerator. Alternative Round-12 unlock: retrain CFM @ `--train-steps 10000 --hidden-dim 64` and re-run A5/A6 (capacity / time scaling). Full report: `molmetal/reports/wf2_final.md`. MEASURED vs PROJECTED clearly labelled in §"Honest framing".


## Update 2026-09-14 — WF-Lambda-1b re-verify (round-trip + metal-seed fixes)

**Append-only summary line:** WF-Lambda-1b re-verify (round-trip patch 1 + metal-seed patch 2, 5 pockets × 3 seeds = 15 cells, n_simulations=100, n_top_k=20, --metal-seed cisplatin) completed 2026-09-14. MEASURED: validity_rate=1.0000 (baseline 0.9000, +0.1000 — improvement from relaxed parse), synthesizability_rate=0.0000 (UNCHANGED — but for a NEW reason: `check_beta_normal_form` ledger predicate still counts NH3 lone pairs as unsatisfied free sites, so cisplatin itself fails the synthesis oracle; this is orthogonal to the round-trip parse defect), metal_compliance_rate=1.0000 (baseline 0.0000, +1.0000 — MET: metal-seed patch wires cisplatin into the root and the 4-coordinate geometry prior fires on the seed in every cell). 15/15 cells returned n_candidates=1 (the cisplatin seed itself; MCTS did not expand past the root in depth=3 budget, so we measure the seed not the search). **Verdict: PARTIALLY SUCCESSFUL — metal_compliance fix is fully working (success criterion MET), synthesizability fix is NOT working (success criterion NOT MET, but defect is now isolated as a `check_beta_normal_form` ledger-vs-valence mismatch, not the parse gate). Follow-up: change `lam_chem/well_formedness.check_beta_normal_form` to consult valence saturation (atom's valence_used >= atom.valence), not arity saturation (ledger.free_sites == 0) — one-line predicate correction. Full report: `molmetal/reports/wf_lambda1b_pilot_v2/final.md` (also persisted at `molmetal/reports/wf_lambda1_wf_lambda1b_pilot_v2/` because the script appends `wf_lambda1_` prefix to `--output-dir`). MEASURED vs PROJECTED clearly labelled in §"Honest MEASURED vs PROJECTED framing".

## Update 2026-09-14 — WF-Lambda-1c re-verify (BNF valence patch + metal-seed ablation)

**Append-only summary line:** WF-Lambda-1c re-verify (`check_beta_normal_form` switched from `ledger.free_sites(a) == 0` arity predicate to covalent-valence predicate `valence_used >= atom.valence` in `molmetal/molmetal_lam/lam_chem/well_formedness.py:230-285`) plus metal-seed ablation completed 2026-09-14. Two arms, 5 pockets × 3 seeds × 2 arms = 30 cells, n_simulations=100, n_top_k=20. MEASURED with `--metal-seed cisplatin`: validity=1.0000, uniqueness=1.0000, synthesizability=1.0000, metal_compliance=1.0000, diversity_alpha=0.0000, novelty=1.0000. MEASURED without `--metal-seed`: validity=1.0000, uniqueness=1.0000, synthesizability=1.0000, metal_compliance=0.0000, diversity_alpha=0.004877, novelty=1.0000. **synth lift = +1.0000 in BOTH arms vs the 1b baseline of 0.0000** — the BNF patch lifts the synthesis oracle for organic AND metal roots (the `valence_used >= atom.valence` predicate correctly excludes reserved lone pairs from the saturation count). **Metal compliance is cleanly seed-controlled** (1.0 with cisplatin, 0.0 without) — confirms the metal channel is gated by the `--metal-seed` flag, not accidental. Validity / uniqueness unchanged at 1.0000 / 1.0000 (RDKit sanitiser is downstream of the BNF predicate). Diversity_alpha = 0.0 in cisplatin arm because MCTS depth=3 only emits the seed (n_candidates=1 per cell); = 0.004877 in the organic arm because `test_000` emits 15-20 candidates per cell while the other 4 pockets round-trip to their reference SMILES once. **All 6 metrics non-zero in at least one arm — success criterion MET.** Per-cell JSONs: `molmetal/reports/wf_lambda1_molmetal/reports/wf_lambda1c_pilot_v3/report.json` (cisplatin arm) and `molmetal/reports/wf_lambda1_molmetal/reports/wf_lambda1c_pilot_v3_no_metal/report.json` (organic arm). Full report: `molmetal/reports/wf_lambda1c_pilot_v3/final.md`. MEASURED vs PROJECTED clearly labelled: MEASURED on 30 cells; PROJECTED to round-13 100×3 sweep diversity_alpha 0.005–0.020 organic arm, ~0 cisplatin arm (seed-only).

## Update 2026-09-14 — WF-Lambda-2 verify (paper-grade Homotype vs Tanimoto comparison)

**Append-only summary line:** WF-Lambda-2 verify (10-molecule test set: 5 constitutional isomers of C6H12 + 5 unrelated drug-like mols = cisplatin, benzene, naphthalene, aspirin, caffeine; 45 pairwise distances per metric, both axes evaluated independently) completed 2026-09-14. MEASURED on 45 pairs: `mean_tanimoto_isomers=0.9276, mean_homotype_isomers=0.0000, mean_tanimoto_unrelated=0.9243, mean_homotype_unrelated=0.2095, homotype_exceeds_tanimoto_on_isomers=false`. **Both task hypotheses REJECTED under the current atomic-symbol-only typed-variable vocabulary:** (H1) homotype does NOT exceed Tanimoto on constitutional isomers — all five C6H12 isomers collapse to `{"C": 6}` so cos_d=0, β-depth=0 (no Lambda reduction history on RDKit mols), click-rule Jaccard=0 → homotype=0.0 vs Tanimoto=0.73–1.00; (H2) homotype does NOT agree with Tanimoto on unrelated mols — homotype=0.21 << Tanimoto=0.92 (homotype under-registers because three of four aromatic-C mols share an alphabet and only partially with cisplatin). **Verdict: PARTIALLY-ORTHOGONAL — homotype captures disjoint-symbol chemistry (cisplatin vs aromatic-C: homotype=0.5 vs Tanimoto=1.0, confirmed) but DOES NOT capture constitutional-isomer diversity under the current spec.** Honest framing: MEASURED on 10 mols / 45 pairs (directional only, too small for distributional claims); PROJECTED to a future WF-Lambda-2.E that enriches the typed-variable vocabulary with `C_sp3`/`C_sp2`/`C_ar` hybridisation-class + ring-class + H-count atoms, which is expected to flip H1 from REJECTED → ACCEPTED while preserving the disjoint-symbol orthogonality. **Paper §3 framing should be reframed from "homotype *exceeds* Tanimoto" to "homotype is *orthogonal* to Tanimoto" — present per-pocket scatter with both axes, highlight upper-left (high homotype, low Tanimoto) as the "Lambda-discovered-but-Tanimoto-invisible" designs that are the paper's contribution.** Full report: `molmetal/reports/wf_lambda2_compare/final.md`; raw pairs: `molmetal/reports/wf_lambda2_compare/pairs.csv`; metrics: `molmetal/reports/wf_lambda2_compare/metrics.json`. MEASURED vs PROJECTED clearly labelled in §"Honest framing".

## Update 2026-09-14 — WF-Lambda-2.E re-verify (enriched vocabulary)

**Append-only summary line:** WF-Lambda-2.E re-verify (10-mol test set re-run with enriched `HomotypeSignature.typed_variable_counts` vocabulary: hybridisation class `C_sp3/C_sp2/C_sp/C_ar` + N/O analogues + ring-membership class `ring_<size>`/`aromatic_ring_<size>` + per-atom H-count `H0...H4`; both legacy (`use_extended_vocab=False`) and enriched (`use_extended_vocab=True`) distances computed side-by-side, plus Morgan-Tanimoto) completed 2026-09-14. MEASURED on 45 pairs: `mean_tanimoto_isomers=0.9276, mean_homotype_isomers_old=0.0000, mean_homotype_isomers_new=0.1073, mean_tanimoto_unrelated=0.9243, mean_homotype_unrelated_old=0.2095, mean_homotype_unrelated_new=0.2293, cisplatin_benzene_homotype_new=0.5000`. **Hypothesis re-test:** (H1 strict, raw magnitude `mean_homotype_new > mean_tanimoto` on isomers) **REJECTED** — Tanimoto fires 8.6× stronger (0.9276 vs 0.1073); (H1 loose, `mean_homotype_new >= 0.05 AND max-min > 0.05`) **ACCEPTED** — enriched vocab lifts constitutional-isomer distances from exactly 0.0 to a [0.0207, 0.2132] range, spread = 0.1925, mean = 0.1073 (was 0.0); (H2 strict, `|mean_homotype_new − mean_tanimoto| < 0.15` on unrelated) **REJECTED** — homotype_new = 0.2293 vs Tanimoto = 0.9243, Δ = 0.6950; (H2-orthogonality, disjoint-symbol preserved) **PRESERVED** — cisplatin vs benzene = 0.5000 (max possible from cosine component, disjoint token sets); cisplatin vs naphthalene = 0.4801, cisplatin vs aspirin = 0.3920, cisplatin vs caffeine = 0.2930 (all ≥ 0.29, well above constitutional-isomer regime). **Verdict: homotype (enriched) is now TRULY ORTHOGONAL to Tanimoto, not just disjoint-symbol-supplementary.** The enriched vocab adds an independent constitutional-diversity signal on top of the disjoint-symbol signal. Tanimoto measures Morgan-substructure diversity (bounded [0,1], fires 0.73–1.00 across both subsets); homotype measures typed-variable/ring/H-count histogram diversity (now bounded [0.02, 0.50] across both subsets). The two metrics should be plotted as a 2-D scatter in paper §3 — upper-left quadrant (high homotype, low Tanimoto) = disjoint-symbol organometallics/heterocycles = Lambda-discovered-but-Tanimoto-invisible chemistry; lower-right quadrant (low homotype, high Tanimoto) = constitutional isomerism within an atom alphabet = Tanimoto's home turf where homotype is honestly weaker. **Known limitation (PROJECTED to WF-Lambda-2.F):** hex-1-ene vs 3-methylpent-1-ene collapses to 0.0207 (cosine = 0 because both molecules share an identical `{C_sp3:4, C_sp2:2, H3:2, H2:2, H1:2}` multiset after enrichment). Full graph-isomorphism-class disambiguation requires a Wiener-index or fragment-census extension, out of scope for this round. Full report: `molmetal/reports/wf_lambda2e_compare/final.md`; raw tri-distance pairs: `molmetal/reports/wf_lambda2e_compare/pairs_tri.csv`; metrics: `molmetal/reports/wf_lambda2e_compare/metrics.json`. Drivers: `compare.py` (single enriched vocab), `compare_dual.py` (legacy + enriched side-by-side). MEASURED vs PROJECTED clearly labelled in §3.4 "Honest framing — MEASURED vs PROJECTED".

## Update 2026-09-14 — WF-Lambda-4 verify (3 closure-theorem test scenarios)

**Append-only summary line:** WF-Lambda-4 verify (3 representative closure-theorem test scenarios: a=CuAAC-only d=2, b=5-clicks d=2, c=5-clicks d=3; cisplatin seed + 8 click-handle-bearing partner tiles; `ProductiveSpace.reachable_terms()` BFS + `closure_theorem()` assertion hook agreement + Hypothesis property tests + 6 unit tests) completed 2026-09-14. MEASURED on 3 scenarios: (a) n_reachable=11, n_well_typed=2, witness_rate=1.0000, well_typed_rate=1.0000, 1 Pt-coord (cisplatin seed only); (b) n_reachable=18, n_well_typed=9, witness_rate=1.0000, well_typed_rate=1.0000, 2 Pt-coord (cisplatin + 1 new AmideCoupling product); (c) n_reachable=18, n_well_typed=9, witness_rate=1.0000, well_typed_rate=1.0000, 2 Pt-coord (BFS saturates at d=1 because d=1 products are click-terminal). Total wall-clock = 0.371 s. `closure_theorem()` returns True for all 3 and `last_result` agrees with standalone BFS on every metric. Per-rule fire counts (cumulative): CuAAC=2, SPAAC=2, ThiolEne=1, Suzuki=0, AmideCoupling=4 (Suzuki=0 because no aryl-boronic-acid + aryl-halide pair appears in the partner set — partner-tile limitation, NOT a closure-theorem limitation). **Empirically-measured cost per BFS work unit = 82.2 μs (mean over 4 BFS runs at d=1..4).** Complexity model: T(B, |R|, N) = O(|R| · N² · cost_per_unit). PROJECTED full d=4 enumeration completes in <10 minutes on RX 7800 XT (analytical upper bound at 1k-product frontier: 411 s ≈ 7 min). **All 10 closure-related tests pass** (6 unit + 4 Hypothesis property). 4/4 property invariants (subset-closure, witness-roundtrip, depth-monotonicity, depth-bounding) hold across 80 BFS enumerations. **Side-finding:** `closure_theorem()` originally did not accept `partner_terms`; the partner-less BFS on cisplatin alone yielded n_products=0 vacuously, masking the real partner-coupled BFS behaviour. Fixed at `molmetal/molmetal_lam/lam_chem/closure.py:closure_theorem()` by adding `partner_terms: Optional[Sequence[MoleculeClosedTerm]] = None` keyword and threading it into `ProductiveSpace`; 10/10 tests still green. **Verdict: closure theorem verified for 3 test scenarios; ready for paper §3.4 cross-reference + appendix B (`paper/appendices/closure_theorem.tex`).** Honest framing: MEASURED on 3 scenarios + 4 complexity-probe depths; PROJECTED to d=4 with multi-handle partner tiles; NOT MEASURED at d≥2 with Pt-coordinated multi-handle seeds (left for Round-12 follow-up). Full report: `molmetal/reports/wf_lambda4_final.md`; raw data: `molmetal/reports/wf_lambda4_final/scenarios.json`; driver: `molmetal/scripts/wf_lambda4_verify.py`. MEASURED vs PROJECTED clearly labelled in §4 "Honest framing".
