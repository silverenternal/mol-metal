# Wf-A200 -- CROSS-WAVE LOCATOR AUDIT (read-only)

## §A -- RECORD (2026-11-02) -- a DATED record of the locators it reports

root            : /home/hugo/codes/mol-metal
census method   : git ls-files (+ --others --exclude-standard; + 129 gitignored file(s) under 6 prefix(es): molmetal/reports/wfa158_frontier_postproc_matrix, molmetal/reports/wfa161_frontier_labelling, molmetal/reports/wfa175_band_provenance, molmetal/reports/wfa175_frontier_comparability, molmetal/reports/wfa175_synthesis, molmetal/reports/wfa186_frontier_primary_source)
files in census : 10230
files scanned   : 8314  (skipped large=16 binary=0)
locator hits    : 11982
wall seconds    : 17.574

### counts by class

* `RESOLVES-AND-MATCHES` : 386
* `RESOLVES-BUT-DIFFERS` : 108
* `RESOLVES-NO-CLAIM` : 6109
* `TARGET-ABSENT` : 320
* `TARGET-AMBIGUOUS` : 720
* `TARGET-IS-SELF` : 50
* `DATED-SUPERSESSION` : 3516
* `SNAPSHOT` : 773

### LIVE DRIFT (RESOLVES-BUT-DIFFERS) -- the actionable set: 108

* `TODO/completed/34_p7_zenodo_wrapper_done.md` line 101 -> `paper/sections/03_method.tex` line 120
    * the cited line now reads: the Round-10 6-seed $\times$ 2-pocket $\times$ 2-cfg $\times$
    * the claim was: the same class of failure on open-shell / invalid-valence
* `TODO/completed/75_blackwell_retrain.md` line 117 -> `TODO/03_plans/R18_PLAN.md` line 18
    * the cited line now reads: | m1 Ru AUC | ≥ 0.86 | 0.712 | 0.81 | 3-seed × 100-epoch + tmQM pretrain → 0.74 (no significant lift) | **BLOCK** |
    * the claim was: r17 baseline 0.712, r18 target 0.81, block
* `TODO/completed/a126_critic_residual_repair.md` line 42 -> `RECONCILIATION.md` line 51
    * the cited line now reads: - **MEASURED (2026-10-31, before restore):** `git status --porcelain -- results/wf124_p3_per_innovation_ablation_g6/` → **74 lines, every one ` D`**
    * the claim was: 2 lines, all d
* `TODO/completed/a126_critic_residual_repair.md` line 43 -> `RECONCILIATION.md` line 63
    * the cited line now reads: Post-restore: **0** deletions, **79** files on disk = 76 tracked + 2 gitignored `.log` + 1
    * the claim was: 139 files on disk
* `TODO/completed/plan_a_step1_dmpnn_value_wIRED.md` line 8 -> `bench_mcts_search_quality.py` line 104
    * the cited line now reads: n_simulations_executed: int = 0
    * the claim was: - a/b bench: molmetal/reports/r24_step1/ arma_control,armb_dmpnn /search_quality_bench.json
* `TODO/completed/wfa125_guard_and_gate_residuals.md` line 160 -> `TODO/completed/321d_preflight_gate_audit.md` line 47
    * the cited line now reads: ### §2.1 `all` pipeline — 129 gates
    * the claim was: 126 gates
* `TODO/completed/wfa125_guard_and_gate_residuals.md` line 172 -> `molmetal/scripts/pre_flight_check.py` line 10228
    * the cited line now reads: name, priority, pipeline, metric_path, repo_root=repo_root
    * the claim was: all 5 yamls have eval targets within baseline paper ranges
* `TODO/completed/wfa125_guard_and_gate_residuals.md` line 190 -> `molmetal/scripts/wf229_smoke_test.py` line 46
    * the cited line now reads: sys.exit(rc)
    * the claim was: superseded 2026-10-31 (wf-a127): rows 1 (the git-ignored cpu lock), 3
* `TODO/completed/wfa125_guard_and_gate_residuals.md` line 190 -> `molmetal/scripts/wf229_smoke_test.py` line 46
    * the cited line now reads: sys.exit(rc)
    * the claim was: superseded 2026-10-31 (wf-a127): rows 1 (the git-ignored cpu lock), 3
* `docs/SUBMISSION_PLAN.md` line 61 -> `PAPER_SCOPE.md` line 14
    * the cited line now reads: | strongest number | cross-metal collapse, see below | bond EXISTENCE F1 0.9961 vs OpenBabel 0.9771 and exact-type rate 0.0978 vs 0.0300, same 900 com
    * the claim was: bond_type_f1 0.9194 vs openbabel 0.505
* `docs/wf_a44_verify_receipts.md` line 22 -> `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` line 8261
    * the cited line now reads: if _ring_weight > 0.0 and getattr(model, "last_ring_prior_inputs", None):
    * the claim was: fragment_loss_weight 0.05
* `docs/wf_a44_verify_receipts.md` line 24 -> `molmetal/loss/regularization_losses.py` line 85
    * the cited line now reads: return
    * the claim was: entropy_term_weight 0.1
* `docs/wf_a44_verify_receipts.md` line 24 -> `train_cfm_yuelbond_wasserstein.py` line 7918
    * the cited line now reads: #     Default OFF (False); when ON, the per-cohort config is
    * the claim was: entropy_term_weight 0.1
* `docs/wf_a44_verify_receipts.md` line 25 -> `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` line 3141
    * the cited line now reads: ``dropout=0.10``, but it is not the head this trainer builds.)
    * the claim was: consistency_term_weight: float
* `docs/wf_a44_verify_receipts.md` line 26 -> `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` line 8271
    * the cited line now reads: ring_prior_term = torch.zeros((), device=device)
    * the claim was: molecular_property_target_weight 0.05
* `docs/wf_a44_verify_receipts.md` line 27 -> `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` line 513
    * the cited line now reads: if _has_metal(mol):
    * the claim was: 511,531p
* `molmetal/reports/cfm_c7_fullgen_2026_10_20/synthesis.md` line 156 -> `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` line 20159
    * the cited line now reads: assemble_fragments=float(
    * the claim was: :20440, :20553, :20578) and the trainer s own comment records why
* `molmetal/reports/cfm_c7_fullgen_2026_10_20/synthesis.md` line 187 -> `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` line 1614
    * the cited line now reads: # budget, not a hard-coded 1.0.  The hard-coded 1.0 was an
    * the claim was: remains unobtainable for this pool, and the dominant cause of the
* `molmetal/reports/paper_outline.md` line 34 -> `TODO/08_references/sota_landscape.md` line 35
    * the claim was: mol-metal s egnn + flow matching is exactly the η-conversion + β-reduction operators of the molecular lambda calculus.
* `molmetal/reports/paper_outline.md` line 34 -> `TODO/01_research/sota_landscape.md` line 120
    * the cited line now reads: **Description:** REINVENT4 from AstraZeneca is the definitive open-source framework for AI-driven molecular design, supporting de novo design, scaffol
    * the claim was: mol-metal s egnn + flow matching is exactly the η-conversion + β-reduction operators of the molecular lambda calculus.
* `molmetal/reports/paper_outline.md` line 34 -> `TODO/13_lambda_clickchem/molecular_lambda_calculus.md` line 512
    * the cited line now reads: 1. **形式化基础** (Foundational): 第一次把化学定义为 λ-演算,把药物设计定义为构造性证明搜索。
    * the claim was: mol-metal s egnn + flow matching is exactly the η-conversion + β-reduction operators of the molecular lambda calculus.
* `molmetal/reports/paper_outline.md` line 212 -> `molmetal/reports/honest_baseline_summary.md` line 152
    * the claim was: gbdt outperforms d-mpnn when (a) the dataset is large enough for 500-tree bagging to converge (ru n 19 k), and (b) the t
* `molmetal/reports/ultracode_audit/wf1_recon_decoder.md` line 43 -> `r10_cfg_real_crossdocked.py` line 42
    * the claim was: s own early-rejection atom_outside_training_vocabulary becomes dead-code on the masked path (no z vocab ever reaches it)
* `molmetal/reports/ultracode_audit/wf2_recon_cfm_train.md` line 25 -> `molmetal/models/bond_head.py` line 340
    * the cited line now reads: Conservative on purpose: True when an explicit
    * the claim was: head consumes pre-computed distances and angle deviations
* `molmetal/reports/ultracode_audit/wf2_recon_cfm_train.md` line 76 -> `molmetal/reports/ultracode_audit/wf1_recon_training_data.md` line 21
    * the cited line now reads: The tmQM loader (`molmetal/data/tmqm.py:232-291`) merges three shards (`parse_xyz_headers`, `parse_bo_metal`, `parse_properties`) keyed on `csd_code`,
    * the claim was: a sibling parse_bo_all() would emit one row per atom-pair with bo 0.05
* `molmetal/reports/wf281c_g6_readiness/readiness_report.md` line 24 -> `scripts/launch_wf124_p3_g5.sh` line 100
    * the cited line now reads: # After flag-scan, capture positional --prereg <path> if not already set
    * the claim was: --no-enforce-budget so all 15 planned
* `molmetal/reports/wf282c_lint_pass/lint_report.md` line 64 -> `bench_catherine_30metals.py` line 148
    * the cited line now reads: n = max(1, min(30, int(n)))
    * the claim was: sign-flipped from previous buggy
* `molmetal/reports/wf287c_cf266_gate_audit/audit_report.md` line 34 -> `train_chemprop_v2_v3.py` line 1524
    * the cited line now reads: "MasterWassersteinGradientFlow n_mc when --use-master-wass "
    * the claim was: v3-fallback
* `molmetal/reports/wf42_mock_audit/mock_inventory.md` line 530 -> `verify_paper_build.py` line 178
    * the cited line now reads: "sections/03_4_se3_schrodinger_bridge.tex": "§3.4 SE(3) Schrödinger Bridge (Wf-3 ship, W2.1)",
    * the claim was: sections/03_method.tex
* `molmetal/reports/wf_a40_m3_real/BLOCKED.md` line 63 -> `molmetal/scripts/train_cisplatin_resistance.py` line 351
    * the cited line now reads: elif k in own:
    * the claim was: ic50_mcf7_nm
* `molmetal/reports/wf_bug2_fix/final.md` line 14 -> `TODO/INDEX.md` line 212
    * the cited line now reads: > state. The m21 figure `0.8007±0.0160` was replaced twice. The BLK-PA12
    * the claim was: bug-2: pocket_macro_inference.py:101,104 deflex v2 checkpoint
* `molmetal/reports/wf_bug2_fix/final.md` line 16 -> `pocket_macro_inference.py` line 101
    * the cited line now reads: #: File name of the trained state-dict checkpoint (no directory).
    * the claim was: cwd-relative path (5-line fix)
* `molmetal/reports/wf_cfm_frontier_research/final.md` line 97 -> `molmetal/adapters/flow_matching_lipman/__init__.py` line 2439
    * the cited line now reads: state_dict = ckpt["state_dict"]
    * the claim was: midpoint
* `molmetal/reports/wf_deflex_pocket_macro_skeleton/phase3_ca2_fix.md` line 147 -> `pocket_macro_skeleton.py` line 35
    * the claim was: linear(per_residue_features, hidden_dim 32)
* `molmetal/reports/wf_lambda_div_rotation/integrate.md` line 125 -> `r4_lambda_only_run.py` line 1466
    * the cited line now reads: vals.append(v)
    * the claim was: is lifted so the cli --n-simulations 1000 flag actually
* `molmetal/reports/wf_last_cpu_fixes/MASTER.md` line 12 -> `pocket_macro_inference.py` line 101
    * the cited line now reads: #: File name of the trained state-dict checkpoint (no directory).
    * the claim was: models/pocket_macro_skeleton_v2.pt
* `molmetal/reports/wf_lift_n_sim_cap/final.md` line 7 -> `r4_lambda_only_run.py` line 2063
    * the cited line now reads: from molmetal_lam.search_alg.proof_search import RewardAggregator
    * the claim was: works end-to-end cell.n_simulations 1000 in every cell record).
* `molmetal/reports/wf_pitfall_audit/p2_generator_arch.md` line 391 -> `posebusters_adapter.py` line 227
    * the cited line now reads: smiles=smiles,
    * the claim was: wf-pb-dock-mode-wire 2026-09-14
* `molmetal/reports/wf_pitfall_audit/p4_eval_novelty.md` line 15 -> `molmetal/molmetal_lam/lam_chem/tmqm_dataset.py` line 683
    * the cited line now reads: for _ in range(1, n):
    * the claim was: cisplatin + carboplatin + oxaliplatin + nedaplatin + satraplatin
* `molmetal/reports/wf_pitfall_audit/p5_wetlab_validation.md` line 22 -> `paper/section_06_limitations.tex` line 19
    * the cited line now reads: Finally, the anticancer metric suite provides chemical proxies.  hERG,
    * the claim was: wet-lab cytotoxicity are not measured by the present pipeline and require dedicated predictors or experiments
* `molmetal/reports/wf_pitfall_audit/p5_wetlab_validation.md` line 56 -> `paper/section_07_future_work.tex` line 14
    * the cited line now reads: and ultimately wet-lab validation are required before therapeutic claims can
    * the claim was: broader metal coverage, explicit dna-fragment docking, redox and aquation predictors, and ultimately wet-lab validation 
* `molmetal/reports/wf_pivot_followup/f1_verify.md` line 58 -> `04_evaluation.tex` line 844
    * the cited line now reads: top-1 was produced. The smoke test was therefore not executed; we
    * the claim was: hybrid vs de novo typed-term mcts ablation
* `molmetal/reports/wf_pivot_followup/f2_platinai_s4.md` line 27 -> `test_platinai_dataset.py` line 152
    * the cited line now reads: train = PlatinAIDataset(n_max=n, split="train")
    * the claim was: ml-predicted, not wet-lab
* `molmetal/reports/wf_pivot_followup/f3_crossref.md` line 155 -> `04_evaluation.tex` line 1815
    * the cited line now reads: \toprule
    * the claim was: candidates not present in the platinai corpus return null and are excluded from the per-pocket numerator.
* `molmetal/reports/wf_remove_smoke/phase3g_reports_done.md` line 123 -> `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py` line 557
    * the cited line now reads: n_simulations=10,
    * the claim was: wf_lambda1b_smoke_cisplatin
* `molmetal/reports/wf_remove_smoke/phase3g_reports_done.md` line 124 -> `scripts/generate.py` line 279
    * the cited line now reads: Returns
    * the claim was: smoke tests
* `molmetal/reports/wf_round12_lambda_patha_10x3/final.md` line 19 -> `molmetal/scripts/r4_lambda_only_run.py` line 1716
    * the cited line now reads: # vector from voxelization.py:208,255 / vina_adapter.py:961 is the
    * the claim was: 4. partner tiles (partner_tiles_v2 with 3 azides + 3 boronic
* `molmetal/reports/wf_round12_sota_subset/final.md` line 69 -> `inference.py` line 120
    * the claim was: models not found locally and failed to download them from remote_urls
* `molmetal/reports/wfa163_c7_rederive/report.md` line 152 -> `.../generated_smiles.csv` line 9848
    * the claim was: maximum bfs search size exceeded
* `molmetal/reports/wfa163_refute_gpucheck/verdict.md` line 47 -> `molmetal/tests/test_registry_generate_import_path.py` line 54
    * the cited line now reads: return subprocess.run(
    * the claim was: pythonpath
* `molmetal/reports/wfa163_refute_v3/verdict.md` line 56 -> `wfa163_c7_rederive.py` line 212
    * the cited line now reads: comps_full = components(adj, all_idx)
    * the claim was: identical under all four vocabularies
* `molmetal/reports/wfa163_synthesis.md` line 144 -> `molmetal/tests/test_registry_generate_import_path.py` line 54
    * the cited line now reads: return subprocess.run(
    * the claim was: pythonpath
* `molmetal/reports/wfa166_run_ledger_census/census.md` line 58 -> `molmetal/scripts/run_ledger.py` line 215
    * the cited line now reads: data_sha = _sha256_file(dp)
    * the claim was: reformatted
* `molmetal/reports/wfa166_run_ledger_census/census.md` line 58 -> `run_ledger.py` line 322
    * the cited line now reads: "ledger_value": None,
    * the claim was: reformatted
* `molmetal/reports/wfa166_synthesis/synthesis.md` line 22 -> `run_ledger.py` line 215
    * the cited line now reads: data_sha = _sha256_file(dp)
    * the claim was: reformatted
* `molmetal/reports/wfa166_synthesis/synthesis.md` line 22 -> `run_ledger.py` line 322
    * the cited line now reads: "ledger_value": None,
    * the claim was: reformatted
* `molmetal/reports/wfa166_synthesis/synthesis.md` line 171 -> `TODO/completed/321d_preflight_gate_audit.md` line 50
    * the cited line now reads: > live import: `len(build_gates("all")) == 129`,
    * the claim was: 129 gates
* `molmetal/reports/wfa169_b2_refutation/refutation.md` line 28 -> `AGENTS.md` line 36
    * the cited line now reads: 4. ~~**Pre-flight v2 25/25 PASS gate**~~ → **Pre-flight v2 PASS gate — 129 gates on the
    * the claim was: 129 gates (131 on launch)
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1610 -> `molmetal/scripts/pre_flight_check.py` line 5879
    * the cited line now reads: # re-measurement).  See PIPELINES and the "launch" entry below.
    * the claim was: chemprop_v2_v3
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1631 -> `molmetal/scripts/pre_flight_check.py` line 10643
    * the cited line now reads: in :data:`EXPECTED_GATE_COUNTS`.
    * the claim was: expected_gate_counts constant). update the lambda
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1633 -> `molmetal/scripts/pre_flight_check.py` line 10655
    * the cited line now reads: drift the auto-compute mode had been hiding).  Only the unit tests
    * the claim was: build_gates( pipeline r ) raised: type(exc).__name__ : exc
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1639 -> `molmetal/scripts/pre_flight_check.py` line 10674
    * the cited line now reads: cause is direct in the evidence string.  The actual drift check
    * the claim was: expected_total was omitted, so this compares the count
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1640 -> `molmetal/scripts/pre_flight_check.py` line 10675
    * the cited line now reads: is unchanged for valid pipelines.
    * the claim was: to itself; pass expected_total expected_gate_counts
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1641 -> `molmetal/scripts/pre_flight_check.py` line 10679
    * the cited line now reads: (actual > expected = gate bloat; actual < expected = silent
    * the claim was: (explicit expected_total drift check active)
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1642 -> `molmetal/scripts/pre_flight_check.py` line 10685
    * the cited line now reads: ``build_gates("all")`` now passes an EXPLICIT ``expected_total``
    * the claim was: gate count consistent for pipeline pipeline r :
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1643 -> `molmetal/scripts/pre_flight_check.py` line 10686
    * the cited line now reads: (``EXPECTED_GATE_COUNTS["all"]``), so the audit gate on the commit
    * the claim was: actual actual_total expected expected_total suffix
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1645 -> `molmetal/scripts/pre_flight_check.py` line 10699
    * the cited line now reads: return GateResult(
    * the claim was: gate count drift for pipeline pipeline r : actual actual_total
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1646 -> `molmetal/scripts/pre_flight_check.py` line 10700
    * the cited line now reads: name,
    * the claim was: vs expected expected_total [ direction ].
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1654 -> `molmetal/scripts/pre_flight_check.py` line 15669
    * the cited line now reads: (
    * the claim was: p3_gate_count_consistency_audit
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1656 -> `molmetal/scripts/pre_flight_check.py` line 15671
    * the cited line now reads: lambda: gate_wf321e_ablation_registry_complete(
    * the claim was: p3_gate_count_consistency_audit
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 1811 -> `molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py` line 265
    * the cited line now reads: text = PRE_FLIGHT_SCRIPT.read_text()
    * the claim was: expected_gate_counts[ frontier ] should be 24 (was 23)
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2330 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 107
    * the cited line now reads: # "the 24 gates were DELETED ON PURPOSE, not lost" — is in
    * the claim was: 3.4 expected_gate_counts[
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2331 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 109
    * the cited line now reads: #
    * the claim was: expected_gate_counts[ all ] must be 127 ( 95 + 32); got
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2332 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 110
    * the cited line now reads: # This test was written against the Wf-432-A1 design and has been RED since
    * the claim was: _pfc.expected_gate_counts[ all ] . per wf-345-b3 the runtime
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2333 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 111
    * the cited line now reads: # 2026-10-02.  It is repaired here to assert the DECIDED state:
    * the claim was: len(build_gates( all )) is the authoritative source; this
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2334 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 117
    * the cited line now reads: # Nothing was lost from the codebase: all 32 gate FUNCTIONS are still
    * the claim was: 3.4 build_gates( all ) unique names count 127 (auto-recompute).
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2336 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 121
    * the cited line now reads: LIVE_WF432A1_GATES = (
    * the claim was: unique gate names in build_gates( all ) must be 127; got
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2338 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 127
    * the cited line now reads: "p3_mock_to_real_m17_drugood_assay_REAL_3_seeds",
    * the claim was: 2- 6 every new gate is registered in build_gates( all ).
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2340 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 133
    * the cited line now reads: "p3_honest_neg_closure_m1_re_fold",
    * the claim was: build_gates( all ) missing len(missing) new gates: missing
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2341 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 341
    * the cited line now reads: for pipeline in ("frontier", "chemprop_v2_v3", "cfm"):
    * the claim was: 3.4 every new gate appears in build_gates( all ) (not frontier/cfm).
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2343 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 344
    * the cited line now reads: assert runtime == expected, (
    * the claim was: gate_name missing from build_gates( all )
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2344 -> `molmetal/tests/test_wf433a1_gate_v2_wire.py` line 346
    * the cited line now reads: f"(= EXPECTED_GATE_COUNTS[{pipeline!r}]); got {runtime}."
    * the claim was: frontier
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2413 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 78
    * the cited line now reads: "molmetal/tests/test_wf415a3_preflight_fail_audit.py",
    * the claim was: measured by live import: live expected_gate_counts, per pipeline.
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2416 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 83
    * the cited line now reads: #: The files whose stale pins the Wf-A172 repair retired.  A census that
    * the claim was: update expected_gate_counts in pre_flight_check.py.
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2434 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 188
    * the cited line now reads: target.write_text(body, encoding="utf-8")
    * the claim was: the expected_gate_counts[
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2437 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 301
    * the cited line now reads: f"{target} is not covered by the pin census at all — "
    * the claim was: names [n for n, _ in build_gates( all ) if n.startswith( p3_ )] n
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2438 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 360
    * the cited line now reads: """A monotonic floor (``>= 71``) is never a stale pin.
    * the claim was: import molmetal.scripts.wfa169_gate_count_pins as m;
* `molmetal/reports/wfa169_dead_gate_repair/prose_mentions.txt` line 2439 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 362
    * the cited line now reads: The repo's own convention (Wf-345A2) is that a hard equality fires on
    * the claim was: all : m.live_gate_counts()[ all ] ))
* `molmetal/reports/wfa169_dead_gate_repair/registration_table.md` line 47 -> `test_wf433a1_gate_v2_wire.py` line 340
    * the cited line now reads: """
    * the claim was: no main-layer-0 routing
* `molmetal/reports/wfa169_dead_gate_repair/registration_table.md` line 58 -> `pre_flight_check.py` line 14002
    * the cited line now reads: priority,
    * the claim was: metalcytotox_5metal_multitask_v4.csv
* `molmetal/reports/wfa169_dead_gate_repair/verdict_table.md` line 117 -> `pre_flight_check.py` line 17595
    * the cited line now reads: file=sys.stderr,
    * the claim was: if n_fail 0: return 1. warn has no branch in that tail, so the 4
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 791 -> `docs/HONEST_NEGATIVE_CATALOG.md` line 20817
    * the cited line now reads: | **DENOMINATOR** | 92.7–94 % measured over **10,000–15,000** generated molecules from a training-set-distribution sampling | **ONE** pool of **2,112*
    * the claim was: denominator 92.7-94% measured over 10,000-15,000 generated molecules
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 1174 -> `molmetal/reports/wfa161_frontier_labelling/report.md` line 139
    * the cited line now reads: 2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel) and a different post-processing, so it is not on
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 1175 -> `molmetal/reports/wfa161_frontier_labelling/report.md` line 139
    * the cited line now reads: 2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel) and a different post-processing, so it is not on
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 1183 -> `molmetal/scripts/wfa161_frontier_labelling.py` line 934
    * the cited line now reads: validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
    * the claim was: stated fragment handling at all (geom-dd, ggflow). our 92.7-94% band comes from a
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 1286 -> `molmetal/reports/wfa161_frontier_labelling/report.md` line 139
    * the cited line now reads: 2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel)
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 1287 -> `molmetal/reports/wfa161_frontier_labelling/report.md` line 139
    * the cited line now reads: 2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel)
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 1374 -> `molmetal/scripts/wfa161_frontier_labelling.py` line 934
    * the cited line now reads: validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel) and a different post-processing
* `molmetal/reports/wfa175_band_provenance/_occurrences_v1_dash_any_v2_incl_ignored.json` line 1375 -> `molmetal/scripts/wfa161_frontier_labelling.py` line 934
    * the cited line now reads: validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel) and a different post-processing
* `molmetal/reports/wfa175_band_provenance/refutation/_write_refutation.py` line 967 -> `molmetal/scripts/wf_a155_ligand_connected_rate.py` line 223
    * the cited line now reads: #: silent edit of history.  The dated note that supersedes it is
    * the claim was: the superseded rendering asserted one provenance reading as settled:
* `molmetal/reports/wfa175_band_provenance/report.md` line 139 -> `molmetal/scripts/wfa161_frontier_labelling.py` line 934
    * the cited line now reads: validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel) and a different
* `molmetal/reports/wfa175_band_provenance/report.md` line 143 -> `molmetal/reports/wfa161_frontier_labelling/report.md` line 139
    * the cited line now reads: 2. **Post-processing.** The frontier's connectivity rates are NOT computed on the atom-count largest connected component after RDKit sanitization: EF-
    * the claim was: generated from the script, so the defect has two on-disk surfaces)
* `molmetal/reports/wfa175_band_provenance/report.md` line 344 -> `molmetal/scripts/wfa161_frontier_labelling.py` line 934
    * the cited line now reads: validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
    * the claim was: open babel
* `molmetal/reports/wfa175_pin_repair_refutation/report.md` line 255 -> `test_wfa159_timeout_verdict_census.py` line 430
    * the cited line now reads: assert len(pre_flight.build_gates("all")) == 129
    * the claim was: 0 stale pins repo-wide
* `molmetal/reports/wfa175_synthesis/synthesis.md` line 34 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 289
    * the cited line now reads: Wf-A172 (2026-11-01) moved ten stale count pins across three files
    * the claim was: stale 15 4
* `molmetal/reports/wfa175_synthesis/synthesis.md` line 58 -> `molmetal/scripts/wfa161_frontier_labelling.py` line 934
    * the cited line now reads: validity on QM9 -- a SIX-CHECK pose verdict (bond lengths/angles,
    * the claim was: our 92.7-94% band comes from a different perception branch (open babel) and a different post-processing
* `molmetal/reports/wfa175_synthesis/synthesis.md` line 172 -> `molmetal/tests/test_wfa169_gate_count_pins.py` line 289
    * the cited line now reads: Wf-A172 (2026-11-01) moved ten stale count pins across three files
    * the claim was: stale 15 4
* `molmetal/reports/wfa175_synthesis/synthesis.md` line 187 -> `molmetal/reports/wfa175_refs_bib_audit/refs_corrected_STAGED.bib` line 399
    * the cited line now reads: number       = {12},
    * the claim was: crossref issue 12; pubmed 63(12):8749-8760

## §B -- HOW TO READ THIS RECORD (its classes, its limits, its errors)

### §B.1 the classes

| class | meaning | actionable? |
|---|---|---|
| `RESOLVES-BUT-DIFFERS` | the cited path resolves to ONE file, the cited line exists, and the quoted claim next to the locator is NOT on that line | **YES -- this is the defect class** |
| `RESOLVES-AND-MATCHES` | the cited line does carry the claim | no |
| `RESOLVES-NO-CLAIM` | the locator resolves, but the text next to it is a POINTER (a sub-reference inside a longer sentence), so the content is not machine-verifiable | no -- reported, not asserted |
| `DATED-SUPERSESSION` | the occurrence sits inside a DATED section, key or constant -- a supersession record that AGENTS.md 5.5 rule #6 requires to survive | no |
| `SNAPSHOT` | the cited file is pinned by sha256, or is a file INSIDE a sha256-pinned e-print tarball, so its line numbers cannot drift | no |
| `TARGET-ABSENT` | the cited path names no file in the census | usually a stale path, but often just a foreign path |
| `TARGET-AMBIGUOUS` | the cited path matches SEVERAL files, so it does not identify one artefact and cannot be machine-verified | no |
| `TARGET-IS-SELF` | the cited path resolves to the citing file itself | no |

### §B.2 what the CLAIM next to a locator is, and how it is chosen

A locator asserts drift only when the text next to it is a MACHINE-VERIFIABLE claim about the cited line.  The claim is:

1. a quoted string that starts within 120 characters AFTER the locator; or
2. a markdown block quote on one of the next three lines; or
3. otherwise a POINTER -- reported, never asserted as drift.

A locator that sits INSIDE a quoted phrase is always a POINTER: the quote is about something else and the locator is a sub-reference inside it -- the shape of a QUOTED OLD TITLE such as "'THE ONLY MIS-ATTRIBUTION IS report.md:139' IS INCOMPLETE".

### §B.3 the KNOWN error modes -- read the findings with these in hand

* **FALSE POSITIVE (the likely one).**  The claim is taken from the text NEXT TO the locator, not from the sentence that carries the citation's subject.  A line that packs several citations and several quotes together can attach the wrong quote to a locator.  Every finding in this record prints both "the cited line now reads" and "the claim was" -- READ BOTH before acting on a finding.
* **FALSE NEGATIVE.**  A locator whose claim is a paraphrase, not a quotation, is a POINTER and is never asserted as drift.  The drift count is therefore a LOWER bound on the defect class.
* **UNSCANNED.**  This run's census method: `git ls-files (+ --others --exclude-standard; + 129 gitignored file(s) under 6 prefix(es): molmetal/reports/wfa158_frontier_postproc_matrix, molmetal/reports/wfa161_frontier_labelling, molmetal/reports/wfa175_band_provenance, molmetal/reports/wfa175_frontier_comparability, molmetal/reports/wfa175_synthesis, molmetal/reports/wfa186_frontier_primary_source)`.  It read 8314 of 10230 files in the census; 0 were skipped as binary and 16 as too large.  The census is `git ls-files` + untracked-not-ignored, so the repo's gitignored files (e.g. `molmetal/reports/**/*.json`) are NOT read unless the run is restricted to a prefix -- a restricted run adds the gitignored files UNDER that prefix.
* **NOT A GATE.**  This auditor is READ-ONLY.  It is not wired into `pre_flight_check.py`, it adds no gate, and it does not change `EXPECTED_GATE_COUNTS`.

read-only: True | gpu: 0 (CPU-only; no torch import, no CUDA call)
