# Wf-A175 — close-out brief: 5 read-only audit branches, adversarially refuted, merged

**Date:** 2026-11-01 · **CPU-only** (`CUDA_VISIBLE_DEVICES=""` + `/home/hugo/codes/mol-metal/.venv/bin/python`) · **GPU used: 0** · **no git mutation** · **writes confined to** `molmetal/reports/wfa175_synthesis/`.

Read-only respected: `paper/**`, `docs/HONEST_NEGATIVE_CATALOG.md`, `AGENTS.md`, `TODO/**`, `README.md`, `molmetal/scripts/pre_flight_check.py`, `metrics/**`.

## 0. Tally

- Branch claims: **1 REFUTED / 30 SURVIVED / 15 IMPRECISE** across 5 branches.
- Merged findings: **15** — Tier A (can corrupt a published number) 6, Tier B (can make a gate lie) 6, Tier C (documentation / bookkeeping) 3. **5 are BLOCKED** on `paper/**` write auth, this wave's read-only scope, or git write authority.

Every number below is **MEASURED** and traces to a path on disk. Nothing is PROJECTED; nothing is fabricated.

---

## 1. SURVIVED findings, ranked by blast radius

| # | tier | claim | evidence | remediation | blocked |
|---|---|---|---|---|---|
| A1 | A-published-number | Our "same 3-step post-processing as Buttenschoen" is 2 of 3 steps: the UFF step ran on 0 of 12,668 molecules, so the frontier report's direct-comparability claim is unsound as written. | `metrics/by_metric/cfm_fragmentation_postproc_arm.json -> postprocessed_axis.uff_ok_rate_over_postprocessed = 0.0, uff_reasons {"ValueError: Bad Confor` | Read-only: amend report.md:79 to "2 of 3 steps (LargestFragmentChooser -> Chem.AddHs); the UFF step is inapplicable to this corpus' metal complexes (0 | no |
| A2 | A-published-number | READING-SAFE: the exact source cells of the band are NOT settled on disk -- COMPETING readings (`band_provenance` vs `sources`.`EF-TALFM`); the NUMBER of readings is NOT settled on disk. | `frontier_matrix.json -> band_provenance (Reading A) vs the same file -> sources.EF-TALFM (Reading B); the numerals disagree (92.7 / 94.0 vs 92.98 / 94.33) -- the band's source cells are NOT settled on disk` | Read-only: the band must never be placed on OUR scale as a comparator, and its endpoint assignment must not be published as settled -- record the metr | no |
| A3 | A-published-number | The Open-Babel mis-attribution of the band is still LIVE in the generator, not just in stale prose: molmetal/scripts/wfa161_frontier_labelling.py:934 still emits "Our 92.7-94 % band comes fr | `molmetal/scripts/wfa161_frontier_labelling.py:930-936 (re-measured by synthesis` | Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT on this wave's protected list). Replace th | no |
| A4 | A-published-number | The band census is self-inflating and blind to the gitignored tree: 82 sites / 34 files is COMPLETE ONLY for tracked+untracked-not-ignored; 21 gitignored files outside the branch carry the b | `molmetal/reports/wfa175_band_provenance/sites.json (n_sites=82, distinct files=34, population="git ls-files --cached + git ls-files --others --exclude` | Read-only: re-run the census with the gitignored report population included, and close Wf-A171's OPEN-5. No GPU. This is a census-scope repair, not a  | no |
| A5 | A-published-number | The band report's own §2.2 is INVERTED against its own sites.json: report.md:106 says "For 6 of the 7 `post` sites the statement is *correct and per-cell*", but sites.json shows 6 of the 7 a | `molmetal/reports/wfa175_band_provenance/report.md:106 vs molmetal/reports/wfa175_band_provenance/sites.json (raw_or_post Counter {unstated 66, raw 9, ` | Read-only: correct report.md:106 to state that 6 of the 7 `post` sites are the blanket-REFUTED ones and only frontier_matrix.md:68 is correct-and-per- | no |
| A6 | A-published-number | The frontier report's §2.1 ratio column contradicts its OWN computed artifact: 4 of the ratios printed in the report disagree with molmetal/reports/wfa175_frontier_comparability/tools/ratios | `report.md:66 1.03175x vs ratios.json 1.0319905213270142` | Read-only: regenerate report.md's §2.1 ratio column from tools/ratios.json rather than re-typing it. Documentation-only blast radius (rounding magnitu | no |
| B1 | B-gate-truth | The refs_bib gate would FAIL (paper/refs.bib has 0 of the 3 required keys) but is DEAD in the working tree: it is DEFINED at pre_flight_check.py:11118 and registered in NO pipeline, so it ca | `pre_flight_check.py:11118 (gate def) - `grep` finds 1 occurrence = the def only` | Read-only: nothing to change today. For the follow-on wave, the bibliography fix and the gate wiring MUST land in ONE write, bibliography FIRST - land | **YES** |
| B2 | B-gate-truth | 88 TRACKED CONFIGS declare a required pre-flight gate that no pipeline owns: 82 configs name `p1_pytorch_cuda_available` (a gate name with ZERO occurrences in any .py and ZERO occurrences in | `pre_flight_check.py:17444-17466 (--gates resolution + hard error, return 1 at :17466). Synthesis census over all 276 tracked YAML: 88 carry directives` | Read-only config repair: drop the stale `p1_pytorch_cuda_available` key from the 82 configs' directives.pre_flight_gates_required, and drop the 2 dead | no |
| B3 | B-gate-truth | 24 of the 32 Wf-432-A1 gates are registered in NO pipeline and have NEVER run; the 8 that are registered are exactly the 5 L3 + 3 L4 gates. Since 19:17:33 the repo's own test now ASSERTS the | `molmetal/tests/test_wf433a1_gate_v2_wire.py:54-92 (EXPECTED_NEW_GATES = 32)` | Read-only: documentation only. Any wiring must FIRST update the deferral record (test comment block :104-115 + the catalog §6.267 entry) - otherwise t | no |
| B4 | B-gate-truth | The 9 L5 protocol-validation gates compare NOTHING: `verdict_field` is copied into the hook payload and never compared against a measured value; and their emitter writes a hook JSON on EVERY | `molmetal/scripts/pre_flight_check.py:12322 (mkdir(parents=True) + atomic_write_file) inside _hook_passthrough_payload, called by all 9 L5 gates` | BLOCKED for a real fix: the emitter lives inside molmetal/scripts/pre_flight_check.py, which is READ-ONLY under this wave's hard rule 3. Read-only int | **YES** |
| B5 | B-gate-truth | The DrugOOD protocol-alignment reference is dangling and stale: PROTOCOL_ALIGNMENT_MATRIX.md has §0-§4 and NO §1.4; the canonical arg set points at the non-existent TODO/03_plans/PROTOCOL_AL | `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md (headers §0-§4, no §1.4) vs TODO/03_plans/PRE_TRAINING_GATE_v2.md:287 and :311 (spec text cites §1.4) a` | Read-only edits to TODO/03_plans/PRE_TRAINING_GATE_v2.md and TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md - but TODO/** is READ-ONLY under this wave | **YES** |
| B6 | B-gate-truth | Exactly 3 of the 582 metrics/by_metric/*.json files carry the band AND have no registry row (579 registered + 3 unregistered = 582); one of the 3 is STALE against its generator. | `metrics/by_metric/wfa155_ligand_connected_rate.json (band lines 224/228/248), metrics/by_metric/wfa155_metal_coordination_sphere.json (band lines 22/2` | BLOCKED: metrics/** is READ-ONLY under this wave's hard rule 3. Follow-on wave: add the 3 registry rows in registry/metrics.yaml (and refresh the STAL | **YES** |
| C1 | C-bookkeeping | The gate-count pin census - the thing that enforces 'no stale gate-count pins' - is UNTRACKED, so a clean checkout of HEAD has no census and no CI or pre-commit hook can see it; the repo-wid | `git ls-files molmetal/scripts/wfa169_gate_count_pins.py molmetal/tests/test_wfa169_gate_count_pins.py -> EMPTY (both are `??` untracked)` | Needs git write authority (`git add` of the two files) - forbidden by this wave's hard rule 4 (no git mutations). Follow-on wave: track both files, an | **YES** |
| C2 | C-bookkeeping | The pin census scans report-directory .py files: scan_gate_count_pins() is called without the `exclude` parameter it declares, and only the PROSE scan excludes the census's own output dir -  | `molmetal/scripts/wfa169_gate_count_pins.py:853 (exclude param declared), :995 (pin_census calls scan_gate_count_pins with no exclude), :765 (only the ` | Read-only edit of molmetal/scripts/wfa169_gate_count_pins.py (untracked, agent-owned, NOT on the protected list): pass exclude=[CENSUS_OUTPUT_REL] at  | no |
| C3 | C-bookkeeping | The retired-pin accounting does not balance: the Wf-A172 docstring says "moved ten stale count pins across three files ... and moved the four stale baseline assertions", and the artifact rec | `molmetal/tests/test_wfa169_gate_count_pins.py:289-291 (docstring) vs molmetal/reports/wfa169_dead_gate_repair/pin_census.md:22 ("STALE | 15 | 4") whos` | Read-only: correct the pin_census.md accounting note so the 15/4 figure is reconciled against the 5-pins-in-one-file table. Documentation-only. | no |

### A1 — Our "same 3-step post-processing as Buttenschoen" is 2 of 3 steps: the UFF step ran on 0 of 12,668 molecules, so the frontier report's direct-comparability claim is unsound as written.

- **Evidence:** metrics/by_metric/cfm_fragmentation_postproc_arm.json -> postprocessed_axis.uff_ok_rate_over_postprocessed = 0.0, uff_reasons {"ValueError: Bad Conformer Id": 12668}; claim site molmetal/reports/wfa175_frontier_comparability/report.md:79 ("the same 3-step PP as Buttenschoen") and report.md §6.2
- **Status:** SURVIVED (frontier branch's own refutation named it THE most important correction)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: amend report.md:79 to "2 of 3 steps (LargestFragmentChooser -> Chem.AddHs); the UFF step is inapplicable to this corpus' metal complexes (0/12,668, ValueError: Bad Conformer Id)" and strike "directly comparable" in §6.2. The 11.75139146567718x connectivity ratio itself is UNAFFECTED (the rate is fixed by step 1).
- **BLOCKED:** no
- **Files:** `molmetal/reports/wfa175_frontier_comparability/report.md`
- **Note:** This is the only finding that touches a number a paper table would print.

### A2 — READING-SAFE: the exact source cells of the band are NOT settled on disk -- COMPETING readings (`band_provenance` vs `sources`.`EF-TALFM`); the NUMBER of readings is NOT settled on disk.

> The published "92.7-94 %" band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under EVERY reading the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

- **Evidence:** Reading A -- molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance`: The band's endpoints are GeoLDM's Validity & Uniq (92.7 %, post_processed: false, arXiv:2603.27113v1 Table 1(b) row 'Val&Uniq (%)' column 'GeoLDM') and HLTF's PB-Valid (94.0 %, post_processed: false, the same table's row 'PB-Valid (%)' column 'HLTF (ours)'), both on QM9; YuelBond's per-bond-type classification F1 (92.7 %, bioRxiv 10.1101/2025.05.06.652517v1) is a THIRD candidate for the same numeral. Reading B -- molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `sources`.`EF-TALFM`, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4: The band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell C. = 92.98 % (counted BEFORE the largest-component retention) with its post-processed N. & U. & S. = 94.33 % companion cell. The competing readings do not even agree on the numerals (92.7 / 94.0 vs 92.98 / 94.33), so the band's source cells are NOT settled on disk.
- **Status:** CONTESTED on disk (Wf-A177 / Wf-A180 / Wf-A196): the band's provenance is NOT settled -- COMPETING readings of the frontier cells it names, and the NUMBER of competing readings is itself NOT settled on disk; only the claims that hold under EVERY reading may be published.
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: the band must never be placed on OUR scale as a comparator, and its endpoint assignment must not be published as settled -- record the metric-definition break in the honest-negative catalog and quote the reading-safe clause instead.
- **BLOCKED:** no
- **Files:** `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`, `docs/HONEST_NEGATIVE_CATALOG.md`
- **Note:** docs/HONEST_NEGATIVE_CATALOG.md is READ-ONLY under this wave's hard rule 3.

### A3 — The Open-Babel mis-attribution of the band is still LIVE in the generator, not just in stale prose: molmetal/scripts/wfa161_frontier_labelling.py:934 still emits "Our 92.7-94 % band comes from a DIFFERENT perception branch (Open Babel) and a DIFFERENT post-processing", so re-running it re-emits the error.

- **Evidence:** molmetal/scripts/wfa161_frontier_labelling.py:930-936 (re-measured by synthesis; string present verbatim)
- **Status:** SURVIVED (frontier CLAIM 3, IMPRECISE -> the correction is that the mis-attribution is NOT single)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT on this wave's protected list). Replace the Open-Babel clause with the READING-SAFE statement: the band is EXTERNAL and is NOT one homogeneous quantity; the repo holds COMPETING provenance readings for the frontier cells it names (frontier_matrix.json -> band_provenance :51-89 vs the same file -> sources.EF-TALFM :360/:378-379; EF-TALFM arXiv:2609.08333v4) that do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). The NUMBER of competing readings is itself NOT settled on disk: the band-provenance record (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance.competing_reading_and_what_this_does_not_settle`.`readings`) enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. So NO endpoint may be declared a connected-component count and the exact source cells are NOT settled on disk; OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.
- **BLOCKED:** no
- **Files:** `molmetal/scripts/wfa161_frontier_labelling.py`
- **Note:** Only writer-side defect in the tier: a script that would regenerate the false statement.

### A4 — The band census is self-inflating and blind to the gitignored tree: 82 sites / 34 files is COMPLETE ONLY for tracked+untracked-not-ignored; 21 gitignored files outside the branch carry the band on 81 lines, including the very artifact a band-provenance audit should have read (Wf-A171 band_classification.json, 32 band lines).

- **Evidence:** molmetal/reports/wfa175_band_provenance/sites.json (n_sites=82, distinct files=34, population="git ls-files --cached + git ls-files --others --exclude-standard"); .gitignore:158 (molmetal/reports/**/*.json); molmetal/reports/wfa171_band_publication_audit/band_classification.json = 34,046 B / 625 lines / 32 band-bearing lines (re-measured by synthesis)
- **Status:** SURVIVED (band CLAIM 1 SURVIVED + the refutation's scope-cap correction; Wf-A171 OPEN-5 already named it)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: re-run the census with the gitignored report population included, and close Wf-A171's OPEN-5. No GPU. This is a census-scope repair, not a code repair.
- **BLOCKED:** no
- **Files:** `molmetal/reports/wfa175_band_provenance/_census.py`
- **Note:** A guard's own census is not a coverage measurement - the same lesson the branch teaches, applied to itself.

### A5 — The band report's own §2.2 is INVERTED against its own sites.json: report.md:106 says "For 6 of the 7 `post` sites the statement is *correct and per-cell*", but sites.json shows 6 of the 7 are the blanket-REFUTED ones; only molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.md:68 is correct.

- **Evidence:** molmetal/reports/wfa175_band_provenance/report.md:106 vs molmetal/reports/wfa175_band_provenance/sites.json (raw_or_post Counter {unstated 66, raw 9, post 7}; post sites = catalog:20817, wfa155_ligand_connected_rate.json:228, wfa155_metal_coordination_sphere.json:26, wfa155_posebusters_validity.json:608, frontier_matrix.md:68, wfa161_frontier_labelling/report.md:139, wfa161_frontier_labelling.py:934)
- **Status:** SURVIVED (band CLAIM 2 IMPRECISE -> this inversion is the correction)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: correct report.md:106 to state that 6 of the 7 `post` sites are the blanket-REFUTED ones and only frontier_matrix.md:68 is correct-and-per-cell. A repairer following §2.2 verbatim would leave 6 wrong sites unrepaired.
- **BLOCKED:** no
- **Files:** `molmetal/reports/wfa175_band_provenance/report.md`

### A6 — The frontier report's §2.1 ratio column contradicts its OWN computed artifact: 4 of the ratios printed in the report disagree with molmetal/reports/wfa175_frontier_comparability/tools/ratios.json.

- **Evidence:** report.md:66 1.03175x vs ratios.json 1.0319905213270142; report.md:69 1.48071x vs 1.481042654028436; report.md:72 1.27766x vs 1.2776287210936503; report.md:76 1.15501x vs the GEOM row of ratios.json
- **Status:** SURVIVED (frontier CLAIM 2 SURVIVED; this ratio-column defect is the IMPRECISE part)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: regenerate report.md's §2.1 ratio column from tools/ratios.json rather than re-typing it. Documentation-only blast radius (rounding magnitude), but it is a self-inconsistent provenance chain in a paper-facing table.
- **BLOCKED:** no
- **Files:** `molmetal/reports/wfa175_frontier_comparability/report.md`

### B1 — The refs_bib gate would FAIL (paper/refs.bib has 0 of the 3 required keys) but is DEAD in the working tree: it is DEFINED at pre_flight_check.py:11118 and registered in NO pipeline, so it cannot turn `all` RED and cannot block any launch TODAY. It becomes a launch-blocker the moment the Wf-A169 wiring patch lands.

- **Evidence:** pre_flight_check.py:11118 (gate def) - `grep` finds 1 occurrence = the def only; control gate gate_p3_protocol_alignment_yaml_no_inline_cli has 2 (def at :11497 + registration at :14960). Live build_gates() union over PIPELINES ('chemprop_v2_v3','cfm','all','frontier','launch') contains NEITHER p3_protocol_alignment_baseline_paper_refs_bib NOR p3_protocol_alignment_user_ok_gate. paper/refs.bib: 100 keys, missing all of krasnov2026metalcytotoxdb / rusanov2025platinai / ji2022drugood (required_keys at :11124-11127). Staged wiring patch: molmetal/reports/wfa169_dead_gate_repair/patch_wire_24_gates.diff (NOT applied).
- **Status:** SURVIVED AS A MEASUREMENT, BUT THE WIRING REFUTATION'S 'blocks every launch' VERDICT IS CORRECTED (synthesis's own adversarial re-check)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: nothing to change today. For the follow-on wave, the bibliography fix and the gate wiring MUST land in ONE write, bibliography FIRST - landing the wiring first turns `all` RED and exits 1 through scripts/launch_wf124_p3_g5.sh:288-294 (`|| { echo "FAIL: pre_flight_check"; exit 1; }`), which uses --pipeline launch (a superset of `all`). The 3 keys must be added SURGICALLY to paper/refs.bib - the STAGED bib is a 13-entry overlay vs 100 live entries, so copying it over would delete 87 entries.
- **BLOCKED:** YES — BLOCKED: paper/refs.bib is under AGENTS.md §5.3 / TODO-84 - zero writes without the user's explicit OK.
- **Files:** `paper/refs.bib`, `molmetal/scripts/pre_flight_check.py`

### B2 — 88 TRACKED CONFIGS declare a required pre-flight gate that no pipeline owns: 82 configs name `p1_pytorch_cuda_available` (a gate name with ZERO occurrences in any .py and ZERO occurrences in pre_flight_check.py's git history - `git log -S` returns empty), and 6 configs name the 2 dead protocol gates. `pre_flight_check.py --gates` resolves against the UNION of pipelines and HARD-ERRORS (return 1) on unknown names.

- **Evidence:** pre_flight_check.py:17444-17466 (--gates resolution + hard error, return 1 at :17466). Synthesis census over all 276 tracked YAML: 88 carry directives.pre_flight_gates_required and ALL 88 name unregistered gates. p1_pytorch_cuda_available: `git grep -l` -> 83 tracked files (82 configs + 1 TODO doc), 0 of them .py; `git ls-files '*.py' | xargs grep -l` -> 0. The 6 two-gate configs = 5 in configs/training/ + submission/paper_a/configs/training/nci60_9cell_5seed_50ep_v1.yaml. LATENT today: no tracked .sh/.py launcher passes --gates (only pre_flight_check.py itself and test_wf380a4_vina_params.py, which asserts only a list LENGTH of 4, never a gate name).
- **Status:** SURVIVED - NEW FINDING BY SYNTHESIS (not covered by any of the 5 branches; the catalog records only the 6-config half)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only config repair: drop the stale `p1_pytorch_cuda_available` key from the 82 configs' directives.pre_flight_gates_required, and drop the 2 dead protocol gates from the 6 configs (or register real gates). configs/** is NOT on this wave's protected list, so this is actionable without further authorization.
- **BLOCKED:** no
- **Files:** `configs/cfm_5metal_default.yaml`, `configs/training/nci60_9cell_5seed_50ep_v1.yaml`, `configs/training/nci60_9cell_coverage_5seed_50ep_v1.yaml`, `configs/training/nci60_9cell_probe_seed0_cap512_d4.yaml`, `configs/training/nci60_9cell_probe_seed12_cap512_d4.yaml`, `configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml`, `submission/paper_a/configs/training/nci60_9cell_5seed_50ep_v1.yaml`
- **Note:** 82 of the 88 files are the configs/cfm_*.yaml family - enumerated in full in synthesis.json -> files_to_create_edit.config_stale_gate_key.

### B3 — 24 of the 32 Wf-432-A1 gates are registered in NO pipeline and have NEVER run; the 8 that are registered are exactly the 5 L3 + 3 L4 gates. Since 19:17:33 the repo's own test now ASSERTS the 24 stay absent (a deferral tripwire), so this is a RECORDED DELIBERATE deferral (catalog §6.267), not an unnoticed gap.

- **Evidence:** molmetal/tests/test_wf433a1_gate_v2_wire.py:54-92 (EXPECTED_NEW_GATES = 32); synthesis live build_gates() union -> 24 MISSING, 8 present = p3_mock_to_real_m3/m15/m14/m17/m21 + p3_honest_neg_closure_m2_r2/m6_kappa/m1_re_fold; tripwire at :242 test_wf432a1_live_gates_registered_and_deferred_ones_absent + the deferral record at :104-115 and :292
- **Status:** SURVIVED (wiring CLAIM 1; the time-index correction comes from the pins branch)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: documentation only. Any wiring must FIRST update the deferral record (test comment block :104-115 + the catalog §6.267 entry) - otherwise the tripwire fires. Do not wire on this wave's authority.
- **BLOCKED:** no
- **Files:** `molmetal/scripts/pre_flight_check.py`, `molmetal/tests/test_wf433a1_gate_v2_wire.py`
- **Note:** molmetal/scripts/pre_flight_check.py is READ-ONLY under this wave's hard rule 3.

### B4 — The 9 L5 protocol-validation gates compare NOTHING: `verdict_field` is copied into the hook payload and never compared against a measured value; and their emitter writes a hook JSON on EVERY invocation (before the verdict), so even a WARNing gate writes molmetal/reports/protocol_validation/<baseline>_hook.json.

- **Evidence:** molmetal/scripts/pre_flight_check.py:12322 (mkdir(parents=True) + atomic_write_file) inside _hook_passthrough_payload, called by all 9 L5 gates; live probe verdict vector 19 PASS / 4 WARN / 1 FAIL / 0 RAISED over 24 gates (molmetal/reports/wfa175_dead_gate_wiring/refutation/probe24.json); the 3 hook JSONs the probe wrote are at molmetal/reports/wfa175_dead_gate_wiring/refutation/_pv_hook_dir/
- **Status:** SURVIVED (wiring CLAIM 4)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** BLOCKED for a real fix: the emitter lives inside molmetal/scripts/pre_flight_check.py, which is READ-ONLY under this wave's hard rule 3. Read-only interim: document that these gates PASS on file-existence alone and their hook JSON is not evidence of a comparison.
- **BLOCKED:** YES — A gate that cannot fail is worse than no gate: it manufactures PASS.
- **Files:** `molmetal/scripts/pre_flight_check.py`

### B5 — The DrugOOD protocol-alignment reference is dangling and stale: PROTOCOL_ALIGNMENT_MATRIX.md has §0-§4 and NO §1.4; the canonical arg set points at the non-existent TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md; the plan doc's own L5 spec text cites "§1.4" twice; and the matrix's m17 row still says "NOT ALIGN - fallback cohort" while the metric JSON records the OFFICIAL DrugOOD IC50 splits.

- **Evidence:** TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md (headers §0-§4, no §1.4) vs TODO/03_plans/PRE_TRAINING_GATE_v2.md:287 and :311 (spec text cites §1.4) and molmetal/tests/test_wf433a1_gate_v2_wire.py:235 (arg set points at the non-existent TODO/03_plans/ path); metrics/by_metric/drugood_ood_auc.json -> data_source = drugood_official_lbap_refined_ic50, per_axis_auc_delta_vs_imold = +0.009347711256380342 / +0.001844904852242868 / -0.0028530871876835118, assay OOD AUC 0.7304477112563803 (vs the doc's stale 0.7479)
- **Status:** SURVIVED (wiring CLAIM 7)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only edits to TODO/03_plans/PRE_TRAINING_GATE_v2.md and TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md - but TODO/** is READ-ONLY under this wave's hard rule 3, so this needs a follow-on wave's write scope.
- **BLOCKED:** YES — BLOCKED: TODO/** is on this wave's protected list.
- **Files:** `TODO/03_plans/PRE_TRAINING_GATE_v2.md`, `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md`

### B6 — Exactly 3 of the 582 metrics/by_metric/*.json files carry the band AND have no registry row (579 registered + 3 unregistered = 582); one of the 3 is STALE against its generator.

- **Evidence:** metrics/by_metric/wfa155_ligand_connected_rate.json (band lines 224/228/248), metrics/by_metric/wfa155_metal_coordination_sphere.json (band lines 22/26), metrics/by_metric/wfa155_posebusters_validity.json (band line 608) - synthesis census: 582 files, 4 lack a registry row by metric-id, of which exactly 3 carry the band (the 4th, cfm_wfa75a_guidance_sweep.json, carries none); band values 92.7 / 94.0 / 92.98 / 94.33 at wfa155_posebusters_validity.json:554,555,558,561
- **Status:** SURVIVED (band CLAIM 4, IMPRECISE only on the label - the counts are exact)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** BLOCKED: metrics/** is READ-ONLY under this wave's hard rule 3. Follow-on wave: add the 3 registry rows in registry/metrics.yaml (and refresh the STALE generator relation), so `registry.py check` covers them.
- **BLOCKED:** YES — BLOCKED: metrics/** and registry/ are outside this wave's write scope.
- **Files:** `registry/metrics.yaml`, `metrics/by_metric/wfa155_ligand_connected_rate.json`, `metrics/by_metric/wfa155_metal_coordination_sphere.json`, `metrics/by_metric/wfa155_posebusters_validity.json`

### C1 — The gate-count pin census - the thing that enforces 'no stale gate-count pins' - is UNTRACKED, so a clean checkout of HEAD has no census and no CI or pre-commit hook can see it; the repo-wide zero-stale invariant is unguarded (only a FILE-SCOPED zero-stale guard over the 4 Wf-A172-repaired files exists).

- **Evidence:** git ls-files molmetal/scripts/wfa169_gate_count_pins.py molmetal/tests/test_wfa169_gate_count_pins.py -> EMPTY (both are `??` untracked); git grep -ln stale_pin -- . over TRACKED files -> 0 hits; .pre-commit-config.yaml and .github/workflows/ci.yml contain no pin-census gate. Live census re-run by synthesis: stale_pin_count = 0, stale_pin_files = [], counts_match = True.
- **Status:** SURVIVED (pins CLAIM 4 IMPRECISE -> the precise version; plus the branch's supplementary finding #1)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Needs git write authority (`git add` of the two files) - forbidden by this wave's hard rule 4 (no git mutations). Follow-on wave: track both files, and add a repo-wide `stale_pin_count == 0` assertion.
- **BLOCKED:** YES — BLOCKED: no git mutations in this wave. Note the census test also depends on 2 files that exist on disk but not in HEAD - committing the test without them breaks a clean checkout.
- **Files:** `molmetal/scripts/wfa169_gate_count_pins.py`, `molmetal/tests/test_wfa169_gate_count_pins.py`

### C2 — The pin census scans report-directory .py files: scan_gate_count_pins() is called without the `exclude` parameter it declares, and only the PROSE scan excludes the census's own output dir - so a .py dropped in a report dir flips stale 0->6 while the census suite still reports 18 passed.

- **Evidence:** molmetal/scripts/wfa169_gate_count_pins.py:853 (exclude param declared), :995 (pin_census calls scan_gate_count_pins with no exclude), :765 (only the PROSE scan excludes CENSUS_OUTPUT_REL), :724-737 (_iter_python_files has no exclusion). Live experiment by the pins refutation: before stale 0 / files_scanned_python 3054 -> with fixture stale 6 / 3055 -> census suite 18 passed -> after delete stale 0.
- **Status:** SURVIVED (pins CLAIM 5)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only edit of molmetal/scripts/wfa169_gate_count_pins.py (untracked, agent-owned, NOT on the protected list): pass exclude=[CENSUS_OUTPUT_REL] at the :995 call site.
- **BLOCKED:** no
- **Files:** `molmetal/scripts/wfa169_gate_count_pins.py`

### C3 — The retired-pin accounting does not balance: the Wf-A172 docstring says "moved ten stale count pins across three files ... and moved the four stale baseline assertions", and the artifact records "STALE | 15 | 4", but the file's own table lists 5 stale pins in test_wf433a1_gate_v2_wire.py - at least one retired pin is unaccounted for. Exact-status coverage also regressed 9 files/24 pins -> 6 files/8 pins.

- **Evidence:** molmetal/tests/test_wfa169_gate_count_pins.py:289-291 (docstring) vs molmetal/reports/wfa169_dead_gate_repair/pin_census.md:22 ("STALE | 15 | 4") whose rows 3-7 are 5 stale pins in test_wf433a1_gate_v2_wire.py (lines 108,120,139,140,365); pre-repair pin_census.json = executable 24 pins / 9 files vs live 8 pins / 6 files
- **Status:** SURVIVED (pins CLAIM 2); the "KNOWN_PIN_FILES was edited down" half of pins CLAIM 7 is REFUTED (length stayed 6 - a 3-out/3-in swap)
- **Tag:** MEASURED (not PROJECTED)
- **Remediation (minimal, read-only):** Read-only: correct the pin_census.md accounting note so the 15/4 figure is reconciled against the 5-pins-in-one-file table. Documentation-only.
- **BLOCKED:** no
- **Files:** `molmetal/reports/wfa169_dead_gate_repair/pin_census.md`

---

## 2. REFUTED / IMPRECISE — do not resurrect

> **Wf-A180 (2026-11-01) — this section is a quarantine.** Every entry below is REFUTED or IMPRECISE and must not be quoted. In particular, the band-endpoint statements inside them (which pair the band with specific `post_processed: false` frontier cells) are SUPERSEDED: the band's source cells are NOT settled on disk, and their line-number locators are pre-Wf-A177 and therefore not stable. See the Wf-A180 CORRECTION at the end of this document.

### BIBLIOGRAPHY · CLAIM 6 - the AttentiveFP version-of-record issue is 12

- **Why:** REFUTED on the issue number. Four independent primary records say 16 (Crossref `issue` + `journal-issue.issue`; PubMed esummary `issue`; PubMed MEDLINE `VI - 63` / `IP - 16`; OpenAlex `biblio.issue`). paper/refs.bib:141 ALREADY carries `number = {16}`. The branch's STAGED file (molmetal/reports/wfa175_refs_bib_audit/refs_corrected_STAGED.bib:399 and :424) writes `number = {12}` and its note at :405 cites 'Crossref issue 12; PubMed 63(12):8749-8760' - landing it would INTRODUCE a new error into a currently-correct field. `audit.json` stores no `crossref_issue` field at all, so 12 was asserted, never measured. DO NOT APPLY the number change.
- **What survives:** CLAIM 6's title correction and both author-name corrections (Zhenjun->Zhaojun, Xiang->Xiaomin) SURVIVE and should be applied.

### FRONTIER · CLAIM 2b - our RAW 8.51 % is below EVERY frontier RAW connectivity measured

- **Why:** IMPRECISE, not false in direction. The branch's own §1 row 3 measures TABASCO connectivity = 0.00 (RAW, at 10-20 steps), which is BELOW our 8.51 %; and DiffSBDD's 63.37 % is P(connected|valid) (main.tex:929; validity 96.32 % => ~61.0 % unconditional) against our unconditional P(connected|parsed). The honest-negative direction survives; the universal quantifier does not.

### FRONTIER · CLAIM 3 - band is frontier-sourced, never in paper/, never at HEAD

- **Why:** IMPRECISE. Both negatives CONFIRMED and strengthened (0 hits over all 9,551 HEAD blobs and all 134 paper/ files). But the report's own census records 123 paper/ files, not 134; '0 of 10,060 committed blobs' mislabels the denominator (HEAD = 9,551); '94.0 is 0 of 134 in paper/' is false (1 SVG coordinate hit); the census excluded gitignored paths repo-wide; and the mis-attribution is NOT single (see finding A3).

### FRONTIER · CLAIM 5 - EF-TALFM's prose sentence contradicts its own table

- **Why:** IMPRECISE. The conclusion (92.98 % must be RAW; no RAW/POST pair exists in the paper, so the ratio is uncomputable) is right, but the premise is wrong: src/appendix.tex:60 states 'Connectivity: % of molecules with exactly one connected component, i.e., one fragment.' - :183 is a loose caption, not a contradiction. Do NOT file EF-TALFM as a frontier metric-definition contradiction.

### BAND · CLAIM 2 - six counts and the '79 %' canonical subset

- **Why:** IMPRECISE. All six counts are exact from sites.json, but '79 %' belongs to the canonical 59/75 subset (today 66/82 = 80.5 %), and the report's §2.2 prose is inverted - see finding A5. Undisclosed scope cap: 21 gitignored files carry the band on 81 lines - see finding A4.

### BAND · CLAIM 3 - the band's endpoints and Open-Babel refutation

- **Why:** IMPRECISE on citation, not on substance. Both quotes and the Open-Babel refutation verified verbatim; but the two band endpoints' post_processed:false sit at frontier_matrix.json:58 (92.7 %) and :80 (94 %) - :67 is the YuelBond F1 candidate, a different quantity. 'Every other site calls it external' overstates: 5 further sites attach 'the repo's' to the band.

### BAND · CLAIM 4 - the 3 'published metric artifacts' are published

- **Why:** IMPRECISE on the LABEL only. All 6 surfaces, the 4 dash blocks and the 20817/20819 geometry verify line-exact. The three artifacts are exactly the 3 of 582 metrics/by_metric/*.json with NO registry row - unregistered, one STALE vs its generator. See finding B6.

### BAND · CLAIM 5 - paper/ is band-free except 99.9

- **Why:** IMPRECISE. The headline SURVIVES: all 123 tracked paper/** files re-swept, 0 band hits under 3 notations; every 92.7 is the YuelBond F1 string or SVG coordinate noise. But the value sweep is not '0 except 99.9' (99.9->12, 8.51->19, 92.7->18, 92.0->10, 98.8->7, 87.9->6, 91.9->5, 91.2->3 hits, all noise) and the SVG split '4/5/3/4' does not reproduce (total 16 reproduces; the 9\d\.\d split is 13/3/0/0).

### WIRING · CLAIM 2 - 'NOT ONE of the 24 evaluates its own spec threshold'

- **Why:** IMPRECISE: it is 22 of 24. drugood_3_splits does evaluate a 3-split COUNT - against the wrong file, with a 'batch_size'-driven false positive on 'size' (scaffold lives under per_axis_auc / gate_pass, not in any top-level key) - and dataset_match evaluates a 6-path count where the spec says 7.

### WIRING · CLAIM 3 - the 11/7/5/1 and 8/7/2/1/6 histograms are MEASURED

- **Why:** IMPRECISE: both histograms are exactly reproducible and each sums to 24, but 3 of the LABELS are declared interpretations, not measurements (metric_match=UNSPEC; honest_limitation=7/7 is a loose reading where the repo's stricter rule scores 5/7; dataset_match=MET is 6/6 where the spec says 7). A case-sensitive \bm15\b scan returns 0 - the entry exists as metric_id = m15_rmsd in §6.23, so the 7/7 label is correct.

### WIRING · CLAIM 5 - 'the other 6 numerically MET'

- **Why:** IMPRECISE: arithmetic error. The 9 records are 3 SHORTFALL + 1 MISSING_SOURCE + 5 MEETS (4 named + 6 > 9 is impossible; the 6th residual is the MISSING_SOURCE one just named). The 3 drugood MEETS also come from an ALTERNATE source: the spec's primary molmetal/reports/wf265_collect/all_results.json has primary_status = MISSING_KEY, so the MEETS come from metrics/by_metric/drugood_ood_auc.json.

### WIRING · CLAIM 8 - 'read-only repair' / 'named flags occur 0x' / seed_cohort blocker

- **Why:** IMPRECISE on three counts: section_6_auto_emit needs a NEW FILE under molmetal/scripts/ (and once it exists the gate PASSes on its existence alone - the vacuous-PASS shape); '5-fold' occurs 5x in train_metal_cytotox.py (prose) and 'per_cell' occurs in both trainers as an output key, so only the CLI-flag reading is right; and seed_cohort is closable by correcting the SPEC TEXT alone (the trainers already emit per-seed JSON) with no trainer change and no §5.7 smoke. Also SUPERSEDED at 19:17:33: wiring only 4 no longer leaves the repo's own test RED - the assertion was replaced by the deferral tripwire (see finding B3).

### WIRING · CLAIM 6 - 'the refs_bib gate turns `all` RED and blocks every launch'

- **Why:** SURVIVED AS A MEASUREMENT, CORRECTED ON THE CONSEQUENCE by the synthesis's own re-check: in the WORKING TREE the gate is registered in NO pipeline (pre_flight_check.py:11118 is the ONLY occurrence of the function name - def only; the control gate has def + registration), so it cannot turn `all` RED and cannot block any launch today. It becomes a launch-blocker only when the Wf-A169 wiring patch lands - which is why the bibliography fix and the gate wiring must be ONE write, bibliography FIRST. See finding B1.

### PINS · CLAIM 4 - 'no test anywhere asserts stale_pin_count is 0'

- **Why:** IMPRECISE: a FILE-SCOPED zero-stale guard exists and holds - test_the_wfa172_repaired_files_hold_no_stale_pin (molmetal/tests/test_wfa169_gate_count_pins.py:286-305) asserts no stale pin in each of the 4 Wf-A172-repaired files. PRECISE: the REPO-WIDE zero-stale invariant is unguarded; the only zero-stale assertion is file-scoped. MEASURED consequence: a 6-stale-pin fixture in a report dir left the census suite at 18/18 PASS.

### PINS · CLAIM 7 - 'KNOWN_PIN_FILES was edited down'

- **Why:** REFUTED on that clause. Its length was 6 BEFORE the repair and is 6 AFTER - a 3-out / 3-in swap (pre-repair {wfa127, wfa128, wf433a1, wf350c1, wf323f, wf415a3} -> today's {preflight_ssot_guard_wiring, wf350a2, wf350c1, wf415a3, wfa127, wfa159}); the guard's form did not weaken (len(exact) >= len(KNOWN_PIN_FILES)). The FIRST clause survives: exact-status coverage regressed 9 files/24 pins -> 6 files/8 pins while file-level coverage of all 9 was preserved.

### PINS · CLAIM 8 - the Wf-A172 loop form is invisible to the census

- **Why:** IMPRECISE on the citation only. The substance holds (a `for pipeline in (...)` loop form yields 0 rows of any kind to the exact-pin vocabulary; the direct-SSOT form yields 1 'dynamic' row). CITATION ERROR: the loop is at molmetal/tests/test_wf433a1_gate_v2_wire.py:199 at measurement time and :341 after the 19:17:33 rewrite - NOT :180-186, which are the repair's explanatory comment.

### PINS · CLAIM 9 - 'the floor is VIOLATED at HEAD's 126'

- **Why:** IMPRECISE: mixes two trees. HEAD's own 385-line copy of the test file carries NO `>= 127` floor - it has `assert _pfc.EXPECTED_GATE_COUNTS["all"] == 127` (:108) and `assert len(set(names)) == 127` (:120). PRECISE: the WT-only floor is satisfied in the WT tree (129 >= 127) but is incompatible with HEAD's 126-gate runtime, so the WT test file cannot be run against HEAD's pre-flight.

---

## 3. Checked and found CLEAN (negative results are results)

- paper/** - 0 hits of the 92.7-94 % band under 3 notations across all 123 tracked paper/ files; every 92.7 in paper/ is the YuelBond F1 string or SVG coordinate noise. (build_log.txt is tracked + ISO-8859 + 8 hits; PATH ugrep's -I skips it, so the repo's own scanner is required.)
- The Open-Babel '92.7-94 % band' attribution string - 0 hits over all 9,551 HEAD blobs and all 134 paper/ files on disk (the mis-attribution exists only in molmetal/scripts/wfa161_frontier_labelling.py:934 and in report prose).
- All 4 frontier e-print sha256 reproduced byte-exact on re-fetch; the cached extracted text is byte-identical to the re-fetched tarball; every cited quote verbatim at the cited line.
- The 87.9 % vs 89.4 % dispute - reproduced `89.4`=0 / `87.9`=2 in the e-print and `89.4`=0 / `87.9`=3 on the live abs page; the 87.9 sentence is verbatim. (Minor: the locator is src/results_unconditional.tex:35, not sections/...; the abs-page half is not archived - no bytes/sha in §0.)
- Live gate counts: all=129, launch=131, frontier=25, chemprop_v2_v3=18, cfm=8 - identical to EXPECTED_GATE_COUNTS, with NO scope cap (those 5 ARE the repo's own PIPELINES tuple).
- The 4 registered p3_protocol_alignment_yaml_* gates are correctly registered (control for the dead-gate family) - the dead set is specific, not a global registration failure.
- --gates resolves against the UNION of all pipelines (not just `all`) - correct by design (pre_flight_check.py:17444-17466).
- No tracked .sh/.py launcher passes --gates -> the config gate-list trap is LATENT, not a live RED.
- Live pin census re-run by synthesis: stale_pin_count = 0, stale_pin_files = [], counts_match = True.
- git grep -ln stale_pin -- . over TRACKED files -> 0 hits (no tracked consumer of the census).
- metrics/by_metric/*.json = 582 files; 579 registry rows -> exactly the 3 band-carrying files are the unregistered ones (the arithmetic closes).
- The Krasnov record is real and correctly quoted: the abstract says verbatim '26,500 IC50 values for 7050 metal complexes against 754 cell lines from 1921 articles' (no '60 cancer cell lines' anywhere); the real record is J. Med. Chem. 2026, 69(8):8838-8851, 10.1021/acs.jmedchem.5c02755.
- The ChemRxiv Rusanov preprint 10.26434/chemrxiv-2025-pp32k is real; its abstract contains 'MB-Finder' 0 times and PlatinAI is ONE novel bis-carbene platinum(II) complex.
- Both .tools/sci-paper DOIs return responseCode:100 (fabricated); the doubled AAAI DOI 10.1609/aaai.v37i.v37i7.25970 also returns 100, the real one being 10.1609/aaai.v37i7.25970 = AAAI 37(7):8023-8031.
- paper/refs.bib is tracked, has 100 @-entries and is not truncated; detect_truncation returned None on all three STAGED bib deliverables.
- The 3 STAGED bib deliverables preserve all 11/11 original entries verbatim, carry all 3 required keys, and have unique cite keys.
- The drugood 3-split / dataset_match gate-source arithmetic (25/125, 9/45, 3/12, 3/3 splits, 2/2, 1/1) and the 266,522 data rows per axis CSV are exact.

---

## 4. Exact file set for an implementer

### read_only_now_agent_owned

- `molmetal/reports/wfa175_frontier_comparability/report.md` — line 79 + §6.2: say '2 of 3 steps', disclose the UFF 0/12,668 failure, strike 'directly comparable' (finding A1); regenerate the §2.1 ratio column from tools/ratios.json (finding A6).
- `molmetal/reports/wfa175_band_provenance/report.md` — line 106: correct the inverted §2.2 ('6 of the 7 post sites are correct' -> '6 of the 7 are the blanket-REFUTED ones; only frontier_matrix.md:68 is correct') (finding A5).
- `molmetal/reports/wfa175_band_provenance/_census.py` — include the gitignored report population so the census is a coverage measurement (finding A4).
- `molmetal/scripts/wfa161_frontier_labelling.py` — line 934: replace the Open-Babel mis-attribution string (finding A3).
- `molmetal/scripts/wfa169_gate_count_pins.py` — line 995: pass exclude=[CENSUS_OUTPUT_REL] to scan_gate_count_pins (finding C2).
- `molmetal/reports/wfa169_dead_gate_repair/pin_census.md` — line 22: reconcile the STALE 15 / 4 accounting against the 5-pins-in-one-file table (finding C3).
- `configs/cfm_5metal_default.yaml + 81 sibling configs/cfm_*.yaml` — drop the stale `p1_pytorch_cuda_available` key from directives.pre_flight_gates_required (finding B2); full list in files_to_create_edit.config_stale_gate_key.
- `configs/training/nci60_9cell_5seed_50ep_v1.yaml` — drop `p3_protocol_alignment_user_ok_gate` and `p3_protocol_alignment_baseline_paper_refs_bib` from directives.pre_flight_gates_required (finding B2).
- `configs/training/nci60_9cell_coverage_5seed_50ep_v1.yaml` — same as above (finding B2).
- `configs/training/nci60_9cell_probe_seed0_cap512_d4.yaml` — same as above (finding B2).
- `configs/training/nci60_9cell_probe_seed12_cap512_d4.yaml` — same as above (finding B2).
- `configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml` — same as above (finding B2).
- `submission/paper_a/configs/training/nci60_9cell_5seed_50ep_v1.yaml` — same as above (finding B2).

### blocked_on_paper_ok_agmd_5_3

- `paper/refs.bib` — SURGICALLY add the 3 required keys (krasnov2026metalcytotoxdb, rusanov2025platinai, ji2022drugood) + apply the surviving CLAIM-6 title/author corrections. DO NOT copy the STAGED overlay (13 live entries vs 100) and DO NOT apply its number={12} (finding B1).

### blocked_on_readonly_scope_this_wave

- `molmetal/scripts/pre_flight_check.py` — register or retire the 24 dead gates (B3); make the 9 L5 hooks compare a measured value and stop writing on WARN (B4). READ-ONLY under hard rule 3.
- `TODO/03_plans/PRE_TRAINING_GATE_v2.md` — fix the dangling 'PROTOCOL_ALIGNMENT_MATRIX.md §1.4' spec cross-references (:287, :311) (B5). READ-ONLY under hard rule 3.
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — refresh the stale m17 'NOT ALIGN - fallback cohort' row against metrics/by_metric/drugood_ood_auc.json (B5). READ-ONLY under hard rule 3.
- `metrics/by_metric/wfa155_ligand_connected_rate.json + wfa155_metal_coordination_sphere.json + wfa155_posebusters_validity.json` — registry rows + STALE generator relation (B6). READ-ONLY under hard rule 3.
- `docs/HONEST_NEGATIVE_CATALOG.md` — record the band metric-definition break + the band-census scope cap (A2, A4). READ-ONLY under hard rule 3.
- `AGENTS.md` — update the §6.267 dead-gate entry: 24 of 32 unregistered (now a recorded deferral), and the 82-config p1_pytorch_cuda_available population (B2, B3). READ-ONLY under hard rule 3.

### blocked_on_git_write_authority

- `molmetal/scripts/wfa169_gate_count_pins.py` — git add (track it) - hard rule 4 forbids git mutations (C1).
- `molmetal/tests/test_wfa169_gate_count_pins.py` — git add (track it) + add a repo-wide stale_pin_count == 0 assertion; note it depends on 2 files that exist on disk but not in HEAD (C1).

### new_files_this_wave

- `molmetal/reports/wfa175_synthesis/synthesis.json` — this wave's deliverable.
- `molmetal/reports/wfa175_synthesis/synthesis.md` — this wave's deliverable.

### full list: the 82 configs carrying the stale `p1_pytorch_cuda_available` key

```
configs/cfm_5metal_default.yaml
configs/cfm_5metal_default_log_normal_2026_10_16.yaml
configs/cfm_5metal_smoke_2026_10_17.yaml
configs/cfm_5metal_v5_full.yaml
configs/cfm_5metal_wfa68_smoke.yaml
configs/cfm_5metal_wfa69_smoke.yaml
configs/cfm_5metal_wfa70_w1_edge_mpnn.yaml
configs/cfm_5metal_wfa70_w2_temperature_pr_auc.yaml
configs/cfm_5metal_wfa70_w3_smoke.yaml
configs/cfm_5metal_wfa73_operators.yaml
configs/cfm_5metal_wfa74_residual.yaml
configs/cfm_5metal_wfa75_bond_guidance.yaml
configs/cfm_5metal_wfa75a_w000.yaml
configs/cfm_5metal_wfa75a_w002.yaml
configs/cfm_5metal_wfa75a_w005.yaml
configs/cfm_5metal_wfa75a_w020.yaml
configs/cfm_5metal_wfa77_allegro_scaffold.yaml
configs/cfm_5metal_wfa78_ring_prior.yaml
configs/cfm_5metal_wfa78_ring_prior_cpu_smoke.yaml
configs/cfm_5metal_wfa80_per_class_cpu_smoke.yaml
configs/cfm_5metal_wfa80_per_class_decision.yaml
configs/cfm_blk_c5_repair_2026_10_09.yaml
configs/cfm_bond_head_30ep_onecycle.yaml
configs/cfm_fm_ab_2026_10_09.yaml
configs/cfm_fm_control_2026_10_09.yaml
configs/cfm_full_retrain.yaml
configs/cfm_gen_collapse_fm.yaml
configs/cfm_pcgrad_off_2026_10_15.yaml
configs/cfm_pcgrad_on_2026_10_14.yaml
configs/cfm_per_innovation_ablation_p1.4.yaml
configs/cfm_real_decode_lift_p1.3.yaml
configs/cfm_tmqm_all6.yaml
configs/default.yaml
configs/fm_pocket_multiph.yaml
configs/smoke_t41_best_block.yaml
configs/smoke_wfa108j_lsq.yaml
configs/wf124_p3_50epoch_full_sweep_g5.yaml
configs/wf124_p3_per_innovation_ablation_g6.yaml
configs/wf124_p3_remaining_metals_g4.yaml
configs/wfa107_control.yaml
configs/wfa107_count.yaml
configs/wfa107_count_w1e3.yaml
configs/wfa107_count_w1e4.yaml
configs/wfa107_edge_ring_w03.yaml
configs/wfa107_edge_ring_w1.yaml
configs/wfa107_features.yaml
configs/wfa107_per_atom.yaml
configs/wfa107e_fm_control.yaml
configs/wfa107e_fm_edge_ring_w1.yaml
configs/wfa107g_fm_sigma_control.yaml
configs/wfa107g_fm_sigma_edge_ring_w1.yaml
configs/wfa108_radial_masked_centroid.yaml
configs/wfa108_vector_channels.yaml
configs/wfa108b_budget20_vector_channels.yaml
configs/wfa108b_liveness_probe.yaml
configs/wfa108b_liveness_probe_e1.yaml
configs/wfa108b_lr1e3_vector_channels.yaml
configs/wfa108e_x1_cpu_smoke.yaml
configs/wfa108h_b32_chunk2_vector_channels.yaml
configs/wfa108h_b48_chunk2_vector_channels.yaml
configs/wfa108h_b64_chunk2_vector_channels.yaml
configs/wfa108j_budget36_2026_10_25.yaml
configs/wfa108j_fixed_cpu_2026_10_25.yaml
configs/wfa108j_gcv36_2026_10_25.yaml
configs/wfa108l_geom_smoke_2026_10_25.yaml
configs/wfa108l_t3_seed0_2026_10_25.yaml
configs/wfa108l_t3_seed1_2026_10_25.yaml
configs/wfa108l_t3_seed2_2026_10_25.yaml
configs/wfa108l_t3_smoke_2026_10_25.yaml
configs/wfa110_fmode_ckpt_2026_10_25.yaml
configs/wfa110_fmode_ckpt_control_2026_10_25.yaml
configs/wfa110_typed_edges_smoke20_2026_10_25.yaml
configs/wfa110_typed_edges_smoke8_2026_10_25.yaml
configs/wfa110_typed_edges_smoke_2026_10_25.yaml
configs/wfa112_cn_cond_smoke_null_2026_10_26.yaml
configs/wfa112_cn_cond_smoke_off_2026_10_26.yaml
configs/wfa112_cn_cond_smoke_on_2026_10_26.yaml
molmetal/configs/lambda_click_physical_smoke.yaml
molmetal/configs/m2_r2_per_cohort.yaml
molmetal/configs/multi_template_config.yaml
molmetal/configs/sota_aligned_targetdiff.yaml
molmetal/configs/vina_docking_params.yaml
```

---

## 5. The two things that would actually hurt if left alone

1. **A1 — the UFF step.** `metrics/by_metric/cfm_fragmentation_postproc_arm.json` records `uff_ok_rate_over_postprocessed = 0.0` and `uff_reasons {"ValueError: Bad Conformer Id": 12668}`. The frontier report says our pipeline is *"the same 3-step PP as Buttenschoen"*. It is 2 of 3 steps. The 11.75139× connectivity ratio is unaffected (the rate is fixed by step 1); the **comparability claim** is not. This is the only finding that touches a number a paper table would print.
2. **B1 — the coupling.** `paper/refs.bib` is missing all 3 keys the `refs_bib` gate requires, so that gate **would** FAIL. It does not run today (it is registered in no pipeline), so it cannot block a launch today. It becomes a launch-blocker the instant the Wf-A169 wiring patch lands — and the wiring patch and the bibliography fix must be **one write, bibliography first**. The bibliography half needs the user's AGENTS.md §5.3 OK on `paper/**`.

---

*Generated 2026-11-01T17:55:29.486815+00:00 · source artifacts: `molmetal/reports/wfa175_refs_bib_audit/refutation/refutation.json`, `molmetal/reports/wfa175_frontier_comparability/refutation/refutation.json`, `molmetal/reports/wfa175_band_provenance/refutation/refutation.json`, `molmetal/reports/wfa175_dead_gate_wiring/refutation/refutation.json`, `molmetal/reports/wfa175_pin_repair_refutation/refutation/refutation.json`.*

---

## Wf-A177 CORRECTION (2026-11-01) — the A2 claim is contested on disk

**Superseded A2 claim (Wf-A175, quoted VERBATIM — preserved, not deleted):**

> The quoted "92.7-94 %" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.

**Why it is superseded.** The repo holds TWO COMPETING provenance readings for the frontier cells the "92.7-94 %" band names, and they do not even agree on the numerals. The sentence above states Reading A as if it were settled; under Reading B the lower endpoint IS a connected-component count (EF-TALFM's own raw `C. = 92.98 %` cell). A published sentence must not assert either reading as fact, so only the claims that hold under BOTH readings may be published.

**Corrected wording (reading-safe — the only form that may be quoted):**

> The published 92.7-94 % band is EXTERNAL to Mol-Metal (it is a FRONTIER composite) and it is NOT one homogeneous quantity. The repo holds TWO COMPETING provenance readings for which cells it names, and they do not even agree on the numerals: (A) `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` (:51-89): GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid 94.0 % on QM9, both `post_processed: false` (plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) -- this reading reproduces the quoted numerals exactly but makes NO endpoint a connected-component count; (B) the same file -> `sources.EF-TALFM` (:360, :378-379), corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its post-processed N. & U. & S. companion cell (94.33 %) -- this reading has the connectivity+post shape but does NOT reproduce the quoted numerals (92.98 / 94.33). Under BOTH readings the band is external, is not one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk.

**Reading B locator (MEASURED, 2026-11-01).** Reading B lives at `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `sources.EF-TALFM` (`:360` `cells.POST-PROCESSING.note`, `:378-379` `extra_observations`), corroborated by the `molmetal/reports/wfa168_frontier_axis_mapping/` branch. The band-provenance artefact carries NO key literally named `frontier_axis_mapping`, so the path quoted above is the resolvable one.

**What is NOT changed by this correction.** The A2 finding's `id`, `tier`, `rank`, `evidence`, `measured_or_projected`, `status`, `remediation`, `blocked`, `files` and `note` fields are untouched. `synthesis.json` keeps the superseded claim VERBATIM under `findings[id=A2].claim` and carries both renderings in the sibling key `a2_claim_superseded_2026_11_01`. No line above this section was edited or removed (AGENTS.md §5.5 rule #6 — append-only).

### Wf-A177 locator addendum (2026-11-01) — the cited line numbers are pre-Wf-A177

The corrected wording above cites `band_provenance` (:51-89) and `:360, :378-379`. Those are the **PRE-Wf-A177** line numbers of `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`: the branch that owns that file added its own Wf-A177 key to `band_provenance`, which shifted every line at or after the old :52. The current positions on disk are — Reading A: `band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140; Reading B: `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence).

**The ROBUST locators are the JSON PATHS, not the line numbers**: `band_provenance` and `sources`.`EF-TALFM`. The owning artefact records the same note at `band_provenance.competing_reading_and_what_this_does_not_settle.line_number_note`, and `synthesis.json` carries it as `findings[id=A2].a2_claim_superseded_2026_11_01.line_number_note`.

Nothing above this subsection was edited or removed (AGENTS.md §5.5 rule #6 — append-only).

---

## Wf-A180 CORRECTION (2026-11-01) — the renderer now emits the reading-safe clause

**The superseded A2 renderings the Wf-A175 close-out carried (quoted VERBATIM — preserved, not deleted):**

> **claim:** The quoted "92.7-94 %" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.
> - **Evidence:** molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58 (92.7 % endpoint, post_processed:false), :67 (the YuelBond F1 candidate - a DIFFERENT quantity), :80 (94 % endpoint, post_processed:false); headline_finding at :89
> - **Status:** SURVIVED (band branch CLAIM 3, IMPRECISE only on citation slips; substance stands)
> - **Remediation (minimal, read-only):** Read-only: the band must never be presented as a connectivity comparator; record the metric-definition break in the honest-negative catalog. Endpoint citation slip to fix in place: the two post_processed:false endpoints are :58 and :80, NOT :67.

The `evidence` line above cites LINE-NUMBER locators (`:58`, `:67`, `:80`, `:89`) that the Wf-A177 key added to `frontier_matrix.json` shifted. Line numbers are **not** stable locators — use the JSON PATHS in the LOCATORS paragraph below.

**Why they are superseded.** The repo holds COMPETING provenance readings for the frontier cells the '92.7-94 %' band names, and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). The NUMBER of competing readings is itself NOT settled on disk: the band-provenance record (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance.competing_reading_and_what_this_does_not_settle`.`readings`) enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. The superseded wording asserts Reading A as settled -- 'None is a connected-component count' -- but under Reading B the lower endpoint IS a connectivity count. A published sentence must not assert either reading as fact.

**Corrected wording (reading-safe — the only form that may be quoted):**

> The published "92.7-94 %" band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under EVERY reading the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

**The sibling key's own Wf-A177 rendering of the same correction** (`findings[id=A2].a2_claim_superseded_2026_11_01`.`corrected_claim_clause_level`), with its pre-Wf-A177 line-number locators stripped:

> The quoted "92.7-94 %" band cannot be placed on our scale: the repo holds COMPETING provenance readings for which frontier cells it names (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` vs the same file -> `sources`.`EF-TALFM`, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4), and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). The NUMBER of competing readings is itself NOT settled on disk: the band-provenance record (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance.competing_reading_and_what_this_does_not_settle`.`readings`) enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. Under EVERY reading the band is external, is NOT one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk.

**The competing readings the sibling key records, addressed by JSON PATH (never by line number):**

> The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals.

- **Reading A** — molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance`
  - The band's endpoints are GeoLDM's Validity & Uniq (92.7 %, post_processed: false, arXiv:2603.27113v1 Table 1(b) row 'Val&Uniq (%)' column 'GeoLDM') and HLTF's PB-Valid (94.0 %, post_processed: false, the same table's row 'PB-Valid (%)' column 'HLTF (ours)'), both on QM9; YuelBond's per-bond-type classification F1 (92.7 %, bioRxiv 10.1101/2025.05.06.652517v1) is a THIRD candidate for the same numeral.
  - reproduces the quoted numerals: YES (92.7 / 94.0); lower endpoint is a connected-component count: NO
- **Reading B** — molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `sources`.`EF-TALFM`, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4
  - The band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell C. = 92.98 % (counted BEFORE the largest-component retention) with its post-processed N. & U. & S. = 94.33 % companion cell.
  - reproduces the quoted numerals: NO (92.98 / 94.33); lower endpoint is a connected-component count: YES

**LOCATORS.** The ROBUST locators are the JSON PATHS `band_provenance` and `sources`.`EF-TALFM` in `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`. The token `frontier_axis_mapping` is **NOT** a key of that record — it survives only inside a string value, so pointing a reader at it is a **DEFECT**.

**What is NOT changed by this correction.** The A2 finding's `id`, `tier`, `rank`, `measured_or_projected`, `blocked`, `files` and `note` fields are untouched, and the sibling key `a2_claim_superseded_2026_11_01` carries BOTH renderings. No line above this section is edited or removed (AGENTS.md §5.5 rule #6 — append-only).

This renderer used to re-render the SUPERSEDED A2 wording into the .md table row, the .md heading and the .md A2 block, and emitted NONE of the correction section — so a re-run of it DESTROYED the Wf-A177 correction. It now renders the READING-SAFE clause and emits BOTH correction sections, so a re-run REPRODUCES them.

**THE READING-SAFE STATEMENT (the only form that may be quoted):**

> The published "92.7-94 %" band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under EVERY reading the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

**The only claims that may be published (they hold under EVERY reading):**

1. the band is EXTERNAL to Mol-Metal (it is a FRONTIER composite)
2. it is NOT one homogeneous quantity
3. it is NOT on our scale
4. OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else
5. the exact source cells of the band are NOT settled on disk

**Claims that are FALSE or SETTLED-WRONG and must NOT be asserted:**

- **X1** — "None is a connected-component count, ours (8.51 %) is." — TRUE under Reading A (lower endpoint is a connected-component count: NO) and FALSE under Reading B (lower endpoint is a connected-component count: YES).
- **X2** — "the published band is measured on the published POST-PROCESSED stream" / "our number is not post-processed and theirs is": FALSE — under Reading A both band endpoints carry `post_processed: false`, and under Reading B the band MIXES one RAW cell with one post-processed companion cell (The band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell C. = 92.98 % (counted BEFORE the largest-component retention) with its post-processed N. & U. & S. = 94.33 % companion cell.). OUR number is RAW under either reading.
- **X3** — "their Connectivity is computed AFTER retaining the atom-count largest connected component": REFUTED on disk — The band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell C. = 92.98 % (counted BEFORE the largest-component retention) with its post-processed N. & U. & S. = 94.33 % companion cell.
- **X4** — "the band is minted here: lower endpoint 92.7 % = GeoLDM's Val&Uniq, upper endpoint 94 % = HLTF's own PB-Valid": asserts ONE reading as settled (Reading A reproduces the quoted numerals: 92.7 / 94.0; Reading B: 92.98 / 94.33). Qualify it, or use the reading-safe clause above.

**Producer state.** This generator now EMITS the reading-safe clause and BOTH correction sections, so a re-run reproduces this correction instead of dropping it. The superseded A2 wording is preserved VERBATIM in the dated source comment above and in the Wf-A177 section above (AGENTS.md §5.5 rule #6).

---

## Wf-A187 CORRECTION (2026-11-01) — the band-provenance LINE-NUMBER locators are DEMOTED

**Status:** CPU-only (`CUDA_VISIBLE_DEVICES=""`) · single venv `/home/hugo/codes/mol-metal/.venv/bin/python` · GPU used: 0 · no git mutation · **ADDITIVE**: nothing above this section is edited or removed (AGENTS.md §5.5 rule #6).

### The locator class — why a line number may not be a live locator

> A LINE NUMBER IS A PERISHABLE LOCATOR. The record `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` GROWS, so every line-number locator into it is STALE BY CONSTRUCTION the moment it is written. The ROBUST locator for that record is a JSON PATH: `band_provenance` and `sources`.`EF-TALFM`.

### The superseded line-number locators, quoted VERBATIM

Each value below is the value the named field carried BEFORE Wf-A187. They are preserved EXACTLY as measured on 2026-11-01 and are NOT re-measured here — see "Not re-measured" below.

**`findings[1].a2_claim_superseded_2026_11_01.readings[0].locator`**

> molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> band_provenance (:51-89; endpoint candidates :52-87, headline_finding :89)

**`findings[1].a2_claim_superseded_2026_11_01.readings[0].current_line_numbers_on_disk`**

> `band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140

**`findings[1].a2_claim_superseded_2026_11_01.readings[1].locator`**

> molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> sources.EF-TALFM (:360 cells.POST-PROCESSING.note, :378-379 extra_observations), corroborated by molmetal/reports/wfa168_frontier_axis_mapping/; EF-TALFM arXiv:2609.08333v4

**`findings[1].a2_claim_superseded_2026_11_01.readings[1].current_line_numbers_on_disk`**

> `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence)

**`findings[1].a2_claim_superseded_2026_11_01.corrected_claim`**

> The published 92.7-94 % band is EXTERNAL to Mol-Metal (it is a FRONTIER composite) and it is NOT one homogeneous quantity. The repo holds TWO COMPETING provenance readings for which cells it names, and they do not even agree on the numerals: (A) `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` (:51-89): GeoLDM's Validity & Uniq 92.7 % and HLTF's PB-Valid 94.0 % on QM9, both `post_processed: false` (plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) -- this reading reproduces the quoted numerals exactly but makes NO endpoint a connected-component count; (B) the same file -> `sources.EF-TALFM` (:360, :378-379), corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4: the band mixes EF-TALFM's OWN Table 3(a) raw connectivity cell (C. = 92.98 %, counted BEFORE the largest-component retention) with its post-processed N. & U. & S. companion cell (94.33 %) -- this reading has the connectivity+post shape but does NOT reproduce the quoted numerals (92.98 / 94.33). Under BOTH readings the band is external, is not one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk.

**`findings[1].a2_claim_superseded_2026_11_01.corrected_claim_clause_level`**

> The quoted "92.7-94 %" band cannot be placed on our scale: the repo holds TWO COMPETING provenance readings for which frontier cells it names (`molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance` :51-89 vs the same file -> `sources.EF-TALFM` :360/:378-379, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4), and they do not even agree on the numerals (92.7/94.0 vs 92.98/94.33). Under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale -- while OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else. The exact source cells of the band are NOT settled on disk.

**`findings[1].a2_claim_superseded_2026_11_01.line_number_note`**

> The ':51-89' range cited for Reading A and the ':360, :378-379' range cited for Reading B are the PRE-Wf-A177 line numbers of frontier_matrix.json; the Wf-A177 key added to that file's `band_provenance` shifted every line at or after the old :52. Current positions on disk: Reading A -> `band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140; Reading B -> `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence). The ROBUST locators are the JSON PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- not the line numbers. The owning artefact records the same note at `band_provenance.competing_reading_and_what_this_does_not_settle.line_number_note`.

**`findings[2].remediation`**

> Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT on this wave's protected list). Replace the Open-Babel clause with the measured truth: the band's endpoints are frontier Validity&Uniq / PB-Valid / bond-type F1, none a connected-component count.

**The PUBLISHED .md's own superseded passages, quoted VERBATIM** (they lived only in the published .md — the renderer cannot reconstruct them from `synthesis.json`, so they are carried here as the dated historical record):

> **Reading B locator (MEASURED, 2026-11-01).** Reading B lives at `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `sources.EF-TALFM` (`:360` `cells.POST-PROCESSING.note`, `:378-379` `extra_observations`), corroborated by the `molmetal/reports/wfa168_frontier_axis_mapping/` branch. The band-provenance artefact carries NO key literally named `frontier_axis_mapping`, so the path quoted above is the resolvable one.
>
> The corrected wording above cites `band_provenance` (:51-89) and `:360, :378-379`. Those are the **PRE-Wf-A177** line numbers of `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json`: the branch that owns that file added its own Wf-A177 key to `band_provenance`, which shifted every line at or after the old :52. The current positions on disk are — Reading A: `band_provenance` :50 (object), `band_as_quoted_in_the_repo` :51, `endpoint_candidates` :103-, `headline_finding` :140; Reading B: `sources` :383 -> `EF-TALFM` :384, the EF-TALFM `POST-PROCESSING` note :411 (carries the 'C. = 92.98 %' cell), `extra_observations` :429-430 (carry the '92.7-94 % band ... mixes the LOWEST connectivity' sentence).

### The LIVE locator — JSON PATHS

- **Reading A** — molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance`
- **Reading B** — molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `sources`.`EF-TALFM`, corroborated by `molmetal/reports/wfa168_frontier_axis_mapping/`; EF-TALFM arXiv:2609.08333v4

### Not re-measured (race-free)

> NOT re-measured by Wf-A187 (race-free). The record is concurrently regenerated by the wave that owns it, so re-measuring the line numbers would race against that wave. The historical values below are preserved EXACTLY as measured on 2026-11-01.

### The X1 renderings this document still carries — SUPERSEDED

**The §A2 heading (quoted VERBATIM):**

> ### A2 — The quoted "92.7-94 %" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict); a third paper quotes 92.7 % as a per-bond-type classification F1. None is a connected-component count, ours (8.51 %) is.

**The A3 remediation bullet (quoted VERBATIM):**

> - **Remediation (minimal, read-only):** Read-only edit of one string literal in molmetal/scripts/wfa161_frontier_labelling.py:934 (that file is NOT on this wave's protected list). Replace the Open-Babel clause with the measured truth: the band's endpoints are frontier Validity&Uniq / PB-Valid / bond-type F1, none a connected-component count.

The §A2 heading asserts X1 — "None is a connected-component count" — which is **TRUE under Reading A** and **FALSE under Reading B**; the A3 remediation asserted the same X1. The renderer now emits the reading-safe form for both; the PUBLISHED .md predates that renderer, so this section is what supersedes the two passages quoted above (they are kept, not deleted).

**The reading-safe replacement (the only form that may be quoted):**

> The published "92.7-94 %" band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under EVERY reading the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

(A re-run of this renderer also emits the same clause as its own `## Wf-A180 CORRECTION` section; this section quotes it inline so that the correction is self-contained wherever it is read.)

**The A2 finding's own Wf-A175 rendering, preserved VERBATIM** (the sibling key `a2_locator_demotion_2026_11_01` -> `superseded_renderings_preserved_verbatim`):

> These three fields are the Wf-A175 renderings of the A2 finding, preserved BYTE-IDENTICAL (AGENTS.md §5.5 rule #6) and quoted VERBATIM inside the dated '## Wf-A177 CORRECTION' section of synthesis.md. Their line-number locators are STALE BY CONSTRUCTION -- do not navigate by them; use the JSON PATHS.

### What is NOT changed by this correction

No number in this document is touched. The A2 finding's `id`, `tier`, `rank`, `measured_or_projected`, `status`, `blocked`, `files` and `note` fields are untouched, and the sibling keys `a2_claim_superseded_2026_11_01` and `a2_locator_demotion_2026_11_01` carry BOTH renderings. Only the **locator form** is corrected, and only additively and dated.

---

## Wf-A196 CORRECTION (2026-11-02) — the reading COUNT is NOT settled on disk, and the producer reproduces the dated sections

**Status:** CPU-only (`CUDA_VISIBLE_DEVICES=""`) · single venv `/home/hugo/codes/mol-metal/.venv/bin/python` · GPU used: 0 · no git mutation · **ADDITIVE**: nothing above this section is edited or removed (AGENTS.md §5.5 rule #6).

### The defect class — a COUNT CLAIM about the size of the contest

A sentence that puts a NUMBER on the size of the contest asserts a settled count that the repo does not settle. Writing a LARGER number is the SAME defect in the other direction — how many DISTINCT readings exist is itself a judgement call (a further reading may be a variant of another). The honest repair asserts NO settled count.

**The clause that may be quoted (it asserts no count):**

> The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals.

**The full reading-safe clause (it asserts no count):**

> The published "92.7-94 %" band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). The NUMBER of competing readings is itself NOT settled on disk: this record enumerates two, and the repo's primary-source verification (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json` -> `invariants.competing_readings`) records a further reading that likewise does not reproduce the quoted numerals. They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under EVERY reading the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

### The four invariants — STILL PRESENT

Quoted VERBATIM from `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json` -> `band_provenance.competing_reading_and_what_this_does_not_settle`.`the_only_claims_that_may_be_published`:

1. **i_the_band_is_external** — The published 92.7-94 % band is EXTERNAL to Mol-Metal -- it is a FRONTIER composite, not our measurement.
2. **ii_it_is_not_one_homogeneous_quantity** — It is NOT one homogeneous quantity: even under Reading A its two endpoints are two different observables (a sanitisation-and-duplication rate and a PoseBusters validity verdict -- HLTF states NO check count for the latter, NOT-STATED-IN-SOURCE), and under Reading B one endpoint is a raw connectivity count while the other is a post-processed composite.
3. **iii_it_is_not_on_our_scale** — It is NOT on our scale.  OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.
4. **iv_the_source_cells_are_not_settled** — The source cells of the band are NOT settled on disk.

### D1 — the producer now reproduces the PUBLISHED Wf-A177 dated sections byte-for-byte

BEFORE Wf-A196 this renderer re-derived the '## Wf-A177 CORRECTION' body from `synthesis.json` and emitted NO '### Wf-A177 locator addendum' subsection at all, so a re-run of the producer REWROTE the dated Wf-A177 body IN PLACE and DROPPED the locator-addendum subsection — a dated block destroyed by a re-run (AGENTS.md §5.5 rule #6). The two published sections are now carried as the NAMED DATED CONSTANTS `_MD_WF177_CORRECTION_BODY` and `_MD_WF177_LOCATOR_ADDENDUM` in `molmetal/reports/wfa175_synthesis/_write_md.py`, each naming the dated section it feeds, and they are emitted byte-for-byte.

### D2 — the LIVE summary-table row carried the refuted X1 claim

The PUBLISHED .md's summary-table row for A2 asserted X1 — "the quoted "92.7-94 %" band is NOT a connectivity band" — which is TRUE under Reading A and FALSE under Reading B (whose lower cell IS a connectivity count). The LIVE row now carries the reading-safe form; the X1 row is preserved VERBATIM below and is NOT deleted.

**The superseded LIVE row (quoted VERBATIM — superseded by this section, not deleted):**

> | A2 | A-published-number | The quoted "92.7-94 %" band is NOT a connectivity band: its two endpoints are GeoLDM Validity&Uniq (92.7 %, sanitisation+uniqueness) and PoseBusters PB-Valid (94.0 %, a 6-check pose verdict) | `molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json:58 (92.7 % endpoint, post_processed:false), :67 (the YuelBond F1 candidate - a D` | Read-only: the band must never be presented as a connectivity comparator; record the metric-definition break in the honest-negative catalog. Endpoint  | no |

**The reading-safe LIVE row (the only form that may be quoted):**

> | A2 | A-published-number | READING-SAFE: the exact source cells of the band are NOT settled on disk -- COMPETING readings (`band_provenance` vs `sources`.`EF-TALFM`); the NUMBER of readings is NOT settled on disk. | `frontier_matrix.json -> band_provenance (Reading A) vs the same file -> sources.EF-TALFM (Reading B); the numerals disagree (92.7 / 94.0 vs 92.98 / 94.33) -- the band's source cells are NOT settled on disk` | Read-only: the band must never be placed on OUR scale as a comparator, and its endpoint assignment must not be published as settled -- record the metr | no |

### The superseded 2026-11-01 renderings, preserved VERBATIM

**The reading-safe clause as it was rendered on 2026-11-01 (quoted VERBATIM — superseded by this section, not deleted):**

> The published "92.7-94 %" band's provenance is NOT settled on disk: the frontier post-processing matrix record (`frontier_matrix.json`) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94.0 %, both `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 % as a third candidate for the same numeral) and the second under `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity `C. = 92.98 %`, counted BEFORE the largest-component retention, with the post-processed `N. & U. & S. = 94.33 %` companion cell). They do not even agree on the numerals (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT settled on disk. Under BOTH readings the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale -- OUR 8.51 % IS a connected-component count on the RAW emitted SMILES, and nothing else.

**The summary-table label as it was rendered on 2026-11-01 (quoted VERBATIM — superseded, not deleted):**

> READING-SAFE: the exact source cells of the band are NOT settled on disk -- TWO COMPETING readings (`band_provenance` vs `sources`.`EF-TALFM`).

**The A2 `status` as it was rendered on 2026-11-01 (quoted VERBATIM — superseded, not deleted):**

> CONTESTED on disk (Wf-A177 / Wf-A180): the band's provenance is NOT settled -- TWO COMPETING readings of the frontier cells it names; only the claims that hold under BOTH readings may be published.

**The A2 evidence tail as it was rendered on 2026-11-01 (quoted VERBATIM — superseded, not deleted):**

> The two readings do not even agree on the numerals (92.7 / 94.0 vs 92.98 / 94.33), so the band's source cells are NOT settled on disk.

The `## Wf-A177 CORRECTION` body and the `## Wf-A187 CORRECTION` section above also carry the 2026-11-01 wording, including its reading-count phrasing, because both are DATED records preserved byte-for-byte. THIS section supersedes that phrasing: the NUMBER of competing readings is NOT settled on disk.

### D3 — the Wf-A189 report's published-only line count

Wf-A189's report states that the renderer-vs-published diff has **16 published-only lines**. The corrected count is **17**. Measurement (MEASURED 2026-11-02, CPU-only): `diff -u` between the PUBLISHED `molmetal/reports/wfa175_synthesis/synthesis.md` as it stood on 2026-11-01 (sha256 `427b3541e1cabdecc6c5611daffafa63249c1b723dcb57471d4593402036624e`) and that same producer's regeneration of it yields **5 hunks** and **18** lines beginning with `-`, of which **one** is the `---` file header — i.e. **17 published-only lines** (13 non-empty and 4 blank). The regenerated side yields **45 regenerated-only lines** (46 lines beginning with `+` minus the `+++` header). The figure 16 is therefore off by one. Wf-A189's report is NOT edited here (another wave owns it); this dated section records the corrected count.

### What is NOT changed by this correction

No number in this document is touched. The A2 finding's `id`, `tier`, `rank`, `measured_or_projected`, `blocked`, `files` and `note` fields are untouched, and `synthesis.json` keeps the superseded claim VERBATIM under `findings[id=A2].claim` and in the dated sibling keys `a2_claim_superseded_2026_11_01` and `a2_x5_count_claim_repair_2026_11_02`. The four invariants above are unchanged. Only the **reading-count form**, the **LIVE summary-table row** and the **producer's reproduction of the dated Wf-A177 sections** are corrected, and only additively and dated.

