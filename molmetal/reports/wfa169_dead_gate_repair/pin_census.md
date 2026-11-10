# Wf-A169 — the gate-count pin census

MEASURED 2026-11-01 by live import of `molmetal/scripts/pre_flight_check.py` (CPU-only, no GPU, no torch).

## 0 — the live counts

| pipeline | live `len(build_gates(p))` | `EXPECTED_GATE_COUNTS[p]` |
|---|---|---|
| `all` | 129 | 129 |
| `cfm` | 8 | 8 |
| `chemprop_v2_v3` | 18 | 18 |
| `frontier` | 25 | 25 |
| `launch` | 131 | 131 |

`counts_match = True` — the Wf-321D invariant (`live == EXPECTED_GATE_COUNTS`) HOLDS on this tree.

## 1 — the two populations

| population | rows | files |
|---|---|---|
| EXECUTABLE-PIN (exact `==`) | 24 | 9 |
| — of which STALE | 15 | 4 |
| — of which current | 9 | 7 |
| FLOOR guard (`>=`, `<=`) | 6 | 5 |
| DYNAMIC invariant (no literal) | 14 | 9 |
| DOC-DERIVED pin (literal in a document) | 4 | 3 |
| PROSE (never a pin) | 2460 | 515 |

Denominator: **3025** tracked/untracked `.py` files scanned by AST + **7878** `.md`/`.sh`/`.yaml`/`.yml`/`.json`/`.txt` files scanned as prose. No file was skipped: `files_skipped = 0` (the scan reads bytes and decodes with `errors="replace"`; PATH `grep` here is `ugrep -I` and would silently skip the repo's non-UTF-8 tracked text files).

## 2 — the STALE executable pins (the RED files)

| # | file:line | pipeline | pinned | live | status |
|---|---|---|---|---|---|
| 1 | `molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py:262` | `all` | 57 | 129 | stale |
| 2 | `molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py:263` | `frontier` | 24 | 25 | stale |
| 3 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:108` | `all` | 127 | 129 | stale |
| 4 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:120` | `all` | 127 | 129 | stale |
| 5 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:139` | `chemprop_v2_v3` | 14 | 18 | stale |
| 6 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:140` | `cfm` | 6 | 8 | stale |
| 7 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:365` | `all` | 127 | 129 | stale |
| 8 | `molmetal/tests/test_wfa127_residue_closure.py:152` | `all` | 127 | 129 | stale |
| 9 | `molmetal/tests/test_wfa127_residue_closure.py:153` | `launch` | 129 | 131 | stale |
| 10 | `molmetal/tests/test_wfa127_residue_closure.py:154` | `all` | 127 | 129 | stale |
| 11 | `molmetal/tests/test_wfa127_residue_closure.py:155` | `launch` | 129 | 131 | stale |
| 12 | `molmetal/tests/test_wfa128_preflight_classification.py:482` | `all` | 127 | 129 | stale |
| 13 | `molmetal/tests/test_wfa128_preflight_classification.py:483` | `launch` | 129 | 131 | stale |
| 14 | `molmetal/tests/test_wfa128_preflight_classification.py:484` | `all` | 127 | 129 | stale |
| 15 | `molmetal/tests/test_wfa128_preflight_classification.py:485` | `launch` | 129 | 131 | stale |

**15 stale pins in 4 files.** Each row is a site that FAILS today for a reason unrelated to the code it guards.

## 3 — the CURRENT executable pins

| # | file:line | pipeline | pinned | live |
|---|---|---|---|---|
| 1 | `molmetal/tests/test_preflight_ssot_guard_wiring.py:312` | `all` | 129 | 129 |
| 2 | `molmetal/tests/test_wf350a2_compute_manifest.py:361` | `frontier` | 25 | 25 |
| 3 | `molmetal/tests/test_wf350c1_gpu_env_probe.py:544` | `all` | 129 | 129 |
| 4 | `molmetal/tests/test_wf415a3_preflight_fail_audit.py:150` | `all` | 129 | 129 |
| 5 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:138` | `frontier` | 25 | 25 |
| 6 | `molmetal/tests/test_wfa127_residue_closure.py:108` | `all` | 129 | 129 |
| 7 | `molmetal/tests/test_wfa127_residue_closure.py:142` | `all` | 129 | 129 |
| 8 | `molmetal/tests/test_wfa159_timeout_verdict_census.py:430` | `all` | 129 | 129 |
| 9 | `molmetal/tests/test_wfa159_timeout_verdict_census.py:431` | `launch` | 131 | 131 |

## 4 — FLOOR guards (monotonic, never stale)

The repo's own convention (Wf-345A2) is that a hard equality fires on every legitimate gate addition while a floor still catches shrinkage. These are **not** stale pins.

| # | file:line | pipeline | floor | live |
|---|---|---|---|---|
| 1 | `molmetal/tests/test_wf345a2_yuelbond_env.py:442` | `all` | >= 68 | 129 |
| 2 | `molmetal/tests/test_wf346a3_anchor_registered.py:125` | `all` | >= 63 | 129 |
| 3 | `molmetal/tests/test_wf350a1_runtime_energy.py:331` | `all` | >= 71 | 129 |
| 4 | `molmetal/tests/test_wf350a2_compute_manifest.py:302` | `all` | >= 71 | 129 |
| 5 | `molmetal/tests/test_wf350a2_compute_manifest.py:319` | `chemprop_v2_v3` | >= 14 | 18 |
| 6 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:374` | `all` | >= 127 | 129 |

## 5 — DOC-DERIVED pins (the literal lives in a document)

Catalog §6.236 says the count lives in FOUR places: the code constant, a test assertion, the SSOT doc header and the SSOT doc summary table. These two rows are the document half — the literal is read out of a file at test time, so the document is a pinned artefact.

| # | file:line | pipeline | pinned | live | document |
|---|---|---|---|---|---|
| 1 | `molmetal/tests/test_preflight_ssot_guard_wiring.py:312` | `all` | 129 | 129 | `document=TODO/completed/321d_preflight_gate_audit.md; regex=### §2\.1\s+`all`\s+pipeline\s+—\s+(\d+)\s+gates` |
| 2 | `molmetal/tests/test_verify_narrative_claims.py:178` | `all` | None | 129 | `document=molmetal/scripts/pre_flight_check.py; regex=EXPECTED_GATE_COUNTS = \{(.*?)\n\}` |
| 3 | `molmetal/tests/test_wfa127_residue_closure.py:108` | `all` | 129 | 129 | `document=TODO/completed/321d_preflight_gate_audit.md; regex=### §2\.1\s+`all`\s+pipeline\s+—\s+(\d+)\s+gates` |
| 4 | `molmetal/tests/test_wfa127_residue_closure.py:142` | `all` | 129 | 129 | `document=molmetal/tests/test_wf415a3_preflight_fail_audit.py; regex=assert preflight_summary\["total"\]\s*==\s*(\d+)` |

Unresolved (the literal is extracted by a second-level regex; the document is identified but the number is not machine-extracted):

- `molmetal/tests/test_verify_narrative_claims.py:178` — document=molmetal/scripts/pre_flight_check.py; regex=EXPECTED_GATE_COUNTS = \{(.*?)\n\}

## 6 — DYNAMIC invariants (no literal; compare constant to live)

| # | file:line | pipeline |
|---|---|---|
| 1 | `molmetal/tests/test_wf343c4_integration.py:625` | `all` |
| 2 | `molmetal/tests/test_wf343c4_integration.py:707` | `frontier` |
| 3 | `molmetal/tests/test_wf345a2_yuelbond_env.py:423` | `all` |
| 4 | `molmetal/tests/test_wf350a1_runtime_energy.py:321` | `all` |
| 5 | `molmetal/tests/test_wf350a2_compute_manifest.py:292` | `all` |
| 6 | `molmetal/tests/test_wf350a2_compute_manifest.py:309` | `chemprop_v2_v3` |
| 7 | `molmetal/tests/test_wf350a2_compute_manifest.py:354` | `frontier` |
| 8 | `molmetal/tests/test_wf350c1_gpu_env_probe.py:518` | `all` |
| 9 | `molmetal/tests/test_wf446_l5_v1_known_leaky_gate.py:149` | `all` |
| 10 | `molmetal/tests/test_wf_r43a3_m7_subpocket_uniform_guard.py:215` | `all` |
| 11 | `molmetal/tests/test_wfa125_preflight_warn_contract.py:125` | `all` |
| 12 | `molmetal/tests/test_wfa125_preflight_warn_contract.py:129` | `launch` |
| 13 | `molmetal/tests/test_wfa128_fragmentation_gate_script.py:955` | `all` |
| 14 | `molmetal/tests/test_wfa128_fragmentation_gate_script.py:956` | `frontier` |

## 7 — PROSE

**2460 prose mentions in 515 files.** The full `file:line` list is in `prose_mentions.txt` (one row per mention, with the source line). Prose is **not** edited by Wf-A169: AGENTS.md §5.5 rule 6 — the superseded rendering of a document is preserved verbatim and a correction is additive and dated.

| extension | prose rows |
|---|---|
| `.md` | 1476 |
| `.py` | 894 |
| `.txt` | 81 |
| `.json` | 5 |
| `.sh` | 4 |

### 7.1 — prose that still names a SUPERSEDED count (230 rows)

Inventory only — **do not edit**.  These documents record what the count was at the time they were written; changing them would rewrite history (rule 6).  One exception is flagged below: a *runtime log line* prints a wrong number to a reader today.

| # | file:line | pipeline | claimed | live |
|---|---|---|---|---|
| 1 | `TODO/02_metrics/METRICS_TARGETS.md:716` | `all` | 94 | 129 |
| 2 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md:55` | `all` | 95 | 129 |
| 3 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md:55` | `chemprop_v2_v3` | 14 | 18 |
| 4 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md:55` | `cfm` | 6 | 8 |
| 5 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md:90` | `all` | 95 | 129 |
| 6 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md:450` | `all` | 95 | 129 |
| 7 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md:450` | `chemprop_v2_v3` | 14 | 18 |
| 8 | `TODO/03_plans/PRE_TRAINING_GATE_v2.md:450` | `cfm` | 6 | 8 |
| 9 | `TODO/INDEX.md:9` | `all` | 68 | 129 |
| 10 | `TODO/README.md:9` | `all` | 68 | 129 |
| 11 | `TODO/completed/321d_preflight_gate_audit.md:71` | `all` | 127 | 129 |
| 12 | `TODO/completed/321d_preflight_gate_audit.md:72` | `launch` | 129 | 131 |
| 13 | `TODO/completed/321d_preflight_gate_audit.md:440` | `all` | 127 | 129 |
| 14 | `TODO/completed/321d_preflight_gate_audit.md:440` | `launch` | 129 | 131 |
| 15 | `TODO/completed/321d_preflight_gate_audit.md:440` | `cfm` | 7 | 8 |
| 16 | `TODO/completed/321d_preflight_gate_audit.md:440` | `all` | 127 | 129 |
| 17 | `TODO/completed/321d_preflight_gate_audit.md:440` | `launch` | 129 | 131 |
| 18 | `TODO/completed/322e_todo_148_loader_api_doc.md:129` | `all` | 55 | 129 |
| 19 | `TODO/completed/322e_todo_148_loader_api_doc.md:129` | `frontier` | 21 | 25 |
| 20 | `TODO/completed/322e_todo_148_loader_api_doc.md:129` | `chemprop_v2_v3` | 13 | 18 |
| 21 | `TODO/completed/322e_todo_148_loader_api_doc.md:129` | `cfm` | 6 | 8 |
| 22 | `TODO/completed/322e_todo_148_loader_api_doc.md:130` | `all` | 56 | 129 |
| 23 | `TODO/completed/322e_todo_148_loader_api_doc.md:130` | `frontier` | 21 | 25 |
| 24 | `TODO/completed/322e_todo_148_loader_api_doc.md:130` | `chemprop_v2_v3` | 13 | 18 |
| 25 | `TODO/completed/322e_todo_148_loader_api_doc.md:130` | `cfm` | 6 | 8 |
| 26 | `TODO/completed/323f_m4_sa_qed_subcriterion.md:22` | `all` | 57 | 129 |
| 27 | `TODO/completed/323f_m4_sa_qed_subcriterion.md:22` | `frontier` | 24 | 25 |
| 28 | `TODO/completed/327_ultracode_remaining_problems_closeout.md:73` | `all` | 62 | 129 |
| 29 | `TODO/completed/327_ultracode_remaining_problems_closeout.md:122` | `all` | 62 | 129 |
| 30 | `TODO/completed/327f11_gate_count_docs_fix.md:36` | `all` | 61 | 129 |
| 31 | `TODO/completed/329_synthesis.md:128` | `all` | 62 | 129 |
| 32 | `TODO/completed/333_final_synthesis.md:176` | `all` | 62 | 129 |
| 33 | `TODO/completed/335_final_synthesis.md:234` | `all` | 62 | 129 |
| 34 | `TODO/completed/335_final_synthesis.md:336` | `all` | 62 | 129 |
| 35 | `TODO/completed/337u3_gpu_smoke_wrapper.md:115` | `all` | 62 | 129 |
| 36 | `TODO/completed/341_final_closeout.md:137` | `all` | 62 | 129 |
| 37 | `TODO/completed/341_final_closeout.md:703` | `all` | 62 | 129 |
| 38 | `TODO/completed/341f1_preflight_final.md:87` | `all` | 71 | 129 |
| 39 | `TODO/completed/341f1_preflight_final.md:302` | `all` | 71 | 129 |
| 40 | `TODO/completed/342_final_closeout.md:118` | `all` | 66 | 129 |
| 41 | `TODO/completed/342k2_w2s_3_5_3_6.md:220` | `all` | 57 | 129 |
| 42 | `TODO/completed/342k2_w2s_3_5_3_6.md:220` | `frontier` | 24 | 25 |
| 43 | `TODO/completed/342k3_preflight_final.md:61` | `all` | 66 | 129 |
| 44 | `TODO/completed/342k3_preflight_final.md:279` | `all` | 66 | 129 |
| 45 | `TODO/completed/343a5_audit_207.md:71` | `all` | 66 | 129 |
| 46 | `TODO/completed/343a5_audit_207.md:71` | `chemprop_v2_v3` | 13 | 18 |
| 47 | `TODO/completed/343a5_audit_207.md:71` | `cfm` | 6 | 8 |
| 48 | `TODO/completed/343a6_audit_per_wave.md:80` | `all` | 66 | 129 |
| 49 | `TODO/completed/343a6_audit_per_wave.md:330` | `all` | 66 | 129 |
| 50 | `TODO/completed/343a6_audit_per_wave.md:332` | `chemprop_v2_v3` | 13 | 18 |
| 51 | `TODO/completed/343a6_audit_per_wave.md:333` | `cfm` | 6 | 8 |
| 52 | `TODO/completed/343a6_audit_per_wave.md:379` | `all` | 66 | 129 |
| 53 | `TODO/completed/343c4_integration.md:164` | `all` | 66 | 129 |
| 54 | `TODO/completed/343d1.md:22` | `all` | 66 | 129 |
| 55 | `TODO/completed/343d1.md:119` | `all` | 66 | 129 |
| 56 | `TODO/completed/343d1.md:130` | `all` | 66 | 129 |
| 57 | `TODO/completed/343d1.md:133` | `all` | 66 | 129 |
| 58 | `TODO/completed/343d1.md:180` | `all` | 66 | 129 |
| 59 | `TODO/completed/343d2.md:113` | `all` | 66 | 129 |
| 60 | `TODO/completed/343d3.md:22` | `all` | 66 | 129 |
| 61 | `TODO/completed/343d3.md:34` | `all` | 66 | 129 |
| 62 | `TODO/completed/343d3.md:47` | `all` | 71 | 129 |
| 63 | `TODO/completed/343d3.md:93` | `all` | 66 | 129 |
| 64 | `TODO/completed/343d3.md:187` | `all` | 66 | 129 |
| 65 | `TODO/completed/344b3_gate_count.md:53` | `all` | 66 | 129 |
| 66 | `TODO/completed/345_final_synthesis.md:60` | `all` | 71 | 129 |
| 67 | `TODO/completed/345_final_synthesis.md:84` | `all` | 93 | 129 |
| 68 | `TODO/completed/345a1_hyperparam_config_complete.md:241` | `all` | 66 | 129 |
| 69 | `TODO/completed/345a3_anchor_freshness_audit.md:128` | `all` | 66 | 129 |
| 70 | `TODO/completed/345b2_docstring_dedup.md:20` | `all` | 71 | 129 |
| 71 | `TODO/completed/345b2_docstring_dedup.md:46` | `all` | 71 | 129 |
| 72 | `TODO/completed/345b2_docstring_dedup.md:57` | `all` | 71 | 129 |
| 73 | `TODO/completed/346_final_synthesis.md:102` | `all` | 66 | 129 |
| 74 | `TODO/completed/346_final_synthesis.md:267` | `all` | 69 | 129 |
| 75 | `TODO/completed/346a1_register_hyperparams_gate.md:20` | `all` | 66 | 129 |
| 76 | `TODO/completed/346a2_register_yuelbond_gate.md:71` | `all` | 66 | 129 |
| 77 | `TODO/completed/346a2_register_yuelbond_gate.md:96` | `all` | 66 | 129 |
| 78 | `TODO/completed/346a2_register_yuelbond_gate.md:163` | `all` | 66 | 129 |
| 79 | `TODO/completed/346a3_register_anchor_gate.md:25` | `all` | 66 | 129 |
| 80 | `TODO/completed/346a3_register_anchor_gate.md:152` | `all` | 66 | 129 |
| 81 | `TODO/completed/346a4_register_wave_budget_gate.md:64` | `all` | 66 | 129 |
| 82 | `TODO/completed/346a4_register_wave_budget_gate.md:86` | `all` | 66 | 129 |
| 83 | `TODO/completed/346a4_register_wave_budget_gate.md:147` | `all` | 66 | 129 |
| 84 | `TODO/completed/346b2_orphan_gate_audit.md:27` | `all` | 66 | 129 |
| 85 | `TODO/completed/346b2_orphan_gate_audit.md:27` | `chemprop_v2_v3` | 13 | 18 |
| 86 | `TODO/completed/346b2_orphan_gate_audit.md:61` | `all` | 66 | 129 |
| 87 | `TODO/completed/346b2_orphan_gate_audit.md:110` | `all` | 66 | 129 |
| 88 | `TODO/completed/346c2_doc_reconciliation_343d3.md:14` | `all` | 66 | 129 |
| 89 | `TODO/completed/346c2_doc_reconciliation_343d3.md:18` | `all` | 66 | 129 |
| 90 | `TODO/completed/346c2_doc_reconciliation_343d3.md:20` | `all` | 68 | 129 |
| 91 | `TODO/completed/346c2_doc_reconciliation_343d3.md:40` | `all` | 66 | 129 |
| 92 | `TODO/completed/346c2_doc_reconciliation_343d3.md:47` | `all` | 66 | 129 |
| 93 | `TODO/completed/346c2_doc_reconciliation_343d3.md:68` | `all` | 66 | 129 |
| 94 | `TODO/completed/346c2_doc_reconciliation_343d3.md:105` | `all` | 66 | 129 |
| 95 | `TODO/completed/346c3_index_sync.md:88` | `all` | 66 | 129 |
| 96 | `TODO/completed/347_final_synthesis.md:174` | `all` | 94 | 129 |
| 97 | `TODO/completed/347_final_synthesis.md:301` | `all` | 94 | 129 |
| 98 | `TODO/completed/347a1_wf296_integration.md:20` | `all` | 68 | 129 |
| 99 | `TODO/completed/347a1_wf296_integration.md:96` | `all` | 66 | 129 |
| 100 | `TODO/completed/347a2_wf334_integration.md:40` | `all` | 66 | 129 |
| 101 | `TODO/completed/347a2_wf334_integration.md:40` | `all` | 66 | 129 |
| 102 | `TODO/completed/347a2_wf334_integration.md:126` | `all` | 66 | 129 |
| 103 | `TODO/completed/347c2_agmd_sync.md:91` | `all` | 68 | 129 |
| 104 | `TODO/completed/347c3_doc_reconciliation.md:9` | `all` | 66 | 129 |
| 105 | `TODO/completed/347c3_doc_reconciliation.md:18` | `all` | 71 | 129 |
| 106 | `TODO/completed/347c3_doc_reconciliation.md:45` | `all` | 66 | 129 |
| 107 | `TODO/completed/347c3_doc_reconciliation.md:64` | `all` | 66 | 129 |
| 108 | `TODO/completed/347c3_doc_reconciliation.md:64` | `all` | 71 | 129 |
| 109 | `TODO/completed/347c3_doc_reconciliation.md:70` | `all` | 66 | 129 |
| 110 | `TODO/completed/347c3_doc_reconciliation.md:80` | `all` | 66 | 129 |
| 111 | `TODO/completed/347c3_doc_reconciliation.md:81` | `all` | 66 | 129 |
| 112 | `TODO/completed/347c3_doc_reconciliation.md:94` | `all` | 71 | 129 |
| 113 | `TODO/completed/347c3_doc_reconciliation.md:166` | `all` | 71 | 129 |
| 114 | `TODO/completed/347d1_full_verify.md:76` | `all` | 94 | 129 |
| 115 | `TODO/completed/348_final_synthesis.md:289` | `all` | 62 | 129 |
| 116 | `TODO/completed/348a3_index_sync_wf341.md:57` | `all` | 66 | 129 |
| 117 | `TODO/completed/349b1_wf340_p2_index_sync.md:158` | `all` | 66 | 129 |
| 118 | `TODO/completed/349b3_wf340_p4_index_sync.md:26` | `all` | 66 | 129 |
| 119 | `TODO/completed/349b3_wf340_p4_index_sync.md:175` | `all` | 66 | 129 |
| 120 | `TODO/completed/349c3_wf339_improvement_audit.md:275` | `all` | 66 | 129 |
| 121 | `TODO/completed/350_final_synthesis.md:141` | `all` | 94 | 129 |
| 122 | `TODO/completed/350a1_runtime_energy_report.md:130` | `all` | 71 | 129 |
| 123 | `TODO/completed/350a2_compute_manifest.md:149` | `all` | 71 | 129 |
| 124 | `TODO/completed/350a2_compute_manifest.md:161` | `all` | 71 | 129 |
| 125 | `TODO/completed/350a3_hp_provenance_log.md:239` | `all` | 70 | 129 |
| 126 | `TODO/completed/350b3_repro_manifest_extension.md:150` | `all` | 71 | 129 |
| 127 | `TODO/completed/350c1_gpu_env_probe.md:140` | `all` | 94 | 129 |
| 128 | `TODO/completed/352_final_synthesis.md:13` | `all` | 71 | 129 |
| 129 | `TODO/completed/352_final_synthesis.md:13` | `chemprop_v2_v3` | 14 | 18 |
| 130 | `TODO/completed/352_final_synthesis.md:13` | `cfm` | 6 | 8 |
| 131 | `TODO/completed/352a1_expected_gate_counts_bump.md:12` | `all` | 66 | 129 |
| 132 | `TODO/completed/352a1_expected_gate_counts_bump.md:107` | `all` | 71 | 129 |
| 133 | `TODO/completed/352b1_wf321d_audit_doc_sync.md:19` | `all` | 71 | 129 |
| 134 | `TODO/completed/352c1_full_verify.md:136` | `all` | 93 | 129 |
| 135 | `TODO/completed/352c1_full_verify.md:136` | `chemprop_v2_v3` | 14 | 18 |
| 136 | `TODO/completed/352c1_full_verify.md:136` | `cfm` | 6 | 8 |
| 137 | `TODO/completed/353_final_synthesis.md:114` | `all` | 94 | 129 |
| 138 | `TODO/completed/353_final_synthesis.md:243` | `all` | 94 | 129 |
| 139 | `TODO/completed/353c2_index_sync.md:103` | `all` | 93 | 129 |
| 140 | `TODO/completed/355_final_synthesis.md:26` | `all` | 62 | 129 |
| 141 | `TODO/completed/355_final_synthesis.md:246` | `all` | 62 | 129 |
| 142 | `TODO/completed/355_final_synthesis.md:303` | `all` | 62 | 129 |
| 143 | `TODO/completed/355a3_r20_p5_tmqm_wrapper.md:73` | `all` | 62 | 129 |
| 144 | `TODO/completed/356a2_anchor_refresh.md:86` | `all` | 62 | 129 |
| 145 | `TODO/completed/357a2_wf343c1_audit.md:90` | `all` | 94 | 129 |
| 146 | `TODO/completed/357a2_wf343c1_audit.md:99` | `all` | 94 | 129 |
| 147 | `TODO/completed/357a2_wf343c1_audit.md:143` | `all` | 94 | 129 |
| 148 | `TODO/completed/357a2_wf343c1_audit.md:191` | `all` | 94 | 129 |
| 149 | `TODO/completed/357a5_neurips_reproducibility_audit.md:38` | `chemprop_v2_v3` | 14 | 18 |
| 150 | `TODO/completed/357c3_workflow_analysis.md:122` | `all` | 94 | 129 |
| 151 | `TODO/completed/358a3_speedup_flags.md:207` | `all` | 62 | 129 |
| 152 | `TODO/completed/359b1_honest_neg_catalog.md:249` | `all` | 62 | 129 |
| 153 | `TODO/completed/359c1_index_sync.md:98` | `all` | 94 | 129 |
| 154 | `TODO/completed/359c1_index_sync.md:128` | `all` | 94 | 129 |
| 155 | `TODO/completed/360b2_per_pipeline_metric_audit.md:40` | `chemprop_v2_v3` | 14 | 18 |
| 156 | `TODO/completed/360b2_per_pipeline_metric_audit.md:41` | `cfm` | 6 | 8 |
| 157 | `TODO/completed/360b2_per_pipeline_metric_audit.md:43` | `all` | 94 | 129 |
| 158 | `TODO/completed/360b2_per_pipeline_metric_audit.md:237` | `all` | 94 | 129 |
| 159 | `TODO/completed/360b2_per_pipeline_metric_audit.md:364` | `all` | 94 | 129 |
| 160 | `TODO/completed/360b2_per_pipeline_metric_audit.md:364` | `chemprop_v2_v3` | 14 | 18 |
| 161 | `TODO/completed/360b2_per_pipeline_metric_audit.md:364` | `cfm` | 6 | 8 |
| 162 | `TODO/completed/360c1_index_sync.md:47` | `all` | 94 | 129 |
| 163 | `TODO/completed/361a5_wf352b3_verify.md:104` | `chemprop_v2_v3` | 14 | 18 |
| 164 | `TODO/completed/361a5_wf352b3_verify.md:104` | `cfm` | 6 | 8 |
| 165 | `TODO/completed/361a5_wf352b3_verify.md:104` | `all` | 71 | 129 |
| 166 | `TODO/completed/364a4_section6_template.md:103` | `all` | 62 | 129 |
| 167 | `TODO/completed/368a2_wf364_truncated_verify.md:106` | `all` | 94 | 129 |
| 168 | `TODO/completed/370a3_session_synthesis.md:114` | `all` | 94 | 129 |
| 169 | `TODO/completed/380a4_vina_param_tuning.md:189` | `all` | 94 | 129 |
| 170 | `TODO/completed/380a5_multi_template_mm_gbsa.md:149` | `all` | 94 | 129 |
| 171 | `TODO/completed/380a6_sa_generation_penalty.md:215` | `all` | 94 | 129 |
| 172 | `TODO/completed/382a4_strategic_plans_refresh.md:256` | `all` | 94 | 129 |
| 173 | `TODO/completed/383b1_index_sync.md:48` | `all` | 94 | 129 |
| 174 | `TODO/completed/404a2_todo_204_scaffold.md:196` | `all` | 94 | 129 |
| 175 | `TODO/completed/420a8_wf125_controlled_pair_retrain.md:47` | `chemprop_v2_v3` | 14 | 18 |
| 176 | `TODO/completed/424a7_wf380_scaffold_retrain.md:33` | `all` | 117 | 129 |
| 177 | `TODO/completed/425b5_wf125_controlled_pair_investigator.md:415` | `all` | 117 | 129 |
| 178 | `TODO/completed/432a1_pre_training_gate_v2.md:171` | `all` | 127 | 129 |
| 179 | `TODO/completed/A7_tui_deep_integration_closeout.md:41` | `all` | 117 | 129 |
| 180 | `TODO/decisions/0010-scaffold-loss-preflight-gate.md:9` | `all` | 94 | 129 |
| 181 | `docs/HONEST_NEGATIVE_CATALOG.md:19605` | `all` | 68 | 129 |
| 182 | `docs/HONEST_NEGATIVE_CATALOG.md:21231` | `all` | 95 | 129 |
| 183 | `docs/SESSION_FINAL_SYNTHESIS.md:294` | `all` | 94 | 129 |
| 184 | `molmetal/reports/wf374b1_preflight.txt:879` | `all` | 94 | 129 |
| 185 | `molmetal/reports/wfa125_verification.md:115` | `all` | 127 | 129 |
| 186 | `molmetal/reports/wfa166_b2_refutation/refutation.md:14` | `all` | 127 | 129 |
| 187 | `molmetal/reports/wfa166_b2_refutation/refutation.md:14` | `all` | 94 | 129 |
| 188 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:9` | `all` | 130 | 129 |
| 189 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:10` | `launch` | 132 | 131 |
| 190 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:15` | `all` | 127 | 129 |
| 191 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:15` | `launch` | 129 | 131 |
| 192 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:15` | `cfm` | 7 | 8 |
| 193 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:15` | `all` | 127 | 129 |
| 194 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:15` | `launch` | 129 | 131 |
| 195 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:20` | `all` | 130 | 129 |
| 196 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:21` | `launch` | 132 | 131 |
| 197 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:24` | `all` | 130 | 129 |
| 198 | `molmetal/reports/wfa166_run_ledger_provenance/blast_radius.md:25` | `launch` | 132 | 131 |
| 199 | `molmetal/reports/wfa166_synthesis/synthesis.md:158` | `all` | 127 | 129 |
| 200 | `molmetal/reports/wfa166_synthesis/synthesis.md:160` | `all` | 94 | 129 |
| 201 | `molmetal/scripts/pre_flight_check.py:15680` | `all` | 68 | 129 |
| 202 | `molmetal/scripts/wfa169_gate_count_pins.py:938` | `all` | 66 | 129 |
| 203 | `molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py:262` | `all` | 57 | 129 |
| 204 | `molmetal/tests/test_wf345a2_yuelbond_env.py:429` | `all` | 68 | 129 |
| 205 | `molmetal/tests/test_wf345a2_yuelbond_env.py:434` | `all` | 127 | 129 |
| 206 | `molmetal/tests/test_wf345a3_anchor_freshness.py:47` | `all` | 66 | 129 |
| 207 | `molmetal/tests/test_wf346a1_hyperparams_registered.py:9` | `all` | 66 | 129 |
| 208 | `molmetal/tests/test_wf346a1_hyperparams_registered.py:179` | `all` | 66 | 129 |
| 209 | `molmetal/tests/test_wf346a3_anchor_registered.py:34` | `all` | 66 | 129 |
| 210 | `molmetal/tests/test_wf346a3_anchor_registered.py:118` | `all` | 66 | 129 |
| 211 | `molmetal/tests/test_wf350a1_runtime_energy.py:39` | `all` | 70 | 129 |
| 212 | `molmetal/tests/test_wf350a1_runtime_energy.py:306` | `all` | 71 | 129 |
| 213 | `molmetal/tests/test_wf350a2_compute_manifest.py:32` | `all` | 71 | 129 |
| 214 | `molmetal/tests/test_wf350a2_compute_manifest.py:272` | `all` | 71 | 129 |
| 215 | `molmetal/tests/test_wf350a2_compute_manifest.py:282` | `all` | 71 | 129 |
| 216 | `molmetal/tests/test_wf350a2_compute_manifest.py:283` | `chemprop_v2_v3` | 14 | 18 |
| 217 | `molmetal/tests/test_wf350c1_gpu_env_probe.py:43` | `all` | 94 | 129 |
| 218 | `molmetal/tests/test_wf350c1_gpu_env_probe.py:487` | `all` | 94 | 129 |
| 219 | `molmetal/tests/test_wf350c1_gpu_env_probe.py:491` | `all` | 94 | 129 |
| 220 | `molmetal/tests/test_wf350c1_gpu_env_probe.py:529` | `all` | 94 | 129 |
| 221 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:107` | `all` | 95 | 129 |
| 222 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:108` | `all` | 127 | 129 |
| 223 | `molmetal/tests/test_wfa127_residue_closure.py:154` | `all` | 127 | 129 |
| 224 | `molmetal/tests/test_wfa127_residue_closure.py:155` | `launch` | 129 | 131 |
| 225 | `molmetal/tests/test_wfa128_preflight_classification.py:484` | `all` | 127 | 129 |
| 226 | `molmetal/tests/test_wfa128_preflight_classification.py:485` | `launch` | 129 | 131 |
| 227 | `molmetal/tests/test_wfa169_gate_count_pins.py:139` | `all` | 999 | 129 |
| 228 | `scripts/gpu_pipeline_master.sh:93` | `all` | 66 | 129 |
| 229 | `scripts/gpu_pipeline_master.sh:149` | `all` | 66 | 129 |
| 230 | `scripts/gpu_smoke_when_vram_free.sh:53` | `all` | 62 | 129 |

---

## 8 — CORRECTION (2026-11-01, Wf-A169 C2 + C3) — the retired-pin accounting, counted in SITES

APPEND-ONLY (AGENTS.md §5.5 rule #6): everything above this line is the
2026-11-01 12:31 snapshot, preserved VERBATIM.  This section is the dated
correction.  It supersedes the §1 STALE row and the §2 table; nothing above
is edited and nothing above is deleted.

### 8.1 — the superseded rendering, quoted VERBATIM

`pin_census.md:22` — the §1 population table's STALE row:

    | — of which STALE | 15 | 4 |

`pin_census.md:29` — the §1 denominator line:

    Denominator: **3025** tracked/untracked `.py` files scanned by AST + **7878** `.md`/`.sh`/`.yaml`/`.yml`/`.json`/`.txt` files scanned as prose. No file was skipped: `files_skipped = 0` (the scan reads bytes and decodes with `errors="replace"`; PATH `grep` here is `ugrep -I` and would silently skip the repo's non-UTF-8 tracked text files).

`pin_census.md:37-41` — §2 rows 3-7, the five `test_wf433a1_gate_v2_wire.py`
sites:

    | 3 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:108` | `all` | 127 | 129 | stale |
    | 4 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:120` | `all` | 127 | 129 | stale |
    | 5 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:139` | `chemprop_v2_v3` | 14 | 18 | stale |
    | 6 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:140` | `cfm` | 6 | 8 | stale |
    | 7 | `molmetal/tests/test_wf433a1_gate_v2_wire.py:365` | `all` | 127 | 129 | stale |

### 8.2 — why the accounting did not balance, and the corrected balance

The census counts a PIN SITE (one asserted literal at one `file:line`).  The
Wf-A172 close-out narrative counts a TEST FUNCTION in its second clause.  The
two units differ in exactly one place, and that is where the accounting
leaked.

SUPERSEDED narrative, quoted VERBATIM from
`molmetal/tests/test_wfa169_gate_count_pins.py:289-293`:

    Wf-A172 (2026-11-01) moved ten stale count pins across three files
    onto the SSOT constant, and moved the four stale baseline assertions
    in ``test_wf433a1_gate_v2_wire.py`` onto the SSOT and the runtime.
    A census that still reported a stale pin in any of those files would
    mean the repair silently kept a literal.

MEASURED 2026-11-01 — the 15 retired pin SITES (from the 12:31 census
snapshot, `molmetal/reports/wfa169_dead_gate_repair/pin_census.json` →
`pins[].status == "stale"`), the test FUNCTIONS that now carry the
re-synced assertion (AST walk over the current tree; a function counts when
its source segment contains the marker `Wf-A172 (2026-11-01) REPAIR`), and
the balance:

| file | retired pin SITES | re-synced test FUNCTIONS | site lines (12:31 tree) |
|---|---|---|---|
| `molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py` | 2 | 1 | 262, 263 |
| `molmetal/tests/test_wfa127_residue_closure.py` | 4 | 1 | 152, 153, 154, 155 |
| `molmetal/tests/test_wfa128_preflight_classification.py` | 4 | 1 | 482, 483, 484, 485 |
| `molmetal/tests/test_wf433a1_gate_v2_wire.py` | **5** | 4 | 108, 120, 139, 140, 365 |
| **TOTAL** | **15** | **7** | |

**The corrected balance, in SITES: 15 = 10 (three files) + 5
(`test_wf433a1_gate_v2_wire.py`).**  The superseded narrative's "ten …
across three files" is a SITE count and is correct; its "the four stale
baseline assertions in `test_wf433a1_gate_v2_wire.py`" is a FUNCTION count
presented as an assertion count, and it is ONE SHORT of the five SITES the
census measured there.

**The one unaccounted retired pin is `:140`** — the `cfm == 6` site in
`molmetal/tests/test_wf433a1_gate_v2_wire.py`.  It shares its test function
with `:139` (the `chemprop_v2_v3 == 14` site): both live inside
`test_wf432a1_frontier_pipeline_subset_unchanged` (now at `:323` after the
repair), so the narrative's FUNCTION count of 4 gives that function ONE
slot where the census measured TWO SITES.  `:140` is the site that falls
outside the "four".  Counting FUNCTIONS gives 4; counting SITES gives 5.  No
pin was left behind by the repair — every one of the 15 SITES is re-synced
(§8.3 measures `stale_pin_count = 0` on the live tree) — the prose simply
mixed the two units.

### 8.3 — the TRUE count today (re-derived, not inherited)

The tree was repaired after the 12:31 census, so §1/§2 above describe a
state that no longer exists.  The live census, re-run 2026-11-01 after the
C2 fix (§8.4) and written to
`molmetal/reports/wfa169_dead_gate_repair/pin_census_post_c2c3_2026-11-01.json`:

| population | 12:31 snapshot (`pin_census.json`) | LIVE 2026-11-01 (`pin_census_post_c2c3_2026-11-01.json`) |
|---|---|---|
| EXECUTABLE-PIN (exact `==`) | 24 in 9 files | **8 in 6 files** |
| — of which STALE | **15 in 4 files** | **0 in 0 files** |
| — of which current | 9 in 7 files | 8 in 6 files |
| FLOOR guard (`>=`, `<=`) | 6 in 5 files | 6 in 5 files |
| DYNAMIC invariant (no literal) | 14 in 9 files | 20 in 12 files |
| DOC-DERIVED pin (literal in a document) | 4 in 3 files | 4 in 3 files |
| PROSE (never a pin) | 2460 in 515 files | 2645 in 531 files |
| `.py` files scanned (denominator) | 3025 | **3056** |
| prose files scanned | 7878 | 7935 |
| prose rows naming a superseded count | 230 | 235 |

**All 15 retired SITES are gone from the live census** — `stale_pin_count = 0`
and `stale_pin_file_count = 0` in the live JSON — consistent with §8.2: the
repair retired 15 SITES carried by 7 test FUNCTIONS.  The 12:31 snapshot JSON
is kept UNCHANGED at `molmetal/reports/wfa169_dead_gate_repair/pin_census.json`
so that the §1/§2 tables above stay citable; the refreshed census is a NEW
file beside it rather than an overwrite.

### 8.4 — the C2 fix (the declared exclude was never passed)

`molmetal/scripts/wfa169_gate_count_pins.py` DECLARED the exclusion of the
census's own output directory and never applied it to the pin scan.  Line
numbers below are PRE-FIX:

* `CENSUS_OUTPUT_REL = "molmetal/reports/wfa169_dead_gate_repair"` (line 98)
  — the constant, documented as "EXCLUDED from the scan";
* `scan_gate_count_pins(..., exclude: Sequence[str] = ())` (line 853) — the
  exclude parameter, honoured at line 877 (`rel.startswith(ex)`);
* the CALL SITE, `pin_census` → `scan_gate_count_pins(root, include=include,
  live=live)` (line 986) — **no `exclude=` was passed**, so the declared
  exclusion was dead code at the only place that mattered;
* `_iter_prose_files` DID apply it (line 765) but `_iter_python_files` did
  not (lines 724-737) — the prose population excluded the census's own
  output while the python population did not.

FIXED (three edits, all in `molmetal/scripts/wfa169_gate_count_pins.py`):

1. `pin_census` now passes `exclude=(CENSUS_OUTPUT_REL,)` to
   `scan_gate_count_pins` — the declared exclude is actually passed;
2. `_iter_python_files` now skips `CENSUS_OUTPUT_REL`, so the python
   population and its denominator agree with the prose population;
3. the `exclude` parameter is documented in the `scan_gate_count_pins`
   docstring.

MEASURED before/after (same command, same tree, CPU-only — no torch, no
GPU, no trainer):

```
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/wfa169_gate_count_pins.py --json <out>
```

| counter | before the fix | after the fix |
|---|---|---|
| `executable_pin_count` / `executable_pin_file_count` | 8 / 6 | 8 / 6 (SAME) |
| `stale_pin_count` / `stale_pin_file_count` | 0 / 0 | 0 / 0 (SAME) |
| `ok_pin_count` | 8 | 8 (SAME) |
| `floor_guard_count` / `floor_guard_file_count` | 6 / 5 | 6 / 5 (SAME) |
| `dynamic_invariant_count` | 20 | 20 (SAME) |
| `doc_derived_pin_count` | 4 | 4 (SAME) |
| `prose_mention_count` / `prose_mention_file_count` | 2645 / 531 | 2645 / 531 (SAME) |
| `prose_stale_number_count` | 235 | 235 (SAME) |
| `files_scanned_prose` | 7935 | 7935 (SAME) |
| **`files_scanned_python`** | **3059** | **3056** (DIFF, −3) |
| the `pins` list itself | — | identical (SAME) |

**The pin population is UNCHANGED — the fix is a NO-OP on every pin
counter.**  MEASURED reason: the three `.py` files under the census's own
output directory (`regen_block.py`, `regen_patch.py`,
`wfa169b2_exec_verdict.py`, enumerated by `wfa156_text_scan.iter_repo_files`
with `untracked=True`) hold **0** executable gate-count pins — scanned
individually with `scan_file_for_pins`, each returns an empty pin list.  The
only number that moves is the DENOMINATOR, `files_scanned_python` 3059 →
3056: the census no longer counts its own output directory as scanned
python, which is what `CENSUS_OUTPUT_REL` has always claimed.
