# User Decision Prep — 2026-09-14

## Overview

Last 6 user-gated decisions identified by the ultracode audit. Each has a default, a deadline, a recommended option with cost/risk/dependency, and the alternative options. The user picks one (or "Other"); Claude proceeds.

This file aggregates open decisions surfaced across the ultracode audit layer set:
`molmetal/reports/ultracode_audit/{agent_a_planning,agent_b_evidence,agent_c_lambda,agent_d_model,PROJECT_STATUS}.md`,
plus the previously approved `TODO/decisions.md` (D1-D5 closed 2026-09-12; D6/D7 still open).
The four smaller calls (test_005 cohort, cite-only SOTA column, MW-range flag, journal) are
listed in PROJECT_STATUS.md §8 and agent_a_planning.md §3 but not yet on file in decisions.md.

D6 and D7 are already on file in `TODO/decisions.md`; they are reproduced here so the
user has a single sheet with all six pending decisions. They are not duplicated in `decisions.md`.

---

## D6 — REINVENT4 install path

**Options:**
- (a) separate venv at `/mnt/storage/env-projects/reinvent4-rocm` (already working for install + NLL RPC; multiproperty bridge still RDKit proxy)
- (b) shared venv with molmetal (one fewer env to maintain; but RDKit/torch conflict risk)
- (c) Docker container (full isolation; +overhead; reproducibility win)

**Recommended: (a) — already verified, lowest risk.**
- **Cost:** already paid (env already created at `/mnt/storage/env-projects/reinvent4-rocm`; install + learned NLL RPC verified per `molmetal/reports/reinvent4_learned_smoke/`).
- **Risk:** multiproperty bridge is still an RDKit proxy — NLL ≠ activity. Per R5, the multiproperty layer remains OBS until a real task-specific worker is configured.
- **Dependency:** TODO-15 anticancer metric suite could optionally use REINVENT4's multiproperty layer once a real worker is wired.
- **Deadline:** **2026-09-19.**
- **Reference:** `molmetal/reports/ultracode_audit/PROJECT_STATUS.md` §8; `TODO/decisions.md` D6; `TODO/pending/05_reinvent4_install.md`.

---

## D7 — Vina→QVina swap activation

**Options:**
- (a) keep Vina 1.2.7 default
- (b) flip to QuickVina 2 default
- (c) both engines in headline table (default per audit)

**Recommended: (c) — byte-identity proved, parity experiment in flight (workflow r11-parity-n50).**
- **Cost:** 1 ultracode round for the N=50 parity run (rounds 11-12).
- **Risk:** scoring-identity gap may invalidate the (c) path — fallback is (a) if QVina empirical MAD > 0.6 kcal/mol vs published QVina runs.
- **Dependency:** `round11_engine_parity_n50.md` write-up (not yet executed; `round9_qvina_parity.md` §5 calls this "the open empirical question").
- **Deadline:** after r11 parity run completes.
- **Reference:** `PROJECT_STATUS.md` §8; `molmetal/reports/quickvina2_binary_identity.md`; `TODO/decisions.md` D7.

---

## test_005 (4RN0 ASP B101) cohort decision

**Options:**
- (a) keep as separately labelled "modeled atoms" experiment
- (b) drop from Round-12/13 cohorts entirely

**Recommended: (a) — labeled transparency preserves the 10-receptor coverage target; (b) loses statistical power.**
- **Cost:** small (write-up overhead per pocket table — explicit "modeled ASP B101 sidechain" footnote).
- **Risk:** (a) requires explicit framing in every cohort summary; reviewers may still query the modeled receptor. Acceptable: per `crossdocked_first10_resolved/README.md` and `agent_b_evidence.md` §3.2 item 12, the ASP B:101 CG/OD1/OD2 atoms are physically absent from deposited 4RN0.pdb — this is a real data integrity limitation, not an engineering choice.
- **Dependency:** none — affects write-up only.
- **Deadline:** before Round-12 scientific-budget run.
- **Reference:** `TODO/pending/17_aggregate_weak_impls_and_pending.md`; `agent_b_evidence.md` §3.2 item 12; `PROJECT_STATUS.md` §8 (test_005 row); `TODO/decisions.md` D6 mentions the 9/10 strict-resolved status.

---

## Cite-only SOTA column

**Options:**
- (a) ship cite-only SOTA column when GPU checks underperform (per D1 cite-only path)
- (b) only include measured rows

**Recommended: (a) — aligns with D1 (DiffSBDD cite-only), preserves scientific honesty without dropping context.**
- **Cost:** small (one additional column per cohort table + per-row footnote).
- **Risk:** reviewer may treat cite-only rows as "Mol-Metal claims" if not clearly labelled — explicit per-row footnote required (pattern: "*(not re-run by Mol-Metal; cited from <ref>)*").
- **Dependency:** D1 already approved (2026-09-12).
- **Deadline:** before paper draft.
- **Reference:** `TODO/decisions.md` D1; `PROJECT_STATUS.md` §8 (cite-only SOTA column row); `agent_a_planning.md` §3.

---

## MW-range flag (Lipinski MW vs IV 300-700 Da)

**Options:**
- (a) keep Lipinski MW descriptive flag only
- (b) add 300-700 Da IV anticancer MW-range flag
- (c) both flags side-by-side

**Recommended: (c) — anticancer metric survey recommends logP 2-5 + TPSA 60-150 + IV MW 300-700 alongside standard flags; dual flag = maximum interpretability.**
- **Cost:** trivial (add column to per-pocket table + cohort aggregates).
- **Risk:** (b) may look like claim of clinical relevance — frame as "IV-appropriate window, not clinical efficacy" (parallels the "heuristics do not establish efficacy" framing already in `priors/anticancer_metric_suite.py`).
- **Dependency:** TODO-15 anticancer metric suite (currently wired for logP/TPSA/RotB/MW + IV/metal-adjusted flags via `AnticancerMetricSuite.descriptor_report()`).
- **Deadline:** before paper draft.
- **Reference:** `molmetal/reports/anticancer_vs_general_metrics_survey.md` §6; `PROJECT_STATUS.md` §8 (MW range flag row); `TODO/pending/14_full_100pocket_paper_r13.md`.

---

## Journal choice (Round-13)

**Options:**
- (a) Digital Discovery (RSC, Q1 IF ~8.5, ML × chemistry)
- (b) J. Chem. Inf. Model. (ACS, Q1 IF ~5.6, methods-heavy)
- (c) Patterns (Cell, Q1 IF ~6.0, broader scope)
- (d) Briefings in Bioinformatics (Oxford, Q1 IF ~7.0, interpretability)

**Recommended: depends on the dominant story:**
- If metal-anticancer benchmarks win: (b) JCIM — methods + pharma audience.
- If Triton-on-consumer-GPU + lipman-FM win: (a) Digital Discovery.
- If MCTS-on-metal-complex + symbolic-discovery win: (c) Patterns.
- If closed-loop Lambda proof-theoretic story wins: (d) Briefings in Bioinformatics.

- **Cost:** deferred to Round-13 (~7-14 d after Round-12 acceptance).
- **Risk:** journal mismatch = desk reject; defer until Round-12 acceptance clarifies story. Note: Round-13 work is currently *designed not started* per `PROJECT_STATUS.md` §7.
- **Dependency:** Round-12 scientific-budget results + which axis (A-F) of the 6-axis ablation has the strongest measured on/off delta.
- **Deadline:** Round-13 submission window (target W7 / 2026-10-31).
- **Reference:** `TODO/pending/14_full_100pocket_paper_r13.md`; `PROJECT_STATUS.md` §8 (journal choice row); `agent_a_planning.md` §3.

---

## Decision status table

| ID | Decision | Default | Deadline | Status |
|---|---|---|---|---|
| D6 | REINVENT4 install path | (a) separate venv | 2026-09-19 | awaiting user input |
| D7 | Vina→QVina swap | (c) both engines | post-r11-parity | awaiting parity run |
| test_005 | cohort inclusion | (a) labeled modeled-atom | pre-R12-budget | awaiting user input |
| cite-only-SOTA | ship column | (a) ship with footnote | pre-paper-draft | awaiting user input |
| MW-range flag | dual flag | (c) both | pre-paper-draft | awaiting user input |
| journal | Round-13 pick | story-dependent | 2026-10-31 | awaiting Round-12 results |

---

## How to decide

For each item, the user replies with the option letter or "Other". Claude proceeds.

---

## Cross-references

- `TODO/completion_audit_2026-09-13.md` — authoritative state
- `TODO/decisions.md` — D1-D5 already approved; D6/D7 reproduced here for one-sheet view
- `TODO/risks.md` — open blockers (R1 cite-only SOTA, R2 Vina-vs-QVina parity, R3 PoseBusters CuAAC, R4 metal_hybrid stability re-measure BLOCKED, R5 REINVENT4 live score still OBS at multiproperty layer, R6 tmQM pretraining wireup resolved, R7 PDBbind blocked, R8 pIC50 Pearson r=0.407)
- `TODO/pending/17_aggregate_weak_impls_and_pending.md` — weak impls
- `TODO/pending/14_full_100pocket_paper_r13.md` — Round-13 paper outline
- `molmetal/reports/ultracode_audit/PROJECT_STATUS.md` — full audit synthesis
- `molmetal/reports/ultracode_audit/agent_a_planning.md` — planning layer (synthesis in §3)
- `molmetal/reports/ultracode_audit/agent_b_evidence.md` — evidence layer (measurement gaps §3.2)
- `molmetal/reports/ultracode_audit/agent_c_lambda.md` — lambda line
- `molmetal/reports/ultracode_audit/agent_d_model.md` — model line
- `molmetal/reports/quickvina2_binary_identity.md` — QVina 2 byte-identity proof
- `molmetal/reports/anticancer_vs_general_metrics_survey.md` — MW / logP / TPSA / RotB / hERG recommendations
- `molmetal/reports/round9_qvina_parity.md` — parity open question
