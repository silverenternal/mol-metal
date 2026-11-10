# Wf-A175 — ADVERSARIAL REFUTATION of the WIRING branch's dead-gate audit

**Branch:** WIRING  ·  **Refuting agent:** Wf-A175 refutation branch
**Date:** 2026-11-01 · **CPU-only** (`CUDA_VISIBLE_DEVICES=""`, `/home/hugo/codes/mol-metal/.venv/bin/python`)
**Writes confined to:** `molmetal/reports/wfa175_dead_gate_wiring/refutation/`
**Read-only respected:** `paper/**`, `docs/HONEST_NEGATIVE_CATALOG.md`, `AGENTS.md`, `TODO/**`, `README.md`,
`molmetal/scripts/pre_flight_check.py`, `metrics/**`. No git mutation.

**VERDICT: REFUTED 0 | SURVIVED 4 | IMPRECISE 4.** No WIRING claim is false. Two of the eight are
imprecisely worded; the corrections are counted, not conceptual.

---

## Method

1. Re-ran `build_gates()` live for every pipeline, against the repo's own `PIPELINES` constant
   (`molmetal/scripts/pre_flight_check.py:451`) — not against a list the WIRING branch chose.
2. Re-ran all 24 gates against the real tree with the canonical arg set
   (`molmetal/tests/test_wf433a1_gate_v2_wire.py:259-349`) — `_probe24.py` → `probe24.json`.
3. Read the source of all 24 gate functions + `_hook_passthrough_payload` + `_emit_protocol_validation_hook_json`.
4. Re-ran the repo's own `molmetal/scripts/wfa171_l5_threshold.py` and read every record.
5. Re-measured every disk claim (paths, counts, row counts, BibTeX keys, metric JSON values).

**One documented deviation.** The canonical arg set passes
`pv_dir = molmetal/reports/protocol_validation` to the 9 L5 gates, and the L5 emitter **creates** that
directory. To avoid writing outside the refutation directory the probe redirects `pv_dir` to
`refutation/_pv_hook_dir`. The L5 verdict depends only on (a) `source_present` (unchanged) and
(b) whether `mkdir`+`atomic_write` succeeded (same outcome in either directory) — so the verdict
vector is unaffected, and I re-verified it is 19/4/1/0 independently.

---

## Per-claim verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | 24 gates registered in NO pipeline; `all`=129 `launch`=131 `frontier`=25 `chemprop_v2_v3`=18 `cfm`=8 | **SURVIVED** |
| 2 | 19 PASS / 4 WARN / 1 FAIL / 0 RAISED; 20 path / 3 substring / 1 parse; "NOT ONE evaluates its own spec threshold" | **IMPRECISE** |
| 3 | 11 MET / 7 SHORT / 5 MISSING_SOURCE / 1 UNSPEC; 8/7/2/1/6 verdicts | **IMPRECISE** |
| 4 | The 9 L5 hooks are not read-only; they create `protocol_validation/*_hook.json` | **SURVIVED** |
| 5 | krasnov_m1 25/125, krasnov_m21 9/45, rusanov_m2 3/12 SHORT; rusanov_m2_4cell MISSING_SOURCE; "the other 6 numerically MET"; the 9 gates compare nothing | **IMPRECISE** |
| 6 | `refs_bib` FAILs 0/3, Tier-1 BLOCKING, turns `all` RED and blocks every launch, needs paper auth | **SURVIVED** |
| 7 | DrugOOD 0.7479 / "NOT ALIGN — fallback cohort" is stale; PROTOCOL_ALIGNMENT_MATRIX has no §1.4 and is at `TODO/03_baselines/` | **SURVIVED** |
| 8 | 4 wire-as-is / 4 wire-after-repair / 16 must-not-wire | **IMPRECISE** |

### CLAIM 1 — SURVIVED
Live counts reproduce exactly and equal the repo's `EXPECTED_GATE_COUNTS`. **No scope cap:** the five
pipelines the claim names ARE the repo's own `PIPELINES` tuple, so "registered in NO pipeline" was
searched exhaustively. **Completeness checked:** the plan doc defines 32 gates; the other 8 (5 L3 + 3 L4)
*are* registered in `all`/`launch`, so the dead set is exactly 24, not a truncation. The repo's own
`EXPECTED_NEW_GATES` (test file:54-92) is the same 32.

### CLAIM 2 — IMPRECISE
**19 PASS / 4 WARN / 1 FAIL / 0 RAISED (denominator 24) — reproduced independently**, matching
`docs/HONEST_NEGATIVE_CATALOG.md` §6.266.5. The 20/3/1 implementation-kind split is correct (verified
from each function's source).

*Correction:* "**NOT ONE** evaluates its own spec threshold" is over-stated for **2** of 24.
`p3_protocol_recommended_drugood_3_splits_full_alignment` DOES evaluate a count threshold
(`len(splits_present) >= 3`); and `p3_protocol_alignment_dataset_match` evaluates a 6-path count where
its spec says 7. The drugood gate is the sharper case — running its **own** detector against the
correct file `metrics/by_metric/drugood_ood_auc.json` returns `['assay','size']` = **2/3**, because
`'size'` is matched only by the top-level `batch_size` key and `'scaffold'` appears in **no** top-level
key (it lives under `per_axis_auc` / `gate_pass`). So a path re-point alone would still WARN. The
branch's own "re-point + fix nested-key detector" repair is right; its "NOT ONE" framing is not.

### CLAIM 3 — IMPRECISE
Both histograms are **exactly** reproducible from `decision_table.json` and each sums to 24. The
underlying counts check out on disk (6/6 datasets, 7/7 catalog entries, 0/3 BibTeX keys, 25/125, 9/45,
3/12, 3/3 splits, 2/2, 1/1).

*Correction:* the `MEASURED` tag over-claims for the **labels**. Three are declared interpretations,
not measurements: `metric_match` = UNSPEC is a judgement that no oracle exists; `honest_limitation` =
MET 7/7 is a **loose** reading (the branch's own blocker discloses the repo's stricter rule scores
5/7); `dataset_match` = MET is 6/6 where the spec says 7. Correct tag: *the counts are MEASURED; the
labels are measured-or-declared.*

*Regex blind-spot note (this cut FOR the branch):* a case-sensitive `\bm15\b` scan of the catalog
returns **0** hits — which would have looked like "no §6 entry for m15". The entry exists as
`metric_id = m15_rmsd` in §6.23. The 7/7 label is correct.

### CLAIM 4 — SURVIVED
`_emit_protocol_validation_hook_json` (`pre_flight_check.py:12322`) does
`hook_dir.mkdir(parents=True, exist_ok=True)` then `atomic_write_file({baseline}_hook.json)`, and
`_hook_passthrough_payload` calls it on **every** invocation of all 9 L5 gates — before the verdict is
computed, so even a WARNing gate writes. My probe produced
`{krasnov,rusanov,drugood}_hook.json`, each carrying its baseline's 3 hook_ids (the D2 merge is live).

*Two corrections:* (a) one call creates **one** file, not all three; (b) "the tree is as found" was true
for the branch's own probe but is **no longer true on disk** — `molmetal/reports/protocol_validation/`
was absent at 19:02 and reappeared at **2026-11-01 19:04:25** with the same three files, i.e. a
**concurrent branch** ran the same canonical probe and left the artifact. The write side-effect is
real; the "clean tree" assertion is a snapshot, not a property.

### CLAIM 5 — IMPRECISE
All four named numbers are **exact** (25/125, 9/45, 3/12 SHORTFALL; rusanov_m2_4cell MISSING_SOURCE),
and the module is genuinely read-only (its only file op is `open(path, 'r')`).

*Correction (arithmetic):* "**the other 6** numerically MET" cannot be right. The 9 records are
**3 SHORTFALL + 1 MISSING_SOURCE + 5 MEETS**. The claim names 4 and then says "the other 6" — 4 + 6 = 10
> 9 — and the 6th remaining record is the MISSING_SOURCE one it had just named, not a MET. Correct
residual: **5**.

*Additional caveat:* the 3 drugood MEETS come from an **alternate** source. The spec's primary
`molmetal/reports/wf265_collect/all_results.json` is present but **lacks the counting key**
(`primary_status = MISSING_KEY`); the MEETS come from `metrics/by_metric/drugood_ood_auc.json`
(+0.009347711256380342 / +0.001844904852242868 / −0.0028530871876835118 against a −0.02 bar).

The claim's second half — "the 9 gates compare nothing" — is **confirmed**: `verdict_field` is copied
into the hook payload and never compared to a measured value.

### CLAIM 6 — SURVIVED
`paper/refs.bib` contains **0 of 3** required keys (checked individually, not as a regex). Live verdict
FAIL. The spec's own classification is `Wave: pre-retrain (BLOCKING; 0 paper/* touches per AGENTS.md
§5.3)` (`PRE_TRAINING_GATE_v2.md:176`). Blocking path traced end to end: `main()` returns **1** when
`n_fail > 0` (`pre_flight_check.py:17595`), and `scripts/launch_wf124_p3_g5.sh:288-294` runs the
pre-flight with `|| { echo "FAIL: pre_flight_check"; exit 1; }`. The launcher runs `--pipeline launch`,
and `build_gates("launch")` normalises to `all` + launch-only additions — so an `all` registration is
also a launch registration. **"Blocks every launch" holds.**

### CLAIM 7 — SURVIVED
`metrics/by_metric/drugood_ood_auc.json`: `data_source = drugood_official_lbap_refined_ic50`,
per-axis `n_official_test_rows` {53291, 53304, 51922} (total 158517),
`per_axis_auc_delta_vs_imold` = **+0.009347711256380342 / +0.001844904852242868 / −0.0028530871876835118**
(the quoted +0.009348/+0.001845/−0.002853 match to 6 dp). The three
`.data/molmetal/drugood/ic50/csv/lbap_refined_ic50_*.csv` each hold **266523 lines = 266522 data rows**
+ header. `PROTOCOL_ALIGNMENT_MATRIX.md` §1's m17 row still reads
"NOT ALIGN — fallback cohort (5,207-row `metacytotox_fallback`, not the official DrugOOD IC50 splits …)"
— stale against the metric JSON, whose assay OOD AUC is 0.7304477112563803, not the plan doc's 0.7479.

For (b): the file exists **only** at `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md`; its headers are
§0/§1/§2/§3/§4 — **no §1.4**; the canonical arg set points at the **non-existent**
`TODO/03_plans/PROTOCOL_ALIGNMENT_MATRIX.md` (`test_wf433a1_gate_v2_wire.py:235`).

*Strengthening:* the dangling "§1.4" reference is not only in the test's arg set — the **plan doc's own
L5 spec text** cites `PROTOCOL_ALIGNMENT_MATRIX.md §1.4` (e.g. `PRE_TRAINING_GATE_v2.md:287`, `:311`).
The staged patch must correct the SPEC cross-reference too.

### CLAIM 8 — IMPRECISE
The 4/4/16 partition and every rationale check out against disk (leakage_utils has **0** scaffold
tokens while `molmetal/data/splits.py:610` defines `class ScaffoldSplitter`; `auto_emit_section_6_minor.py`
does not exist; `eval_per_seed_per_split` occurs **0** times in all three trainers; `m2_r2_per_cohort.json`
carries `label_provenance.labels = "SIMULATED"` and cites `docs/DATA_INVENTORY.json` classifying
`platinai_4cell.csv` as **DO-NOT-TRAIN** — "a null test, not a measurement"). All four "wire as-is" gates
PASS live; 129 + 4 = 133.

*Three corrections:*
1. **"read-only repair" is the wrong word for `section_6_auto_emit`** — closing it means **authoring a
   new file** under `molmetal/scripts/` (the branch's own blocker text says "a NEW file under
   `molmetal/scripts/`"). And once the file exists the gate PASSes on its **existence alone** — the same
   vacuous-PASS shape the branch refuses elsewhere.
2. **"named flags occur 0×" is literally false for two of the three.** `'5-fold'` occurs **5** times in
   `train_metal_cytotox.py` (lines 4, 374, 399, 404, 412 — prose about Krasnov's protocol), and
   `per_cell` occurs in both trainers (1 and 5 hits) as an output JSON key. Only the **CLI-flag** reading
   is right; the conclusion (trainer wire-in needed) still holds.
3. **`seed_cohort` has a cheaper route the branch omits:** the trainers already emit per-seed JSON
   (`--seeds` default [0,1,2]), so correcting the **spec text** alone closes it with no trainer change and
   no §5.7 smoke. The branch's own blocker says this — listing it as "must not wire" without it
   overstates the blocker.

*Two additions the branch does not state:*
- **Wiring only 4 does not turn the repo green.** `test_wf432a1_all_32_new_gates_present_in_build_gates`
  (`test_wf433a1_gate_v2_wire.py:171`) asserts **every** one of the 32 `EXPECTED_NEW_GATES` is in
  `build_gates("all")` — with 4 of 24 wired it stays **RED with 20 missing**.
- **A latent trap is already wired at the config layer.**
  `configs/training/nci60_9cell_5seed_50ep_v1.yaml:243-244` declares
  `directives.pre_flight_gates_required: [p3_protocol_alignment_user_ok_gate,
  p3_protocol_alignment_baseline_paper_refs_bib, …]`, and `pre_flight_check.py --gates` resolves against
  the union of `PIPELINES` and **hard-errors** (`return 1`) on an unknown gate name
  (`pre_flight_check.py:17393-17397`). Both of the first two are among the 24 dead. No tracked shell
  script currently passes `--gates` (the flag appears only in the tool, docs and the catalog), so this is
  **latent, not a live RED** — but it is a live trap the moment a launcher honours the config list.

---

## Artefacts

| Path | What |
|---|---|
| `refutation/_probe24.py` | re-runnable CPU-only probe of the 24 gates (pv_dir redirected) |
| `refutation/probe24.json` | 24 live verdicts — 19 PASS / 4 WARN / 1 FAIL / 0 RAISED |
| `refutation/_pv_hook_dir/` | the 3 hook JSONs my probe wrote (proof of the L5 write side-effect) |
| `refutation/refutation.json` | per-claim verdicts + counter-evidence + measured values |
| `refutation/refutation.md` | this file |

`molmetal/reports/protocol_validation/` was **not** created by this branch (redirected away) — it exists
on disk because a **concurrent** branch left it at 19:04:25.
