# Wf-343c-I1 — Pre-Retrain Check Script

**Date**: 2026-09-29 (post-Wf-309..323 ship wave; pre-Wf-343 launch)
**Owner**: Hugo Lee + ultracode agents
**Status**: SHIPPED (per_gate_count = 26; n_tests = 8; integration_status = PASS)
**Parent**: `科研进展_pre_retrain_checklist_2026-09-28.md` (pre-retrain acceptance checklist)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
                 + `mol-metal-gpu-execution` §6 (pre-launch gate; 5-seed × 50-epoch)

---

## §0 Why this script exists

The pre-retrain checklist at `科研进展_pre_retrain_checklist_2026-09-28.md`
captures **24 honest-framing lessons** from Wf-243..252 + Wf-309..323.
Before this script, the checklist was **manual-only** — every gate was
filled by a human eyeballing the code, which is both time-consuming and
error-prone (per Wf-244-B: the indices aliasing bug at line 512 sat
unfixed for several waves because no one re-grepped the file before
retrain).

This script **automates the auto-verifiable subset** so:

1. **Pre-launch gate is reproducible** — same gates every time, no
   honest errors possible.
2. **Honest framing is preserved** — post-retrain gates return SKIP (not
   FAIL); data-fact gates (BindingDB external) return SKIP with reason.
3. **JSON output is machine-readable** — for `jq` post-processing or
   wave-budget tracking in `molmetal/reports/wf343c1_pre_retrain/`.

---

## §1 Ship surface

| Artefact | Path | Status |
|---|---|---|
| Script | `molmetal/scripts/wf343c1_pre_retrain_check.py` | ✅ SHIPPED |
| Tests | `molmetal/tests/test_wf343c1_pre_retrain_check.py` | ✅ SHIPPED (8/8 PASS) |
| Doc | `TODO/completed/343c1_pre_retrain_check.md` | ✅ SHIPPED (this file) |

**Stdout invariant**: every gate emits `PASS` / `FAIL` / `SKIP` (with
`--strict` for SKIP-as-fail). Exit codes: 0=PASS, 1=FAIL, 2=SKIP.

**JSON schema**: `{timestamp, schema_version, pipeline, summary, gates,
parent_doc_reference}` — mirrors the Wf-259 `pre_flight_check.py`
schema so downstream tooling can read both reports uniformly.

---

## §2 Gate registry (per-pipeline)

Per the parent doc §1.1 + §1.3 + §2.1 + §3.1 + §5.11-5.14 + §8.15:

| Pipeline | Gate count | Gate IDs | Auto-verifiable? |
|---|---:|---|---|
| **code** | 10 | §1.1.1, §1.1.2, §1.1.3, §1.1.4, §1.1.5, §1.1.6, §1.1.7, §1.1.8, §1.1.9, §1.1.10 | YES (grep + help) |
| **ablation** | 3 | §1.3.1, §1.3.2, §1.3.3 | YES (help) |
| **data** | 7 | §2.1.1, §2.1.2, §2.1.3, §2.1.4, §2.1.5, §2.1.6 (SKIP), §2.1.7 | YES (file_exists + min_lines); §2.1.6 SKIP |
| **test** | 1 | §3.1.5 (NaN guard test file present) | YES (file_exists) |
| **post_retrain** | 4 | §5.11, §5.12, §5.13, §5.14 | SKIP today (post-retrain evidence only) |
| **honest_framing** | 1 | §8.15 (entry count ≥ 15) | YES (regex on parent doc) |
| **Total** | **26** | (across all 6 pipelines) | 21 PASS / 5 SKIP today |

**Per-gate primitive mapping** (mirrors `pre_flight_check.py`):

| Primitive | Used by | Notes |
|---|---|---|
| `gate_grep` | §1.1.1, §1.1.2, §1.1.3, §1.1.4, §1.1.5, §1.1.6, §1.1.9 | Substring search in source file |
| `gate_help` | §1.1.7, §1.1.8, §1.1.10, §1.3.1, §1.3.2, §1.3.3 | CLI `--help` token presence |
| `gate_file_exists` | §2.1.3, §2.1.4, §2.1.5, §3.1.5 | File/dir existence |
| `gate_min_lines` | §2.1.1, §2.1.2, §2.1.7 | Line-count fingerprint |
| `gate_post_retrain_skip` | §5.11, §5.12, §5.13, §5.14 | Always SKIP (post-retrain evidence) |
| `gate_section8_entry_count` | §8.15 | Regex `^\| \*\*8\.\d+\*\*` count |

---

## §3 Honest framing (party-twenty-spirit §3.5)

### §3.1 What this script DOES verify

1. **Code-level fixes from Wf-243..252 are still on disk** — 10 grep
   gates cover the indices aliasing fix, Kendall log_sigma, DANN
   per_metal mode, NaN guard, gradient clipping, epoch-avg logging,
   ablation flags, EnhancedMetalHead, click-aug, and master modules.
2. **Data-level prerequisites are present** — 7 gates cover
   MetalCytoToxDB 5-metal CSV, raw MetalCytoToxDB.csv, DrugOOD dir,
   NCI-60 dir, PlatinAI 4-cell, click yields literature.
3. **Test infrastructure is in place** — 1 gate verifies the Wf-253-C
   NaN guard test file is present (so we can re-run it as part of
   §3.1.5 verification on demand).
4. **Honest framing catalog has full content** — 1 gate verifies the
   parent doc §8 table has ≥ 15 entries (currently exactly 15 per
   §8.1-§8.15).

### §3.2 What this script DOES NOT verify (honest-negative)

1. **Post-retrain evidence** (§5.11-§5.14) — these are post-retrain
   outcomes (per-seed std, Os AUC non-degenerate, Ru w2 epoch-avg
   non-zero, no NaN test_auc across 5×5=25 runs). Returned as SKIP
   today; filled in after the retrain wave completes.
2. **BindingDB retrain** (§2.1.6) — data fact, not blocker. BindingDB
   has ~50 precious-metal entities; NOT used in retrain per parent doc.
   Returned as SKIP with explicit honest-negative reason.
3. **Wall-time budget approval** (§4 + §10.5 + §10.9) — manual user
   OK required per the standing "kill then adapt" directive (AGENTS.md
   §5.1) — NOT in scope for an auto-verifier.
4. **Per-lever lift table** (§7) — requires the full 60-run ablation
   (4-config × 5-metal × 3-seed). Deferred to GPU retrain.
5. **Post-mortem 12 lessons** (§0) — these are retrospective, not
   auto-verifiable. Stay in the parent doc as-is.

---

## §4 Test surface (8/8 PASS)

| # | Test | Validates |
|---|---|---|
| **1** | `test_cli_runs` | `--json path` exits 0/1 + JSON file is parseable |
| **2** | `test_json_schema` | Top-level + per-gate schema keys; status in {PASS, FAIL, SKIP}; priority in {P1, P2}; wall_sec is numeric |
| **3** | `test_pipeline_filter` | `--pipeline code` runs ONLY code gates (no data/ablation/etc.) |
| **4** | `test_summary_counts_match_gates` | pass+fail+skip == len(gates) invariant |
| **5** | `test_gate_count_consistency` | EXPECTED_GATE_COUNTS sums to 26; per-pipeline spot-checks (code=10, ablation=3, data=7, test=1, post_retrain=4, honest_framing=1) |
| **6** | `test_post_retrain_skip_is_honest_negative` | §5.11-§5.14 + §2.1.6 (BindingDB) return SKIP (not FAIL) |
| **7** | `test_parent_doc_reference_present` | JSON `parent_doc_reference` field includes parent doc filename + §-family markers |
| **8** | `test_gate_result_helpers_unit` | Direct unit tests of `GateResult.to_dict` + `gate_grep` PASS/FAIL + `gate_file_exists` PASS/FAIL + `gate_min_lines` PASS/FAIL + `_grep_file` / `_line_count` helpers |

**Test run** (verified 2026-09-29):

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf343c1_pre_retrain_check.py -v
...
8 passed, 1 warning in 124.13s (0:02:04)
```

---

## §5 Honest framing on the §8.15 entry-count gate

The §8 honest framing catalog at `科研进展_pre_retrain_checklist_2026-09-28.md`
§8.1-§8.15 has **exactly 15 entries**. The gate uses `min_entries=15`
to enforce that the catalog stays at-or-above 15 entries as future
workflows add NEW entries.

The gate pattern `^\| \*\*8\.\d+\*\*` anchors on the markdown table-row
format used in the parent doc. This matches the convention used by the
existing pre-flight check (`gate_section9_entry_count` in
`pre_flight_check.py`) so the two reports stay consistent in tooling.

---

## §6 CLI usage

### §6.1 Default (all pipelines, stdout only)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/wf343c1_pre_retrain_check.py \
  --pipeline all --verbose
```

### §6.2 JSON output (for wave-budget tracking)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/wf343c1_pre_retrain_check.py \
  --pipeline all \
  --json molmetal/reports/wf343c1_pre_retrain/audit.json
```

### §6.3 Strict mode (SKIP exits non-zero)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/wf343c1_pre_retrain_check.py \
  --pipeline all --strict
```

This will exit 2 (not 0) when any post-retrain gate SKIPs. Useful for
"are we 100% ready to launch?" checks; default mode is more permissive
(SKIP is informational, not blocker).

### §6.4 Single-pipeline filter

```bash
# Just code-level fixes
molmetal/scripts/wf343c1_pre_retrain_check.py --pipeline code

# Just data-level prerequisites
molmetal/scripts/wf343c1_pre_retrain_check.py --pipeline data

# Just ablation flags
molmetal/scripts/wf343c1_pre_retrain_check.py --pipeline ablation
```

---

## §7 Integration with pre_flight_check.py

This script is **complementary** to `pre_flight_check.py` (Wf-259 +
Wf-306C + Wf-321D), not a replacement:

| Surface | `pre_flight_check.py` | `wf343c1_pre_retrain_check.py` |
|---|---|---|
| **Source doc** | `TODO/completed/207_unified_pre_flight_checklist_v2.md` | `科研进展_pre_retrain_checklist_2026-09-28.md` |
| **Focus** | Pre-flight v2 (auto-verify P1 + P2 + P3 gates across chemprop + CFM + frontier math) | Pre-retrain self-audit (auto-verify §1.1 + §1.3 + §2.1 + §3.1 + §5 + §8.15 gates) |
| **Gate count** | 100+ (across all + cfm + chemprop + frontier pipelines) | 26 (across 6 sub-pipelines) |
| **Schema version** | `207.v1` | `343c1.v1` |
| **JSON shape** | identical (top-level + per-gate) | identical (top-level + per-gate) |

Both scripts:
- Are CPU-only (no `torch.cuda` calls).
- Use the same `GateResult` dataclass + `gate_grep` / `gate_help` /
  `gate_file_exists` / `gate_min_lines` / `gate_pytest` primitives.
- Emit identical JSON shape so downstream tooling can swap.
- Exit 0=PASS, 1=FAIL, 2=SKIP (with `--strict`).

Run both before retrain: `pre_flight_check.py --pipeline cfm` +
`wf343c1_pre_retrain_check.py --pipeline all` to cover both surfaces.

---

## §8 MEASURED-vs-PROJECTED tag discipline (party-twenty-spirit §3.5)

Per AGENTS.md §5.2 honest framing mandate, this doc records:

| Metric | Value | MEASURED? | Source |
|---|---:|:---:|---|
| per_gate_count | 26 | YES | `EXPECTED_GATE_COUNTS` in script; verified by `test_gate_count_consistency` |
| n_tests | 8 | YES | `molmetal/tests/test_wf343c1_pre_retrain_check.py` |
| integration_status | PASS | YES | pytest run 2026-09-29 (8/8 passed, 124.13s) |
| §8.15 entries | 15 | YES | regex match on parent doc §8 table |
| Post-retrain SKIPs | 4 | YES | §5.11-§5.14 always-SKIP contract |
| BindingDB SKIP | 1 | YES | §2.1.6 data-fact, not blocker |

**Zero PROJECTED numbers in main results.** All metrics traced to JSON /
report / test output on disk.

---

## §9 Cross-references

- `科研进展_pre_retrain_checklist_2026-09-28.md` — parent pre-retrain checklist (the §1.1, §1.3, §2.1, §3.1, §5, §8 source-of-truth)
- `molmetal/scripts/pre_flight_check.py` — Wf-259 + Wf-306C + Wf-321D pre-flight v2 auto-verifier (companion script)
- `molmetal/scripts/train_chemprop_v2_v3.py` — chemprop v2/v3 driver (anchor of §1.1.x grep gates)
- `molmetal/tests/test_train_chemprop_v2_v3_nan_guard.py` — Wf-253-C NaN guard test (anchor of §3.1.5 file_exists gate)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` — pre-flight v2 checklist (companion source)
- `TODO/completed/259_wf259_pre_flight_checklist.md` — Wf-259 ship surface (precedent for the v2 doc schema)
- `TODO/completed/321d_preflight_gate_audit.md` — Wf-321D gate-count audit (precedent for `test_gate_count_consistency`)

---

## §10 Sign-off

| # | Sign-off | Status |
|---|---|---|
| **10.1** | Script implemented per Wf-343c-I1 brief | ✅ |
| **10.2** | 8 tests implemented per Wf-343c-I1 brief | ✅ |
| **10.3** | pytest 8/8 PASS | ✅ |
| **10.4** | Doc written at `TODO/completed/343c1_pre_retrain_check.md` | ✅ |
| **10.5** | Honest framing catalog reviewed (§3.1, §3.2) | ✅ |
| **10.6** | Schema-version stamp (`343c1.v1`) recorded | ✅ |
| **10.7** | MEASURED-vs-PROJECTED tags (§8) reviewed | ✅ |
| **10.8** | Cross-references (§9) linked to parent doc + companion scripts | ✅ |

**Auto-fill verification log**: 8/10 gates verified; 2 N/A (post-retrain sign-off gates).

---

## §11 Honest framing disclaimer (party-twenty-spirit §3.5)

This script is **a tool to enforce honest framing**, not a guarantee.
Every retrain run MUST:

1. Trace every number back to JSON / log line / file
2. Report PROJECTED as PROJECTED, MEASURED as MEASURED
3. Acknowledge structural limitations (e.g. Os LOMO collapse per §1.2.1)
4. Cite partial work honestly (e.g. "4/12 Lambda gaps MEASURED, 3/12 honest-negative")
5. Never cite best-seed selection

**No fabrication. No paper-over. 0 容忍 不掩盖.**

---

*Created post-Wf-309..323 to ship the pre-retrain auto-verify surface.
Mirrors the Wf-259 pre_flight_check.py pattern (same GateResult shape,
same schema discipline, same honest-framing convention). Standing gate
document — run `wf343c1_pre_retrain_check.py --pipeline all` before any
retrain launch.*
