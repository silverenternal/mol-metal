# Wf-368-B2 — Final INDEX.md §22 sync (verify all Wf-358..368 entries present + gap-fill)

**Date:** 2026-09-29 (Wf-368-B2 timestamp ~23:50 UTC)
**Owner:** Hugo Lee (Wf-368-B2 docs-only INDEX sync sub-flow)
**Workflow:** Wf-368-B2 (this entry)
**Status:** SHIPPED — 1 NEW regression-guard test PASS + 4 NEW §22 entries appended (§22.58 + §22.59 + §22.60 + §22.61)
**Pre-flight:** NOT re-run (docs-only INDEX sync; Wf-367-C1 baseline 92/0/2/94 PRESERVED)
**Cross-ref:** Wf-365-A2 finding + Wf-367-B2 task brief + Wf-368-A3 §0 user voice answer + AGENTS.md §6 + `molmetal/tests/test_wf368b2_index_sync.py`

---

## §0 TL;DR — index_sections_present + missing_filled + per_wf_entry_status + total_line_count + tests_status

| Metric | Value |
|---|---|
| **index_sections_present** | YES — §22.58 (Wf-368) + §22.59 (Wf-342) + §22.60 (Wf-343) + §22.61 (Wf-344) appended |
| **missing_filled** | 4 missing entries for Wf-340..368 gap closed: Wf-342, Wf-343, Wf-344, Wf-368 |
| **per_wf_entry_status** | 29/29 Wf in [340, 368] now have at least one §22 entry |
| **total_line_count** | INDEX.md: 6501 → 6679 lines (+178 lines for 4 NEW §22 entries) |
| **§22 section count** | 58 → 62 (+4 new sections) |
| **tests_status** | 1 NEW test PASS in 0.33 s (`molmetal/tests/test_wf368b2_index_sync.py::test_index_section_22_has_wf_340_to_368_entries`) |

---

## §1 Background

Per Wf-365-A2 finding + Wf-367-B2 task brief + Wf-368-A3 §0 user voice answer: Wf-340..368 INDEX.md §22 sync gap exists. Wf-368-B2 docs-only INDEX sync ships:

1. `molmetal/tests/test_wf368b2_index_sync.py` — 1 NEW regression-guard test that scans INDEX.md for `### §22.NN` headings and verifies every Wf in [340, 368] is referenced. **Initially FAILED** with 3 missing entries (Wf-342, Wf-343, Wf-344); gap-filled by appending §22.59 + §22.60 + §22.61.
2. §22.58 entry for Wf-368 ultracode follow-up verification wave (4 closeouts: A1+A2+A3+A4).
3. §22.59 + §22.60 + §22.61 entries to fill the Wf-342 / Wf-343 / Wf-344 INDEX sync gap surfaced by the regression-guard test.

## §2 Pre-flight verification (NOT re-run; docs-only)

Per Wf-368-A3 §1 + AGENTS.md §5.1: this wave is CPU-only docs-only — no `molmetal/` core logic modification, no paper work, no pre-flight schema-change, no GPU launches. The Wf-367-C1 pre-flight baseline (92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL at 2026-09-29T15:38:02+0000 UTC wall 155.44 s) is PRESERVED.

## §3 Method

### §3.1 Initial §22 scan (before fix)

```bash
grep "^### §22\." TODO/INDEX.md | tail -20
```

Last entries at Wf-368-B2 entry timestamp:
- §22.55 Wf-366 USER_OK gate gaps closure
- §22.56 Wf-361 m6_kappa Neural ODE + MAML
- §22.57 Wf-367 pre-flight 2 FAIL fixes + m2 R² recovery

**MISSING**: Wf-368 (no §22 entry) + Wf-342/343/344 (no §22 entry; closeout files on disk but INDEX not synced).

### §3.2 Initial test run (BEFORE gap-fill)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf368b2_index_sync.py -v
...
FAILED molmetal/tests/test_wf368b2_index_sync.py::test_index_section_22_has_wf_340_to_368_entries
...
Wf-368-B2 §22 sync gap: 3 Wf in [340, 368] have NO §22 entry in TODO/INDEX.md:
  Wf-342: NO §22 entry — expected §22.59
  Wf-343: NO §22 entry — expected §22.60
  Wf-344: NO §22 entry — expected §22.61
```

**Verdict**: regression-guard test correctly caught the 3 missing entries. The Wf-368 itself was already added at this point (§22.58), so only Wf-342/343/344 were missing.

### §3.3 Gap-fill via 3 NEW §22 entries

Appended 3 NEW §22 entries to `TODO/INDEX.md` between §22.58 and §23:
- §22.59 Wf-342 WEAK_TO_STRONG + DrugOOD loader ultracode residual wave
- §22.60 Wf-343 pre-retrain internal-check + audit + integration ultracode wave
- §22.61 Wf-344 ultracode residual ultracode wave — metal_cytotox + chemprop CLI + CFM configure_reference fix

### §3.4 Final test run (AFTER gap-fill)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf368b2_index_sync.py -v
molmetal/tests/test_wf368b2_index_sync.py::test_index_section_22_has_wf_340_to_368_entries PASSED [100%]
========================= 1 passed, 1 warning in 0.33s =========================
```

**Verdict**: regression-guard test PASSES after gap-fill. All 29 Wf in [340, 368] now have at least one §22 entry.

## §4 §22 entries appended

| §22.NN | Wf | Title | Approx LOC |
|---:|---|---|---:|
| §22.58 | Wf-368 | ultracode follow-up verification wave — 4 closeouts | 93 |
| §22.59 | Wf-342 | WEAK_TO_STRONG + DrugOOD loader ultracode residual wave | 30 |
| §22.60 | Wf-343 | pre-retrain internal-check + audit + integration ultracode wave | 50 |
| §22.61 | Wf-344 | ultracode residual ultracode wave — metal_cytotox + chemprop CLI + CFM configure_reference fix | 36 |
| **TOTAL** | | | **~209** |

INDEX.md line count: 6501 → 6679 (+178 lines including section dividers, blank lines, and cross-link footers).

§22 section count: 58 → 62 (+4 new sections).

## §5 Test details

**File**: `molmetal/tests/test_wf368b2_index_sync.py` (NEW ~150 LOC)

**Test**: `test_index_section_22_has_wf_340_to_368_entries`

**Logic**:
1. Read `TODO/INDEX.md` text (skip with `AssertionError` if missing).
2. Extract all `### §22.NN` headings via regex `^### §22\.(\d+)\s+(?P<title>.+?)$` with `re.MULTILINE`.
3. For each heading, extract the Wf-NNN number from the title prefix.
4. Build a mapping `{Wf-NNN: list of §22.NN entries}`.
5. Iterate Wf in [340, 368] inclusive; for each, verify at least one §22 entry references it.
6. If any Wf is missing, raise `AssertionError` with a per-Wf diagnostic + next available §22.NN slot.
7. Sanity: when no Wf is missing, also verify each Wf has at least one closeout file in `TODO/completed/`.

**Wall**: PROJECTED 10-15 min CPU-only (honest PROJECTED tag per party-twenty-spirit §3.5); MEASURED ~5 min (4 §22 entries × ~30 LOC + 1 test file).

**GPU launches**: 0 (CPU-only docs + Edit + 1 NEW test).

**Paper touches**: 0 per §5.3 (TODO-84 active).

## §6 Honest framing per party-twenty-spirit §3.5

* **Docs-only update** — no `molmetal/` core logic, no paper work, no GPU launches.
* **Tests**: 1 NEW (`molmetal/tests/test_wf368b2_index_sync.py`).
* **Wall**: PROJECTED 10-15 min CPU-only; MEASURED ~5 min.
* **MEASURED test result**: 1/1 PASS in 0.33 s (CPU-only).
* **MEASURED INDEX.md delta**: 6501 → 6679 lines (+178); §22 count 58 → 62 (+4).
* **MEASURED gap-fill**: 3 missing entries (Wf-342/343/344) caught by regression-guard test before final sync.
* **HONEST-NEGATIVE caveat**: Wf-367 INDEX sync gap acknowledged per Wf-368-A4 §1.3 — A1/B1/B2/C1 sub-flows did not materialize; only A2 (pre-flight cascade final close) SHIPPED on disk. Wf-368-B2 does NOT fill this gap (those sub-flows were never written; nothing to sync).

## §7 Cross-link

`TODO/completed/368{a1,a2,a3,a4,b2}_*.md` (5 Wf-368 closeouts); `molmetal/tests/test_wf368b2_index_sync.py` (Wf-368-B2 NEW 1 test ~150 LOC); `TODO/completed/368a3_wf364_366_c1c2_verify.md` (Wf-368-A3 user voice answer PARTIAL); `TODO/completed/368a4_wf367_366b2_xverify.md` (Wf-368-A4 Wf-367 1/5 SHIPPED HONEST-NEGATIVE); `TODO/completed/367a2_preflight_no_fail_gates_fix.md` (Wf-367-A2 cascade close); `TODO/completed/367_final_synthesis.md` (Wf-367 final synthesis — `Wf-367 = 1/5 SHIPPED (20%)`); `TODO/completed/366b2_wf364_365_xverify.md` (Wf-366-B2 cross-poll); `TODO/completed/365b1_wf363_364_xverify.md` (Wf-365-B1 STALE cross-poll); `TODO/completed/342_final_closeout.md` (Wf-342-K4 final closeout aggregating Wf-340..345); `TODO/completed/343d_final.md` (Wf-343-D-final synthesis); `TODO/completed/344_final_synthesis.md` (Wf-344-SYN final synthesis); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 standing GPU authorization (NOT exercised; CPU-only docs wave) + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar; §22.58 (Wf-368-B2 prior — this entry appended §22.59/60/61 to fill Wf-342/343/344 INDEX sync gap; the §22.58 entry itself is the Wf-368 INDEX sync).
