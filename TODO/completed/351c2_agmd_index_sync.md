# Wf-351-C2: AGENTS.md + TODO/INDEX.md docs sync for Wf-351 SCI submission ship surface

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Wall:** PROJECTED 5-10 min CPU-only; actual ≈ 5 min (docs-only, 1 Edit on INDEX.md)
**Wall:** docs-only — zero code changes.

---

## §1 Scope (per task brief)

Per the Wf-351-C2 task brief from the orchestrator:

1. ✅ Append `mol-metal-sci-paper-submission` to `AGENTS.md §3.2` (Domain / surface skills)
2. ✅ Append §22 entry to `TODO/INDEX.md` documenting Wf-351 ship wave
3. ✅ Verify edits
4. ✅ Document this closeout

---

## §2 AGENTS.md §3 update status

The `mol-metal-sci-paper-submission` row was already added to `AGENTS.md §3.2` during
**Wf-351-B3** (the skill-creation sub-flow). Wf-351-C1 §3 verified the registration. Wf-351-C2
re-confirms:

```
$ grep -n "mol-metal-sci-paper-submission" AGENTS.md
63:## §3 Skill catalog (29 skills in `.claude/skills/` — 20 baseline + 8 NEW per Wf-321B + 1 NEW per Wf-351-B3; all re-validated per Wf-329-W8 + Wf-351-B3)
96:| `mol-metal-sci-paper-submission` | SCI paper submission workflow — end-to-end build → validate → submit → archive using audited open-source tools (sn-jnl + tectonic + Zotero + matplotlib/scienceplots). READ-only on `paper/*` per TODO-84 + §5.3. | "submit paper", "SCI submission", "投稿", "cover letter", "ScholarOne", "Editorial Manager", "Mol. Divers.", "JCAMD", "J. Mol. Graph. Model.", "build PDF", "tectonic", "Zotero export", "Better BibTeX", "CSL style" |
251:| 2026-09-29 | ...**Wf-351-B3 SCI submission skill** (NEW per user directive 2026-09-29 "你也别闲着，现在配置一下那种发sci要用的工作流和skill吧，直接找开源的，不用自己写（记得审查）" — end-to-end open-source submission pipeline: sn-jnl CC-BY-4.0 + tectonic MIT + Zotero AGPL-3.0 + Better BibTeX MIT + CSL BSD + matplotlib PSF + scienceplots MIT, all audited; SKILL.md 172 LOC at `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` with §0 audit table + §1 4-stage build→validate→submit→archive + §2 cross-refs + §3 standing caveats + §4 audit trail; 7 NEW tests PASS in 0.12s at `molmetal/tests/test_wf351b3_sci_skill.py`; READ-only on `paper/*` per TODO-84 + §5.3; ...
```

**Verdict:** 3 matches found at lines 63 (§3 header), 96 (§3.2 row), 251 (§6 recent Wf history pointer). All three are correctly in place from Wf-351-B3 + Wf-351-C1 prior work.

### §2.1 Skill catalog count delta

| Stage | Skill count | Source |
|---|---:|---|
| Pre-Wf-351 | 28 | `ls .claude/skills/ \| wc -l = 28` per Wf-336-D2 / Wf-341-F3 baseline |
| Post-Wf-351-B3 | **29** | +1 NEW per Wf-351-B3 = `mol-metal-sci-paper-submission` |
| Post-Wf-351-C2 | 29 (UNCHANGED) | docs-only sync, no NEW skill created |

---

## §3 TODO/INDEX.md §22 update

**Section-number note (honest framing per `party-twenty-spirit` §3.5)**: The Wf-351-C2 task
brief requested §22.34 for this INDEX-sync entry, but §22.34 was already populated by
**Wf-341-F2** (Wf-341 ultracode residual closeout wave — comprehensive 13-sub-flow detailed
table, 2026-09-29 Wf-341-F2 INDEX sync). To preserve sequential ordering and avoid
section-number collision (same convention used in §22.34 itself + §22.35), this entry is
appended as **§22.36** (the next available §22.* slot after §22.35). The TASK-BRIEF
request for §22.34 is **NOT** honoured because §22.34 was already taken at the Wf-351-C2
timestamp; no EDIT performed on §22.34 to insert new content.

### §3.1 Per-file edit inventory

| # | Path | Edit | LOC delta |
|---:|---|---|---:|
| 1 | `TODO/INDEX.md` | Append new §22.36 entry (Wf-351 SCI submission ship wave) | +~85 LOC |
| 2 | `TODO/completed/351c2_agmd_index_sync.md` | NEW (this file) | +~140 LOC |

### §3.2 §22.36 entry summary

Appended at the end of the §22 block (between §22.35 and §23):

- **Subject**: Wf-351 SCI paper submission workflow + skill setup (2026-09-29)
- **Per Wf-351 ship wave**: 4 audits (A1 LaTeX + A2 reference manager + A3 LaTeX engine + A4
  figure style) + 3 setups (B2 Zotero + B3 skill + C1 setup verify) + 1 docs sync (C2 this entry)
- **Tools adopted (audited open-source)**: sn-jnl CC-BY-4.0 + tectonic MIT + Zotero AGPL-3.0 +
  Better BibTeX MIT + CSL BSD + matplotlib PSF + scienceplots MIT (per Wf-351-B3 SKILL.md §0
  audit table)
- **Skill**: `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` (172 LOC, 8 sections,
  YAML frontmatter) — registered at AGENTS.md §3.2 row 96 + §3 header line 63
- **Tests**: 9 NEW = 9/9 PASS in 0.83 s wall (2 in `test_wf351b2_sci_paper.py` + 7 in
  `test_wf351b3_sci_skill.py`); **0 NEW** from C2 (docs-only sync)
- **Wall**: PROJECTED 5-10 min CPU-only (per task brief); actual ≈ 5 min for C2
- **Pre-flight delta**: 0 NEW gates added; EXPECTED_GATE_COUNTS["all"] 62 PRESERVED at
  Wf-335-SYN baseline (per Wf-351-B3 §0; Wf-351-C2 does NOT re-run pre-flight — docs-only)

---

## §4 Honest framing per `party-twenty-spirit` §3.5

**MEASURED today (2026-09-29, this Wf-351-C2)**:
- `AGENTS.md §3.2` row for `mol-metal-sci-paper-submission` is correctly in place at line 96
  (added by Wf-351-B3, verified by Wf-351-C1, re-confirmed by Wf-351-C2)
- `AGENTS.md §3` header line 63 says "29 skills ... 1 NEW per Wf-351-B3" — correct count
- `AGENTS.md §6` recent Wf history includes Wf-351-B3 entry at line 251
- `TODO/INDEX.md §22.36` entry appended documenting Wf-351 ship wave (next available slot
  after §22.34 + §22.35 already taken)
- 0 NEW tests added (Wf-351-C2 is docs-only sync per task brief honest framing)
- 0 GPU launches performed (CPU-only docs sync)
- 0 NEW pre-flight gates added (`EXPECTED_GATE_COUNTS["all"] 62 PRESERVED` per Wf-335-SYN baseline)

**Honest-negative caveats preserved**:
1. **§22.34 slot conflict**: TASK-BRIEF requested §22.34 but the slot was already taken by
   Wf-341-F2 (comprehensive 13-sub-flow detailed table). Per the established convention from
   §22.34 + §22.35 notes, this entry is appended as §22.36 instead. No EDIT performed on §22.34
   to insert new content.
2. **LaTeX stage deferred**: `.tools/latex/sn-jnl.cls` + `test_minimal.tex` MISSING per
   Wf-351-C1 §1; deferred to Wf-352 (`.tex` integration needs user OK per TODO-84 + §5.3).
3. **BibTeX style swap deferred**: `unsrtnat` → `splncs04.bst` / `spbasic.bst` swap deferred
   to Wf-352 (per 351a1 §6 + 351b2 §6).
4. **Pandoc + Vale prose-lint deferred**: heavier install (~300 MB for Pandoc); deferred to
   Wf-352+.

---

## §5 Standing-directive invariants

- AGENTS.md §1.1 MEASURED-vs-PROJECTED tag discipline — RESPECTED (this closeout uses
  `MEASURED today` block + `PROJECTED` annotations per the standing rule)
- AGENTS.md §5.1 standing GPU authorization — N/A (Wf-351-C2 is docs-only, zero GPU launches)
- AGENTS.md §5.2 honest framing — RESPECTED (party-twenty-spirit §3.5 MEASURED vs PROJECTED
  discipline preserved; §22.34 slot conflict declared honestly)
- AGENTS.md §5.3 no paper work without authorization — RESPECTED (zero touches to `paper/*`;
  LaTeX stage deferred to Wf-352 per TODO-84)
- AGENTS.md §5.4 Tier-2 journal target — RESPECTED (Mol. Divers. CSL style configured; full
  submission workflow deferred to Wf-352 per §5.4 Tier-2 acceptance bar)

---

## §6 Cross-references

- `TODO/completed/351a1_latex_template_audit.md` — A1 LaTeX template audit (defer `.cls` to Wf-352)
- `TODO/completed/351a2_reference_manager_audit.md` — A2 Zotero + Better BibTeX + JabRef + papis audit
- `TODO/completed/351a3_latex_engine_audit.md` — A3 tectonic + latexmk build engine audit
- `TODO/completed/351a4_figure_style_audit.md` — A4 matplotlib + scienceplots figure-style audit
- `TODO/completed/351b2_zotero_setup.md` — B2 Zotero + Better BibTeX + CSL + pybtex actual setup
- `TODO/completed/351b3_sci_skill.md` — B3 skill wiring + tests + AGENTS.md sync (the §3.2 row)
- `TODO/completed/351c1_setup_verify.md` — C1 setup verification (re-uses Wf-351-B3 §3 verdict)
- `TODO/completed/351c2_agmd_index_sync.md` — **C2 this closeout** (docs-only sync, §22.36 entry appended)
- `.claude/skills/mol-metal-sci-paper-submission/SKILL.md` — the new skill itself (172 LOC, 8 sections, YAML frontmatter)
- `AGENTS.md §3` header line 63 — "29 skills ... 1 NEW per Wf-351-B3"
- `AGENTS.md §3.2` row 96 — `mol-metal-sci-paper-submission` entry
- `AGENTS.md §6` line 251 — Wf-351-B3 entry in 2026-09-29 ship wave row
- `TODO/INDEX.md §22.36` — new entry appended by Wf-351-C2 (this closeout's primary INDEX deliverable)
- `TODO/INDEX.md §22.34` — pre-existing Wf-341-F2 entry (slot taken; C2 uses §22.36 per convention)
- `TODO/INDEX.md §22.35` — pre-existing Wf-349-B2 entry (slot taken; C2 uses §22.36 per convention)
- `molmetal/tests/test_wf351b2_sci_paper.py` — 2 NEW tests (B2)
- `molmetal/tests/test_wf351b3_sci_skill.py` — 7 NEW tests (B3)
- AGENTS.md §5.1 standing GPU authorization + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar.

---

**Wall:** ≈ 5 min CPU-only (within PROJECTED 5-10 min band per task brief).
**Ship verdict:** Wf-351-C2 SHIPPED — `AGENTS.md §3.2` already correctly synced from Wf-351-B3
(verified by Wf-351-C1 + Wf-351-C2); `TODO/INDEX.md §22.36` appended documenting Wf-351 wave;
LaTeX stage + BibTeX style swap + Pandoc/Vale prose-lint deferred to Wf-352 per TODO-84 + §5.3.
