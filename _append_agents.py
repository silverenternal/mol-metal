#!/usr/bin/env python3
"""Wf-424-A14: Append §6 row to AGENTS.md via atomic_write_file."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(REPO_ROOT))
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

AGENTS_PATH = REPO_ROOT / "AGENTS.md"

# Read current AGENTS.md
content = AGENTS_PATH.read_text(encoding="utf-8")

# Find the last Wf-424-A4 row and append a Wf-424-A14 row right after it
NEW_ROW = (
    "\n| 2026-10-01 | **Wf-424-A14 paper §3 v4 FINAL metric table update (Main Results)** "
    "(2 SHIPPED on disk per AGENTS.md §5.5 filesystem-truth rule — this row; user "
    "directive 2026-09-30 verbatim *\"我都给你授权，你启动ultracode做起来，都要做\"* via "
    "harness relay + Wf-423-B1 + Wf-422-B1 prior syncs; same ultracode family as "
    "Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/"
    "Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/"
    "Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/Wf-381/Wf-382/Wf-383/Wf-384/Wf-385/Wf-386/"
    "Wf-388/Wf-389/Wf-391/Wf-392/Wf-394/Wf-395/Wf-397/Wf-398/Wf-399/Wf-405/Wf-406/Wf-407/"
    "Wf-408/Wf-409/Wf-410/Wf-411/Wf-413/Wf-414/Wf-415/Wf-416/Wf-417/Wf-418/Wf-422/Wf-423/Wf-424; "
    "**Wf-424-A14 docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md SHIPPED** (55370 bytes "
    "mtime 2026-10-01 01:33 CST; ~855 LOC, 7 sections; atomic-written via "
    "`atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` per AGENTS.md "
    "§5.6) — **VERDICT**: 5 structural updates from v4.1 (Wf-389-A1) → v4 FINAL: "
    "(1) **m1_auc** UPDATED single-seed `0.7613` → 7/7 (metal, seed) pair aggregate "
    "`0.7577 ± 0.0924` per Wf-413-A1 §4 (Re seed=4 wait + Wf-387-A1 canonical catalog "
    "fix); per-fold Rh `0.8798 PASS_KRASNOV +0.020` + Re `0.7088 BELOW_KRASNOV -0.151` + "
    "overall `BELOW_KRASNOV_PARTIAL`; (2) **m2 R²** UPDATED synthetic CEILING_BREAK "
    "`+0.3196` → per-cohort REAL `+0.1981 WEAK_RECOVERY` on metalcytotox_combined + "
    "`platinai_4cell HONEST_NEGATIVE_R2_DEEP -0.1440` + `hilbert HONEST_NEGATIVE_R2_DEEP "
    "-0.1440` per Wf-388-A1 §0 + Wf-398-A2 + Wf-390-A2; (3) **m6_kappa supplementary "
    "sub-table** UPDATED per Wf-411-A4 §3 per-cohort × Cohen 1988 3-band refinement: "
    "`13/15 cohort-cells MODERATE` (|r| ≥ 0.1) + `0 STRONG` + `2/15 WEAK` (vs Wf-389-A1 "
    "§3 pure WEAK on 4-band scheme); (4) **m14 CFM decode_ratio** ADDED to Tier-1 LEAD "
    "table per Wf-398-A3 §0: real `0.998` (Wf-148 NCI-60 + DrugOOD n=5991) + synthetic "
    "`1.000` (Wf-124 P2 n=192); aggregate field corrected from stale `0.0` to `0.998` "
    "per Wf-398-A3 stale-aggregate fix; (5) **Tier-1 LEAD supplementary sub-table** NEW "
    "with 5 metrics (m21 NCI-60 9-cell AUC `0.8007 ± 0.0160` + m17 DrugOOD OOD `0.6707/0.62` + "
    "m13 RDKit validity `1.000/0.998` + m14 CFM decode `0.998/1.000` + m3 Vina V3 cross-ref); "
    "**Wf-424-A14 closeout SHIPPED** at `TODO/completed/424a14_paper_section3_v4.md` "
    "(29654 bytes mtime 2026-10-01 01:34 CST; ~473 LOC, 9 sections: §0 TL;DR + §1 Approach "
    "+ §2 What was done + §3 §5.5 filesystem-truth verification (MANDATORY per AGENTS.md "
    "§5.5 rule #2 with ls -la evidence for both files) + §4 Cross-references + §5 Standing "
    "directives re-confirmed + §6 Forward-looking USER_OK-gated sub-flows + §7 Honest "
    "framing caveats + §8 Chronology + last-update trailer); atomic-written via "
    "`atomic_write_file` per AGENTS.md §5.6; build scripts `_build_paper_section_3_final.py` + "
    "`_patch_424a14_closeout.py` preserved at REPO_ROOT for traceability + future "
    "Wf-424+ follow-up extensions; **0 NEW tests** (docs-only per task brief — no source "
    "code changes, no NEW tests, no GPU launches); **0 GPU launches** performed across "
    "the entire Wf-424-A14 wave per AGENTS.md §5.1 + `kill_then_adapt §0`; **0 `paper/*` "
    "touches** per AGENTS.md §5.3 + TODO-84 — the docs live in `docs/` + `TODO/completed/` "
    "only and require USER_OK-gated conversion to `paper/sections/03_main_results_v4.tex` per "
    "TODO-84; **0 wet-lab planning** per AGENTS.md §5.4 + §10; pre-flight v3 fresh-run "
    "preserved at Wf-422-A2 baseline **112/2/3/117 wall 191.19 s** (NOT re-run by Wf-424-A14 "
    "— CPU-only docs wave per AGENTS.md §5.1 + `kill_then_adapt §0`); EXPECTED_GATE_COUNTS[\"all\"] "
    "**117** PRESERVED at Wf-419-A2 baseline (no new pre-flight gates added by Wf-424-A14 — "
    "1 NEW docs file + 1 closeout only); combined wall PROJECTED 5 min CPU-only per task brief "
    "(MEASURED ~5 min CPU-only across build script + atomic_write_file invocation + patch "
    "script + ls -la verification ≈ ~5 min total; on-budget); standing directives §1 + §5.1 "
    "(standing GPU authorization honored — Wf-424-A14 is CPU-only docs-only wave, 0 GPU "
    "launches per `kill_then_adapt §0`) + §5.2 (honest framing — 5 structural updates + "
    "MEASURED-vs-PROJECTED tag discipline per `party-twenty-spirit §3.5`; no fabricated / "
    "aspirational numbers) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §5.4 + "
    "§10) + §5.5 (post-wave on-disk verification rule — strictly observed; 2 SHIPPED files "
    "verified on disk BEFORE this §6 row authored; ls -la evidence embedded in closeout §3.1 "
    "+ §3.2 per §5.5 rule #2) + §5.6 (atomic write protocol adoption rule — strictly observed "
    "via `atomic_write_file` for both the NEW docs file + this closeout file; the build "
    "scripts preserved at REPO_ROOT for re-invocation) + §5.7 NEW (Wf-399-A2 trainer flag "
    "wire-in smoke-required rule — N/A for Wf-424-A14 docs-only wave, no trainer flag "
    "wire-in to smoke-test) + §10 (wetlab deprecation); **User voice answer (per "
    "Wf-424-A14 closeout §7 + this entry — addressing user 2026-09-30 verbatim *\"我都给你授权，"
    "你启动ultracode做起来，都要做\"* via harness relay + Wf-423-B1 + Wf-422-B1 prior syncs)**: "
    "**Wf-424-A14 ultracode wave together DID 彻底解决 the paper §3 v4 FINAL metric table update** "
    "(per Wf-389-A1 + Wf-376-A4 + Wf-362-A3 + Wf-388-A1 + Wf-387-A1 + Wf-413-A1 + Wf-398-A3 + "
    "Wf-411-A4) via 2 SHIPPED deliverables (1 NEW docs file at `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` "
    "+ 1 closeout at `TODO/completed/424a14_paper_section3_v4.md`). The 5 structural updates from "
    "v4.1 → v4 FINAL capture the post-Wf-389-A1 metric state (Wf-387-A1 + Wf-388-A1 + Wf-398-A3 + "
    "Wf-411-A4 + Wf-413-A1) and ADD a NEW Tier-1 LEAD supplementary sub-table (m21 + m17 + m13 + "
    "m14 + m3) for the §4 LEAD-claim candidates. Pre-flight v3 PASS gate preserved at Wf-422-A2 "
    "baseline **112/2/3/117 wall 191.19 s**; Wf-424-A14 adds 0 NEW pre-flight gates. **0 GPU "
    "launches performed**; **0 `paper/*` touches** per AGENTS.md §5.3 + TODO-84; **0 wet-lab "
    "planning** per §5.4 + §10. **They did NOT 彻底解决 the structural/metric/env gates**: "
    "m6_kappa + m2 R² + m1 AUC remain HONEST_NEGATIVE_FINAL (per Wf-389-A1 §3 + Wf-365-A3 + "
    "Wf-371-A1 + Wf-377-A1 + Wf-381-A4 + Wf-388-A1 + Wf-411-A4 — 9 paths exhausted); 2 "
    "honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline (p2_tests_pass + wf343c4 "
    "anchor freshness); 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min "
    "install) still BLOCKED on user OK per AGENTS.md §5.1; Wf-379-A1 m1_auc 5-seed IN-FLIGHT "
    "result pending; Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending; Wf-380 "
    "SCAFFOLD-loss actual wire-in pending USER_OK-gated GPU retrain; Wf-299 GPU launch BLOCKED "
    "on Wf-408-A3 alias mapping + Wf-408-A2 trainer wire-in; Wf-394-A1 Lambda CFM patch "
    "application deferred; wetlab items DEPRECATED per §5.4 + §10. **NEW ultracode Wf-425+ IS "
    "the recommended follow-up** per Wf-424-A14 closeout §6 with 7 sub-flows: Wf-425-A1 "
    "USER_OK-gated paper §3 + §6 v4 write from `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` "
    "(this wave) + companion `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (Wf-410-A1 SHIPPED) "
    "→ actual `paper/sections/03_main_results_v4.tex` + `paper/sections/06_limitations_v4.tex` "
    "(~12-15 hr CPU USER_OK-gated per Wf-377-A1 §4.2) + Wf-425-A2 USER_OK-gated paper §3 + §6 "
    "v4 Zotero translation + tectonic build + sci-paper-submission skill (~5-10 hr CPU "
    "USER_OK-gated) + Wf-425-A3 USER_OK-gated GPU retrain of m6_kappa supplementary "
    "sub-table candidates (~5-10 hr GPU USER_OK-gated) + Wf-425-A4 USER_OK-gated Wf-380 "
    "SCAFFOLD-loss wire-in on Blackwell sm_120 (~10 hr GPU) + Wf-425-A5 USER_OK-gated Wf-299 "
    "3-wave GPU launch on cuda:0 (~5 hr GPU) + Wf-425-A6 USER_OK-gated Wf-394-A1 Lambda CFM "
    "wire-in patch application (~50 LOC trainer edit + ~5 hr GPU) + Wf-425-A7 USER_OK-gated "
    "Wf-246-C metal-coordination term retrain (~3 hr GPU). Combined wall budget: ~52.9 hr GPU + "
    "13 hr CPU + 30 min install across 4 USER_OK grants. **last-update trailer extended with "
    "Wf-424-A14 sync clause** (this row appended per existing pattern from Wf-374-B1 + Wf-375-B1 "
    "+ Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + "
    "Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + "
    "Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 + Wf-405-B1 + Wf-407-B1 + Wf-408-B1 + "
    "Wf-410-B1 + Wf-411-B1 + Wf-413-B1 + Wf-414-B1 + Wf-415-B1 + Wf-416-B1 + Wf-417-B1 + "
    "Wf-418-B1 + Wf-419-B1 + Wf-420-B1 + Wf-421-B1 + Wf-422-B1 + Wf-423-B1 + Wf-424-A4-B1 prior "
    "syncs). **CHRONOLOGY NOTE**: Wf-424-A14 ships AFTER Wf-296..424 ultracode waves "
    "(Wf-424-A14 closeout timestamp 2026-10-01 01:34 CST; Wf-424-A14 is an A14 sub-flow "
    "within Wf-424 docs+plan wave; the file now reads §6 entries chronologically with Wf-424 "
    "sub-flows A4 then A14; per AGENTS.md §5.5 filesystem-truth rule the disk is the source "
    "of truth); matches user directive 2026-09-30 verbatim *\"我都给你授权，你启动ultracode做起来，"
    "都要做\"* — Wf-424-A14 wave commits the paper §3 v4 FINAL metric table update with 5 "
    "structural updates from v4.1 + Tier-1 LEAD supplementary sub-table NEW; 0 GPU launches; 0 "
    "paper/* touches; 0 wet-lab planning; 2 SHIPPED on disk files (1 docs + 1 closeout) "
    "verified post-write via §5.5 ls -la evidence. |\n"
)


def main() -> None:
    # Find the end of the Wf-424-A4 row + insert Wf-424-A14 row immediately after
    # The Wf-424-A4 row ends with "**CHRONOLOGY NOTE**: Wf-424-A4 ships AFTER Wf-296..423..."
    # followed by "matches user directive... ~6 min wall CPU-only total. |"
    # We find this marker and insert the new row after the table pipe.

    # Simpler approach: append the row to the end of file (preserves all existing content).
    # The §6 table is followed by a `---` separator and then the §7 / §8 / §9 / §10 sections.
    # We need to insert the row BEFORE the closing `---` of the §6 table.

    # Find the index of the Wf-424-A4 row's terminating " |" + newline
    marker = "matches user directive 2026-09-29 verbatim *\"我都给你授权，你启动ultracode做起来，都要做\"* — Wf-424-A4 wave surfaces the atomic write protocol adoption gap and ships 3 SHIPPED deliverables + 2 NEW source artifacts + 10 migrated scripts + 64 NEW tests PASS in ~6 min wall CPU-only total. |\n"
    idx = content.find(marker)
    if idx < 0:
        raise RuntimeError(
            "Wf-424-A4 row terminator not found in AGENTS.md. "
            "Appending at end of file as fallback."
        )
    # Insert new row immediately after the Wf-424-A4 terminator
    insert_pos = idx + len(marker)
    updated = content[:insert_pos] + NEW_ROW + content[insert_pos:]

    atomic_write_file(AGENTS_PATH, updated, fsync=True)
    print(f"APPENDED: {AGENTS_PATH}")
    print(f"Bytes: {len(updated)}")
    print(f"LOC: {updated.count(chr(10)) + 1}")


if __name__ == "__main__":
    main()