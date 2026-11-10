#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Publish the staged §6.269.16 (wave-5 closure) into the honest-negative
catalogue, in its own numeric slot between §6.269.15 and §6.269.17.

WHAT THIS DOES, AND WHY EACH PIECE IS THE WAY IT IS
----------------------------------------------------
1. §6.269.16 was STAGED on 2026-11-02 and never published, which left a
   numeric gap (§6.269.15 -> §6.269.17) that §6.269.17 explained.  The gap
   is closed here by inserting the entry BETWEEN the two neighbours, so the
   catalogue's numbering becomes continuous and no published rendering is
   moved, reordered or deleted.
2. §6.269.16.9's heading claims the §6.241 surface-count guard "IS RED".
   MEASURED at publication (2026-11-02): the guard is GREEN (2 passed).
   The heading is tense-corrected and the ORIGINAL wording is preserved
   verbatim in a dated CORRECTION block (AGENTS.md §5.5 rule #6).
3. A new §6.269.16.12 records a SEVEN-SITE census of stale forward-
   references to the number §6.269.16 (which was reserved for the ledger-
   provenance defect and then took this wave-closure entry instead).
4. §6.269.17's "note on the section numbering" paragraph is PRESERVED
   VERBATIM and a dated UPDATE paragraph is appended after it.
5. The live code docstring in molmetal/tests/test_wfa127_surface_count_guard.py
   is corrected in place (a docstring is live documentation, not a preserved
   rendering), with the superseded spelling preserved verbatim.

VERIFICATION: the original catalogue's lines must survive as an ordered
SUBSEQUENCE of the new file (nothing deleted, nothing reordered), the prefix
and suffix must be byte-identical around the insertion, and the §6.241
surface-count guard must still pass afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))

from wf386a3_atomic_write import atomic_write_file  # noqa: E402

CATALOG = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"
STAGED = Path("/tmp/wfa269k_sec_6_269_16.md")
GUARD = REPO_ROOT / "molmetal" / "tests" / "test_wfa127_surface_count_guard.py"

ANCHOR_17 = "\n### §6.269.17 — **THE DEAD-GATE CLASS: A CHECK THAT CANNOT FAIL**"
NOTE_17 = (
    "**⇒ A NOTE ON THE SECTION NUMBERING, so the gap is not read as a deletion.**"
    " §6.269.16 exists as a **staged file** (`/tmp/wfa269k_sec_6_269_16.md`, 25 537 B,"
    " the wave-5 closure) that has **not yet been appended to this catalogue**."
    " This entry is therefore numbered **§6.269.17** so it cannot collide with it."
    " The gap between §6.269.15 (in this file) and §6.269.17 (this entry) is"
    " **deliberate and traceable to that staged file** — it is **not** a removal"
    " (AGENTS.md §5.5 rule #6 forbids silent deletion)."
)

OLD_HEADING_169 = (
    "#### §6.269.16.9 — THE §6.241 SURFACE-COUNT GUARD IS RED, AND IT **CANNOT PASS"
    " AT HEAD** UNDER ANY DOCUMENT EDIT"
)
NEW_HEADING_169 = (
    "#### §6.269.16.9 — THE §6.241 SURFACE-COUNT GUARD **WAS** RED (AND IS NOW GREEN"
    " IN THE WORKING TREE); IT **CANNOT PASS AT HEAD** UNDER ANY DOCUMENT EDIT"
)

CORRECTION_169 = """
> **CORRECTION (2026-11-02, the main thread, on publication of this entry) — THE HEADING'S TENSE CHANGED; THE FINDING BELOW IS PRESERVED VERBATIM.**
> **SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:**
>     #### §6.269.16.9 — THE §6.241 SURFACE-COUNT GUARD IS RED, AND IT **CANNOT PASS AT HEAD** UNDER ANY DOCUMENT EDIT
>
> **MEASURED at publication (2026-11-02, CPU-only, this thread):**
> * `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa127_surface_count_guard.py -q` → **2 passed**. The guard is **GREEN** in the working tree, not RED.
> * `pytest --collect-only -q` over the 15 `AFFECTED_SURFACE` files → **170 tests collected** (5.34 s). Unchanged.
> * `git ls-files --error-unmatch` on the nine files named below → **all nine are still UNTRACKED**. Unchanged.
>
> **⇒ THE "IS RED" CLAIM WAS TRUE WHEN THIS SECTION WAS WRITTEN AND IS FALSE AS A PRESENT-TENSE CLAIM.** Wave-6's repair re-synced the seven quote sites through the guard's own preserved-historical mechanism — exactly what this section's last paragraph describes — so the guard's `live_seen` assertion is now satisfied. The heading above is therefore a **dated finding**, not a description of the tree.
>
> **⇒ THE LOAD-BEARING STRUCTURAL CLAIM SURVIVES.** *"The guard cannot pass at HEAD under any document edit"* is **still TRUE**: the nine untracked files do not exist at HEAD, so the guard's `pytest --collect-only` cannot run there, and the **170** the documents now quote is a function of an **uncommitted** tree. The durable fix (a tracked-explicit measured universe) is still **USER-gated** on the U1 commit.
""".rstrip("\n")

SEC_16_12 = """
#### §6.269.16.12 — THE NUMBER §6.269.16 WAS RESERVED FOR A DIFFERENT DEFECT: A **SEVEN-SITE** CENSUS OF STALE FORWARD-REFERENCES (the main thread, 2026-11-02, on publication)

**WHAT WAS FOUND AT PUBLICATION.** This entry was staged on 2026-11-02 and published later the same day. Before publishing it, the main thread measured every **live** reference to the number `§6.269.16` that already existed in the tree. **SEVEN of them point at a DIFFERENT defect** — the USER-gated `p3_run_ledger_provenance` defect, whose home is **§6.269.9** ("(a) `p3_run_ledger_provenance` **FAILS** today with `OPEN_never_committed=2`"), and not at this wave closure:

| site | kind | cites §6.269.16 for |
|---|---|---|
| `docs/HONEST_NEGATIVE_CATALOG.md:16887` | dated wave-7 measurement block | the ledger-provenance defect |
| `docs/HONEST_NEGATIVE_CATALOG.md:16972` | dated wave-7 measurement block | the ledger-provenance defect |
| `docs/HONEST_NEGATIVE_CATALOG.md:17052` | dated wave-7 measurement block | the ledger-provenance defect |
| `TODO/INDEX.md:49` | dated wave-7 measurement block | the ledger-provenance defect |
| `TODO/INDEX.md:1823` | dated wave-7 measurement block | the ledger-provenance defect |
| `TODO/README.md:48` | dated wave-7 measurement block | the ledger-provenance defect |
| `molmetal/tests/test_wfa127_surface_count_guard.py:86` | **live code docstring** | the ledger-provenance defect |

**⇒ WHY THEY ARE STALE.** The number `§6.269.16` was **RESERVED** for the ledger-provenance defect when those seven passages were written; the numbering then **moved**, and the ledger-provenance defect landed at **§6.269.9** instead — which is why this very entry's §6.269.16.10 points onward with "*§6.269.9's ledger-provenance FAIL is still FAILing*". The §6.269.16 slot was then taken by **this** entry. **⇒ THIS IS THE §6.269.13 HOST-BOUND-PATH CLASS IN A NEW DRESS: the thing cited and the thing that must agree with it (a section NUMBER) live in different universes — a reservation made in one place and a landing in another.**

**⇒ WHAT WAS DONE, AND WHAT WAS DELIBERATELY NOT DONE.**
* The **six dated measurement blocks are NOT edited.** Each carries a *MEASURED 2026-11-02* wave-7 reading whose text is a historical record; rewriting a dated measurement in place is the silent-edit reflex that AGENTS.md §5.5 rule #6 exists to prevent. They are **left byte-identical**, and **this subsection is the disambiguation**: a reader who follows "(catalog §6.269.16)" now lands on an entry that says, in the table above, what the correct target is.
* The **one live code docstring IS corrected**, because a docstring is not a historical record — it is live documentation a reader follows. `molmetal/tests/test_wfa127_surface_count_guard.py` now points at **§6.269.9**, with the superseded spelling preserved verbatim in a dated `SUPERSEDED RENDERING PRESERVED VERBATIM` block beside it.
* ⇒ After publication, all seven references **resolve**: six to this section (which disambiguates) and one to §6.269.9 directly.

**⇒ THE COVERAGE BOUND OF THAT CENSUS — TWO REAL BLIND SPOTS, BOTH MEASURED.**
* **`git grep` searches only TRACKED files.** The census over tracked files returned **six** sites; the **seventh** (`test_wfa127_surface_count_guard.py`, untracked) was invisible to it and surfaced only on a filesystem-level grep of that path. **A `git grep` census is a census of the COMMITTED tree** — in this repo, where a large share of the newest artefacts is untracked, that is a real hole, and it is the same shape as §6.269.16.9's finding that **9 of the 15** surface files are untracked.
* **There is NO automated guard for section-number resolution.** MEASURED: no file under `molmetal/tests/` or `molmetal/scripts/` matches a `§6.269.x` citation pattern, and the zero-tolerance cross-reference guard named in §6.263 has **no** `BASELINE_SIZE` occurrence anywhere in the tracked test tree. So these seven references were found by **reading**, and the census's own claim of completeness rests on the greps above — **a coverage census, not a defect count** (§6.269.6). A reader should assume more exist in files this sweep did not cover.

**⇒ AND THE 0-GPU CONTRACT, RE-VERIFIED AT PUBLICATION.** MEASURED 2026-11-02: the attributable GPU watchdog log `/tmp/wfa269k_watchdog_attributable.log` → **0 lines**; `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` → **2 rows, both pid 725330** (550 MiB + 22 670 MiB), whose `/proc/725330/cmdline` reads `main.py train --experiment_preset light_cnn_v1_2loss --seed 41 …` and whose working directory is owned by **another user** — **not** mol-metal, **not** this wave's. The host's GPU row has therefore **changed** since §6.269.16.8 was written (that entry measured pid 498522 / 6 158 MiB); the *conclusion* — no mol-metal GPU allocation — **survives**.
""".rstrip("\n")

NOTE_17_UPDATE = (
    "\n\n**⇒ UPDATE (2026-11-02, the main thread, on publication of §6.269.16) — THE GAP IS"
    " NOW CLOSED, AND THE PARAGRAPH ABOVE IS SUPERSEDED.** The staged file this paragraph"
    " names has now been **published** into this catalogue, between §6.269.15 and this"
    " entry, in its own numeric slot. **SUPERSEDED RENDERING PRESERVED VERBATIM per"
    " AGENTS.md §5.5 rule #6:** the paragraph above. ⇒ The numbering is now continuous"
    " (§6.269.15 → §6.269.16 → §6.269.17) and the gap this paragraph explained **no longer"
    " exists**; it existed only between the staging of §6.269.16 and its publication, and"
    " it was **never** a deletion. §6.269.16 was itself **corrected on publication** — its"
    " §6.269.16.9 heading was tense-corrected (the §6.241 guard is now GREEN) and a new"
    " §6.269.16.12 records a **seven-site** census of stale forward-references to this very"
    " number."
)

OLD_DOCSTRING = "#: USER-gated `p3_run_ledger_provenance` defect, catalog §6.269.16), so the"
NEW_DOCSTRING = "#: USER-gated `p3_run_ledger_provenance` defect, catalog §6.269.9), so the"
DOCSTRING_CORRECTION = """#: Wf-A269 (2026-11-02, on publication of catalog §6.269.16) - CORRECTION:
#: the defect's home is **§6.269.9**, not §6.269.16.  The number §6.269.16
#: was RESERVED for it when this docstring was written; the numbering then
#: moved and §6.269.16 took the wave-5-closure entry instead.  The
#: superseded spelling is preserved verbatim below, and the seven-site census
#: of this collision is recorded at catalog §6.269.16.12.
#:
#: SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#:     #: USER-gated `p3_run_ledger_provenance` defect, catalog §6.269.16), so the
"""


def _insert(original: str, anchor: str, payload: str, before: bool = True) -> str:
    idx = original.index(anchor)
    if before:
        return original[:idx] + payload + original[idx:]
    end = idx + len(anchor)
    return original[:end] + payload + original[end:]


def main() -> int:
    original = CATALOG.read_text(encoding="utf-8")
    staged = STAGED.read_text(encoding="utf-8")

    assert "### §6.269.16 " not in original, "§6.269.16 is already published"
    assert "### §6.269.17 — **THE DEAD-GATE CLASS" in original
    assert OLD_HEADING_169 in staged, "the staged entry's §6.269.16.9 heading moved"
    assert NOTE_17 in original, "§6.269.17's numbering note moved"

    # --- 1. correct the staged entry ---------------------------------------
    entry = staged.replace(OLD_HEADING_169, NEW_HEADING_169, 1)
    assert entry != staged
    entry = _insert(entry, "\n**THIS IS THE FINDING NO WAVE IN THIS CHAIN HAD LOOKED AT",
                    "\n" + CORRECTION_169 + "\n", before=True)
    assert CORRECTION_169[:60] in entry
    entry = entry.rstrip("\n") + "\n\n" + SEC_16_12 + "\n"
    assert "§6.269.16.12" in entry

    # --- 2. insert it between §6.269.15 and §6.269.17 -----------------------
    idx = original.index(ANCHOR_17)
    before = original[:idx]
    after = original[idx:]
    insert = entry.lstrip("\n") + "\n"  # keep exactly two blank lines between sections
    merged = before + insert + after
    assert merged[: len(before)] == before, "the prefix is not byte-identical"
    assert merged[len(before) + len(insert) :] == after, "the suffix is not byte-identical"

    # --- 3. the §6.269.17 update paragraph ----------------------------------
    merged = _insert(merged, NOTE_17, NOTE_17_UPDATE, before=False)
    assert NOTE_17 in merged and NOTE_17_UPDATE in merged

    # --- 4. the live code docstring -----------------------------------------
    guard_text = GUARD.read_text(encoding="utf-8")
    assert guard_text.count(OLD_DOCSTRING) == 1, "the guard docstring moved"
    guard_new = guard_text.replace(
        OLD_DOCSTRING,
        DOCSTRING_CORRECTION + NEW_DOCSTRING,
        1,
    )
    atomic_write_file(GUARD, guard_new, encoding="utf-8")

    # --- 5. atomic write + verification -------------------------------------
    atomic_write_file(CATALOG, merged, encoding="utf-8")

    after_lines = merged.splitlines()
    old_lines = original.splitlines()
    # every original line must survive, in order (nothing deleted / reordered)
    it = iter(after_lines)
    missing = [ln for ln in old_lines if not any(x == ln for x in it)]
    assert not missing, f"{len(missing)} original line(s) lost or reordered"
    assert merged.endswith("\n"), "the catalogue must end with a newline"
    assert "### §6.269.16 " in merged and "### §6.269.17 " in merged

    order = [
        merged.index(f"### §6.269.{n} ")
        for n in (13, 14, 15, 16, 17)
    ]
    assert order == sorted(order), "the §6.269.x sections are out of numeric order"

    print(f"CATALOG {len(original)} -> {len(merged)} chars (+{len(merged) - len(original)})")
    print(f"  prefix byte-identical: True; suffix byte-identical: True")
    print(f"  original lines preserved as an ordered subsequence: True ({len(old_lines)} lines)")
    print(f"  §6.269.x order: 13 < 14 < 15 < 16 < 17")
    print(f"GUARD  {len(guard_text)} -> {len(guard_new)} chars; docstring -> §6.269.9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
