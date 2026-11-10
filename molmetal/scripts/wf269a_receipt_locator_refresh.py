#!/usr/bin/env python3
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A269A — the PRODUCER for `docs/wf_a44_verify_receipts.md`'s locator table.

AGENTS.md §5.5 rule #6 + the §6.269.4h finding: the Wf-A44 verify-receipt is a
**static prose table with no producer and no gate**, so its line-numbers rotted
silently (9 of its 13 grep-verified rows point at dead lines) and nothing in the
repo noticed.  This module is that missing producer.

THE DESIGN RULE (this is the operational form of §6.269.4h's generalised rule):

    **A locator must be a RESOLVER (a symbol that resolves to a current line),
    not a FROZEN LINE NUMBER.**

A frozen line number is a textual probe: it cannot distinguish "the symbol was
renamed" from "the symbol was deleted", and it breaks on every unrelated edit
above it.  A symbol probe answers the question a human following the receipt
actually has ("where is this thing NOW?") and stays true across line drift.

⇒ The sidecar (`docs/wf_a44_verify_receipts.locators.json`) stores the SYMBOL;
  the guard (`molmetal/tests/test_wfa269_receipt_locator_guard.py`) resolves it
  against the tree and FAILS if a claim no longer holds.  The receipt's own
  prose keeps the historical line-numbers (preserved VERBATIM per §5.5 rule #6)
  and gains, in its dated re-verification section, the resolved line AS OF the
  re-verification date — explicitly labelled as a snapshot, with the symbol
  named as the durable locator.

================================================================================
DESIGN v4 — SEPARATE THE CLAIM FROM THE RESOLUTION
================================================================================

Everything the sidecar stores falls into exactly TWO classes, and the class
decides whether a mutation may FAIL the guard:

  * **ASSERTED** (`CLAIM_FIELDS`) — the DURABLE claims: does the symbol still
    exist, is the count still above its floor, is the verdict still right, and
    does the receipt's own wording still hold.  A mutation to any of these
    MUST fail the guard.  They are compared against a FRESH re-resolution of
    the tree, never against a stored copy of themselves.
  * **RESOLUTION** (`RESOLUTION_FIELDS`) — the DERIVED measurements: resolved
    line numbers, hit counts, the file list, byte offsets, content hashes.
    These are **informational drift only** and are NEVER compared for
    equality.  A line number moves on every unrelated edit above it; a guard
    that fails on that is a treadmill, not a guard.

⇒ **A guard may FAIL only on a DURABLE claim.**  Every derived field — line
  numbers, byte offsets, content hashes — is REPORTED as informational drift,
  never ASSERTED.

⇒ **REFRESHING IS A SEPARATE, NON-BLOCKING REGENERATION STEP.**  Re-running
  this producer rewrites the resolution fields and the timestamp.  It never
  decides whether a claim holds: the consumer does.  (That is the same
  producer/consumer split as AGENTS.md §4's stale-flag rule — see below.)

⇒ **AGENTS.md §4's STALE-FLAG RULE APPLIES.**  The PRODUCER stamps
  (`resolved_at`); the CONSUMER decides.  A stamp older than 24 h means FAIL,
  and a missing, unparseable or future-dated stamp fails CLOSED.  The sidecar's
  own raw BYTE SIZE is NOT a durable locator for the same reason (it changes
  when only the timestamp changes) — `rows_sha256` is: it hashes the rows
  array ALONE, with the volatile envelope excluded.

FLOORS, NOT VACUOUS THRESHOLDS
------------------------------
Every `count_ge` probe declares an explicit `floor`.  A floor is a LOWER BOUND
the live count must not fall below — so the tree may GROW freely without
failing anything.  A *freeze* (`live == stored`) would turn every legitimate
addition into a failure; that is the treadmill this design removes.  The
previous producer defaulted the threshold to `1`, which is VACUOUS (one hit
passes) — the floor is now mandatory and explicit for every `count_ge` probe.

THE THREE-PART PREDICATE (the only place `count` and `files` are used, and
never as an equality)
-----------------------------------------------------------------------------
For every `count_ge` row the verdict is a THREE-PART conjunction:

    (a) live_count >= floor                            — detects SHRINKAGE
    (b) len(sidecar.files) >= floor                    — detects TAMPERING
    (c) every sidecar-named path still carries the marker — detects RELOCATION

`live_files == sidecar.files` is explicitly NOT compared: that is the
treadmill.  (c) is a SUBSET check, so growth is free; only a path that LOST
the marker fails.

THE `said` PROBE — THE PRODUCER'S OWN BLIND SPOT
-------------------------------------------------
`row["said"]` copies the receipt's claim VERBATIM, so the sidecar could never
detect a stale (or false) `said`: three receipt claims are FALSE today and the
old sidecar said nothing about it.  So the producer RE-MEASURES the claim and
stores the verdict in `row["said_ok"]` (ASSERTED).  The forms, in precedence
order:

  1. **LINE** — the claim names `L<n>` (or `L<a>-<b>`, or `L<n> (<a>-<b>)`).
     `said_lines` = the measured line list; `said_ok` = the measured set
     intersects the claimed line / range.
  2. **HITS** — the claim names `N hits`.  `said_count` = the measured count
     of the probe TOKEN (for an identifier-like symbol the identifier, so a
     receipt that counted "def + 5 call sites" is compared against the
     identifier's occurrences, not against the `def` line alone);
     `said_ok` = (measured >= claimed).
  3. **TESTS** — the claim names `N tests`.  `said_count` = the measured
     number of `def test_` functions in the file; `said_ok` = (measured >=
     claimed).
  4. **BYTES** — the claim names `N bytes`.  `said_count` = the measured file
     size; `said_ok` = (measured >= claimed).
  5. **PRESENCE** — the claim names the row's own artefact (its symbol, or the
     path it cited).  `said_ok` = whether that artefact is PRESENT.  This is
     what catches the receipt rows that were FALSE AT BIRTH (a command that
     cannot have exited 0, a path that never existed).
  6. Otherwise `said_ok = None` — there is no machine-checkable claim.

COVERAGE BOUND (stated explicitly, per the repo's own convention that a guard's
coverage is not a defect count):

  * COVERED — the receipt's 15 MEDIUM rows (M1..M15), §6.71a, §6.71b, §6.71f,
    §6.71e (J1 fix + docstring), the 13 trainer-flag flips, and the two §6.71d
    yaml counts.  That is 40 probes over 12 files.
  * NOT COVERED — the 19 JUDGEMENT-CALL ledger rows (they live in the
    catalogue's prose, not in a file locator), and the receipt's pre-flight
    claim (`110 PASS / 3 FAIL / 10 SKIP / 124 total`), which is a wave-scoped
    snapshot of a live tool and is not a locator.
  * The guard does NOT run pytest.  It pins the ARTEFACTS (the test files and
    the named test functions exist); the repo's own test run executes them.

================================================================================
DESIGN v5 (2026-11-07) — THREE FIELD CLASSES, A CLAIM-SIDE PROBE KIND, AND A
FLOOR THAT CANNOT BE VACUOUS
================================================================================

DESIGN v4 (above) split every stored field into TWO classes.  v5 KEEPS that
split and ADDS the third class the v4 wording could not express, because ONE
field belongs to different classes on different ROWS.

  * **ASSERTED** — the DURABLE claims.  A mutation to any of these MUST fail
    the guard: `id`, `file`, `expect`, `symbol`, `said`, `said_kind`, `floor`,
    `ok`, `section`.
  * **ASSERTED-VERDICT** — the guard RE-MEASURES the live tree and asserts that
    the RECORDED verdict EQUALS the LIVE verdict.  It does NOT assert that the
    verdict is TRUE (a retired flag's ABSENCE is a TRUE claim).
      - `ok` — on every row.
      - `said_ok` — on every row EXCEPT a row whose `said_kind` is `line`, and
        EXCEPT a row whose `said_ok` is `None`.
  * **RESOLUTION** — informational drift only, NEVER compared for equality:
    `count`, `lines`, `files`, `exists`, `note`, `said_claim`, `said_count`,
    `said_lines`, AND the `said_ok` of a `said_kind == 'line'` row.

⇒ **A GUARD MAY FAIL ONLY ON A DURABLE CLAIM.**  Every derived field — line
  numbers, byte offsets, content hashes — is REPORTED as informational drift,
  never ASSERTED.

WHY `said_ok` IS ROW-DEPENDENT.  A `line` claim is a LOCATOR: it moves on
every unrelated edit above the symbol, so asserting its equality would make
the guard a treadmill.  A `hits` / `tests` / `bytes` / `presence` claim is a
DURABLE statement about the artefact ("the identifier occurs at least N
times", "the flag is still there"), so asserting its equality is meaningful.

⇒ **THE ROW'S OWN `said_kind` FIELD IS THE SELECTOR.**  The guard reads
  `row["said_kind"]` to decide which assertion applies to that row, and it
  also asserts `row["said_kind"] == row["said_claim"]["kind"]` — the two are
  set from ONE place, so a drift between them is itself a defect.

FLOORS THAT CANNOT BE VACUOUS (2026-11-07)
-------------------------------------------
Every `count_ge` probe MUST declare an explicit positive integer `floor`, and
the floor is compared in BOTH branches (the directory branch and the file
branch).  A `count_ge` spec WITHOUT a floor now RAISES instead of falling back
to a threshold that passes trivially — the previous default (`1`) was VACUOUS
(one hit passed, so the check could not fail).  A floor BELOW the measured
live count is also vacuous and is not acceptable.

MEASURED 2026-11-07: the population of `count_ge` probes is **2** —
`S671d-scaffold` (floor **29**) and `S671d-legacy` (floor **16**) — and BOTH
floors EQUAL the measured live TRACKED count, so neither is vacuous.  The
floors are a REVIEWER-RATIFIED list: the two numbers are MEASURED, the DECISION
that they are the right floors belongs to the reviewer.

THE `said` PROBE'S OWN BLIND SPOT — THE `hits` VERDICT IS A LOWER BOUND
-----------------------------------------------------------------------
The `hits` verdict is `measured >= claimed`: a LOWER BOUND.  It therefore
CANNOT certify a claim that carries a BREAKDOWN.  MEASURED 2026-11-07 on the
receipt's M2 row — the receipt's prose says
`grep -n "_has_metal" molmetal/molmetal_lam/sbdd_env/metrics_v2.py`
"returns 6 hits (line 110 def + 5 call sites)": the live grep returns **7**
lines (L110 def, L141 — a COMMENT that mentions the identifier — and the five
call sites L220/285/364/513/572).  The verdict `7 >= 6` is TRUE and
`said_ok` is True, YET the receipt's stated COUNT (6) is not what the command
prints (7), and the verdict says NOTHING about the breakdown.
⇒ **A claim that is FALSE on its breakdown can still yield `said_ok = True`.**
  This is a KNOWN LIMIT of the probe, stated here and NOT repaired: repairing
  it means teaching the probe to read an English breakdown, which is a
  different (and far more fragile) contract.

DATED CORRECTION (2026-11-07) — "three receipt claims are FALSE today"
---------------------------------------------------------------------
The sentence in the v4 section above is PRESERVED VERBATIM (AGENTS.md §5.5
rule #6), but it names a POPULATION that is neither the 29 nor the 4.  Three
populations, one label (the §6.269.40 class-6 costume).  All three numbers
below are MEASURED on 2026-11-07.

  * **3** — the receipt's own FALSE claims among its 13 grep-verified rows:
    that is the receipt's own R0 tally, cross-checked against the live tree.
    **M3** (`--self-test` — 0 occurrences in the file; the command exits 2,
    not 0), **M9** (`consistency_term_weight` — 0 hits today, retired with a
    published retraction), **M10** (`molecular_property_target_weight` —
    0 hits today).  ⇒ THE DESIGN v5 BRIEF NAMES THIS POPULATION
    "(M1, M2, M5)"; ON DISK THAT IS A DIFFERENT SET: M1 and M5 are
    STALE-LOCATOR rows (the receipt's own class "stale locator, claim still
    TRUE", which has EIGHT members: M1, M2, M5, M6, M8×3, M12, M13, M14) and
    M2's probe verdict is TRUE (the `hits` lower-bound blind spot above), so
    {M1, M2, M5} is a set of three STALE LOCATORS, not of FALSE claims.  The
    disk's reading is the one this producer asserts; the brief's reading is
    recorded here so that NEITHER is lost.
  * **29** — the sidecar ROWS whose PROBE VERDICT is False: **25** of kind
    `line` (locator drift — informational under v5) + **3** of kind `presence`
    (M3a, M15c, M15d) + **1** of kind `hits` (S671d-legacy).
  * **4** — the DURABLE-KIND false claims, the only ones a guard may fail on:
    **M3a**, **M15c**, **M15d** and **S671d-legacy**.

CLI:
    python molmetal/scripts/wf269a_receipt_locator_refresh.py            # check
    python molmetal/scripts/wf269a_receipt_locator_refresh.py --write    # refresh
    python molmetal/scripts/wf269a_receipt_locator_refresh.py --json

Read-only unless `--write` is passed.  CPU-only.  No GPU.  No network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SIDECAR = REPO / "docs" / "wf_a44_verify_receipts.locators.json"
RECEIPT = REPO / "docs" / "wf_a44_verify_receipts.md"

TRAINER = "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
METRICS_V2 = "molmetal/molmetal_lam/sbdd_env/metrics_v2.py"
STEREO = "molmetal/eval/scoring/stereo_postprocess.py"

# --------------------------------------------------------------------------- #
# DESIGN v4 — THE TWO FIELD CLASSES.
#
# EVERY field a row stores is in exactly one of these tuples, and
# `build_sidecar()` asserts that invariant on every run.  A hand-written
# SUBSET of the fields is itself the defect class (the repo's own
# §6.269.35.3: the sidecar stored 12 fields and the gate compared 3, then 12
# minus 9 more, and every added field silently escaped).  Two disjoint
# tuples that COVER the row shape cannot fall behind it.
#
# DESIGN v5 (2026-11-07) keeps both tuples DISJOINT (the invariant above is
# unchanged) and adds the THIRD class — ASSERTED-VERDICT — which is a
# PER-ROW rule, not a per-field one: see `VERDICT_SCOPE` below.  A field that
# is ASSERTED on one row may be RESOLUTION on another (`said_ok` on a
# `line`-kind row), which is why the rule is carried by the row's own
# `said_kind` field rather than by a third tuple.
# --------------------------------------------------------------------------- #
#
# ASSERTED — a mutation to any of these MUST fail the guard.
CLAIM_FIELDS: tuple[str, ...] = (
    "error",      # only present when a probe's file is unreadable (never today)
    "expect",
    "file",
    "floor",
    "id",
    "ok",
    "said",
    "said_kind",  # v5: the CLAIM-SIDE probe kind — the guard SELECTS on it
    "said_ok",
    "section",
    "symbol",
)
# RESOLUTION — informational drift only; NEVER compared for equality.
RESOLUTION_FIELDS: tuple[str, ...] = (
    "count",
    "exists",
    "files",
    "lines",
    "note",
    "said_claim",
    "said_count",
    "said_lines",
)
# ASSERTED-VERDICT — the guard RE-MEASURES the live tree and asserts that the
# RECORDED verdict EQUALS the LIVE verdict (it does NOT assert that the
# verdict is TRUE).  The scope of each verdict field is PER-ROW; the row's own
# `said_kind` is the selector.
VERDICT_SCOPE: dict[str, str] = {
    "ok": "every row",
    "said_ok": (
        "every row EXCEPT a row whose said_kind == 'line' (a locator — it "
        "drifts on every unrelated edit above the symbol) and EXCEPT a row "
        "whose said_ok is None (no machine-checkable claim to be consistent "
        "about)"
    ),
}

# --------------------------------------------------------------------------- #
# The probe table.  `expect` is the ASSERTION, `said` is what the receipt
# claimed, `note` explains a divergence.  Nothing here is derived by hand at
# read time — every value below is resolved from the tree on every run.
# --------------------------------------------------------------------------- #
SPECS: list[dict[str, Any]] = [
    # ---- M1 -------------------------------------------------------------- #
    {"id": "M1", "file": STEREO, "expect": "contains",
     "symbol": '_ALLOWED_STEREO_MODES = frozenset({"noop", "remove_atom"})',
     "said": "L32", "section": "M1 §6.69a"},
    # ---- M2 -------------------------------------------------------------- #
    {"id": "M2", "file": METRICS_V2, "expect": "contains",
     "symbol": "def _has_metal(", "said": "6 hits (def + 5 call sites)",
     "section": "M2 §6.69b"},
    # ---- M3 — the receipt's verification command never existed ----------- #
    {"id": "M3a", "file": "molmetal/scripts/wf_a40_pivot_long_to_wide.py",
     "expect": "absent", "symbol": "--self-test",
     "said": "`--self-test` exit 0",
     "section": "M3 §6.69c",
     "note": "The flag was NEVER in the file: `git show b3482fc9:<file> | "
             "grep -c -- 'self.test'` = 0, and the script's argparse declares "
             "exactly two args (`--in`, `--out`), both required.  So the "
             "receipt's verification command cannot have exited 0.  FALSE AT BIRTH."},
    {"id": "M3b", "file": "molmetal/scripts/wf_a40_pivot_long_to_wide.py",
     "expect": "contains", "symbol": 'dest="out_path"',
     "said": "NEW script, 6639 bytes", "section": "M3 §6.69c"},
    # ---- M4 / M5 --------------------------------------------------------- #
    {"id": "M4", "file": "molmetal/loss/distillation_losses.py", "expect": "contains",
     "symbol": "weight: float = 0.05,", "said": "L34",
     "section": "M4 §6.69d",
     "note": "The receipt's term-NAME `shortcut_distill_term` does not exist; "
             "the class is `ShortcutDistillLoss` and its name attr is "
             "`shortcut_distill_loss`."},
    {"id": "M5", "file": "molmetal/loss/aggregation_losses.py", "expect": "contains",
     "symbol": "weight: float = 0.05,", "said": "L135",
     "section": "M5 §6.69e"},
    # ---- M6 -------------------------------------------------------------- #
    {"id": "M6", "file": TRAINER, "expect": "contains",
     "symbol": "fragment_loss_weight: float = 0.05,", "said": "L3291",
     "section": "M6 §6.69f",
     "note": "The receipt's literal `fragment_loss_weight=0.05` (no spacing) "
             "returns 0 hits; the real form is `fragment_loss_weight: float = 0.05,`."},
    # ---- M7 -------------------------------------------------------------- #
    {"id": "M7", "file": "molmetal/tests/test_bond_aromatic_lo_invariance.py",
     "expect": "file_exists", "symbol": "", "said": "NEW pytest, 5819 B",
     "section": "M7 §6.69g"},
    # ---- M8 — the claim §6.269.4b declared "exists nowhere" ------------- #
    {"id": "M8a", "file": "molmetal/loss/regularization_losses.py", "expect": "contains",
     "symbol": "class EntropyTerm:", "said": "L85",
     "section": "M8 §6.69h",
     "note": "§6.269.4b published that `entropy_term.weight = 0.1` \"exists "
             "nowhere in the tracked tree\".  It DOES exist.  Only the line and "
             "the identifier were stale."},
    {"id": "M8b", "file": "molmetal/loss/regularization_losses.py", "expect": "contains",
     "symbol": "weight: float = 0.1,", "said": "L85",
     "section": "M8 §6.69h"},
    {"id": "M8c", "file": TRAINER, "expect": "contains",
     "symbol": "entropy_reg_weight: float = 0.1,", "said": "L3235",
     "section": "M8 §6.69h",
     "note": "The receipt's literal `entropy_term_weight=0.1` returns 0 hits; "
             "the trainer carries the RENAMED `entropy_reg_weight`."},
    # ---- M9 — the flag was RETIRED deliberately, with a green guard ------ #
    {"id": "M9a", "file": TRAINER, "expect": "absent",
     "symbol": "consistency_term_weight", "said": "L3141",
     "section": "M9 §6.69i",
     "note": "Retired on purpose: `metrics/by_metric/cfm_epoch_wall.json` "
             "`/retractions/0/why` records the retirement, and "
             "`molmetal/tests/test_wfa105_diagnostic_terms.py` asserts the flag "
             "is GONE (green, CPU-only).  The receipt's VALUE (0.05) is also "
             "wrong: the live defaults are 0.0."},
    {"id": "M9b", "file": TRAINER, "expect": "contains",
     "symbol": "consistency_weight: float = 0.0,", "said": "L3141",
     "section": "M9 §6.69i"},
    # ---- M10 — no such token in ANY .py ---------------------------------- #
    {"id": "M10", "file": TRAINER, "expect": "absent",
     "symbol": "molecular_property_target_weight", "said": "L3296",
     "section": "M10 §6.69j",
     "note": "The token occurs in the tracked tree in exactly TWO files, BOTH "
             "prose (this receipt + the catalogue) and in 0 `.py` files."},
    # ---- M11 ------------------------------------------------------------- #
    {"id": "M11", "file": METRICS_V2, "expect": "contains",
     "symbol": "if _has_metal(mol):", "said": "L513 (511-530)",
     "section": "M11 §6.69k"},
    # ---- M12 ------------------------------------------------------------- #
    {"id": "M12", "file": TRAINER, "expect": "contains",
     "symbol": "DEFAULT_EARLY_STOP_PATIENCE = 10", "said": "L800",
     "section": "M12 §6.68a"},
    # ---- M13 / M14 ------------------------------------------------------- #
    {"id": "M13", "file": TRAINER, "expect": "contains",
     "symbol": "tmqm_min_per_metal", "said": "4 hits", "section": "M13 §6.66c"},
    {"id": "M14", "file": "molmetal/data/tmqm.py", "expect": "contains",
     "symbol": "TRANSITION_METALS", "said": "3 hits", "section": "M14 §6.66d",
     "note": "The receipt's count was 3; the token now resolves 4 times."},
    # ---- M15 — the receipt's paths NEVER EXISTED ------------------------- #
    {"id": "M15a", "file": "molmetal/adapters/fpocket_wrapper.py",
     "expect": "file_exists", "symbol": "", "said": "—",
     "section": "M15 §6.66f",
     "note": "The artefact the receipt MEANT.  Moved here by d4bbafb5 "
             "(Wf-A41-fixup-1)."},
    {"id": "M15b", "file": "molmetal/tests/test_fpocket_wrapper.py",
     "expect": "file_exists", "symbol": "", "said": "—",
     "section": "M15 §6.66f",
     "note": "The test the receipt MEANT.  Renamed from `test_fpocket_path.py`."},
    {"id": "M15c", "file": "molmetal/scripts/fpocket_wrapper.py",
     "expect": "file_absent", "symbol": "", "said": "molmetal/scripts/fpocket_wrapper.py",
     "section": "M15 §6.66f",
     "note": "The path the receipt CITED.  `git log --all --diff-filter=A` "
             "proves it was NEVER added to the repo.  The row was WRONG WHEN "
             "WRITTEN, not stale."},
    {"id": "M15d", "file": "molmetal/tests/test_fpocket_path.py",
     "expect": "file_absent", "symbol": "", "said": "molmetal/tests/test_fpocket_path.py",
     "section": "M15 §6.66f",
     "note": "Same: never existed.  The receipt row is FALSE AT BIRTH."},
    # ---- §6.71a / §6.71b / §6.71e / §6.71f ------------------------------ #
    {"id": "S671a", "file": METRICS_V2, "expect": "contains",
     "symbol": "p = p + _metal_amend(mol)", "said": "L567-573",
     "section": "§6.71a"},
    {"id": "S671b1", "file": "molmetal/tests/test_metrics_v2.py", "expect": "contains",
     "symbol": "def test_ppb_cisplatin_has_metal_amend(", "said": "L215-247",
     "section": "§6.71b"},
    {"id": "S671b2", "file": "molmetal/tests/test_metrics_v2.py", "expect": "contains",
     "symbol": "def test_ppb_organic_no_amend(", "said": "L215-247",
     "section": "§6.71b"},
    {"id": "S671e", "file": STEREO, "expect": "contains",
     "symbol": "§6.66e", "said": "L54", "section": "§6.71e docstring"},
    {"id": "S671eJ1", "file": "molmetal/tests/test_sascorer_fork_divergence.py",
     "expect": "contains",
     "symbol": "def test_opt1_direction_reverses_under_the_canonical_scorer(",
     "said": "L403", "section": "§6.71e J1 fix"},
    {"id": "S671f", "file": "molmetal/tests/test_assemble_term_arithmetic.py",
     "expect": "file_exists", "symbol": "", "said": "NEW, 3 tests",
     "section": "§6.71f"},
    # ---- the 13 trainer-flag flips (§6.71c) ------------------------------ #
    # 9 SURVIVE under the same name; 4 were RETIRED and the repo's own green
    # guard (test_wfa105_diagnostic_terms.py::REMOVED_FLAGS) asserts they are
    # gone.  The receipt's table listed 13 live flags.
    *[{"id": f"S671c:{f}", "file": TRAINER, "expect": "contains",
       "symbol": f"{f}: float", "said": "L3143-3286", "section": "§6.71c"}
      for f in ("fm_ode_term_weight", "energy_conservation_weight",
                "sboo3_term_weight", "chem_loss_weight", "qed_loss_weight",
                "molecule_loss_weight", "stereo_loss_weight")],
    # the 4 retired ones — ABSENT is the ASSERTION (the repo asserts it)
    *[{"id": f"S671c-retired:{f}", "file": TRAINER, "expect": "absent",
       "symbol": f, "said": "L3143-3144", "section": "§6.71c",
       "note": "RETIRED deliberately; `molmetal/tests/test_wfa105_diagnostic_terms.py` "
               "REMOVED_FLAGS asserts the flag is GONE."}
      for f in ("edm_term_weight", "shortcut_distill_term_weight",
                "consistency_term_weight", "so3_term_weight")],
    # the replacements that carry the semantics forward
    {"id": "S671c-repl:edm", "file": TRAINER, "expect": "contains",
     "symbol": "edm_preconditioning: bool = False,", "said": "—", "section": "§6.71c",
     "note": "The EDM term's semantics live on under a boolean + a sigma, not a weight."},
    {"id": "S671c-repl:sd", "file": TRAINER, "expect": "contains",
     "symbol": "shortcut_distill_weight: float = 1.0,", "said": "—", "section": "§6.71c"},
    # ---- §6.71d — the two yaml counts ----------------------------------- #
    # The FLOOR is the MEASURED count at the time this spec was written.  A
    # floor is a LOWER BOUND, never a freeze: the tree may GROW without
    # failing anything, but it may not fall below the floor.  The previous
    # producer defaulted the threshold to 1 — VACUOUS (one hit passed).
    {"id": "S671d-scaffold", "file": "configs/", "expect": "count_ge",
     "symbol": "tmqm_split_kind: scaffold", "floor": 29, "said": "3 hits",
     "section": "§6.71d",
     "note": "The receipt's `3` was the number of files THIS WAVE flipped, not "
             "the number the tree carried.  TWO POPULATIONS must be labelled and "
             "they are NOT the same: the TRACKED count (git grep over tracked "
             "files only) and the ON-DISK count (every file under configs/ on "
             "disk, tracked AND untracked).  Both are measured on every run and "
             "appended below.  SCOPING (2026-11-07, DESIGN v5): the `29` in "
             "this row's floor is the TRACKED count (git grep over TRACKED "
             "files only) — it is NOT an unqualified 'the tree carries 29'; "
             "the ON-DISK count is a DIFFERENT population, is measured on "
             "every run, and is labelled separately below.  A bare count must "
             "always be read with its population label."},
    {"id": "S671d-legacy", "file": "configs/", "expect": "count_ge",
     "symbol": "LEGACY OVERRIDE", "floor": 16, "said": "17 hits",
     "section": "§6.71d",
     "note": "The receipt's `17 hits` is FALSE against the tracked population "
             "and FALSE against the on-disk population.  The 17th file the "
             "receipt names (`configs/cfm_tmqm_all6.yaml`) received a REAL "
             "split flip (random -> scaffold) in e74832c8 — see the receipt's "
             "dated re-verification section.  Both populations are measured on "
             "every run and appended below.  SCOPING (2026-11-07, DESIGN v5): "
             "the `16` in this row's floor is the TRACKED count (git grep over "
             "TRACKED files only); the ON-DISK count is a DIFFERENT population "
             "— the two are EQUAL today, but they are NOT the same population "
             "and a bare count must always be read with its label."},
]


# --------------------------------------------------------------------------- #
# The `said` probe's grammar.  See the module docstring for the precedence
# order: LINE > HITS > TESTS > BYTES > PRESENCE > None.
# --------------------------------------------------------------------------- #
_SAID_LINE_RE = re.compile(r"L(\d+)")
_SAID_PAREN_RANGE_RE = re.compile(r"\(\s*(\d+)\s*-\s*(\d+)\s*\)")
_SAID_HITS_RE = re.compile(r"(\d+)\s+hits?\b")
_SAID_TESTS_RE = re.compile(r"(\d+)\s+tests?\b")
_SAID_BYTES_RE = re.compile(r"(\d+)\s*(?:bytes?\b|\bB\b)")
_FILE_ENTRY_RE = re.compile(r"^(?P<path>.*?)\s+\((?P<n>\d+)\)$")


# --------------------------------------------------------------------------- #
def _read(rel: str) -> list[str] | None:
    p = REPO / rel
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace").split("\n")


def _grep_in_dir(dir_rel: str, token: str) -> list[tuple[str, int]]:
    """Tracked-file grep inside a directory.  Uses `git grep` so that
    gitignored trees are not silently included and untracked junk is not."""
    r = subprocess.run(
        ["git", "grep", "-a", "-I", "-c", token, "--", dir_rel],
        cwd=REPO, capture_output=True, text=True,
    )
    out: list[tuple[str, int]] = []
    for ln in r.stdout.strip().split("\n"):
        if not ln:
            continue
        path, _, cnt = ln.rpartition(":")
        try:
            out.append((path, int(cnt)))
        except ValueError:
            continue
    return out


def _marker_paths_in_dir(dir_rel: str, token: str) -> set[str]:
    """The tracked paths under ``dir_rel`` that currently carry ``token``."""
    r = subprocess.run(
        ["git", "grep", "-a", "-I", "-l", token, "--", dir_rel],
        cwd=REPO, capture_output=True, text=True,
    )
    return {ln for ln in r.stdout.strip().split("\n") if ln}


def _ondisk_populations(dir_rel: str, token: str) -> tuple[int, int]:
    """(files, hits) over EVERY file on disk under ``dir_rel`` — tracked AND
    untracked.  This is the second population the §6.71d rows must label; the
    `git grep` population above only ever sees TRACKED files."""
    base = REPO / dir_rel
    if not base.is_dir():
        return (0, 0)
    n_files = 0
    n_hits = 0
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n = text.count(token)
        if n:
            n_files += 1
            n_hits += n
    return (n_files, n_hits)


def _probe_token(symbol: str) -> str:
    """The token a COUNT claim is measured against.

    A receipt that says "6 hits (def + 5 call sites)" counted OCCURRENCES OF
    THE IDENTIFIER, not occurrences of the literal probe string (which is the
    `def` line alone).  So for an identifier-like symbol the probe token is
    the identifier; for a phrase it is the phrase itself."""
    tok = symbol.strip()
    if tok.startswith("def "):
        tok = tok[4:]
    if tok.endswith("("):
        tok = tok[:-1]
    return tok


def _file_paths(entries: list[str]) -> list[str]:
    """The bare paths of a stored `files` list ("path (n)" -> "path")."""
    out: list[str] = []
    for e in entries:
        m = _FILE_ENTRY_RE.match(e)
        out.append(m.group("path") if m else e)
    return out


def _count_ge_units(row: dict[str, Any]) -> tuple[str, list[Any]]:
    """The STORED units a `count_ge` row's predicate is evaluated over.

    A dir probe stores the FILES that carry the marker; a file probe stores
    the LINES that carry the symbol.  Both are the units the receipt's count
    claim was about."""
    files = row.get("files") or []
    if files:
        return ("path", _file_paths(list(files)))
    return ("line", list(row.get("lines") or []))


def _spec_floor(spec: dict[str, Any]) -> int:
    """The EXPLICIT floor of a `count_ge` probe.

    A floor is MANDATORY: the previous producer defaulted the threshold to
    `1`, which is VACUOUS (one hit passed, so the check could not fail).  A
    `count_ge` spec without a positive integer floor now RAISES.

    A floor BELOW the measured live count is ALSO vacuous (it can never fire);
    that is not checked here — it is a REVIEWER decision on the MEASURED
    number, and the sidecar records the measured live count beside the floor
    on every run."""
    if "floor" not in spec:
        raise ValueError(
            f"count_ge probe {spec.get('id')!r} declares NO floor — a floor is "
            "mandatory (the previous default of 1 was vacuous: one hit "
            "passed, so the check could not fail)"
        )
    floor = spec["floor"]
    if isinstance(floor, bool) or not isinstance(floor, int) or floor <= 0:
        raise ValueError(
            f"count_ge probe {spec.get('id')!r} has a non-positive or "
            f"non-integer floor {floor!r} — a floor must be a positive integer"
        )
    return floor


def count_ge_verdict(
    spec: dict[str, Any],
    stored_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """THE THREE-PART PREDICATE for a `count_ge` probe.

        (a) live_count >= floor                              — SHRINKAGE
        (b) len(stored units) >= floor                       — TAMPERING
        (c) every stored unit still carries the marker        — RELOCATION

    ``live_files == stored_files`` is explicitly NOT compared: that is the
    treadmill (every legitimate ADDITION would fail it).  (c) is a SUBSET
    check, so the tree may GROW freely; only a unit that LOST the marker
    fails.

    ``stored_row`` is the PREVIOUS sidecar row for the same probe id.  When it
    is ``None`` the units are the LIVE ones, so (b) and (c) hold trivially and
    the verdict reduces to (a).  That is what the WRITE path does (it records
    the tree's verdict); the GUARD passes the stored row, which is what makes
    (b) and (c) able to fire.
    """
    floor = _spec_floor(spec)
    rel = spec["file"]
    token = spec.get("symbol", "")
    if rel.endswith("/"):
        hits = _grep_in_dir(rel, token)
        live_count = sum(c for _, c in hits)
        live_units: tuple[str, list[Any]] = ("path", sorted(p for p, _ in hits))
    else:
        src = _read(rel)
        live_count = 0
        live_units = ("line", [])
        if src is not None:
            live_lines = [i for i, ln in enumerate(src, 1) if token in ln]
            live_count = len(live_lines)
            live_units = ("line", live_lines)

    if stored_row is None:
        stored_units = live_units
    else:
        stored_units = _count_ge_units(stored_row)

    a_shrinkage = live_count >= floor
    b_tamper = len(stored_units[1]) >= floor
    if stored_units[0] == "path":
        live_paths = _marker_paths_in_dir(rel, token)
        c_relocation = all(p in live_paths for p in stored_units[1])
    else:
        src = _read(rel)
        c_relocation = True
        if src is not None:
            for ln in stored_units[1]:
                if not (1 <= ln <= len(src)) or token not in src[ln - 1]:
                    c_relocation = False
                    break
    return {
        "floor": floor,
        "a_shrinkage": bool(a_shrinkage),
        "b_tamper": bool(b_tamper),
        "c_relocation": bool(c_relocation),
        "verdict": bool(a_shrinkage and b_tamper and c_relocation),
        "live_count": live_count,
        "stored_units": len(stored_units[1]),
    }


def _names_artefact(spec: dict[str, Any], said: str) -> str | None:
    """Does the receipt's `said` name the row's own artefact?

    Returns "symbol" / "path" / None.  This is the PRESENCE form of the said
    probe: the receipt claimed the artefact was THERE, so the re-measurement
    is whether it is present."""
    sym = spec.get("symbol", "")
    if sym and sym in said:
        return "symbol"
    rel = spec.get("file", "")
    if rel and not rel.endswith("/") and rel in said:
        return "path"
    return None


def _said_probe(
    spec: dict[str, Any],
    row: dict[str, Any],
    *,
    src: list[str] | None,
    token_hits: int | None,
    present: bool | None,
) -> None:
    """RE-MEASURE the receipt's `said` claim into ``row``.

    Sets ``said_claim`` (what was claimed), ``said_kind`` (the CLAIM-SIDE
    probe kind — `'line' | 'hits' | 'tests' | 'bytes' | 'presence' | 'none'`,
    set from the SAME place as ``said_claim["kind"]`` so the two cannot
    disagree), ``said_lines`` / ``said_count`` (the measurement that backs the
    claim) and ``said_ok`` (the verdict).  ``said_ok`` is None when the claim
    is not machine-checkable."""
    said = row["said"]
    claimed_lines = [int(x) for x in _SAID_LINE_RE.findall(said)]
    paren = _SAID_PAREN_RANGE_RE.findall(said)

    # 1. LINE ----------------------------------------------------------------
    if claimed_lines:
        if paren:
            lo, hi = int(paren[0][0]), int(paren[0][1])
        elif len(claimed_lines) >= 2:
            lo, hi = min(claimed_lines), max(claimed_lines)
        else:
            lo = hi = claimed_lines[0]
        row["said_lines"] = list(row["lines"])
        row["said_claim"] = {"kind": "line", "claimed": [lo, hi], "text": said}
        row["said_kind"] = "line"
        row["said_ok"] = any(lo <= ln <= hi for ln in row["said_lines"])
        return

    # 2. HITS ----------------------------------------------------------------
    m = _SAID_HITS_RE.search(said)
    if m:
        claimed = int(m.group(1))
        measured = int(token_hits or 0)
        row["said_count"] = measured
        row["said_claim"] = {"kind": "hits", "claimed": claimed, "text": said}
        row["said_kind"] = "hits"
        row["said_ok"] = measured >= claimed
        return

    # 3. TESTS ---------------------------------------------------------------
    m = _SAID_TESTS_RE.search(said)
    if m:
        claimed = int(m.group(1))
        measured = 0
        if src is not None:
            measured = sum(1 for ln in src if re.match(r"\s*def test_", ln))
        row["said_count"] = measured
        row["said_claim"] = {"kind": "tests", "claimed": claimed, "text": said}
        row["said_kind"] = "tests"
        row["said_ok"] = measured >= claimed
        return

    # 4. BYTES ---------------------------------------------------------------
    m = _SAID_BYTES_RE.search(said)
    if m:
        claimed = int(m.group(1))
        measured = 0
        p = REPO / spec["file"]
        if p.is_file():
            measured = p.stat().st_size
        row["said_count"] = measured
        row["said_claim"] = {"kind": "bytes", "claimed": claimed, "text": said}
        row["said_kind"] = "bytes"
        row["said_ok"] = measured >= claimed
        return

    # 5. PRESENCE -----------------------------------------------------------
    named = _names_artefact(spec, said)
    if named is not None:
        row["said_claim"] = {"kind": "presence", "claimed": named, "text": said}
        row["said_kind"] = "presence"
        row["said_ok"] = bool(present)
        return

    # 6. no machine-checkable claim ----------------------------------------
    row["said_claim"] = {"kind": "none", "claimed": None, "text": said}
    row["said_kind"] = "none"
    row["said_ok"] = None


def resolve(spec: dict[str, Any], stored: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ONE probe against the current tree.  Read-only.

    ``stored`` is the PREVIOUS sidecar row for the same probe id, when the
    caller has one.  It is what makes the `count_ge` predicate's parts (b) and
    (c) — TAMPERING and RELOCATION — able to fire; with no stored row the
    predicate reduces to (a) SHRINKAGE."""
    rel = spec["file"]
    expect = spec["expect"]
    symbol = spec.get("symbol", "")
    row: dict[str, Any] = {
        "id": spec["id"],
        "file": rel,
        "expect": expect,
        "symbol": symbol,
        "said": spec.get("said", ""),
        "section": spec.get("section", ""),
        "floor": spec.get("floor"),
        "ok": False,
        "said_kind": "none",
        "said_ok": None,
        # RESOLUTION fields — present on EVERY row so the row shape is uniform
        # and every stored field is classified by exactly one tuple.
        "count": 0,
        "lines": [],
        "files": [],
        "exists": None,
        "note": spec.get("note", ""),
        "said_claim": {"kind": "none", "claimed": None, "text": spec.get("said", "")},
        "said_count": None,
        "said_lines": [],
    }

    # ---- file-level probes ------------------------------------------------ #
    if expect in ("file_exists", "file_absent"):
        exists = (REPO / rel).is_file()
        row["exists"] = exists
        row["count"] = 1 if exists else 0
        row["ok"] = (exists if expect == "file_exists" else not exists)
        src = _read(rel) if exists else None
        _said_probe(spec, row, src=src, token_hits=row["count"], present=exists)
        return row

    # ---- directory-level probes ------------------------------------------- #
    if rel.endswith("/"):
        hits = _grep_in_dir(rel, symbol)
        total = sum(c for _, c in hits)
        row["count"] = total
        row["files"] = [f"{p} ({c})" for p, c in hits]
        floor = _spec_floor(spec)
        verdict = count_ge_verdict(spec, stored)
        row["floor"] = floor
        row["ok"] = bool(verdict["verdict"])
        # Label BOTH populations: the TRACKED count (git grep) and the ON-DISK
        # count (tracked + untracked).  The receipt's number was wave-scoped;
        # neither population is "the" number without a label.
        od_files, od_hits = _ondisk_populations(rel, symbol)
        row["note"] = (
            row["note"]
            + "  MEASURED populations: TRACKED files = "
            + f"{len(hits)} (tracked hits = {total}); ON-DISK files = {od_files} "
            + f"(on-disk hits = {od_hits})."
        )
        _said_probe(spec, row, src=None, token_hits=total, present=total > 0)
        return row

    # ---- file-level symbol probes ----------------------------------------- #
    src = _read(rel)
    if src is None:
        row["ok"] = False
        row["error"] = "file not found"
        _said_probe(spec, row, src=None, token_hits=0, present=False)
        return row

    lines = [i for i, ln in enumerate(src, 1) if symbol in ln]
    row["count"] = len(lines)
    row["lines"] = lines
    token = _probe_token(symbol)
    token_hits = sum(1 for ln in src if token in ln) if token else 0

    if expect == "contains":
        row["ok"] = len(lines) > 0
    elif expect == "absent":
        row["ok"] = len(lines) == 0
    elif expect == "count_ge":
        verdict = count_ge_verdict(spec, stored)
        row["floor"] = _spec_floor(spec)
        row["ok"] = bool(verdict["verdict"])
    else:
        raise ValueError(f"unknown expect: {expect}")

    _said_probe(spec, row, src=src, token_hits=token_hits, present=len(lines) > 0)
    return row


def resolve_all(stored_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    stored_by_id = {r.get("id"): r for r in (stored_rows or [])}
    return [resolve(s, stored_by_id.get(s["id"])) for s in SPECS]


def canonical_rows_bytes(rows: list[dict[str, Any]]) -> bytes:
    """THE canonical serialisation of the ROWS array - ONE definition, shared.

    Wf-A269 wave-32d (2026-11-07).  This used to be INLINE inside
    `build_sidecar`, so the only way to reproduce the sidecar's `rows_sha256`
    was to RE-IMPLEMENT it.  That is the repo's own §6.269.35.3 lesson (a copy
    drifts silently) and it was load-bearing here: the rows carry non-ASCII
    characters, and Python's DEFAULT `ensure_ascii=True` ESCAPES them, so the
    SAME rows hash DIFFERENTLY under the two flags.

    MEASURED 2026-11-07 on this repo's own sidecar (43 rows):
      * `ensure_ascii=False` (this function)
        -> 02b151b3a36671205286ccaedb5894d647e75f9c2a128fa57b1dd37746afe71f
        (22 428 B) == the stored value
      * `ensure_ascii=True` (Python's default)
        -> 586a51903ca757d0721ed2d7878b1668f47bb0f9142b7c989ddf3d4db6708caf
        (22 641 B) MISMATCH

    THREE POPULATIONS, ONE LABEL (repo §6.269.35.2).  The non-ASCII character
    count has TWO different populations, and an earlier revision of this
    docstring named the WRONG one next to the row count:
      * in the **HASHED ROWS** (what this function serialises) = **56**
        non-ASCII chars (**45** `§` + **11** `—`)
      * in the **WHOLE SIDECAR FILE** = **66** non-ASCII chars
        (**47** `§` + **19** `—`)
    The two differ because the file also carries the envelope's own
    resolution prose.  State which population a non-ASCII count belongs to.

    => A consumer that re-implements this with the default flag reports a FALSE
    MISMATCH.  The consumer MUST import this function instead
    (`molmetal/tests/test_wfa269_receipt_locator_guard.py` does).
    """
    return json.dumps(rows, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def build_sidecar(stored_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the sidecar document.

    ``stored_rows`` defaults to the rows of the sidecar ON DISK — that is the
    comparison the GUARD wants (the fresh verdict re-derived against what the
    sidecar claims).  Pass ``[]`` for the WRITE path, which records the tree's
    own verdict."""
    if stored_rows is None:
        stored_rows = _sidecar_rows(SIDECAR) or []
    rows = resolve_all(stored_rows)

    # DESIGN v5 (2026-11-07) — a `count_ge` probe WITHOUT an explicit floor
    # RAISES here rather than falling back to a vacuous threshold.  A floor is
    # a LOWER BOUND, never a freeze.
    for spec in SPECS:
        if spec.get("expect") == "count_ge":
            _spec_floor(spec)

    # The invariant that keeps the two field classes honest: every field a row
    # stores is classified by exactly one tuple, so a field added later cannot
    # silently escape the guard (the repo's own §6.269.35.3 defect class).
    classified = set(CLAIM_FIELDS) | set(RESOLUTION_FIELDS)
    for r in rows:
        unclassified = sorted(set(r) - classified)
        if unclassified:
            raise AssertionError(
                f"row {r['id']} stores unclassified field(s) {unclassified} — add "
                "them to CLAIM_FIELDS or RESOLUTION_FIELDS"
            )
        # DESIGN v5 (2026-11-07) — `said_kind` is the CLAIM-SIDE selector and
        # it is set from the SAME place as `said_claim["kind"]`, so the two
        # MUST agree; a drift between them is itself a defect.
        claim_kind = (r.get("said_claim") or {}).get("kind")
        if r.get("said_kind") != claim_kind:
            raise AssertionError(
                f"row {r['id']} has said_kind={r.get('said_kind')!r} but "
                f"said_claim.kind={claim_kind!r} — they are set from ONE place "
                "and must agree"
            )

    canonical_rows = canonical_rows_bytes(rows)
    return {
        "schema": "wf269a.receipt_locators.v1",
        "receipt": "docs/wf_a44_verify_receipts.md",
        "producer": "molmetal/scripts/wf269a_receipt_locator_refresh.py",
        "gate": "molmetal/tests/test_wfa269_receipt_locator_guard.py",
        "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim_fields": sorted(CLAIM_FIELDS),
        "resolution_fields": sorted(RESOLUTION_FIELDS),
        "verdict_scope": VERDICT_SCOPE,
        "rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "design_rule": (
            "DESIGN v5 — CLAIM vs RESOLUTION (THREE classes).  (i) "
            "CLAIM/RESOLUTION SEPARATION: every stored field is classified — "
            "ASSERTED (a mutation MUST fail the guard), ASSERTED-VERDICT (the "
            "guard RE-MEASURES the live tree and asserts that the RECORDED "
            "verdict EQUALS the LIVE verdict — it does NOT assert that the "
            "verdict is TRUE), or RESOLUTION (informational drift, NEVER "
            "compared for equality).  A guard may FAIL only on a DURABLE "
            "claim — does the symbol still exist, is the count still above its "
            "floor, is the verdict still right; every derived field (line "
            "numbers, byte offsets, content hashes) is REPORTED as "
            "informational drift, never ASSERTED.  The row's own `said_kind` "
            "(line | hits | tests | bytes | presence | none) SELECTS which "
            "assertion applies to that row: `ok` is asserted on every row, and "
            "`said_ok` on every row EXCEPT a `line`-kind row (a locator drifts "
            "on every unrelated edit) and EXCEPT a row whose said_ok is None.  "
            "(ii) FLOOR SEMANTICS: a count_ge probe carries a FLOOR, never a "
            "freeze — a floor is a LOWER BOUND the live count must not fall "
            "below, so the tree may GROW without failing; a count_ge spec "
            "without an explicit positive integer floor RAISES (the previous "
            "default of 1 was vacuous: one hit passed, so the check could not "
            "fail), and a floor BELOW the measured live count is ALSO vacuous "
            "and is not acceptable; the count_ge verdict is the three-part "
            "predicate (a) live_count >= floor, (b) len(sidecar.files) >= "
            "floor, (c) every sidecar-named path still carries the marker — "
            "and live_files == sidecar.files is NOT compared (that is the "
            "treadmill).  (iii) REFRESHING IS A SEPARATE, NON-BLOCKING "
            "REGENERATION STEP: it rewrites the resolution fields and the "
            "timestamp and never decides whether a claim holds.  (iv) "
            "AGENTS.md §4's STALE-FLAG RULE: the PRODUCER stamps "
            "(`resolved_at`), the CONSUMER decides, a stamp older than 24 h "
            "means FAIL, and a missing, unparseable or future-dated stamp "
            "fails CLOSED.  Plus Wf-A269 §6.269.4h's generalised rule: a "
            "locator is a RESOLVER (symbol -> current line), not a FROZEN LINE "
            "NUMBER.  KNOWN LIMIT: the `hits` said-verdict is a LOWER BOUND "
            "(measured >= claimed), so it cannot certify a claim's BREAKDOWN."
        ),
        "coverage": {
            "covered_probes": len(rows),
            "covered_files": len({r["file"] for r in rows}),
            "not_covered": [
                "the 19 JUDGEMENT-CALL ledger rows (catalogue prose, not file locators)",
                "the receipt's pre-flight claim (110 PASS / 3 FAIL / 10 SKIP / 124 total) "
                "— a wave-scoped snapshot of a live tool, not a locator",
            ],
            "does_not_run_pytest": (
                "the guard pins the ARTEFACTS; the repo's own test run executes them"
            ),
        },
        "rows": rows,
    }


def _fmt_lines(lines: list[int]) -> str:
    if not lines:
        return "—"
    if len(lines) == 1:
        return f"L{lines[0]}"
    return f"L{lines[0]}…L{lines[-1]} ({len(lines)} hits)"


def _fmt_bool(v: bool | None) -> str:
    if v is None:
        return "—"
    return "TRUE" if v else "FALSE"


def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|")


def _said_verdict_cell(row: dict[str, Any]) -> str:
    """The `said` verdict cell, labelled by the row's CLASS.

    DESIGN v5: `said_ok` is ASSERTED-VERDICT on every row EXCEPT a
    `line`-kind row (a LOCATOR — it drifts on every unrelated edit above the
    symbol) and EXCEPT a row whose `said_ok` is None (no machine-checkable
    claim).  Printing such a value as a bare TRUE/FALSE would print a
    RESOLUTION value as if it were a claim, which is exactly what the two
    tables exist to prevent."""
    kind = row.get("said_kind") or "none"
    said_ok = row.get("said_ok")
    if kind == "line":
        return f"{_fmt_bool(said_ok)} (informational — kind=line)"
    if said_ok is None:
        return "— (no machine-checkable claim)"
    return _fmt_bool(said_ok)


def render_table(rows: list[dict[str, Any]]) -> str:
    """Render the sidecar's rows as TWO tables.

    WHAT IS TRUE HERE.  The table is RENDERED FROM THE SIDECAR, and the
    receipt EMBEDS A COPY of it.  This function never writes the receipt, so
    the copy is refreshed by a SEPARATE step and the two CAN drift apart;
    drift is REPORTED (the guard compares the receipt's embedded copy against
    a fresh render), never PREVENTED.  (The previous docstring claimed the
    receipt and the sidecar "cannot drift apart" — that was false.)

    The ASSERTED columns and the RESOLUTION columns are printed in SEPARATE
    tables on purpose: a resolution column (a resolved line number, a hit
    count) printed among the claims reads as a claim, and it is not one.
    DESIGN v5 (2026-11-07) applies the same rule to the `said` verdict: it
    is printed with the row's `said_kind` and is labelled INFORMATIONAL on a
    `line`-kind row, because on that row it is RESOLUTION, not an asserted
    verdict."""
    claimed = [
        "| id | expect | the file it is about | the symbol | the receipt's claim | "
        "floor | the claim holds | the said kind | the said verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        floor = r.get("floor")
        claimed.append(
            f"| {r['id']} | `{r['expect']}` | `{r['file']}` | "
            f"`{_esc(r['symbol'] or '(file)')}` | {_esc(r['said'] or '—')} | "
            f"{floor if floor is not None else '—'} | "
            f"{'**TRUE**' if r.get('ok') else '**FALSE**'} | "
            f"`{r.get('said_kind') or 'none'}` | {_said_verdict_cell(r)} |"
        )

    resolved = [
        "| id | measured count | resolved lines | files | exists |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        files = r.get("files") or []
        resolved.append(
            f"| {r['id']} | {r.get('count')} | {_fmt_lines(r.get('lines') or [])} | "
            f"{len(files)} file(s) | "
            f"{('EXISTS' if r.get('exists') else 'DOES NOT EXIST') if r.get('exists') is not None else '—'} |"
        )

    return (
        "**ASSERTED — the claims the guard enforces "
        "(a mutation to any of these fails the guard)**\n\n"
        + "\n".join(claimed)
        + "\n\n**RESOLUTION — measured now; informational drift only, "
          "NEVER compared for equality**\n\n"
        + "\n".join(resolved)
    )


def _sidecar_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("rows")
    except Exception:
        return None


def _row_asserted_fields(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The fields ASSERTED for ONE row, and the fields that are RESOLUTION for it.

    DESIGN v5 (2026-11-07): `said_ok` is ASSERTED-VERDICT on every row EXCEPT
    a `line`-kind row (a LOCATOR — it drifts on every unrelated edit above the
    symbol) and EXCEPT a row whose `said_ok` is None (there is no
    machine-checkable claim to be consistent about).  On those rows `said_ok`
    is RESOLUTION, so a drift in it is REPORTED as informational drift and
    never as a claim violation."""
    asserted = list(CLAIM_FIELDS)
    resolution = list(RESOLUTION_FIELDS)
    if row.get("said_kind") == "line" or row.get("said_ok") is None:
        if "said_ok" in asserted:
            asserted.remove("said_ok")
            resolution.append("said_ok")
    return sorted(asserted), sorted(resolution)


def _diff(old: list[dict[str, Any]] | None, new: list[dict[str, Any]]) -> list[str]:
    """Report the drift between the sidecar on disk and a fresh resolution.

    CLAIM drift (an ASSERTED field moved) is reported FIRST and separately
    from RESOLUTION drift (a derived measurement moved) — they are not the
    same kind of event and must not be reported as one.  DESIGN v5: the
    classification is PER-ROW, because `said_ok` is RESOLUTION on a
    `line`-kind row (see `_row_asserted_fields`)."""
    if old is None:
        return ["(no sidecar on disk — everything is new)"]
    out: list[str] = []
    old_by = {r["id"]: r for r in old}
    new_by = {r["id"]: r for r in new}
    for rid, r in new_by.items():
        o = old_by.get(rid)
        if o is None:
            out.append(f"+ {rid}: NEW probe")
            continue
        asserted, resolution = _row_asserted_fields(r)
        claim_drift = sorted(k for k in asserted if o.get(k) != r.get(k))
        if claim_drift:
            out.append(
                f"~ CLAIM {rid}: {claim_drift} differ "
                f"(disk={ {k: o.get(k) for k in claim_drift} } "
                f"fresh={ {k: r.get(k) for k in claim_drift} })  [{r['file']}]"
            )
        res_drift = sorted(k for k in resolution if o.get(k) != r.get(k))
        if res_drift:
            out.append(
                f"~ RESOLUTION {rid}: {res_drift} differ  [{r['file']}]"
            )
    for rid in old_by:
        if rid not in new_by:
            out.append(f"- {rid}: DROPPED")
    return out


def _said_table(rows: list[dict[str, Any]]) -> list[str]:
    """id | said | measured | said_ok | kind — for every row whose `said`
    carries a machine-checkable claim.

    The `kind` column is the CLAIM-SIDE probe kind (DESIGN v5): it is what
    tells a reader whether the verdict is an ASSERTED-VERDICT (hits / tests /
    bytes / presence) or RESOLUTION (a `line` locator, which drifts on every
    unrelated edit above the symbol)."""
    out = ["  id              | the receipt said                    | measured            "
           "| said_ok  | kind"]
    for r in rows:
        if r.get("said_ok") is None:
            continue
        claim = r.get("said_claim") or {}
        kind = claim.get("kind")
        if kind == "line":
            measured = _fmt_lines(r.get("said_lines") or [])
        elif kind == "presence":
            measured = "PRESENT" if r.get("said_ok") else "ABSENT"
        else:
            measured = f"{r.get('said_count')} {kind}"
        out.append(
            f"  {r['id']:<15} | {str(r.get('said'))[:32]:<32} | "
            f"{measured:<19} | {_fmt_bool(r.get('said_ok')):<8} | {kind}"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="atomically write the sidecar (refresh)")
    ap.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = ap.parse_args(argv)

    if args.json:
        print(json.dumps(build_sidecar(), ensure_ascii=False, indent=2))
        return 0

    # The WRITE path records the tree's own verdict (no stored row ⇒ the
    # predicate reduces to (a) SHRINKAGE).  The CHECK path re-derives the
    # verdict against what the sidecar CLAIMS, which is what makes TAMPERING
    # and RELOCATION visible.
    disk_rows = _sidecar_rows(SIDECAR)
    side = build_sidecar(stored_rows=[] if args.write else disk_rows)
    rows = side["rows"]

    print(render_table(rows))
    print()
    bad = [r for r in rows if not r["ok"]]
    print(f"probes: {len(rows)}  PASS: {len(rows) - len(bad)}  FAIL: {len(bad)}")
    for r in bad:
        print(f"  FAIL {r['id']}: {r['file']} :: {r.get('symbol')!r} "
              f"(expect {r['expect']}, got {r.get('count')})")

    print()
    print("count_ge probes (floor = a LOWER BOUND, never a freeze):")
    for spec in SPECS:
        if spec["expect"] != "count_ge":
            continue
        row = next(r for r in rows if r["id"] == spec["id"])
        v = count_ge_verdict(spec, next(
            (r for r in (disk_rows or []) if r.get("id") == spec["id"]), None))
        print(f"  {spec['id']}: floor={spec['floor']}  live_count={v['live_count']}  "
              f"stored_units={v['stored_units']}  "
              f"a_shrinkage={v['a_shrinkage']}  b_tamper={v['b_tamper']}  "
              f"c_relocation={v['c_relocation']}  verdict={v['verdict']}")
        print(f"      {row['note'].strip()}")

    print()
    print("the `said` probe (the receipt's claim, RE-MEASURED):")
    for line in _said_table(rows):
        print(line)

    d = _diff(disk_rows, rows)
    print()
    print("sidecar diff:")
    for line in d:
        print(f"  {line}")

    if args.write:
        sys.path.insert(0, str(REPO))
        from molmetal.scripts.wf386a3_atomic_write import (  # noqa: PLC0415
            atomic_write_file,
            detect_truncation,
            safe_overwrite_check,
        )
        body = json.dumps(side, ensure_ascii=False, indent=2) + "\n"
        original = SIDECAR.read_text(encoding="utf-8") if SIDECAR.is_file() else ""
        print(f"\nsafe_overwrite_check: {safe_overwrite_check(SIDECAR, body, raise_on_risk=False)}")
        trunc = detect_truncation(body, expected_min_length=max(1000, len(original)))
        assert trunc is None, f"truncation detected: {trunc}"
        atomic_write_file(SIDECAR, body, encoding="utf-8", fsync=True)
        # MEASURED 2026-11-07 (Wf-A269 producer repair): the previous label
        # printed `len(body)` — a CHARACTER count — with a "B" suffix.  The
        # sidecar carries **66** non-ASCII characters (**47** `§` + **19** `—`),
        # so the printed number was **85** bytes BELOW the file's true size
        # (**36 364** chars vs **36 449** bytes on disk).  A byte claim that is
        # not a byte count is the §6.269.38 class (a number in a
        # machine-consumed string that no longer matches the file it
        # describes); the number printed with a "B" suffix is now the TRUE
        # byte count, and the character count is still shown, labelled as
        # such.  (Nothing parses this line — it is a human-facing label.)
        print(f"WROTE {SIDECAR.relative_to(REPO)} "
              f"({len(body.encode('utf-8'))} B on disk, {len(body)} chars)")
    elif d and d != ["(no sidecar on disk — everything is new)"]:
        print("\n(sidecar is stale — re-run with --write to refresh)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
