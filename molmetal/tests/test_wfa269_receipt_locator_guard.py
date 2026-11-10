#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Wf-A269 — the GATE for `docs/wf_a44_verify_receipts.md`'s locator table.

WHY THIS EXISTS.  `docs/wf_a44_verify_receipts.md` is the Wf-A44 verify-receipt:
a prose table that claims "76 LOC/audit items, all on disk at 2026-10-15", each
row backed by a `grep` / `pytest` command.  MEASURED 2026-11-02: **9 of its 13
grep-verified rows point at dead line-numbers, 2 of its rows were false at birth,
and nothing in the repo noticed** — the receipt is a static prose table with **no
producer and no gate**.  Catalogue §6.269.11 item 24 recorded that gap; this
file closes it.

THE RULE THIS ENCODES (Wf-A269 §6.269.4h's generalised rule, made operational):

    **A locator must be a RESOLVER (a symbol that resolves to a current line),
    not a FROZEN LINE NUMBER.**

A frozen line number is a *textual probe*: it cannot distinguish "the symbol was
renamed" from "the symbol was deleted", and it breaks on every unrelated edit
above it.  So this gate asserts **symbol-level claims**, which is both more
robust and more meaningful.

================================================================================
DESIGN v5 (2026-11-07) — THE CLAIM IS SEPARATED FROM THE RESOLUTION
================================================================================

Everything the sidecar stores falls into exactly THREE classes, and the class
decides whether a mutation may FAIL this guard:

  * **ASSERTED** — the DURABLE claims.  A mutation to any of these MUST fail
    the guard: `id`, `file`, `expect`, `symbol`, `said`, `said_kind`, `floor`,
    `section`.
  * **ASSERTED-VERDICT** — the guard RE-MEASURES the live tree and asserts
    that the RECORDED verdict EQUALS the LIVE verdict.  It does **NOT** assert
    that the verdict is `True` (a retired flag's ABSENCE is a TRUE claim, and
    four receipt claims are legitimately FALSE today and disclosed by the
    receipt itself).
      - `ok`      — on every row.
      - `said_ok` — on every row EXCEPT a row whose `said_kind == 'line'` (a
        LOCATOR: it drifts on every unrelated edit above the symbol) and EXCEPT
        a row whose `said_ok` is `None` (no machine-checkable claim to be
        consistent about).
  * **RESOLUTION** — the DERIVED measurements: resolved line numbers, hit
    counts, the file list, byte offsets.  **informational drift only, NEVER
    compared for equality.**  A line number moves on every unrelated edit above
    it; a guard that fails on that is a treadmill, not a guard.

⇒ **A GUARD MAY FAIL ONLY ON A DURABLE CLAIM.**  Every derived field is
  REPORTED as informational drift, never ASSERTED.  The drift is reported
  through the PRODUCER's own `_diff` channel, so there is exactly one
  classifier in the repo and this guard does not keep a second, drifting copy.

⇒ **THE THREE-PART PREDICATE** (`count_ge` rows only) is the one place the
  stored units are used, and never as an equality:

      (a) live_count >= floor                       — SHRINKAGE
      (b) len(stored units) >= floor                — TAMPERING
      (c) every stored unit still carries the marker — RELOCATION

  `live_files == stored_files` is explicitly NOT compared (that is the
  treadmill).  (c) is a MEMBERSHIP test in the live marker-path set — NOT an
  `os.path.exists` test: a stored entry is `"path (n)"` (the hit-count suffix
  must be stripped) and the marker may legitimately live in a DIFFERENT file.

⇒ **INDEPENDENCE.**  This guard implements the three-part predicate ITSELF and
  then CROSS-CHECKS it against the producer's own `count_ge_verdict` (loaded
  with `importlib` and CALLED, never copied).  Two independent implementations
  that AGREE are evidence; one implementation called twice is not.  If they
  disagree, this guard FAILS and says which one disagrees — it never papers
  over a disagreement.

⇒ **THE PROBE'S OWN BLIND SPOT IS REPORTED, NOT CERTIFIED.**  The `hits`
  verdict is `measured >= claimed` — a LOWER BOUND — so it CANNOT certify a
  claim that carries a BREAKDOWN.  MEASURED 2026-11-07: the receipt's **M2**
  row says "6 hits (def + 5 call sites)" and the live grep prints **7** lines,
  yet `said_ok` is `True` (7 >= 6).  A green verdict is therefore NOT a
  certification of such a row, and this guard says so in its report.

WHAT IT DOES NOT COVER (the coverage bound, stated — a guard's coverage is not a
defect count):
  * the 19 JUDGEMENT-CALL ledger rows (catalogue prose, not file locators);
  * the receipt's pre-flight claim (`110 PASS / 3 FAIL / 10 SKIP / 124 total`,
    a wave-scoped snapshot of a live tool, not a locator);
  * it does NOT run pytest — it pins the ARTEFACTS; the repo's own test run
    executes them.

CPU-only. No GPU. No network. No writes (every mutation is made on an in-memory
COPY; the sidecar and the tree are never left mutated).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SIDECAR = REPO / "docs" / "wf_a44_verify_receipts.locators.json"
RECEIPT = REPO / "docs" / "wf_a44_verify_receipts.md"
PRODUCER_PATH = REPO / "molmetal" / "scripts" / "wf269a_receipt_locator_refresh.py"


# --------------------------------------------------------------------------- #
# 0. THE PRODUCER — loaded with `importlib` as a STANDALONE module.
#
#    The producer is a SCRIPT with a `main()`, not an importable package API.
#    It is loaded BY PATH so that importing it does not drag the whole
#    `molmetal` package (and its import-time banner) into a guard that needs
#    nothing but `re`, `json`, `hashlib` and `subprocess`.  The guard CALLS the
#    producer's resolution functions — `resolve`, `count_ge_verdict`,
#    `_file_paths`, `_marker_paths_in_dir` — rather than copying their logic:
#    a copy drifts silently (the repo's own §6.269.35.3 lesson).
# --------------------------------------------------------------------------- #
def _load_producer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "wfa269a_receipt_locator_refresh_under_test", PRODUCER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the producer at {PRODUCER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROD = _load_producer()

# --------------------------------------------------------------------------- #
# 1. THE FIELD CLASSES — the guard's SCOPE, stated in one place.
#
#    ASSERTED          — a mutation MUST fail the guard.  `id`, `file`,
#                        `expect`, `symbol`, `said`, `said_kind`, `floor`,
#                        `section` (and `error`, which is set only when a
#                        probe's file is unreadable).  `ok` is ALSO listed here
#                        in the spec, but it is enforced through the
#                        ASSERTED-VERDICT channel below — a mutation to it
#                        fails the guard, and the mechanism that catches it
#                        is "recorded verdict != live verdict".
#    ASSERTED-VERDICT  — the guard RE-MEASURES the live tree and asserts the
#                        RECORDED verdict EQUALS the LIVE verdict.  It does
#                        NOT assert the verdict is True.
#                          * `ok`      — on EVERY row.
#                          * `said_ok` — on every row EXCEPT a row whose
#                            `said_kind == 'line'` (a LOCATOR: it drifts on
#                            every unrelated edit above the symbol) and EXCEPT
#                            a row whose `said_ok` is None (no machine-
#                            checkable claim to be consistent about).
#    RESOLUTION        — `count`, `lines`, `files`, `exists`, `note`,
#                        `said_claim`, `said_count`, `said_lines`, and the
#                        `said_ok` of a `said_kind == 'line'` row.
#                        Informational drift only, NEVER compared for
#                        equality.
# --------------------------------------------------------------------------- #
ASSERTED_FIELDS: tuple[str, ...] = (
    "id",
    "file",
    "expect",
    "symbol",
    "said",
    "said_kind",
    "floor",
    "section",
    "error",
)
VERDICT_FIELDS: tuple[str, ...] = ("ok", "said_ok")

#: The receipt's ASSERTED table (the producer's `render_table` header).
CLAIM_TABLE_HEADER = (
    "id", "expect", "the file it is about", "the symbol",
    "the receipt's claim", "floor", "the claim holds", "the said kind",
    "the said verdict",
)
#: The receipt's RESOLUTION table (a dated snapshot — never asserted).
RESOLUTION_TABLE_HEADER = ("id", "measured count", "resolved lines", "files", "exists")

#: A `said` that carries a parenthesised BREAKDOWN the probe does not check.
_BREAKDOWN_RE = re.compile(r"\([^)]*\)")
#: Staleness markers in the receipt's own prose.
_SNAPSHOT_RE = re.compile(r"(?i)dated snapshot")
_STALENESS_RE = re.compile(r"(?i)(will move again|drift|stale|never a claim)")


# --------------------------------------------------------------------------- #
# 2. Small helpers.
# --------------------------------------------------------------------------- #
def _disk_sidecar() -> dict[str, Any]:
    assert SIDECAR.is_file(), (
        f"{SIDECAR.relative_to(REPO)} is missing — regenerate it with "
        "`python molmetal/scripts/wf269a_receipt_locator_refresh.py --write`"
    )
    return json.loads(SIDECAR.read_text(encoding="utf-8"))


def _disk_rows() -> list[dict[str, Any]]:
    return list(_disk_sidecar()["rows"])


def _copy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A DEEP COPY of a row list — every mutation in this file happens here."""
    return json.loads(json.dumps(rows, ensure_ascii=False))


def _sidecar_sha256() -> str:
    return hashlib.sha256(SIDECAR.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 2b. THE ENVELOPE PIN — `rows_sha256` MUST BE CONSUMED, not merely STORED.
#
#     Wf-A269 wave-32d (2026-11-07).  Until now NOTHING read the sidecar's
#     `rows_sha256`: this guard had 0 hits for it and never imported the
#     producer's `canonical_rows_bytes`.  A stored integrity field with no
#     consumer is NOT an integrity field — it could not fail, and a hand-edit
#     of the sidecar was invisible to the guard.  (MEASURED by the wave-32d
#     adversarial refutation: overwriting the stored pin with "0"*64 left the
#     guard's failure count at 0 and BOTH envelope tests green.)
#
#     WHY THIS CHANNEL EXISTS EVEN THOUGH THE GUARD DELIBERATELY DOES NOT
#     ASSERT THE RESOLUTION FIELDS.  DESIGN v5 says a derived field (line
#     numbers, byte offsets, content hashes) is REPORTED as informational
#     drift and NEVER compared for equality — asserting it would be the
#     treadmill.  That rule leaves exactly one hole: a HAND-EDIT of a
#     RESOLUTION field.  The pin is the control for that hole, and it is a
#     DIFFERENT question from the one the guard refuses to ask: it asserts
#     that the DOCUMENT is exactly what the producer wrote (its integrity),
#     not that a value is still fresh (a line number still correct).  A
#     producer REWRITE updates rows AND pin together, so the pin never
#     creates the treadmill; only a hand-edit breaks it.
#
#     THE CANONICALISATION IS IMPORTED, NOT RE-IMPLEMENTED.  The rows carry
#     non-ASCII characters, so `ensure_ascii=True` (Python's DEFAULT) hashes
#     DIFFERENTLY from `ensure_ascii=False` — MEASURED 2026-11-07 on this
#     repo's own sidecar: 56 non-ASCII chars inside the hashed rows
#     (66 in the whole FILE — two different populations, one label), so the
#     two flags give 22 428 B / `02b151b3…` (correct) vs 22 641 B /
#     `586a5190…` (wrong).  The producer owns the definition; the guard
#     CALLS it.  A re-implementation here would be a second serialisation
#     contract that drifts silently (AGENTS.md 6.269.35.3).
# --------------------------------------------------------------------------- #
def envelope_pin_failures(envelope: dict[str, Any]) -> list[str]:
    """Check the envelope's `rows_sha256` pin against its own `rows`.

    Returns a list of failure strings (empty == the pin HOLDS).
    """
    stored = envelope.get("rows_sha256")
    if stored is None:
        return [
            "the sidecar carries NO `rows_sha256` pin — the envelope's "
            "integrity field was REMOVED (regenerate the sidecar with the "
            "producer)"
        ]
    rows = envelope.get("rows")
    if rows is None:
        return ["the sidecar carries NO `rows` array — the pin cannot be checked"]
    recomputed = hashlib.sha256(PROD.canonical_rows_bytes(rows)).hexdigest()
    if stored != recomputed:
        return [
            f"the sidecar's `rows_sha256` pin does not match its own `rows`: "
            f"stored={stored} recomputed={recomputed}.  The sidecar was "
            "HAND-EDITED (or written under a different serialisation) — "
            "regenerate it with "
            "`molmetal/scripts/wf269a_receipt_locator_refresh.py --write`"
        ]
    return []


def _dir_fingerprint(directory: Path) -> str:
    """A cheap, stable fingerprint of a directory's CONTENTS (path + size)."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(directory)}:{p.stat().st_size}\n".encode("utf-8"))
    return h.hexdigest()


def _is_durable(row: dict[str, Any]) -> bool:
    """Is this row's `said_ok` an ASSERTED-VERDICT?

    A `line`-kind row's `said_ok` is a LOCATOR verdict: it flips on every
    unrelated edit above the symbol, so asserting it is the treadmill this
    design removes.  A row whose `said_ok` is `None` has no machine-checkable
    claim to be consistent about."""
    return row.get("said_kind") != "line" and row.get("said_ok") is not None


def _split_md_row(line: str) -> list[str]:
    """Split a markdown table row into its cells.

    The producer escapes a literal `|` inside a cell as ``\\|``, so the split
    must not treat an escaped pipe as a cell boundary."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _md_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown table in `text` as `(header_cells, [row_cells, ...])`."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            block: list[str] = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            if len(block) >= 2:
                header = _split_md_row(block[0])
                rows = [
                    _split_md_row(b)
                    for b in block[1:]
                    if not set(b.strip()) <= set("|-: ")
                ]
                tables.append((header, rows))
            i = j
            continue
        i += 1
    return tables


def _find_table(text: str, header: tuple[str, ...]) -> list[list[str]] | None:
    """The rows of the ONE table whose header matches, or None."""
    hits = [rows for hdr, rows in _md_tables(text) if tuple(hdr) == header]
    if len(hits) != 1:
        return None
    return hits[0]


# --------------------------------------------------------------------------- #
# 3. THE THREE-PART PREDICATE — THE GUARD'S OWN IMPLEMENTATION.
#
#    This is deliberately NOT a call to `PROD.count_ge_verdict`: it is the
#    SECOND implementation that the cross-check below compares against the
#    producer's FIRST one.  The raw measurements come from the producer's
#    RESOLUTION PRIMITIVES (`_grep_in_dir` for the live hit count,
#    `_file_paths` for the stored units, `_marker_paths_in_dir` for the live
#    marker set) so that the RESOLUTION logic is not copied; the CONJUNCTION
#    and the ARITHMETIC are the guard's own.  The live count is therefore
#    MEASURED HERE, not read back out of the producer's row — a guard that
#    reuses the producer's number is not an independent implementation.
#
#    `floor` is a parameter because the guard enforces the predicate against
#    the sidecar's OWN recorded floor (that is the claim it is checking) while
#    the CROSS-CHECK runs the same code at the producer's ratified floor (that
#    is the like-for-like comparison).
# --------------------------------------------------------------------------- #
def three_part_predicate(
    spec: dict[str, Any],
    stored_row: dict[str, Any],
    floor: int,
) -> dict[str, Any]:
    """(a) SHRINKAGE ∧ (b) TAMPERING ∧ (c) RELOCATION."""
    rel = spec["file"]
    token = spec.get("symbol", "")
    stored_files = list(stored_row.get("files") or [])
    stored_lines = list(stored_row.get("lines") or [])
    missing: list[Any] = []

    # The live count, MEASURED HERE (not read out of the producer's row).
    if rel.endswith("/"):
        hits = PROD._grep_in_dir(rel, token)
        live_count = sum(c for _, c in hits)
    else:
        src_live = PROD._read(rel)
        live_count = sum(1 for ln in (src_live or []) if token in ln)

    # The STORED units — the paths (dir probe) or lines (file probe) the
    # receipt's count claim was about.  A stored entry carries a " (n)"
    # hit-count suffix, which must be stripped.
    if stored_files:
        units = list(PROD._file_paths(stored_files))
        # (c) is MEMBERSHIP in the live marker-path set — NOT os.path.exists.
        live_paths = PROD._marker_paths_in_dir(rel, token)
        missing = [p for p in units if p not in live_paths]
        c_relocation = not missing
    else:
        units = list(stored_lines)
        src = PROD._read(rel)
        c_relocation = True
        if src is not None:
            for ln in units:
                if not (1 <= ln <= len(src)) or token not in src[ln - 1]:
                    c_relocation = False
                    missing.append(ln)
                    break

    a_shrinkage = live_count >= floor
    b_tamper = len(units) >= floor
    return {
        "floor": floor,
        "a_shrinkage": bool(a_shrinkage),
        "b_tamper": bool(b_tamper),
        "c_relocation": bool(c_relocation),
        "verdict": bool(a_shrinkage and b_tamper and c_relocation),
        "live_count": live_count,
        "stored_units": len(units),
        "missing": missing,
    }


# --------------------------------------------------------------------------- #
# 4. THE EVALUATION — the guard's verdict, as a FUNCTION of a row list.
#
#    Taking the rows as an ARGUMENT is what lets every mutation test prove
#    the guard bites: the real artefact is never touched.
# --------------------------------------------------------------------------- #
def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a sidecar's ROWS against the LIVE tree.

    Returns:
      failures       — the CLAIM / VERDICT violations (the guard's verdict)
      drift          — the producer's `_diff` channel (INFORMATIONAL only)
      uncertifiable  — rows the `said` probe CANNOT certify (a breakdown)
      lower_bound    — rows whose `said` verdict is a lower bound
      counts         — the population bookkeeping
    """
    failures: list[str] = []
    disk_by_id: dict[Any, dict[str, Any]] = {}
    for r in rows:
        rid = r.get("id")
        if rid in disk_by_id:
            failures.append(f"the sidecar carries a DUPLICATE row id {rid!r}")
        disk_by_id[rid] = r
    spec_by_id = {s["id"]: s for s in PROD.SPECS}

    missing = sorted(set(spec_by_id) - set(disk_by_id))
    extra = sorted((k for k in disk_by_id if k not in spec_by_id), key=str)
    if missing or extra:
        failures.append(
            "the sidecar's row-id set does not match the producer's: "
            f"MISSING {missing}; EXTRA {extra}"
        )

    # The LIVE re-measurement, with the GIVEN rows as the stored rows — so a
    # tampered copy really drives the `count_ge` predicate's (b) and (c).
    fresh_rows = PROD.resolve_all(list(rows))
    fresh_by_id = {r["id"]: r for r in fresh_rows}

    for rid, spec in spec_by_id.items():
        row = disk_by_id.get(rid)
        if row is None:
            continue
        fresh = fresh_by_id[rid]

        # (1) ASSERTED — the CLAIM fields, compared against the SPEC-derived
        #     live resolution (the code is the authority, not the sidecar).
        for field in ASSERTED_FIELDS:
            got, want = row.get(field), fresh.get(field)
            if got != want:
                failures.append(
                    f"{rid}: CLAIM field {field!r}: sidecar={got!r} but the "
                    f"producer's live resolution says {want!r}"
                )

        # (2) ASSERTED-VERDICT — `ok`, on every row.
        if row.get("ok") != fresh.get("ok"):
            failures.append(
                f"{rid}: the RECORDED verdict ok={row.get('ok')!r} does NOT "
                f"equal the LIVE verdict ok={fresh.get('ok')!r} "
                f"[{row.get('file')}]"
            )

        # (3) ASSERTED-VERDICT — `said_ok`, on every DURABLE row.
        if _is_durable(row) and row.get("said_ok") != fresh.get("said_ok"):
            failures.append(
                f"{rid}: the RECORDED said_ok={row.get('said_ok')!r} does NOT "
                f"equal the LIVE verdict {fresh.get('said_ok')!r} "
                f"(kind={row.get('said_kind')!r})"
            )

        # (4) THE THREE-PART PREDICATE — enforced, then CROSS-CHECKED.
        if spec.get("expect") == "count_ge":
            recorded_floor = row.get("floor")
            enforced_floor = (
                recorded_floor
                if isinstance(recorded_floor, int) and not isinstance(recorded_floor, bool)
                else int(spec["floor"])
            )
            mine = three_part_predicate(spec, row, enforced_floor)
            # The cross-check runs at the producer's ratified floor so that
            # the two implementations are compared LIKE FOR LIKE.
            mine_at_spec_floor = three_part_predicate(spec, row, int(spec["floor"]))
            theirs = PROD.count_ge_verdict(spec, row)
            for part in ("a_shrinkage", "b_tamper", "c_relocation", "verdict"):
                if bool(mine_at_spec_floor[part]) != bool(theirs.get(part)):
                    failures.append(
                        f"{rid}: the guard's OWN three-part predicate and the "
                        f"PRODUCER'S DISAGREE on {part} — guard="
                        f"{mine_at_spec_floor[part]!r} producer={theirs.get(part)!r}. "
                        f"The guard does not paper over a disagreement."
                    )
            if not mine["verdict"]:
                failures.append(
                    f"{rid}: the three-part predicate FAILS — floor={mine['floor']} "
                    f"live_count={mine['live_count']} stored_units={mine['stored_units']} "
                    f"a_shrinkage={mine['a_shrinkage']} b_tamper={mine['b_tamper']} "
                    f"c_relocation={mine['c_relocation']} missing={mine['missing']}"
                )

    # The drift channel — the PRODUCER's own classifier, never a second copy.
    drift = PROD._diff(list(rows), fresh_rows)

    # THE PROBE'S BLIND SPOT — reported, never certified.
    lower_bound: list[str] = []
    uncertifiable: list[str] = []
    for rid in sorted((r for r in disk_by_id), key=str):
        row = disk_by_id[rid]
        kind = row.get("said_kind")
        if kind in ("hits", "tests", "bytes"):
            lower_bound.append(rid)
            said = row.get("said") or ""
            if _BREAKDOWN_RE.search(said):
                uncertifiable.append(
                    f"{rid}: the receipt's claim {said!r} carries a BREAKDOWN the "
                    f"`{kind}` probe does not check — the verdict is measured >= "
                    f"claimed (a LOWER BOUND), so this row's claim is NOT certified "
                    f"by a green verdict"
                )

    durable = [r for r in rows if _is_durable(r)]
    return {
        "failures": failures,
        "drift": drift,
        "uncertifiable": uncertifiable,
        "lower_bound": lower_bound,
        "counts": {
            "rows": len(rows),
            "durable_said_rows": len(durable),
            "durable_false_verdicts": sorted(
                r["id"] for r in durable if r.get("said_ok") is False
            ),
            "lower_bound_rows": len(lower_bound),
            "uncertifiable_rows": len(uncertifiable),
        },
    }


# --------------------------------------------------------------------------- #
# 5. THE TESTS.
# --------------------------------------------------------------------------- #
def test_every_receipt_claim_still_holds():
    """The receipt's claims, re-resolved against the tree, all still hold."""
    rows = _disk_rows()
    stored = {r["id"]: r for r in rows}
    failures = []
    for spec in PROD.SPECS:
        row = PROD.resolve(spec, stored.get(spec["id"]))
        if not row["ok"]:
            failures.append(
                f"{row['id']}: {row['file']} :: {row.get('symbol')!r} "
                f"(expect {row['expect']}, got {row.get('count')} hit(s) at "
                f"{row.get('lines') or row.get('files')})"
            )
    assert not failures, (
        "the receipt's claims no longer hold — re-verify and append a dated "
        "correction to docs/wf_a44_verify_receipts.md:\n  " + "\n  ".join(failures)
    )


def test_sidecar_claims_and_verdicts_match_the_live_tree():
    """THE GUARD.  Every CLAIM field and every RECORDED verdict still matches
    the LIVE tree; every RESOLUTION field is reported, never asserted."""
    rows = _disk_rows()
    rep = evaluate(rows)
    print(
        f"[receipt-locator guard] rows={rep['counts']['rows']} "
        f"durable_said_rows={rep['counts']['durable_said_rows']} "
        f"durable_false_verdicts={rep['counts']['durable_false_verdicts']} "
        f"uncertifiable={rep['counts']['uncertifiable_rows']} "
        f"drift_entries={len(rep['drift'])}"
    )
    for line in rep["drift"]:
        print(f"[receipt-locator guard] RESOLUTION DRIFT (informational): {line}")
    for line in rep["uncertifiable"]:
        print(f"[receipt-locator guard] NOT CERTIFIABLE: {line}")
    assert not rep["failures"], (
        "the sidecar's CLAIMS or VERDICTS no longer match the live tree:\n  "
        + "\n  ".join(rep["failures"])
    )


def test_the_envelopes_rows_sha256_pin_holds_against_its_own_rows():
    """THE PIN IS CONSUMED — `rows_sha256` is checked against `rows`.

    The sidecar's envelope stores a sha256 over its own `rows` array.  This
    test is that field's ONLY consumer (until wave-32d it had none, and a
    hand-edited sidecar was invisible to the guard).  The pin is checked with
    the PRODUCER's own canonicalisation — never a re-implementation.
    """
    env = _disk_sidecar()
    failures = envelope_pin_failures(env)
    assert not failures, (
        "the sidecar's `rows_sha256` pin does not match its own `rows`:\n  "
        + "\n  ".join(failures)
    )


def test_the_pin_channel_bites_on_both_a_pin_tamper_and_a_row_tamper():
    """The pin channel must FAIL on a hand-edit — in BOTH directions.

    (a) the PIN itself is overwritten  -> the channel must report it.
    (b) a RESOLUTION field inside a row is overwritten -> the channel must
        report it EVEN THOUGH the guard deliberately does NOT assert the
        RESOLUTION fields for equality (DESIGN v5).  This is the whole point
        of the pin: it is the control over the half the guard stopped
        asserting, and it is the ONLY thing that can catch a hand-edit of
        that half.

    Both mutations happen on IN-MEMORY COPIES.  The sidecar and the tree are
    never left mutated.
    """
    env = _disk_sidecar()
    assert not envelope_pin_failures(env), "the pin channel is RED on the real sidecar"

    # (a) the PIN itself
    tampered_pin = json.loads(json.dumps(env))
    tampered_pin["rows_sha256"] = "0" * 64
    pin_failures = envelope_pin_failures(tampered_pin)
    assert pin_failures, (
        "the pin channel did NOT fail when the stored `rows_sha256` was "
        "overwritten with 64 zeros — the channel is DEAD"
    )
    assert "rows_sha256" in pin_failures[0], (
        f"the pin failure does not name the field:\n  {pin_failures[0]}"
    )

    # (b) a RESOLUTION field — the half the guard does NOT assert for equality
    tampered_rows = json.loads(json.dumps(env))
    target = next(
        r for r in tampered_rows["rows"] if r.get("lines") is not None
    )
    target["lines"] = [999999]
    row_failures = envelope_pin_failures(tampered_rows)
    assert row_failures, (
        "the pin channel did NOT fail when a row's RESOLUTION field `lines` "
        "was hand-edited — the control over the half the guard does not "
        "assert is DEAD"
    )

    # ... and the control: an untouched copy is still green.
    assert not envelope_pin_failures(
        json.loads(json.dumps(env))
    ), "the pin channel is RED on an untouched copy — it is broken"


def test_the_sidecars_envelope_declares_the_producers_field_classes():
    """The sidecar's declared `claim_fields` / `resolution_fields` are the
    producer's own tuples — so the guard's scope cannot drift from the
    producer's silently."""
    env = _disk_sidecar()
    assert sorted(env["claim_fields"]) == sorted(PROD.CLAIM_FIELDS), (
        "the sidecar's declared claim_fields do not match the producer's "
        f"CLAIM_FIELDS: {env['claim_fields']} vs {list(PROD.CLAIM_FIELDS)}"
    )
    assert sorted(env["resolution_fields"]) == sorted(PROD.RESOLUTION_FIELDS), (
        "the sidecar's declared resolution_fields do not match the producer's "
        f"RESOLUTION_FIELDS: {env['resolution_fields']} vs {list(PROD.RESOLUTION_FIELDS)}"
    )
    assert env["schema"] == "wf269a.receipt_locators.v1", "the sidecar schema drifted"


def test_sidecar_coverage_bound_is_true():
    """The sidecar's stated coverage matches reality."""
    side = _disk_sidecar()
    cov = side["coverage"]
    assert cov["covered_probes"] == len(PROD.SPECS), "the coverage probe count is wrong"
    assert cov["covered_files"] == len({s["file"] for s in PROD.SPECS}), "the coverage file count is wrong"
    assert cov["not_covered"], "the coverage bound must state what it does NOT cover"
    assert cov["does_not_run_pytest"], "the coverage bound must state the pytest limit"


def test_receipt_embeds_the_sidecars_claim_columns():
    """T7 — the receipt's embedded table carries the sidecar's CLAIM columns.

    The RESOLUTION half is a DATED SNAPSHOT: the receipt must LABEL it as such
    and carry a staleness notice.  Its VALUES are never asserted — they drift
    on every unrelated edit, and asserting them is the treadmill."""
    text = RECEIPT.read_text(encoding="utf-8")
    rows = _disk_rows()

    table = _find_table(text, CLAIM_TABLE_HEADER)
    assert table is not None, (
        "the receipt no longer carries the producer's CLAIM table "
        f"({list(CLAIM_TABLE_HEADER)}) — regenerate the receipt section from "
        "the producer"
    )
    assert len(table) == len(rows), (
        f"the receipt's CLAIM table has {len(table)} row(s) but the sidecar has "
        f"{len(rows)}"
    )

    for row, cells in zip(rows, table):
        assert len(cells) == len(CLAIM_TABLE_HEADER), (
            f"the receipt's row {row['id']} has {len(cells)} cell(s), expected "
            f"{len(CLAIM_TABLE_HEADER)}"
        )
        floor = row.get("floor")
        expected = [
            str(row.get("id")),
            f"`{row.get('expect')}`",
            f"`{row.get('file')}`",
            f"`{PROD._esc(row.get('symbol') or '(file)')}`",
            PROD._esc(row.get("said") or "—"),
            str(floor) if floor is not None else "—",
            "**TRUE**" if row.get("ok") else "**FALSE**",
            f"`{row.get('said_kind') or 'none'}`",
        ]
        for col, want, got in zip(CLAIM_TABLE_HEADER[:-1], expected, cells[:-1]):
            assert got == want, (
                f"the receipt's CLAIM column {col!r} for {row['id']} is {got!r} "
                f"but the sidecar says {want!r}"
            )

        # The LAST column is the `said` VERDICT — an ASSERTED-VERDICT on a
        # durable row, RESOLUTION on a `line` row (a locator, which drifts on
        # every unrelated edit and must NOT be asserted).
        verdict_cell = cells[-1]
        if row.get("said_kind") == "line":
            assert "informational" in verdict_cell.lower(), (
                f"the receipt must label {row['id']}'s said verdict as "
                f"informational (kind=line), got {verdict_cell!r}"
            )
        elif row.get("said_ok") is None:
            assert verdict_cell == PROD._said_verdict_cell(row), (
                f"the receipt's said-verdict cell for {row['id']} is "
                f"{verdict_cell!r}, expected {PROD._said_verdict_cell(row)!r}"
            )
        else:
            assert verdict_cell == PROD._said_verdict_cell(row), (
                f"the receipt's said-verdict cell for {row['id']} is "
                f"{verdict_cell!r}, expected {PROD._said_verdict_cell(row)!r}"
            )

    # The RESOLUTION half must be PRESENT, LABELLED as a dated snapshot, and
    # carry a staleness notice.  Its VALUES are never asserted.
    resolution = _find_table(text, RESOLUTION_TABLE_HEADER)
    assert resolution is not None, (
        "the receipt no longer carries the producer's RESOLUTION table — a "
        "resolution column printed among the claims reads as a claim"
    )
    assert _SNAPSHOT_RE.search(text), (
        "the receipt must label its RESOLUTION half as a DATED SNAPSHOT"
    )
    tail = text[text.index("| id | measured count | resolved lines | files | exists |"):]
    tail = tail[len(RESOLUTION_TABLE_HEADER) + 4:]
    assert _STALENESS_RE.search(tail[:4000]), (
        "the receipt must carry a staleness notice next to its RESOLUTION table"
    )


def test_receipt_preserves_its_original_body():
    """§5.5 rule #6: the pre-2026-11-02 body is preserved VERBATIM (the catalogue
    cites this file by line at :22334 / :22628 / :22634)."""
    text = RECEIPT.read_text(encoding="utf-8")
    marker = "# ⚠️ RE-VERIFICATION — 2026-11-02 (the main thread, Wf-A269)"
    assert marker in text, "the dated re-verification section is missing"
    head = text.split(marker)[0]
    for s in (
        "grep -c \"Wf-A44\"",
        "tmqm_split_kind: scaffold",
        "17 docs-only LEGACY OVERRIDE",
        "110 PASS / 3 FAIL / 10 SKIP / 124 total",
        "--self-test",
        "M15 §6.66f",
    ):
        assert s in head, f"a superseded rendering was lost from the receipt body: {s!r}"


def test_the_gate_actually_bites():
    """Self-test: a broken probe must resolve to ok=False in BOTH directions."""
    # (a) a `contains` probe for a symbol that does not exist
    broken_contains = {
        "id": "SELFTEST-contains", "file": "molmetal/data/tmqm.py",
        "expect": "contains", "symbol": "THIS_SYMBOL_DOES_NOT_EXIST_9f3a",
    }
    assert PROD.resolve(broken_contains)["ok"] is False, (
        "the gate does NOT fail on a missing symbol — it cannot detect a rename/delete"
    )

    # (b) an `absent` probe for a symbol that DOES exist (the retired-flag shape)
    broken_absent = {
        "id": "SELFTEST-absent", "file": "molmetal/data/tmqm.py",
        "expect": "absent", "symbol": "TRANSITION_METALS",
    }
    assert PROD.resolve(broken_absent)["ok"] is False, (
        "the gate does NOT fail when a symbol that must be gone is still present"
    )

    # (c) a `file_exists` probe for a path that does not exist
    broken_file = {
        "id": "SELFTEST-file", "file": "molmetal/scripts/does_not_exist_9f3a.py",
        "expect": "file_exists", "symbol": "",
    }
    assert PROD.resolve(broken_file)["ok"] is False, (
        "the gate does NOT fail on a missing file"
    )

    # (d) and the control: a genuinely-true probe still passes
    control = {
        "id": "SELFTEST-control", "file": "molmetal/data/tmqm.py",
        "expect": "contains", "symbol": "TRANSITION_METALS",
    }
    assert PROD.resolve(control)["ok"] is True, "the gate fails on a TRUE claim — it is broken"


# --------------------------------------------------------------------------- #
# THE INDEPENDENCE CROSS-CHECK IS NOT DEAD.  Two implementations that AGREE
# are evidence — but a cross-check that cannot report a disagreement is a
# second copy of the thing it was supposed to audit.  This test forces the
# producer's verdict to disagree and requires the guard to SAY SO.
# --------------------------------------------------------------------------- #
def test_the_independence_cross_check_can_report_a_disagreement():
    """The guard's own predicate and the producer's must be compared, and a
    disagreement must FAIL the guard with the differing arm named."""
    original = PROD.count_ge_verdict

    def disagreeing(spec, stored_row):
        verdict = dict(original(spec, stored_row))
        verdict["c_relocation"] = not verdict["c_relocation"]
        return verdict

    PROD.count_ge_verdict = disagreeing
    try:
        rep = evaluate(_disk_rows())
    finally:
        PROD.count_ge_verdict = original

    disagreements = [f for f in rep["failures"] if "DISAGREE" in f]
    assert disagreements, (
        "the guard did NOT report a forced disagreement between its own "
        "three-part predicate and the producer's — the cross-check is DEAD"
    )
    assert any("c_relocation" in f for f in disagreements), (
        "the disagreement report does not name the differing arm:\n  "
        + "\n  ".join(disagreements)
    )
    assert not evaluate(_disk_rows())["failures"], (
        "the guard is RED after the producer's verdict was restored"
    )


# --------------------------------------------------------------------------- #
# T1 — CLAIM TAMPER FAILS.  A mutation to any ASSERTED field must fail the
#       guard, and must be ATTRIBUTED to that field.
# --------------------------------------------------------------------------- #
def test_claim_tamper_fails_the_guard():
    """T1 — a mutation to any ASSERTED field MUST fail the guard."""
    base = _disk_rows()
    before = _sidecar_sha256()
    assert not evaluate(base)["failures"], (
        "the guard is already RED on the unmutated sidecar — the mutation "
        "proofs below would be vacuous"
    )

    # (field, row id, new value) — one mutation at a time, on a COPY.
    mutations = [
        ("id", "M1", "M1-TAMPERED"),
        ("file", "M1", "molmetal/data/tmqm.py"),
        ("expect", "M1", "absent"),
        ("symbol", "M1", "SOME_OTHER_SYMBOL"),
        ("said", "M1", "L99999"),
        ("said_kind", "M1", "hits"),
        ("floor", "S671d-scaffold", 28),
        ("ok", "M1", False),
        ("section", "M1", "M1 §9.99z"),
        # a DURABLE row's recorded verdict (kind=presence, said_ok=False today)
        ("said_ok", "M15c", True),
    ]

    for field, row_id, value in mutations:
        rows = _copy_rows(base)
        target = next(r for r in rows if r["id"] == row_id)
        assert field in target, f"the fixture row {row_id} has no {field!r} field"
        target[field] = value
        rep = evaluate(rows)
        assert rep["failures"], (
            f"TAMPERING {field!r} on {row_id} did NOT fail the guard — a guard "
            f"that cannot fail is worse than no guard"
        )
        joined = "\n  ".join(rep["failures"])
        assert str(value) in joined, (
            f"the failure for the {field!r} tamper on {row_id} does not NAME "
            f"the tampered value {value!r}:\n  {joined}"
        )
        assert _sidecar_sha256() == before, "the real sidecar was mutated"


# --------------------------------------------------------------------------- #
# T2 — THE NEGATIVE CONTROL.  RESOLUTION drift is REPORTED, never ASSERTED.
#       This is the test that proves the resolution/claim split is real.
# --------------------------------------------------------------------------- #
def test_resolution_drift_does_not_fail_the_guard():
    """T2 — RESOLUTION drift (lines +32, count +1) must NOT fail the guard.

    THE NEGATIVE CONTROL.  Without it, "the guard passes" could just mean
    "the guard compares nothing"."""
    base = _disk_rows()
    before = _sidecar_sha256()

    for row_id, has_lines in (("M13", True), ("S671d-scaffold", False)):
        rows = _copy_rows(base)
        target = next(r for r in rows if r["id"] == row_id)
        if has_lines:
            assert target["lines"], f"{row_id} must carry resolved lines"
            target["lines"] = [int(ln) + 32 for ln in target["lines"]]
        target["count"] = int(target["count"]) + 1

        rep = evaluate(rows)
        assert not rep["failures"], (
            f"RESOLUTION drift on {row_id} FAILED the guard — resolution drift "
            f"is informational only:\n  " + "\n  ".join(rep["failures"])
        )
        assert any(row_id in d for d in rep["drift"]), (
            f"the drift channel did not report the {row_id} mutation — a "
            f"resolution field that drifts UNREPORTED is silent rot"
        )
        assert _sidecar_sha256() == before, "the real sidecar was mutated"


# --------------------------------------------------------------------------- #
# T3 — THE THREE-PART PREDICATE FIRES.
# --------------------------------------------------------------------------- #
def test_the_three_part_predicate_fires():
    """T3 — each arm of the three-part predicate must be able to FAIL."""
    base = _disk_rows()
    before = _sidecar_sha256()
    row_id = "S671d-scaffold"
    spec = next(s for s in PROD.SPECS if s["id"] == row_id)

    # (i) the FLOOR dropped below the count that was measured with it.
    #     A lowered floor can NEVER fail a `>= floor` predicate (every arm
    #     gets easier), so this arm is caught by the CLAIM comparison — the
    #     recorded floor must equal the producer's ratified floor, and the
    #     ratified floor IS the measured count.
    rows = _copy_rows(base)
    target = next(r for r in rows if r["id"] == row_id)
    live_count = int(target["count"])
    target["floor"] = int(target["floor"]) - 1
    rep = evaluate(rows)
    assert rep["failures"], "(i) dropping the floor did NOT fail the guard"
    joined = "\n  ".join(rep["failures"])
    assert "floor" in joined, (
        f"(i) the failure does not name the floor:\n  {joined}"
    )

    # (ii) a stored file deleted so that len(files) < floor — TAMPERING.
    rows = _copy_rows(base)
    target = next(r for r in rows if r["id"] == row_id)
    assert len(target["files"]) >= int(target["floor"]), "fixture precondition"
    target["files"] = list(target["files"])[:-1]
    rep = evaluate(rows)
    assert rep["failures"], "(ii) deleting a stored file did NOT fail the guard"
    assert any("b_tamper" in f for f in rep["failures"]), (
        "(ii) the predicate's TAMPERING arm did not fire:\n  "
        + "\n  ".join(rep["failures"])
    )

    # (iii) the marker MOVED — a stored path that no longer carries it.
    rows = _copy_rows(base)
    target = next(r for r in rows if r["id"] == row_id)
    live_paths = PROD._marker_paths_in_dir(spec["file"], spec.get("symbol", ""))
    moved_to = "configs/blk_c5_head_ab_dmpnn.yaml"
    assert moved_to not in live_paths, (
        "the fixture path now carries the marker — pick another one"
    )
    target["files"][0] = f"{moved_to} (1)"
    rep = evaluate(rows)
    assert rep["failures"], "(iii) moving the marker did NOT fail the guard"
    assert any("c_relocation" in f for f in rep["failures"]), (
        "(iii) the predicate's RELOCATION arm did not fire:\n  "
        + "\n  ".join(rep["failures"])
    )

    assert _sidecar_sha256() == before, "the real sidecar was mutated"


# --------------------------------------------------------------------------- #
# T4 — THE said RE-VERIFICATION: VERDICT CONSISTENCY, NOT TRUTH.
# --------------------------------------------------------------------------- #
def test_said_verdict_consistency_is_asserted_not_truth():
    """T4 — the recorded verdict must EQUAL the live verdict.  It is NOT
    asserted to be `True`: four receipt claims are legitimately FALSE today
    (M3a, M15c, M15d, S671d-legacy) and the receipt carries the truth about
    them.  The guard's job is to notice that the TRUTH VALUE MOVED."""
    rows = _disk_rows()
    fresh = {r["id"]: r for r in PROD.resolve_all(list(rows))}
    durable = [r for r in rows if _is_durable(r)]
    assert durable, "there must be at least one DURABLE row to be consistent about"

    for row in durable:
        assert row["said_ok"] == fresh[row["id"]]["said_ok"], (
            f"{row['id']}: the recorded said_ok={row['said_ok']!r} does NOT equal "
            f"the live verdict {fresh[row['id']]['said_ok']!r}"
        )

    false_rows = sorted(r["id"] for r in durable if r["said_ok"] is False)
    assert false_rows, (
        "the fixture must contain at least one durable row whose verdict is "
        "legitimately False — otherwise this test cannot show that the guard "
        "does NOT assert truth"
    )
    rep = evaluate(rows)
    assert not rep["failures"], (
        "the guard failed on a legitimately-FALSE durable verdict — the "
        "verdict-consistency check must not assert truth:\n  "
        + "\n  ".join(rep["failures"])
    )
    for rid in false_rows:
        assert not any(rid in f for f in rep["failures"]), (
            f"{rid} is legitimately False and must NOT be a guard failure"
        )


# --------------------------------------------------------------------------- #
# T4b — THE VERDICT-CONSISTENCY MUTATIONS.  Both must FAIL the guard.
# --------------------------------------------------------------------------- #
def test_said_verdict_consistency_mutations_fail():
    """T4b — (i) flipping a recorded verdict and (ii) flipping the LIVE truth
    must both fail the guard."""
    base = _disk_rows()
    before = _sidecar_sha256()

    # (i) flip a DURABLE row's recorded said_ok in a COPY of the sidecar.
    rows = _copy_rows(base)
    target = next(r for r in rows if r["id"] == "M15c")
    assert target["said_kind"] != "line" and target["said_ok"] is False, (
        "M15c must be a durable row whose recorded verdict is False"
    )
    target["said_ok"] = True
    rep = evaluate(rows)
    assert rep["failures"], (
        "(i) flipping a durable row's recorded said_ok did NOT fail the guard"
    )
    assert any("said_ok" in f for f in rep["failures"]), (
        "(i) the failure does not name said_ok:\n  " + "\n  ".join(rep["failures"])
    )
    assert _sidecar_sha256() == before, "the real sidecar was mutated"

    # (ii) change the LIVE TREE so a durable claim's truth value flips:
    #      create one of the two paths the receipt says NEVER EXISTED.
    target_path = REPO / "molmetal" / "scripts" / "fpocket_wrapper.py"
    assert not target_path.exists(), (
        f"{target_path} already exists — M15c's claim would not be FALSE"
    )
    sidecar_before = _sidecar_sha256()
    scripts_before = _dir_fingerprint(REPO / "molmetal" / "scripts")
    try:
        target_path.write_text(
            "# TEMPORARY fixture created by the receipt-locator guard's "
            "verdict-consistency mutation test.\n",
            encoding="utf-8",
        )
        rep = evaluate(_disk_rows())
        assert rep["failures"], (
            "(ii) creating a path the receipt says never existed did NOT fail "
            "the guard"
        )
        assert any("M15c" in f for f in rep["failures"]), (
            "(ii) the failure does not name M15c:\n  " + "\n  ".join(rep["failures"])
        )
    finally:
        target_path.unlink(missing_ok=True)
    assert not target_path.exists(), "the fixture file was left behind"
    assert _sidecar_sha256() == sidecar_before, "the real sidecar was mutated"
    assert _dir_fingerprint(REPO / "molmetal" / "scripts") == scripts_before, (
        "the tree was NOT restored byte-for-byte after the live-tree mutation"
    )


# --------------------------------------------------------------------------- #
# T4c — THE PROBE BLIND SPOT IS REPORTED, NOT CERTIFIED.
# --------------------------------------------------------------------------- #
def test_probe_blind_spot_is_reported_not_certified():
    """T4c — the `hits` verdict is a LOWER BOUND, so it cannot certify a
    claim that carries a BREAKDOWN.  The guard must NAME the rows it cannot
    certify — and a green verdict must not be presented as a certification."""
    rep = evaluate(_disk_rows())
    assert not rep["failures"], (
        "the guard is RED on the current tree — the report below would be "
        "unreadable:\n  " + "\n  ".join(rep["failures"])
    )
    joined = "\n  ".join(rep["uncertifiable"])
    assert "M2" in joined, (
        "the report does NOT name M2 — the receipt's M2 row says '6 hits "
        "(def + 5 call sites)' while the live probe prints 7 lines, and the "
        f"`hits` lower bound cannot certify that breakdown.\n  {joined}"
    )
    assert "M2" in rep["lower_bound"], (
        "M2 must be reported as a LOWER-BOUND row"
    )
    assert not any("M2" in f for f in rep["failures"]), (
        "M2 must not be a guard FAILURE — the receipt discloses it, and the "
        "guard's job is to REPORT it, not to fail on it"
    )


# --------------------------------------------------------------------------- #
# T8 — THE RESOLUTION DRIFT IS REPORTED THROUGH THE PRODUCER'S `_diff`
#       CHANNEL, AND THE VERDICT DOES NOT DEPEND ON IT.
# --------------------------------------------------------------------------- #
def test_resolution_drift_is_reported_through_the_producers_diff_channel():
    """T8 — the guard routes its drift report through the PRODUCER's `_diff`
    (one classifier in the repo, not two that drift apart) and the drift
    never changes the verdict."""
    rows = _copy_rows(_disk_rows())
    target = next(r for r in rows if r["id"] == "M13")
    target["lines"] = [int(ln) + 32 for ln in target["lines"]]
    target["count"] = int(target["count"]) + 1

    rep = evaluate(rows)
    assert rep["drift"], "the drift channel reported nothing for a real mutation"
    assert rep["drift"] == PROD._diff(list(rows), PROD.resolve_all(list(rows))), (
        "the guard does not route its drift report through the producer's "
        "_diff channel"
    )
    assert not rep["failures"], (
        "the verdict DEPENDS on the resolution drift — it must not:\n  "
        + "\n  ".join(rep["failures"])
    )
