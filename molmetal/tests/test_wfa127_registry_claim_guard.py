#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""(Wf-A127 TRACK 3) — a registry claim must be checkable against the artifact it names.

**The residual.**  Wf-A127 closed the "a document describes an artifact and
nothing checks the description" class for the four SSOT **markdown**
documents (``test_wfa127_dangling_reference_guard.py`` for dead
cross-references, ``test_wfa127_size_citation_guard.py`` for byte sizes).
Two **registry** artifacts carry the same class of claim and had no guard of
their own:

``registry/narrative_claims.yaml``
    47 claims, each naming the JSON artifact it came from and the dotted
    field path inside it.  Every claim is a statement about a file on disk.

``registry/file_provenance.json``
    which git hunks in two contaminated files were authored in the 2026-10-10
    session: hunk anchors, added-line counts, the attribution method and its
    tokens, the commit list.  Every one of those is a statement about git
    history.

**What is checkable, and what is left alone.**  A claim is checkable when the
file names an artifact that can be re-derived: a path on disk, a dotted field
path, a git hunk anchor, a commit subject, a count of regions.  A claim is
NOT checkable when it is a verdict or a rationale — the prose in ``purpose``,
``why_this_exists``, ``attribution_method``, ``CORRECTION_to_earlier_disclosure``,
``DISCLOSURE_GAP_found_by_its_own_test``, ``risk_assessment.caveat``, the
``claim`` one-liners and the ``note`` explanations in the YAML.  Those are
opinions; this guard does not touch them.

**The value comparison is delegated, not duplicated.**  Whether a claim's
``value`` still equals the number on disk is owned by
``molmetal/scripts/verify_narrative_claims.py`` and its own test.  This
guard checks what that checker cannot see — the structure around the claims
(schema, narrative pointer, artifact path, field path) and the provenance
record's git-derived statements.

**Scope.**  The two registry artifacts plus the git history they describe.
Disjoint from the two markdown guards, which cover the four SSOT documents.

**Deliberately NOT gated** (measured findings, reported to the wave instead):

  * ``risk_assessment.new_top_level_symbols_in_pre_existing_hunks`` says
    ``{"trainer": 5, "panel": 0}``.  5 is reproducible only by counting
    distinct ``def``/``class`` names at ANY indent in the not-authored added
    lines; a strict column-0 "top level" reading gives 3.  The record does not
    pin the counting rule, so the number is not machine-checkable.
  * ``tol: 1e-6`` / ``tol: 1e-9`` in the YAML parse as STRINGS (PyYAML's
    float resolver needs a dot before the exponent).  The only consumer
    coerces with ``float()``, so no number is wrong today, but the declared
    type is not the parsed type.
  * The YAML header calls ``id`` "the stable key the narrative cites"; the
    pinned narrative cites 0 of the 47 ids (the one textual hit is the source
    file name ``rdkit_validity.json`` inside a table cell).
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

#: Overridable so the guard can be run END-TO-END against a perturbed COPY in
#: /tmp (the negative test) without ever modifying the repository.
PROVENANCE_PATH = Path(
    os.environ.get("WFA127_REGISTRY_PROVENANCE")
    or REPO / "registry" / "file_provenance.json"
)
CLAIMS_PATH = Path(
    os.environ.get("WFA127_REGISTRY_CLAIMS")
    or REPO / "registry" / "narrative_claims.yaml"
)

#: The schema versions the two registry artifacts declare.
EXPECTED_CLAIMS_SCHEMA = "molmetal-narrative-claims-v1"
EXPECTED_PROVENANCE_SCHEMA = "molmetal.file_provenance.v1"

#: The YAML header's own rule: "A claim that is CAPPED, PROXY, or otherwise
#: not a direct measurement must say so in ``note``."
CAP_LIKE_RE = re.compile(r"cap|saturat|proxy|ceiling|truncat", re.IGNORECASE)

#: ``@@ -a,b +c,d @@`` — the hunk header of a unified diff.
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")

#: How close (in old-file lines) two hunks must sit to count as one "region".
PROXIMITY_WINDOW = 25

#: The labels the record's prose uses for the two contaminated files.
PROSE_FILE_LABELS = {
    "trainer": "train_cfm_yuelbond_wasserstein.py",
    "panel": "cfm_property_panel.py",
}


# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _git_ok(*args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).returncode
        == 0
    )


def _parse_hunks(diff_text: str) -> list[tuple[str, list[str]]]:
    """Every hunk of a unified diff as ``(anchor, added line texts)``."""
    hunks: list[tuple[str, list[str]]] = []
    anchor: str | None = None
    added: list[str] = []
    for line in diff_text.splitlines():
        if HUNK_RE.match(line):
            if anchor is not None:
                hunks.append((anchor, added))
            anchor = line.split("@@")[1].strip()
            added = []
            continue
        if anchor is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    if anchor is not None:
        hunks.append((anchor, added))
    return hunks


def _old_start(anchor: str) -> int:
    """The old-side start line of a hunk anchor (``-169,17 +189,183`` -> 169)."""
    return int(anchor.split(",")[0].lstrip("-"))


def _candidate_states(base_commit: str, path: str) -> list[tuple[str, str]]:
    """``(commit, blob)`` for every distinct state of ``path`` after ``base``."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    log = _git("log", "--format=%H", f"{base_commit}..HEAD", "--", path)
    for commit in log.split():
        blob = _git("rev-parse", f"{commit}:{path}").strip()
        if blob in seen:
            continue
        seen.add(blob)
        out.append((commit, blob))
    return out


#: Memoised: the history scan is the expensive part of this guard.
_STATE_CACHE: dict[
    tuple[str, str, frozenset[str]], list[tuple[str, list[tuple[str, list[str]]]]]
] = {}


def _states_with_anchors(
    base_commit: str, path: str, anchors: frozenset[str]
) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    """Every historical state of ``path`` whose diff has exactly ``anchors``."""
    key = (base_commit, path, anchors)
    if key in _STATE_CACHE:
        return _STATE_CACHE[key]
    base_blob = _git("rev-parse", f"{base_commit}:{path}").strip()
    matches: list[tuple[str, list[tuple[str, list[str]]]]] = []
    for commit, blob in _candidate_states(base_commit, path):
        hunks = _parse_hunks(_git("diff", "--unified=3", base_blob, blob))
        if frozenset(a for a, _ in hunks) == anchors:
            matches.append((commit, hunks))
    _STATE_CACHE[key] = matches
    return matches


def _dig(obj: Any, dotted: str) -> tuple[bool, Any]:
    """Resolve ``a.b.0.c`` against nested dicts/lists."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return False, None
            if not (-len(cur) <= idx < len(cur)):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def _load_provenance() -> dict[str, Any]:
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def _load_claims() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(CLAIMS_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the provenance audit
# --------------------------------------------------------------------------


def _audit_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Check every git-derivable statement in the provenance record.

    Returns a dict with one problem list per concern plus the counts that
    tell the caller how much was actually checked (non-vacuity).
    """
    base = str(record["base_commit"])
    files = record.get("files") or {}
    tokens = [str(t) for t in (record.get("attribution_tokens") or [])]

    hunk_problems: list[str] = []
    token_problems: list[str] = []
    commit_problems: list[str] = []
    prose_problems: list[str] = []

    states: dict[str, str] = {}
    diff_hunks: dict[str, list[tuple[str, list[str]]]] = {}
    checked_hunks = 0

    for path, entry in sorted(files.items()):
        recorded_authored = list(entry.get("authored_this_session") or [])
        recorded_not = list(entry.get("not_authored_this_session") or [])
        recorded = recorded_authored + recorded_not
        recorded_anchors = frozenset(str(h["anchor"]) for h in recorded)
        authored_anchors = frozenset(str(h["anchor"]) for h in recorded_authored)
        not_anchors = frozenset(str(h["anchor"]) for h in recorded_not)

        if len(recorded_anchors) != len(recorded):
            hunk_problems.append(f"{path}: the record lists a hunk anchor twice")

        matches = _states_with_anchors(base, path, recorded_anchors)
        if not matches:
            hunk_problems.append(
                f"{path}: NO historical state of this file reproduces the recorded "
                f"hunk anchors ({len(recorded_anchors)} of them) — the anchors are "
                "not git history"
            )
            continue
        if len(matches) > 1:
            hunk_problems.append(
                f"{path}: {len(matches)} historical states reproduce the recorded "
                "hunk anchors — the record is ambiguous"
            )
        commit, hunks = matches[0]
        states[path] = commit
        diff_hunks[path] = hunks
        by_anchor = {a: added for a, added in hunks}
        checked_hunks += len(hunks)

        # (1) every recorded anchor is a real hunk, with the real added count
        for h in recorded:
            anchor = str(h["anchor"])
            if anchor not in by_anchor:
                hunk_problems.append(
                    f"{path}: recorded hunk anchor {anchor!r} is not a git hunk"
                )
                continue
            real = len(by_anchor[anchor])
            if real != h["added_lines"]:
                hunk_problems.append(
                    f"{path}: hunk {anchor!r} is recorded with added_lines="
                    f"{h['added_lines']} but git shows {real} added lines"
                )
        # (2) every git hunk is recorded exactly once
        for anchor in by_anchor:
            if anchor not in recorded_anchors:
                hunk_problems.append(
                    f"{path}: git hunk {anchor!r} is missing from the record"
                )
        # (3) the recorded aggregate counts equal the git-derived ones
        real_authored = sum(
            len(by_anchor[a]) for a in authored_anchors if a in by_anchor
        )
        real_not = sum(len(by_anchor[a]) for a in not_anchors if a in by_anchor)
        if entry.get("added_lines_authored") != real_authored:
            hunk_problems.append(
                f"{path}: added_lines_authored={entry.get('added_lines_authored')} "
                f"but git shows {real_authored} added lines in the authored hunks"
            )
        if entry.get("added_lines_not_authored") != real_not:
            hunk_problems.append(
                f"{path}: added_lines_not_authored="
                f"{entry.get('added_lines_not_authored')} but git shows {real_not} "
                "added lines in the not-authored hunks"
            )
        if entry.get("n_authored") != len(recorded_authored):
            hunk_problems.append(
                f"{path}: n_authored={entry.get('n_authored')} but "
                f"{len(recorded_authored)} hunks are listed as authored"
            )
        if entry.get("n_not_authored") != len(recorded_not):
            hunk_problems.append(
                f"{path}: n_not_authored={entry.get('n_not_authored')} but "
                f"{len(recorded_not)} hunks are listed as not authored"
            )

        # (4) the record's own attribution method reproduces the partition
        for anchor, added in hunks:
            hit = next(
                (tok for tok in tokens for line in added if tok in line),
                None,
            )
            if anchor in authored_anchors and hit is None:
                token_problems.append(
                    f"{path}: hunk {anchor!r} is recorded as AUTHORED but none of "
                    "its added lines contain an attribution token"
                )
            if anchor in not_anchors and hit is not None:
                token_problems.append(
                    f"{path}: hunk {anchor!r} is recorded as NOT authored but its "
                    f"added lines contain the attribution token {hit!r}"
                )

        # (5) the "N regions" prose equals the git-derived proximity regions
        for label, filename in PROSE_FILE_LABELS.items():
            if not path.endswith(filename):
                continue
            computed = set()
            for a in authored_anchors:
                for b in not_anchors:
                    sa, sb = _old_start(a), _old_start(b)
                    if abs(sa - sb) <= PROXIMITY_WINDOW:
                        computed.add(tuple(sorted((sa, sb))))
            reason = str(
                (record.get("history_was_NOT_rewritten") or {}).get("reason", "")
            )
            m = re.search(rf"{label}:\s*((?:\d+/\d+(?:,\s*)?)+)", reason)
            if m is None:
                prose_problems.append(
                    f"{path}: the record's prose names no {label!r} region pairs, "
                    f"but git finds {len(computed)}"
                )
                continue
            named = {
                tuple(sorted(int(x) for x in pair.split("/")))
                for pair in m.group(1).split(",")
                if pair.strip()
            }
            if named != computed:
                prose_problems.append(
                    f"{path}: the record names {sorted(named)} as the {label} "
                    f"proximity regions, but git finds {sorted(computed)}"
                )

    # (6) the "N regions" total and the "966 lines" figure
    reason = str((record.get("history_was_NOT_rewritten") or {}).get("reason", ""))
    total_regions = sum(
        1
        for path in diff_hunks
        for a in frozenset(
            str(h["anchor"]) for h in (files[path].get("authored_this_session") or [])
        )
        for b in frozenset(
            str(h["anchor"])
            for h in (files[path].get("not_authored_this_session") or [])
        )
        if abs(_old_start(a) - _old_start(b)) <= PROXIMITY_WINDOW
    )
    m = re.search(r"(\d+)\s+regions", reason)
    if m is None:
        prose_problems.append(
            "the record states no region count, but git finds "
            f"{total_regions} authored/not-authored hunks within "
            f"{PROXIMITY_WINDOW} lines of each other"
        )
    elif int(m.group(1)) != total_regions:
        prose_problems.append(
            f"the record states {m.group(1)} regions where authored and "
            f"pre-existing hunks sit within {PROXIMITY_WINDOW} lines of each "
            f"other, but git finds {total_regions}"
        )
    m = re.search(r"entire content is (\d+)\s+lines", reason)
    trainer_path = next(
        (p for p in files if p.endswith(PROSE_FILE_LABELS["trainer"])), None
    )
    if m is not None and trainer_path is not None:
        stated = int(m.group(1))
        real = (files[trainer_path] or {}).get("added_lines_not_authored")
        if real != stated:
            prose_problems.append(
                f"the record says the never-reviewed trainer diff is {stated} "
                f"lines, but the record's own added_lines_not_authored says {real}"
            )

    # (7) the correction's "4 hunks / 354 lines ... 6 hunks / 49 lines"
    correction = str(record.get("CORRECTION_to_earlier_disclosure") or "")
    m = re.search(
        r"(\d+)\s+hunks?\s*/\s*(\d+)\s+lines\s+authored\s+here\s+and\s+"
        r"(\d+)\s+hunks?\s*/\s*(\d+)\s+lines\s+not",
        correction,
    )
    panel_path = next(
        (p for p in files if p.endswith(PROSE_FILE_LABELS["panel"])), None
    )
    if m is not None and panel_path is not None:
        panel = files[panel_path] or {}
        for prose_val, field in (
            (int(m.group(1)), "n_authored"),
            (int(m.group(2)), "added_lines_authored"),
            (int(m.group(3)), "n_not_authored"),
            (int(m.group(4)), "added_lines_not_authored"),
        ):
            if panel.get(field) != prose_val:
                prose_problems.append(
                    f"the correction states {prose_val} for {field}, but the "
                    f"record's {panel_path} entry says {panel.get(field)}"
                )

    # (8) the commit list: count note, and every commit against git
    note = str(record.get("commits_count_note") or "")
    m = re.search(r"(\d+)\s+after", note)
    commits = list(record.get("commits_touching_these_files") or [])
    if m is not None and int(m.group(1)) != len(commits):
        prose_problems.append(
            f"the record's commit-count note says {m.group(1)} entries after "
            f"reconciliation, but the list holds {len(commits)}"
        )
    contaminated = set(files)
    for entry in commits:
        sha = str(entry.get("sha") or "")
        short = str(entry.get("short") or "")
        if not _git_ok("cat-file", "-e", f"{sha}^{{commit}}"):
            commit_problems.append(f"commit {short or sha} is not in git")
            continue
        if not short or not sha.startswith(short):
            commit_problems.append(
                f"commit short {short!r} is not a prefix of its sha {sha!r}"
            )
        real_subject = _git("log", "-1", "--format=%s", sha).strip()
        if str(entry.get("subject") or "") != real_subject:
            commit_problems.append(
                f"commit {short} is recorded with subject "
                f"{entry.get('subject')!r} but git says {real_subject!r}"
            )
        names = {
            n
            for n in _git("show", "--format=", "--name-only", sha).splitlines()
            if n.strip()
        }
        real_touches = sorted(n for n in names if n in contaminated)
        if sorted(entry.get("touches") or []) != real_touches:
            commit_problems.append(
                f"commit {short} is recorded with touches "
                f"{sorted(entry.get('touches') or [])} but git says {real_touches}"
            )

    # (9) the SHAs named in the prose all exist, and the one that touched
    #     neither file is not in the commit list
    why = str(record.get("why_this_exists") or "")
    named_shas = set(re.findall(r"\b[0-9a-f]{10}\b", why))
    for sha in sorted(named_shas):
        if not _git_ok("cat-file", "-e", f"{sha}^{{commit}}"):
            prose_problems.append(f"the prose names commit {sha}, which is not in git")
    for sha in sorted(named_shas):
        if any(str(c.get("sha") or "").startswith(sha) for c in commits):
            continue
        names = {
            n
            for n in _git("show", "--format=", "--name-only", sha).splitlines()
            if n.strip()
        }
        touched = sorted(n for n in names if n in contaminated)
        if touched:
            prose_problems.append(
                f"the prose names commit {sha} but the commit list omits it, and "
                f"it did touch {touched}"
            )

    return {
        "hunk_problems": hunk_problems,
        "token_problems": token_problems,
        "commit_problems": commit_problems,
        "prose_problems": prose_problems,
        "states": states,
        "diff_hunks": diff_hunks,
        "counts": {
            "files": len(files),
            "hunks": checked_hunks,
            "commits": len(commits),
            "tokens": len(tokens),
        },
    }


# --------------------------------------------------------------------------
# the claims audit
# --------------------------------------------------------------------------


def _audit_claims(doc: dict[str, Any]) -> tuple[list[str], int]:
    """Every structural claim in the YAML that names an artifact on disk."""
    problems: list[str] = []
    if doc.get("schema") != EXPECTED_CLAIMS_SCHEMA:
        problems.append(
            f"schema is {doc.get('schema')!r}, expected {EXPECTED_CLAIMS_SCHEMA!r}"
        )
    narrative_doc = doc.get("narrative_doc")
    if not isinstance(narrative_doc, str) or not (REPO / narrative_doc).is_file():
        problems.append(f"narrative_doc {narrative_doc!r} does not name a file on disk")
    claims = doc.get("claims")
    if not isinstance(claims, list):
        problems.append("the file has no top-level 'claims' list")
        return problems, 0
    ids = [str(c.get("id")) for c in claims]
    if len(set(ids)) != len(ids):
        problems.append("the claims list contains a duplicate claim id")
    for claim in claims:
        cid = claim.get("id")
        for required in ("id", "claim", "value", "source", "field"):
            if required not in claim:
                problems.append(f"claim {cid!r} is missing the key {required!r}")
        tol = claim.get("tol", 5e-4)
        try:
            tol_value = float(tol)
        except (TypeError, ValueError):
            problems.append(f"claim {cid!r}: tol {tol!r} is not a number")
        else:
            if not math.isfinite(tol_value) or tol_value < 0:
                problems.append(
                    f"claim {cid!r}: tol {tol!r} is not a finite non-negative "
                    "tolerance"
                )
        source = str(claim.get("source") or "")
        src = REPO / source
        if not src.is_file():
            problems.append(
                f"claim {cid!r}: its source {source!r} is not on disk — the claim "
                "cannot be resolved against any artifact"
            )
            continue
        try:
            payload = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(
                f"claim {cid!r}: its source {source!r} is unreadable: {exc}"
            )
            continue
        field = str(claim.get("field") or "")
        found, _ = _dig(payload, field)
        if not found:
            problems.append(
                f"claim {cid!r}: field {field!r} does not resolve inside " f"{source!r}"
            )
        wording = " ".join(str(claim.get(k, "")) for k in ("id", "claim", "field"))
        if CAP_LIKE_RE.search(wording) and not str(claim.get("note") or "").strip():
            problems.append(
                f"claim {cid!r}: the file requires a note for a CAPPED/PROXY "
                "claim, and this one has none"
            )
    return problems, len(claims)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_the_provenance_schema_is_the_one_this_guard_understands() -> None:
    record = _load_provenance()
    assert record.get("schema") == EXPECTED_PROVENANCE_SCHEMA, (
        f"registry/file_provenance.json declares schema "
        f"{record.get('schema')!r}, expected {EXPECTED_PROVENANCE_SCHEMA!r} — "
        "this guard was written against the v1 layout"
    )


def test_every_recorded_hunk_is_real_git_history() -> None:
    """The record's anchors and added-line counts must reproduce from git."""
    record = _load_provenance()
    audit = _audit_provenance(record)
    assert not audit["hunk_problems"], (
        "registry/file_provenance.json does not match git history: "
        f"{audit['hunk_problems']}"
    )


def test_the_attribution_tokens_reproduce_the_authored_partition() -> None:
    """The record's own stated method must reproduce its own partition."""
    record = _load_provenance()
    audit = _audit_provenance(record)
    assert not audit["token_problems"], (
        "the attribution-token rule does not reproduce the authored / "
        f"not-authored split: {audit['token_problems']}"
    )


def test_every_recorded_commit_matches_git() -> None:
    """The commit list's shas, shorts, subjects and touches must be true."""
    record = _load_provenance()
    audit = _audit_provenance(record)
    assert not audit["commit_problems"], (
        "registry/file_provenance.json misdescribes its commits: "
        f"{audit['commit_problems']}"
    )


def test_the_records_prose_counts_match_its_structure() -> None:
    """The record's quoted numbers must equal the git-derived ones."""
    record = _load_provenance()
    audit = _audit_provenance(record)
    assert not audit["prose_problems"], (
        "the prose of registry/file_provenance.json disagrees with the "
        f"structure it describes: {audit['prose_problems']}"
    )


def test_every_narrative_claim_points_at_an_artifact_that_exists() -> None:
    """Every claim's source and field path must resolve against disk."""
    doc = _load_claims()
    problems, n_claims = _audit_claims(doc)
    assert not problems, (
        f"registry/narrative_claims.yaml has {n_claims} claims and "
        f"{len(problems)} of them do not resolve against an artifact on disk: "
        f"{problems}"
    )


def test_the_pre_flight_gate_fails_when_a_claim_artifact_is_missing(
    tmp_path,
) -> None:
    """A claim whose artifact is gone must FAIL the pre-flight gate.

    MEASURED 2026-10-31: the gate ``p3_narrative_claims_check`` invokes
    ``verify_narrative_claims.py`` WITHOUT ``--strict``.  The checker exits 0
    when the only problem is an UNVERIFIABLE claim (a missing source file or
    an unresolvable field), so the gate PASSes while a §5.2 traceability
    claim points at nothing.  The gate's own failure branch counts "??" rows
    that this invocation can never produce.
    """
    from molmetal.scripts import pre_flight_check
    from molmetal.scripts import verify_narrative_claims as claims_mod

    # (a) the checker's exit codes, measured on a COPY with one broken claim
    doc = _load_claims()
    broken = copy.deepcopy(doc)
    broken["claims"][0]["source"] = "metrics/by_metric/__WFA127_DOES_NOT_EXIST__.json"
    broken_path = tmp_path / "broken_claims.yaml"
    import yaml

    broken_path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")
    rc_plain = claims_mod.main(["--claims", str(broken_path)])
    rc_strict = claims_mod.main(["--claims", str(broken_path), "--strict"])
    assert (rc_plain, rc_strict) == (0, 1), (
        "the checker's exit codes changed: without --strict it returned "
        f"{rc_plain}, with --strict {rc_strict} (expected 0 and 1) — re-derive "
        "the invariant this guard pins"
    )

    # (b) the argv the gate actually uses
    captured: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return 0, "", ""

    original = pre_flight_check._run
    pre_flight_check._run = _fake_run
    try:
        result = pre_flight_check.gate_narrative_claims(
            "p3_narrative_claims_check",
            "P3",
            "ag52_narrative_claims",
            REPO / "molmetal" / "scripts" / "verify_narrative_claims.py",
        )
    finally:
        pre_flight_check._run = original

    argv = captured[0]
    assert "--strict" in argv, (
        "the p3_narrative_claims_check gate invokes the checker WITHOUT "
        f"--strict (argv={argv}), so a claim whose artifact is missing exits 0 "
        f"and the gate reports {result.status!r}. AGENTS.md §5.2 requires every "
        "quoted number to trace to a path on disk. Fix: add '--strict' to the "
        "argv in gate_narrative_claims "
        "(molmetal/scripts/pre_flight_check.py)."
    )


def test_the_guard_is_not_vacuous() -> None:
    """The guard must fail if the registry stops declaring anything."""
    record = _load_provenance()
    audit = _audit_provenance(record)
    counts = audit["counts"]
    assert counts["files"] >= 2, f"only {counts['files']} files are recorded"
    assert counts["hunks"] >= 40, f"only {counts['hunks']} hunks were checked"
    assert counts["commits"] >= 3, f"only {counts['commits']} commits were checked"
    assert counts["tokens"] >= 5, f"only {counts['tokens']} attribution tokens"
    problems, n_claims = _audit_claims(_load_claims())
    assert n_claims >= 40, f"only {n_claims} claims are declared"
    assert len(audit["states"]) >= 2, "no historical state could be resolved"


def test_a_perturbed_provenance_record_is_rejected() -> None:
    """The guard must reject a record whose claims are false.

    This is the guard's own non-vacuity proof: each perturbation below is a
    value a repair could plausibly write, and each one must be caught with
    the offending value named.
    """
    record = _load_provenance()
    panel = "molmetal/scripts/cfm_property_panel.py"

    # (a) an added-line count that git contradicts
    bad = copy.deepcopy(record)
    entry = bad["files"][panel]["authored_this_session"][0]
    truth = entry["added_lines"]
    entry["added_lines"] = 999
    problems = _audit_provenance(bad)["hunk_problems"]
    assert problems, "the guard accepted a record with a wrong added_lines count"
    assert any("999" in p and str(truth) in p for p in problems), (
        f"the guard did not name both values (999 recorded vs {truth} in git): "
        f"{problems}"
    )

    # (b) a hunk moved to the wrong side of the authored / not-authored split
    bad = copy.deepcopy(record)
    moved = bad["files"][panel]["authored_this_session"].pop()
    bad["files"][panel]["not_authored_this_session"].append(moved)
    problems = _audit_provenance(bad)["token_problems"]
    assert problems, "the guard accepted a record with a mis-split hunk"
    assert any(
        moved["anchor"] in p for p in problems
    ), f"the guard did not name the offending hunk {moved['anchor']}: {problems}"

    # (c) a commit subject that git contradicts
    bad = copy.deepcopy(record)
    truth_subject = bad["commits_touching_these_files"][0]["subject"]
    bad["commits_touching_these_files"][0]["subject"] = "a subject that never existed"
    problems = _audit_provenance(bad)["commit_problems"]
    assert problems, "the guard accepted a record with a wrong commit subject"
    assert any(
        "a subject that never existed" in p and truth_subject in p for p in problems
    ), f"the guard did not name both subjects: {problems}"


def test_a_perturbed_claims_file_is_rejected(tmp_path) -> None:
    """A claim whose artifact path is wrong must be caught."""
    doc = _load_claims()
    bad = copy.deepcopy(doc)
    bad["claims"][0]["source"] = "metrics/by_metric/__WFA127_DOES_NOT_EXIST__.json"
    problems, _ = _audit_claims(bad)
    assert problems, "the guard accepted a claim whose artifact is missing"
    assert any(
        "__WFA127_DOES_NOT_EXIST__" in p for p in problems
    ), f"the guard did not name the offending source: {problems}"
