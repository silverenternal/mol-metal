#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A194 (2026-11-02) -- closure guard for the MIRROR band-provenance class.

WHAT THIS GUARDS
----------------
The published 92.7-94 % frontier band has TWO competing provenance readings on
disk, and they do not even agree on the numerals, so the band's source cells are
NOT settled on disk.  Five renderings of that band are REFUTED and must never
appear as a live assertion (AGENTS.md §5.2, §5.5 rule #6):

  X1 -- "no endpoint is a connected-component count" / "THE BAND IS NOT A
        CONNECTIVITY BAND" / "NEITHER is a connected-component count"
        (TRUE under the `band_provenance` reading, FALSE under the
        `sources`.`EF-TALFM` reading, whose lower cell IS a connectivity count)
  X2 -- "the band is measured on the published POST-PROCESSED stream"
        (FALSE for this band: both `band_provenance` endpoints carry
        `post_processed: false`, and the `sources`.`EF-TALFM` arm is MIXED)
  X3 -- "their Connectivity is computed AFTER retaining the largest component"
        (REFUTED: EF-TALFM's own text takes the count BEFORE the retention)
  X4 -- "the band is minted here: lower = GeoLDM Val&Uniq, upper = HLTF
        PB-Valid" (FALSE as a SETTLED statement: the record holds TWO
        competing readings whose numerals disagree)
  MIRROR -- "both arms are RAW" / "both endpoints are RAW" / "band as a whole
        is RAW" (as false as the blanket POST-PROCESSED claim: under the
        `sources`.`EF-TALFM` reading the frontier arm is MIXED)

A rendering that appears inside a DATED SUPERSESSION block, a REFUTATION
record, or a MUST-NOT-ASSERT list is NOT a live claim and must NOT be
"repaired" -- repairing it would destroy the evidence that the defect was
found.  This guard therefore separates LIVE positions from DATED ones BY PATH
(and, inside a producer, by the name of the dated constant), and only the LIVE
positions are scanned.

The paths this guard scans are overridable through the environment so that a
broken COPY of an artefact can be fed to the guard:

  WFA194_ARTEFACT_DIR -- the directory holding the three metric JSONs
  WFA194_SCRIPT_DIR   -- the directory holding the three emitters

THE BITE PROOF (run by the wave, not by pytest): copy the artefacts and the
emitters into a /tmp directory, write a refuted rendering back into a LIVE
position of the copy, and run THIS module against the copy with the two
environment variables above.  The guard must FAIL.  A guard that passes
against a broken artefact is a FAILURE.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

ARTEFACT_DIR = Path(
    os.environ.get("WFA194_ARTEFACT_DIR", REPO_ROOT / "metrics" / "by_metric")
)
SCRIPT_DIR = Path(
    os.environ.get("WFA194_SCRIPT_DIR", REPO_ROOT / "molmetal" / "scripts")
)

ARTEFACT_NAMES = (
    "wfa155_ligand_connected_rate.json",
    "wfa155_posebusters_validity.json",
    "wfa155_metal_coordination_sphere.json",
)

EMITTER_NAMES = (
    "wf_a155_ligand_connected_rate.py",
    "wf_a155_posebusters_validity.py",
    "wf_a155_metal_coordination_sphere.py",
)

# --------------------------------------------------------------------------
# The five refuted renderings
# --------------------------------------------------------------------------

#: X1 -- "no endpoint is a connected-component count" and its two siblings.
#: The literals are the repo's own must-not-assert list
#: (molmetal/reports/wfa164_coordination_sphere_narrative/
#:  pool_specific_literal_census.json -> literals_deliberately_LEFT_ALONE[0]).
X1_PATTERNS: Tuple[str, ...] = (
    r"no endpoint is a connected-component count",
    r"neither (?:endpoint|band endpoint) is a connected-component count",
    r"neither is a connected-component count",
    r"none is a connected-component count",
    r"the band is not a connectivity band",
    r"not a connectivity band",
    r"two different observables, not a connectivity band",
)

#: X2 -- "the band is measured on the published POST-PROCESSED stream".
X2_PATTERNS: Tuple[str, ...] = (
    r"measured on the published POST-PROCESSED stream",
    r"the band is measured on the published .?post-processed.? stream",
    r"our number is not post-processed and theirs is",
)

#: X3 -- "their Connectivity is computed AFTER retaining the largest component".
X3_PATTERNS: Tuple[str, ...] = (
    r"computed after retaining the largest component",
    r"after retaining the largest component",
    r"after retaining the atom-count largest connected component",
)

#: X4 -- "the band is minted here: lower = GeoLDM Val&Uniq, upper = HLTF
#: PB-Valid", asserted as a SETTLED statement.
X4_PATTERNS: Tuple[str, ...] = (
    r"the band is minted here",
    r"lower = GeoLDM Val&Uniq, upper = HLTF PB-Valid",
    r"lower endpoint .* = GeoLDM",
    r"upper endpoint .* = HLTF",
)

#: MIRROR -- the blanket "both arms are RAW" over-claim.
MIRROR_PATTERNS: Tuple[str, ...] = (
    r"both arms are RAW",
    r"both endpoints are RAW",
    r"two endpoints are (?:both )?RAW",
    r"band as a whole is RAW",
)

REFUTED_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "X1": X1_PATTERNS,
    "X2": X2_PATTERNS,
    "X3": X3_PATTERNS,
    "X4": X4_PATTERNS,
    "MIRROR": MIRROR_PATTERNS,
}

#: A string that carries one of these markers is READING-SCOPED: it already
#: says that the band's provenance is NOT settled on disk, so the raw "RAW"
#: wording inside it is the CORRECT form, not the MIRROR over-claim.
READING_SCOPE_MARKERS: Tuple[str, ...] = (
    "not settled on disk",
    "under the `band_provenance` reading",
    "in that reading",
    "band_provenance` reading",
    "under the competing",
    "competing provenance reading",
    "one reading",
)

#: "both arms are RAW" is only the MIRROR over-claim when the SAME string
#: also refers to the frontier band.  In wfa155_ligand_connected_rate.json the
#: phrase "Both arms are RAW" names THIS metric's own two arms (the
#: metal-inclusive arm and the ligand-only arm), both of which are genuinely
#: RAW -- a different referent, and a TRUE statement.
BAND_REFERENTS: Tuple[str, ...] = ("band", "frontier")


def _norm(text: str) -> str:
    """Collapse whitespace so a line-wrapped rendering still matches."""
    return re.sub(r"\s+", " ", text)


def _matches(patterns: Tuple[str, ...], text: str) -> bool:
    flat = _norm(text)
    return any(re.search(p, flat, re.IGNORECASE) for p in patterns)


def _is_reading_scoped(text: str) -> bool:
    low = _norm(text).lower()
    return any(m.lower() in low for m in READING_SCOPE_MARKERS)


def _refuted_hits(text: str) -> List[str]:
    """Return the refuted classes carried by ``text`` as a LIVE assertion.

    X1-X4 are unqualified by nature: if the text carries them, it is a
    defect wherever it lives (the dated records that quote them are excluded
    by PATH before this function is reached).

    MIRROR needs two extra conditions, because the CORRECT form also
    contains the word "RAW":

      (a) the string must not be reading-scoped, and
      (b) "both arms are RAW" only counts when the same string refers to the
          frontier band (see ``BAND_REFERENTS``).
    """
    hits: List[str] = []
    for cls, patterns in REFUTED_PATTERNS.items():
        if not _matches(patterns, text):
            continue
        if cls == "MIRROR":
            if _is_reading_scoped(text):
                continue
            if re.search(r"both arms are RAW", _norm(text), re.IGNORECASE) and not any(
                r.lower() in _norm(text).lower() for r in BAND_REFERENTS
            ):
                # names THIS metric's own arms, not the band's arm
                continue
        hits.append(cls)
    return hits


# --------------------------------------------------------------------------
# DATED positions -- a JSON path component that carries one of these is a
# dated supersession / repair record, never a live claim.
# --------------------------------------------------------------------------

DATED_PATH_MARKERS: Tuple[str, ...] = (
    "superseded",
    "supersedes",
    "_repair_20",
    "_backing_repair_20",
    "why_superseded",
)


def _is_dated_path(path: str) -> bool:
    low = path.lower()
    return any(m in low for m in DATED_PATH_MARKERS)


# --------------------------------------------------------------------------
# Artefact walking
# --------------------------------------------------------------------------


def _string_leaves(node: Any, path: str = "") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.extend(_string_leaves(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            out.extend(_string_leaves(value, f"{path}[{idx}]"))
    elif isinstance(node, str):
        out.append((path, node))
    return out


def _load(name: str) -> Dict[str, Any]:
    path = ARTEFACT_DIR / name
    assert path.is_file(), f"missing artefact: {path}"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Producer walking -- AST, so COMMENTS (which quote the defect on purpose)
# are excluded by construction.
# --------------------------------------------------------------------------

DATED_CONST_NAME = re.compile(r"superseded", re.IGNORECASE)


def _dated_constant_nodes(tree: ast.Module) -> set:
    """The id() of every string-literal NODE that lives inside a dated
    supersession constant.

    The exclusion is by NODE IDENTITY, not by VALUE: a dated constant holds
    the superseded rendering as a VALUE, so a value-based exclusion would
    also hide the very same string if it were re-inserted anywhere else in
    the file (which is exactly what a regression looks like).
    """
    out: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not any(DATED_CONST_NAME.search(n) for n in names):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.add(id(sub))
    return out


def _live_source_strings(tree: ast.Module) -> List[Tuple[int, str]]:
    """Every string literal in the module that is NOT inside a dated constant."""
    dated = _dated_constant_nodes(tree)
    out: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in dated:
                continue
            out.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if id(value) in dated:
                        continue
                    parts.append(value.value)
            out.append((node.lineno, "".join(parts)))
    return out


def _parse_emitter(name: str) -> ast.Module:
    path = SCRIPT_DIR / name
    assert path.is_file(), f"missing emitter: {path}"
    return ast.parse(path.read_text())


def _load_module(name: str):
    path = SCRIPT_DIR / name
    assert path.is_file(), f"missing emitter: {path}"
    spec = importlib.util.spec_from_file_location(f"wfa194_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The two renderings THIS wave (Wf-A194) replaced
# --------------------------------------------------------------------------

#: The pre-Wf-A194 module docstring rendering of the LIG emitter.
LIG_OLD_DOCSTRING = (
    "The frontier's connectivity band (92.7-94 %) is measured on the WHOLE\n"
    "emitted molecule."
)

#: The pre-Wf-A194 module docstring rendering of the POSE emitter.
PB_OLD_DOCSTRING = (
    "The frontier band we are compared against (92.7--94 % connected / 99.9 % "
    "valid /\n91.9 % PoseBusters-valid, per arXiv 2609.08333 and FlowMol3 "
    "arXiv 2508.12629)\nis quoted as a *percentage*."
)


# --------------------------------------------------------------------------
# 1. The LIVE positions of the three artefacts carry no refuted rendering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("artefact", ARTEFACT_NAMES)
def test_live_artefact_positions_carry_no_refuted_rendering(artefact: str) -> None:
    doc = _load(artefact)
    offenders: List[str] = []
    for path, text in _string_leaves(doc):
        if _is_dated_path(path):
            continue  # a dated supersession / repair record: not a live claim
        for cls in _refuted_hits(text):
            offenders.append(f"{artefact} :: {path} :: {cls}")
    assert not offenders, (
        "LIVE positions carry a refuted band-provenance rendering:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("emitter", EMITTER_NAMES)
def test_producer_live_strings_carry_no_refuted_rendering(emitter: str) -> None:
    tree = _parse_emitter(emitter)
    offenders: List[str] = []
    for lineno, text in _live_source_strings(tree):
        for cls in _refuted_hits(text):
            offenders.append(f"{emitter}:{lineno} :: {cls}")
        for old in (LIG_OLD_DOCSTRING, PB_OLD_DOCSTRING):
            if old in text:
                offenders.append(f"{emitter}:{lineno} :: Wf-A194-superseded-rendering")
    assert not offenders, (
        "LIVE strings in the producer carry a refuted band-provenance "
        "rendering:\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------
# 2. The two renderings this wave replaced must NOT be live anywhere
# --------------------------------------------------------------------------


def test_wfa194_superseded_docstring_renderings_are_not_live() -> None:
    offenders: List[str] = []
    for artefact in ARTEFACT_NAMES:
        doc = _load(artefact)
        for path, text in _string_leaves(doc):
            if _is_dated_path(path):
                continue
            for old in (LIG_OLD_DOCSTRING, PB_OLD_DOCSTRING):
                if old in text:
                    offenders.append(f"{artefact} :: {path}")
    for emitter in EMITTER_NAMES:
        tree = _parse_emitter(emitter)
        for lineno, text in _live_source_strings(tree):
            for old in (LIG_OLD_DOCSTRING, PB_OLD_DOCSTRING):
                if old in text:
                    offenders.append(f"{emitter}:{lineno}")
    assert not offenders, (
        "the superseded Wf-A194 docstring rendering is LIVE again:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------
# 3. Every replaced rendering survives VERBATIM in a DATED position
# --------------------------------------------------------------------------


#: (artefact, json path, verbatim text) -- the dated rule-#6 records that must
#: keep carrying the superseded rendering, character for character.
DATED_VERBATIM_RECORDS: Tuple[Tuple[str, str, str], ...] = (
    (
        "wfa155_ligand_connected_rate.json",
        "comparability.superseded_frontier_2026_11_01_wfa183.rendering",
        "the published band's two endpoints are RAW (post_processed: false in "
        "the band-provenance record)",
    ),
    (
        "wfa155_ligand_connected_rate.json",
        "comparability.superseded_frontier_2026_11_01_wfa189.rendering",
        "the published band's two endpoints are RAW (post_processed: false in "
        "the band-provenance record)",
    ),
    (
        "wfa155_posebusters_validity.json",
        "comparability_vs_frontier_band.superseded_reading_2026_11_01."
        "superseded[0].text",
        "measured on the published POST-PROCESSED stream",
    ),
    (
        "wfa155_posebusters_validity.json",
        "comparability_vs_frontier_band.superseded_reading_2026_11_01."
        "superseded[1].text",
        "our number is not post-processed and theirs is",
    ),
    (
        "wfa155_posebusters_validity.json",
        "comparability_vs_frontier_band.superseded_reading_2026_11_01_wfa182."
        "superseded[1].text",
        "post-processing is a THIRD, separate axis on top of the METRIC break, "
        "not the break itself: both arms are RAW, and the decisive break is "
        "that under BOTH readings the band is external, is NOT one homogeneous "
        "quantity, and is NOT on our scale (OUR number is a "
        "connected-component count on the RAW emitted SMILES, and nothing "
        "else)",
    ),
    (
        "wfa155_metal_coordination_sphere.json",
        "comparability_statement.break_3_post_processing_superseded_2026_11_01"
        "_wfa176.rendering",
        "POST-PROCESSING — the published band is measured on the published "
        "POST-PROCESSED stream; the pool measured here is RAW (no "
        "post-processing, no repair).",
    ),
)


def _dig(doc: Any, path: str) -> Any:
    cur = doc
    for token in path.split("."):
        match = re.fullmatch(r"([^\[\]]*)(\[(\d+)\])?", token)
        assert match is not None, f"unparsable path token: {token!r}"
        name, _, index = match.groups()
        if name:
            cur = cur[name]
        if index is not None:
            cur = cur[int(index)]
    return cur


@pytest.mark.parametrize(
    "artefact,path,verbatim",
    DATED_VERBATIM_RECORDS,
    ids=[f"{a}:{p}" for a, p, _ in DATED_VERBATIM_RECORDS],
)
def test_superseded_rendering_survives_verbatim_in_a_dated_position(
    artefact: str, path: str, verbatim: str
) -> None:
    doc = _load(artefact)
    value = _dig(doc, path)
    assert isinstance(value, str), f"{artefact} :: {path} is not a string"
    assert verbatim in value, (
        f"{artefact} :: {path} no longer carries the superseded rendering "
        f"VERBATIM -- the rule-#6 record was edited or dropped"
    )
    assert _is_dated_path(path), f"{path} is not a dated position"


#: (emitter, artefact, superseded rendering) -- the Wf-A194 dated docstring
#: record, and where it has to REACH.
DOCSTRING_EMITTER_PARITY: Tuple[Tuple[str, str, str], ...] = (
    (
        "wf_a155_ligand_connected_rate.py",
        "wfa155_ligand_connected_rate.json",
        LIG_OLD_DOCSTRING,
    ),
    (
        "wf_a155_posebusters_validity.py",
        "wfa155_posebusters_validity.json",
        PB_OLD_DOCSTRING,
    ),
)


def test_wfa194_docstring_supersession_is_emitted_into_the_published_artifact() -> None:
    """The Wf-A194 dated docstring record must be EMITTED, not merely DEFINED.

    A constant that exists in a producer's source but is never LOADED by that
    producer's artefact-assembly code is the producer-artefact parity defect
    class: a re-run reproduces the artefact WITHOUT the dated record, so the
    dated history is dropped with no diff to review.  Every OTHER dated
    rule-#6 record in both producers IS emitted this way (the LIG emitter
    emits `comparability.superseded_frontier_2026_11_01_wfa183` / `_wfa189`;
    the POSE emitter emits
    `comparability_vs_frontier_band.superseded_reading_2026_11_01` + `_wfa180`
    + `_wfa182`, `prior_posebusters_artifacts_measured
    .superseded_field_name_2026_11_01`, `narrative_self_consistency
    .superseded_2026_11_01_wfa182` and
    `published_cross_check_superseded_2026_11_01`).

    This asserts the property that actually matters -- the record REACHES the
    artefact -- in two independent ways:

      (1) SOURCE: the producer's code LOADS the dated constant, so a re-run
          reproduces the record instead of dropping it;
      (2) ARTEFACT: the published artefact carries the superseded docstring
          rendering VERBATIM, at a DATED position (a live position would trip
          `test_wfa194_superseded_docstring_renderings_are_not_live`).

    AS OF 2026-11-02 (Wf-A202) BOTH CHECKS FAIL FOR BOTH PRODUCERS: neither
    producer emits the record and neither published artefact carries the
    superseded docstring text.  The producers' own constant records the
    refusal (`emitted_into_the_artifact: "no -- deliberately"`, because
    emitting it would change the artefact's bytes, its sha256 and its
    registry row without changing a measurement), so the disposition is the
    OWNING wave's call.  Wf-A202 does not edit the two producers.  See the
    NAMED cross-wave finding XW-A202-02.
    """
    problems: List[str] = []
    for emitter, artefact, old in DOCSTRING_EMITTER_PARITY:
        # ---- (1) SOURCE: is the dated constant LOADED by the producer? ----
        tree = _parse_emitter(emitter)
        loaded = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "DOCSTRING_SUPERSEDED_2026_11_02_WFA194"
            and isinstance(node.ctx, ast.Load)
        ]
        if not loaded:
            problems.append(
                f"{emitter}: DOCSTRING_SUPERSEDED_2026_11_02_WFA194 is DEFINED but "
                f"never LOADED -- the producer does not emit it into {artefact} "
                "(producer-artefact parity gap)"
            )
        # ---- (2) ARTEFACT: does the published artefact carry it? ---------
        doc = _load(artefact)
        hits = [path for path, text in _string_leaves(doc) if old in text]
        if not hits:
            problems.append(
                f"{artefact}: no string leaf carries the superseded Wf-A194 "
                "docstring rendering VERBATIM -- the dated record never reached "
                "the artefact"
            )
        elif not any(_is_dated_path(path) for path in hits):
            problems.append(
                f"{artefact}: the superseded Wf-A194 docstring rendering is carried "
                f"at a LIVE position ({hits[0]}), not at a dated one"
            )
    assert not problems, (
        "the Wf-A194 dated docstring record is DEFINED but NOT EMITTED into the "
        "published artefacts (producer-artefact parity gap, NAMED cross-wave "
        "finding XW-A202-02):\n  " + "\n  ".join(problems)
    )


# --------------------------------------------------------------------------
# 4. The LIVE band-provenance renderings are the reading-safe ones
# --------------------------------------------------------------------------

#: (artefact, json path) -- the live positions that carry the band's
#: provenance and MUST be reading-scoped.
LIVE_BAND_POSITIONS: Tuple[Tuple[str, str], ...] = (
    (
        "wfa155_ligand_connected_rate.json",
        "comparability.breaks[2].frontier",
    ),
    (
        "wfa155_posebusters_validity.json",
        "comparability_vs_frontier_band.breaks[2].frontier",
    ),
    (
        "wfa155_posebusters_validity.json",
        "comparability_vs_frontier_band.breaks[2].consequence",
    ),
    (
        "wfa155_metal_coordination_sphere.json",
        "comparability_statement.breaks[2].frontier",
    ),
    (
        "wfa155_metal_coordination_sphere.json",
        "comparability_statement.break_3_post_processing",
    ),
    (
        "wfa155_metal_coordination_sphere.json",
        "frontier_reference.what_the_band_is",
    ),
)


@pytest.mark.parametrize(
    "artefact,path", LIVE_BAND_POSITIONS, ids=[f"{a}:{p}" for a, p in LIVE_BAND_POSITIONS]
)
def test_live_band_provenance_is_reading_scoped(artefact: str, path: str) -> None:
    doc = _load(artefact)
    value = _dig(doc, path)
    assert isinstance(value, str)
    flat = _norm(value)
    assert "NOT settled on disk" in flat, (
        f"{artefact} :: {path} does not say that the band's source cells are "
        f"NOT settled on disk"
    )
    assert "both arms are RAW" not in flat, (
        f"{artefact} :: {path} carries the MIRROR over-claim"
    )
    assert "measured on the published POST-PROCESSED stream" not in flat, (
        f"{artefact} :: {path} carries the X2 over-claim"
    )


def test_emitter_docstrings_are_reading_scoped() -> None:
    lig = _load_module("wf_a155_ligand_connected_rate.py")
    pb = _load_module("wf_a155_posebusters_validity.py")
    for module, name in ((lig, "wf_a155_ligand_connected_rate.py"),
                          (pb, "wf_a155_posebusters_validity.py")):
        doc = module.__doc__ or ""
        flat = _norm(doc)
        assert "NOT settled on disk" in flat, (
            f"{name}: the module docstring does not say that the band's "
            f"provenance is NOT settled on disk"
        )
        assert LIG_OLD_DOCSTRING not in doc
        assert PB_OLD_DOCSTRING not in doc


# --------------------------------------------------------------------------
# 5. The artefacts themselves still pass their own narrative guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("artefact", ARTEFACT_NAMES)
def test_artefact_narrative_self_consistency_still_passes(artefact: str) -> None:
    doc = _load(artefact)
    guard = doc.get("narrative_self_consistency")
    assert isinstance(guard, dict), f"{artefact}: no narrative_self_consistency"
    assert guard.get("pass") is True, f"{artefact}: narrative guard does not pass"
