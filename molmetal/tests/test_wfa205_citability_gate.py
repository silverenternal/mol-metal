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

"""Wf-A205 -- the MECHANICAL citability gate: no caveat-less paper headline.

WHY THIS FILE EXISTS
--------------------
Wf-A203 (2026-11-02) added the missing ``headline_caveat`` to the two
paper-facing headline rows that had none and left this open in its own
report: *"no mechanical rule blocks a third caveat-less headline row
(docs/METRIC_CITABILITY_RULE.md:288-292 still calls headline_caveat
optional)"*.  The rule was enforced BY HAND that time.  This file enforces
it BY TEST.

THE RULE AS SHIPPED
-------------------
Let ``R`` be the registry rows in ``registry/metrics.yaml``.

  SCOPE  A row is *governed* iff it satisfies all three of:
           ``quotable: true``            (a real measurement of this metric)
           ``headline_grade: true``      (table-grade, not prose-only)
           ``registry_role: paper_line`` (it is a paper-facing number)

  R2 (registry row)  Every governed row carries a ``headline_caveat`` that is
      a ``str`` whose ``.strip()`` is non-empty.

  R3 (payload direction -- the DURABLE direction)  Every governed row's
      ``source_json`` payload on disk ALSO carries a non-empty
      ``headline_caveat``.  This is the direction that survives a regenerate;
      see the payload-direction proof below.

  R4 (scope is DERIVED, not hand-listed)  The governed set is computed from
      ``molmetal.scripts.registry.LINE_METRICS`` (registry.py:103-135) and
      must equal the set of rows carrying ``registry_role: paper_line``.
      A new paper line is therefore covered automatically -- the gate cannot
      silently stop watching a row, because the scope is re-derived from code
      on every run and compared against the registry.

  R5 (parity, and it covers the OUT-OF-SCOPE rows too)  For EVERY row that is
      ``quotable`` AND ``headline_grade`` -- governed or not -- the registry
      row's ``headline_caveat`` must equal what ``assess_citability`` derives
      from the payload.  A registry row may not carry a caveat the payload
      does not: a hand-injected registry caveat is erased by the next
      ``registry.py generate`` and is therefore not a disclosure, it is a
      decoration.

WHY THE SCOPE IS ``paper_line`` AND NOT EVERY ``quotable + headline_grade``
row
-----------------------------------------------------------------------------
MEASURED 2026-11-02 on this working tree (``registry/metrics.yaml``,
sha256 9ddd971eae8e333ed7bf169660da28f95ab500528e53d6a4eb1c17df1480e684):
582 rows; 79 ``quotable``; 6 ``quotable + headline_grade``; of those 6,
**4 carry a caveat and 2 do not** -- ``nci60_precious_9cell_auc`` and
``platinai_3cell_resistance_real_strat``, and BOTH have
``registry_role: artifact``.  Every ``paper_line`` row that is
``quotable + headline_grade`` (3 of them) carries a caveat.

The scope is therefore *paper-facing*, for four reasons that are all
checkable rather than editorial:

  * ``registry.py cite`` -- the report the citability rule points the reader
    at -- prints ONLY paper-line rows: ``if r.get("registry_role") !=
    "paper_line": continue`` (registry.py:2276-2277).
  * ``docs/METRIC_CITABILITY_RULE.md`` §1 defines the three marks over "one
    line per paper-line metric"; §7 counts 11 paper-line metrics.
  * ``registry.py check`` counts headline-grade over ``paper_rows``
    (registry.py:1708-1721).
  * ``registry.py`` assigns ``registry_role`` from the two build passes:
    ``"paper_line"`` for a LINE_METRICS member (registry.py:1397) and
    ``"artifact"`` for the disk-walk rows (registry.py:1487).

Requiring a caveat of a row that no paper-facing surface can print would be
a rule over a population the citability rule does not govern, and it would
block a correct registry.  The two artifact-role rows that lack a caveat are
REPORTED as a bounded census (see ``gate/row_census.json``) and as a
residual -- they are not silently dropped.

THE PAYLOAD-DIRECTION PROOF (why the payload, not the registry row)
--------------------------------------------------------------------
``registry.py generate`` REBUILDS the whole metric layer from the payloads:

  * ``build_metrics`` starts from an EMPTY dict -- ``out: Dict[str, Any] =
    {}`` (registry.py:1362) -- and every row it writes is built from the
    on-disk JSON (registry.py:1366-1429 for the LINE_METRICS pass,
    registry.py:1435-1529 for the disk walk).  There is NO preservation pass
    for metrics (the preservation helper, ``_preserve_underived_datasets``,
    is DATASETS-only -- registry.py:2022-2024).
  * each row's ``headline_caveat`` is produced by ``assess_citability``:
    ``caveat = payload.get("headline_caveat")`` (registry.py:1239) and
    ``"headline_caveat": (caveat.strip() if isinstance(caveat, str) and
    caveat.strip() else None)`` (registry.py:1245-1247).
  * ``build_metrics`` calls it as ``rec.update(assess_citability(payload,
    rec["value"], _rel(src_path), tracked))`` (registry.py:1426-1429).
  * ``generate`` then dumps those rows to ``registry/metrics.yaml``
    (``metrics_doc`` at registry.py:2056, ``yaml.safe_dump`` +
    ``atomic_write_file`` at registry.py:2072-2079).

Therefore: a ``headline_caveat`` that exists ONLY in the registry row is
ERASED by the next ``registry.py generate``.  R3 is the durable direction and
R5 makes a registry-only caveat a FAILURE rather than a decoration.

THE LOCATORS ABOVE -- CORRECTED 2026-11-02 (the statements never moved)
---------------------------------------------------------------------
The 2026-11-02 rendering of this docstring cited ``registry.py:1361``,
``:2061``, ``:2077``, ``:103-136``, ``:1364-1434`` and ``:1436-1500``; those
locators had ROTTED against the file on disk.  MEASURED 2026-11-02 against
the working tree ``molmetal/scripts/registry.py`` (sha256
``18fae08ab25dfb866dd857dc1a92654e4647d63b9b75f37a36f33004a568a438``):
1361 -> **1362** (1361 is the closing ``\"\"\"`` of ``build_metrics``'
docstring), 2061 -> **2056** (2061 is a *member* of the ``metrics_doc``
literal), 2077 -> **2072** (2077 is the no-op write guard), 103-136 ->
**103-135**, 1364-1434 -> **1366-1429**, 1436-1500 -> **1435-1529**.  The
CITE STATEMENTS are unchanged.  The superseded rendering is preserved
VERBATIM in ``gate/payload_direction_proof.json`` -> ``corrections`` (AGENTS.md
5.5 rule #6) and the guard that keeps it from rotting again is
``test_every_code_citation_points_at_a_line_that_carries_its_text`` at the
bottom of this file.

THE REGISTRY PARSE
------------------
This file reads the registry with the SAME loader ``registry.py`` uses --
``_load_yaml`` (registry.py:255-261):

    yaml.safe_load(path.read_text(encoding="utf-8")) or {}

and the SAME key names ``assess_citability`` writes (registry.py:1241-1247):
``quotable``, ``headline_grade``, ``headline_caveat``.

CPU-only.  No GPU.  No network.  No ``os.walk`` (the repo is 201 GB).  Writes
NOTHING under ``registry/`` or ``metrics/``: the bite scenarios below mutate
a SHADOW copy of the registry written under pytest's ``tmp_path`` (outside
the repo -- asserted), never the repo's own file.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REGISTRY = REPO / "registry" / "metrics.yaml"

#: The rule, in one line, so it can be quoted verbatim in a report.
GATE_RULE = (
    "every registry row with quotable: true AND headline_grade: true AND "
    "registry_role: paper_line must carry a non-empty headline_caveat in the "
    "registry row AND in its source_json payload on disk"
)

#: The key names the rule reads.  These are the names `assess_citability`
#: writes into the row (registry.py:1241-1247).
KEY_QUOTABLE = "quotable"
KEY_HEADLINE_GRADE = "headline_grade"
KEY_CAVEAT = "headline_caveat"
KEY_ROLE = "registry_role"
KEY_SOURCE_JSON = "source_json"
ROLE_PAPER_LINE = "paper_line"


# ---------------------------------------------------------------------------
# registry access -- the SAME loader registry.py uses
# ---------------------------------------------------------------------------


def _yaml_load(text: str) -> Dict[str, Any]:
    """Parse a registry document with the SAME semantics as ``yaml.safe_load``.

    ``registry.py`` uses ``yaml.safe_load`` (registry.py:255-261).  PyYAML's
    ``CSafeLoader`` is the C implementation of ``SafeLoader`` -- identical
    semantics, ~7x faster on this 684 KB registry (MEASURED 2026-11-02:
    0.76 s vs 0.10 s).  ``test_loader_equivalence`` asserts the two produce
    the SAME document on the real registry, so the fast path cannot change
    what the gate sees.
    """
    if getattr(yaml, "CSafeLoader", None) is not None:
        loaded = yaml.load(text, Loader=yaml.CSafeLoader)
    else:  # pragma: no cover - PyYAML built without libyaml
        loaded = yaml.safe_load(text)
    return loaded or {}


def _yaml_dump(doc: Dict[str, Any]) -> str:
    """Serialise a registry document the way ``registry.py`` does."""
    if getattr(yaml, "CSafeDumper", None) is not None:
        return yaml.dump(doc, Dumper=yaml.CSafeDumper, sort_keys=False, allow_unicode=True)
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)  # pragma: no cover


def _load_registry(path: pathlib.Path) -> Dict[str, Any]:
    """Load a registry file the way ``registry.py`` loads it.

    ``registry.py:255-261``::

        def _load_yaml(path: Path) -> Dict[str, Any]:
            if yaml is None or not path.is_file():
                return {}
            try:
                return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                return {}
    """
    if not path.is_file():
        return {}
    try:
        return _yaml_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rows(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return (doc or {}).get("metrics") or {}


def _paper_facing_ids() -> set:
    """The paper-line class, DERIVED from code -- not hand-listed.

    ``LINE_METRICS`` (registry.py:103-135) is the registry's own declaration
    of which metric ids belong to a paper line.
    """
    from molmetal.scripts.registry import LINE_METRICS

    return {metric_id for mapping in LINE_METRICS.values() for metric_id in mapping}


def _caveat_state(value: Any) -> Optional[str]:
    """Mirror ``assess_citability``'s caveat handling exactly.

    registry.py:1245-1247::

        "headline_caveat": (
            caveat.strip() if isinstance(caveat, str) and caveat.strip() else None
        ),
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _payload_path(rel: str) -> Optional[pathlib.Path]:
    """Resolve a registry row's ``source_json`` against the repo root."""
    if not isinstance(rel, str) or not rel:
        return None
    candidate = (REPO / rel).resolve()
    try:
        candidate.relative_to(REPO.resolve())
    except ValueError:
        return None  # a row may not point outside the repo
    return candidate


def _load_payload(rel: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return ``(payload, why)``; ``payload`` is None when unreadable."""
    path = _payload_path(rel)
    if path is None:
        return None, f"source_json {rel!r} does not resolve inside the repo"
    if not path.is_file():
        return None, f"source_json {rel!r} is not on disk"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed payload
        return None, f"source_json {rel!r} is unreadable: {exc}"
    if not isinstance(data, dict):
        return None, f"source_json {rel!r} is not a JSON object"
    return data, "ok"


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _is_headline(row: Dict[str, Any]) -> bool:
    return bool(row.get(KEY_QUOTABLE)) and bool(row.get(KEY_HEADLINE_GRADE))


def _is_governed(row: Dict[str, Any]) -> bool:
    return _is_headline(row) and row.get(KEY_ROLE) == ROLE_PAPER_LINE


def _violations(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every way the registry at ``doc`` breaks the gate.

    Each violation names the metric and the reason, so a failure says WHICH
    metric and WHICH clause -- not "the gate broke".
    """
    out: List[Dict[str, Any]] = []
    for metric_id, row in sorted(_rows(doc).items()):
        if not _is_headline(row):
            continue  # R5's converse: a non-headline row needs no caveat
        governed = _is_governed(row)
        row_caveat = _caveat_state(row.get(KEY_CAVEAT))

        if governed and row_caveat is None:
            out.append(
                {
                    "metric_id": metric_id,
                    "kind": "registry_caveat_missing",
                    "detail": (
                        "quotable + headline_grade + registry_role=paper_line, but the "
                        f"registry row's {KEY_CAVEAT!r} is absent/blank "
                        f"(raw={row.get(KEY_CAVEAT)!r})"
                    ),
                }
            )

        payload, why = _load_payload(row.get(KEY_SOURCE_JSON))
        if payload is None:
            if governed:
                out.append(
                    {
                        "metric_id": metric_id,
                        "kind": "payload_unreadable",
                        "detail": why,
                    }
                )
            continue

        payload_caveat = _caveat_state(payload.get(KEY_CAVEAT))

        if governed and payload_caveat is None:
            out.append(
                {
                    "metric_id": metric_id,
                    "kind": "payload_caveat_missing",
                    "detail": (
                        f"the payload {row.get(KEY_SOURCE_JSON)!r} carries no "
                        f"{KEY_CAVEAT!r}; a regenerate would erase any registry-only "
                        "caveat, so the disclosure is not durable"
                    ),
                }
            )

        # R5 -- parity.  Holds for OUT-OF-SCOPE rows too: a row may not carry a
        # caveat the payload does not (hand-injected, erased by a regenerate).
        if row_caveat != payload_caveat:
            out.append(
                {
                    "metric_id": metric_id,
                    "kind": "caveat_not_durable",
                    "detail": (
                        f"registry {KEY_CAVEAT!r}={row_caveat!r} != payload "
                        f"{KEY_CAVEAT!r}={payload_caveat!r} "
                        f"(source_json={row.get(KEY_SOURCE_JSON)!r})"
                    ),
                }
            )
    return out


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_doc() -> Dict[str, Any]:
    return _real_doc_cache()


@pytest.fixture(scope="module")
def real_rows(real_doc) -> Dict[str, Dict[str, Any]]:
    return _rows(real_doc)


@pytest.fixture(scope="module")
def governed_ids(real_rows) -> List[str]:
    return sorted(mid for mid, row in real_rows.items() if _is_governed(row))


def _shadow(tmp_path: pathlib.Path, mutate) -> pathlib.Path:
    """Write a SHADOW copy of the registry under ``tmp_path`` and mutate it.

    NEVER the repo's own file: the copy is asserted to live outside REPO.
    """
    import copy

    doc = copy.deepcopy(_real_doc_cache())
    metrics = (doc.get("metrics") or {})
    mutate(metrics)
    out = tmp_path / "metrics_shadow.yaml"
    out.write_text(_yaml_dump(doc), encoding="utf-8")
    assert REPO not in out.resolve().parents, "the shadow registry must live outside the repo"
    return out


_REAL_DOC: Optional[Dict[str, Any]] = None


def _real_doc_cache() -> Dict[str, Any]:
    """Parse the repo's registry ONCE per session (682 KB, 0.1 s with the C loader)."""
    global _REAL_DOC
    if _REAL_DOC is None:
        _REAL_DOC = _load_registry(REGISTRY)
    return _REAL_DOC


# ---------------------------------------------------------------------------
# R4 -- the scope is derived from code, not hand-listed
# ---------------------------------------------------------------------------


def test_loader_equivalence():
    """The fast loader is the same loader.

    ``registry.py:255-261`` reads the registry with ``yaml.safe_load``; this
    module reads it with PyYAML's ``CSafeLoader`` (the C implementation of the
    same SafeLoader).  Assert they produce the SAME document, so the speed-up
    cannot change what the gate sees.
    """
    text = REGISTRY.read_text(encoding="utf-8")
    assert yaml.safe_load(text) == _yaml_load(text), (
        "the C loader and yaml.safe_load disagree on the real registry"
    )


def test_governed_scope_matches_line_metrics(real_rows):
    """The paper-facing class is re-derived from ``registry.py`` on every run.

    If a paper line is added to ``LINE_METRICS`` and the registry has not been
    regenerated, the two sets drift and this fails -- so the gate cannot
    silently stop watching a row.
    """
    from_registry = {mid for mid, row in real_rows.items() if row.get(KEY_ROLE) == ROLE_PAPER_LINE}
    from_code = _paper_facing_ids()
    assert from_registry == from_code, (
        "paper-facing scope drifted: "
        f"registry-only={sorted(from_registry - from_code)} "
        f"code-only={sorted(from_code - from_registry)}"
    )


# ---------------------------------------------------------------------------
# R2 / R3 -- the real registry passes
# ---------------------------------------------------------------------------


def test_real_registry_passes_the_gate(real_doc):
    violations = _violations(real_doc)
    assert not violations, "gate violations:\n" + "\n".join(
        f"  {v['metric_id']}: {v['kind']}: {v['detail']}" for v in violations
    )


def test_the_governed_population_is_not_empty(governed_ids):
    assert governed_ids, (
        "no row is quotable + headline_grade + paper_line; the gate would be "
        "watching an empty set and would prove nothing"
    )


@pytest.mark.parametrize("metric_id", sorted({"m19_lomo", "m21_nci60_9cell", "m2_r2_per_cell"}))
def test_named_paper_headlines_are_governed(real_rows, metric_id):
    """The three rows Wf-A203's predecessor waves touched are in scope."""
    row = real_rows.get(metric_id)
    assert row is not None, f"{metric_id} is absent from the registry"
    assert _is_governed(row), (
        f"{metric_id} is not governed: quotable={row.get(KEY_QUOTABLE)} "
        f"headline_grade={row.get(KEY_HEADLINE_GRADE)} "
        f"role={row.get(KEY_ROLE)}"
    )


@pytest.mark.parametrize("metric_id", sorted({"m19_lomo", "m21_nci60_9cell", "m2_r2_per_cell"}))
def test_registry_row_carries_a_non_empty_caveat(real_rows, metric_id):
    caveat = real_rows[metric_id].get(KEY_CAVEAT)
    assert isinstance(caveat, str), f"{metric_id}: caveat is not a string ({caveat!r})"
    assert caveat.strip(), f"{metric_id}: caveat is blank"


@pytest.mark.parametrize("metric_id", sorted({"m19_lomo", "m21_nci60_9cell", "m2_r2_per_cell"}))
def test_payload_carries_the_caveat(real_rows, metric_id):
    """R3 -- the DURABLE direction: the payload on disk carries the caveat."""
    rel = real_rows[metric_id][KEY_SOURCE_JSON]
    payload, why = _load_payload(rel)
    assert payload is not None, why
    caveat = payload.get(KEY_CAVEAT)
    assert isinstance(caveat, str) and caveat.strip(), (
        f"{metric_id}: payload {rel} carries no non-empty {KEY_CAVEAT!r}"
    )


@pytest.mark.parametrize("metric_id", sorted({"m19_lomo", "m21_nci60_9cell", "m2_r2_per_cell"}))
def test_assess_citability_reproduces_the_caveat_from_the_payload(real_rows, metric_id):
    """A regenerate would NOT erase the caveat -- proven through the very
    function ``build_metrics`` calls (registry.py:1426-1429)."""
    from molmetal.scripts.registry import assess_citability

    row = real_rows[metric_id]
    payload, why = _load_payload(row[KEY_SOURCE_JSON])
    assert payload is not None, why
    observed = assess_citability(
        payload, row.get("value"), row[KEY_SOURCE_JSON], None
    )
    assert observed[KEY_CAVEAT] == row[KEY_CAVEAT], (
        f"{metric_id}: assess_citability rebuilt {observed[KEY_CAVEAT]!r} from the "
        f"payload but the registry row holds {row[KEY_CAVEAT]!r}"
    )
    assert observed[KEY_QUOTABLE] is True
    assert observed[KEY_HEADLINE_GRADE] is True


# ---------------------------------------------------------------------------
# R5 -- parity, including the OUT-OF-SCOPE headline rows
# ---------------------------------------------------------------------------


def test_headline_rows_registry_caveat_equals_payload_caveat(real_rows):
    """Applies to EVERY quotable + headline_grade row, governed or not.

    This is the invariant that keeps the two artifact-role rows honest: a row
    may not carry a registry caveat the payload does not.
    """
    checked, problems = 0, []
    for metric_id, row in sorted(real_rows.items()):
        if not _is_headline(row):
            continue
        payload, why = _load_payload(row.get(KEY_SOURCE_JSON))
        if payload is None:
            continue
        checked += 1
        row_caveat = _caveat_state(row.get(KEY_CAVEAT))
        payload_caveat = _caveat_state(payload.get(KEY_CAVEAT))
        if row_caveat != payload_caveat:
            problems.append(
                f"{metric_id}: registry={row_caveat!r} != payload={payload_caveat!r} "
                f"({row.get(KEY_SOURCE_JSON)})"
            )
    assert checked, "no quotable + headline_grade row had a readable payload"
    assert not problems, "registry/payload caveat drift:\n" + "\n".join(problems)


def test_no_headline_row_carries_a_registry_only_caveat(real_rows):
    """The negative form of R5, stated over the whole registry."""
    offenders = []
    for metric_id, row in sorted(real_rows.items()):
        caveat = row.get(KEY_CAVEAT)
        if caveat is None:
            continue
        payload, _why = _load_payload(row.get(KEY_SOURCE_JSON))
        if payload is None:
            continue
        if _caveat_state(caveat) != _caveat_state(payload.get(KEY_CAVEAT)):
            offenders.append(metric_id)
    assert not offenders, (
        "rows whose registry caveat has no payload counterpart (a regenerate "
        f"erases it): {offenders}"
    )


# ---------------------------------------------------------------------------
# the converse -- a quotable-but-not-headline row needs NO caveat
# ---------------------------------------------------------------------------


def test_quotable_but_not_headline_rows_need_no_caveat(real_rows, governed_ids):
    """The converse must not fire spuriously.

    MEASURED 2026-11-02: 73 rows are ``quotable`` and NOT ``headline_grade``,
    5 of them paper-facing (``m14_decode_ratio``, ``m7_diversity``,
    ``m1_ru_auc``, ``l6_closure``, ``m6_kappa``).  Four of those five carry NO
    caveat today and the gate still passes -- a caveat is required only of a
    TABLE-GRADE row.
    """
    soft = {
        mid for mid, row in real_rows.items()
        if row.get(KEY_QUOTABLE) and not row.get(KEY_HEADLINE_GRADE)
    }
    assert soft, "no quotable-but-not-headline row exists; the converse is untested"
    assert not (soft & set(governed_ids)), (
        f"overlap between the soft set and the governed set: {sorted(soft & set(governed_ids))}"
    )
    caveatless_soft = sorted(
        mid for mid in soft if _caveat_state(real_rows[mid].get(KEY_CAVEAT)) is None
    )
    assert caveatless_soft, (
        "every quotable-but-not-headline row carries a caveat; the converse is "
        "untested because nothing exercises the no-caveat-needed path"
    )


def test_a_soft_row_without_a_caveat_does_not_fail_the_gate(tmp_path, real_rows):
    """Bite of the CONVERSE: the gate stays green for a soft caveat-less row."""
    soft_paper = sorted(
        mid for mid, row in real_rows.items()
        if row.get(KEY_QUOTABLE)
        and not row.get(KEY_HEADLINE_GRADE)
        and row.get(KEY_ROLE) == ROLE_PAPER_LINE
        and _caveat_state(row.get(KEY_CAVEAT)) is None
    )
    assert soft_paper, "no caveat-less quotable-but-not-headline paper row exists"

    def mutate(metrics):
        for mid in soft_paper:
            metrics[mid].pop(KEY_CAVEAT, None)

    shadow = _shadow(tmp_path, mutate)
    assert not _violations(_load_registry(shadow)), (
        "the gate fired on a quotable-but-not-headline row: "
        f"{_violations(_load_registry(shadow))}"
    )


# ---------------------------------------------------------------------------
# THE BITES -- the gate must fail when it should
# ---------------------------------------------------------------------------


def _violation_kinds(doc) -> Dict[str, List[str]]:
    kinds: Dict[str, List[str]] = {}
    for v in _violations(doc):
        kinds.setdefault(v["kind"], []).append(v["metric_id"])
    return kinds


def test_bite_1_blank_caveat_fails(tmp_path):
    """Bite 1: the caveat is blanked to the empty string."""
    def mutate(metrics):
        metrics["m21_nci60_9cell"][KEY_CAVEAT] = ""

    shadow = _shadow(tmp_path, mutate)
    kinds = _violation_kinds(_load_registry(shadow))
    assert "m21_nci60_9cell" in kinds.get("registry_caveat_missing", []), (
        f"a blanked caveat was not caught; kinds={kinds}"
    )


def test_bite_2_whitespace_caveat_fails(tmp_path):
    """Bite 2: the caveat is whitespace only."""
    def mutate(metrics):
        metrics["m21_nci60_9cell"][KEY_CAVEAT] = "   \n\t  \r\n "

    shadow = _shadow(tmp_path, mutate)
    kinds = _violation_kinds(_load_registry(shadow))
    assert "m21_nci60_9cell" in kinds.get("registry_caveat_missing", []), (
        f"a whitespace-only caveat was not caught; kinds={kinds}"
    )


def test_bite_3_removed_caveat_key_fails(tmp_path):
    """Bite 3: the caveat key is removed from the row entirely."""
    def mutate(metrics):
        metrics["m21_nci60_9cell"].pop(KEY_CAVEAT, None)

    shadow = _shadow(tmp_path, mutate)
    kinds = _violation_kinds(_load_registry(shadow))
    assert "m21_nci60_9cell" in kinds.get("registry_caveat_missing", []), (
        f"a removed caveat key was not caught; kinds={kinds}"
    )


def test_bite_4_promoted_artifact_row_without_a_caveat_fails(tmp_path):
    """Bite 4: THE HAZARD CLASS.

    A caveat-less artifact row is promoted to the paper-facing surface.  The
    gate must fail and name it -- this is the "third caveat-less headline"
    arriving by a route other than hand-editing a paper row.
    """
    doc = _load_registry(REGISTRY)
    candidates = sorted(
        mid for mid, row in _rows(doc).items()
        if _is_headline(row)
        and row.get(KEY_ROLE) != ROLE_PAPER_LINE
        and _caveat_state(row.get(KEY_CAVEAT)) is None
    )
    assert candidates, (
        "no caveat-less artifact row is headline-grade; bite 4 has no target"
    )
    target = candidates[0]

    def mutate(metrics):
        metrics[target][KEY_ROLE] = ROLE_PAPER_LINE

    shadow = _shadow(tmp_path, mutate)
    kinds = _violation_kinds(_load_registry(shadow))
    assert target in kinds.get("registry_caveat_missing", []), (
        f"promoting {target} to paper_line was not caught; kinds={kinds}"
    )
    assert target in kinds.get("payload_caveat_missing", []), (
        f"{target}'s payload carries no caveat either; kinds={kinds}"
    )


def test_bite_5_registry_only_caveat_is_flagged_not_durable(tmp_path):
    """Bite 5: the caveat lives ONLY in the registry row.

    ``registry.py generate`` rebuilds the row from the payload
    (registry.py:1362 -> 1426-1429 -> 2056 -> 2072-2079; the 2026-11-02
    rendering said 1361 / 2061 / 2077-2079 -- corrected 2026-11-02), so a
    registry-only caveat is erased by the next regenerate.  The gate must
    say so.
    """
    def mutate(metrics):
        metrics["m21_nci60_9cell"][KEY_CAVEAT] = (
            "hand-injected registry-only caveat: not in the payload, so a "
            "regenerate erases it"
        )

    shadow = _shadow(tmp_path, mutate)
    kinds = _violation_kinds(_load_registry(shadow))
    assert "m21_nci60_9cell" in kinds.get("caveat_not_durable", []), (
        f"a registry-only caveat was not flagged; kinds={kinds}"
    )


def test_bite_6_shadow_copy_lives_outside_the_repo(tmp_path):
    """Guard on the bites themselves: the shadow is a /tmp copy."""
    shadow = _shadow(tmp_path, lambda metrics: None)
    assert REPO not in shadow.resolve().parents
    assert str(shadow.resolve()).startswith("/tmp") or "pytest" in str(shadow.resolve()), (
        f"the shadow registry is not in a temp area: {shadow.resolve()}"
    )


def test_bite_6b_the_shadow_file_round_trips_to_the_same_verdict(tmp_path):
    """The bites bite the FILE, not an in-memory fiction.

    The shadow is written to disk, read back with the same loader
    ``registry.py`` uses, and the verdict on the re-read document is asserted
    to equal the verdict on the document that was written.
    """
    def mutate(metrics):
        metrics["m21_nci60_9cell"][KEY_CAVEAT] = ""

    shadow = _shadow(tmp_path, mutate)
    from_file = _load_registry(shadow)
    kinds = _violation_kinds(from_file)
    assert "m21_nci60_9cell" in kinds.get("registry_caveat_missing", []), (
        f"the shadow FILE, read back from disk, did not trip the gate; kinds={kinds}"
    )
    # The repo's own registry is untouched by the bite.
    assert not _violations(_load_registry(REGISTRY)), "the bite reached the repo's registry"


def test_bite_7_a_payload_without_a_caveat_rebuilds_a_row_without_one():
    """The mechanism, demonstrated on the function itself.

    ``assess_citability`` is the ONLY source of a row's ``headline_caveat``
    (registry.py:1239 -> 1245-1247).  Strip the key from a payload and the
    rebuilt row has no caveat -- so a payload that does not carry the caveat
    cannot yield a row that does.
    """
    from molmetal.scripts.registry import assess_citability

    payload = {
        "status": "MEASURED",
        "value": 0.5,
        "n_seeds": 5,
        "dataset": "x.csv",
        "split": "4-fold scaffold split",
    }
    assert assess_citability(payload, 0.5, "metrics/by_metric/x.json", None)[KEY_CAVEAT] is None
    payload[KEY_CAVEAT] = "a computed disclosure"
    assert assess_citability(payload, 0.5, "metrics/by_metric/x.json", None)[KEY_CAVEAT] == (
        "a computed disclosure"
    )
    payload[KEY_CAVEAT] = "   "
    assert assess_citability(payload, 0.5, "metrics/by_metric/x.json", None)[KEY_CAVEAT] is None


# ---------------------------------------------------------------------------
# no-GPU contract
# ---------------------------------------------------------------------------


def test_no_gpu_is_used():
    """CPU-only.  The gate reads YAML/JSON and never touches a device."""
    assert "torch" not in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "torch is loaded and a GPU is visible"
    )


# ---------------------------------------------------------------------------
# Wf-A205 repair (2026-11-02) -- the CITATION-LINE guard
#
# WHY THIS SECTION EXISTS.  The proof file
# ``gate/payload_direction_proof.json`` cites 8 statements in
# ``molmetal/scripts/registry.py`` BY LINE NUMBER.  Three of those numbers
# had ROTTED (1361 -> 1362, 2061 -> 2056, 2077 -> 2072) and NOTHING noticed:
# the repo's own locator auditor (Wf-A200,
# ``molmetal/scripts/wfa200_crosswave_locator_audit.py``) cannot see a
# machine-readable ``{"file": ..., "line": ...}`` record -- its ``LOCATOR_RE``
# needs path-then-colon-then-digits ADJACENCY -- so it flagged 0 of the 3,
# and its single ``live_drift`` on that tree was a FALSE POSITIVE.
#
# These tests resolve every citation against the file ON DISK, by CONTENT,
# and fail when a citation points at a line that does not carry its text --
# unless a dated correction in the artifact records the move.  The anchors
# below are written INDEPENDENTLY of ``gate/evidence.py`` (which emits the
# artifact): if the two ever disagree, the cross-check test fails.
# ---------------------------------------------------------------------------

PROOF_PATH = (
    REPO
    / "molmetal"
    / "reports"
    / "wfa205_annotation_reproducibility"
    / "gate"
    / "payload_direction_proof.json"
)
REGISTRY_PY = REPO / "molmetal" / "scripts" / "registry.py"

#: citation index -> (scope, anchor block).  ``scope`` is the enclosing
#: top-level ``def``/``class``, or the literal ``"module"``.
CITATION_ANCHORS = {
    0: ("build_metrics", ["out: Dict[str, Any] = {}"]),
    1: ("assess_citability", ['caveat = payload.get("headline_caveat")']),
    2: (
        "assess_citability",
        [
            '"headline_caveat": (',
            "caveat.strip() if isinstance(caveat, str) and caveat.strip() else None",
            "),",
        ],
    ),
    3: (
        "build_metrics",
        [
            "rec.update(",
            'assess_citability(payload, rec["value"], _rel(src_path), tracked)',
            ")",
        ],
    ),
    4: ("generate", ["metrics_doc = {", '"schema": "molmetal-registry-metrics-v1",']),
    5: ("generate", ["text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)"]),
    6: ("main", ['if r.get("registry_role") != "paper_line":', "continue"]),
    7: ("module", ["LINE_METRICS: Dict[str, Dict[str, str]] = {"]),
}


def _sha256_of(path: pathlib.Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _enclosing_scope(lines: List[str], idx: int) -> Optional[str]:
    """The nearest enclosing top-level ``def``/``class`` name, or None."""
    for i in range(idx, -1, -1):
        line = lines[i]
        if line.startswith("def ") or line.startswith("class "):
            return line.split("(")[0].split()[-1]
    return None


def _resolve_anchor(lines: List[str], block: List[str], scope: str) -> Optional[int]:
    """The 1-based line where ``block`` starts, or None when not unique."""
    n = len(block)
    hits: List[int] = []
    for i in range(len(lines) - n + 1):
        if [lines[i + k].strip() for k in range(n)] != block:
            continue
        found = _enclosing_scope(lines, i)
        if scope == "module":
            if found is not None:
                continue
        elif found != scope:
            continue
        hits.append(i + 1)
    return hits[0] if len(hits) == 1 else None


def _resolved_citation_lines() -> Dict[int, int]:
    """{citation index: the line in registry.py that carries its text}."""
    lines = REGISTRY_PY.read_text(encoding="utf-8").splitlines()
    out: Dict[int, int] = {}
    for idx, (scope, block) in CITATION_ANCHORS.items():
        resolved = _resolve_anchor(lines, block, scope)
        assert resolved is not None, (
            f"citation {idx}'s anchor no longer resolves in registry.py; the "
            "source moved and the artifact must be re-anchored"
        )
        out[idx] = resolved
    return out


def _correction_entries(proof: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    entries: Dict[int, Dict[str, Any]] = {}
    for corr in proof.get("corrections", []):
        assert corr.get("dated"), "a correction block entry carries no date"
        for entry in corr.get("entries", []):
            entry = dict(entry)
            entry["dated"] = corr.get("dated")
            entries[entry["citation_index"]] = entry
    return entries


def test_every_code_citation_points_at_a_line_that_carries_its_text():
    """The guard that was missing: a citation must resolve to its own line.

    On the PRE-correction artifact this FAILS -- three citations point at a
    line that does not carry their text and no dated correction records the
    move.  MEASURED 2026-11-02: with the pre-correction proof file
    (the 8 citations at 1361 / 2061 / 2077 and NO ``corrections`` key) this
    test raises on all three; with the corrected artifact it passes.
    """
    proof = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    citations = proof["code_citations"]
    assert len(citations) == len(CITATION_ANCHORS), (
        f"the proof carries {len(citations)} citations but this guard anchors "
        f"{len(CITATION_ANCHORS)}; re-anchor before trusting it"
    )
    resolved = _resolved_citation_lines()
    corrections = _correction_entries(proof)
    problems: List[str] = []
    for idx, cit in enumerate(citations):
        cited = cit["line"]
        want = resolved[idx]
        if cited == want:
            continue
        entry = corrections.get(idx)
        if entry is None:
            problems.append(
                f"code_citations[{idx}] cites line {cited} but its text is at "
                f"line {want}; no dated correction records the move"
            )
        elif entry.get("corrected_line") != want:
            problems.append(
                f"code_citations[{idx}]'s correction names line "
                f"{entry.get('corrected_line')} but its text is at line {want}"
            )
        elif entry.get("superseded_line") != cited:
            problems.append(
                f"code_citations[{idx}]'s correction preserves superseded line "
                f"{entry.get('superseded_line')} but the artifact still cites {cited}"
            )
    assert not problems, "; ".join(problems)


def test_the_correction_block_preserves_the_superseded_rendering():
    """A correction is ADDITIVE: it keeps the old rendering and dates the fix."""
    proof = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    citations = proof["code_citations"]
    corrections = _correction_entries(proof)
    assert corrections, (
        "the artifact carries drifted citations but no correction block; "
        "AGENTS.md 5.5 rule #6 requires the superseded rendering to be kept "
        "and corrected additively with a date"
    )
    resolved = _resolved_citation_lines()
    for idx, entry in corrections.items():
        assert idx < len(citations)
        superseded = entry.get("superseded_citation")
        assert isinstance(superseded, dict), f"correction {idx} keeps no verbatim record"
        assert superseded.get("line") == entry.get("superseded_line")
        assert superseded.get("text") == citations[idx].get("text")
        assert superseded.get("meaning") == citations[idx].get("meaning")
        # the superseded line really was wrong, and the corrected one is right
        assert resolved[idx] != entry["superseded_line"]
        assert resolved[idx] == entry["corrected_line"]
        assert entry.get("corrected_citation", {}).get("line") == resolved[idx]


def test_the_artifacts_own_citation_check_agrees_with_this_guard():
    """The artifact's self-report must agree with an INDEPENDENT resolution."""
    proof = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    check = proof.get("citation_line_check")
    assert check, "the artifact carries no citation_line_check"
    resolved = _resolved_citation_lines()
    assert check.get("registry_py_sha256") == _sha256_of(REGISTRY_PY), (
        "the artifact's citation check was made against a different "
        "registry.py; re-run gate/evidence.py"
    )
    for entry in check.get("citations", []):
        idx = entry["citation_index"]
        assert entry["resolved_line"] == resolved[idx]
    assert check.get("uncorrected_drift") == []

