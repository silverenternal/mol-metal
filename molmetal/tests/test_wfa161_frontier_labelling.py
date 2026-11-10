"""Wf-A161 — regression tests for the frontier-labelling extractor.

Scope (deliberately narrow, per the branch brief):
  * the committed artefacts exist and are well-formed;
  * every source row carries a verdict from the allowed vocabulary;
  * every source row records a URL and a retrieved flag;
  * a retrieved row carries a verbatim excerpt;
  * a NOT-RETRIEVED row carries NO excerpt (we never guess a number);
  * the proposed house format string is present and is flagged as a
    proposal (it must NOT have been applied to any metric artefact).

It deliberately does NOT assert a specific paper count or a specific
verdict for a specific paper: the paper set may grow, and a verdict is
only meaningful together with the evidence stored beside it.

Run with:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
        molmetal/tests/test_wfa161_frontier_labelling.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.wfa161_frontier_labelling import (  # noqa: E402
    PAPER_KEYS,
    VERDICTS,
    PROPOSED_HOUSE_FORMAT,
)

REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wfa161_frontier_labelling"
SOURCES_JSON = REPORT_DIR / "sources.json"
REPORT_MD = REPORT_DIR / "report.md"


@pytest.fixture(scope="module")
def sources() -> dict:
    assert SOURCES_JSON.exists(), f"missing artefact: {SOURCES_JSON}"
    with SOURCES_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Artefact well-formedness
# ---------------------------------------------------------------------------
def test_sources_json_is_an_object_with_a_source_list(sources):
    assert isinstance(sources, dict)
    assert isinstance(sources.get("sources"), list)
    assert sources["sources"], "sources.json carries no source rows"


def test_report_md_exists_and_mentions_every_paper(sources):
    assert REPORT_MD.exists(), f"missing artefact: {REPORT_MD}"
    text = REPORT_MD.read_text(encoding="utf-8")
    assert text.strip(), "report.md is empty"
    for row in sources["sources"]:
        assert row["short"] in text, f"report.md does not mention {row['short']}"


def test_vocabulary_is_published(sources):
    assert set(sources.get("verdict_vocabulary", [])) == set(VERDICTS)


# ---------------------------------------------------------------------------
# Per-source-row contract
# ---------------------------------------------------------------------------
def test_every_source_row_has_a_key_url_and_verdict(sources):
    for row in sources["sources"]:
        assert row.get("key"), "source row has no key"
        assert row.get("url", "").startswith("https://"), (
            f"{row['key']}: url is not an https URL"
        )
        assert row.get("verdict") in VERDICTS, (
            f"{row['key']}: verdict {row.get('verdict')!r} is not in {VERDICTS}"
        )
        assert isinstance(row.get("retrieved"), bool), (
            f"{row['key']}: 'retrieved' must be a boolean"
        )


def test_source_keys_are_unique(sources):
    keys = [row["key"] for row in sources["sources"]]
    assert len(keys) == len(set(keys)), f"duplicate source keys: {keys}"


def test_every_row_has_a_verdict_reason(sources):
    for row in sources["sources"]:
        assert (row.get("verdict_reason") or "").strip(), (
            f"{row['key']}: verdict has no reason recorded"
        )


def test_retrieved_rows_carry_a_verbatim_excerpt(sources):
    """A retrieved row either shows the metric (with excerpts) or records
    that the metric is absent — never a silent gap."""
    for row in sources["sources"]:
        if not row["retrieved"]:
            continue
        headline = row.get("headline_sites") or []
        if row.get("n_metric_sites", 0) == 0:
            assert not headline, (
                f"{row['key']}: no metric site counted but a headline site is recorded"
            )
            continue
        assert headline, f"{row['key']}: retrieved but no headline site recorded"
        excerpts = [
            ex
            for site in headline
            for ex in (site.get("headline_excerpt") or [])
        ]
        assert excerpts, f"{row['key']}: retrieved but no verbatim excerpt recorded"
        for ex in excerpts:
            assert isinstance(ex, str) and ex.strip(), (
                f"{row['key']}: empty excerpt"
            )


def test_not_retrieved_rows_carry_no_excerpt(sources):
    """A failed fetch must never be papered over with a guessed number."""
    for row in sources["sources"]:
        if row["verdict"] != "NOT-RETRIEVED":
            continue
        assert not row["retrieved"], (
            f"{row['key']}: NOT-RETRIEVED but retrieved is True"
        )
        assert (row.get("error") or "").strip(), (
            f"{row['key']}: NOT-RETRIEVED without an error message"
        )
        assert not (row.get("headline_sites") or []), (
            f"{row['key']}: NOT-RETRIEVED but a headline site is recorded"
        )


def test_retrieved_rows_record_provenance(sources):
    for row in sources["sources"]:
        if not row["retrieved"]:
            continue
        assert (row.get("sha256") or "").strip(), (
            f"{row['key']}: retrieved without a sha256 provenance record"
        )
        assert row.get("abs_url", "").startswith("https://"), (
            f"{row['key']}: no abs_url recorded"
        )


def test_agree_verdict_requires_all_three_qualifiers_at_the_number(sources):
    """AGREES is only defensible if population, post-processing AND the
    definition are all printed at the number."""
    for row in sources["sources"]:
        if row["verdict"] != "AGREES":
            continue
        at = row.get("qualifier_summary_at_number") or {}
        for quad in ("population", "postproc", "definition"):
            assert at.get(quad), (
                f"{row['key']}: AGREES but {quad} is not printed at the number"
            )


def test_verdict_is_consistent_with_the_recorded_evidence(sources):
    """A row with no metric site at all can only be NOT-STATED, and a
    retrieved row can never be NOT-RETRIEVED."""
    for row in sources["sources"]:
        if not row["retrieved"]:
            assert row["verdict"] == "NOT-RETRIEVED", (
                f"{row['key']}: not retrieved but verdict is {row['verdict']}"
            )
            continue
        assert row["verdict"] != "NOT-RETRIEVED", (
            f"{row['key']}: retrieved but verdict is NOT-RETRIEVED"
        )
        if row.get("n_metric_sites", 0) == 0:
            assert row["verdict"] == "NOT-STATED", (
                f"{row['key']}: no metric site found but verdict is {row['verdict']}"
            )


# ---------------------------------------------------------------------------
# The proposal must stay a proposal
# ---------------------------------------------------------------------------
def test_proposed_house_format_is_recorded_and_labelled_a_proposal(sources):
    assert sources.get("proposed_house_format"), "no proposed house format recorded"
    assert PROPOSED_HOUSE_FORMAT.strip(), "proposed house format is empty"
    report = REPORT_MD.read_text(encoding="utf-8")
    assert "PROPOSAL ONLY" in report, "report.md does not flag the format as a proposal"
    assert "not been applied" in report.lower(), (
        "report.md does not state that the proposal was not applied"
    )


def test_proposed_house_format_carries_the_mandatory_slots():
    """The four slots the frontier papers leave to a distant caption."""
    for slot in ("N=", "post-processing", "denominator", "hydrogens"):
        assert slot in PROPOSED_HOUSE_FORMAT, (
            f"proposed house format is missing the {slot!r} slot"
        )


def test_paper_keys_in_module_match_the_committed_rows(sources):
    committed = {row["key"] for row in sources["sources"]}
    assert set(PAPER_KEYS) <= committed, (
        "the script's paper list contains a key with no committed source row: "
        f"{sorted(set(PAPER_KEYS) - committed)}"
    )
