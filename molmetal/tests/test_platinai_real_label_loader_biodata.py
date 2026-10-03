#!/usr/bin/env python3
"""Offline regression tests for the platinai_real_label_loader biodata mode.

Context (2026-10-03, "kill the false honest-negative"): the loader used to
assert in its docstring/meta that per-cell-line IC50 NUMERIC values are NOT
exposed by the public MB-Finder API.  That was false — POST
/api/v2/get-ceil-line-bio-data returns real measured values and is public.

These tests pin the CORRECTED route so it cannot silently regress:

- the endpoint name used for the per-cell-line fetch,
- the value-string parsing for BOTH live forms ("1.2" and "0.6 ± 0.1"),
- the compound-code join key (endpoint-C imageUrl basename -> endpoint-B imageId),
- the structure join (SMILES/mw/substance_id attached; unmatched kept + counted).

They run FULLY OFFLINE against a recorded fixture
(molmetal/tests/fixtures/mbfinder_a2780cis_biodata.json, recorded live
2026-10-03).  If the loader regresses — endpoint renamed, parser broken, join
key broken, join dropped — these tests fail.  Proof of that failure is
recorded in TODO/completed/pt_data_route_correction.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.scripts import platinai_real_label_loader as loader


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "mbfinder_a2780cis_biodata.json"
)
BIODATA_ENDPOINT = "get-ceil-line-bio-data"


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture()
def fake_post(monkeypatch, fixture):
    """Serve recorded pages from the fixture; record every call."""
    calls = []

    def _fake_post(endpoint, body, *, max_retries=4, timeout=20):
        calls.append((endpoint, dict(body)))
        if endpoint == BIODATA_ENDPOINT:
            page = str(body.get("page", 1))
            rows = fixture["biodata"]["pages"].get(page)
            if rows is None:
                return {
                    "data": [],
                    "meta": {"last_page": fixture["biodata"]["last_page"]},
                }
            return {
                "data": rows,
                "meta": {"last_page": fixture["biodata"]["last_page"]},
            }
        if endpoint == "get-substances":
            page = str(body.get("page", 1))
            rows = fixture["substances"]["pages"].get(page)
            if rows is None:
                return {"data": [], "meta": {}}
            return {"data": rows, "meta": {}}
        # Any other endpoint means the loader regressed to a wrong route.
        return None

    monkeypatch.setattr(loader, "_post", _fake_post)
    return calls


def test_parse_measurement_value_handles_both_live_forms():
    assert loader.parse_measurement_value("1.2") == (1.2, None)
    assert loader.parse_measurement_value("0.6 ± 0.1") == (0.6, 0.1)
    assert loader.parse_measurement_value("70 ± 10") == (70.0, 10.0)
    assert loader.parse_measurement_value("1") == (1.0, None)
    assert loader.parse_measurement_value(None) == (None, None)
    assert loader.parse_measurement_value("not a number") == (None, None)


def test_compound_code_extraction_matches_live_image_url():
    url = "https://api.mb-finder.org/storage/images/MB-Pt-000017.svg?v=1770252062"
    assert loader.compound_code_from_image_url(url) == "MB-Pt-000017"
    assert loader.compound_code_from_image_url("") == ""


def test_biodata_pagination_offline(fake_post):
    records, requests = loader.fetch_cell_line_biodata_pages(
        "A2780cis",
        max_pages=2,
        per_request_delay=0.0,
    )
    assert requests == 2
    assert len(records) == 10
    # Regression guard: every network call must hit the real endpoint name.
    assert {ep for ep, _ in fake_post} == {BIODATA_ENDPOINT}
    assert all(body.get("cell_line") == "A2780cis" for _, body in fake_post)
    first = records[0]
    assert first["compound_code"] == "MB-Pt-000017"
    assert first["value_mean"] == 1.2
    assert first["value_raw"] == "1.2"
    assert first["doi"] == "10.1021/ic8012359"
    # The "mean ± SD" form must be parsed as (0.6, 0.1), not dropped.
    pm = [r for r in records if r["value_sd"] is not None]
    assert pm, "no 'mean ± SD' row parsed"
    assert pm[0]["compound_code"] == "MB-Pt-000132"
    assert (pm[0]["value_mean"], pm[0]["value_sd"]) == (0.6, 0.1)


def test_structure_join_offline(fake_post):
    records, _ = loader.fetch_cell_line_biodata_pages(
        "A2780cis",
        max_pages=2,
        per_request_delay=0.0,
    )
    index, index_requests = loader.build_structure_index(1545, 3, 0.0)
    assert index_requests == 3
    assert set(index) == {f"MB-Pt-{n:06d}" for n in range(5, 20)}
    assert index["MB-Pt-000017"]["substance_id"] == 13
    assert index["MB-Pt-000017"]["smiles"].startswith("CC(C)=[NH][Pt]1")
    assert index["MB-Pt-000017"]["mw"] == "648.67"

    joined, n_matched, n_unmatched = loader.join_structures(records, index)
    assert n_matched + n_unmatched == len(records)
    assert n_matched >= 1
    hit = [r for r in joined if r["compound_code"] == "MB-Pt-000017"][0]
    assert hit["smiles"] == index["MB-Pt-000017"]["smiles"]
    assert "[Pt]" in hit["smiles"]
    assert hit["substance_id"] == 13
    # Unmatched rows are kept (honest) with empty smiles, not dropped.
    assert len([r for r in joined if not r["smiles"]]) == n_unmatched


def test_biodata_mode_writes_outputs_offline(fake_post, tmp_path):
    records, bio_requests = loader.fetch_cell_line_biodata_pages(
        "A2780cis",
        max_pages=2,
        per_request_delay=0.0,
    )
    index, index_requests = loader.build_structure_index(1545, 3, 0.0)
    joined, n_matched, n_unmatched = loader.join_structures(records, index)
    paths = loader.write_biodata_outputs(
        joined,
        tmp_path,
        "smoke_biodata",
        {
            "cell_line": "A2780cis",
            "total_api_requests": bio_requests + index_requests,
            "n_matched_structures": n_matched,
            "n_unmatched_structures": n_unmatched,
        },
    )
    assert Path(paths["csv"]).exists()
    meta = json.loads(Path(paths["meta"]).read_text())
    assert meta["n_rows"] == 10
    assert meta["n_rows_with_numeric_value"] == 10
    assert meta["n_matched_structures"] == n_matched
    assert "corrects the earlier false honest-negative" in meta["framing"]
    header = Path(paths["csv"]).read_text().splitlines()[0]
    assert "value_mean" in header and "smiles" in header
