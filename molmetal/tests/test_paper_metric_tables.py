"""MET-11 tests — the paper tables must be derived, never transcribed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.scripts import paper_metric_tables as pmt


# ---------------------------------------------------------------------------
# pointer resolution
# ---------------------------------------------------------------------------


def test_resolve_walks_dicts_and_lists():
    doc = {"a": {"b": [10, {"c": 20}]}}
    assert pmt.resolve(doc, "/a/b/0") == 10
    assert pmt.resolve(doc, "/a/b/1/c") == 20


def test_resolve_supports_list_filters_on_a_field():
    doc = {"curve": [{"coverage": 0.1, "v": 1}, {"coverage": 0.5, "v": 2}]}
    assert pmt.resolve(doc, "/curve[coverage=0.5]/v") == 2


def test_resolve_unescapes_tilde_sequences():
    doc = {"a/b": {"c~d": 7}}
    assert pmt.resolve(doc, "/a~1b/c~0d") == 7


def test_resolve_raises_on_a_missing_key():
    with pytest.raises(KeyError):
        pmt.resolve({"a": 1}, "/b")


def test_resolve_raises_on_a_missing_filter_hit():
    with pytest.raises(KeyError):
        pmt.resolve({"c": [{"k": 1}]}, "/c[k=2]/v")


def test_resolve_rejects_a_pointer_without_a_leading_slash():
    with pytest.raises(ValueError):
        pmt.resolve({"a": 1}, "a")


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------


def test_a_missing_pointer_becomes_the_missing_marker(tmp_path: Path):
    (tmp_path / "x.json").write_text(json.dumps({"a": 1}))
    cache = pmt.ArtifactCache(tmp_path)
    text, status, where = pmt.P("x.json", "/nope").value(cache)
    assert (text, status) == (pmt.MISSING, "missing")
    assert "x.json" in where


def test_a_missing_file_becomes_the_missing_marker(tmp_path: Path):
    cache = pmt.ArtifactCache(tmp_path)
    text, status, _ = pmt.P("absent.json", "/a").value(cache)
    assert (text, status) == (pmt.MISSING, "missing")


def test_an_optional_cell_renders_dash_and_is_not_missing(tmp_path: Path):
    (tmp_path / "x.json").write_text(json.dumps({"a": 1}))
    cache = pmt.ArtifactCache(tmp_path)
    text, status, where = pmt.P("x.json", "/nope", optional=True).value(cache)
    assert (text, status) == ("-", "absent")
    assert "x.json" in where


def test_a_delta_cell_subtracts_two_pointers(tmp_path: Path):
    (tmp_path / "x.json").write_text(json.dumps({"a": 0.9, "b": 0.4}))
    cache = pmt.ArtifactCache(tmp_path)
    text, status, _ = pmt.D("x.json", "/a", "/b").value(cache)
    assert status == "ok"
    assert text == "0.5000"


def test_formats_cover_ci_pct_int_and_raw():
    assert pmt._fmt([0.1, 0.2], "ci") == "[0.1000, 0.2000]"
    assert pmt._fmt(0.1234, "pct") == "12.34%"
    assert pmt._fmt(7.9, "int") == "7"
    assert pmt._fmt("BLOCKED", "raw") == "BLOCKED"


def test_an_unknown_format_is_an_error():
    with pytest.raises(ValueError):
        pmt._fmt(1.0, "f9")


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def _run(tmp_path: Path, monkeypatch, tables) -> dict:
    monkeypatch.setattr(pmt, "TABLES", tables)
    monkeypatch.setattr(pmt, "METRICS_DIR", tmp_path / "by_metric")
    (tmp_path / "by_metric").mkdir(exist_ok=True)
    out = tmp_path / "out"
    code = pmt.main(["--out-dir", str(out)])
    assert code == 0
    return json.loads((out / "manifest.json").read_text())


def test_values_are_read_from_the_artifact_not_hardcoded(tmp_path: Path, monkeypatch):
    (tmp_path / "by_metric").mkdir()
    art = tmp_path / "by_metric" / "x.json"

    def build() -> pmt.Table:
        return pmt.Table("T_x", "X", ["Value"], rows=[pmt.Row("a", [pmt.P("x.json", "/a")])])

    art.write_text(json.dumps({"a": 0.1234}))
    _run(tmp_path, monkeypatch, (build,))
    md = (tmp_path / "out" / "T_x.md").read_text()
    assert "0.1234" in md

    art.write_text(json.dumps({"a": 0.4321}))
    pmt.main(["--out-dir", str(tmp_path / "out")])
    md = (tmp_path / "out" / "T_x.md").read_text()
    assert "0.4321" in md
    assert "0.1234" not in md


def test_manifest_records_sources_and_missing_counts(tmp_path: Path, monkeypatch):
    (tmp_path / "by_metric").mkdir()
    (tmp_path / "by_metric" / "x.json").write_text(json.dumps({"a": 1}))

    def build() -> pmt.Table:
        return pmt.Table(
            "T_x",
            "X",
            ["Value"],
            rows=[pmt.Row("ok", [pmt.P("x.json", "/a")]), pmt.Row("bad", [pmt.P("x.json", "/b")])],
        )

    manifest = _run(tmp_path, monkeypatch, (build,))
    assert manifest["missing_total"] == 1
    assert manifest["tables"][0]["missing"]
    assert "x.json" in manifest["sources"]


def test_strict_exits_nonzero_when_a_cell_is_missing(tmp_path: Path, monkeypatch):
    (tmp_path / "by_metric").mkdir()
    (tmp_path / "by_metric" / "x.json").write_text(json.dumps({"a": 1}))

    def build() -> pmt.Table:
        return pmt.Table("T_x", "X", ["Value"], rows=[pmt.Row("bad", [pmt.P("x.json", "/b")])])

    monkeypatch.setattr(pmt, "TABLES", (build,))
    monkeypatch.setattr(pmt, "METRICS_DIR", tmp_path / "by_metric")
    assert pmt.main(["--out-dir", str(tmp_path / "out"), "--strict"]) == 1


def test_strict_exits_zero_when_only_optional_cells_are_absent(tmp_path: Path, monkeypatch):
    (tmp_path / "by_metric").mkdir()
    (tmp_path / "by_metric" / "x.json").write_text(json.dumps({"a": 1}))

    def build() -> pmt.Table:
        return pmt.Table(
            "T_x", "X", ["Value"], rows=[pmt.Row("bad", [pmt.P("x.json", "/b", optional=True)])]
        )

    monkeypatch.setattr(pmt, "TABLES", (build,))
    monkeypatch.setattr(pmt, "METRICS_DIR", tmp_path / "by_metric")
    assert pmt.main(["--out-dir", str(tmp_path / "out"), "--strict"]) == 0


def test_a_rebuild_is_byte_identical(tmp_path: Path, monkeypatch):
    (tmp_path / "by_metric").mkdir()
    (tmp_path / "by_metric" / "x.json").write_text(json.dumps({"a": 1}))

    def build() -> pmt.Table:
        return pmt.Table("T_x", "X", ["Value"], rows=[pmt.Row("a", [pmt.P("x.json", "/a")])])

    monkeypatch.setattr(pmt, "TABLES", (build,))
    monkeypatch.setattr(pmt, "METRICS_DIR", tmp_path / "by_metric")
    out = tmp_path / "out"
    assert pmt.main(["--out-dir", str(out)]) == 0
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert pmt.main(["--out-dir", str(out)]) == 0
    second = {p.name: p.read_bytes() for p in out.iterdir()}
    assert first == second


def test_only_filters_by_id_prefix(tmp_path: Path, monkeypatch):
    (tmp_path / "by_metric").mkdir()
    (tmp_path / "by_metric" / "x.json").write_text(json.dumps({"a": 1}))

    def one() -> pmt.Table:
        return pmt.Table("T1_a", "A", ["Value"], rows=[pmt.Row("a", [pmt.P("x.json", "/a")])])

    def two() -> pmt.Table:
        return pmt.Table("T2_b", "B", ["Value"], rows=[pmt.Row("a", [pmt.P("x.json", "/a")])])

    monkeypatch.setattr(pmt, "TABLES", (one, two))
    monkeypatch.setattr(pmt, "METRICS_DIR", tmp_path / "by_metric")
    out = tmp_path / "out"
    assert pmt.main(["--out-dir", str(out), "--only", "T2"]) == 0
    assert not (out / "T1_a.md").exists()
    assert (out / "T2_b.md").exists()


def test_the_real_tables_build_without_a_missing_cell(tmp_path: Path):
    """The shipped specs must resolve against the shipped artifacts."""
    code = pmt.main(["--out-dir", str(tmp_path), "--strict"])
    assert code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["missing_total"] == 0
    assert len(manifest["tables"]) == len(pmt.TABLES)
    for entry in manifest["tables"]:
        assert (tmp_path / Path(entry["markdown"]).name).exists()
        assert (tmp_path / Path(entry["csv"]).name).exists()
