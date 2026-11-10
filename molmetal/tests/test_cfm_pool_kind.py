# Licensed under the Apache License, Version 2.0.
"""Wf-A114 CLUSTER C — the pool-kind rule has ONE owner.

A CFM run leaves a ``generated_smiles.csv`` behind and TWO different code paths
can write one:

* ``wf_cfm_decode_dump_smiles.py`` writes ``metal_z`` — in its FIRST era
  (before 2026-10-12) a bond-head RECONSTRUCTION of real tmQM geometries, in
  its SECOND era (2026-10-12 onwards) the model's own FREE GENERATION;
* ``train_cfm_yuelbond_wasserstein.py`` writes ``n_frags_raw`` — the trainer's
  free GENERATION path.

The two are not comparable.  A reconstruction pool inherits the properties of
the real chemistry it started from, so calling its connectivity rate a
"fragmentation rate of the generator" is false.  The discriminator is NEVER the
run name (two of the reconstruction runs on disk are named ``cfm_fullcorpus_*``).

THE RULE IS NOT HEADER-ONLY, AND THAT IS THE POINT
---------------------------------------------------
One script wrote ONE header in TWO eras, so the header alone cannot decide a
``metal_z`` pool.  The rule therefore reads the header AND at most ONE data
row:

* an explicit ``pool_kind`` column is authoritative;
* ``n_frags_raw`` in the header (and no ``metal_z``) -> generation;
* ``metal_z`` in the header (and no ``n_frags_raw``) -> decided by the CONTENT
  of the first data row (``sanitized`` + ``metal_z``);
* both markers, a missing row or disagreeing signals -> unknown.

These tests pin the rule, its ONE owner, and the properties that make it safe
to use:

1. the rule never buckets silently: UNKNOWN is reported with a reason, and the
   refusal to use an unclassified pool is a separate, explicit act
   (``require_known_kind``);
2. the body of a pool is never read beyond ONE data row (the c7 pool holds
   12,668 rows);
3. the cross-check joins the scored rows back to the source rows BY INDEX, not
   by position — a positional zip misaligns everything after the first skipped
   row (it once reported 2299 false mismatches).

CPU-only, RDKit-only, no GPU.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import c7_scale_matched_pools as c7  # noqa: E402
from molmetal.scripts.cfm_fragmentation_gate import (  # noqa: E402
    score_run_dir,
    score_smiles_pool,
)
from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    CONTENT_MARKER,
    GENERATION_MARKER,
    KIND_GENERATION,
    KIND_RECONSTRUCTION,
    KIND_UNKNOWN,
    MEASUREMENT_MEANING,
    POOL_KIND_COLUMN,
    RECONSTRUCTION_MARKER,
    UnknownPoolKindError,
    VERDICT_SCOPE,
    classify_pool,
    classify_pool_header,
    require_known_kind,
)

REPORTS = REPO_ROOT / "molmetal" / "reports"
WRITER = REPO_ROOT / "molmetal" / "scripts" / "wf_cfm_decode_dump_smiles.py"

#: A ``metal_z`` pool: the header alone does NOT decide it, the content does.
RECON_HEADER = ["smiles", "sanitized", "metal_z"]
#: A generation pool carries the trainer's free-generation marker.
GEN_HEADER = ["smiles", "n_frags_raw", "largest_frag_frac"]


def _write_pool(path: Path, header, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# --- the rule itself --------------------------------------------------------


def test_a_metal_z_pool_with_sanitized_one_is_a_reconstruction(tmp_path):
    # era A: the bond head applied to a REAL tmQM geometry -- the row carries
    # the bond head's own sanitized flag (1) and the real cloud's metal.
    p = _write_pool(
        tmp_path / "pool.csv",
        RECON_HEADER,
        [{"smiles": "CCO", "sanitized": "1", "metal_z": "77"}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_RECONSTRUCTION
    assert kind.marker == CONTENT_MARKER
    assert kind.header == tuple(RECON_HEADER)
    assert kind.path == p
    assert kind.is_known
    # The reason states WHICH signal decided it.
    assert "content" in kind.reason
    assert "sanitized" in kind.reason
    assert "metal_z" in kind.reason


def test_a_metal_z_pool_with_sanitized_zero_and_no_metal_is_a_generation(tmp_path):
    # era B: the ODE sampler.  Its decode_chemistry dict has NO "sanitized"
    # key, so the column is structurally always 0, and metal_z is empty
    # unless the generated molecule happened to contain a matched metal.
    p = _write_pool(
        tmp_path / "pool.csv",
        RECON_HEADER,
        [{"smiles": "CCO", "sanitized": "0", "metal_z": ""}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_GENERATION
    assert kind.marker == CONTENT_MARKER
    assert kind.is_known
    assert "content" in kind.reason
    assert "sanitized" in kind.reason
    assert "metal_z" in kind.reason


def test_a_metal_z_header_alone_is_not_decidable():
    # The header says "this pool came from the decode-dump writer", which in
    # two eras means two different things.  The header-only answer is
    # therefore unknown, with a reason that says the content must be read.
    kind = classify_pool_header(["smiles", "sanitized", "metal_z"])
    assert kind.kind == KIND_UNKNOWN
    assert not kind.is_known
    assert "CONTENT" in kind.reason
    assert "metal_z" in kind.reason


def test_n_frags_raw_header_classifies_as_generation(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv",
        GEN_HEADER,
        [{"smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_GENERATION
    assert kind.marker == GENERATION_MARKER
    assert kind.is_known


def test_a_header_with_both_markers_is_unknown():
    # Both markers mean the pool's writer changed: the evidence is
    # contradictory, so the honest reading is "unknown", not "reconstruction".
    kind = classify_pool_header(["smiles", "metal_z", "n_frags_raw"])
    assert kind.kind == KIND_UNKNOWN
    assert kind.marker is None
    assert "both" in kind.reason.lower()
    assert RECONSTRUCTION_MARKER in kind.reason
    assert GENERATION_MARKER in kind.reason


def test_a_pool_with_both_markers_is_unknown(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv",
        ["smiles", "metal_z", "n_frags_raw"],
        [{"smiles": "CCO", "metal_z": "77", "n_frags_raw": 1}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_UNKNOWN
    assert kind.marker is None
    assert "both" in kind.reason.lower()


def test_a_header_with_neither_marker_is_unknown():
    kind = classify_pool_header(["smiles", "charge"])
    assert kind.kind == KIND_UNKNOWN
    assert kind.marker is None
    assert not kind.is_known
    # The reason names BOTH markers, so a reader knows what was looked for.
    assert RECONSTRUCTION_MARKER in kind.reason
    assert GENERATION_MARKER in kind.reason


def test_classify_pool_header_is_pure_and_rejects_a_bare_string():
    # A bare string would classify itself character by character.
    with pytest.raises(TypeError):
        classify_pool_header("metal_z")
    # Pure: the same header always gives the same answer, and the result
    # carries no path (nothing was read).
    a = classify_pool_header(["metal_z"])
    b = classify_pool_header(["metal_z"])
    assert a == b
    assert a.path is None


# --- the explicit pool_kind column is authoritative -------------------------


def test_an_explicit_pool_kind_column_of_generation_wins(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv",
        ["smiles", "pool_kind", "n_attempted"],
        [{"smiles": "CCO", "pool_kind": "generation", "n_attempted": 192}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_GENERATION
    assert kind.marker == POOL_KIND_COLUMN
    assert POOL_KIND_COLUMN in kind.reason


def test_an_explicit_pool_kind_column_of_reconstruction_wins(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv",
        ["smiles", "pool_kind"],
        [{"smiles": "CCO", "pool_kind": "reconstruction"}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_RECONSTRUCTION
    assert kind.marker == POOL_KIND_COLUMN


def test_a_bogus_pool_kind_value_is_unknown(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv",
        ["smiles", "pool_kind"],
        [{"smiles": "CCO", "pool_kind": "banana"}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_UNKNOWN
    assert kind.marker is None
    assert "banana" in kind.reason


def test_an_explicit_pool_kind_column_beats_the_header_markers(tmp_path):
    # The column is authoritative: it wins even over a header that carries a
    # marker (the writer stamps the column, so it is the newer truth).
    p = _write_pool(
        tmp_path / "pool.csv",
        ["smiles", "pool_kind", "n_frags_raw"],
        [{"smiles": "CCO", "pool_kind": "reconstruction", "n_frags_raw": 1}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_RECONSTRUCTION
    assert kind.marker == POOL_KIND_COLUMN


def test_a_pool_kind_column_without_a_data_row_is_unknown(tmp_path):
    p = tmp_path / "pool.csv"
    p.write_text("smiles,pool_kind\n", encoding="utf-8")
    kind = classify_pool(p)
    assert kind.kind == KIND_UNKNOWN
    assert "no data row" in kind.reason.lower()


# --- the content is parsed robustly -----------------------------------------


def test_the_sanitized_flag_is_parsed_as_a_number_not_a_string(tmp_path):
    for text in ("1", "1.0", " 1 "):
        p = _write_pool(
            tmp_path / f"pool_{text.strip()}.csv",
            RECON_HEADER,
            [{"smiles": "CCO", "sanitized": text, "metal_z": "77"}],
        )
        kind = classify_pool(p)
        assert kind.kind == KIND_RECONSTRUCTION, text


def test_an_unparseable_sanitized_flag_is_not_one(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv",
        RECON_HEADER,
        [{"smiles": "CCO", "sanitized": "yes", "metal_z": "77"}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_UNKNOWN


def test_an_empty_metal_z_may_be_spelled_in_several_ways(tmp_path):
    for text in ("", " ", "nan", "none", "null", "-"):
        p = _write_pool(
            tmp_path / "pool.csv",
            RECON_HEADER,
            [{"smiles": "CCO", "sanitized": "0", "metal_z": text}],
        )
        kind = classify_pool(p)
        assert kind.kind == KIND_GENERATION, text


# --- unknown is first-class -------------------------------------------------


def test_disagreeing_content_signals_are_unknown_and_name_the_disagreement(
    tmp_path,
):
    # sanitized says "the bond head succeeded", metal_z says "no metal was
    # recorded" -- the two signals cannot both be true of one row.
    p = _write_pool(
        tmp_path / "pool.csv",
        RECON_HEADER,
        [{"smiles": "CCO", "sanitized": "1", "metal_z": ""}],
    )
    kind = classify_pool(p)
    assert kind.kind == KIND_UNKNOWN
    assert kind.marker is None
    assert "disagree" in kind.reason.lower()
    assert "sanitized" in kind.reason
    assert "metal_z" in kind.reason


def test_a_metal_z_pool_without_a_data_row_is_unknown(tmp_path):
    p = tmp_path / "pool.csv"
    p.write_text("smiles,sanitized,metal_z\n", encoding="utf-8")
    kind = classify_pool(p)
    assert kind.kind == KIND_UNKNOWN
    assert "no data row" in kind.reason.lower()


def test_a_missing_file_is_unknown_not_an_exception(tmp_path):
    kind = classify_pool(tmp_path / "does_not_exist.csv")
    assert kind.kind == KIND_UNKNOWN
    assert kind.reason.startswith("unreadable: ")
    assert kind.path == tmp_path / "does_not_exist.csv"


def test_an_unreadable_path_is_unknown_not_an_exception(tmp_path):
    # A directory is an OSError too (IsADirectoryError).
    d = tmp_path / "a_directory"
    d.mkdir()
    kind = classify_pool(d)
    assert kind.kind == KIND_UNKNOWN
    assert kind.reason.startswith("unreadable: ")


def test_an_empty_file_is_unknown_with_the_empty_file_reason(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    kind = classify_pool(p)
    assert kind.kind == KIND_UNKNOWN
    assert kind.reason == "empty file: no header row"
    assert kind.header == ()


def test_require_known_kind_refuses_an_unknown_pool(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv", ["smiles", "charge"], [{"smiles": "CCO", "charge": "0"}]
    )
    with pytest.raises(UnknownPoolKindError) as excinfo:
        require_known_kind(p)
    err = excinfo.value
    assert err.path == p
    assert err.header == ("smiles", "charge")
    assert RECONSTRUCTION_MARKER in err.reason
    assert GENERATION_MARKER in err.reason


def test_require_known_kind_returns_the_kind_for_a_known_pool(tmp_path):
    p = _write_pool(
        tmp_path / "pool.csv",
        GEN_HEADER,
        [{"smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
    )
    kind = require_known_kind(p)
    assert kind.kind == KIND_GENERATION
    assert kind.marker == GENERATION_MARKER


def test_the_per_kind_semantics_have_one_owner():
    # MEASUREMENT_MEANING / VERDICT_SCOPE were duplicated between the C7 audit
    # and the fragmentation audit; they now live only here, so the two cannot
    # disagree about what a reconstruction pool's number MEANS.
    assert set(MEASUREMENT_MEANING) == {KIND_GENERATION, KIND_RECONSTRUCTION, KIND_UNKNOWN}
    assert set(VERDICT_SCOPE) == {KIND_GENERATION, KIND_RECONSTRUCTION, KIND_UNKNOWN}
    assert VERDICT_SCOPE[KIND_UNKNOWN] == "unclassified"
    assert "decoder" in MEASUREMENT_MEANING[KIND_RECONSTRUCTION].lower()
    assert "sampler" in MEASUREMENT_MEANING[KIND_GENERATION].lower()
    # A reconstruction pool's number must never be described as the
    # generator's fragmentation rate.
    assert "not a fragmentation rate of the generator" in (
        MEASUREMENT_MEANING[KIND_RECONSTRUCTION].lower()
    )
    # The reconstruction meaning is scoped to ERA A: it must no longer claim
    # "never the sampler" (the rule no longer decides from the header, and
    # the 2026-10-12+ decode-dump path IS the sampler).
    assert "never the sampler" not in MEASUREMENT_MEANING[KIND_RECONSTRUCTION].lower()
    # The generation meaning must cover BOTH free-generation writers.
    assert "wf_cfm_decode_dump_smiles.py" in MEASUREMENT_MEANING[KIND_GENERATION]


# --- at most ONE data row is read -------------------------------------------


def test_classify_pool_reads_only_the_header_row(tmp_path, monkeypatch):
    class _CountingFile:
        """Wraps a text file and counts the characters actually pulled out."""

        def __init__(self, handle):
            self._handle = handle
            self.chars_read = 0

        def __iter__(self):
            return self

        def __next__(self):
            line = next(self._handle)
            self.chars_read += len(line)
            return line

        def close(self):
            self._handle.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    p = tmp_path / "pool.csv"
    body = "".join(f"CCO,{i}\n" for i in range(5000))
    p.write_text("smiles,n_frags_raw\n" + body, encoding="utf-8")

    real_open = Path.open
    holder = {}

    def counting_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        wrapped = _CountingFile(handle)
        holder["wrapped"] = wrapped
        return wrapped

    monkeypatch.setattr(Path, "open", counting_open)
    kind = classify_pool(p)
    monkeypatch.undo()

    assert kind.kind == KIND_GENERATION
    assert kind.header == ("smiles", "n_frags_raw")
    # The body was NOT read: only the header line came out of the file.
    assert holder["wrapped"].chars_read < p.stat().st_size
    assert holder["wrapped"].chars_read <= len("smiles,n_frags_raw\n")


def test_classify_pool_reads_at_most_one_data_row(tmp_path, monkeypatch):
    # A metal_z pool NEEDS its content, so exactly ONE data row is read --
    # never the body (the c7 pool holds 12,668 rows).
    class _CountingFile:
        def __init__(self, handle):
            self._handle = handle
            self.chars_read = 0

        def __iter__(self):
            return self

        def __next__(self):
            line = next(self._handle)
            self.chars_read += len(line)
            return line

        def close(self):
            self._handle.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    p = tmp_path / "pool.csv"
    p.write_text(
        "smiles,sanitized,metal_z\n" + "".join(f"CCO,1,77\n" for _ in range(5000)),
        encoding="utf-8",
    )

    real_open = Path.open
    holder = {}

    def counting_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        wrapped = _CountingFile(handle)
        holder["wrapped"] = wrapped
        return wrapped

    monkeypatch.setattr(Path, "open", counting_open)
    kind = classify_pool(p)
    monkeypatch.undo()

    assert kind.kind == KIND_RECONSTRUCTION
    header_line = "smiles,sanitized,metal_z\n"
    first_row = "CCO,1,77\n"
    assert holder["wrapped"].chars_read <= len(header_line) + len(first_row)
    assert holder["wrapped"].chars_read < p.stat().st_size


# --- the writer stamps its own provenance -----------------------------------


def test_the_decode_dump_writer_stamps_its_pool_kind_and_attempt_count():
    # ROOT CAUSE FIX (Wf-A114D).  The writer had ONE header for TWO eras, so
    # the header could not say which era wrote a pool.  It now writes an
    # explicit `pool_kind` column (it always runs the ODE path) and the
    # number of draws it made (`n_attempted`), so no FUTURE pool is ambiguous.
    src = WRITER.read_text(encoding="utf-8")
    assert '"pool_kind": "generation"' in src
    assert '"n_attempted": int(n_tot)' in src
    assert '"pool_kind",' in src
    assert '"n_attempted",' in src
    # The stale banner is gone: the docstring must say the ERA, not deny the
    # 2026-10-12+ ODE path unconditionally.
    assert "!! THESE ARE NOT GENERATED MOLECULES. MEASURED 2026-10-11. !!" not in src
    assert "FALSE FOR EVERY FILE WRITTEN AFTER IT" in src
    assert "2026-10-12" in src


# --- the shipped pools ------------------------------------------------------


def test_every_shipped_pool_classifies_as_known():
    # MEASURED 2026-10-30 (Wf-A114D, the NEW content-based rule): the 9
    # ``molmetal/reports/cfm_*/generated_smiles.csv`` pools are 7 generation +
    # 2 reconstruction, 0 unknown.  cfm_fullcorpus_2026_10_10 MOVED from
    # reconstruction to generation: its first row is sanitized=0 with an empty
    # metal_z, i.e. the 2026-10-12+ ODE decode-dump path (the project's own
    # metrics/by_metric/cfm_free_generation_panel_seed0.json says the same
    # pool is free generation: 192 attempted, 66 valid).
    #
    # The audit's wider scan sees 13 pools (10 generation + 2 reconstruction +
    # 1 unknown), the unknown one being ``wf124_p2_cfm_full``.
    pools = sorted(p for p in REPORTS.glob("cfm_*/generated_smiles.csv") if p.is_file())
    kinds = [classify_pool(p) for p in pools]
    assert len(pools) == 9
    assert sum(1 for k in kinds if k.kind == KIND_GENERATION) == 7
    assert sum(1 for k in kinds if k.kind == KIND_RECONSTRUCTION) == 2
    assert sum(1 for k in kinds if k.kind == KIND_UNKNOWN) == 0
    recon = {p.parent.name for p, k in zip(pools, kinds) if k.kind == KIND_RECONSTRUCTION}
    assert recon == {
        "cfm_fullcorpus_b64_2026_10_11",
        "cfm_objective_terms_2026_10_11",
    }


def test_the_fullcorpus_pool_is_classified_as_generation():
    # The pool the defect named.  It carries `metal_z`, so the OLD rule called
    # it a reconstruction; its CONTENT (sanitized=0, metal_z empty) says it is
    # the ODE sampler's free generation.
    pool = REPORTS / "cfm_fullcorpus_2026_10_10" / "generated_smiles.csv"
    assert pool.is_file()
    kind = classify_pool(pool)
    assert kind.kind == KIND_GENERATION
    assert kind.marker == CONTENT_MARKER


def test_the_wider_scan_of_every_pool_on_disk():
    # MEASURED 2026-10-30: 13 pools under molmetal/reports/ -> 10 generation,
    # 2 reconstruction, 1 unknown (wf124_p2_cfm_full, a header with neither
    # marker: ``epoch,smiles``).
    pools = sorted(p for p in REPORTS.glob("*/generated_smiles.csv") if p.is_file())
    kinds = [classify_pool(p) for p in pools]
    assert len(pools) == 13
    assert sum(1 for k in kinds if k.kind == KIND_GENERATION) == 10
    assert sum(1 for k in kinds if k.kind == KIND_RECONSTRUCTION) == 2
    unknown = {p.parent.name for p, k in zip(pools, kinds) if k.kind == KIND_UNKNOWN}
    assert unknown == {"wf124_p2_cfm_full"}


# --- the index join (the 2299-false-mismatch bug) ---------------------------


def test_the_shuffled_pool_cross_check_joins_by_index_not_by_position(tmp_path):
    # REGRESSION (Wf-A114, 2026-10-30).  The pool skips empty and unparseable
    # strings, so the scored rows and the source rows have DIFFERENT lengths.
    # Lining them up POSITIONALLY misaligns everything after the first skip:
    # on the c7 pool (one unparseable row at index 10370) the naive zip
    # reported 2299 "mismatches" against a re-derivation that agrees on all
    # 12668 scored rows.  The join must be by INPUT INDEX.
    valid = [
        ("CCO", 1),
        ("C.C.C", 3),
        ("CC(C)(C)", 1),
        ("C.C.C.C", 4),
        ("c1ccccc1O", 1),
        ("C.C", 2),
    ]
    skipped = [("not_a_smiles", 0), ("", 0), ("also not a smiles", 0)]
    rows = [
        {"epoch": 0, "smiles": s, "n_frags_raw": n, "largest_frag_frac": 0.0}
        for s, n in valid + skipped
    ]
    # Shuffled so the skips are NOT all at the end.
    random.Random(20261030).shuffle(rows)

    d = tmp_path / "run"
    _write_pool(
        d / "generated_smiles.csv",
        ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"],
        rows,
    )
    scored = score_run_dir(d)
    cc = scored["sources"]["generated_smiles_csv"]["csv_column_cross_check"]
    assert cc["n_checked"] == len(valid)      # the skipped rows are not checked
    assert cc["n_mismatch"] == 0              # the index join agrees

    # The test is NOT vacuous: the same rows joined POSITIONALLY do mismatch.
    pool = score_smiles_pool([r["smiles"] for r in rows])
    per_row = list(pool["rows"])
    indexes = list(pool["row_indexes"])
    assert indexes != list(range(len(per_row)))   # the join is non-trivial
    naive_mismatches = 0
    for chem, r in zip(per_row, rows):            # POSITIONAL -- the bug
        csv_frag = r["n_frags_raw"]
        if csv_frag is None:
            continue
        if abs(float(csv_frag) - float(chem.get("n_fragments", 0) or 0)) > 1e-9:
            naive_mismatches += 1
    assert naive_mismatches > 0, "the fixture no longer exercises the bug"


# --- the gate records the kind ---------------------------------------------


def test_the_gate_records_the_pool_kind_of_every_scored_pool(tmp_path):
    gen = tmp_path / "gen"
    _write_pool(
        gen / "generated_smiles.csv",
        ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"],
        [{"epoch": 0, "smiles": "CCO", "n_frags_raw": 1, "largest_frag_frac": 1.0}],
    )
    src = score_run_dir(gen)["sources"]["generated_smiles_csv"]
    assert src["pool_kind"] == KIND_GENERATION
    assert src["pool_kind_marker"] == GENERATION_MARKER
    assert src["pool_kind_reason"]

    recon = tmp_path / "recon"
    _write_pool(
        recon / "generated_smiles.csv",
        ["epoch", "smiles", "sanitized", "metal_z"],
        [{"epoch": 0, "smiles": "CCO", "sanitized": "1", "metal_z": "77"}],
    )
    src = score_run_dir(recon)["sources"]["generated_smiles_csv"]
    assert src["pool_kind"] == KIND_RECONSTRUCTION
    assert src["pool_kind_marker"] == CONTENT_MARKER
    assert src["pool_kind_reason"]


def test_the_strict_pool_kind_switch_is_an_env_var_and_not_a_cli_flag():
    # The one migration that could move a number (the C7 cross-kind ``by_tier``)
    # stays OPT-IN, and the switch is an environment variable -- repo
    # precedent WFA2_CONFIG_STRICT=1 / WF343C4_GATE=1 -- NOT a new argparse
    # flag, so the default launch path is byte-identical.
    src = Path(c7.__file__).read_text(encoding="utf-8")
    assert "CFM_POOL_KIND_STRICT" in src
    assert "--strict-pool-kind" not in src
    assert "--strict_pool_kind" not in src
