"""Wf-A158 BRANCH POP -- tests for ``wfa158_population_reconciliation.py``.

WHAT THIS FILE GUARDS
---------------------
1. THE CENSUS AGGREGATION MATHS ON A SYNTHETIC FIXTURE -- the median, the
   rendering-collision grouping, the pool-CSV recount and the whole census
   roll-up, computed on fixtures built inside ``tmp_path`` so the assertions
   do not depend on the repo's live numbers drifting.
2. THE FAMILY-CLASSIFICATION RULE -- the same JSON key means DIFFERENT things
   in different producers (``connected_rate`` is the CONDITIONAL rate in the
   pool-quality file and the POST-PROCESSED rate in the post-processed arm),
   so the rule must be FILE-AWARE.
3. THE JSON SCHEMA of the emitted artifact.
4. THE NO-CUDA IMPORT CHECK -- importing the script must not pull in torch.

CPU-ONLY: no torch, no CUDA, no RDKit, no checkpoint.  Nothing here writes to
``metrics/``, ``registry/`` or the run ledger; every fixture is built inside
pytest's ``tmp_path``.

Run:
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        -m pytest molmetal/tests/test_wfa158_population_reconciliation.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts import wfa158_population_reconciliation as mod  # noqa: E402


# ---------------------------------------------------------------------------
# 0. the no-CUDA import check
# ---------------------------------------------------------------------------
# WHY A SUBPROCESS: this repository's own ``molmetal/tests/conftest.py``
# imports torch at module import time (it forces the scatter backend), so an
# in-process ``"torch" not in sys.modules`` assertion would be FALSE for a
# reason that has nothing to do with the script under test.  A clean
# interpreter is the only honest measurement.
_NO_CUDA_PROBE = r"""
import json, sys
sys.path.insert(0, {repo!r})
import molmetal.scripts.wfa158_population_reconciliation as mod
print(json.dumps({{
    "torch": "torch" in sys.modules,
    "cupy": "cupy" in sys.modules,
    "rdkit": "rdkit" in sys.modules,
    "n_families": len(mod.FAMILIES),
}}))
"""


def test_importing_the_script_pulls_in_no_torch_and_no_rdkit():
    """MEASURED in a clean interpreter (see the docstring's WHY)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", _NO_CUDA_PROBE.format(repo=str(REPO))],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["torch"] is False, "importing the script pulled in torch"
    assert payload["cupy"] is False, "importing the script pulled in cupy"
    assert payload["rdkit"] is False, "importing the script pulled in rdkit"
    assert payload["n_families"] == len(mod.FAMILIES)


def test_script_source_never_mentions_torch_or_cuda():
    """A STATIC check that does not depend on the interpreter."""
    source = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("import torch", "torch.cuda", "nvidia-smi", "cuda("):
        assert banned not in source, "the script mentions %r" % (banned,)


# ---------------------------------------------------------------------------
# 1. the family-classification rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "stem,key,expected",
    [
        # the headline: the audit's own end-to-end rate
        ("cfm_fragmentation_audit", "single_molecule_rate_end_to_end",
         ("F1_pool_e2e", "fraction")),
        # the conditional rate over PARSEABLE rows
        ("cfm_fragmentation_audit", "single_fragment_ratio",
         ("F2_pool_cond", "fraction")),
        # the rate over the pool's OWN rows
        ("cfm_fragmentation_audit", "single_molecule_rate_over_pool_rows",
         ("F3_pool_rows", "fraction")),
        # the SAME key name, DIFFERENT meaning -- the pool-quality file's
        # connected_rate is the CONDITIONAL rate ...
        ("c7_fullgen_pool_quality", "connected_rate",
         ("F2_pool_cond", "fraction")),
        # ... while the post-processed arm's connected_rate_over_parsed is
        # the POST-PROCESSED rate
        ("cfm_fragmentation_postproc_arm", "connected_rate_over_parsed",
         ("F10_postproc_connected", "fraction")),
        ("cfm_fragmentation_postproc_arm", "connected_rate_over_postprocessed",
         ("F10_postproc_connected", "fraction")),
        # the raw axis the post-processed arm is compared against stays F2
        ("cfm_fragmentation_postproc_arm", "single_fragment_rate_over_parsed",
         ("F2_pool_cond", "fraction")),
        # the metal-stripped counterfactual
        ("wfa155_ligand_connected_rate", "single_fragment_rate",
         ("F11_ligand_only_connected", "fraction")),
        # the trainer-side (per-run) families
        ("cfm_run_census", "gen_single_molecule_rate_end",
         ("F5_census_last_epoch", "fraction")),
        ("cfm_run_census", "gen_single_molecule_rate_best",
         ("F6_census_best_epoch", "fraction")),
        ("cfm_run_census", "gen_validity_end", ("F7_validity", "fraction")),
        ("cfm_run_census", "gen_molecule_validity_end",
         ("F8_molecule_validity", "fraction")),
        ("cfm_free_generation_panel_seed0", "single_fragment_ratio",
         ("F2_pool_cond", "fraction")),
        # decode_ratio's value key -- ONLY inside decode_ratio.json
        ("decode_ratio", "best_measured_real", ("F9_decode_ratio", "fraction")),
    ],
)
def test_classify_key_is_file_aware(stem, key, expected):
    assert mod.classify_key(stem, key) == expected


@pytest.mark.parametrize(
    "stem,key",
    [
        # the decode_ratio value key name must NOT be claimed outside
        # decode_ratio.json -- 12 unrelated metrics carry it
        ("some_unrelated_metric", "best_measured_real"),
        # a count that names a fragmentation quantity but is not a rate
        ("anything", "n_frags"),
        ("anything", "n_rows"),
        ("anything", "n_attempted"),
        # an unrelated key
        ("anything", "totally_unrelated_key"),
    ],
)
def test_classify_key_declines_non_rates(stem, key):
    assert mod.classify_key(stem, key) is None


def test_every_rule_targets_a_declared_family():
    """A rule that names an UNDECLARED family would silently drop rows."""
    declared = set(mod.FAMILIES)
    for pattern, family_id, unit in mod.GENERIC_KEY_RULES:
        assert family_id in declared, pattern
        assert unit in {"fraction", "count"}, pattern
    for stem, rules in mod.FILE_OVERRIDES.items():
        for pattern, family_id, unit in rules:
            assert family_id in declared, (stem, pattern)
            assert unit in {"fraction", "count"}, (stem, pattern)


def test_every_declared_family_states_its_four_axes():
    """population / numerator / denominator / measurement_point -- the four
    axes Q1 asks for -- must be non-empty for every family."""
    for family_id, spec in mod.FAMILIES.items():
        for axis in ("population", "numerator", "denominator", "measurement_point"):
            assert spec.get(axis), (family_id, axis)
        assert spec.get("label"), family_id
        assert spec.get("unit") in {"fraction", "count", "band"}, family_id


@pytest.mark.parametrize(
    "key,expected",
    [
        ("connected_rate", "rate"),
        ("single_molecule_rate_end_to_end", "rate"),
        ("n_single", "component_count"),
        ("n_connected", "component_count"),
        ("n_unparseable", "component_count"),
        ("largest_frag_frac_mean", "companion_statistic"),
        ("share_n_frags_2", "companion_statistic"),
    ],
)
def test_classify_role(key, expected):
    assert mod.classify_role(key) == expected


# ---------------------------------------------------------------------------
# 2. the census aggregation maths, on a synthetic fixture
# ---------------------------------------------------------------------------
def test_render_pct_is_two_decimals():
    assert mod.render_pct(0.08506944444444445) == "8.51"
    assert mod.render_pct(0.25047348484848486) == "25.05"
    assert mod.render_pct(1.0) == "100.00"
    assert mod.render_pct(0.0) == "0.00"


def test_find_rendering_collisions_groups_distinct_values_that_print_the_same():
    """THE MECHANISM behind the ambiguity: five DIFFERENT published numbers
    render as "8.51 %".  A collision is a bucket with >1 DISTINCT value."""
    rows = [
        {"file": "a.json", "jsonpath": "/x", "value": 0.08506944444444445,
         "family_id": "F1_pool_e2e"},
        # the same VALUE again, a different jsonpath: still ONE distinct value
        {"file": "b.json", "jsonpath": "/y", "value": 0.08506944444444445,
         "family_id": "F1_pool_e2e"},
        {"file": "c.json", "jsonpath": "/z", "value": 0.08508958875996527,
         "family_id": "F3_pool_rows"},
        {"file": "d.json", "jsonpath": "/w", "value": 0.08509630565203663,
         "family_id": "F2_pool_cond"},
        # a value with NO twin -- not a collision
        {"file": "e.json", "jsonpath": "/v", "value": 0.25047348484848486,
         "family_id": "F1_pool_e2e"},
        # a non-numeric value is ignored entirely
        {"file": "f.json", "jsonpath": "/u", "value": None,
         "family_id": "F1_pool_e2e"},
    ]
    collisions = mod.find_rendering_collisions(rows)
    assert len(collisions) == 1
    collision = collisions[0]
    assert collision["rendered_percent"] == "8.51"
    assert collision["n_distinct_values"] == 3
    assert collision["values"] == sorted(
        {0.08506944444444445, 0.08508958875996527, 0.08509630565203663}
    )
    # the duplicate row is still a member, but it does not add a distinct value
    assert len(collision["members"]) == 4
    assert {m["family_id"] for m in collision["members"]} == {
        "F1_pool_e2e", "F2_pool_cond", "F3_pool_rows",
    }


def test_median_matches_the_producer_rule():
    """cfm_fragmentation_audit.py:333-348 -- sorted, odd n -> vals[n // 2],
    even n -> the mean of the two central values."""
    assert mod._median([3.0, 1.0, 2.0]) == 2.0
    assert mod._median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert mod._median([5.0]) == 5.0
    assert mod._median([]) is None


def test_token_fraction_reads_a_leading_dot_as_a_fraction():
    """The bug that made an earlier draft report '0.0851' as 0.09 % and miss
    every fragmentation hit."""
    assert mod._token_fraction("0.0851", 0.0851) == 0.0851
    assert mod._token_fraction("8.51", 8.51) == pytest.approx(0.0851)
    assert mod._token_fraction("25.05", 25.05) == pytest.approx(0.2505)
    assert mod._token_fraction("92.7", 92.7) == pytest.approx(0.927)


def _write_pool_csv(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,smiles,n_frags_raw,largest_frag_frac"]
    for epoch, smiles, n_frags, frac in rows:
        lines.append("%s,%s,%s,%s" % (epoch, smiles, n_frags, frac))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_pool_csv_recount_on_a_synthetic_fixture(tmp_path, monkeypatch):
    """The F4 family: count the writer's own n_frags_raw column, no RDKit."""
    monkeypatch.setattr(mod, "REPO", tmp_path)  # _read_pool_csv paths are REPO-relative
    pool = _write_pool_csv(
        tmp_path / "molmetal" / "reports" / "synth_pool" / "generated_smiles.csv",
        [
            ("1", "C.C.C", 1, "1.0"),
            ("1", "C.C", 2, "0.6"),
            ("1", "C", 1, "1.0"),
            ("2", "C.C.C", 1, "1.0"),
            ("2", "C.C.C.C", 1, "1.0"),
            ("2", "C.C.C.C.C", 2, "0.4"),
        ],
    )
    counted = mod._read_pool_csv(pool)
    assert counted["exists"] is True
    assert counted["n_rows"] == 6
    assert counted["n_single"] == 4
    assert counted["rate_over_rows"] == pytest.approx(4 / 6)
    assert counted["header"][:4] == ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
    assert counted["per_epoch"]["1"] == {
        "n_rows": 3, "n_single": 2, "rate_over_rows": pytest.approx(2 / 3),
    }
    assert counted["per_epoch"]["2"]["n_rows"] == 3


def _synthetic_repo(tmp_path: Path) -> Dict[str, Path]:
    """A miniature repo: three metric files, one pool CSV, one registry."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    audit = metrics_dir / "cfm_fragmentation_audit.json"
    audit.write_text(
        json.dumps(
            {
                "summary": {
                    "generation_side": {
                        "single_molecule_rate_end_to_end": {
                            "n": 2, "median": 0.08506944444444445,
                        },
                        "single_fragment_ratio": {
                            "n": 2, "median": 0.0649203119169274,
                        },
                    }
                },
                "runs": [
                    {
                        "run": "pool_a",
                        "single_molecule_rate_end_to_end": 0.08506944444444445,
                        "single_molecule_rate_over_pool_rows": 0.08508958875996527,
                        "single_fragment_ratio": 0.0649203119169274,
                    },
                    {
                        "run": "pool_b",
                        "single_molecule_rate_end_to_end": 0.25047348484848486,
                        "single_molecule_rate_over_pool_rows": 0.25047348484848486,
                        "single_fragment_ratio": 0.25,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    census = metrics_dir / "cfm_run_census.json"
    census.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run": "r1",
                        "gen_single_molecule_rate_end": 0.0625,
                        "gen_single_molecule_rate_best": 0.35546875,
                        "gen_validity_end": 0.9,
                    },
                    {
                        "run": "r2",
                        "gen_single_molecule_rate_end": 0.07,
                        "gen_single_molecule_rate_best": 0.30,
                        "gen_validity_end": 0.8,
                    },
                ],
                "fragmentation_summary": {
                    "end_to_end_single_molecule_rate": {"median": 0.06625, "n": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    quality = metrics_dir / "synth_pool_quality.json"
    quality.write_text(
        json.dumps(
            {
                "overall": {
                    "connected_rate": 0.08509630565203663,
                    "validity": 0.97,
                    "n_parse": 100,
                }
            }
        ),
        encoding="utf-8",
    )

    pool = _write_pool_csv(
        tmp_path / "molmetal" / "reports" / "synth_pool" / "generated_smiles.csv",
        [
            ("1", "C.C.C", 1, "1.0"),
            ("1", "C.C", 2, "0.6"),
            ("1", "C", 1, "1.0"),
            ("2", "C.C.C", 1, "1.0"),
            ("2", "C.C.C.C", 1, "1.0"),
            ("2", "C.C.C.C.C", 2, "0.4"),
        ],
    )

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "metrics:\n"
        "  synth_pool_quality:\n"
        "    source_json: metrics/by_metric/synth_pool_quality.json\n"
        "    value: 0.08509630565203663\n"
        "    unit_canonical: fraction\n"
        "    status: current\n"
        "    quotable: false\n"
        "    registry_role: artifact\n",
        encoding="utf-8",
    )

    return {
        "REPO": tmp_path,
        "METRICS_DIR": metrics_dir,
        "AUDIT_JSON": audit,
        "CENSUS_JSON": census,
        "POSTPROC_JSON": metrics_dir / "postproc.json",
        "LIGAND_JSON": metrics_dir / "ligand.json",
        "C7_QUALITY_JSON": quality,
        "PANEL_JSON": metrics_dir / "panel.json",
        "REGISTRY_YAML": registry,
        "C8_POOL": pool,
    }


def test_census_q2_on_a_synthetic_fixture(tmp_path, monkeypatch):
    """The whole roll-up, computed on fixtures -- no repo dependency."""
    paths = _synthetic_repo(tmp_path)
    for name, value in paths.items():
        monkeypatch.setattr(mod, name, value)

    q2 = mod.census_q2()

    # 6 rows from the audit + 6 from the census + 2 from the pool-quality
    # file + 1 from the pool CSV.
    assert q2["status"] == "OK"
    assert q2["n_census_rows"] == 15
    assert q2["n_metrics_scanned"] == 3
    assert q2["rows_per_metric"] == {
        "cfm_fragmentation_audit": 6,
        "cfm_run_census": 6,
        "synth_pool_quality": 2,
    }

    table = q2["family_table"]
    assert table["F1_pool_e2e"]["n_rate_rows"] == 2       # the two e2e runs
    assert table["F3_pool_rows"]["n_rate_rows"] == 2      # over-pool-rows
    assert table["F2_pool_cond"]["n_rate_rows"] == 3      # 2 audit + 1 quality
    assert table["F5_census_last_epoch"]["n_rate_rows"] == 2
    assert table["F6_census_best_epoch"]["n_rate_rows"] == 2
    assert table["F7_validity"]["n_rate_rows"] == 3       # 2 census + 1 quality
    assert table["F4_csv_count"]["n_rate_rows"] == 1       # the pool CSV
    # the external families are DECLARED but carry no rows of their own
    for external in mod.EXTERNAL_FAMILIES:
        assert table[external]["n_census_rows"] == 0
        assert external in table

    # the collision detector fires on the synthetic "8.51"
    collisions = {c["rendered_percent"]: c for c in q2["rendering_collisions"]}
    assert "8.51" in collisions
    assert collisions["8.51"]["n_distinct_values"] == 3
    assert set(collisions["8.51"]["values"]) == {
        0.08506944444444445, 0.08508958875996527, 0.08509630565203663,
    }
    # a single-distinct-value bucket is not a collision
    assert "25.05" not in collisions

    # every row names a declared family and a unit
    for row in q2["rows"]:
        assert row["family_id"] in mod.FAMILIES, row
        assert row["unit"] in {"fraction", "count"}, row
        assert row["role"] in {"rate", "component_count", "companion_statistic"}
        assert row["file"], row

    # the registry link is READ-ONLY and attached to the rows it covers
    reg_rows = q2["registry_rows"]
    if reg_rows and "error" not in reg_rows[0]:
        assert reg_rows[0]["registry_key"] == "synth_pool_quality"
        linked = [r for r in q2["rows"] if r["metric"] == "synth_pool_quality"]
        assert linked and all(r["registry_row"] == "synth_pool_quality" for r in linked)


def test_census_q2_does_not_write_to_metrics_or_registry(tmp_path, monkeypatch):
    """The census is READ-ONLY over ``metrics/`` and ``registry/``."""
    paths = _synthetic_repo(tmp_path)
    for name, value in paths.items():
        monkeypatch.setattr(mod, name, value)
    before = {
        p: p.stat().st_mtime_ns for p in paths.values() if p.is_file()
    }
    mod.census_q2()
    after = {p: p.stat().st_mtime_ns for p in paths.values() if p.is_file()}
    assert before == after


# ---------------------------------------------------------------------------
# 3. the gates must actually bite
# ---------------------------------------------------------------------------
def _fake_q1(complete: bool = True, reason: str = "SAME DEFINITION, DIFFERENT POPULATION"):
    definition = {
        "population": "9 generation pools",
        "numerator": "n_single",
        "denominator": "samples drawn",
        "measurement_point": "raw SMILES after RDKit sanitisation",
        "published_value": 0.08506944444444445,
        "recomputed_median_from_runs": 0.08506944444444445,
        "recomputed_matches_published": True,
        "matches_audit_published_value": True,
        "csv_recount": {"rate_over_rows": 0.25047348484848486},
    }
    if not complete:
        definition.pop("denominator")
    return {
        "status": "OK",
        "definitions": {
            "8.51_percent": dict(definition),
            "25.05_percent": dict(definition),
        },
        "reason_the_two_differ": reason,
    }


def _fake_q2():
    return {
        "n_families_declared": 1,
        "n_families_with_rows": 1,
        "family_table": {"F1_pool_e2e": {"n_rate_rows": 1}},
        "n_hits": 0,
    }


def _fake_q3(verdict: str = "MATCHED"):
    return {"hits": [{"verdict": verdict, "file": "x.md"}], "n_hits": 1,
            "n_unmatched": 0, "n_ambiguous": 0}


def test_run_gates_emits_the_expected_gate_set():
    gates = mod.run_gates(_fake_q1(), _fake_q2(), _fake_q3())
    names = [g["gate"] for g in gates]
    assert names == [
        "q1_median_recomputed",
        "q1_c8_csv_recount_matches_published",
        "q1_all_four_axes_established",
        "q1_reason_established",
        "q2_every_family_has_a_rate_row",
        "q3_every_hit_classified",
        "q3_no_paper_read",
        "no_cuda_no_torch",
    ]
    for gate in gates:
        assert gate["status"] in {"PASS", "FAIL", "SKIP"}
        assert gate["detail"]
    by_name = {g["gate"]: g for g in gates}
    for name in (
        "q1_median_recomputed",
        "q1_c8_csv_recount_matches_published",
        "q1_all_four_axes_established",
        "q1_reason_established",
        "q2_every_family_has_a_rate_row",
        "q3_every_hit_classified",
        "q3_no_paper_read",
    ):
        assert by_name[name]["status"] == "PASS", (name, by_name[name]["detail"])


def test_a_missing_axis_makes_the_axis_gate_fail():
    """A gate that cannot FAIL is not a gate."""
    gates = {g["gate"]: g for g in mod.run_gates(_fake_q1(complete=False), _fake_q2(), _fake_q3())}
    assert gates["q1_all_four_axes_established"]["status"] == "FAIL"


def test_an_unclassified_hit_makes_the_q3_gate_fail():
    gates = {g["gate"]: g for g in mod.run_gates(_fake_q1(), _fake_q2(), _fake_q3(verdict=""))}
    assert gates["q3_every_hit_classified"]["status"] == "FAIL"


def test_a_paper_hit_makes_the_paper_gate_fail():
    q3 = _fake_q3()
    q3["hits"] = [{"verdict": "MATCHED", "file": "paper/main.tex"}]
    gates = {g["gate"]: g for g in mod.run_gates(_fake_q1(), _fake_q2(), q3)}
    assert gates["q3_no_paper_read"]["status"] == "FAIL"


def test_an_unestablished_reason_makes_the_reason_gate_fail():
    gates = {
        g["gate"]: g
        for g in mod.run_gates(
            _fake_q1(reason="NOT ESTABLISHED -- the link is missing"),
            _fake_q2(),
            _fake_q3(),
        )
    }
    assert gates["q1_reason_established"]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# 4. the JSON schema of the emitted artifact
# ---------------------------------------------------------------------------
ARTIFACT = REPO / "molmetal" / "reports" / "wfa158_population_reconciliation" / "reconciliation.json"
MARKDOWN = REPO / "molmetal" / "reports" / "wfa158_population_reconciliation" / "census.md"

REQUIRED_TOP_LEVEL = {
    "metric", "tag", "date", "wave", "gpu", "purpose",
    "q1", "q2", "q3", "gates", "verdict", "write_scope", "read_scope",
    "not_established",
}
REQUIRED_Q1 = {
    "status", "headline", "reason_the_two_differ", "definitions", "csv_recounts",
    "provenance",
}
REQUIRED_Q2 = {
    "status", "n_census_rows", "n_families_declared", "n_families_with_rows",
    "family_table", "family_headlines", "rendering_collisions", "rows",
}
REQUIRED_Q3 = {
    "status", "n_files_scanned", "n_hits", "n_files_with_hits", "hits",
    "patch_list", "grep_note",
}


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="the artifact has not been generated yet")
def test_emitted_artifact_schema():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert REQUIRED_TOP_LEVEL <= set(payload)
    assert payload["tag"] == "MEASURED"
    assert payload["verdict"] in {"PASS", "FAIL", "SKIP"}
    assert REQUIRED_Q1 <= set(payload["q1"])
    assert REQUIRED_Q2 <= set(payload["q2"])
    assert REQUIRED_Q3 <= set(payload["q3"])

    for key in ("8.51_percent", "25.05_percent"):
        definition = payload["q1"]["definitions"][key]
        for axis in ("population", "numerator", "denominator", "measurement_point"):
            assert definition.get(axis), (key, axis)
        assert definition["published_jsonpath"].startswith("/")
        assert definition["published_metric_file"].startswith("metrics/")

    assert isinstance(payload["gates"], list)
    assert payload["gates"]
    for gate in payload["gates"]:
        assert set(gate) == {"gate", "status", "detail"}
        assert gate["status"] in {"PASS", "FAIL", "SKIP"}

    for hit in payload["q3"]["hits"]:
        assert set(hit) >= {
            "file", "line", "quoted_token", "sentence", "verdict", "family_ids",
        }
        assert hit["verdict"].strip()
        assert not str(hit["file"]).startswith("paper/")

    assert set(payload["q2"]["family_table"]) == set(mod.FAMILIES)


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="the artifact has not been generated yet")
def test_artifact_numbers_agree_with_the_files_on_disk():
    """FILESYSTEM TRUTH: the published 8.51 % must still be the median of the
    9 per-pool rates in the audit, and the published 25.05 % must still be
    the CSV recount of the c8 pool."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    audit = json.loads(
        (REPO / "metrics" / "by_metric" / "cfm_fragmentation_audit.json").read_text(encoding="utf-8")
    )
    values = [
        run["single_molecule_rate_end_to_end"]
        for run in audit["runs"]
        if isinstance(run.get("single_molecule_rate_end_to_end"), (int, float))
    ]
    published = audit["summary"]["generation_side"]["single_molecule_rate_end_to_end"]
    assert len(values) == published["n"]
    assert mod._median(values) == pytest.approx(published["median"])
    assert payload["q1"]["headline"]["8.51_percent"] == pytest.approx(published["median"])

    c8 = payload["q1"]["csv_recounts"]["c8_pool"]
    assert c8["n_single"] == 529
    assert c8["n_rows"] == 2112
    assert c8["rate_over_rows"] == pytest.approx(529 / 2112)
    assert payload["q1"]["headline"]["25.05_percent"] == pytest.approx(529 / 2112)


@pytest.mark.skipif(not MARKDOWN.is_file(), reason="the census.md has not been generated yet")
def test_markdown_artifact_carries_the_three_tables():
    text = MARKDOWN.read_text(encoding="utf-8")
    assert "Q1" in text and "Q2" in text and "Q3" in text
    assert "8.51" in text and "25.05" in text
    # the two headline families must both be named
    assert "F1_pool_e2e" in text


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="the artifact has not been generated yet")
def test_rendering_neighbourhood_labels_every_number_with_its_own_printing():
    """REGRESSION: an earlier draft filed the DIRECT column count (0.08516852,
    which PRINTS as '8.52') under a key that claimed 'renders as 8.51'.  Every
    entry must now carry its OWN rendered_percent."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    block = payload["q1"]["c7_pool_rendering_neighbourhood"]
    entries = {
        key: value
        for key, value in block.items()
        if isinstance(value, dict) and "value" in value
    }
    assert len(entries) >= 4
    for entry in entries.values():
        assert "rendered_percent" in entry, entry
        assert entry["rendered_percent"] == mod.render_pct(entry["value"]), entry
    # the two counts of the SAME pool straddle the 2-dp boundary
    assert entries["c7_pool_over_its_own_rows__CSV_COLUMN_COUNT"]["rendered_percent"] == "8.52"
    assert entries["c7_pool_over_its_own_rows__AUDIT_RECOUNT"]["rendered_percent"] == "8.51"


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="the artifact has not been generated yet")
def test_cross_checks_record_the_c7_and_c8_reconciliation():
    """The CSV recount and the audit's recount must be compared, not assumed."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    checks = payload["q1"]["cross_checks"]
    c8 = checks["c8_pool_csv_recount_vs_audit"]
    assert c8["agrees"] is True
    assert c8["csv"]["n_single"] == 529 and c8["csv"]["n_rows"] == 2112
    c7 = checks["c7_pool_csv_recount_vs_audit"]
    assert c7["agrees"] is False
    assert c7["difference_in_rows"] == 1
    assert c7["audit"]["n_unparseable"] == 1
    assert c7["audit"]["csv_column_cross_check"]["n_mismatch"] == 0
    # the difference must be EXPLAINED, not merely recorded
    assert "UNPARSEABLE" in c7["note"]
