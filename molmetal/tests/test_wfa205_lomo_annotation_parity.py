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
"""Wf-A205 -- the m19 headline caveat must be COMPUTED, not transcribed.

The defect this covers: the producer of
``metrics/by_metric/metal_lomo_auc_wasserstein.json`` emits only the
measurement keys, while the caveat that annotates the headline was added
by hand.  A fresh execution of the producer therefore drops the caveat
silently.  Worse, a caveat that hard-codes a measured number becomes a
live false claim the moment the producer runs on other data -- a false
caveat is worse than a missing one.

So the caveat must be a pure function of the payload, and these tests
pin that down from four sides:

  * a SYNTHETIC payload with deliberately different metals, counts and
    fold values must yield the synthetic figures and none of the
    published ones;
  * the builder must be deterministic;
  * the builder, fed the PUBLISHED payload plus the resolvability inputs
    the published caveat was derived from, must reproduce every figure the
    published caveat states -- and must NOT assert the two figures that
    belong to another artefact or to a withdrawn claim;
  * the PUBLISHED payload on its own carries no resolvability inputs, so
    the builder must degrade instead of inventing them.

The producer is checked by AST only.  The trainer is never executed: it
needs a cohort and 30 epochs.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wfa205_lomo_annotation_builders import (  # noqa: E402
    audit_style_mean,
    build_lomo_headline_caveat,
    describe_lomo_headline,
    resolve_min_unique_for_gate,
)

BUILDER_PATH = REPO_ROOT / "molmetal" / "scripts" / "wfa205_lomo_annotation_builders.py"
PRODUCER_PATH = REPO_ROOT / "molmetal" / "scripts" / "train_metal_lomo_wasserstein.py"
PUBLISHED_METRIC = REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
RESOLVABILITY_AUDIT = REPO_ROOT / "metrics" / "by_metric" / "lomo_resolvability_audit.json"
LOMO_CSV = REPO_ROOT / "molmetal" / "data" / "metal_lomo.csv"

#: Figures and metal names that must never be transcribed into the builder.
#: The check is a case-sensitive SUBSTRING check (not a word boundary),
#: which is deliberately stricter than it has to be: it makes it impossible
#: to add a transcribed figure to the builder without tripping this test.
FORBIDDEN_LITERALS = (
    "0.5575",
    "0.5974",
    "0.3812",
    "0.5947",
    "1311",
    "538",
    "Rh",
    "Os",
    "Re",
    "Ru",
    "Ir",
    "200",
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _published_payload() -> dict:
    return json.loads(PUBLISHED_METRIC.read_text(encoding="utf-8"))


def _audit_folds() -> dict:
    audit = json.loads(RESOLVABILITY_AUDIT.read_text(encoding="utf-8"))
    return {fold["metal"]: int(fold["n_unique"]) for fold in audit["folds"]}


def _audit_threshold() -> int:
    audit = json.loads(RESOLVABILITY_AUDIT.read_text(encoding="utf-8"))
    return int(audit["design"]["min_unique_for_gate"])


def _parity_payload() -> tuple:
    """The published payload plus the resolvability inputs its caveat used.

    The published artefact carries NO per-metal unique-ligand counts, so
    the published caveat's counts came from the project's resolvability
    audit.  That block is what the producer now supplies itself.
    """
    published = _published_payload()
    payload = dict(published)
    payload["per_metal_resolvability"] = {
        "rule": "MIN_UNIQUE_FOR_GATE",
        "rule_source": "molmetal/scripts/metal_activity_transfer_probe.py",
        "min_unique_for_gate": _audit_threshold(),
        "n_unique_ligands": _audit_folds(),
    }
    return payload, published


SYNTHETIC_PAYLOAD = {
    "value": 0.6266666666666667,
    "per_fold_auc_mean": {"Au": 0.61, "Pt": 0.72, "Ag": 0.55},
    "gate_target": 0.6,
    "per_metal_resolvability": {
        "rule": "MIN_UNIQUE_FOR_GATE",
        "min_unique_for_gate": 200,
        "n_unique_ligands": {"Au": 900, "Pt": 30, "Ag": 120},
    },
}


# ---------------------------------------------------------------------------
# 1. the builder computes, it does not transcribe
# ---------------------------------------------------------------------------
def test_synthetic_payload_yields_the_synthetic_figures_only():
    caveat = build_lomo_headline_caveat(SYNTHETIC_PAYLOAD)

    # the synthetic figures are present
    assert "0.6267" in caveat          # equal-weight mean over the 3 folds
    assert "0.6100" in caveat          # the resolvable fold
    assert "0.5500" in caveat          # the weakest fold
    assert "900" in caveat              # the resolvable fold's count
    assert "30" in caveat               # a fold below the threshold
    assert "120" in caveat              # a fold below the threshold
    assert "Au" in caveat and "Pt" in caveat and "Ag" in caveat

    # the published figures are absent
    for forbidden in ("0.5575", "0.5974", "0.3812", "1311", "538", "Rh", "Os"):
        assert forbidden not in caveat, forbidden

    # and the published metals never appear as fold names
    assert "The headline 0.6267 is the EQUAL-WEIGHT mean over 3 metal folds" in caveat


def test_builder_is_deterministic():
    first = build_lomo_headline_caveat(SYNTHETIC_PAYLOAD)
    second = build_lomo_headline_caveat(dict(SYNTHETIC_PAYLOAD))
    assert first == second
    assert build_lomo_headline_caveat(_parity_payload()[0]) == build_lomo_headline_caveat(
        _parity_payload()[0]
    )


def test_builder_rejects_an_unreadable_payload():
    with pytest.raises(ValueError):
        build_lomo_headline_caveat({})
    with pytest.raises(ValueError):
        build_lomo_headline_caveat({"per_fold_auc_mean": {}})
    with pytest.raises(ValueError):
        build_lomo_headline_caveat({"per_fold_auc_mean": {"Au": float("nan")}})


def test_builder_source_carries_no_transcribed_figure():
    source = BUILDER_PATH.read_text(encoding="utf-8")
    for literal in FORBIDDEN_LITERALS:
        assert literal not in source, "transcribed literal in the builder: %r" % literal


# ---------------------------------------------------------------------------
# 2. parity with the published, hand-written caveat
# ---------------------------------------------------------------------------
def test_published_payload_reproduces_every_published_figure():
    payload, published = _parity_payload()
    caveat = build_lomo_headline_caveat(payload)
    published_caveat = published["headline_caveat"]

    folds = published["per_fold_auc_mean"]
    counts = _audit_folds()
    threshold = _audit_threshold()
    resolvable = [m for m in folds if counts.get(m, 0) >= threshold]
    unresolvable = [m for m in folds if counts.get(m, 0) < threshold]

    # every figure the published caveat states, computed from the payload
    assert "{0:.4f}".format(published["value"]) in caveat
    for metal, value in folds.items():
        assert "{0} {1:.4f}".format(metal, value) in caveat
    assert "MIN_UNIQUE_FOR_GATE={0}".format(threshold) in caveat
    for metal in unresolvable:
        assert "{0} {1:,}".format(metal, counts[metal]) in caveat
    for metal in resolvable:
        assert "{0} {1:,}".format(metal, counts[metal]) in caveat
    subset = audit_style_mean([folds[m] for m in resolvable])
    assert "{0:.4f}".format(subset) in caveat
    weakest = min(folds.items(), key=lambda kv: (kv[1], kv[0]))
    assert "{0} {1:.4f}".format(weakest[0], weakest[1]) in caveat
    assert "gate_target={0:g}".format(published["gate_target"]) in caveat

    # the published caveat states these two; the computed one must not,
    # because neither is derivable from a measurement payload:
    #   * the gap of the WITHDRAWN claim  (a historical claim, not a run)
    #   * the seed-ensemble robustness figure (a different artefact)
    assert "0.5947" not in caveat
    assert "0.04" not in caveat

    # the withdrawal is pointed at, never asserted
    assert "docs/PREDICTIVE_LINE_INDEX.md" in caveat
    assert "RETRACTED" not in caveat
    assert "metrics/by_metric/lomo_seed_ensemble.json" in caveat


def test_published_caveat_and_computed_caveat_agree_on_the_conclusion():
    payload, published = _parity_payload()
    caveat = build_lomo_headline_caveat(payload)
    published_caveat = published["headline_caveat"]

    # same conclusion: the headline is not the quotable figure, the
    # resolvable-subset mean is.
    assert "must not be quoted as the gated headline" in caveat
    assert "must not be quoted as the LOMO headline" in published_caveat
    # same direction against the gate
    assert "sits below this metric's own gate constant" in caveat
    assert "sits below the metric's own gate constant" in published_caveat
    # same weakest fold, named the same way
    folds = published["per_fold_auc_mean"]
    weakest = min(folds.items(), key=lambda kv: (kv[1], kv[0]))[0]
    assert "The weakest fold in the equal-weight mean is {0}".format(weakest) in caveat
    assert "real_only_fold_value is {0:.4f} (the {1} fold)".format(
        published["real_only_fold_value"], weakest
    ) in published_caveat


def test_published_payload_without_resolvability_degrades_instead_of_inventing():
    published = _published_payload()
    caveat = build_lomo_headline_caveat(published)
    assert "RESOLVABILITY NOT COMPUTED" in caveat
    # no count is fabricated: no count appears as a standalone number
    for count in _audit_folds().values():
        assert re.search(r"(?<![\d.]){0}(?![\d.])".format(count), caveat) is None
    # and no resolvable-subset figure is stated
    counts = _audit_folds()
    folds = published["per_fold_auc_mean"]
    threshold = _audit_threshold()
    resolvable = [m for m in folds if counts.get(m, 0) >= threshold]
    subset = audit_style_mean([folds[m] for m in resolvable])
    assert "{0:.4f}".format(subset) not in caveat
    # the headline and the weakest fold still come from the payload
    assert "{0:.4f}".format(published["value"]) in caveat
    weakest = min(published["per_fold_auc_mean"].values())
    assert "{0:.4f}".format(weakest) in caveat


# ---------------------------------------------------------------------------
# 3. the producer is wired (AST only -- the trainer is never executed)
# ---------------------------------------------------------------------------
def _producer_tree() -> ast.Module:
    return ast.parse(PRODUCER_PATH.read_text(encoding="utf-8"))


def _head_producer_source():
    try:
        finished = subprocess.run(
            ["git", "show", "HEAD:molmetal/scripts/train_metal_lomo_wasserstein.py"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:  # pragma: no cover - no git available
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout


def test_producer_calls_the_builder_and_stores_its_output():
    tree = _producer_tree()

    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("wfa205_lomo_annotation_builders"):
                imported = any(
                    alias.name == "build_lomo_headline_caveat" for alias in node.names
                )
    assert imported, "the producer does not import the caveat builder"

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_lomo_headline_caveat"
    ]
    assert calls, "the producer never calls the caveat builder"
    assert any(
        call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "payload"
        for call in calls
    ), "the builder is not called with the measurement payload"

    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and node.targets
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Constant)
        and node.targets[0].slice.value == "headline_caveat"
    ]
    assert stores, "the computed caveat is not stored under headline_caveat"


def test_producer_payload_keys_are_strictly_additive():
    """No existing key removed, renamed or replaced -- only additions."""
    head_source = _head_producer_source()
    if head_source is None:
        pytest.skip("git is unavailable, cannot compare against the committed producer")

    def payload_keys(source: str) -> set:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
                if (
                    targets
                    and isinstance(targets[0], ast.Name)
                    and targets[0].id == "payload"
                    and isinstance(node.value, ast.Dict)
                ):
                    return {
                        key.value
                        for key in node.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
        raise AssertionError("no payload dict literal found")

    before = payload_keys(head_source)
    after = payload_keys(PRODUCER_PATH.read_text(encoding="utf-8"))
    assert before <= after, "a payload key was removed or renamed"
    assert after - before == {"per_metal_resolvability"}


def test_producer_cli_surface_is_unchanged():
    head_source = _head_producer_source()
    if head_source is None:
        pytest.skip("git is unavailable, cannot compare against the committed producer")

    def flags(source: str) -> set:
        return set(re.findall(r'"(--[a-z0-9][a-z0-9\-]*)"', source))

    assert flags(head_source) == flags(PRODUCER_PATH.read_text(encoding="utf-8"))


def test_producer_write_path_is_unchanged():
    source = PRODUCER_PATH.read_text(encoding="utf-8")
    assert "metric_path.write_text(json.dumps(payload, indent=2, default=str))" in source


# ---------------------------------------------------------------------------
# 4. the resolvability inputs are DERIVED from the run's own cohort
# ---------------------------------------------------------------------------
def test_rule_threshold_comes_from_the_project_constant():
    from molmetal.scripts.metal_activity_transfer_probe import MIN_UNIQUE_FOR_GATE

    assert resolve_min_unique_for_gate({}) == MIN_UNIQUE_FOR_GATE
    # an explicit argument wins over the payload and over the project rule
    assert resolve_min_unique_for_gate({}, min_unique_for_gate=7) == 7
    assert resolve_min_unique_for_gate(
        {"per_metal_resolvability": {"min_unique_for_gate": 11}}
    ) == 11


def test_producer_cohort_agrees_with_the_audit_on_the_resolvable_set():
    if not LOMO_CSV.exists():
        pytest.skip("the LOMO cohort CSV is not present")
    producer = pytest.importorskip("molmetal.scripts.train_metal_lomo_wasserstein")

    namespace = argparse.Namespace(
        csv=str(LOMO_CSV),
        time_threshold=24.0,
        dedup_ligands=False,
        quick=False,
        rh_oversample=0,
        rh_cohort_json="",
        rh_cohort_active_json="",
    )
    dataset, _ = producer._build_dataset(namespace, rh_over=0)
    block = producer._per_metal_resolvability(dataset)
    counts = block["n_unique_ligands"]
    threshold = block["min_unique_for_gate"]
    audit_counts = _audit_folds()

    # the two cohorts are not identical (the trainer filters at 24 h, the
    # audit reads the raw CSV), so the counts may differ -- but the
    # RESOLVABLE SET must not, and the folds that fail the rule must carry
    # the same count in both.
    def resolvable(counter):
        return {m for m, n in counter.items() if n >= threshold}

    assert resolvable(counts) == resolvable(audit_counts)
    for metal, count in audit_counts.items():
        if count < threshold:
            assert counts[metal] == count


def test_fabricated_rows_are_excluded_from_the_count():
    if not LOMO_CSV.exists():
        pytest.skip("the LOMO cohort CSV is not present")
    producer = pytest.importorskip("molmetal.scripts.train_metal_lomo_wasserstein")
    from molmetal.data.cytotox import MetalCytotoxDataset

    frame = _fake_frame()
    dataset = MetalCytotoxDataset(df=frame, cache_dir="/tmp/wfa205_lomo_cache")
    block = producer._per_metal_resolvability(dataset)
    assert block["n_fabricated_rows_excluded"] == 2
    # the two fabricated ligands must NOT be counted for their metal
    assert block["n_unique_ligands"]["Rh"] == 1


def _fake_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "Metal": ["Rh", "Rh", "Rh", "Ru", "Ru"],
            "SMILES_Ligands": ["a", "b", "c", "x", "y"],
            "Cell_line": [
                "SYNTHETIC batch 1",
                "SYNTHETIC batch 1",
                "MCF7",
                "MCF7",
                "MCF7",
            ],
        }
    )


def test_describe_exposes_the_derived_facts():
    payload, _ = _parity_payload()
    facts = describe_lomo_headline(payload)
    assert facts["n_folds"] == 5
    assert facts["headline_matches_equal_weight_mean"] is True
    assert facts["headline_below_gate"] is True
    assert facts["weakest_metal"] == "Rh"
    assert facts["real_only_fold_metal"] == "Rh"
    assert set(facts["unresolvable_metals"]) == {"Os", "Rh", "Re"}
    assert set(facts["resolvable_metals"]) == {"Ru", "Ir"}
