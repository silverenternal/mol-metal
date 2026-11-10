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

r"""Wf-A269 wave-10 — the L5 hook PAYLOAD-VALUE contract.

WHAT THIS FILE GUARDS (Wf-A269 wave-10, 2026-11-02)
----------------------------------------------------
The 9 L5 per-baseline protocol-validation gates of
``TODO/03_plans/PRE_TRAINING_GATE_v2.md`` §4 all route through
``pre_flight_check._hook_passthrough_payload``.  That helper returned
PASS iff ``source_present AND emitted`` — so the gate could not fail on
the VALUE it exists to check: a hook that emitted an EMPTY (or a
wrong-valued) payload still PASSed, and the evidence string claimed
"hook emitted with source_present=True" while the payload carried no
measurement at all.  That is the §6.266 D3-OPEN dead-gate class (a check
that cannot fail) — the same class as the 7 launch gates repaired in
Wf-A269 wave-10.

THE REPAIR
----------
``L5_HOOK_VALUE_CONTRACT`` names, per hook_id, the values the emitted
payload must carry, where each one is read from, and the range (or the
exact value) it must fall in.  The payload now carries the extracted
``values`` and the gate FAILS when a required value is missing, is out
of range, or disagrees with the payload that actually reached the disk.

THE CONTRACTS THIS FILE PINS
----------------------------
1. WRONG-VALUED PAYLOAD -> FAIL (the headline: a syntactically valid
   hook payload carrying a wrong value must not PASS).
2. EMPTY PAYLOAD -> FAIL (no ``values`` at all -> FAIL).
3. OUT-OF-RANGE SOURCE VALUE -> FAIL (the source is present and
   parseable, but the number it holds is not an AUC).
4. MISSING REQUIRED KEY -> FAIL (the source is present but does not
   carry the key the hook exists to report).
5. MISSING SOURCE FILE -> WARN (the pre-repair branch, preserved: a
   hook whose primary source is absent has nothing to fail).
6. CORRECT SOURCE -> PASS, and the emitted hook file carries the
   checked values (so the record shows the measurement, not just a
   file existing).
7. ALL 9 HOOK IDS DECLARE A CONTRACT (a hook without one cannot
   return PASS).
8. ZERO LIVE IMPACT — the 9 L5 gates are registered in NO pipeline, so
   the live gate counts are unchanged (129 / 131 / 25 / 18 / 8).
   SUPERSEDED 2026-11-04 (Wf-A269 wave-19): the counts in that sentence
   are the HISTORICAL reading; the live counts are 133 / 135 / 25 / 18 / 8
   (see test_live_gate_counts_are_unchanged).

Every test runs on CPU with ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import pre_flight_check as pfc  # noqa: E402

# The 9 L5 hook_ids (3 baselines x 3 hooks) — the keys of the contract.
L5_HOOK_IDS = (
    "krasnov_m1_per_seed_per_split_5fold_5metal",
    "krasnov_m21_per_cell_line_per_seed",
    "krasnov_protocol_alignment_diffs_documented",
    "rusanov_m2_per_cell_line_per_cohort",
    "rusanov_m2_4cell_baseline_comparison",
    "rusanov_protocol_alignment_diffs_documented",
    "drugood_m17_assay_split_auc",
    "drugood_m17_scaffold_split_auc",
    "drugood_m17_size_split_auc",
)

# Real on-disk sources (MEASURED on this tree, 2026-11-02).
DRUGOOD_METRIC = REPO_ROOT / "metrics" / "by_metric" / "drugood_ood_auc.json"
METRIC_M1 = REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
NCI60_DIR = REPO_ROOT / "molmetal" / "reports" / "wf_nci60_precious_9cell"
NCI60_FINAL_MD = NCI60_DIR / "final.md"


def _drugood_shape(**overrides) -> dict:
    """A minimal DrugOOD-shaped metric JSON (assay axis)."""
    data = {
        "per_axis_auc": {"assay": 0.7304477112563803},
        "imold_reference": {"assay": 0.7211},
        "erm_baseline": {"assay": 0.7163},
        "per_axis_auc_delta_vs_imold": {"assay": 0.009347711256380342},
        "results": {"assay": {"n_seeds": 3}},
        "n_official_test_rows": {"assay": 53291},
    }
    for key, value in overrides.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _run_assay_gate(hook_dir: Path, source: Path):
    return pfc.gate_drugood_m17_assay_split_auc(
        "drugood_m17_assay_split_auc", "P3", "all", hook_dir, source
    )


def _install_fake_emitter(monkeypatch, payload: dict, hook_dir: Path):
    """Replace the emitter with one that writes ``payload`` verbatim and
    reports success — i.e. a syntactically valid hook file whose VALUES
    are wrong (or absent)."""

    def _fake_emit(hook_dir_arg, baseline_name, hook_id, _payload):
        hook_dir_arg.mkdir(parents=True, exist_ok=True)
        out = hook_dir_arg / f"{baseline_name}_hook.json"
        merged = {}
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    merged.update(existing)
            except Exception:  # noqa: BLE001
                merged = {}
        merged[hook_id] = payload
        out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        return True

    monkeypatch.setattr(pfc, "_emit_protocol_validation_hook_json", _fake_emit)


# ---------------------------------------------------------------------------
# 1. THE HEADLINE — a wrong-valued payload must FAIL, not PASS.
# ---------------------------------------------------------------------------


def test_wrong_valued_payload_fails_instead_of_passing(tmp_path, monkeypatch):
    """PRE-REPAIR this returned PASS: the gate only asked whether a
    source file existed and whether the emitter returned True, so a hook
    payload carrying ``ood_auc = 7.5`` (an impossible ROC-AUC) was
    reported as a PASS with the evidence "hook emitted with
    source_present=True"."""
    hook_dir = tmp_path / "protocol_validation"
    wrong_payload = {
        "baseline": "drugood",
        "hook_id": "drugood_m17_assay_split_auc",
        "hook_emitted": True,
        "source_present": True,
        "verdict_field": "iMoLD=0.7211, ERM=0.7163 (baseline)",
        "values": {
            "ood_auc": 7.5,
            "imold_reference": 0.7211,
            "erm_baseline": 0.7163,
            "delta_vs_imold": 0.009347711256380342,
            "delta_is_consistent": 0.009347711256380342,
            "n_seeds": 3,
            "n_official_test_rows": 53291,
        },
    }
    _install_fake_emitter(monkeypatch, wrong_payload, hook_dir)

    res = _run_assay_gate(hook_dir, DRUGOOD_METRIC)

    assert res.status == "FAIL", (
        f"a wrong-valued hook payload must FAIL, not {res.status}; "
        f"evidence={res.evidence}"
    )
    assert "PASS" not in res.evidence


def test_empty_payload_fails_instead_of_passing(tmp_path, monkeypatch):
    """PRE-REPAIR this returned PASS too: an EMPTY payload (no values at
    all) was indistinguishable from a correct one, because the gate
    never looked inside the payload."""
    hook_dir = tmp_path / "protocol_validation"
    empty_payload = {
        "baseline": "drugood",
        "hook_id": "drugood_m17_assay_split_auc",
        "hook_emitted": True,
        "source_present": True,
        "verdict_field": "iMoLD=0.7211, ERM=0.7163 (baseline)",
    }
    _install_fake_emitter(monkeypatch, empty_payload, hook_dir)

    res = _run_assay_gate(hook_dir, DRUGOOD_METRIC)

    assert res.status == "FAIL", (
        f"an empty hook payload must FAIL, not {res.status}; "
        f"evidence={res.evidence}"
    )


# ---------------------------------------------------------------------------
# 2. The SOURCE side — an out-of-range or absent value must FAIL.
# ---------------------------------------------------------------------------


def test_out_of_range_source_value_fails(tmp_path):
    """PRE-REPAIR this returned PASS: the source file existed, so the
    gate passed — even though the file held an AUC of 7.5, which is not
    a possible ROC-AUC."""
    hook_dir = tmp_path / "protocol_validation"
    source = _write_json(
        tmp_path / "drugood_ood_auc.json",
        _drugood_shape(
            per_axis_auc={"assay": 7.5},
            per_axis_auc_delta_vs_imold={"assay": 7.5 - 0.7211},
        ),
    )

    res = _run_assay_gate(hook_dir, source)

    assert res.status == "FAIL", (
        f"an out-of-range value must FAIL, not {res.status}; "
        f"evidence={res.evidence}"
    )
    assert "outside" in res.evidence, res.evidence


def test_missing_required_key_fails(tmp_path):
    """PRE-REPAIR this returned PASS: the file existed, but the key the
    hook exists to report (``per_axis_auc``) was absent from it."""
    hook_dir = tmp_path / "protocol_validation"
    source = _write_json(
        tmp_path / "drugood_ood_auc.json",
        _drugood_shape(per_axis_auc=None),
    )

    res = _run_assay_gate(hook_dir, source)

    assert res.status == "FAIL", (
        f"a missing required key must FAIL, not {res.status}; "
        f"evidence={res.evidence}"
    )
    assert "missing JSON key" in res.evidence, res.evidence


def test_out_of_range_text_value_fails(tmp_path):
    """The text-sourced contract (L5-Krasnov-2, the per-cell-line
    report) is checked the same way: the mean AUC in the report must be
    an AUC."""
    hook_dir = tmp_path / "protocol_validation"
    report_dir = tmp_path / "nci60"
    report_dir.mkdir()
    patched = NCI60_FINAL_MD.read_text(encoding="utf-8").replace(
        "Per-cell AUC mean = 0.8007", "Per-cell AUC mean = 1.7000"
    )
    (report_dir / "final.md").write_text(patched, encoding="utf-8")

    res = pfc.gate_krasnov_m21_per_cell_line_per_seed(
        "krasnov_m21_per_cell_line_per_seed", "P3", "all", hook_dir, report_dir
    )

    assert res.status == "FAIL", (
        f"an out-of-range reported AUC must FAIL, not {res.status}; "
        f"evidence={res.evidence}"
    )
    assert "outside" in res.evidence, res.evidence


def test_missing_secondary_source_fails(tmp_path):
    """The Rusanov 4-cell comparison needs the Rusanov baseline
    artefact.  When it is absent the gate cannot make the comparison,
    so it must NOT report PASS (PRE-REPAIR it did: the metric JSON was
    present, so the OR-ed ``source_present`` was True)."""
    hook_dir = tmp_path / "protocol_validation"
    missing_baseline = tmp_path / "rusanov_baseline_dir_absent"

    res = pfc.gate_rusanov_m2_4cell_baseline_comparison(
        "rusanov_m2_4cell_baseline_comparison",
        "P3",
        "all",
        hook_dir,
        missing_baseline,
        REPO_ROOT / "metrics" / "by_metric" / "m2_r2_per_cohort.json",
    )

    assert res.status == "FAIL", (
        f"a missing baseline artefact must FAIL, not {res.status}; "
        f"evidence={res.evidence}"
    )
    assert "missing" in res.evidence, res.evidence


# ---------------------------------------------------------------------------
# 3. The WARN branch is preserved, and the PASS path still works.
# ---------------------------------------------------------------------------


def test_missing_source_file_still_warns(tmp_path):
    """The pre-repair WARN branch (a missing source file) is preserved:
    there is nothing to check, so the hook reports WARN — never PASS,
    never FAIL."""
    hook_dir = tmp_path / "protocol_validation"
    absent = tmp_path / "not_a_real_metric.json"

    res = _run_assay_gate(hook_dir, absent)

    assert res.status == "WARN", (
        f"a missing source file must stay WARN, got {res.status}; "
        f"evidence={res.evidence}"
    )


def test_unwritable_hook_dir_still_warns(tmp_path):
    """A hook whose payload cannot be written stays WARN (pre-repair
    branch, preserved): the emit failure is not a value failure."""
    blocked = tmp_path / "protocol_validation"
    blocked.write_text("not a directory", encoding="utf-8")

    res = _run_assay_gate(blocked, DRUGOOD_METRIC)

    assert res.status == "WARN", (
        f"a failed emit must stay WARN, got {res.status}; evidence={res.evidence}"
    )
    assert "NOT emitted" in res.evidence, res.evidence


def test_correct_source_passes_and_the_hook_file_carries_the_values(tmp_path):
    """The PASS path: the real DrugOOD metric satisfies the contract,
    and the emitted hook file records the checked values."""
    hook_dir = tmp_path / "protocol_validation"

    res = _run_assay_gate(hook_dir, DRUGOOD_METRIC)

    assert res.status == "PASS", (
        f"the real drugood metric must PASS the contract, got {res.status}; "
        f"evidence={res.evidence}"
    )
    written = json.loads(
        (hook_dir / "drugood_hook.json").read_text(encoding="utf-8")
    )
    payload = written["drugood_m17_assay_split_auc"]
    assert payload["values"]["ood_auc"] == 0.7304477112563803, payload
    assert payload["values"]["n_seeds"] == 3, payload
    assert payload["value_contract"] == "drugood_m17_assay_split_auc", payload


# ---------------------------------------------------------------------------
# 4. Coverage + zero live impact.
# ---------------------------------------------------------------------------


def test_every_l5_hook_id_declares_a_value_contract():
    """Each of the 9 hooks must name the values it checks — otherwise a
    future hook could be added without a contract and would inherit the
    presence-only PASS this repair removed."""
    contract = pfc.L5_HOOK_VALUE_CONTRACT
    for hook_id in L5_HOOK_IDS:
        assert hook_id in contract, f"{hook_id} has no value contract"
        entry = contract[hook_id]
        assert entry["fields"], f"{hook_id} declares no field to check"
        assert entry["primary_source"] in ("metric_json", "dir"), entry
        for field in entry["fields"]:
            assert field["key"], entry
            assert field["kind"], entry


def test_live_gate_counts_are_unchanged():
    """The live gate counts, re-frozen against the tree.

    Wf-A269 wave-19 (2026-11-04) — THE STALE-FROZEN-PIN REPAIR.

    This assertion was a RED TEST that had been hiding three added gates:
    it read 129 / 131 while the live tree measured 132 / 134.  A frozen
    expectation that no longer matches the tree does not fail loudly — it
    fails as a mystery, and it hides exactly the change it was written to
    detect.

    The counts are now 133 / 135, because
    ``p3_script_contract_check`` — the FOURTH §5.11 gate, the one that
    covers the class the other three structurally cannot see (a repo-local
    script whose name is NOT ``train_*``) — was REGISTERED on the pre-flight
    rather than shipped as a standalone script that nothing ran.

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6
    (the pin as it stood, and the reading it asserted):
        \"\"\"The repair touches only the helper of the 9 UNREGISTERED L5
        gates, so the live pre-flight is unchanged: 129 / 131 / 25 / 18 / 8.\"\"\"
        counts = {
            k: len(pfc.build_gates(k))
            for k in ("all", "launch", "frontier", "chemprop_v2_v3", "cfm")
        }
        assert counts == {
            "all": 129,
            "launch": 131,
            "frontier": 25,
            "chemprop_v2_v3": 18,
            "cfm": 8,
        }, f"live gate counts drifted: {counts}"
    """
    counts = {
        k: len(pfc.build_gates(k))
        for k in ("all", "launch", "frontier", "chemprop_v2_v3", "cfm")
    }
    assert counts == {
        "all": 133,
        "launch": 135,
        "frontier": 25,
        "chemprop_v2_v3": 18,
        "cfm": 8,
    }, f"live gate counts drifted: {counts}"
    assert counts == pfc.EXPECTED_GATE_COUNTS
