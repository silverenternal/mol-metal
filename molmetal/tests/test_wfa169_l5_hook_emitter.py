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

"""Wf-A169 — tests for the L5 protocol-validation hook emitter.

Wf-A169 MEASURED two defects in ``_emit_protocol_validation_hook_json``
(the helper behind the 9 L5 per-baseline protocol-validation gates of
``TODO/03_plans/PRE_TRAINING_GATE_v2.md`` §4) on this tree, 2026-11-01:

  (D1) THE HOOK WAS NEVER EMITTED.  The emitter was a no-op whenever
       ``hook_dir`` did not already exist and nothing in the repo creates
       it.  ``molmetal/reports/protocol_validation`` does not exist, so
       all 9 L5 hooks wrote NOTHING while their gate still returned PASS
       with the evidence string "hook emitted with source_present=True".

  (D2) SIBLING hook_idS WERE SILENTLY DROPPED.  Every hook of one
       baseline wrote ``{hook_id: payload}`` to the SAME
       ``{baseline}_hook.json``, replacing the file.  Running the 3
       Krasnov hooks left exactly ONE key in the file; the spec expects
       3 hook_ids per baseline (9 hooks = 3 baselines x 3 hook_ids).

The contracts this file pins:

1. AUTO-EMIT — a missing ``hook_dir`` is CREATED and the hook is written
   (the spec says the hooks "auto-emit").
2. NO SILENT OVERWRITE — the 3 hooks of one baseline coexist in ONE file.
3. NO LYING EVIDENCE — the gate's evidence string states whether the hook
   was emitted; if the write failed the gate must NOT return PASS.
4. SPEC SHAPE — 3 baselines x 3 hook_ids = 9 hooks across 3 files.
5. ZERO LIVE IMPACT — the live gate counts are unchanged (the 9 L5 gates
   are registered in NO pipeline, so this repair is inert for the
   pre-flight) and the repair is confined to the dead helper.

Every test runs on CPU with ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import pre_flight_check as pfc  # noqa: E402

# The 9 L5 gates, grouped by baseline (spec §4: 3 baselines x 3 hook_ids).
KRASNOV_GATES = (
    "gate_krasnov_m1_per_seed_per_split_5fold_5metal",
    "gate_krasnov_m21_per_cell_line_per_seed",
    "gate_krasnov_protocol_alignment_diffs_documented",
)
RUSANOV_GATES = (
    "gate_rusanov_m2_per_cell_line_per_cohort",
    "gate_rusanov_m2_4cell_baseline_comparison",
    "gate_rusanov_protocol_alignment_diffs_documented",
)
DRUGOOD_GATES = (
    "gate_drugood_m17_assay_split_auc",
    "gate_drugood_m17_scaffold_split_auc",
    "gate_drugood_m17_size_split_auc",
)
ALL_L5_GATES = KRASNOV_GATES + RUSANOV_GATES + DRUGOOD_GATES

# Sources that exist on this tree (used so the gates reach their PASS path).
METRIC_M1 = REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
METRIC_M2 = REPO_ROOT / "metrics" / "by_metric" / "m2_r2_per_cohort.json"
NCI60_DIR = REPO_ROOT / "molmetal" / "reports" / "wf_nci60_precious_9cell"
CATALOG = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"
MATRIX = REPO_ROOT / "TODO" / "03_baselines" / "PROTOCOL_ALIGNMENT_MATRIX.md"
DRUGOOD_RESULTS = REPO_ROOT / "molmetal" / "reports" / "wf265_collect" / "all_results.json"


def _run_krasnov(hook_dir: Path) -> list:
    """Run the 3 Krasnov L5 gates against ``hook_dir``; return the results."""
    return [
        pfc.gate_krasnov_m1_per_seed_per_split_5fold_5metal(
            "g1", "P3", "all", hook_dir, METRIC_M1
        ),
        pfc.gate_krasnov_m21_per_cell_line_per_seed(
            "g2", "P3", "all", hook_dir, NCI60_DIR
        ),
        pfc.gate_krasnov_protocol_alignment_diffs_documented(
            "g3", "P3", "all", hook_dir, CATALOG, MATRIX
        ),
    ]


def test_d1_missing_hook_dir_is_created_and_the_hook_is_written(tmp_path):
    """D1 (negative control on the PRE-repair behaviour): the pre-repair
    emitter was a no-op when hook_dir was missing, so the hook never
    reached the disk.  The repair creates the directory and writes it."""
    hook_dir = tmp_path / "protocol_validation"
    assert not hook_dir.exists()

    results = _run_krasnov(hook_dir)

    assert hook_dir.exists(), "the emitter must create hook_dir (auto-emit)"
    out = hook_dir / "krasnov_hook.json"
    assert out.exists(), "the hook JSON must reach the disk"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload) == 3, f"expected 3 hook_ids, got {sorted(payload)}"
    for res in results:
        assert res.status in ("PASS", "WARN")


def test_d2_sibling_hook_ids_are_merged_not_overwritten(tmp_path):
    """D2 (negative control on the PRE-repair behaviour): the pre-repair
    emitter replaced the file, so only the LAST hook_id survived."""
    hook_dir = tmp_path / "protocol_validation"
    _run_krasnov(hook_dir)

    payload = json.loads((hook_dir / "krasnov_hook.json").read_text(encoding="utf-8"))
    expected = {
        "krasnov_m1_per_seed_per_split_5fold_5metal",
        "krasnov_m21_per_cell_line_per_seed",
        "krasnov_protocol_alignment_diffs_documented",
    }
    assert set(payload) == expected, (
        f"the 3 hooks of one baseline must coexist in ONE file; got {sorted(payload)}"
    )


def test_spec_shape_three_baselines_three_hooks_nine_total(tmp_path):
    """The spec (§4) names 3 hook files, each holding 3 hook_ids."""
    hook_dir = tmp_path / "protocol_validation"
    _run_krasnov(hook_dir)

    pfc.gate_rusanov_m2_per_cell_line_per_cohort(
        "g1", "P3", "all", hook_dir, METRIC_M2
    )
    pfc.gate_rusanov_m2_4cell_baseline_comparison(
        "g2", "P3", "all", hook_dir,
        REPO_ROOT / "molmetal" / "reports" / "rusanov_2025_paper_table_s4_s5_baseline",
        METRIC_M2,
    )
    pfc.gate_rusanov_protocol_alignment_diffs_documented(
        "g3", "P3", "all", hook_dir, CATALOG, MATRIX
    )
    pfc.gate_drugood_m17_assay_split_auc(
        "drugood_m17_assay_split_auc", "P3", "all", hook_dir, DRUGOOD_RESULTS
    )
    pfc.gate_drugood_m17_scaffold_split_auc(
        "drugood_m17_scaffold_split_auc", "P3", "all", hook_dir, DRUGOOD_RESULTS
    )
    pfc.gate_drugood_m17_size_split_auc(
        "drugood_m17_size_split_auc", "P3", "all", hook_dir, DRUGOOD_RESULTS
    )

    files = sorted(p.name for p in hook_dir.glob("*_hook.json"))
    assert files == [
        "drugood_hook.json",
        "krasnov_hook.json",
        "rusanov_hook.json",
    ], f"the spec names 3 hook files; got {files}"

    total = 0
    for name in files:
        payload = json.loads((hook_dir / name).read_text(encoding="utf-8"))
        assert len(payload) == 3, f"{name} must hold 3 hook_ids; got {sorted(payload)}"
        total += len(payload)
    assert total == 9, f"9 hooks = 3 baselines x 3 hook_ids; got {total}"


def test_evidence_never_claims_emitted_when_the_write_failed(tmp_path):
    """D3 — the pre-repair evidence string said "hook emitted" even when
    the emitter was a no-op.  A write failure must surface as a WARN and
    the evidence must not say "emitted"."""
    # Make hook_dir a FILE so mkdir() must fail.
    blocked = tmp_path / "protocol_validation"
    blocked.write_text("not a directory", encoding="utf-8")

    res = pfc.gate_krasnov_m1_per_seed_per_split_5fold_5metal(
        "g1", "P3", "all", blocked, METRIC_M1
    )

    assert res.status == "WARN", f"a failed emit must not PASS; got {res.status}"
    assert "NOT emitted" in res.evidence, res.evidence
    assert "hook emitted" not in res.evidence, res.evidence


def test_zero_live_impact_gate_counts_are_unchanged():
    """The repair is confined to a helper only the 9 UNREGISTERED L5 gates
    call, so the live pre-flight must be byte-for-byte identical in
    behaviour.

    The ORIGINAL pin was **129 / 131 / 25 / 18 / 8** (Wf-A169, 2026-11-01).
    That rendering is PRESERVED VERBATIM above per AGENTS.md §5.5 rule #6;
    the live census has since moved to **132 / 134 / 25 / 18 / 8** because
    later waves of THIS chain REGISTERED gates (``p3_launch_contract_check``
    and two more), each of which moved the module's own
    ``EXPECTED_GATE_COUNTS`` in lockstep.  The load-bearing assertion is the
    second one — the live counts must equal the MODULE's pin — and the
    literal below is the recorded measurement of that pin.

    RE-MEASURED 2026-11-04 (Wf-A269 wave-17/18): **132 / 134 / 25 / 18 / 8**
    (the module's ``EXPECTED_GATE_COUNTS`` and ``len(build_gates(...))``
    agree exactly; the stale 129/131 literal was the only defect, and it had
    been stale since the waves that registered the three extra gates).
    """
    counts = {k: len(pfc.build_gates(k)) for k in
              ("all", "launch", "frontier", "chemprop_v2_v3", "cfm")}
    assert counts == {
        "all": 132,
        "launch": 134,
        "frontier": 25,
        "chemprop_v2_v3": 18,
        "cfm": 8,
    }, f"live gate counts drifted: {counts}"
    assert counts == pfc.EXPECTED_GATE_COUNTS


def test_the_nine_l5_gates_are_still_registered_in_no_pipeline():
    """None of the 9 L5 gate names appears in a build_gates registration
    tuple — they are DEAD code.  If this ever changes, the repair above
    stops being inert and the gate-count pins must move with it."""
    src = (REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    registered = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    registered.add(elt.value)
    for gate in ALL_L5_GATES:
        assert gate not in registered, (
            f"{gate} is now registered — the live gate count changed and the "
            "pin in EXPECTED_GATE_COUNTS must be updated in lockstep"
        )


def test_cpu_only_import_graph():
    """The pre-flight module must not pull torch into this test's graph."""
    src = (REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    assert "torch" not in imported, "the pre-flight must stay CPU-only / torch-free"
