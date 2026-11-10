"""Wf-A169 — re-generate the 24-gate wiring patch against the CURRENT
working-tree molmetal/scripts/pre_flight_check.py.

WHY THIS EXISTS
---------------
`molmetal/scripts/pre_flight_check.py` is owned by the main thread of this
wave and was being edited CONCURRENTLY while this patch was generated.  The
committed diff therefore applies to a SPECIFIC snapshot (its sha256 is
recorded in the diff header and in apply_probe.json).  If the file moves
again, `git apply --check` will fail with a context mismatch.

Run this script to re-base the patch onto whatever the file looks like now:

    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/reports/wfa169_dead_gate_repair/regen_patch.py

It rewrites, in place:
    molmetal/reports/wfa169_dead_gate_repair/patch_wire_24_gates.diff
    molmetal/reports/wfa169_dead_gate_repair/apply_probe.json

It NEVER touches the repo's working tree, never imports torch, never runs
a trainer, never decodes a molecule.  All measurement is CPU-only.

The registration block itself lives in `regen_block.py` next to this file
(same directory) so the block text and the generator travel together.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from regen_block import (  # noqa: E402
    ANCHOR,
    ANCHOR_REPLACEMENT,
    NEW_MATRIX,
    OLD_MATRIX,
)

REPO = HERE.parents[2]
TARGET_REL = "molmetal/scripts/pre_flight_check.py"
SRC = REPO / TARGET_REL
WORK = Path("/tmp/wfa169_regen")
DIFF = HERE / "patch_wire_24_gates.diff"
PROBE = HERE / "apply_probe.json"

NAMES = [
    "p3_protocol_alignment_dataset_match",
    "p3_protocol_alignment_split_match",
    "p3_protocol_alignment_metric_match",
    "p3_protocol_alignment_seed_cohort_per_seed_per_split",
    "p3_protocol_alignment_honest_limitation_minor_section",
    "p3_protocol_alignment_paper_section_6_template",
    "p3_protocol_alignment_user_ok_gate",
    "p3_protocol_alignment_baseline_paper_refs_bib",
    "p3_protocol_recommended_seed_count_3",
    "p3_protocol_recommended_per_cell_breakdown",
    "p3_protocol_recommended_5_fold_cv_compatibility_50x",
    "p3_protocol_recommended_krasnov_paper_table_4_replication",
    "p3_protocol_recommended_drugood_3_splits_full_alignment",
    "p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup",
    "p3_protocol_recommended_section_6_honest_minor_section_auto_emit",
    "krasnov_m1_per_seed_per_split_5fold_5metal",
    "krasnov_m21_per_cell_line_per_seed",
    "krasnov_protocol_alignment_diffs_documented",
    "rusanov_m2_per_cell_line_per_cohort",
    "rusanov_m2_4cell_baseline_comparison",
    "rusanov_protocol_alignment_diffs_documented",
    "drugood_m17_assay_split_auc",
    "drugood_m17_scaffold_split_auc",
    "drugood_m17_size_split_auc",
]


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def build_mirror(target_dir: Path) -> Path:
    """Symlink farm over the repo; only the patched file is real."""
    mirror = Path("/tmp/wfa169_regen_mirror")
    if mirror.exists():
        shutil.rmtree(mirror)
    mirror.mkdir(parents=True)
    for entry in sorted(REPO.iterdir()):
        if entry.name in (".git", "molmetal"):
            continue
        (mirror / entry.name).symlink_to(entry)
    mm = mirror / "molmetal"
    mm.mkdir()
    for entry in sorted((REPO / "molmetal").iterdir()):
        if entry.name == "scripts":
            continue
        (mm / entry.name).symlink_to(entry)
    sd = mm / "scripts"
    sd.mkdir()
    for entry in sorted((REPO / "molmetal" / "scripts").iterdir()):
        if entry.name == "pre_flight_check.py":
            continue
        (sd / entry.name).symlink_to(entry)
    shutil.copy2(target_dir / TARGET_REL, sd / "pre_flight_check.py")
    return mirror


def main() -> int:
    if not SRC.exists():
        print(f"FATAL: {SRC} not found", file=sys.stderr)
        return 2

    if WORK.exists():
        shutil.rmtree(WORK)
    base = WORK / "base" / TARGET_REL
    base.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, base)
    raw = base.read_bytes()
    orig = raw.decode("utf-8")
    snap_sha = hashlib.sha256(raw).hexdigest()

    assert orig.count(OLD_MATRIX) == 1, "OLD_MATRIX anchor is not unique"
    new = orig.replace(OLD_MATRIX, NEW_MATRIX)
    assert new.count(ANCHOR) == 1, "ANCHOR is not unique"
    new = new.replace(ANCHOR, ANCHOR_REPLACEMENT)
    for n in NAMES:
        b = orig.count(n)
        a = new.count(n)
        assert b >= 1 and a == b + 3, f"{n}: {b} -> {a}"
    print(f"OK: 24 gates wired; snapshot sha256={snap_sha} bytes={len(raw)}")

    a_side = WORK / "a" / TARGET_REL
    b_side = WORK / "b" / TARGET_REL
    for p in (a_side, b_side):
        p.parent.mkdir(parents=True, exist_ok=True)
    a_side.write_text(orig, encoding="utf-8")
    b_side.write_text(new, encoding="utf-8")

    diff_text = run(
        [
            "diff", "-u",
            "--label", f"a/{TARGET_REL}",
            "--label", f"b/{TARGET_REL}",
            str(a_side), str(b_side),
        ]
    ).stdout
    header = (
        f"# Wf-A169 (2026-11-01) — register the 24 Wf-432-A1 gates that were\n"
        f"# DEFINED but NEVER REGISTERED in molmetal/scripts/pre_flight_check.py.\n"
        f"#\n"
        f"# Generated against a SNAPSHOT of the working-tree file, NOT against HEAD:\n"
        f"#   path            : {TARGET_REL}\n"
        f"#   snapshot sha256 : {snap_sha}\n"
        f"#   snapshot bytes  : {len(raw)}\n"
        f"#   snapshot mtime  : {base.stat().st_mtime}\n"
        f"#   Re-generate with:\n"
        f"#     CUDA_VISIBLE_DEVICES=\"\" .venv/bin/python \\\n"
        f"#       molmetal/reports/wfa169_dead_gate_repair/regen_patch.py\n"
        f"#\n"
        f"# This patch registers 24 gates; it does NOT update\n"
        f"# EXPECTED_GATE_COUNTS (owned by another branch of this wave).\n"
    )
    DIFF.write_text(header + diff_text, encoding="utf-8")
    print(f"OK: wrote {DIFF} ({DIFF.stat().st_size} bytes, "
          f"{diff_text.count(chr(10) + '@@')} hunks)")

    # ---- scratch-copy apply + measure ------------------------------------
    apply_dir = WORK / "apply"
    (apply_dir / TARGET_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base, apply_dir / TARGET_REL)
    run(["git", "init", "-q"], cwd=apply_dir)
    run(["git", "add", "-A"], cwd=apply_dir)
    run(["git", "-c", "user.email=a@b.c", "-c", "user.name=wf",
         "commit", "-qm", "scratch base"], cwd=apply_dir)

    check = run(["git", "apply", "--check", "-v", str(DIFF)], cwd=apply_dir)
    appl = run(["git", "apply", "-v", str(DIFF)], cwd=apply_dir)
    applied = (apply_dir / TARGET_REL).read_bytes()
    identical = applied == b_side.read_bytes()
    print(f"OK: git apply --check rc={check.returncode} "
          f"git apply rc={appl.returncode} byte_identical={identical}")

    mirror = build_mirror(apply_dir)
    import importlib.util
    import json

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    base_mod = load(base, "pfc_base")
    patched_mod = load(mirror / TARGET_REL, "pfc_patched")
    pipelines = ("all", "launch", "frontier", "chemprop_v2_v3", "cfm")
    b_counts = {p: len(base_mod.build_gates(p)) for p in pipelines}
    a_counts = {p: len(patched_mod.build_gates(p)) for p in pipelines}
    runners = dict(patched_mod.build_gates("all"))
    verdicts = {}
    for g in NAMES:
        res = runners[g]()
        verdicts[g] = {"status": res.status, "evidence": res.evidence}

    live_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    probe = {
        "tag": "MEASURED",
        "wave": "Wf-A169",
        "measured_on": "2026-11-01",
        "gpu": "none (CUDA_VISIBLE_DEVICES=\"\", no torch import, no trainer, no decode)",
        "method": {
            "scratch_git_repo": str(apply_dir),
            "mirror_for_import": str(mirror),
            "why_mirror": (
                "pre_flight_check.py sets REPO_ROOT = Path(__file__).resolve()"
                ".parents[2]; a patched copy parked at an unrelated path would make "
                "every gate see a missing data path. The mirror is a symlink farm "
                "over the real repo with only the patched file real."
            ),
            "steps": [
                "copy the current working-tree file to a scratch git repo",
                "git apply --check -v <diff>",
                "git apply -v <diff>",
                "byte-compare applied file vs the expected patched file",
                "import the patched module; measure len(build_gates(pipeline))",
                "invoke each of the 24 newly-registered runners and record the verdict",
            ],
            "working_tree_modified": False,
        },
        "patch": {
            "path": "molmetal/reports/wfa169_dead_gate_repair/patch_wire_24_gates.diff",
            "bytes": DIFF.stat().st_size,
            "sha256": hashlib.sha256(DIFF.read_bytes()).hexdigest(),
            "a_side_sha256": snap_sha,
            "b_side_sha256": hashlib.sha256(applied).hexdigest(),
        },
        "git_apply_check": {
            "returncode": check.returncode,
            "ok": check.returncode == 0,
            "stdout": check.stdout[-2000:],
            "stderr": check.stderr[-2000:],
        },
        "git_apply": {
            "returncode": appl.returncode,
            "ok": appl.returncode == 0,
            "stdout": appl.stdout[-2000:],
            "stderr": appl.stderr[-2000:],
        },
        "applied_file_byte_identical_to_expected": identical,
        "live_worktree_sha256_at_probe_time": live_sha,
        "live_worktree_still_matches_a_side": live_sha == snap_sha,
        "gate_counts": {
            "baseline_unpatched": b_counts,
            "patched": a_counts,
            "delta": {p: a_counts[p] - b_counts[p] for p in pipelines},
            "baseline_expected_gate_counts_constant": dict(base_mod.EXPECTED_GATE_COUNTS),
            "patched_expected_gate_counts_constant": dict(patched_mod.EXPECTED_GATE_COUNTS),
            "patched_unique_names_all": len({r[0] for r in patched_mod.build_gates("all")}),
        },
        "registered_new_gate_names_in_all": {g: (g in runners) for g in NAMES},
        "n_new_gates_registered": sum(1 for g in NAMES if g in runners),
        "new_gate_verdicts": verdicts,
        "verdict_tally": {
            s: sum(1 for v in verdicts.values() if v["status"] == s)
            for s in ("PASS", "WARN", "FAIL", "NOT_REGISTERED")
        },
        "notes": [
            "EXPECTED_GATE_COUNTS is NOT bumped by this patch; the explicit "
            "expected_total drift-check gate will FAIL until the count-pin "
            "branch lands. Intentional: EXPECTED_GATE_COUNTS is another branch's file.",
            "frontier / chemprop_v2_v3 / cfm counts are UNCHANGED by this patch.",
            "The 24 verdicts are the gates' CURRENT on-disk verdict, not a "
            "property of the wiring; a FAIL/WARN here is an honest-negative "
            "surfaced by the registration, not a wiring defect.",
        ],
    }
    PROBE.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OK: wrote {PROBE}")
    print(json.dumps({
        "base": b_counts, "patched": a_counts,
        "delta": probe["gate_counts"]["delta"],
        "n_registered": probe["n_new_gates_registered"],
        "verdict_tally": probe["verdict_tally"],
        "live_matches_a_side": probe["live_worktree_still_matches_a_side"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
