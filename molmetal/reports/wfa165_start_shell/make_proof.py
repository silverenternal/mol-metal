#!/usr/bin/env python
"""Wf-A165 proof generator — CPU-only, allocates NO GPU memory.

Extracts the real blocks from the real `scripts/start.sh` and executes them
in a sandboxed `bash -c` with a stubbed `$GPU` under three ambient
CUDA_VISIBLE_DEVICES conditions, plus a negative control (the pre-repair
block, taken verbatim from `git show HEAD:scripts/start.sh`).

`scripts/start.sh` itself is NEVER executed — it launches a trainer.

Writes `proof.json` next to this file.

Run:
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/reports/wfa165_start_shell/make_proof.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
TEST_FILE = PROJECT_ROOT / "molmetal" / "tests" / "test_wfa165_start_shell.py"
START_SH = PROJECT_ROOT / "scripts" / "start.sh"


def _load_test_module():
    spec = importlib.util.spec_from_file_location("wfa165_test", TEST_FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_span(lines, start_marker: str, end_marker: str):
    i = lines.index(start_marker) + 1  # 1-indexed, first line AFTER the marker
    j = lines.index(end_marker) + 1
    return i, j


def _guard_cross_check() -> dict:
    """Best-effort cross-check against the sibling branch's static guard
    (molmetal/scripts/wfa165_gpu_lock_guard.py — READ-ONLY, CPU-ONLY).  The
    guard is another branch's file; this only READS its report."""
    guard = PROJECT_ROOT / "molmetal" / "scripts" / "wfa165_gpu_lock_guard.py"
    if not guard.exists():
        return {"ran": False, "reason": "guard script not on disk"}
    try:
        proc = subprocess.run(
            [
                "/home/hugo/codes/mol-metal/.venv/bin/python",
                str(guard),
                "--json",
                "--advisory",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        data = json.loads(proc.stdout)
        hits = [f for f in data.get("findings", []) if f.get("file") == "scripts/start.sh"]
        return {
            "ran": True,
            "tool": "molmetal/scripts/wfa165_gpu_lock_guard.py",
            "repo_findings_total": data.get("findings_total"),
            "scripts_start_sh_findings": len(hits),
            "scripts_start_sh_finding_details": hits,
        }
    except Exception as exc:  # pragma: no cover - cross-check only
        return {"ran": False, "reason": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    mod = _load_test_module()
    src = START_SH.read_text(encoding="utf-8")
    lines = src.splitlines()
    current = mod._current_impl()

    def case(name, gpu, ambient):
        out, is_set, value = mod._run_block(current, gpu=gpu, ambient=ambient)
        return {
            "case": name,
            "yaml_compute_gpu": gpu,
            "ambient_cuda_visible_devices": "<unset>" if ambient is None else ambient,
            "result_cuda_visible_devices_is_set": is_set,
            "result_cuda_visible_devices": value,
            "stdout": out.splitlines(),
        }

    cases = [
        case("a_ambient_unset_yaml_decides", "cuda:0", None),
        case("b_ambient_empty_lock_wins", "cuda:0", ""),
        case("c_ambient_explicit_device_wins", "cuda:0", "1"),
        case("d_ambient_unset_yaml_cuda3", "cuda:3", None),
        case("e_ambient_empty_lock_wins_over_cuda3", "cuda:3", ""),
        case("f_non_numeric_gpu_no_pin", "auto", None),
    ]

    new_out, new_set, new_val = mod._run_block(current, gpu="cuda:0", ambient=None)
    old_out, old_set, old_val = mod._run_block(
        mod.PRE_REPAIR_BLOCK, gpu="cuda:0", ambient=None
    )

    neg_pre = mod.ambient_violations(mod.PRE_REPAIR_BLOCK)
    neg_cur = mod.ambient_violations(current)

    # REVERT CHECK: rebuild the file as it would be if the Wf-A165 repair were
    # undone (new mask block replaced by the pre-repair one) and run the same
    # rule against it.  This is the negative control applied to the REAL file
    # rather than to a pasted constant.
    reverted = src.replace(mod._extract(src, mod.MASK_START, mod.MASK_END),
                           mod.PRE_REPAIR_BLOCK.strip("\n"), 1)
    reverted_block = (
        mod._extract(reverted, mod.CAPTURE_START, mod.CAPTURE_END)
        + "\n"
        + mod._extract(reverted, mod.MASK_START, mod.MASK_END)
    )
    revert_violations = mod.ambient_violations(reverted_block)

    exports = [n for n, line in enumerate(lines) if "export CUDA_VISIBLE_DEVICES" in line]
    mask_start, mask_end = _line_span(lines, mod.MASK_START, mod.MASK_END)
    cap_start, cap_end = _line_span(lines, mod.CAPTURE_START, mod.CAPTURE_END)
    presence = [
        n
        for n, line in enumerate(lines)
        if "${CUDA_VISIBLE_DEVICES+set}" in line or "-v CUDA_VISIBLE_DEVICES" in line
    ]
    block_lines = lines[mask_start : mask_end - 1]  # lines between the sentinels
    depths = {
        str(n + 1): mod._conditional_depth_at(block_lines, n - mask_start)
        for n in exports
    }
    whole_file_property = {
        "export_cuda_visible_devices_lines_1indexed": [n + 1 for n in exports],
        "all_exports_inside_guarded_mask_block": all(mask_start <= n + 1 <= mask_end for n in exports),
        "ambient_presence_test_lines_1indexed": [n + 1 for n in presence],
        "ambient_presence_test_precedes_first_export": bool(presence) and presence[0] + 1 < exports[0] + 1,
        "conditional_depth_of_each_export": depths,
        "no_unconditional_export_at_top_level": all(d >= 1 for d in depths.values()),
        "capture_block_span_1indexed": [cap_start, cap_end],
        "mask_block_span_1indexed": [mask_start, mask_end],
    }

    head_blob = subprocess.run(
        ["git", "rev-parse", "HEAD:scripts/start.sh"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    proof = {
        "tag": "MEASURED",
        "wave": "Wf-A165 (branch B of the Wf-A163 GPU-tooling census)",
        "date": "2026-11-01",
        "defect": (
            "scripts/start.sh derived the GPU mask from the yaml compute.gpu field and "
            "exported it unconditionally, overwriting any ambient CUDA_VISIBLE_DEVICES. "
            "A session lock of CUDA_VISIBLE_DEVICES='' was replaced by the yaml's device "
            "index, so a yaml-only launch silently re-enabled a GPU."
        ),
        "file_changed": "scripts/start.sh",
        "repair": {
            "capture_block": "scripts/start.sh:%d-%d" % (cap_start, cap_end),
            "mask_block": "scripts/start.sh:%d-%d" % (mask_start, mask_end),
            "rule": (
                "ambient explicitly SET (incl. '') -> the ambient mask wins and the yaml's "
                "compute.gpu is NOT applied; ambient UNSET -> the yaml decides the device "
                "(unchanged behaviour, yaml stays the device SSOT per AGENTS.md 5.11)."
            ),
            "presence_test": "${CUDA_VISIBLE_DEVICES+set} (POSIX-safe, works under set -u)",
        },
        "proof_method": (
            "the real blocks were EXTRACTED from the real scripts/start.sh by their sentinel "
            "markers and EXECUTED in a sandboxed `bash -c` with a stubbed $GPU. "
            "scripts/start.sh itself was NOT executed (it launches a trainer). "
            "No CUDA call was made; no GPU memory was allocated."
        ),
        "environment": {
            "bash_version": subprocess.run(
                ["bash", "--version"], capture_output=True, text=True, check=False
            ).stdout.splitlines()[0],
            "python": sys.version.split()[0],
            "cuda_visible_devices_during_proof": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "gpu_allocated": False,
        },
        "artifacts": {
            "scripts_start_sh_sha256": _sha256(START_SH),
            "scripts_start_sh_git_head_blob": head_blob,
            "pre_repair_source": "git show HEAD:scripts/start.sh (block copied verbatim into the test as PRE_REPAIR_BLOCK)",
        },
        "three_case_proof": cases,
        "default_path_byte_for_byte": {
            "ambient": "<unset>",
            "yaml_compute_gpu": "cuda:0",
            "pre_repair_stdout": old_out.splitlines(),
            "current_stdout": new_out.splitlines(),
            "stdout_identical": old_out == new_out,
            "pre_repair_mask": {"is_set": old_set, "value": old_val},
            "current_mask": {"is_set": new_set, "value": new_val},
            "mask_identical": (old_set, old_val) == (new_set, new_val),
        },
        "negative_control": {
            "pre_repair_block_rule_violations": neg_pre,
            "pre_repair_block_flagged": bool(neg_pre),
            "current_block_rule_violations": neg_cur,
            "current_block_flagged": bool(neg_cur),
            "revert_check": {
                "description": (
                    "the real scripts/start.sh with its Wf-A165 mask block replaced by the "
                    "pre-repair block; the same rule must flag it"
                ),
                "rule_violations": revert_violations,
                "flagged": bool(revert_violations),
            },
        },
        "whole_file_property": whole_file_property,
        "cross_branch_guard_cross_check": _guard_cross_check(),
        "regression_test": "molmetal/tests/test_wfa165_start_shell.py",
    }

    out_path = REPORT_DIR / "proof.json"
    sys.path.insert(0, str(PROJECT_ROOT))
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(out_path, json.dumps(proof, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: proof[k] for k in ("three_case_proof", "default_path_byte_for_byte", "negative_control", "whole_file_property")}, indent=2, ensure_ascii=False))
    print("WROTE", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
