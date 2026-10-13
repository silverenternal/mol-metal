# SPDX-License-Identifier: Apache-2.0
"""Guard: wf_cfm_decode_dump_smiles must GENERATE, not reconstruct.

WHY THIS TEST EXISTS
--------------------
The script's first implementation loaded REAL tmQM clouds and ran only the
bond head over them (featurise -> _select_bonds -> _assemble_mol). It never
called the sampler, so every SMILES it wrote was a bond-head reconstruction of
a known molecule. That defect sat under a docstring that said the output was
"real generated molecules", and it survived because nothing in the output
looked wrong: the panels reported ~100% chemistry validity, because real
molecules really do pass every check.

This test pins the ONE thing that distinguishes the two paths: the script must
call the trainer's generation entry point, which integrates the ODE. It is a
source-level check on purpose -- running the real thing needs a checkpoint and
minutes of GPU, and a test that only runs on the machines that have both is a
test nobody runs.

RED CHECK (reported, not assumed): reverting the call to
evaluate_generation_validity back to the featurise loop makes this FAIL.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "molmetal" / "scripts" / "wf_cfm_decode_dump_smiles.py"


def _source() -> str:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def test_script_calls_the_ode_generation_path():
    """The sampler-backed entry point must be called, not the bond head alone."""
    src = _source()
    assert "evaluate_generation_validity" in src, (
        "wf_cfm_decode_dump_smiles.py no longer calls evaluate_generation_validity. "
        "That function is the ONLY path in the trainer that integrates the ODE "
        "(it calls generate_clouds_from_noise at :4683). Without it the script "
        "reconstructs real tmQM molecules and every panel metric computed from "
        "its CSV describes the bond head, not the generator."
    )


def test_bond_head_only_loop_is_gone():
    """The reconstruction loop must not survive anywhere in main()."""
    tree = ast.parse(_source())
    funcs = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }
    main = funcs.get("main")
    assert main is not None, "main() disappeared from the dump script"
    called = {
        node.func.attr
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    # featurise/_select_bonds/_assemble_mol are the reconstruction primitives.
    # _assemble_mol is also reachable INSIDE evaluate_generation_validity, but
    # here it must not be called directly from main().
    for banned in ("featurise", "_select_bonds", "_bridge_components", "_assemble_mol"):
        assert banned not in called, (
            f"main() still calls {banned} directly — that is the reconstruction "
            "path. Generation must go through evaluate_generation_validity."
        )


def test_generation_is_passed_the_real_decoder():
    """The call must supply `dec` — without the decoder it cannot decode bonds."""
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "evaluate_generation_validity"
        ):
            args = node.args
            assert len(args) >= 3, (
                "evaluate_generation_validity(model, decoder, templates, device) "
                f"-- got {len(args)} positional args"
            )
            return
    raise AssertionError("no evaluate_generation_validity call found")


def test_module_docstring_still_carries_the_provenance_warning():
    """The ODE-vs-reconstruction warning must not be quietly deleted."""
    src = _source()
    assert "NOT GENERATED" in src.upper(), (
        "the module docstring no longer says these are not generated molecules; "
        "that caveat is the only thing standing between this file and the next "
        "reader mislabelling reconstructions as de-novo output"
    )