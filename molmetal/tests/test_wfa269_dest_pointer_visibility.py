# Licensed under the Apache License, Version 2.0.
"""Wf-A269 (2026-11-09) — the ``dest:`` pointer blind spot is CLOSED.

THE DEFECT
----------
``molmetal/scripts/wfA2_config_loader.py::_iter_leaves`` skips any mapping
whose EVERY child key is in ``_NON_TRAINER_SUBKEYS`` — and ``dest`` IS one of
those keys — so a schema-metadata block is skipped WHOLESALE, the ``dest:``
pointer inside it included, and the §5.11 binding gate never saw it.

A ``dest:`` pointer is categorically different from ``type`` / ``default`` /
``desc``: it is a BINDING CLAIM — a machine-readable ``script.py::dest``
reference to an argparse dest.  Before this repair the gate could not fail on
one: a config could name a script that does not exist, or a dest its parser
never declares, and every §5.11 gate still said PASS while its closing line
claimed that every config key reaches a dest of its own ``entry.script``
parser.

MEASURED 2026-11-09 over the live config population: schema-metadata blocks
number **5** and **all 5** carry a ``dest:`` child ⇒ the skip's entire
observable effect was to hide exactly those pointers, and **5 of 5** are FALSE
(4 name a dest absent from the named script's parser, 1 names a script that
does not exist anywhere in the repo).

THE REPAIR
----------
Mechanism: ``wfA2_config_loader.iter_dest_pointers`` — a SEPARATE scanner, so
the loader's own dest-resolution path (``_iter_leaves``) is untouched and a
``dest:`` string is never delivered to an argparse dest.

Disposition: ``scripts/_config_binding_gate.DECLARED_INERT_DEST_POINTERS`` —
one layer down from ``DECLARED_SKIPPED_CONFIGS``, keyed
``"<config file name>::<block path>"``, with a required reason.  A pointer
that resolves is OK; a pointer that does not resolve but is DECLARED is
reported every run and is not fatal; a pointer that does not resolve and is
NOT declared is a VIOLATION, always.

WHAT IS UNDER TEST
------------------
* the loader still SKIPS the schema block as a trainer leaf (the Wf-A57 W9-P2
  behaviour is preserved — the repair is additive);
* the loader surfaces every schema-metadata ``dest:`` pointer;
* a pointer that resolves ⇒ OK, not a violation;
* an UNDECLARED pointer ⇒ a violation ⇒ a FAIL verdict;
* a DECLARED pointer ⇒ reported with its reason, not fatal;
* removing the declaration flips the verdict (the registry is load-bearing);
* the live population has zero undeclared pointers;
* TWO MUTATION PROOFS — the loader copy and the gate copy are each mutated in
  ``tmp_path`` and the check is shown to go BLIND.

CPU-only.  No config is written into the repository: every fixture is a
``tmp_path`` file, and the gate is audited over an explicit config list.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_config_binding_gate.py"
LOADER_PATH = REPO_ROOT / "molmetal" / "scripts" / "wfA2_config_loader.py"

#: The one shipped config that carries schema-metadata blocks (MEASURED
#: 2026-11-09: 5 blocks, all 5 carrying a ``dest:`` child).
LIVE_CONFIG = REPO_ROOT / "configs" / "cfm_patience10_seed0_2026_10_12.yaml"

#: A script that exists in the repo and whose parser is capturable.
LIVE_SCRIPT = "molmetal/scripts/bench_cfm_decode_real.py"

_spec = importlib.util.spec_from_file_location("_config_binding_gate_ptr", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["_config_binding_gate_ptr"] = gate
_spec.loader.exec_module(gate)


def _pointer_config(tmp_path: Path, name: str, pointer: str) -> Path:
    """A minimal config carrying exactly ONE schema-metadata ``dest:`` pointer.

    The block's children are all in ``_NON_TRAINER_SUBKEYS``, so the loader
    treats the block as a leaf — which is precisely the shape the blind spot
    lived in.
    """
    p = tmp_path / name
    p.write_text(
        yaml.safe_dump(
            {
                "entry": {"script": LIVE_SCRIPT},
                "some_block": {
                    "type": "bool",
                    "default": False,
                    "dest": pointer,
                    "desc": "a binding claim",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return p


def _load_module(path: Path, name: str):
    """Import a module from an explicit path (used for the /tmp mutants)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the loader
def test_the_loader_still_skips_a_schema_block_as_a_trainer_leaf() -> None:
    """The Wf-A57 W9-P2 behaviour must survive the repair.

    The repair is ADDITIVE: ``_iter_leaves`` still treats a schema-metadata
    block as a leaf, so no ``type`` / ``default`` / ``desc`` string is ever
    delivered to an argparse dest.  If this test fails, the fix leaked the
    pointer into the loader's dest-resolution path.
    """
    data = yaml.safe_load(LIVE_CONFIG.read_text(encoding="utf-8"))
    leaves = list(gate.L._iter_leaves(data))
    offending = [p for p, _ in leaves if p and p[0] == "strict_large_frag"]
    assert offending == [], (
        "the schema block is no longer skipped as a leaf — the repair changed "
        f"the loader's dest-resolution path: {offending}"
    )


def test_the_loader_surfaces_every_schema_block_dest_pointer() -> None:
    """MEASURED 2026-11-09: 5 schema blocks, all 5 carrying a ``dest:``."""
    data = yaml.safe_load(LIVE_CONFIG.read_text(encoding="utf-8"))
    pointers = dict(
        (".".join(block), pointer)
        for block, pointer in gate.L.iter_dest_pointers(data)
    )
    assert pointers == {
        "strict_large_frag": "bench_cfm_decode_real.py::strict_large_frag",
        "largest_frag_min_frac": "bench_cfm_decode_real.py::largest_frag_min_frac",
        "min_atoms_per_mol": "bench_cfm_decode_real.py::min_atoms_per_mol",
        "emit_admet8": "cfm_property_panel.py::emit_admet8",
        "paper_table_cohort": "bench_cfm_paper_table.py::cohort_id",
    }, f"the surfaced pointer set changed: {pointers}"


# ----------------------------------------------------------------- the gate
def test_a_pointer_that_resolves_is_ok_not_a_violation(tmp_path: Path) -> None:
    """A true binding claim must not be reported as a violation."""
    cfg = _pointer_config(tmp_path, "ok_pointer.yaml", f"{LIVE_SCRIPT.split('/')[-1]}::out_dir")
    report = gate.audit(configs=[cfg])
    assert report["n_dest_pointers"] == 1
    assert report["n_dest_pointer_violations"] == 0, report["violations"]
    assert report["dest_pointer_records"][0]["state"] == "OK"
    assert report["verdict"] == "PASS", report["violations"]


def test_an_undeclared_pointer_is_a_violation_and_the_gate_fails(
    tmp_path: Path,
) -> None:
    """MUTATION: the check must be able to FAIL on a false binding claim."""
    cfg = _pointer_config(
        tmp_path, "undeclared.yaml", "this_script_does_not_exist_at_all.py::no_such_dest"
    )
    report = gate.audit(configs=[cfg])
    assert report["verdict"] == "FAIL", (
        "a dest: pointer that names a script which does not exist still "
        "produced a PASS verdict"
    )
    pointer_violations = [v for v in report["violations"] if v.get("reason") == "dest_pointer"]
    assert pointer_violations, f"no dest_pointer violation was raised: {report['violations']}"
    assert pointer_violations[0]["key"] == "some_block"
    assert "does NOT exist" in pointer_violations[0]["detail"]


def test_an_undeclared_pointer_into_an_existing_script_is_also_a_violation(
    tmp_path: Path,
) -> None:
    """A script that EXISTS but does not declare the dest is a false claim too."""
    cfg = _pointer_config(
        tmp_path, "wrong_dest.yaml", "bench_cfm_decode_real.py::not_a_dest_either"
    )
    report = gate.audit(configs=[cfg])
    assert report["verdict"] == "FAIL", report["violations"]
    pointer_violations = [v for v in report["violations"] if v.get("reason") == "dest_pointer"]
    assert pointer_violations, report["violations"]
    assert "is NOT a dest of" in pointer_violations[0]["detail"]


def test_a_declared_pointer_is_reported_with_its_reason_and_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DECLARED pointer is disclosed every run, and is not fatal."""
    cfg = _pointer_config(
        tmp_path, "declared.yaml", "this_script_does_not_exist_at_all.py::no_such_dest"
    )
    monkeypatch.setitem(
        gate.DECLARED_INERT_DEST_POINTERS,
        "declared.yaml::some_block",
        "a deliberately false binding claim, kept as the record of an intent",
    )
    report = gate.audit(configs=[cfg])
    assert report["verdict"] == "PASS", report["violations"]
    assert report["n_dest_pointer_violations"] == 0
    assert report["n_declared_inert_dest_pointers"] == 1
    record = report["dest_pointer_records"][0]
    assert record["state"] == "DECLARED-INERT"
    assert "record of an intent" in record["declared_reason"]


def test_removing_a_declaration_flips_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION: the declaration is load-bearing.

    The same config, the same fixture, the same gate — only the declaration
    differs.  If the declaration did nothing, this test could not exist.
    """
    cfg = _pointer_config(
        tmp_path, "flip.yaml", "this_script_does_not_exist_at_all.py::no_such_dest"
    )
    monkeypatch.setitem(gate.DECLARED_INERT_DEST_POINTERS, "flip.yaml::some_block", "declared")
    assert gate.audit(configs=[cfg])["verdict"] == "PASS"

    monkeypatch.delitem(gate.DECLARED_INERT_DEST_POINTERS, "flip.yaml::some_block")
    assert gate.audit(configs=[cfg])["verdict"] == "FAIL", (
        "removing the declaration did not change the verdict — the "
        "DECLARED_INERT_DEST_POINTERS registry is not load-bearing"
    )


def test_the_live_population_has_no_undeclared_pointer() -> None:
    """The live repo audit must report every pointer and violate none.

    MEASURED 2026-11-09: 5 pointers, 5 declared-inert, 0 violations.
    """
    report = gate.audit()
    assert report["n_dest_pointers"] == 5, (
        f"the live population's pointer count moved: {report['n_dest_pointers']}"
    )
    assert report["n_dest_pointer_violations"] == 0, report["violations"]
    assert report["n_declared_inert_dest_pointers"] == 5
    assert report["verdict"] == "PASS", report["violations"]
    # Every pointer the gate saw must be adjudicated — none may be unrecorded.
    assert len(report["dest_pointer_records"]) == report["n_dest_pointers"]


# ------------------------------------------------------------ mutation proofs
def test_mutating_the_loader_copy_to_suppress_the_pointer_tier_blinds_the_gate(
    tmp_path: Path,
) -> None:
    """MUTATION PROOF (the loader side).

    ``molmetal/scripts/wfA2_config_loader.py`` is copied to ``tmp_path`` and
    the COPY's ``iter_dest_pointers`` is made to yield nothing (the single
    ``yield path, pointer`` line becomes ``continue``).  The gate is then
    pointed at the mutated loader and MUST go blind: the false pointer it
    used to fail on is no longer seen.  Nothing in the repo is touched.
    """
    mutant = tmp_path / "wfA2_config_loader_mutant.py"
    src = LOADER_PATH.read_text(encoding="utf-8")
    target = None
    for line in src.splitlines(keepends=True):
        if line.strip() == "yield path, pointer":
            target = line
            break
    assert target is not None, "the loader's pointer yield line moved — update the mutation"
    mutated = src.replace(
        target,
        target.replace(
            "yield path, pointer",
            "continue  # MUTANT: the pointer is suppressed",
        ),
        1,
    )
    assert mutated != src
    mutant.write_text(mutated, encoding="utf-8")

    mutant_mod = _load_module(mutant, "_wfa269_loader_mutant")
    original_loader = gate.L
    gate.L = mutant_mod
    try:
        cfg = _pointer_config(
            tmp_path, "blind_loader.yaml", "this_script_does_not_exist_at_all.py::no_such_dest"
        )
        report = gate.audit(configs=[cfg])
    finally:
        gate.L = original_loader

    assert report["n_dest_pointers"] == 0, (
        "the mutated loader still surfaced a pointer — the mutation did not "
        f"neutralise the scanner: {report['dest_pointer_records']}"
    )
    assert report["verdict"] == "PASS", (
        "the gate did not go blind under the mutated loader — the pointer tier "
        "does not actually depend on the loader's scanner"
    )


def test_mutating_the_gate_copy_to_drop_the_pointer_tier_blinds_the_gate(
    tmp_path: Path,
) -> None:
    """MUTATION PROOF (the gate side).

    ``scripts/_config_binding_gate.py`` is copied to ``tmp_path`` and the
    COPY's pointer tier is replaced by an empty loop (the
    ``for rec in _audit_dest_pointers(...)`` line becomes ``for rec in []``).
    The copy's ``REPO_ROOT`` is re-pointed at the real repo root (computed at
    run time, never hard-coded) so the copy still resolves real scripts.  The
    mutated copy MUST go blind on the same false pointer.
    """
    mutant = tmp_path / "_config_binding_gate_mutant.py"
    src = GATE_PATH.read_text(encoding="utf-8")
    needle = "        for rec in _audit_dest_pointers(cfg, data, script_path, pointer_cache):"
    assert needle in src, "the gate's pointer tier call moved — update the mutation"
    mutated = src.replace(
        needle,
        "        for rec in []:  # MUTANT: the pointer tier is disabled",
        1,
    )
    mutated = mutated.replace(
        "REPO_ROOT = Path(__file__).resolve().parent.parent",
        f"REPO_ROOT = Path({str(REPO_ROOT)!r})",
        1,
    )
    assert mutated != src
    mutant.write_text(mutated, encoding="utf-8")

    mutant_gate = _load_module(mutant, "_wfa269_gate_mutant")
    cfg = _pointer_config(
        tmp_path, "blind_gate.yaml", "this_script_does_not_exist_at_all.py::no_such_dest"
    )
    report = mutant_gate.audit(configs=[cfg])

    assert report["n_dest_pointers"] == 0, (
        "the mutated gate still counted a pointer — the mutation did not "
        f"disable the tier: {report['dest_pointer_records']}"
    )
    assert report["verdict"] == "PASS", (
        "the mutated gate did not go blind — the pointer tier is not wired "
        "into the verdict"
    )
