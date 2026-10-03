"""Pytest: Wf-124 P0 #2 — domain-label mapping reconciliation.

Background
----------
Two files in the D-MPNN four-domain pipeline defined the domain→int
mapping differently:

* ``molmetal/scripts/pre_featurize_dmpnn.py:78-83`` (legacy) used
  ``0=metal_cytotox, 1=nci60, 2=binding_db, 3=drugood`` (a stub
  scratch order).
* ``molmetal/scripts/train_dmpnn_multitask.py:152`` (docstring)
  claimed ``0=MetalCytoToxDB 1=DrugOOD 2=NCI-60 3=BindingDB``.

The data-layer (``molmetal/data/four_domain_combined.py``) and the
D-MPNN baseline header both use the canonical mapping
(``DOMAIN_METALCYTOTOX=0 / DOMAIN_DRUGOOD=1 / DOMAIN_NCI60=2 /
DOMAIN_BINDINGDB=3``), so the train_dmpnn_multitask.py docstring was
correct and pre_featurize_dmpnn.py was the broken side.

This test pins both files to the canonical mapping so that any
future drift fails CI.

HONEST FRAMING (Wf-124 P0 #2, 2026-09-27)
------------------------------------------
The canonical mapping is::

    0 = MetalCytoToxDB
    1 = DrugOOD
    2 = NCI-60
    3 = BindingDB

That is what the D-MPNN baseline header, ``four_domain_combined.py``,
and the ``train_dmpnn_multitask.py --domain-id`` argparse help all
agree on. The pre_featurize_dmpnn.py stub was a stale scratch order
left over from a pre-TODO-143 refactor.

MIGRATION
---------
Pre-featurized .pt files written before this fix are mis-labelled
(metal_cytotox=0 is still right but nci60/binding_db/drugood have
swapped slots). To regenerate::

    rm -rf molmetal/data/pre_featurized/*.pt
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/pre_featurize_dmpnn.py

``test_pre_featurize_uses_canonical`` below documents the OLD labels
in those files explicitly so the next person doesn't have to reverse-
engineer what happened.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Canonical mapping — MUST match four_domain_combined.py and the
# train_dmpnn_multitask.py docstring.
CANONICAL_DOMAIN_ID = {
    "metal_cytotox": 0,   # MetalCytoToxDB
    "drugood":       1,   # DrugOOD
    "nci60":         2,   # NCI-60
    "binding_db":    3,   # BindingDB
}

PRE_FEATURIZE = PROJECT_ROOT / "molmetal" / "scripts" / "pre_featurize_dmpnn.py"
TRAIN_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask.py"


# ---------------------------------------------------------------------------
# 1) pre_featurize_dmpnn.py uses the canonical mapping
# ---------------------------------------------------------------------------
def test_pre_featurize_uses_canonical():
    """pre_featurize_dmpnn.py.DOMAIN_ID must match CANONICAL_DOMAIN_ID."""
    from molmetal.scripts import pre_featurize_dmpnn

    assert hasattr(pre_featurize_dmpnn, "DOMAIN_ID"), (
        "pre_featurize_dmpnn.py must expose DOMAIN_ID"
    )
    assert dict(pre_featurize_dmpnn.DOMAIN_ID) == CANONICAL_DOMAIN_ID, (
        f"pre_featurize_dmpnn.DOMAIN_ID drifted from canonical mapping.\n"
        f"  got:      {dict(pre_featurize_dmpnn.DOMAIN_ID)}\n"
        f"  expected: {CANONICAL_DOMAIN_ID}"
    )


# ---------------------------------------------------------------------------
# 2) Existing pre-featurized .pt files are labelled with the OLD mapping —
#    record this so the migration path is documented.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename,old_id,canonical_id,name",
    [
        ("metal_cytotox.pt", 0, 0, "MetalCytoToxDB"),
        ("nci60.pt",         1, 2, "NCI-60"),
        ("binding_db.pt",    2, 3, "BindingDB"),
        ("drugood.pt",       3, 1, "DrugOOD"),
    ],
)
def test_pre_featurize_existing_pt_files_use_old_mapping(
    filename, old_id, canonical_id, name
):
    """Document the OLD (stale) labels in pre-featurized .pt files.

    This test is an HONEST-FRAMING pin: it does NOT regenerate the
    files (that would be a separate workflow), it just asserts the
    current on-disk state so reviewers know the pre-fix world.

    If a contributor regenerates the .pt files with the canonical
    mapping, this test will start failing — that is the signal to
    update the migration notes (and the .pt files will then agree
    with ``CANONICAL_DOMAIN_ID``).
    """
    pt_path = PROJECT_ROOT / "molmetal" / "data" / "pre_featurized" / filename
    if not pt_path.exists():
        pytest.skip(f"{filename} not present — pre_featurize never ran")
    pack = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert "domain" in pack, f"{filename}: missing 'domain' tensor"
    uniq = pack["domain"].unique().tolist()
    assert uniq == [old_id], (
        f"{filename}: expected stale domain id {old_id} ({name}), "
        f"got {uniq}. If you just regenerated the .pt files with the "
        f"canonical mapping ({canonical_id}), update the migration "
        f"table in this test (and the MIGRATION_NOTES block in "
        f"pre_featurize_dmpnn.py)."
    )


# ---------------------------------------------------------------------------
# 3) train_dmpnn_multitask.py docstring claims canonical mapping
# ---------------------------------------------------------------------------
def test_train_script_docstring_matches():
    """The --domain-id argparse help in train_dmpnn_multitask.py must
    advertise the canonical mapping. We parse the AST and pull the
    help= literal directly so the assertion is decoupled from line
    shifts.
    """
    src = TRAIN_SCRIPT.read_text()
    tree = ast.parse(src)

    help_literals: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Look for keyword args named ``help=`` whose value is a
        # constant string.
        for kw in node.keywords:
            if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                help_literals.append(kw.value.value)

    # The --domain-id help line should mention the canonical ordering.
    domain_help = next(
        (h for h in help_literals if "MetalCytoToxDB" in h and "DrugOOD" in h),
        None,
    )
    assert domain_help is not None, (
        "Could not find --domain-id help text mentioning both "
        "MetalCytoToxDB and DrugOOD in train_dmpnn_multitask.py"
    )

    # Regex the canonical positions out of the help text.
    # We require the literal substring "0=MetalCytoToxDB 1=DrugOOD
    # 2=NCI-60 3=BindingDB" (whitespace tolerant).
    pattern = (
        r"0\s*=\s*MetalCytoToxDB"
        r"\s+1\s*=\s*DrugOOD"
        r"\s+2\s*=\s*NCI-?60"
        r"\s+3\s*=\s*BindingDB"
    )
    assert re.search(pattern, domain_help), (
        f"--domain-id help text does not encode the canonical mapping:\n"
        f"  {domain_help!r}"
    )


# ---------------------------------------------------------------------------
# 4) Both files cross-reference each other so future drift fails fast.
# ---------------------------------------------------------------------------
def test_pre_featurize_docstring_documents_canonical():
    """pre_featurize_dmpnn.py's module docstring (the schema block)
    must label the ``domain`` tensor with the canonical mapping.
    """
    src = PRE_FEATURIZE.read_text()
    # The schema docstring contains a ``"domain": Tensor ...`` block
    # listing each (id=name) pair. Pull that block out and confirm
    # the four canonical mappings are present in the right order.
    schema_block = re.search(
        r'"domain":.*?BindingDB\s*\)\s*"',
        src,
        flags=re.DOTALL,
    )
    assert schema_block is not None, (
        "Could not locate the 'domain' schema block in "
        "pre_featurize_dmpnn.py's module docstring."
    )
    block = schema_block.group(0)
    for needle in (
        "0=MetalCytoToxDB",
        "1=DrugOOD",
        "2=NCI-60",
        "3=BindingDB",
    ):
        assert needle in block, (
            f"Expected canonical mapping substring {needle!r} missing "
            f"from pre_featurize_dmpnn.py docstring schema block."
        )


def test_train_script_docstring_documents_canonical():
    """train_dmpnn_multitask.py's --domain-id section in the argparse
    description must explicitly mention all 4 canonical entries.
    """
    src = TRAIN_SCRIPT.read_text()
    for needle in (
        "0 = MetalCytoToxDB",
        "1 = DrugOOD",
        "2 = NCI-60",
        "3 = BindingDB",
    ):
        assert needle in src, (
            f"Expected canonical mapping {needle!r} in "
            f"train_dmpnn_multitask.py (--domain-id docstring section)."
        )
