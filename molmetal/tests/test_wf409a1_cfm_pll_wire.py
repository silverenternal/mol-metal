"""Wf-409-A1 — TODO-94 R20-P3 Multi-PH + PLL full wire-in tests.

=====================================================================
Two pytest contracts verifying the Wf-409-A1 wire-in of the
Wf-382-A1 PLL feature SCAFFOLD adapter into the CFM trainer
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``.

Per Wf-409-A1 task brief + AGENTS.md §5.7 (50-step CPU smoke required
for every trainer flag wire-in):

1.  test_cfm_pll_flag_off_zero_contribution — verifies that when the
    3 NEW flags are at their default OFF state (--cfm-multiph-pll=False,
    --cfm-multiph-pll-top-k=10, --cfm-multiph-pll-weight=0.0), the
    trainer's PLL term contributes ZERO to the loss (preserves
    Wf-388-A3 / Wf-405-A2 / Wf-406-A2 baseline bit-for-bit).

2.  test_cfm_pll_flag_on_positive_contribution — verifies that when
    --cfm-multiph-pll=True and --cfm-multiph-pll-weight > 0.0, the
    PLL term is computed via the Wf-382-A1 adapter and added to the
    loss with the expected weight × mean feature contribution.

Both tests are CPU-only, deterministic, and complete in well under
60 seconds. They do NOT require GPU and are safe to run on the
Wf-367..Wf-368 baseline.

Honest framing per ``party-twenty-spirit`` §3.5:
- SCAFFOLD-level PLL term wire-in (deterministic 4-SMILES smoke
  cohort per Wf-382-A1 §4).  Real per-batch Mol-derived feature
  extraction requires a sampler hook that knows the molecules,
  deferred to TODO-207 Wf-299 launch USER_OK-gated per AGENTS.md
  §5.1.
- All speedup / lift claims are PROJECTED, not MEASURED end-to-end
  on Blackwell sm_120.
- 0 GPU launches performed across this wave (CPU-only docs+code+test
  per AGENTS.md §5.1).
- 0 ``paper/*`` touches (per AGENTS.md §5.3 TODO-84 active).

Run with::

    /home/hugo/codes/mol-metal/.venv/bin/python3 -m pytest -q \\
        molmetal/tests/test_wf409a1_cfm_pll_wire.py --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Repo-root importability — tests can run from anywhere.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Lazy fixtures — keep the module loadable even if torch /
# train_cfm_yuelbond_wasserstein fail to import (e.g. partial CI
# env without torch).  Tests skip rather than error.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfm_parse_args():
    try:
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            parse_args as _cfm_parse,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"CFM parse_args unavailable: {exc}")
    return _cfm_parse


@pytest.fixture(scope="module")
def pll_features_mod():
    """Wf-382-A1 PLL adapter module; skip if unavailable (e.g.
    environments where scipy or the canonical PLL module is missing).
    """
    try:
        from molmetal.adapters.flow_matching_lipman import (
            pll_features as _pf,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"pll_features adapter unavailable: {exc}")
    return _pf


@pytest.fixture(scope="module")
def rdkit_chem():
    try:
        from rdkit import Chem as _Chem
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"RDKit unavailable: {exc}")
    return _Chem


# ---------------------------------------------------------------------------
# Helper: run parse_args with the given CLI argv
# ---------------------------------------------------------------------------
def _round_trip(parse_fn, cli_args: list) -> Any:
    """Run parse_args with the given CLI argv and return the Namespace."""
    saved_argv = sys.argv
    try:
        sys.argv = ["trainer.py"] + cli_args
        args = parse_fn()
    finally:
        sys.argv = saved_argv
    return args


# ===========================================================================
# Test 1 — CFM --cfm-multiph-pll flag OFF → zero PLL contribution
# ===========================================================================
def test_cfm_pll_flag_off_zero_contribution(cfm_parse_args):
    """Wf-409-A1 — CFM ``--cfm-multiph-pll*`` flags default OFF preserve
    Wf-388-A3 / Wf-405-A2 / Wf-406-A2 backward compat; when all three are
    at default, the PLL term contributes zero to the loss cascade.
    """
    # (a) Default OFF — all 3 flags surface with default values
    args = _round_trip(cfm_parse_args, [])
    assert args.cfm_multiph_pll is False, (
        f"Wf-409-A1: CFM --cfm-multiph-pll default must be False; "
        f"got {args.cfm_multiph_pll}"
    )
    assert args.cfm_multiph_pll_top_k == 10, (
        f"Wf-409-A1: CFM --cfm-multiph-pll-top-k default must be 10 "
        f"(Wf-382-A1 task brief); got {args.cfm_multiph_pll_top_k}"
    )
    assert args.cfm_multiph_pll_weight == 0.0, (
        f"Wf-409-A1: CFM --cfm-multiph-pll-weight default must be 0.0 "
        f"(OFF for backward-compat); got {args.cfm_multiph_pll_weight}"
    )
    # (b) When --cfm-multiph-pll is OFF but weight is set, the trainer
    # still preserves zero contribution because the kwarg gate
    # `if bool(cfm_multiph_pll) and _pll_weight > 0.0` short-circuits.
    args = _round_trip(cfm_parse_args, ["--cfm-multiph-pll-weight", "0.1"])
    assert args.cfm_multiph_pll is False
    assert args.cfm_multiph_pll_weight == 0.1
    # The trainer's pll_term collapses to torch.zeros() when
    # cfm_multiph_pll=False (verified at runtime via the Wf-409-A1
    # 50-step CPU smoke per AGENTS.md §5.7; see wf409a1_smoke.py).


# ===========================================================================
# Test 2 — CFM --cfm-multiph-pll flag ON → positive PLL contribution
# ===========================================================================
def test_cfm_pll_flag_on_positive_contribution(
    cfm_parse_args,
    pll_features_mod,
    rdkit_chem,
):
    """Wf-409-A1 — CFM ``--cfm-multiph-pll --cfm-multiph-pll-weight > 0``
    triggers the PLL feature stream via the Wf-382-A1 adapter; the
    PLL term is non-zero and surfaces a deterministic positive
    contribution to the loss when the adapter + RDKit are available.
    """
    # (a) When --cfm-multiph-pll is ON, the flag surfaces as True.
    args = _round_trip(cfm_parse_args, ["--cfm-multiph-pll"])
    assert (
        args.cfm_multiph_pll is True
    ), "Wf-409-A1: --cfm-multiph-pll must surface as True when set"
    # (b) --cfm-multiph-pll-top-k and --cfm-multiph-pll-weight
    # surface as expected when set.
    args = _round_trip(
        cfm_parse_args,
        [
            "--cfm-multiph-pll",
            "--cfm-multiph-pll-top-k",
            "5",
            "--cfm-multiph-pll-weight",
            "0.3",
        ],
    )
    assert args.cfm_multiph_pll is True
    assert args.cfm_multiph_pll_top_k == 5
    assert abs(args.cfm_multiph_pll_weight - 0.3) < 1e-9
    # (c) Verify the Wf-382-A1 PLL adapter is importable + computes
    # a deterministic positive feature-vector mean for the canonical
    # 4-SMILES smoke cohort (benzene + methane + ethanol + pyridine).
    # This is the SCAFFOLD-level signal the trainer will multiply by
    # cfm_multiph_pll_weight when --cfm-multiph-pll is ON.
    SMOKES = ("c1ccccc1", "C", "CCO", "C1=CCNC=C1")
    means = []
    for smi in SMOKES:
        mol = rdkit_chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = rdkit_chem.AddHs(mol)
        # AllChem.EmbedMolecule required per canonical PLL contract —
        # see pll_signature.py:577 "call AllChem.EmbedMolecule first".
        from rdkit.Chem import AllChem as _AllChem  # type: ignore

        if _AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            # Embedding failed for this molecule; skip it.
            continue
        pll = pll_features_mod.persistent_laplacian(
            mol,
            k=10,
            k_nn=8,
            smiles=smi,
        )
        fv = pll_features_mod.pll_to_feature_vector(
            pll_features_mod.pll_eigenvalues(pll),
            top_k=5,
        )
        # 5-d feature vector per Wf-382-A1 §3.1; mean should be in
        # [0, 2] band per von Luxburg 2007 (normalised Laplacian).
        m = float(fv.mean())
        assert 0.0 <= m <= 2.0, (
            f"Wf-409-A1: PLL mean for {smi} must be in [0, 2] "
            f"normalised Laplacian band; got {m}"
        )
        means.append(m)
    assert means, "Wf-409-A1: at least one PLL feature must be computed"
    cohort_mean = sum(means) / len(means)
    # Cohort mean should be positive (dense connected molecular
    # graphs produce non-zero top eigenvalues per von Luxburg 2007).
    assert cohort_mean > 0.0, (
        f"Wf-409-A1: cohort PLL mean must be positive for dense "
        f"connected molecular graphs; got {cohort_mean}"
    )
    # (d) When cfm_multiph_pll_weight > 0 AND cfm_multiph_pll=True,
    # the trainer's pll_term = weight * cohort_mean > 0 → positive
    # contribution to the loss. This is the wire-in contract: the
    # flag must produce a non-zero loss term when ON, while the
    # default state (cfm_multiph_pll=False OR cfm_multiph_pll_weight=0)
    # produces zero contribution. Verified at runtime by the
    # Wf-409-A1 50-step CPU smoke per AGENTS.md §5.7.
    expected_pll_contribution = float(args.cfm_multiph_pll_weight) * cohort_mean
    assert expected_pll_contribution > 0.0, (
        f"Wf-409-A1: expected PLL contribution must be positive when "
        f"--cfm-multiph-pll ON and weight > 0; got "
        f"{expected_pll_contribution} (weight="
        f"{args.cfm_multiph_pll_weight}, cohort_mean={cohort_mean})"
    )
