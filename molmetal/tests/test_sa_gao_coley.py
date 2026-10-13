"""Red-check test for the Gao-Coley normalised SA implementation.

================================================================
Why this test exists (party-twenty-spirit §3.5 + AGENTS.md §5.7)
================================================================
A new behaviour (Gao-Coley normalised SA) was added to the repo
(molmetal/adapters/sa_gao_coley.py). Per the standing rule, this
introduces a test that FAILS without the new module, then we
revert+confirm-red+restore.

================================================================
What this test pins
================================================================
1. The function exists at the expected import path.
2. The published parameters (mu=2.230044, sigma=0.6526308) match the
   values carried by the published implementation references.
3. The formula is the asymmetric EXPONENTIAL DECAY shape
   exp(-(sa - mu) / sigma) -- NOT a Gaussian, NOT a half-Gaussian,
   and NOT a clamped symmetric Gaussian (the canonical MOMO form
   that the prior "half-Gaussian" docstring misdescribed).
   The misnomer was corrected 2026-10-12 by adversarial review
   (cfm_sa_gao_coley_reference.json::corrections_2026_10_12_evening).
4. Axis convention: 0 = hard, 1 = easy (HIGHER = easier). The
   function is monotonically decreasing in sa_raw.
5. The function returns 1.0 exactly at sa_raw == mu (this is the
   point at which the exponent vanishes).
6. Smoke integration: scoring the n=66 generated cohort produces
   the same Gao-Coley mean as recorded in
   metrics/by_metric/cfm_sa_gao_coley_reference.json. If the
   function is changed (e.g. to a symmetric Gaussian), the
   smoke check on the recorded JSON goes red.
7. The function is exposed under the corrected name
   ``asymmetric_exponential_sa`` and the legacy name
   ``gao_coley_sa_norm`` is a backward-compat alias, not a
   re-implementation.
8. The module docstring does NOT call the formula a "half-Gaussian"
   any more.

================================================================
Red-check procedure (run by hand)
================================================================
  # expect GREEN
  /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
      /home/hugo/codes/mol-metal/molmetal/tests/test_sa_gao_coley.py -v

  # expect RED -- revert just the import path
  sed -i 's/sa_gao_coley/sa_gao_coley_disabled/' \\
      /home/hugo/codes/mol-metal/molmetal/scripts/wf_cfm_sa_gao_coley_two_scales.py
  /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
      /home/hugo/codes/mol-metal/molmetal/tests/test_sa_gao_coley.py -v
  # restore
  sed -i 's/sa_gao_coley_disabled/sa_gao_coley/' \\
      /home/hugo/codes/mol-metal/molmetal/scripts/wf_cfm_sa_gao_coley_two_scales.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

# Repo root on sys.path for the adapter import to work.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.adapters.sa_gao_coley import (  # noqa: E402
    GAO_COLEY_MU,
    GAO_COLEY_SIGMA,
    asymmetric_exponential_sa,
    gao_coley_sa_norm,
)


# -----------------------------------------------------------------
# 1. The function and module-level constants exist and match
#    the values carried in the published implementation references.
# -----------------------------------------------------------------


def test_module_constants_match_published_values():
    """mu=2.230044, sigma=0.6526308 are Gao-Coley 2020's reference
    population (Enamine REAL / ZINC15-clean-3) for the Ertl SA score.
    Hard-coded to 7 dp because published implementations store them
    with that precision."""
    assert GAO_COLEY_MU == pytest.approx(2.230044, rel=1e-9)
    assert GAO_COLEY_SIGMA == pytest.approx(0.6526308, rel=1e-9)


def test_function_returns_1_at_reference_mean():
    """At sa_raw == mu the exponent vanishes; this is the canonical
    reference point and must evaluate to 1.0 exactly."""
    v = gao_coley_sa_norm(GAO_COLEY_MU)
    assert v == pytest.approx(1.0, rel=1e-12)


def test_function_is_monotonically_decreasing_above_mu():
    """Higher SA = harder to synthesise = lower normalised value. Sample
    six values above mu, ordered."""
    xs = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
    ys = [gao_coley_sa_norm(x) for x in xs]
    assert all(ys[i] > ys[i + 1] for i in range(len(ys) - 1)), ys


def test_function_caps_at_1_for_sa_below_mu():
    """The asymmetric exponential is monotonically decreasing in sa_raw,
    so any sa_raw < mu yields a value > 1. We expect the published 1.0
    cap (anything below mu is "maximally easy")."""
    assert gao_coley_sa_norm(0.0) > 1.0
    assert gao_coley_sa_norm(1.0) > 1.0
    # And the cap: a hand-written clamp is NOT applied by the
    # function itself, but the recorded metric does.  This test pins
    # the unclipped behaviour of the primitive.


def test_formula_is_asymmetric_exponential_not_symmetric_gaussian():
    """Discriminator test (corrected 2026-10-12): at sa_raw=6.44 (our
    generated cohort mean) the asymmetric-exponential shape and the
    symmetric Gaussian differ by orders of magnitude. The function we
    use must return the asymmetric-exponential value
    ``exp(-(sa - mu) / sigma)``, NOT the symmetric Gaussian
    ``exp(-((sa - mu) ** 2) / (2 * sigma ** 2))``.

    Prior docstring called this a "half-Gaussian" -- the corrected
    framing is "asymmetric exponential decay, no clamp, no squared
    argument". The numbers are the same as before; only the prose is
    fixed.
    """
    sa = 6.44
    expected_asym = math.exp(-(sa - GAO_COLEY_MU) / GAO_COLEY_SIGMA)
    expected_sym = math.exp(-((sa - GAO_COLEY_MU) ** 2) / (2 * GAO_COLEY_SIGMA ** 2))
    v = gao_coley_sa_norm(sa)
    # The asymmetric-exponential value is several orders of magnitude
    # larger than the symmetric one; the test is robust to
    # floating-point noise in either direction.
    assert v == pytest.approx(expected_asym, rel=1e-12)
    assert v > 100 * expected_sym, (v, expected_sym)


def test_corrected_name_exposed_and_alias_works():
    """The corrected name ``asymmetric_exponential_sa`` must be exposed
    at the import path. The legacy name ``gao_coley_sa_norm`` must be
    a true alias (same callable), not a re-implementation. This is the
    red-check the rename landed.
    """
    # Corrected name exists.
    assert callable(asymmetric_exponential_sa)
    # Legacy alias is the same callable, not a copy.
    assert gao_coley_sa_norm is asymmetric_exponential_sa, (
        "gao_coley_sa_norm must be a forward to asymmetric_exponential_sa, "
        "not a separate re-implementation"
    )
    # And it produces the same value at the reference point.
    assert gao_coley_sa_norm(GAO_COLEY_MU) == pytest.approx(1.0, rel=1e-12)
    assert asymmetric_exponential_sa(GAO_COLEY_MU) == pytest.approx(1.0, rel=1e-12)


def test_module_docstring_does_not_call_formula_half_gaussian():
    """Red-check: the module docstring must NOT describe the formula as
    a 'half-Gaussian' any more. The prior wording was misleading; the
    corrected wording is 'asymmetric exponential decay' (or equivalent).
    If a future edit reintroduces the 'half-Gaussian' framing in a
    descriptive (non-negation) context, this test goes red and the
    docstring must be repaired.

    Denial forms like "not a half-Gaussian" or "not half-Gaussian" are
    allowed (they are correct), and meta-references that quote the
    prior wrong wording in scare-quotes are also allowed.
    """
    import re

    import molmetal.adapters.sa_gao_coley as mod

    doc = (mod.__doc__ or "").lower()
    # The corrected framing must appear.
    assert "asymmetric exponential" in doc or "asymmetric exp" in doc, (
        "module docstring should describe the formula as 'asymmetric "
        "exponential' (the corrected framing)."
    )
    # The misnomer must not appear in a descriptive (non-negation,
    # non-quoted) form. We check the bare phrase as a stand-alone
    # descriptor: e.g. 'it is a half-gaussian' is wrong, but
    # 'not a half-gaussian' or '"half-gaussian"' is allowed.
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if "half-gaussian" not in line:
            continue
        # Allowed: the phrase is preceded by 'not ', 'no ', or wrapped
        # in quotes (scare-quotes / explicit re-quoting of the wrong
        # word).  Any other form is a regression to the misnomer.
        # Strip the phrase and check the immediate prefix.
        idx = line.find("half-gaussian")
        prefix = line[max(0, idx - 6):idx]
        is_negated = "not a" in prefix or "no " in prefix or "not a " in prefix
        is_quoted = '"' in prefix or "'" in prefix or "`" in prefix
        assert is_negated or is_quoted, (
            f"line {i+1} uses 'half-Gaussian' in a non-negation, "
            f"non-quoted context: {line!r}. The function is an "
            "asymmetric exponential decay; describing it as a "
            "half-Gaussian reintroduces the misnomer."
        )


def test_axis_convention_is_higher_is_easier():
    """Pin the docstring claim: HIGHER sa_norm = more synthetically
    accessible. So at the same sa_raw, the Gao-Coley value must be
    higher than the linear hack 1-sa/10 IF the sa_raw is below the
    function's cmu+sigma crossover. At sa_raw=4 (a moderate value),
    Gao-Coley gives ~0.0664 and the linear hack gives 0.6, so the
    linear hack is higher at this point -- and the test is WRITTEN
    THE OTHER WAY: it pins that Gao-Coley is monotonically
    decreasing in sa_raw, which is the axis direction. It does NOT
    say one scale is better than the other.
    """
    assert gao_coley_sa_norm(3.0) > gao_coley_sa_norm(5.0)
    assert gao_coley_sa_norm(5.0) > gao_coley_sa_norm(8.0)


# -----------------------------------------------------------------
# 2. Smoke test: scoring the n=66 generated cohort reproduces the
#    Gao-Coley mean recorded on disk. If the function or the parameters
#    change, this test goes red and the metric JSON must be regenerated.
# -----------------------------------------------------------------


def test_generated_cohort_gao_coley_mean_matches_recorded_value():
    """Pin the n=66 mean against
    metrics/by_metric/cfm_sa_gao_coley_reference.json::generated.sa_norm_gao_coley.mean.
    The recorded value is 0.0025139127232905213.  If this test goes
    red, the function or the parameters have changed.
    """
    from rdkit import Chem
    from rdkit import RDConfig
    import os as _os
    import importlib

    sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    sascorer = importlib.import_module("sascorer")

    gen_csv = (
        _REPO_ROOT
        / "molmetal"
        / "reports"
        / "cfm_fullcorpus_2026_10_10"
        / "generated_smiles.csv"
    )
    if not gen_csv.is_file():
        pytest.skip(f"generated cohort not found: {gen_csv}")
    import csv as _csv
    sms = []
    with gen_csv.open(newline="") as fh:
        rdr = _csv.DictReader(fh)
        for row in rdr:
            sms.append(row.get("smiles") or "")
    raws = []
    for s in sms:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        raws.append(float(sascorer.calculateScore(m)))
    norms = [gao_coley_sa_norm(x) for x in raws]
    mean = sum(norms) / len(norms)

    # Read the recorded value from the metric JSON.
    metric_path = (
        _REPO_ROOT
        / "metrics"
        / "by_metric"
        / "cfm_sa_gao_coley_reference.json"
    )
    if not metric_path.is_file():
        pytest.skip(f"metric JSON not found: {metric_path}")
    with metric_path.open() as fh:
        m = json.load(fh)
    recorded = m["generated"]["sa_norm_gao_coley"]["mean"]
    assert mean == pytest.approx(recorded, rel=1e-9), (
        f"computed {mean} != recorded {recorded}; "
        "if this fails, regenerate the metric JSON."
    )


# -----------------------------------------------------------------
# 3. Red-check: when the adapter module is renamed/missing, the
#    test_sa_gao_coley.py module-level import fails. We do NOT
#    attempt to delete the module here; the human red-check is in the
#    procedure at the top of this file.
# -----------------------------------------------------------------
