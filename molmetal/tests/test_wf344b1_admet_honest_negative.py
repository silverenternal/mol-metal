"""Tests for Wf-344-B1 — ADMET HONEST_NEGATIVE field regression suite.

=====================================================================
Wf-344-B1 / Wf-338-A3 finding 6-9 / Wf-325E sister closeout — verifies
that the 4 BELOW druglikeness ADMET metric JSONs all carry the
canonical ``HONEST_NEGATIVE: true`` + ``honest_negative_reason`` +
``wf_tag`` + ``wf_design_ref`` field set so the Wf-308B / Wf-312C R02
contract stays enforced (downstream tooling: paper_dashboard,
pre_flight, wf325e_admet_4_below_documented aggregated gate).

Test surface (4 NEW tests):
  * 1 test: aqueous_solubility_logS JSON carries canonical fields
  * 1 test: hepatotox_index JSON carries canonical fields
  * 1 test: logp7_4 JSON carries canonical fields
  * 1 test: m6_kappa JSON carries canonical fields

Honest framing (party-twenty-spirit §3.5 + AGENTS.md §1):
  * Per pre-flight v3 fresh-run 2026-09-29 all 4 gates ALREADY carry
    HONEST_NEGATIVE=true (this is a VERIFICATION closeout, not a fix).
  * The 4 tests below are regression-protection: they lock in the
    current correct state so future metric JSON regenerations cannot
    silently drop the HONEST_NEGATIVE field set without breaking a
    test (per Wf-308B / Wf-312C R02 contract).
  * ``Wf-325E_closeout`` is referenced for cross-check; the 4 gates
    are: p3_admet_aq_sol_honest_negative_tag,
    p3_admet_hepatotox_honest_negative,
    p3_admet_logp7_4_honest_negative,
    p3_admet_m6_kappa_honest_negative.

Standing-rule compliance:
  * AGENTS.md §1 MEASURED-vs-PROJECTED: 4/4 fields verified on-disk
    via metrics/by_metric/*.json
  * AGENTS.md §5.1 kill_then_adapt: ZERO GPU launches; CPU-only tests
  * AGENTS.md §5.2 honest framing: VERIFICATION-ONLY closeout
  * AGENTS.md §5.4 Tier-2 strategy: pure computational verification
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path


# Repository root = two parents up from molmetal/tests/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Canonical 4 BELOW druglikeness ADMET metrics per Wf-308B / Wf-312C R02.
# Each carries the HONEST_NEGATIVE field set so downstream tooling
# (pre_flight, paper_dashboard, p3_admet_4_below_documented) branches
# on a single canonical field rather than re-deriving the FAIL /
# THRESHOLD comparison at every read site.
_ADMET_METRICS = (
    "aqueous_solubility_logS",  # Wf-303C T1; gate p3_admet_aq_sol_honest_negative_tag
    "hepatotox_index",           # Wf-312C R02; gate p3_admet_hepatotox_honest_negative
    "logp7_4",                   # Wf-312C R02; gate p3_admet_logp7_4_honest_negative
    "m6_kappa",                  # Wf-312C R02 + Wf-325D; gate p3_admet_m6_kappa_honest_negative
)

# Required fields per Wf-308B / Wf-312C R02 contract. ``HONEST_NEGATIVE``
# must be exactly ``True`` (boolean) and ``honest_negative_reason`` must
# be a non-empty string; ``wf_tag`` + ``wf_design_ref`` carry provenance
# so the pre-flight gate can emit an evidence line that cites the source.
_REQUIRED_FIELDS = (
    "HONEST_NEGATIVE",
    "honest_negative_reason",
    "wf_tag",
    "wf_design_ref",
)


class _AdmetHonestNegativeMixin:
    """Shared assertions for each of the 4 BELOW ADMET metrics.

    Each metric's test class inherits this and sets ``METRIC_NAME`` +
    ``GATE_NAME`` to the appropriate values, then calls ``_assert_*``
    helpers from its own single test method.
    """

    METRIC_NAME: str = ""
    GATE_NAME: str = ""

    def _json_path(self) -> Path:
        return _REPO_ROOT / "metrics" / "by_metric" / f"{self.METRIC_NAME}.json"

    def _assert_canonical_honest_negative(self) -> None:
        path = self._json_path()
        # 1. file must exist on disk (MEASURED-vs-PROJECTED §1)
        self.assertTrue(
            path.exists(),
            f"Metric JSON missing on disk: {path} — AGENTS.md §1 MEASURED-vs-PROJECTED violated",
        )
        # 2. JSON parseable
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # 3. metric key matches the JSON filename (defensive — Wf-308B
        #    uses ``metric`` field as the canonical lookup)
        self.assertEqual(
            data.get("metric"),
            self.METRIC_NAME,
            f"Metric JSON {path} carries 'metric'={data.get('metric')!r}, "
            f"expected {self.METRIC_NAME!r}",
        )
        # 4. All 4 required fields present + correctly typed
        for field_name in _REQUIRED_FIELDS:
            self.assertIn(
                field_name,
                data,
                f"Required field {field_name!r} missing from {path}",
            )
        # 5. HONEST_NEGATIVE must be exactly True (boolean — NOT truthy)
        self.assertIs(
            data["HONEST_NEGATIVE"],
            True,
            f"HONEST_NEGATIVE field is not exactly True in {path}: "
            f"got {data['HONEST_NEGATIVE']!r}",
        )
        # 6. honest_negative_reason must be non-empty string
        reason = data["honest_negative_reason"]
        self.assertIsInstance(
            reason,
            str,
            f"honest_negative_reason must be string in {path}: got {type(reason).__name__}",
        )
        self.assertGreater(
            len(reason.strip()),
            0,
            f"honest_negative_reason must be non-empty in {path}",
        )
        # 7. wf_tag + wf_design_ref must be non-empty strings (provenance)
        for prov_field in ("wf_tag", "wf_design_ref"):
            value = data[prov_field]
            self.assertIsInstance(
                value,
                str,
                f"{prov_field} must be string in {path}: got {type(value).__name__}",
            )
            self.assertGreater(
                len(value.strip()),
                0,
                f"{prov_field} must be non-empty in {path}",
            )


class TestAqueousSolubilityLogS(_AdmetHonestNegativeMixin, unittest.TestCase):
    """Wf-303C T1 / R03 — aqueous_solubility_logS BELOW -4.0 threshold."""

    METRIC_NAME = "aqueous_solubility_logS"
    GATE_NAME = "p3_admet_aq_sol_honest_negative_tag"

    def test_honest_negative_field_set_present(self) -> None:
        """aqueous_solubility_logS.json carries canonical HONEST_NEGATIVE: true."""
        self._assert_canonical_honest_negative()


class TestHepatotoxIndex(_AdmetHonestNegativeMixin, unittest.TestCase):
    """Wf-312C R02 — hepatotox_index above 0.5 Hughes 2008 threshold."""

    METRIC_NAME = "hepatotox_index"
    GATE_NAME = "p3_admet_hepatotox_honest_negative"

    def test_honest_negative_field_set_present(self) -> None:
        """hepatotox_index.json carries canonical HONEST_NEGATIVE: true."""
        self._assert_canonical_honest_negative()


class TestLogp74(_AdmetHonestNegativeMixin, unittest.TestCase):
    """Wf-312C R02 — logp7_4 OUT OF Lipinski 0-5 ceiling."""

    METRIC_NAME = "logp7_4"
    GATE_NAME = "p3_admet_logp7_4_honest_negative"

    def test_honest_negative_field_set_present(self) -> None:
        """logp7_4.json carries canonical HONEST_NEGATIVE: true."""
        self._assert_canonical_honest_negative()


class TestM6Kappa(_AdmetHonestNegativeMixin, unittest.TestCase):
    """Wf-312C R02 + Wf-325D — m6_kappa BELOW 0.6 Shape Kappa threshold."""

    METRIC_NAME = "m6_kappa"
    GATE_NAME = "p3_admet_m6_kappa_honest_negative"

    def test_honest_negative_field_set_present(self) -> None:
        """m6_kappa.json carries canonical HONEST_NEGATIVE: true."""
        self._assert_canonical_honest_negative()


if __name__ == "__main__":
    unittest.main()
