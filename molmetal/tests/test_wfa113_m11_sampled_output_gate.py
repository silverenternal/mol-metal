"""Wf-A113 / T32 — M11, the gate that reads the SAMPLED OUTPUT.

The other ten module gates read the model.  MEASURED 2026-10-25, four of them
passed on ``molmetal/reports/cfm_c8_fullgen_2026_10_24`` while that run's pool
averaged 2.5 fragments per molecule and put 84% of its rings at size 3 (real:
0.49%).  M11 closes that hole: it reads a decoded pool and compares three
ratios against a band measured on real chemistry.

The contracts below are the gate's own kill conditions:

* it FAILS on the shipped c8 pool (so it is not another always-green gate);
* it PASSES on real molecules (so it is not just "generated = bad");
* it FAILS when no pool is supplied (a gate that skips is not a gate);
* it FAILS when the pool and the run's own epoch record disagree;
* the reference band is READ from artifacts, not typed in.

CPU-only.  The c8 and tmQM fixtures skip when the data is absent rather than
passing vacuously.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
C8_POOL = REPO / "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
TMQM_Y = REPO / ".data/molmetal/tmQM/tmQM_y.csv"


@pytest.fixture(scope="module")
def gates():
    from molmetal.scripts import cfm_module_gates as _g

    return _g


def _write_pool(path: Path, smiles: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "smiles", "n_frags_raw", "largest_frag_frac"])
        for s in smiles:
            w.writerow([0, s, 1, 1.0])
    return path


def _real_smiles(n: int = 200) -> list[str]:
    """The first ``n`` non-empty SMILES from the real tmQM corpus."""
    raw = TMQM_Y.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(raw.splitlines(), delimiter=";")
    out: list[str] = []
    for row in reader:
        s = (row.get("SMILES") or "").strip()
        if s:
            out.append(s)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# 1. The gate can fail — on the pool that motivated it
# ---------------------------------------------------------------------------
def test_m11_fails_on_the_c8_pool(gates):
    if not C8_POOL.is_file():
        pytest.skip(f"{C8_POOL} not on disk")
    r = gates.gate_m11({"pool_csv": str(C8_POOL)})
    assert r["pass"] is False
    assert r["checks"]["single_fragment_rate"] is False
    assert r["checks"]["ring_3_share"] is False
    # The two checks that are NOT the discriminator must still pass, or the
    # FAIL could be an artefact of a broken reference instead of the pool.
    assert r["checks"]["parse_rate"] is True
    m = r["measured"]
    assert m["single_fragment_rate"] < 0.5 * m["reference"]["single_fragment_rate"]
    assert m["ring_3_share"] > 2.0 * m["reference"]["ring_3_share"]


# ---------------------------------------------------------------------------
# 2. The gate can pass — on real molecules
# ---------------------------------------------------------------------------
def test_m11_passes_on_real_tmQM(gates):
    if not TMQM_Y.is_file():
        pytest.skip(f"{TMQM_Y} not on disk")
    smiles = _real_smiles(200)
    assert len(smiles) >= 100, "the real control needs a real sample"
    tmp = Path("/tmp") / "wfa113_m11_real_pool.csv"
    _write_pool(tmp, smiles)
    r = gates.gate_m11({"pool_csv": str(tmp)})
    assert r["pass"] is True, r["detail"]
    assert all(r["checks"].values())


# ---------------------------------------------------------------------------
# 3. No pool is a FAIL, not a SKIP
# ---------------------------------------------------------------------------
def test_m11_without_a_pool_fails(gates):
    r = gates.gate_m11({"pool_csv": None})
    assert r["pass"] is False
    assert r["checks"] == {"pool_supplied": False}


def test_m11_missing_file_fails(gates):
    r = gates.gate_m11({"pool_csv": "/tmp/definitely_not_a_pool_9137.csv"})
    assert r["pass"] is False
    assert r["checks"] == {"pool_exists": False}


def test_m11_rejects_a_csv_without_a_smiles_column(gates):
    tmp = Path("/tmp") / "wfa113_m11_no_smiles.csv"
    tmp.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'smiles' column"):
        gates.gate_m11({"pool_csv": str(tmp)})


# ---------------------------------------------------------------------------
# 4. The reference band is read, not typed
# ---------------------------------------------------------------------------
def test_m11_reference_comes_from_artifacts(gates):
    ref = gates._m11_reference()
    assert ref["sources"]["ring_content"].endswith("cfm_opt2_ring_content.json")
    assert ref["sources"]["panel_band"].endswith(
        "cfm_panel_reference_band_real_tmqm.json"
    )
    # 41 three-membered rings of 8386 in the artifact's real block.
    assert ref["ring_3_share_n_rings"] == 8386
    assert abs(ref["ring_3_share"] - 41 / 8386) < 1e-12
    assert ref["single_fragment_n"] > 0


# ---------------------------------------------------------------------------
# 5. The pool and the run artifact must describe the same decode
# ---------------------------------------------------------------------------
def test_m11_cross_check_rejects_a_mismatched_artifact(gates):
    if not TMQM_Y.is_file():
        pytest.skip(f"{TMQM_Y} not on disk")
    tmp = Path("/tmp") / "wfa113_m11_real_pool2.csv"
    _write_pool(tmp, _real_smiles(200))
    ep = Path("/tmp") / "wfa113_m11_epoch_bad.json"
    ep.write_text(
        json.dumps({"gen_ring_3_share": 0.75, "gen_single_fragment_ratio": 0.20}),
        encoding="utf-8",
    )
    r = gates.gate_m11({"pool_csv": str(tmp), "epoch_json": str(ep)})
    assert r["checks"]["artifact_matches_pool"] is False
    assert r["pass"] is False

    # And the same artifact, corrected to what the pool actually contains,
    # must not fail the cross-check.
    ep.write_text(
        json.dumps(
            {
                "gen_ring_3_share": r["measured"]["ring_3_share"],
                "gen_single_fragment_ratio": r["measured"]["single_fragment_rate"],
            }
        ),
        encoding="utf-8",
    )
    r2 = gates.gate_m11({"pool_csv": str(tmp), "epoch_json": str(ep)})
    assert r2["checks"]["artifact_matches_pool"] is True
    assert r2["pass"] is True


# ---------------------------------------------------------------------------
# 6. M11 runs with no checkpoint at all
# ---------------------------------------------------------------------------
def test_m11_is_registered_and_runs_without_a_checkpoint(gates, tmp_path):
    if not TMQM_Y.is_file():
        pytest.skip(f"{TMQM_Y} not on disk")
    assert "M11" in gates._GATES
    assert "M11" in gates._GATE_IDS
    pool = _write_pool(tmp_path / "pool.csv", _real_smiles(200))
    out = tmp_path / "m11.json"
    rc = gates.main(
        ["--gate", "M11", "--pool-csv", str(pool), "--json", str(out)]
    )
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["checkpoint"] is None
    assert doc["gates"][0]["id"] == "M11"
    assert doc["gates"][0]["status"] == "PASS"
