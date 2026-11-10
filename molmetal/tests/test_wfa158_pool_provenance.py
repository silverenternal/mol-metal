"""Wf-A158 BRANCH POOL — tests for the pool provenance + round-trip audit.

CPU-only by construction (the Wf-A119/A124/A125/A156/A158 chain runs under a
standing "禁止使用gpu" directive).  Importing the module under test forces
``CUDA_VISIBLE_DEVICES=""``; the module never imports torch.

What is tested
--------------
1. The recomputation maths on SYNTHETIC SMILES — a single metal atom, a
   two-fragment molecule, and a molecule with an explicit dot — so the
   fragment-count and largest-fragment-fraction rules are pinned by
   construction, not by the pool.
2. The JSON schema of ``pool_provenance.json``.
3. The no-CUDA / no-torch import check.
4. That the script's OWN published numbers are reproducible from the pool
   CSV by an INDEPENDENT recomputation written here (the test does not
   call the script's helper for the headline numbers).
5. That the trainer line numbers the artifact cites still say what the
   artifact claims they say — so a drift upstream cannot make the
   provenance story silently wrong.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# The module sets CUDA_VISIBLE_DEVICES="" at import time.  Do it here too so
# the guarantee holds even if the import order changes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from molmetal.scripts import wfa158_pool_provenance as wfa  # noqa: E402

POOL_CSV = REPO / "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
ARTIFACT = (
    REPO / "molmetal/reports/wfa158_pool_provenance/pool_provenance.json"
)
TRAINER = REPO / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"

METAL_Z = frozenset({44, 45, 75, 76, 77, 78})  # Ru Rh Re Os Ir Pt


# --------------------------------------------------------------------------
# 1. Recomputation maths on synthetic SMILES
# --------------------------------------------------------------------------
def test_single_metal_atom_is_one_fragment_and_frac_one():
    """`[Pt]` — one atom, one fragment, the metal ALONE in it."""
    rec = wfa.roundtrip_one("[Pt]")
    assert rec["parsed"] is True
    assert rec["n_frags"] == 1
    assert rec["n_heavy"] == 1
    assert rec["largest_frag_frac"] == pytest.approx(1.0)
    assert rec["metal_alone"] is True
    assert rec["metal_shares_fragment"] is False
    assert rec["metal_survives_as_single_atom"] is True
    assert rec["metal_z"] == [78]


def test_two_fragment_molecule_counts_two_components():
    """`CCO.CCO` — two equal components, so the largest holds HALF the atoms.

    `largest_frag_frac` is (atoms in the largest component) / (atoms in
    the WHOLE molecule) — not a ratio between the components.  Two equal
    halves read 0.5, which is what the producing code computes.
    """
    rec = wfa.roundtrip_one("CCO.CCO")
    assert rec["parsed"] is True
    assert rec["n_frags"] == 2
    assert rec["frag_sizes"] == [3, 3]
    assert rec["n_heavy"] == 6
    assert rec["largest_frag_frac"] == pytest.approx(0.5)


def test_explicit_dot_metal_alone_frac_has_the_metal_in_the_denominator():
    """`[Pt].CCO` — the metal is ALONE, and the frac is 3/4 NOT 3/3.

    This is the column-semantics test: `largest_frag_frac` divides by the
    WHOLE molecule's heavy atoms, the metal included.  A reader who
    dropped the metal from the denominator would read 1.0 here.
    """
    rec = wfa.roundtrip_one("[Pt].CCO")
    assert rec["parsed"] is True
    assert rec["n_frags"] == 2
    assert rec["frag_sizes"] == [1, 3]
    assert rec["n_heavy"] == 4
    assert rec["largest_frag_frac"] == pytest.approx(0.75)
    assert rec["metal_alone"] is True
    assert rec["metal_survives_as_single_atom"] is True


def test_metal_sharing_a_fragment_is_not_alone():
    """`[Pt]C` — the metal shares its fragment with a non-metal atom."""
    rec = wfa.roundtrip_one("[Pt]C")
    assert rec["parsed"] is True
    assert rec["n_frags"] == 1
    assert rec["metal_alone"] is False
    assert rec["metal_shares_fragment"] is True


def test_unparseable_smiles_is_reported_not_guessed():
    """An unparseable string must come back `parsed=False`, never a count."""
    rec = wfa.roundtrip_one("C1CC")
    assert rec["parsed"] is False
    assert rec["n_frags"] is None
    assert rec["largest_frag_frac"] is None
    assert rec["canonical_smiles"] is None


def test_lfrac_tolerance_is_half_a_unit_in_the_last_published_place():
    """The writer rounds to 4 dp, so the equality bound is 5e-5."""
    assert wfa.ROUNDED_DP == 4
    assert wfa.LFRAC_TOL == pytest.approx(5e-5, abs=1e-11)
    # A value that differs by less than the tolerance is "equal" ...
    assert abs(0.9545 - 0.95450) <= wfa.LFRAC_TOL
    # ... and one that differs by more is not.
    assert abs(0.9545 - 0.9540) > wfa.LFRAC_TOL


# --------------------------------------------------------------------------
# 2. no-CUDA / no-torch import check
# --------------------------------------------------------------------------
def test_module_forces_cpu_only_and_never_imports_torch():
    """The script forces CUDA off and never imports torch itself.

    A pytest plugin may have loaded torch before this module was
    imported; that is the harness's doing, not the script's, so the
    assertion is on the module's own source, not on the whole process.
    """
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    src = Path(wfa.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src
    assert not hasattr(wfa, "torch")


def test_script_runs_end_to_end_and_exits_zero(tmp_path):
    """The script runs on this machine, in a CLEAN process, and exits 0.

    Run as a subprocess so the artifact's `torch_imported` flag is the
    script's own process and not the pytest session's.
    """
    out = tmp_path / "pool_provenance.json"
    proc = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            str(REPO / "molmetal/scripts/wfa158_pool_provenance.py"),
            "--out-json",
            str(out),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "PASS" in proc.stdout.splitlines()[-1]
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["verdict"] == "PASS"
    assert payload["cpu_only"]["cuda_visible_devices"] == ""
    assert payload["cpu_only"]["torch_imported"] is False
    assert payload["cpu_only"]["torch_imported_by_this_script"] is False
    assert payload["cpu_only"]["checkpoint_deserialised"] is False


def test_script_exits_one_on_a_missing_pool(tmp_path):
    proc = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            str(REPO / "molmetal/scripts/wfa158_pool_provenance.py"),
            "--pool-csv",
            str(tmp_path / "nope.csv"),
            "--out-json",
            str(tmp_path / "o.json"),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=300,
    )
    assert proc.returncode == 1
    assert "FAIL" in proc.stdout


# --------------------------------------------------------------------------
# 3. JSON schema
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT.is_file():
        pytest.skip(
            "pool_provenance.json not on disk; run "
            "molmetal/scripts/wfa158_pool_provenance.py first"
        )
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_json_schema(artifact):
    for key in (
        "schema_version",
        "generated_at_utc",
        "tool",
        "wall_time_s",
        "cpu_only",
        "inputs",
        "q1_provenance",
        "q2_roundtrip",
        "q3_column_semantics",
        "checks",
        "summary",
        "per_row_digest",
    ):
        assert key in artifact, key
    assert artifact["schema_version"] == "wfa158_pool_provenance.v1"
    assert artifact["summary"]["verdict"] in {"PASS", "FAIL", "SKIP"}
    assert artifact["summary"]["n_failed"] == 0
    assert isinstance(artifact["checks"], list) and artifact["checks"]
    for c in artifact["checks"]:
        assert c["status"] in {"PASS", "FAIL", "SKIP"}
        assert "name" in c and "detail" in c

    q2 = artifact["q2_roundtrip"]
    for key in (
        "n_rows",
        "n_parsed",
        "n_unparsed",
        "frag_count",
        "largest_frag_frac",
        "canonical_roundtrip",
        "metal",
        "postprocess_truncation_impact",
    ):
        assert key in q2, key
    assert q2["frag_count"]["n_equal"] + q2["frag_count"]["n_differ"] == q2["n_rows"]
    assert (
        q2["largest_frag_frac"]["n_equal"] + q2["largest_frag_frac"]["n_differ"]
        == q2["n_rows"]
    )


def test_artifact_provenance_claims_are_all_resolved(artifact):
    claims = artifact["q1_provenance"]["claims"]
    assert claims
    for c in claims:
        assert c["status"] in {"RESOLVED", "UNRESOLVED"}
        assert "question" in c and "value" in c and "evidence" in c
    assert artifact["q1_provenance"]["unresolved_claims"] == []


# --------------------------------------------------------------------------
# 4. The script's own numbers are reproducible from the CSV
# --------------------------------------------------------------------------
def test_artifact_numbers_reproducible_from_the_pool_csv(artifact):
    """Recompute the headline numbers INDEPENDENTLY of the script helper."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    if not POOL_CSV.is_file():
        pytest.skip(f"pool CSV not on disk: {POOL_CSV}")
    with POOL_CSV.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    n = 0
    parsed = 0
    frag_equal = 0
    lfrac_equal = 0
    metal_rows = 0
    metal_alone = 0
    metal_alone_survive = 0
    for r in rows:
        n += 1
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        parsed += 1
        frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        sizes = [len(f) for f in frags] or [0]
        n_heavy = int(mol.GetNumHeavyAtoms())
        lf = (max(sizes) / n_heavy) if n_heavy else 0.0
        if len(sizes) == int(r["n_frags_raw"]):
            frag_equal += 1
        if abs(lf - float(r["largest_frag_frac"])) <= 5e-5:
            lfrac_equal += 1
        metal_idx = {
            a.GetIdx() for a in mol.GetAtoms() if int(a.GetAtomicNum()) in METAL_Z
        }
        if metal_idx:
            metal_rows += 1
            frag_of = {}
            for f in frags:
                for i in f:
                    frag_of[i] = f
            shared = any(
                any((j not in metal_idx) for j in frag_of[mi]) for mi in metal_idx
            )
            if not shared:
                metal_alone += 1
                if any(len(frag_of[mi]) == 1 for mi in metal_idx):
                    metal_alone_survive += 1

    q2 = artifact["q2_roundtrip"]
    assert q2["n_rows"] == n
    assert q2["n_parsed"] == parsed
    assert q2["n_unparsed"] == n - parsed
    assert q2["frag_count"]["n_equal"] == frag_equal
    assert q2["largest_frag_frac"]["n_equal"] == lfrac_equal
    assert q2["metal"]["n_rows_with_metal"] == metal_rows
    assert q2["metal"]["n_metal_alone"] == metal_alone
    assert q2["metal"]["n_metal_alone_surviving_as_single_atom"] == metal_alone_survive
    assert len(artifact["per_row_digest"]) == n


def test_artifact_per_epoch_row_counts_match_the_csv(artifact):
    if not POOL_CSV.is_file():
        pytest.skip("pool CSV not on disk")
    with POOL_CSV.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    per_epoch = Counter(int(r["epoch"]) for r in rows)
    ev = {
        int(e["epoch"]): e["gen_validity_n"]
        for e in artifact["q1_provenance"]["events_summary"]
    }
    assert per_epoch == Counter(ev)


# --------------------------------------------------------------------------
# 5. The cited trainer lines still say what the artifact claims
# --------------------------------------------------------------------------
def test_cited_trainer_lines_still_say_what_the_artifact_claims(artifact):
    """A provenance story that drifts from the source is a false claim.

    The line numbers below are the ones the artifact cites; each is
    re-read from the LIVE trainer source and asserted to still contain
    the construct the artifact attributes to it.
    """
    src = TRAINER.read_text(encoding="utf-8").splitlines()

    def _blob(a, b):
        return "\n".join(src[a - 1 : b])

    # decode_chemistry: the RDKit parse (sanitising by default) and the
    # fragment count the two published columns come from.
    assert "Chem.MolFromSmiles(smiles)" in _blob(9671, 9671)
    assert "Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)" in _blob(
        9674, 9674
    )
    assert "sizes = [len(f) for f in frags]" in _blob(9675, 9675)
    assert "n_heavy = int(mol.GetNumHeavyAtoms())" in _blob(9676, 9676)
    # the two returned values the writer copies out
    assert '"n_fragments": len(sizes)' in _blob(9697, 9697)
    assert '"largest_fragment_frac": (max(sizes) / n_heavy)' in _blob(9703, 9703)
    # the writer block
    assert "generated_smiles.csv" in _blob(19216, 19216)
    assert "n_frags_raw" in _blob(19250, 19251)
    assert "round(_lfrac, 4)" in _blob(19310, 19310)
    assert "_chem = decode_chemistry(_s)" in _blob(19282, 19282)
    # the metal set the script greps out of the source
    assert wfa._METAL_LITERAL in "\n".join(src)


def test_the_pool_is_untracked_so_git_cannot_be_the_provenance_source():
    """The pool CSV is gitignored, so `git log` can never prove its origin."""
    proc = subprocess.run(
        ["git", "check-ignore", "-v", str(POOL_CSV.relative_to(REPO))],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, "pool CSV is no longer gitignored — provenance story changed"
    assert "molmetal/reports/**/*.csv" in proc.stdout
