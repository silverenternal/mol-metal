"""Wf-A203 -- the citability caveat for the two headline-grade rows that had none.

MEASURED census (Wf-A203, 2026-11-02, `registry/metrics.yaml`): of the 11
`registry_role: paper_line` rows, EXACTLY TWO were `quotable: true` AND
`headline_grade: true` AND carried NO `headline_caveat` -- `m19_lomo` and
`m21_nci60_9cell`.  `registry.py cite` printed a bare number for both, so a
reader got 0.5575 and 0.7294 with none of the disclosures the payloads
themselves already carry (the retraction log for m19, the test-set-size funnel
for m21).  This wave adds one `headline_caveat` key to each PAYLOAD (the
durable place -- `registry.py generate` rebuilds each row from its payload, so
a registry-only caveat would be erased on the next regenerate) and re-points
the two registry rows at the new bytes.

These tests are the durable guard on that repair:

  T1  `registry.py cite` prints a `caveat:` line for BOTH metrics, carrying the
      numbers that make the caveat load-bearing (m21: the two filtered panel
      means and the two single-digit-test-set cells; m19: the resolvable-subset
      figure that supersedes the one `cite` prints);
  T2  each payload's MEASURED VALUE is unchanged -- the edit is disclosure
      only, so a changed value is a REGRESSION, not an improvement;
  T3  each registry row's `source_json_sha256` equals the sha256 of its payload
      on disk (otherwise `registry.py check` calls the row STALE);
  T4  neither caveat contains a forbidden claim word;
  T5  `registry.py check` reports NO STALE/MISSING marker for either metric;
  T6  the ledger's LATEST row for each metric names the CURRENT on-disk sha256
      and the unchanged value;
  T7  the registry row's caveat is byte-identical to the payload's, and
      `assess_citability` -- the function `registry.py generate` calls to build
      the row -- reproduces it from the payload alone (the durability proof:
      a regenerate would NOT erase the caveat);
  T8  the ledger row appended by this wave does not fabricate a commit
      equality: the bytes on disk are in NO commit, and the row says so.

CPU-only.  No GPU.  No network.  No `os.walk` (the repo is 201 GB).

ENVIRONMENT NOTE (MEASURED 2026-11-02): the repo-wide Wf-A123 session guard
(`molmetal/tests/conftest.py::_published_tree_guard`) compares EVERY published
metric artifact's sha256 at session start against session teardown, and raises
a teardown ERROR if any of them moved -- regardless of which test moved it.
While the Wf-A155 wave's `wf_a155_posebusters_validity.py` is running in this
working tree it rewrites `metrics/by_metric/wfa155_metal_coordination_sphere.json`
(its own file; it alternates between two serializations), so a session that
straddles one of its writes trips that guard.  THIS MODULE writes nothing under
`metrics/`: `registry.py check` and `registry.py cite` leave that file's sha256
AND mtime unchanged (MEASURED), and the unrelated pre-existing
`molmetal/tests/test_wfa188_registry_ledger_coverage.py` fails on the SAME two
`wfa155_*` rows for the same reason.  The 27 assertions below pass in every
session; only the guard's teardown error is environment-dependent.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REGISTRY = REPO / "registry" / "metrics.yaml"
LEDGER = REPO / "molmetal" / "reports" / "run_ledger.jsonl"
REGISTRY_CLI = REPO / "molmetal" / "scripts" / "registry.py"

#: metric_id -> the metric payload that carries its `headline_caveat`.
PAYLOADS = {
    "m21_nci60_9cell": "metrics/by_metric/nci60_precious_9cell_auc.json",
    "m19_lomo": "metrics/by_metric/metal_lomo_auc_wasserstein.json",
}

#: The MEASURED values, hard-coded from the payloads on disk (Wf-A203,
#: 2026-11-02).  A caveat edit must not move them.
EXPECTED_VALUE = {
    "m21_nci60_9cell": 0.7293802091886383,
    "m19_lomo": 0.5575471269535148,
}

#: The numbers each caveat MUST carry for it to be load-bearing.
#: m21 -- the two filtered panel means + the comparator + the two cells whose
#:        single-digit test sets carry the headline.
#: m19 -- the resolvable-subset figure that supersedes the retracted one, its
#:        seed-ensemble robustness, and the real-only fold value.
REQUIRED_TOKENS = {
    "m21_nci60_9cell": [
        "0.6521",   # panel mean over 7 cells (n_test>=10)
        "0.5923",   # panel mean over 5 cells (n_test>=20)
        "0.81",     # the Krasnov 2026 comparator both fall below
        "DU145",    # the n_test=7 cell scoring AUC = 1.0
        "SKOV3",    # the n_test=3 cell scoring AUC = 1.0
        "AUC = 1.0",
        "n_test=7",
        "n_test=3",
    ],
    "m19_lomo": [
        "0.5974",   # the quotable resolvable-subset figure
        "0.5947",   # seed-ensemble robustness
        "0.3812",   # real_only_fold_value (the Rh fold)
    ],
}

#: FORBIDDEN claim words -- explicit, so the guard is auditable.  A caveat is
#: a disclosure; it must never promote the number into a win claim.
FORBIDDEN_CLAIM_WORDS = [
    "beat",
    "beats",
    "beating",
    "outperform",
    "outperforms",
    "outperforming",
    "state-of-the-art",
    "state of the art",
    "sota",
    "best-in-class",
    "best performing",
    "best-performing",
    "significant improvement",
    "significantly improves",
    "superior performance",
    "best",
]

#: MEASURED 2026-11-02 on this working tree, BEFORE Wf-A203 wrote anything:
#: `registry.py check` reported "579 current, 2 stale, 0 missing" with exactly
#: these two rows STALE -- their metric JSONs are owned by the Wf-A155 wave,
#: which Wf-A203 must NOT touch.  The task's literal "registry.py check exits
#: 0" is therefore NOT satisfiable on this tree without editing another wave's
#: files, so this test pins the STALE SET instead: a NEW stale row fails the
#: test, and the exit code is asserted to agree with that set.
KNOWN_PREEXISTING_STALE = {
    "wfa155_ligand_connected_rate",
    "wfa155_metal_coordination_sphere",
}

#: The sha256 each payload had BEFORE the caveat was added (the bytes that
#: ARE committed at HEAD).  Used by the honesty guard on the ledger row.
PRE_CAVEAT_SHA = {
    "m21_nci60_9cell": "a2bdc1ad6c514ac7d0b5e36cc24106ca4a1d0348581b59cf1a943bec3abf8ce1",
    "m19_lomo": "90a9c0c095605fbe20e49a2dfb830a6a890b961b9204330d6cc234bdb6d4aa74",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _run_registry(cmd: str) -> subprocess.CompletedProcess:
    """Run `registry.py <cmd>` as a subprocess, 0 GPU, from the repo root."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, str(REGISTRY_CLI), cmd],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


_BLOCK_RE = re.compile(r"^\[(CITE|NUM |NO  )\]\s+(\S+)")


def _cite_blocks(text: str) -> dict:
    """Parse `registry.py cite` output into {metric_id: {mark, caveat}}."""
    blocks: dict = {}
    current = None
    for line in text.splitlines():
        m = _BLOCK_RE.match(line)
        if m:
            current = m.group(2)
            blocks[current] = {"mark": m.group(1).strip(), "caveat": None}
            continue
        if current is None:
            continue
        if not line.strip():
            current = None
            continue
        if line.startswith("        caveat: "):
            blocks[current]["caveat"] = line[len("        caveat: ") :].strip()
    return blocks


def _registry_rows() -> dict:
    doc = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    return doc.get("metrics") or {}


def _payload(rel: str) -> dict:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def _ledger_entries() -> list:
    return [
        json.loads(line)
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_blob_sha(commit: str, rel: str):
    """sha256 of the blob `<commit>:<rel>`, or None if it is not in that commit."""
    proc = subprocess.run(
        ["git", "cat-file", "-p", f"{commit}:{rel}"],
        cwd=str(REPO),
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout).hexdigest()


# ---------------------------------------------------------------------------
# T1 -- `registry.py cite` prints the caveat for both
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cite_blocks() -> dict:
    proc = _run_registry("cite")
    assert proc.returncode == 0, proc.stderr
    return _cite_blocks(proc.stdout)


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_cite_prints_a_caveat_line(metric_id, cite_blocks):
    assert metric_id in cite_blocks, f"{metric_id} absent from `registry.py cite`"
    block = cite_blocks[metric_id]
    assert block["caveat"], f"{metric_id}: cite printed no caveat line"
    assert block["mark"] == "CITE", f"{metric_id}: expected the CITE mark, got {block['mark']!r}"


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_cite_caveat_carries_the_required_numbers(metric_id, cite_blocks):
    caveat = cite_blocks[metric_id]["caveat"]
    for token in REQUIRED_TOKENS[metric_id]:
        assert token in caveat, f"{metric_id}: caveat is missing the token {token!r}"


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_cite_caveat_is_byte_identical_to_the_payload_caveat(metric_id, cite_blocks):
    payload = _payload(PAYLOADS[metric_id])
    assert cite_blocks[metric_id]["caveat"] == payload["headline_caveat"]


# ---------------------------------------------------------------------------
# T2 -- the MEASURED values did not move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_payload_value_unchanged(metric_id):
    payload = _payload(PAYLOADS[metric_id])
    # m21's registry value_basis is `value_overall_mean_auc`; m19's is `value`.
    observed = payload.get("value_overall_mean_auc", payload.get("value"))
    assert observed == EXPECTED_VALUE[metric_id]


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_registry_value_unchanged(metric_id):
    row = _registry_rows()[metric_id]
    assert row["value"] == EXPECTED_VALUE[metric_id]


# ---------------------------------------------------------------------------
# T3 -- the registry row points at the payload on disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_registry_sha_matches_payload_on_disk(metric_id):
    row = _registry_rows()[metric_id]
    live = _sha256(REPO / PAYLOADS[metric_id])
    assert row["source_json_sha256"] == live
    assert row["source_json"] == PAYLOADS[metric_id]


# ---------------------------------------------------------------------------
# T4 -- no forbidden claim words
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_caveat_has_no_forbidden_claim_words(metric_id):
    caveat = _payload(PAYLOADS[metric_id])["headline_caveat"]
    for phrase in FORBIDDEN_CLAIM_WORDS:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])"
        hit = re.search(pattern, caveat, flags=re.IGNORECASE)
        assert hit is None, (
            f"{metric_id}: caveat contains the forbidden claim word {phrase!r} "
            f"(at {hit.start()!r})"
        )


# ---------------------------------------------------------------------------
# T5 -- `registry.py check` is clean for the two metrics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def check_proc() -> subprocess.CompletedProcess:
    return _run_registry("check")


def test_check_has_no_stale_or_missing_marker_for_the_two_metrics(check_proc):
    out = check_proc.stdout + check_proc.stderr
    for metric_id in PAYLOADS:
        assert f"STALE {metric_id}" not in out, f"{metric_id} is STALE"
        assert f"MISSING {metric_id}" not in out, f"{metric_id} is MISSING"


def test_check_missing_count_is_zero(check_proc):
    out = check_proc.stdout + check_proc.stderr
    assert re.search(r"missing", out), "check output did not report a missing count"
    assert re.search(r"\b0 missing\b", out), "check reported a missing metric"


def test_check_stale_set_is_only_the_preexisting_out_of_scope_rows(check_proc):
    out = check_proc.stdout + check_proc.stderr
    stale_ids = set(re.findall(r"^\s*STALE (\S+):", out, flags=re.M))
    unexpected = stale_ids - KNOWN_PREEXISTING_STALE
    assert not unexpected, f"unexpected STALE rows: {sorted(unexpected)}"
    # The exit code must agree with the STALE set (registry.py:2269-2270).
    expected_rc = 0 if not stale_ids else 1
    assert check_proc.returncode == expected_rc


# ---------------------------------------------------------------------------
# T6 -- the ledger's LATEST row names the current sha and the unchanged value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_ledger_latest_row_names_current_sha_and_value(metric_id):
    rows = [e for e in _ledger_entries() if e.get("metric_id") == metric_id]
    assert rows, f"{metric_id}: no ledger row"
    latest = rows[-1]
    assert latest["source_json"] == PAYLOADS[metric_id]
    assert latest["source_json_sha256"] == _sha256(REPO / PAYLOADS[metric_id])
    assert latest["value"] == EXPECTED_VALUE[metric_id]


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_ledger_rows_are_append_only_and_the_prior_row_survives(metric_id):
    rows = [e for e in _ledger_entries() if e.get("metric_id") == metric_id]
    assert len(rows) >= 2, f"{metric_id}: the prior ledger row was removed"
    # the Wf-A203 row is the last one
    assert rows[-1]["extra"]["wf"] == "Wf-A203"


# ---------------------------------------------------------------------------
# T7 -- the caveat is durable: `registry.py generate` would reproduce it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_registry_row_caveat_is_byte_identical_to_the_payload(metric_id):
    row = _registry_rows()[metric_id]
    payload = _payload(PAYLOADS[metric_id])
    assert row["headline_caveat"] == payload["headline_caveat"]


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_generate_would_reproduce_the_caveat_from_the_payload(metric_id):
    """`assess_citability` is what `build_metrics` calls to build the row
    (registry.py:1426-1429) and what `generate` dumps (registry.py:2072).
    If the caveat survives this call, a regenerate keeps it."""
    from molmetal.scripts.registry import assess_citability

    payload = _payload(PAYLOADS[metric_id])
    observed = assess_citability(
        payload, EXPECTED_VALUE[metric_id], PAYLOADS[metric_id], None
    )
    assert observed["headline_caveat"] == payload["headline_caveat"]
    assert observed["quotable"] is True
    assert observed["headline_grade"] is True


# ---------------------------------------------------------------------------
# T8 -- the ledger row does not fabricate a commit equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", sorted(PAYLOADS))
def test_wfa203_ledger_row_is_honest_about_the_uncommitted_bytes(metric_id):
    rows = [e for e in _ledger_entries() if e.get("metric_id") == metric_id]
    wfa = [e for e in rows if (e.get("extra") or {}).get("wf") == "Wf-A203"]
    assert wfa, f"{metric_id}: no Wf-A203 ledger row"
    row = wfa[-1]
    extra = row["extra"]

    assert extra["committed"] is False
    assert extra["commit_blob_equals_disk"] is False
    assert extra["value_unchanged"] is True
    assert extra["correction_of_ledger_sha256"] == PRE_CAVEAT_SHA[metric_id]

    # The named commit must hold the PRIOR bytes -- never the current ones.
    disk_sha = _sha256(REPO / PAYLOADS[metric_id])
    blob_sha = _git_blob_sha(row["git_sha"], PAYLOADS[metric_id])
    assert blob_sha == PRE_CAVEAT_SHA[metric_id], (
        f"{metric_id}: the commit named by the ledger row does not hold the prior bytes"
    )
    assert blob_sha != disk_sha, (
        f"{metric_id}: the row's commit equality is fabricated"
    )
    assert extra["committed_sha256"] == PRE_CAVEAT_SHA[metric_id]
    assert extra["disk_sha256"] == disk_sha
    # The reason text must say the bytes are not committed.
    assert "NOT COMMITTED" in extra["reason"]
    assert "METADATA ONLY" in extra["reason"]
    assert "UNCHANGED" in extra["reason"]
