"""Tests for molmetal.scripts.paper_auto_update — Wf-265 paper §4 auto-update.

Covers the public surface of ``paper_auto_update.py``:

1. test_module_imports                          — module + key symbols exist.
2. test_cli_help_exits_zero                     — ``--help`` exits 0 (sanity).
3. test_cli_flag_count                          — argparser exposes >= 4 flags.
4. test_dry_run_default_true                    — default for --dry-run is True.
5. test_atomic_write_text_round_trip            — _atomic_write_text writes + overwrites.
6. test_build_diff_extracts_overall_and_permetal — collect JSON -> [overall, 5 metals].
7. test_parse_old_value_finds_ru_909            — PATTERNS locate Ru 0.909 in §2.1.
8. test_parse_old_value_returns_none_for_missing — unmatched claim returns (None, "").
9. test_render_diff_md_contains_old_and_new     — diff_report.md has both columns per row.
10. test_render_updated_doc_substitutes_values  — atomic apply replaces old -> new.
11. test_render_updated_doc_appends_banner      — atomic apply appends the Wf-265 banner.
12. test_end_to_end_dry_run                     — main() in dry-run writes only the diff.
13. test_end_to_end_apply_atomic                — main() with --no-dry-run writes both files.
14. test_pattern_catalogue_is_structural        — every pattern captures the number it
                                                  finds; mutating the digits still matches.
15. test_m21_patterns_track_the_live_doc        — 0.8007 (legacy) and 0.7294 (live) both parse.
16. test_pattern_miss_is_reported               — a miss reaches stderr, not the void.
17. test_strict_exit_code_on_pattern_miss       — --strict turns a miss into rc=3.
18. test_apply_preserves_markup                 — only the number is swapped, not the
                                                  surrounding ``**`` / table pipes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from molmetal.scripts import paper_auto_update as pau


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_collect(
    overall_mean: float = 0.7209,
    per_metal: dict | None = None,
) -> dict:
    """Build a minimal Wf-265-Collect-shaped fixture dict."""
    if per_metal is None:
        per_metal = {
            "Ru": {"n_pairs": 3, "mean_test_auc": 0.7483},
            "Ir": {"n_pairs": 3, "mean_test_auc": 0.6702},
            "Os": {"n_pairs": 3, "mean_test_auc": 0.7025},
            "Rh": {"n_pairs": 3, "mean_test_auc": 0.7227},
            "Re": {"n_pairs": 3, "mean_test_auc": 0.7700},
        }
    return {
        "g4_overall_aggregate": {
            "n_pairs": 14,
            "mean_test_auc": overall_mean,
            "min_test_auc": 0.6659,
            "max_test_auc": 0.7970,
        },
        "g4_per_metal_aggregate": per_metal,
    }


PAPER_DOC_TEMPLATE = """# Paper §4 draft

## §2.1 Per-metal AUC

| Metal | TODO-154 chemprop v2 | Krasnov | Δ | Verdict |
|---|---:|---:|---:|---|
| **Ru** | **0.909** | 0.81 | +0.099 | STRONG |
| **Ir** | **0.885** | 0.73 | +0.155 | STRONG |
| **Os** | **0.889** | n/a | n/a | STRONG |
| **Rh** | **0.863** | n/a | n/a | STRONG |
| **Re** | **0.882** | n/a | n/a | STRONG |
| **5-metal mean** | **0.886 ± 0.014** | 0.81 | +0.076 | STRONG |

End of doc.
"""


# ---------------------------------------------------------------------------
# 1) Module + key symbols exist
# ---------------------------------------------------------------------------
def test_module_imports():
    assert pau is not None
    for name in (
        "main",
        "build_diff",
        "render_diff_md",
        "render_updated_doc",
        "_atomic_write_text",
        "_extract_claims",
        "_parse_old_value",
        "PATTERNS",
        "CLAIM_LABELS",
        "DEFAULT_PAPER_DOC",
        "DEFAULT_COLLECT_JSON",
        "DEFAULT_DIFF_OUT",
        "DEFAULT_UPDATED_OUT",
    ):
        assert hasattr(pau, name), f"missing public symbol: {name}"


# ---------------------------------------------------------------------------
# 2) --help exits 0 (sanity for the CLI surface)
# ---------------------------------------------------------------------------
def test_cli_help_exits_zero():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.path.insert(0, '.'); "
            "from molmetal.scripts import paper_auto_update as m; "
            "sys.exit(m._build_argparser().parse_args(['--help']))",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    # argparse exits with code 0 after printing --help
    assert proc.returncode == 0, proc.stderr
    assert "Wf-265" in proc.stdout or "paper" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# 3) argparser exposes >= 4 flags
# ---------------------------------------------------------------------------
def test_cli_flag_count():
    parser = pau._build_argparser()
    # Each add_argument() registers one flag; check >= 4 to satisfy G2.
    assert len(parser._actions) >= 5, (
        f"expected >=5 argparse actions (4 flags + help), got "
        f"{len(parser._actions)}: {[a.dest for a in parser._actions]}"
    )


# ---------------------------------------------------------------------------
# 4) Default for --dry-run is True (backward-compat gate G3)
# ---------------------------------------------------------------------------
def test_dry_run_default_true():
    parser = pau._build_argparser()
    args = parser.parse_args(
        [
            "--collect-json",
            "/tmp/x.json",
            "--paper-doc",
            "/tmp/y.md",
        ]
    )
    assert args.dry_run is True, f"dry_run should default to True; got {args.dry_run}"


# ---------------------------------------------------------------------------
# 5) _atomic_write_text writes + overwrites an existing file
# ---------------------------------------------------------------------------
def test_atomic_write_text_round_trip(tmp_path: Path):
    out = tmp_path / "x.md"
    pau._atomic_write_text(out, "hello world\n")
    assert out.read_text(encoding="utf-8") == "hello world\n"
    pau._atomic_write_text(out, "replacement\n")
    assert out.read_text(encoding="utf-8") == "replacement\n"
    # No leftover .tmp files in the directory.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"tempfile leaked: {leftovers}"


# ---------------------------------------------------------------------------
# 6) build_diff pulls overall + per-metal claims from collect JSON
# ---------------------------------------------------------------------------
def test_build_diff_extracts_overall_and_permetal():
    collect = _make_collect()
    paper_text = PAPER_DOC_TEMPLATE
    claims = pau.build_diff(collect, paper_text)
    ids = [c.claim_id for c in claims]
    assert "overall_mean_test_auc" in ids
    for metal in ("Ru", "Ir", "Os", "Rh", "Re"):
        assert f"per_metal_{metal}_mean_test_auc" in ids, f"missing {metal}"
    # All new_values pulled from the fixture round-trip cleanly.
    by_id = {c.claim_id: c for c in claims}
    assert by_id["overall_mean_test_auc"].new_value == pytest.approx(0.7209)
    assert by_id["per_metal_Ru_mean_test_auc"].new_value == pytest.approx(0.7483)
    # And every claim carries a non-empty JSON path.
    assert all(c.json_path for c in claims)


# ---------------------------------------------------------------------------
# 7) _parse_old_value locates Ru 0.909 from §2.1 of the paper doc
# ---------------------------------------------------------------------------
def test_parse_old_value_finds_ru_909():
    val, substr = pau._parse_old_value(PAPER_DOC_TEMPLATE, "per_metal_Ru_mean_test_auc")
    assert val == pytest.approx(0.909)
    assert "0.909" in substr


# ---------------------------------------------------------------------------
# 8) Unmatched claim returns (None, "")  — catastrophic-failure signal
# ---------------------------------------------------------------------------
def test_parse_old_value_returns_none_for_missing():
    val, substr = pau._parse_old_value(
        "No matching numbers here.", "per_metal_Pt_mean_test_auc"
    )
    assert val is None
    assert substr == ""


# ---------------------------------------------------------------------------
# 9) render_diff_md has both old_value + new_value per row (gate G6)
# ---------------------------------------------------------------------------
def test_render_diff_md_contains_old_and_new():
    collect = _make_collect()
    claims = pau.build_diff(collect, PAPER_DOC_TEMPLATE)
    md = pau.render_diff_md(
        claims=claims,
        collect_path=Path("/tmp/collect.json"),
        paper_path=Path("/tmp/paper.md"),
        dry_run=True,
    )
    # Per-claim: old and new both appear in the row.
    for c in claims:
        assert c.claim_id in md, f"{c.claim_id} missing from diff md"
        if c.has_old:
            assert (
                f"{c.old_value:.4f}" in md
            ), f"old_value {c.old_value} missing for {c.claim_id}"
        assert (
            f"{c.new_value:.4f}" in md
        ), f"new_value {c.new_value} missing for {c.claim_id}"
        assert c.json_path in md, f"json_path missing for {c.claim_id}"
    # Header metadata also present.
    assert "Wf-265 paper §4 diff_report" in md
    assert "dry_run:       True" in md


# ---------------------------------------------------------------------------
# 10) render_updated_doc substitutes old values for new values
# ---------------------------------------------------------------------------
def test_render_updated_doc_substitutes_values():
    collect = _make_collect()
    claims = pau.build_diff(collect, PAPER_DOC_TEMPLATE)
    updated = pau.render_updated_doc(PAPER_DOC_TEMPLATE, claims)
    # Old numbers for matched per-metal claims are gone.
    for c in claims:
        if c.has_old:
            # The old value (to 4 dp) should no longer appear in the body
            # — it has been replaced by the new value (also stamped).
            old_str = f"{c.old_value:.4f}"
            # We expect the updated doc to no longer contain the bare
            # old value substring.  Allow it only inside the banner
            # metadata by stripping the banner first.
            body, _, banner = updated.partition("<!-- Wf-265 auto-update banner -->")
            assert (
                old_str not in body or old_str == f"{c.new_value:.4f}"
            ), f"old {old_str} still in body for {c.claim_id}"


# ---------------------------------------------------------------------------
# 11) render_updated_doc appends the Wf-265 banner
# ---------------------------------------------------------------------------
def test_render_updated_doc_appends_banner():
    collect = _make_collect()
    claims = pau.build_diff(collect, PAPER_DOC_TEMPLATE)
    updated = pau.render_updated_doc(PAPER_DOC_TEMPLATE, claims)
    assert "<!-- Wf-265 auto-update banner -->" in updated
    assert "claim(s) substituted" in updated
    assert "generated_utc:" in updated


# ---------------------------------------------------------------------------
# 12) main() in dry-run mode writes only the diff (no paper mutation)
# ---------------------------------------------------------------------------
def test_end_to_end_dry_run(tmp_path: Path):
    collect_path = tmp_path / "collect.json"
    paper_path = tmp_path / "paper.md"
    diff_path = tmp_path / "diff.md"
    collect_path.write_text(json.dumps(_make_collect()), encoding="utf-8")
    paper_path.write_text(PAPER_DOC_TEMPLATE, encoding="utf-8")
    original_paper = paper_path.read_text(encoding="utf-8")

    rc = pau.main(
        [
            "--collect-json",
            str(collect_path),
            "--paper-doc",
            str(paper_path),
            "--out-diff",
            str(diff_path),
            "--updated-out",
            str(tmp_path / "should_not_exist.md"),
        ]
    )
    assert rc == 0
    assert diff_path.exists()
    # Dry-run must NOT have touched the paper doc.
    assert paper_path.read_text(encoding="utf-8") == original_paper
    # And the updated-out file must NOT have been written.
    assert not (tmp_path / "should_not_exist.md").exists()


# ---------------------------------------------------------------------------
# 13) main() with --no-dry-run writes both files and mutates the diff output
# ---------------------------------------------------------------------------
def test_end_to_end_apply_atomic(tmp_path: Path):
    collect_path = tmp_path / "collect.json"
    paper_path = tmp_path / "paper.md"
    diff_path = tmp_path / "diff.md"
    updated_path = tmp_path / "updated.md"
    collect_path.write_text(json.dumps(_make_collect()), encoding="utf-8")
    paper_path.write_text(PAPER_DOC_TEMPLATE, encoding="utf-8")
    original_paper = paper_path.read_text(encoding="utf-8")

    rc = pau.main(
        [
            "--collect-json",
            str(collect_path),
            "--paper-doc",
            str(paper_path),
            "--out-diff",
            str(diff_path),
            "--updated-out",
            str(updated_path),
            "--no-dry-run",
        ]
    )
    assert rc == 0
    # Both files exist.
    assert diff_path.exists()
    assert updated_path.exists()
    # diff_report carries the apply-status marker.
    assert "apply" in diff_path.read_text(encoding="utf-8").lower()
    # Updated doc has the new numbers (NOT the old ones) for matched claims.
    updated_text = updated_path.read_text(encoding="utf-8")
    body, _, _ = updated_text.partition("<!-- Wf-265 auto-update banner -->")
    # The new Ru number (0.7483) should appear; the old (0.909) should not.
    assert "0.7483" in body
    # Old 0.909 should not survive in the body — only new values remain.
    # (It's fine if 0.909 appears inside the banner; we only check body.)
    assert "0.909" not in body
    # The source paper doc must be untouched (atomic write goes to updated_path,
    # NOT in place).
    assert paper_path.read_text(encoding="utf-8") == original_paper


# ---------------------------------------------------------------------------
# 14) The pattern catalogue must be STRUCTURAL, not a set of frozen literals.
#
# This is the regression test for the 2026-10-20 defect: every pattern used to
# embed the old value it was written to find (`(0\.8007)`), so when m21 was
# re-measured the pattern stopped matching and the claim vanished from the diff
# with no error.  Here we take each pattern's own match, replace every digit in
# the captured group with 7, and require the catalogue to still find it and to
# return the mutated number.  A pattern that encodes a literal fails instantly.
# ---------------------------------------------------------------------------
# Claims whose numbers live in report files but have no home in either paper
# doc on disk.  This set is asserted, not skipped: a claim that silently stops
# being anchored anywhere shows up as a test failure.
UNANCHORED_CLAIMS = {
    "wf269c_yuelbond_full_decode_ratio",
    "wf269c_no_yuelbond_decode_ratio",
    "wf271b_m17_reeval_below_random",
    "wf275b_drugood_adapter_rows",
    "wf275c_m14_lead_delta_yuelbond_off",
}


def _corpus() -> list[tuple[str, str]]:
    """(label, text) for every paper doc on disk that the catalogue targets."""
    out = [("<fixture>", PAPER_DOC_TEMPLATE)]
    for rel in (
        "TODO/completed/188_paper_4_krasnov_protocol_aligned.md",
        "docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md",
    ):
        p = Path(_REPO) / rel
        if p.exists():
            out.append((rel, p.read_text(encoding="utf-8")))
    return out


def test_pattern_catalogue_is_structural():
    unanchored = []
    failures = []
    for claim_id, pats in pau.PATTERNS.items():
        found_any = False
        for label, doc in _corpus():
            for pat in pats:
                m = re.search(pat, doc)
                if not m:
                    continue
                found_any = True
                if m.re.groups < 1:
                    failures.append(f"{claim_id}: pattern has no capture group: {pat!r}")
                    continue
                # Mutate ONLY the captured number, so structural anchors such
                # as "5-metal" or "NCI-60 9-cell" stay intact.
                raw = m.group(1)
                mutated_group = re.sub(r"\d", "7", raw)
                mutated_frag = (
                    m.group(0)[: m.start(1) - m.start(0)]
                    + mutated_group
                    + m.group(0)[m.end(1) - m.start(0) :]
                )
                val, _ = pau._parse_old_value(mutated_frag, claim_id, warn=False)
                expected = pau._to_float(mutated_group)
                if val is None or expected is None or abs(val - expected) > 1e-9:
                    failures.append(
                        f"{claim_id}: pattern stopped matching after mutating "
                        f"{raw!r} -> {mutated_group!r} in {label} ({pat!r}); "
                        f"got {val!r}, expected {expected!r}"
                    )
        if not found_any:
            unanchored.append(claim_id)

    assert not failures, "non-structural pattern(s):\n" + "\n".join(failures)
    assert set(unanchored) == UNANCHORED_CLAIMS, (
        f"anchored-claim set changed: {sorted(set(unanchored))} != "
        f"{sorted(UNANCHORED_CLAIMS)}.  If a doc was edited, extend PATTERNS; "
        f"do not delete the claim silently."
    )


# ---------------------------------------------------------------------------
# 15) m21 must parse BOTH the superseded and the live doc form
# ---------------------------------------------------------------------------
def test_m21_patterns_track_the_live_doc():
    legacy = "**Aggregate**: per-cell mean AUC = **0.8007 ± 0.0160** (3 seeds ...)"
    val, _ = pau._parse_old_value(legacy, "m21_nci60_9cell_mean_auc", warn=False)
    assert val == pytest.approx(0.8007)

    live_table = (
        "| T1 | **m21 NCI-60 9-cell AUC** (D-MPNN k-cell-line multi-task) | "
        "`0.7294` (9-cell unweighted mean, 5 seeds) | MEASURED_HONEST_NEGATIVE |"
    )
    val, _ = pau._parse_old_value(live_table, "m21_nci60_9cell_mean_auc", warn=False)
    assert val == pytest.approx(0.7294)

    live_bullet = (
        "- **MEASURED headline**: per-cell ROC-AUC, unweighted mean over 9 cell lines\n"
        "  x 5 seeds x 50 epoch; fixed Bemis-Murcko scaffold split\n"
        "  = `0.7294` (sample SD across cells `0.2137`)\n"
    )
    val, _ = pau._parse_old_value(live_bullet, "m21_nci60_9cell_mean_auc", warn=False)
    assert val == pytest.approx(0.7294)


# ---------------------------------------------------------------------------
# 16) A pattern miss reaches stderr (it used to return (None, "") in silence)
# ---------------------------------------------------------------------------
def test_pattern_miss_is_reported(capsys):
    val, substr = pau._parse_old_value(
        "nothing here", "m21_nci60_9cell_mean_auc", warn=True
    )
    assert (val, substr) == (None, "")
    err = capsys.readouterr().err
    assert "no pattern matched claim 'm21_nci60_9cell_mean_auc'" in err
    assert "catalogue is stale" in err


# ---------------------------------------------------------------------------
# 17) --strict turns a pattern miss into a non-zero exit
# ---------------------------------------------------------------------------
def test_strict_exit_code_on_pattern_miss(tmp_path: Path):
    collect_path = tmp_path / "collect.json"
    paper_path = tmp_path / "paper.md"
    collect_path.write_text(json.dumps(_make_collect()), encoding="utf-8")
    # A doc with none of the expected anchors -> every claim is a miss.
    paper_path.write_text("# empty\n", encoding="utf-8")
    base = [
        "--collect-json",
        str(collect_path),
        "--paper-doc",
        str(paper_path),
        "--out-diff",
        str(tmp_path / "diff.md"),
    ]
    assert pau.main(base) == 0
    assert pau.main(base + ["--strict"]) == 3


# ---------------------------------------------------------------------------
# 18) Apply mode swaps the number and leaves the surrounding markup alone
# ---------------------------------------------------------------------------
def test_apply_preserves_markup():
    claims = pau.build_diff(_make_collect(), PAPER_DOC_TEMPLATE)
    updated = pau.render_updated_doc(PAPER_DOC_TEMPLATE, claims)
    body, _, _ = updated.partition("<!-- Wf-265 auto-update banner -->")
    # Table row for Ru survives as a well-formed row with the new number.
    assert "| **Ru** | **0.7483 <!-- Wf-265 auto-update -->** | 0.81 |" in body
    # The 5-metal mean keeps its "± 0.014" tail.
    assert "± 0.014**" in body
