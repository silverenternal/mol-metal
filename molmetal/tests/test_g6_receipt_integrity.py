# SPDX-License-Identifier: Apache-2.0
"""Wf-A124 — regression guard for the G6 per-innovation ablation's provenance.

WHY THIS EXISTS
---------------
Wf-A124 (2026-10-31) repaired the provenance of the G6 per-innovation ablation
tree ``results/wf124_p3_per_innovation_ablation_g6/``.  The repair found five
independent defect classes, and every one of them is silent unless a test
looks for it:

  1. The tree was **git-only**.  ``git status --porcelain`` reported 74 ``" D"``
     entries (deleted in the worktree, unstaged) while ``git ls-tree -r HEAD``
     still listed 76 committed paths; only 4 files existed on disk.  A receipt
     that recomputes from files that are not on disk is a receipt nobody can
     check.
  2. The two cohort receipts (``no_wass/metrics.json``,
     ``no_multi_task/metrics.json``) record ``test_auc_std`` as the
     **population** stdev (ddof = 0), not the sample stdev (ddof = 1).  A later
     reader who assumes the sample convention mis-reads the spread.
  3. The G4 sibling slot ``C/Ru/seed0/ru/per_seed_metrics_seed0.json`` was a
     **mis-pathed duplicate**: a bare filesystem-path string, not JSON, in a
     slot that looks like a measurement.  ``json.load()`` raised on the
     committed blob; as a live symlink it silently served the G5 metrics.
  4. The published per-lever Δ table (``TODO/INDEX.md`` and its origin doc
     ``TODO/completed/metric_rescue_wfmrr_2026_10_08.md``) does not reproduce
     from the shipped artifacts under any single reference arm — 2 of 6 cells
     under its own declared reference, 0 of 6 under the G5 all-ON baseline with
     the same metal column; 4 of 6 in total, each against a DIFFERENT reference,
     and 2 under nothing.
  5. ``TODO/completed/330b4_regression_smoke.md`` quoted a test_auc of
     **0.7888** that appears nowhere in any G6 artifact.

RED CHECK: delete any of the 74 restored G6 artifacts from the worktree and
``TreeCompleteness`` fails; edit one ``test_auc`` in a per-seed file and
``ReceiptRecompute`` fails; re-add a bare ``0.7888`` assertion to 330b4 and
``UnsourcedZero7888`` fails; restore the retracted Δ table to INDEX.md **or to
metric_rescue_wfmrr_2026_10_08.md** — in the PROSE form **or in the TABULAR
form** — as a live claim and ``DeltaTableGuard`` fails; delete the G5 baseline
tree (or any of its per-seed files) and ``PairedBaselineRecompute`` fails;
flip a §6.218 significance verdict cell from NO to YES and
``VerdictCellGuard`` fails.

RETACTION-MARKER RULE (used by T6 and T7)
------------------------------------------
Both ``0.7888`` and the Δ table literal are still *present* in the docs — on
purpose.  AGENTS.md §5.5 rule #6 requires corrections to be ADDITIVE and
DATED, so the historical claim is preserved and annotated rather than deleted.
A plain substring check would therefore forbid the very retraction notice that
documents the retraction.  The rule implemented here:

    a line that mentions the token is ACCEPTED only if the SAME line also
    carries a retraction marker, case-insensitively, one of:
        "previously claimed", "unsourced", "retracted", "withdrawn",
        "not reproducible"

A NEW bare assertion ("test_auc = 0.7888 for Ru seed 0") carries no marker and
fails.  T6 and T7 each assert the marker rule against a synthetic bare line
(negative control) and a synthetic marked line (positive control), so each
guard proves it can fail before it is trusted on the real file.

SOURCE OF EVERY CONSTANT BELOW
-------------------------------
All numbers are MEASURED 2026-10-31 from git HEAD by Wf-A124 and are quoted
verbatim; they are not derived by this test file.  Re-derivation from the
per-seed artifacts is asserted separately, so a drifting artifact AND a
drifting constant both fail loudly.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

G6_REL = "results/wf124_p3_per_innovation_ablation_g6"
G6 = REPO / G6_REL
G4_REL = "results/wf124_p3_remaining_metals_g4"
G4_STUB_REL = f"{G4_REL}/C/Ru/seed0/ru/per_seed_metrics_seed0.json"
CATALOG_REL = "docs/HONEST_NEGATIVE_CATALOG.md"
INDEX_REL = "TODO/INDEX.md"
DOC_330B4_REL = "TODO/completed/330b4_regression_smoke.md"
# The Δ table's ORIGIN doc ([F15]) — it must be guarded as well as INDEX.md.
METRIC_RESCUE_REL = "TODO/completed/metric_rescue_wfmrr_2026_10_08.md"
G5_REL = "results/wf124_p3_50epoch_full_sweep_g5"

RECEIPTS = ("no_wass", "no_multi_task")
METALS = ("Ru", "Rh")
SEEDS = (0, 1, 2)

# MEASURED 2026-10-31, git HEAD (Wf-A124 [F2]): the receipt's stored
# (test_auc_mean, test_auc_std) for each (cohort, metal) cell.
EXPECTED_RECEIPT = {
    ("no_wass", "Ru"): (0.7863669994349761, 0.0044902747887155875),
    ("no_wass", "Rh"): (0.6645067804024497, 0.01042287200670721),
    ("no_multi_task", "Ru"): (0.7796456131518155, 0.00763148722290009),
    ("no_multi_task", "Rh"): (0.664588801399825, 0.017941466941118877),
}

# MEASURED 2026-10-31, git HEAD (Wf-A124 [F3]): the three per-seed test_auc
# values each receipt aggregates.
EXPECTED_PER_SEED = {
    ("no_wass", "Ru"): (
        0.7867673560967009,
        0.7916553215755378,
        0.7806783206326897,
    ),
    ("no_wass", "Rh"): (
        0.6780675853018373,
        0.662729658792651,
        0.6527230971128609,
    ),
    ("no_multi_task", "Ru"): (
        0.7688537839247517,
        0.7851497511470598,
        0.7849333043836351,
    ),
    ("no_multi_task", "Rh"): (
        0.6823326771653543,
        0.6400098425196851,
        0.6714238845144358,
    ),
}

# The retraction markers, per the docstring's RETRACTION-MARKER RULE.
_RETRACTION_MARKERS = (
    "previously claimed",
    "unsourced",
    "retracted",
    "withdrawn",
    "not reproducible",
)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _per_seed_metrics_rel(cohort: str, metal: str, seed: int) -> str:
    return f"{G6_REL}/{cohort}/{metal}/seed{seed}/{metal.lower()}/per_seed_metrics.json"


def _training_summary_rel(cohort: str, metal: str, seed: int) -> str:
    return f"{G6_REL}/{cohort}/{metal}/seed{seed}/training_summary.json"


def _per_seed_auc(cohort: str, metal: str, seed: int) -> float:
    """The single measured test_auc of one (cohort, metal, seed) run."""
    doc = _read_json(REPO / _per_seed_metrics_rel(cohort, metal, seed))
    per_seed = doc["per_seed"]
    if len(per_seed) != 1:
        raise AssertionError(
            f"{cohort}/{metal}/seed{seed}: expected 1 per-seed entry, got {len(per_seed)}"
        )
    return per_seed[0]["test_auc"]


def _all_keys(node) -> list:
    """Every mapping key reachable from a JSON document."""
    keys = []
    if isinstance(node, dict):
        for key, value in node.items():
            keys.append(key)
            keys.extend(_all_keys(value))
    elif isinstance(node, list):
        for item in node:
            keys.extend(_all_keys(item))
    return keys


def _bare_mentions(text: str, token: str) -> list:
    """Lines that mention ``token`` without labelling the mention a retraction.

    This is the machine form of the RETRACTION-MARKER RULE in the docstring.
    Returns ``[(lineno, line), ...]``.
    """
    bare = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if token not in line:
            continue
        lowered = line.lower()
        if not any(marker in lowered for marker in _RETRACTION_MARKERS):
            bare.append((lineno, line.strip()))
    return bare


class ReceiptRecompute(unittest.TestCase):
    """T1 — the two cohort receipts must recompute from the per-seed files."""

    def test_receipt_mean_equals_the_mean_of_the_three_per_seed_aucs(self) -> None:
        for cohort in RECEIPTS:
            for metal in METALS:
                with self.subTest(cohort=cohort, metal=metal):
                    receipt = _read_json(G6 / cohort / "metrics.json")
                    stored = receipt["per_metal"][metal]["test_auc_mean"]
                    seeds = [_per_seed_auc(cohort, metal, s) for s in SEEDS]
                    self.assertAlmostEqual(
                        stored,
                        sum(seeds) / len(seeds),
                        delta=1e-9,
                        msg=f"{cohort}/{metal}: receipt test_auc_mean does not recompute",
                    )

    def test_receipt_std_is_the_population_stdev_not_the_sample_stdev(self) -> None:
        for cohort in RECEIPTS:
            for metal in METALS:
                with self.subTest(cohort=cohort, metal=metal):
                    receipt = _read_json(G6 / cohort / "metrics.json")
                    stored = receipt["per_metal"][metal]["test_auc_std"]
                    seeds = [_per_seed_auc(cohort, metal, s) for s in SEEDS]
                    self.assertAlmostEqual(
                        stored,
                        statistics.pstdev(seeds),
                        delta=1e-9,
                        msg=f"{cohort}/{metal}: receipt test_auc_std is not the population stdev",
                    )
                    # The sample stdev is strictly larger in every cell, so if the
                    # receipt ever silently switches convention this fires.
                    self.assertGreater(
                        abs(stored - statistics.stdev(seeds)),
                        1e-6,
                        f"{cohort}/{metal}: receipt std matches the SAMPLE stdev; "
                        "the committed convention is the population stdev (ddof=0)",
                    )

    def test_receipt_values_equal_the_expected_constants(self) -> None:
        """Independent check against the MEASURED 2026-10-31 constants."""
        for (cohort, metal), (mean, std) in EXPECTED_RECEIPT.items():
            with self.subTest(cohort=cohort, metal=metal):
                receipt = _read_json(G6 / cohort / "metrics.json")
                cell = receipt["per_metal"][metal]
                self.assertAlmostEqual(cell["test_auc_mean"], mean, delta=1e-9)
                self.assertAlmostEqual(cell["test_auc_std"], std, delta=1e-9)
                seeds = [_per_seed_auc(cohort, metal, s) for s in SEEDS]
                for got, want in zip(seeds, EXPECTED_PER_SEED[(cohort, metal)]):
                    self.assertAlmostEqual(got, want, delta=1e-12)


class SourceFilesPresent(unittest.TestCase):
    """T2 — every file a receipt depends on must exist ON DISK.

    This is the guard that would have caught the git-only window: the receipts
    were committed and recomputable from git, but the files they name were
    absent from the worktree.
    """

    def test_every_source_artifact_named_by_a_receipt_exists_on_disk(self) -> None:
        for cohort in RECEIPTS:
            with self.subTest(cohort=cohort):
                receipt = _read_json(G6 / cohort / "metrics.json")
                sources = list(receipt["source_artifacts"])
                self.assertTrue(
                    sources, f"{cohort}: receipt declares no source_artifacts"
                )
                expected = {
                    _training_summary_rel(cohort, metal, seed)
                    for metal in METALS
                    for seed in SEEDS
                }
                self.assertEqual(
                    set(sources),
                    expected,
                    f"{cohort}: the receipt's declared provenance is not the 6 runs it aggregates",
                )
                missing = [p for p in sources if not (REPO / p).exists()]
                self.assertEqual(
                    missing,
                    [],
                    f"{cohort}: source artifacts missing from the worktree (git-only window)",
                )

    def test_every_per_seed_metrics_file_the_receipt_recomputes_from_exists(
        self,
    ) -> None:
        for cohort in RECEIPTS:
            for metal in METALS:
                for seed in SEEDS:
                    with self.subTest(cohort=cohort, metal=metal, seed=seed):
                        rel = _per_seed_metrics_rel(cohort, metal, seed)
                        self.assertTrue(
                            (REPO / rel).exists(),
                            f"{rel} missing from the worktree (git-only window)",
                        )


class TreeCompleteness(unittest.TestCase):
    """T3 — the G6 tree must hold every path committed in git HEAD."""

    @staticmethod
    def _head_paths() -> list:
        try:
            proc = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", G6_REL],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            raise unittest.SkipTest(f"git unavailable: {exc}") from exc
        if proc.returncode != 0:  # pragma: no cover
            raise unittest.SkipTest(f"git ls-tree failed: {proc.stderr.strip()}")
        return [line for line in proc.stdout.splitlines() if line.strip()]

    def test_every_path_committed_in_head_exists_on_disk(self) -> None:
        paths = self._head_paths()
        self.assertGreaterEqual(
            len(paths),
            76,
            "git HEAD holds fewer G6 paths than the committed 76 — the tree shrank",
        )
        missing = [p for p in paths if not (REPO / p).exists()]
        self.assertEqual(
            missing,
            [],
            "G6 tree lost committed paths on disk (the git-only window reopened)",
        )

    def test_the_tree_still_holds_24_runs_two_receipts_and_the_launcher_files(
        self,
    ) -> None:
        paths = self._head_paths()
        n_summary = sum(1 for p in paths if p.endswith("/training_summary.json"))
        n_per_seed = sum(1 for p in paths if p.endswith("/per_seed_metrics.json"))
        n_per_seed_named = sum(
            1 for p in paths if re.search(r"/per_seed_metrics_seed\d+\.json$", p)
        )
        n_receipts = sum(
            1
            for p in paths
            if p.endswith("/metrics.json")
            and p.rsplit("/", 1)[0].rsplit("/", 1)[-1] in RECEIPTS
        )
        self.assertEqual(
            n_summary,
            24,
            "expected 24 training_summary.json (4 cohorts x 2 metals x 3 seeds)",
        )
        self.assertEqual(n_per_seed, 24, "expected 24 per_seed_metrics.json")
        self.assertEqual(
            n_per_seed_named, 24, "expected 24 per_seed_metrics_seed<N>.json"
        )
        self.assertEqual(
            n_receipts, 2, "expected 2 cohort receipts (no_wass, no_multi_task)"
        )
        self.assertIn(f"{G6_REL}/plan.json", paths)
        self.assertIn(f"{G6_REL}/recovery_manifest.tsv", paths)


class MisPathedStub(unittest.TestCase):
    """T4 — the G4 C/Ru/seed0 slot must be a stub, not a look-alike measurement."""

    def test_the_g4_slot_parses_and_carries_no_metric(self) -> None:
        path = REPO / G4_STUB_REL
        self.assertTrue(path.exists(), f"{G4_STUB_REL} is missing from disk")
        self.assertFalse(
            path.is_symlink(),
            f"{G4_STUB_REL} is a symlink again; as a live symlink it silently "
            "serves the G5 metrics instead of raising on json.load()",
        )
        doc = _read_json(path)
        self.assertIs(
            doc.get("is_a_measurement"),
            False,
            f"{G4_STUB_REL} must declare is_a_measurement == false",
        )
        self.assertEqual(
            doc.get("MEASURED_vs_PROJECTED"),
            "NOT_A_MEASUREMENT",
            f"{G4_STUB_REL} must declare MEASURED_vs_PROJECTED == NOT_A_MEASUREMENT",
        )
        forbidden = {"test_auc", "val_auc", "per_seed"}
        for key in _all_keys(doc):
            self.assertNotIn(
                key,
                forbidden,
                f"{G4_STUB_REL} carries the metric key {key!r}; it must carry NO metric",
            )
            self.assertFalse(
                key.startswith("test_auc") or key.startswith("val_auc"),
                f"{G4_STUB_REL} carries the metric key {key!r}; it must carry NO metric",
            )


class CatalogMarker(unittest.TestCase):
    """T5 — the honest-negative entry must not be silently deleted."""

    _SECTION_RE = re.compile(r"^## §6\.(\d+) — (.*)$", re.MULTILINE)

    def _sections(self) -> list:
        text = (REPO / CATALOG_REL).read_text(encoding="utf-8")
        matches = list(self._SECTION_RE.finditer(text))
        out = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            out.append((int(match.group(1)), match.group(2), text[match.start() : end]))
        return out

    def test_the_wf_a124_g6_entry_still_records_that_no_lever_is_resolvable(
        self,
    ) -> None:
        sections = self._sections()
        hits = [s for s in sections if "G6" in s[1] and "RESOLVABLE" in s[1].upper()]
        self.assertEqual(
            len(hits),
            1,
            "expected exactly one catalog section whose header names the G6 "
            "no-lever-is-resolvable finding; found "
            f"{[s[0] for s in hits]}",
        )
        number, header, body = hits[0]
        # Phrase read from the catalog's own header text
        # (docs/HONEST_NEGATIVE_CATALOG.md, §6.218).  The task brief paraphrases
        # it as "NO G6 LEVER IS STATISTICALLY RESOLVABLE"; the committed text
        # reads "NO G6 LEVER IS RESOLVABLE".
        self.assertIn(
            "NO G6 LEVER IS RESOLVABLE",
            header,
            f"catalog §6.{number}: the Wf-A124 G6 heading text was weakened",
        )
        self.assertIn(
            "G5", header, f"catalog §6.{number}: the heading must name the G5 baseline"
        )
        self.assertIn(
            "Wf-A124", body, f"catalog §6.{number}: the body must name the workflow"
        )
        self.assertIn(
            "paired",
            body.lower(),
            f"catalog §6.{number}: the body must record the paired comparison",
        )

    def test_all_seven_wf_a124_catalog_sections_are_present(self) -> None:
        tagged = [
            n for n, _h, body in self._sections() if "Wf-A124 (2026-10-31" in body
        ]
        for expected in range(216, 223):
            self.assertIn(
                expected,
                tagged,
                f"catalog §6.{expected} (a Wf-A124 entry) is missing or no longer "
                "tagged with the workflow",
            )
        self.assertGreaterEqual(len(tagged), 7)


class UnsourcedZero7888(unittest.TestCase):
    """T6 — 0.7888 must never reappear as a bare assertion.

    0.7888 appears nowhere in any G6 artifact.  The real values are
    no_multi_task/Ru/seed0 = 0.7688537839247517 and seed1 = 0.7851497511470598.
    """

    TOKEN = "0.7888"

    def test_the_330b4_closeout_only_mentions_0_7888_inside_a_retraction_marker(
        self,
    ) -> None:
        path = REPO / DOC_330B4_REL
        self.assertTrue(path.exists(), f"{DOC_330B4_REL} is missing from disk")
        text = path.read_text(encoding="utf-8")
        bare = _bare_mentions(text, self.TOKEN)
        self.assertEqual(
            bare,
            [],
            f"bare {self.TOKEN} assertion(s) re-appeared in 330b4: {bare}",
        )
        # The retraction trail itself must survive (AGENTS.md §5.5 rule #6):
        # a silent edit that deletes every mention is also a defect.
        marked = [
            lineno
            for lineno, line in enumerate(text.splitlines(), 1)
            if self.TOKEN in line
            and any(m in line.lower() for m in _RETRACTION_MARKERS)
        ]
        self.assertTrue(
            marked,
            f"330b4 no longer mentions {self.TOKEN} on a marked line; the "
            "retraction trail was deleted rather than corrected",
        )
        # Controls: the marker rule must reject a bare line and accept a
        # marked one, otherwise this guard cannot fail.
        self.assertEqual(
            len(_bare_mentions("the run scored 0.7888 for Ru seed 0\n", self.TOKEN)),
            1,
            "negative control failed: a bare 0.7888 line was not flagged",
        )
        self.assertEqual(
            _bare_mentions("previously claimed (unsourced): 0.7888\n", self.TOKEN),
            [],
            "positive control failed: a marked 0.7888 line was flagged",
        )


class DeltaTableGuard(unittest.TestCase):
    """T7 — the retracted per-lever Δ table must not reappear as a live claim.

    Two forms of the literal are matched, because the table has TWO forms:

      * the PROSE form  — "Wasserstein Δ (Ru +0.007, Rh −0.0001)"
      * the TABULAR form — "| Wasserstein (`no_wass`) | +0.007 | −0.0001 |"

    The tabular form is the table's ORIGIN form ([F15]: the table lives in
    ``TODO/completed/metric_rescue_wfmrr_2026_10_08.md`` §2 in exactly that
    shape), so a guard that only matches the prose form does not guard the
    table.  Both docs are guarded.
    """

    TOKEN_RES = (
        re.compile(r"Wasserstein\s*Δ\s*\(\s*Ru\s*\+0\.007"),
        re.compile(r"\|\s*Wasserstein\s*\(`?no_wass`?\)\s*\|\s*\+0\.007"),
    )

    def _scan(self, text):
        """Return ``(bare, trail)`` for one document.

        ``bare``  — Δ-table lines with no retraction marker anywhere in their
                    immediate context (a LIVE claim).
        ``trail`` — the line numbers that evidence the retraction: the Δ-table
                    lines that carry a marker themselves, plus the caption lines
                    that cover a marked table block.
        """
        lines = text.splitlines()
        bare = []
        trail = []
        for lineno, line in enumerate(lines, 1):
            if not any(p.search(line) for p in self.TOKEN_RES):
                continue
            if any(m in line.lower() for m in _RETRACTION_MARKERS):
                trail.append(lineno)
                continue
            if line.lstrip().startswith("|"):
                # A table row is accepted when the table's OWN header row or the
                # caption line immediately above the block carries a retraction
                # marker: the historical table is preserved verbatim per
                # AGENTS.md §5.5 rule #6 and is annotated by its caption.
                start = lineno - 1
                while start > 0 and lines[start - 1].lstrip().startswith("|"):
                    start -= 1
                header = lines[start]
                caption = ""
                caption_lineno = 0
                j = start - 1
                while j >= 0 and not lines[j].strip():
                    j -= 1
                if j >= 0:
                    caption = lines[j]
                    caption_lineno = j + 1
                context = (header + " " + caption).lower()
                if any(m in context for m in _RETRACTION_MARKERS):
                    trail.append(caption_lineno)
                    continue
            bare.append((lineno, line.strip()))
        return bare, trail

    def _bare(self, text):
        return self._scan(text)[0]

    def _assert_guarded(self, rel):
        path = REPO / rel
        self.assertTrue(path.exists(), f"{rel} is missing from disk")
        text = path.read_text(encoding="utf-8")
        bare, trail = self._scan(text)
        self.assertEqual(
            bare,
            [],
            f"the retracted per-lever Δ table re-appeared as a live claim in {rel}: {bare}",
        )
        self.assertTrue(
            trail,
            f"{rel} no longer carries the Δ-table literal inside a retraction marker; "
            "the retraction trail was deleted rather than corrected",
        )

    def test_the_index_delta_table_literal_only_appears_inside_a_retraction_marker(
        self,
    ) -> None:
        self._assert_guarded(INDEX_REL)

    def test_the_origin_doc_delta_table_only_appears_inside_a_retraction_marker(
        self,
    ) -> None:
        self._assert_guarded(METRIC_RESCUE_REL)

    # Controls: the marker rule must reject a bare line and accept a marked
    # one, otherwise this guard cannot fail.
    def test_the_marker_rule_rejects_a_bare_line_and_accepts_a_marked_one(self) -> None:
        self.assertEqual(
            len(
                _bare_mentions(
                    "Wasserstein Δ (Ru +0.007, Rh −0.0001)\n",
                    "Wasserstein Δ (Ru +0.007",
                )
            ),
            1,
            "negative control failed: a bare Δ-table line was not flagged",
        )
        self.assertEqual(
            _bare_mentions(
                "Previously claimed — retracted (not reproducible): Wasserstein Δ (Ru +0.007)\n",
                "Wasserstein Δ (Ru +0.007",
            ),
            [],
            "positive control failed: a marked Δ-table line was flagged",
        )
        # Negative control for the TABULAR form (the hole the critic found).
        tabular = "| Wasserstein (`no_wass`) | +0.007 | −0.0001 |\n"
        self.assertTrue(
            any(p.search(tabular) for p in self.TOKEN_RES),
            "negative control failed: the tabular Δ-table form is not matched at all",
        )
        self.assertEqual(
            self._bare(tabular),
            [(1, tabular.strip())],
            "negative control failed: a bare tabular Δ-table row was not flagged",
        )
        self.assertEqual(
            self._bare(
                "| Lever Δ | Ru | Rh |\n"
                "|---|---|---|\n"
                "| Wasserstein (`no_wass`) | +0.007 | −0.0001 |\n"
            ),
            [(3, "| Wasserstein (`no_wass`) | +0.007 | −0.0001 |")],
            "negative control failed: a tabular Δ-table row with an unmarked header "
            "was not flagged",
        )
        self.assertEqual(
            self._bare(
                "caption — retracted, not reproducible\n"
                "\n"
                "| Lever Δ | Ru | Rh |\n"
                "|---|---|---|\n"
                "| Wasserstein (`no_wass`) | +0.007 | −0.0001 |\n"
            ),
            [],
            "positive control failed: a marked caption did not cover the table block",
        )


class PairedBaselineRecompute(unittest.TestCase):
    """T8 — the paired-vs-G5 Δ and t must re-derive from the two trees ON DISK.

    Nothing in the wave re-derived the headline comparison, so deleting the
    whole G5 baseline tree left every other guard green (the hole the critic
    found).  This test recomputes all four arms from the per-seed artifacts and
    checks the published verdicts:  n = 6 paired differences, df = 5,
    significance threshold |t| > 2.5706, every arm NOT significant.
    """

    # MEASURED 2026-10-31 by Wf-A124 (catalog §6.218), recomputed here.
    EXPECTED_T = {
        "no_wass": 1.3177182050052332,
        "no_multi_task": 0.9708467570400957,
        "no_kendall": -0.2647284354420802,
        "no_dann": 0.1606359478438517,
    }
    CRITICAL = 2.570581835636

    def _g5_auc(self, metal, seed):
        rel = f"{G5_REL}/{metal}/seed{seed}/{metal.lower()}/per_seed_metrics.json"
        path = REPO / rel
        self.assertTrue(path.exists(), f"G5 baseline file missing from disk: {rel}")
        doc = _read_json(path)
        return doc["per_seed"][0]["test_auc"]

    def test_the_g5_baseline_tree_is_present_on_disk(self) -> None:
        for metal in METALS:
            for seed in SEEDS:
                rel = f"{G5_REL}/{metal}/seed{seed}/{metal.lower()}/per_seed_metrics.json"
                self.assertTrue(
                    (REPO / rel).exists(),
                    f"the G5 baseline tree is incomplete on disk (missing {rel}); the "
                    "paired comparison in §6.218 cannot be re-derived without it",
                )

    def test_the_paired_deltas_and_t_re_derive_from_disk(self) -> None:
        for cohort, expected_t in self.EXPECTED_T.items():
            deltas = []
            for metal in METALS:
                for seed in SEEDS:
                    deltas.append(
                        _per_seed_auc(cohort, metal, seed) - self._g5_auc(metal, seed)
                    )
            self.assertEqual(len(deltas), 6, "expected 6 paired differences")
            mean = statistics.fmean(deltas)
            sd = statistics.stdev(deltas)
            t = mean / (sd / (len(deltas) ** 0.5))
            self.assertAlmostEqual(
                t,
                expected_t,
                places=9,
                msg=(
                    f"{cohort}: the paired t re-derived from disk ({t!r}) does not match "
                    f"the published value ({expected_t!r}) — the artifacts or the "
                    "published number drifted"
                ),
            )
            self.assertLess(
                abs(t),
                self.CRITICAL,
                f"{cohort}: |t| = {abs(t)} reaches the df=5 significance threshold; "
                "§6.218's 'no lever is resolvable' verdict would no longer hold",
            )


class VerdictCellGuard(unittest.TestCase):
    """T9 — §6.218's significance verdict cells must all still read NO.

    The verdict IS the finding.  A silent edit that flips a cell from NO to YES
    would leave every other guard green, so the cells are parsed directly.
    """

    SECTION_NO = 218

    def _section(self):
        text = (REPO / CATALOG_REL).read_text(encoding="utf-8")
        matches = list(re.finditer(r"^## §6\.(\d+) — (.*)$", text, re.MULTILINE))
        for index, match in enumerate(matches):
            if int(match.group(1)) != self.SECTION_NO:
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            return text[match.start() : end]
        raise AssertionError(f"catalog §6.{self.SECTION_NO} is missing")

    def test_every_arm_row_of_the_paired_table_still_reads_NO(self) -> None:
        body = self._section()
        rows = []
        for line in body.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5:
                continue
            arm = cells[0]
            if not arm.startswith("`"):
                continue
            rows.append((arm, cells))
        self.assertEqual(
            len(rows),
            4,
            f"expected 4 arm rows in the §6.{self.SECTION_NO} paired table, found "
            f"{[r[0] for r in rows]}",
        )
        for arm, cells in rows:
            verdict = cells[4]
            self.assertIn(
                "NO",
                verdict.upper(),
                f"catalog §6.{self.SECTION_NO}: the significance verdict for {arm} no "
                f"longer reads NO (it reads {verdict!r})",
            )
            self.assertNotIn(
                "YES",
                verdict.upper(),
                f"catalog §6.{self.SECTION_NO}: the significance verdict for {arm} was "
                f"flipped to YES ({verdict!r})",
            )


if __name__ == "__main__":
    unittest.main()
