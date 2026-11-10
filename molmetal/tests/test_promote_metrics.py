"""Pytest: Wf-87 Phase 2 — promote MEASURED JSONs from reports/ to metrics/by_metric/.

Verifies the ``promote_metrics.py`` script:

1. ``test_promote_runs`` — running the script exits 0 and writes the log.
2. ``test_promote_at_least_5_new_files`` — at least 5 NEW central metric
   files are produced per the Wf-87 P2 verify gate.
3. ``test_promote_no_collisions_on_first_run`` — first run ships zero
   collisions (every target is brand-new).
4. ``test_promote_existing_target_preserved`` — second run preserves
   existing central files (collision detection).
5. ``test_promote_hilbert_4cell_r2_promoted`` — the Wf-83 4-cell combined
   R² file ships with MEASURED status + per-cell-line breakdown.
6. ``test_promote_metalcytotox_combined_r2_promoted`` — MetalCytoToxDB
   combined R² ships with per-method breakdown.
7. ``test_promote_tmqm_combined_r2_promoted`` — tmqm-dblock + cross-
   dataset summary ships with both per-cohort + cross-dataset blocks.
8. ``test_promote_platinai_combined_r2_promoted`` — PlatinAI 4-cell
   baseline ASTSR R² value is preserved verbatim.
9. ``test_promote_ood_per_metal_promoted`` — Wf-73 P2 per-precious-metal
   OOD reachability ships with 5 metals.
10. ``test_promote_ood_per_element_promoted`` — Wf-73 P3 per-d-block-
    element OOD reachability ships with 5 elements + gate threshold.
11. ``test_promote_dmpnn_ckpt_status_promoted`` — D-MPNN checkpoint
    inventory file ships with ``n_checkpoints >= 0`` (read-only).
12. ``test_promote_vina_status_promoted`` — Vina status file ships with
    ``meeko_installed`` + ``vina_binary_available`` probe results.
13. ``test_promote_dry_run_does_not_write`` — ``--dry-run`` skips writes.
14. ``test_promote_force_overwrites`` — ``--force`` overwrites the
    existing target on second run.
15. ``test_promote_honest_framing_in_promoted_files`` — every promoted
    file contains ``honest_framing_note`` + ``status`` fields.
16. ``test_promote_no_projected_in_central`` — no PROJECTED tokens leak
    into the central metric files (only MEASURED-on-disk evidence).
17. ``test_promote_promotion_log_present`` — promotion_log.json ships
    with promoter_count + promoted_count + skipped_count + new_files.
18. ``test_promote_never_touches_real_published_tree`` — the real
    ``metrics/by_metric/`` tree is byte-identical before and after this
    module runs (Wf-A123).

**Wf-A123 isolation.**  Before this change this module DELETED nine real
files under the repository's ``metrics/by_metric/`` and re-promoted
older content over them, so a plain test run silently rewrote published
artifacts.  Every WRITE now goes into pytest's ``tmp_path`` tree via the
script's ``--target-root`` / ``--log-path`` flags; the SOURCES are still
read from the real repository, so every assertion below still checks
real on-disk evidence.

All tests run on CPU in <5s — no GPU, no RDKit, no metric fabrication.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "promote_metrics.py"
# The REAL published tree — read-only reference.  Nothing in this module
# may write here (Wf-A123).
BY_METRIC_DIR = PROJECT_ROOT / "metrics" / "by_metric"

import pytest  # noqa: E402

# Files this module used to purge out of the REAL tree.  The list is kept
# so the purge still happens — against the tmp target tree.
PURGE_TARGETS = (
    "hilbert_4cell_r2.json",
    "platinai_combined_r2.json",
    "metalcytotox_combined_r2.json",
    "metalcytotox_5metal_r2.json",
    "tmqm_combined_r2.json",
    "ood_per_metal_reachability.json",
    "ood_per_element_reachability.json",
    "dmpnn_ckpt_status.json",
    "vina_status.json",
)


def _sha256_tree(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root``, keyed by path relative to root."""
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    return out


# Snapshot of the REAL published tree taken at collection time — before
# any test in this module has run.
_REAL_TREE_SHA_AT_IMPORT = _sha256_tree(BY_METRIC_DIR)


class _PromoteEnv:
    """Redirects the promotion WRITE side into a tmp tree (Wf-A123)."""

    def __init__(self, target_root: Path, log_path: Path) -> None:
        self.target_root = target_root
        self.log_path = log_path
        self.first_log: dict = {}

    def target(self, relpath: str) -> Path:
        """Resolve a repo-relative path against the tmp target root."""
        return self.target_root / relpath

    def purge(self) -> None:
        """Remove the promoted targets from the TMP tree (never the real one)."""
        for name in PURGE_TARGETS:
            p = self.target_root / "metrics" / "by_metric" / name
            if p.exists():
                p.unlink()
        if self.log_path.exists():
            self.log_path.unlink()

    def run(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Invoke promote_metrics.py with the write side redirected."""
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--target-root",
                str(self.target_root),
                "--log-path",
                str(self.log_path),
                *args,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def load_log(self) -> dict:
        if not self.log_path.exists():
            pytest.skip(f"Promotion log missing — run {SCRIPT_PATH} first")
        return json.loads(self.log_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def promote_env(tmp_path_factory) -> _PromoteEnv:
    """Fresh tmp target tree + the FIRST promotion run against it."""
    root = tmp_path_factory.mktemp("promote_target")
    env = _PromoteEnv(root / "target_root", root / "promotion_log.json")
    env.purge()
    result = env.run()
    assert result.returncode == 0, f"promotion failed: {result.stderr}"
    env.first_log = env.load_log()
    return env


def _load_metric(env: _PromoteEnv, relpath: str) -> dict:
    p = env.target(relpath)
    if not p.exists():
        pytest.skip(f"Promoted metric {relpath} missing")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Smoke
# ---------------------------------------------------------------------------


def test_promote_runs(tmp_path) -> None:
    """Running the script exits 0 and writes the log."""
    env = _PromoteEnv(tmp_path / "target_root", tmp_path / "promotion_log.json")
    env.purge()
    result = env.run()
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert env.log_path.exists(), f"{env.log_path} not written"


def test_promote_at_least_5_new_files(promote_env: _PromoteEnv) -> None:
    """Wf-87 P2 verify gate: ≥5 NEW central metric files promoted."""
    log = promote_env.first_log
    assert (
        log["promoted_count"] >= 5
    ), f"expected ≥5 promoted, got {log['promoted_count']}"
    assert len(log["new_files"]) >= 5, log["new_files"]


def test_promote_no_collisions_on_first_run(promote_env: _PromoteEnv) -> None:
    """First run should ship zero collisions — every target is new."""
    log = promote_env.first_log
    assert log["collisions"] == [], log["collisions"]


def test_promote_existing_target_preserved(promote_env: _PromoteEnv) -> None:
    """Second run without --force preserves existing central files (collision detection)."""
    new_files_first = list(promote_env.first_log["new_files"])
    if not new_files_first:
        pytest.skip("No new files from first run; can't test collision")
    target = new_files_first[0]
    original_size = promote_env.target(target).stat().st_size

    # Run again without --force: existing files should NOT be overwritten.
    result = promote_env.run()
    assert result.returncode == 0, result.stderr
    log_second = promote_env.load_log()
    collision_targets = {c["target"] for c in log_second["collisions"]}
    assert (
        target in collision_targets
    ), f"Expected collision on {target} on second run; got {log_second['collisions']}"
    # The file on disk should be unchanged in size (no overwrite).
    assert (
        promote_env.target(target).stat().st_size == original_size
    ), f"{target} was overwritten without --force"


# ---------------------------------------------------------------------------
# 2. Per-promoted-file content checks
# ---------------------------------------------------------------------------


def test_promote_hilbert_4cell_r2_promoted(promote_env: _PromoteEnv) -> None:
    """Wf-83 4-cell combined R² ships with per-cell-line breakdown + MEASURED status."""
    data = _load_metric(promote_env, "metrics/by_metric/hilbert_4cell_r2.json")
    assert data["metric"] == "hilbert_4cell_r2"
    assert data["status"] in {"MEASURED", "MEASURED_BELOW_GATE"}, data["status"]
    assert data["n_cell_lines"] == 4
    cells = set(data["per_cell_line"].keys())
    assert cells == {"mcf7", "a549", "a2780", "a2780cis"}, cells
    # Each cell-line entry should carry combined_r2_mean + n_seeds.
    for cell, row in data["per_cell_line"].items():
        assert "combined_r2_mean" in row, (cell, row)
        assert "n_seeds" in row, (cell, row)
        assert isinstance(row["combined_r2_mean"], (int, float)), (cell, row)
    # Aggregate should be the mean of the 4 cell-line means.
    means = [r["combined_r2_mean"] for r in data["per_cell_line"].values()]
    assert abs(data["aggregate_combined_r2_mean"] - sum(means) / 4) < 1e-9


def test_promote_metalcytotox_combined_r2_promoted(promote_env: _PromoteEnv) -> None:
    """MetalCytoToxDB combined R² ships with per-method breakdown."""
    data = _load_metric(promote_env, "metrics/by_metric/metalcytotox_combined_r2.json")
    assert data["metric"] == "metalcytotox_combined_r2"
    assert data["status"] == "MEASURED", data["status"]
    assert data["n_methods"] >= 1
    assert "per_method" in data
    methods = set(data["per_method"].keys())
    assert "wf77_p2_astr_baseline/metalcytotox_5metal" in methods, methods
    # Wf-77 baseline should carry a real numeric value (not all-null).
    baseline = data["per_method"]["wf77_p2_astr_baseline/metalcytotox_5metal"]
    assert isinstance(baseline["overall_r2_mean"], (int, float)), baseline


def test_promote_metalcytotox_5metal_r2_promoted(promote_env: _PromoteEnv) -> None:
    """Wf-86 P1 — MetalCytoToxDB 5-metal per-metal combined R² ships
    with 5 metal breakdown + strict-scaffold flag + per-metal sources."""
    p = promote_env.target("metrics/by_metric/metalcytotox_5metal_r2.json")
    if not p.exists():
        pytest.skip(
            "metalcytotox_5metal_r2.json missing — Wf-86 P1 run not "
            "yet complete (skip until wf86_p1_metalcytotox_real/{metal}/per_seed_metrics.json exists)"
        )
    data = _load_metric(promote_env, "metrics/by_metric/metalcytotox_5metal_r2.json")
    assert data["metric"] == "metalcytotox_5metal_r2"
    assert data["status"] in {"MEASURED", "MEASURED_BELOW_GATE"}, data["status"]
    assert data["strict_scaffold"] is True
    assert data["n_metals"] == 5
    assert sorted(data["aggregate_metals"]) == ["Ir", "Os", "Re", "Rh", "Ru"]
    # Each metal entry has a real numeric combined_r2_mean.
    for metal, entry in data["per_metal"].items():
        assert isinstance(entry["combined_r2_mean"], (int, float)), entry
        assert entry["n_seeds"] == 5, entry
    # Aggregate should be the mean of the 5 metal means.
    expected = sum(e["combined_r2_mean"] for e in data["per_metal"].values()) / 5.0
    assert (
        abs(data["aggregate_combined_r2_mean"] - expected) < 1e-9
    ), f"aggregate {data['aggregate_combined_r2_mean']} != mean {expected}"


def test_promote_tmqm_combined_r2_promoted(promote_env: _PromoteEnv) -> None:
    """tmqm-dblock + cross-dataset summary both ship."""
    data = _load_metric(promote_env, "metrics/by_metric/tmqm_combined_r2.json")
    assert data["metric"] == "tmqm_combined_r2"
    assert data["status"] == "MEASURED", data["status"]
    assert data["n_cohorts_total"] >= 29, data["n_cohorts_total"]
    # per_dataset block carries the tmqm_dblock detail.
    assert "tmqm_dblock" in data["per_dataset"], list(data["per_dataset"].keys())
    ds = data["per_dataset"]["tmqm_dblock"]
    assert ds["n_cohorts"] == 29
    assert isinstance(ds["macro_r2_mean"], (int, float)), ds
    # cross_dataset block carries the Wf-71 P3 summary.
    assert "cross_dataset" in data, list(data.keys())
    assert data["cross_dataset"]["n_datasets"] == 3
    assert isinstance(data["cross_dataset_macro_r2_mean"], (int, float))


def test_promote_platinai_combined_r2_promoted(promote_env: _PromoteEnv) -> None:
    """PlatinAI 4-cell baseline ASTSR R² value is preserved verbatim."""
    data = _load_metric(promote_env, "metrics/by_metric/platinai_combined_r2.json")
    assert data["metric"] == "platinai_combined_r2"
    assert data["status"] == "MEASURED", data["status"]
    assert isinstance(data["baseline_astr_r2_mean"], (int, float)), data
    # Wf-77 baseline ASTSR value should match the source JSON
    # (wf77_p2_astr_baseline/platinai_4cell/per_seed_metrics.json
    #  overall_r2_mean = -0.4027960052271771).
    assert abs(data["baseline_astr_r2_mean"] - (-0.4027960052271771)) < 1e-9, data[
        "baseline_astr_r2_mean"
    ]
    # n_methods should be 3 (hilbert baseline + astr baseline + 3-way pll).
    assert data["n_methods"] == 3, data["n_methods"]


def test_promote_ood_per_metal_promoted(promote_env: _PromoteEnv) -> None:
    """Wf-73 P2 per-precious-metal OOD reachability ships with 5 metals."""
    data = _load_metric(promote_env, "metrics/by_metric/ood_per_metal_reachability.json")
    assert data["metric"] == "ood_per_metal_reachability"
    assert data["status"] == "MEASURED", data["status"]
    assert data["n_metals"] == 5
    assert set(data["per_metal"].keys()) == {"Ru", "Ir", "Os", "Rh", "Re"}, data[
        "per_metal"
    ].keys()
    # Each metal row carries ood_pass_rate (0-1) + n_samples.
    for metal, row in data["per_metal"].items():
        assert 0.0 <= row["ood_pass_rate"] <= 1.0, (metal, row)
        assert row["n_samples"] >= 1, (metal, row)


def test_promote_ood_per_element_promoted(promote_env: _PromoteEnv) -> None:
    """Wf-73 P3 per-d-block-element OOD reachability ships with 5 elements + gate."""
    data = _load_metric(
        promote_env, "metrics/by_metric/ood_per_element_reachability.json"
    )
    assert data["metric"] == "ood_per_element_reachability"
    assert data["status"] == "MEASURED", data["status"]
    assert data["gate_threshold"] == 0.4, data["gate_threshold"]
    assert data["n_elements"] == 5
    assert set(data["per_element"].keys()) == {"Co", "Ni", "Cu", "Au", "Hg"}, data[
        "per_element"
    ].keys()
    for elem, row in data["per_element"].items():
        assert 0.0 <= row["ood_pass_rate"] <= 1.0, (elem, row)
        # gate_pass is a boolean derived from ood_pass_rate vs gate.
        assert isinstance(row["gate_pass"], bool), (elem, row)


def test_promote_dmpnn_ckpt_status_promoted(promote_env: _PromoteEnv) -> None:
    """D-MPNN checkpoint inventory ships with n_checkpoints >= 0."""
    data = _load_metric(promote_env, "metrics/by_metric/dmpnn_ckpt_status.json")
    assert data["metric"] == "dmpnn_ckpt_status"
    assert data["status"] == "MEASURED", data["status"]
    assert isinstance(data["n_checkpoints"], int)
    assert data["n_checkpoints"] >= 0
    assert isinstance(data["checkpoints"], list)
    # Each checkpoint row carries filename + size_bytes + mtime_utc.
    for ckpt in data["checkpoints"]:
        assert "filename" in ckpt
        assert "size_bytes" in ckpt
        assert "mtime_utc" in ckpt


def test_promote_vina_status_promoted(promote_env: _PromoteEnv) -> None:
    """Vina status ships with meeko + vina CLI probe results + last measured value."""
    data = _load_metric(promote_env, "metrics/by_metric/vina_status.json")
    assert data["metric"] == "vina_status"
    assert data["status"] == "MEASURED", data["status"]
    assert isinstance(data["meeko_installed"], bool)
    assert isinstance(data["vina_binary_available"], bool)
    assert isinstance(data["n_measurements"], int)
    # Either last_real_vina_value_kcal_per_mol is null (no real Vina yet)
    # or a numeric kcal/mol value.
    last = data["last_real_vina_value_kcal_per_mol"]
    assert last is None or isinstance(last, (int, float)), last


# ---------------------------------------------------------------------------
# 3. Honest-framing + content sanity
# ---------------------------------------------------------------------------


def test_promote_honest_framing_in_promoted_files(promote_env: _PromoteEnv) -> None:
    """Every Wf-87 P2 promoted file contains honest_framing_note + status fields.

    We probe all 8 known target files directly so the test is robust to
    ordering (some tests force collisions that move files out of new_files).
    """
    targets = [
        "metrics/by_metric/hilbert_4cell_r2.json",
        "metrics/by_metric/platinai_combined_r2.json",
        "metrics/by_metric/metalcytotox_combined_r2.json",
        "metrics/by_metric/tmqm_combined_r2.json",
        "metrics/by_metric/ood_per_metal_reachability.json",
        "metrics/by_metric/ood_per_element_reachability.json",
        "metrics/by_metric/dmpnn_ckpt_status.json",
        "metrics/by_metric/vina_status.json",
    ]
    existing = [t for t in targets if promote_env.target(t).exists()]
    assert len(existing) >= 5, existing
    for relpath in existing:
        data = _load_metric(promote_env, relpath)
        assert "status" in data, f"{relpath} missing status"
        assert "honest_framing_note" in data, f"{relpath} missing honest_framing_note"


def test_promote_no_projected_in_central(promote_env: _PromoteEnv) -> None:
    """No PROJECTED tokens leak into central metric files."""
    for relpath in promote_env.first_log["new_files"]:
        p = promote_env.target(relpath)
        text = p.read_text(encoding="utf-8")
        # Allow "PROJECTED" inside honest_framing_note strings only.
        # We don't expect any since we only promote MEASURED-on-disk files.
        # A real PROJECTED token would appear as '"status": "PROJECTED"'.
        assert (
            '"status": "PROJECTED"' not in text
        ), f"{relpath} leaks a PROJECTED status into central metrics"


def test_promote_promotion_log_present(promote_env: _PromoteEnv) -> None:
    """promotion_log.json ships with the required summary fields."""
    log = promote_env.first_log
    for key in (
        "workflow",
        "date",
        "constraint",
        "promoter_count",
        "promoted_count",
        "skipped_count",
        "error_count",
        "new_files",
        "collisions",
        "rows",
    ):
        assert key in log, f"promotion_log missing {key}"
    assert log["promoter_count"] >= 5
    assert len(log["rows"]) == log["promoter_count"]


# ---------------------------------------------------------------------------
# 4. CLI flag behaviour
# ---------------------------------------------------------------------------


def test_promote_dry_run_does_not_write(promote_env: _PromoteEnv) -> None:
    """--dry-run must NOT modify any central metric files."""
    # Snapshot one of the existing files (in the tmp target tree).
    sample = "metrics/by_metric/dmpnn_ckpt_status.json"
    p = promote_env.target(sample)
    if not p.exists():
        pytest.skip(f"{sample} missing; can't test --dry-run")
    pre_text = p.read_text(encoding="utf-8")

    result = promote_env.run("--dry-run")
    assert result.returncode == 0, result.stderr

    post_text = p.read_text(encoding="utf-8")
    assert pre_text == post_text, f"--dry-run modified {sample}"

    # The promotion_log.json should still be written (so reviewers see
    # what *would* have happened).
    assert promote_env.log_path.exists()


def test_promote_force_overwrites(promote_env: _PromoteEnv) -> None:
    """--force must overwrite an existing central file (collision bypass)."""
    sample = "metrics/by_metric/dmpnn_ckpt_status.json"
    p = promote_env.target(sample)
    if not p.exists():
        pytest.skip(f"{sample} missing; can't test --force")

    result = promote_env.run("--force")
    assert result.returncode == 0, result.stderr
    log = promote_env.load_log()
    assert (
        sample in log["new_files"]
    ), f"--force should have listed {sample} as a new file write"


# ---------------------------------------------------------------------------
# 5. Wf-A123 — the real published tree must be untouched by this module
# ---------------------------------------------------------------------------


def test_promote_never_touches_real_published_tree() -> None:
    """The real metrics/by_metric/ tree is byte-identical after this module ran.

    Before Wf-A123 this module purged nine real files and re-promoted
    older content over them, so a plain ``pytest`` run rewrote published
    artifacts (MEASURED: the 8 committed targets changed on disk, and
    ``hilbert_4cell_r2.json``'s ``aggregate_combined_r2_mean`` went from
    -5.662524090882017 to -6.712634861296495).
    """
    after = _sha256_tree(BY_METRIC_DIR)
    changed = sorted(
        k
        for k in set(_REAL_TREE_SHA_AT_IMPORT) | set(after)
        if _REAL_TREE_SHA_AT_IMPORT.get(k) != after.get(k)
    )
    assert not changed, f"published metrics/by_metric/ was modified: {changed}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
