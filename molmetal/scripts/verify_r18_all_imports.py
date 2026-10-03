"""verify_r18_all_imports.py — R18 ship-modules import verification.

Background
----------
R18 ship wave delivered 32 modules across 5 categories:

  1. ``math_skills/``     — 9 mathematical skeleton modules (T1–T9)
  2. ``adapters/``        — 7 integration / SOTA-clone adapters
  3. ``lam_chem/``        — 11 lam_chem modules (new in R18) +
                            1 fg-compatibility veto (under
                            ``reactions/fg_compatibility.py``)
  4. ``reward/``          — 1 hilbert_primal_dual module
  5. ``search_alg/``      — 2 score_repellent + variance_puct
  6. ``flow_matching_lipman/`` — 2 w_flow + m5_schedulers

Each module has been individually smoke-tested during R18 development
but there is no single artifact that says "all 32 import clean under
the canonical Python environment".  This script closes that gap with
two layers:

  * **Import-time check** — for each module, attempt ``importlib.import_module``.
    On success, record the resolved module file path + N public attributes.
    On failure, record the exception class + first 240 chars of message.

  * **Output artefact** — emit a machine-readable JSON at
    ``molmetal/reports/wf_final_verification/import_check.json`` with
    summary fields suitable for wf_dashboard / paper-claim verification.

CLI::

    cd /home/hugo/codes/mol-metal
    python3 molmetal/scripts/verify_r18_all_imports.py            # text + JSON
    python3 molmetal/scripts/verify_r18_all_imports.py --check   # text only, exit 0/1
    python3 molmetal/scripts/verify_r18_all_imports.py --json    # JSON only

Exit codes:
    0 — all modules imported successfully
    1 — at least one module failed to import
    2 — the verifier itself could not run (missing repo root, etc.)

This script is intentionally CPU-only and idempotent; it does not
mutate any project state beyond writing its own JSON output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — keep absolute and script-relative so the verifier can be invoked
# from any CWD.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL = REPO_ROOT / "molmetal"
REPORT_DIR = MOLMETAL / "reports" / "wf_final_verification"
REPORT_FILE = REPORT_DIR / "import_check.json"


# ---------------------------------------------------------------------------
# Module catalogue — the canonical list of R17+R18 ship modules.
#
# Each entry is (logical_id, fully-qualified-module-name, description).
# The logical_id is the human-readable key used in JSON output and in the
# pytest mirror (test_r18_imports.py).  Description is rendered into the
# JSON ``groups`` mapping and helps a reader skim the manifest.
# ---------------------------------------------------------------------------
SHIP_MODULES: list[tuple[str, str, str]] = [
    # ------------------------------------------------------------------
    # math_skills/ — 9 mathematical skeleton modules (T1–T9)
    # ------------------------------------------------------------------
    (
        "math_skills/bass_existence",
        "molmetal.math_skills.bass_existence",
        "T8 Bass martingale coupling existence (Backhoff-Veraguas 2020)",
    ),
    (
        "math_skills/bergman_kernel",
        "molmetal.math_skills.bergman_kernel",
        "T9 Bergman kernel on bounded symmetric domains (麻小南)",
    ),
    (
        "math_skills/monge_ampere_regularity",
        "molmetal.math_skills.monge_ampere_regularity",
        "T3 Monge-Ampère regularity theory (汪徐家)",
    ),
    (
        "math_skills/multi_parameter_ph",
        "molmetal.math_skills.multi_parameter_ph",
        "T5 Multi-parameter persistent homology (Kim-Jung 2026)",
    ),
    (
        "math_skills/ph_hardness_guardrail",
        "molmetal.math_skills.ph_hardness_guardrail",
        "T6 PH approximation hardness guardrail (Kolbe-Mayr 2026)",
    ),
    (
        "math_skills/sb_bass_martingale",
        "molmetal.math_skills.sb_bass_martingale",
        "T7 SB-Bass martingale coupling (Henry-Labordère 2017)",
    ),
    (
        "math_skills/se3_schrodinger_bridge",
        "molmetal.math_skills.se3_schrodinger_bridge",
        "T1 Schrödinger Bridges on SE(3) (Mahmood-Halder-Akhtar 2026)",
    ),
    (
        "math_skills/transport_relaxation",
        "molmetal.math_skills.transport_relaxation",
        "T2 Transport relaxation (Otto-Villani 2024)",
    ),
    (
        "math_skills/wasserstein_gradient_flow",
        "molmetal.math_skills.wasserstein_gradient_flow",
        "T4 Wasserstein gradient flow (薛小平)",
    ),
    # ------------------------------------------------------------------
    # adapters/ — integration + SOTA-clone + coordination
    # ------------------------------------------------------------------
    (
        "adapters/coordination_gnn",
        "molmetal.adapters.coordination_gnn",
        "Coordination-GNN scaffold for metal centre featurisation",
    ),
    (
        "adapters/quotient_space_diffusion",
        "molmetal.adapters.quotient_space_diffusion",
        "Quotient-space diffusion bridge (invariance to SE(3)/permutation)",
    ),
    (
        "adapters/multi_ph_integration",
        "molmetal.adapters.multi_ph_integration",
        "T5 multi-parameter PH integration adapter",
    ),
    (
        "adapters/se3_sb_integration",
        "molmetal.adapters.se3_sb_integration",
        "T1 SE(3) Schrödinger-Bridge integration adapter",
    ),
    (
        "adapters/transport_relaxation_integration",
        "molmetal.adapters.transport_relaxation_integration",
        "T2 transport-relaxation integration adapter",
    ),
    (
        "adapters/diffdock",
        "molmetal.adapters.diffdock",
        "DiffDock-Pose SOTA-clone adapter (bound-pose baseline)",
    ),
    (
        "adapters/equibind",
        "molmetal.adapters.equibind",
        "EquiBind SOTA-clone adapter (regression-allowlist baseline)",
    ),
    # ------------------------------------------------------------------
    # lam_chem/ — 11 new modules in R18 (coupling, decoder, cross-attn,
    # layered-gen, buyable gate, OOD protocol, macro-pH, non-click,
    # skeleton jump, MD relaxation stub) + the fg-compat veto (under
    # reactions/fg_compatibility.py per the T30 P1.2 ship report).
    # ------------------------------------------------------------------
    (
        "lam_chem/coupling_joint_inference",
        "molmetal.molmetal_lam.lam_chem.coupling_joint_inference",
        "Joint inference over CFM coupling + λ-combinator stack",
    ),
    (
        "lam_chem/decoder_m3_sequence",
        "molmetal.molmetal_lam.lam_chem.decoder_m3_sequence",
        "Decoder rework M3 — sequence-coherent metal decoder",
    ),
    (
        "lam_chem/cross_attention_m4",
        "molmetal.molmetal_lam.lam_chem.cross_attention_m4",
        "Cross-attention M4 — pocket↔ligand fusion layer",
    ),
    (
        "lam_chem/layered_generation",
        "molmetal.molmetal_lam.lam_chem.layered_generation",
        "Layered generation (scaffold → attach → decorate, T30 P2.3)",
    ),
    (
        "lam_chem/buyable_gate",
        "molmetal.molmetal_lam.lam_chem.buyable_gate",
        "Buyable-reactant gate (AiZynth-as-gate, T30 P1.3)",
    ),
    (
        "lam_chem/ood_protocol",
        "molmetal.molmetal_lam.lam_chem.ood_protocol",
        "OOD protocol (Tanimoto<0.4 + MMseqs2 scaffold, T30 P6.3)",
    ),
    (
        "lam_chem/macroguide_ph",
        "molmetal.molmetal_lam.lam_chem.macroguide_ph",
        "Macrocyle guide + PH scaffold (T30 P2.x macrocycle pitfall fix)",
    ),
    (
        "lam_chem/non_click_medchem_rules",
        "molmetal.molmetal_lam.lam_chem.non_click_medchem_rules",
        "Non-click med-chem rule pack (peptide / Boc / reductive amination, "
        "T30 P1.1)",
    ),
    (
        "lam_chem/skeleton_jump",
        "molmetal.molmetal_lam.lam_chem.skeleton_jump",
        "Skeleton-jump / fragment-swap rule (T30 P2.4)",
    ),
    (
        "lam_chem/md_relaxation_stub",
        "molmetal.molmetal_lam.lam_chem.md_relaxation_stub",
        "MD-relaxation stub (OpenMM-backed, T30 P2.5 Phase A)",
    ),
    (
        "lam_chem/fg_compat_veto",
        "molmetal.molmetal_lam.reactions.fg_compatibility",
        "FG-compatibility / leave-group veto (T30 P1.2, lives under "
        "reactions/ but counted in lam_chem ship manifest)",
    ),
    # ------------------------------------------------------------------
    # reward/ — Hilbert-space primal-dual reward head
    # ------------------------------------------------------------------
    (
        "reward/hilbert_primal_dual",
        "molmetal.molmetal_lam.reward.hilbert_primal_dual",
        "Hilbert-space primal-dual reward (K-functional + ellipsoid)",
    ),
    # ------------------------------------------------------------------
    # search_alg/ — score-repellent MC + variance-aware PUCT
    # ------------------------------------------------------------------
    (
        "search_alg/score_repellent_mc",
        "molmetal.molmetal_lam.search_alg.score_repellent_mc",
        "Score-repellent Monte-Carlo tree search (REP-UCB)",
    ),
    (
        "search_alg/variance_puct_integration",
        "molmetal.molmetal_lam.search_alg.variance_puct_integration",
        "Variance-aware PUCT integration (UCB-V / UCT-V)",
    ),
    # ------------------------------------------------------------------
    # flow_matching_lipman/ — w_flow + m5_schedulers
    # ------------------------------------------------------------------
    (
        "flow_matching_lipman/w_flow",
        "molmetal.adapters.flow_matching_lipman.w_flow",
        "Wasserstein-flow backbone (Lipman OT-CFM)",
    ),
    (
        "flow_matching_lipman/m5_schedulers",
        "molmetal.adapters.flow_matching_lipman.m5_schedulers",
        "M5 variance-preserving noise schedulers (cosine/EDM/polynomial)",
    ),
]


# ---------------------------------------------------------------------------
# Group rollup — used for the JSON ``groups`` field and the test counts.
# The order mirrors the spec; counts must match the SHIP_MODULES list.
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "math_skills": [m[0] for m in SHIP_MODULES if m[0].startswith("math_skills/")],
    "adapters": [m[0] for m in SHIP_MODULES if m[0].startswith("adapters/")],
    "lam_chem": [m[0] for m in SHIP_MODULES if m[0].startswith("lam_chem/")],
    "reward": [m[0] for m in SHIP_MODULES if m[0].startswith("reward/")],
    "search_alg": [m[0] for m in SHIP_MODULES if m[0].startswith("search_alg/")],
    "flow_matching_lipman": [
        m[0] for m in SHIP_MODULES if m[0].startswith("flow_matching_lipman/")
    ],
}


# ---------------------------------------------------------------------------
# Core verification — try-import each module, capture diagnostic on failure.
# ---------------------------------------------------------------------------
def _safe_import(fqmn: str) -> tuple[bool, str | None, str | None, int | None]:
    """Import *fqmn*; return (ok, file, error, n_public).

    *ok* is True iff the import succeeded.
    *file* is the resolved ``__file__`` (or None on failure).
    *error* is the formatted exception (or None on success).
    *n_public* is the count of public attributes (no leading underscore),
    None on failure.
    """
    try:
        mod = importlib.import_module(fqmn)
    except BaseException as exc:  # noqa: BLE001 — we WANT every failure mode
        msg = f"{type(exc).__name__}: {str(exc)[:240]}"
        return False, None, msg, None
    file = getattr(mod, "__file__", None)
    n_public = sum(1 for n in dir(mod) if not n.startswith("_"))
    return True, file, None, n_public


def _ensure_repo_root_on_path() -> None:
    """Insert the repo root into ``sys.path`` if missing.

    Mirrors the r10_cfg bootstrap in R10 — without this, running the
    verifier from a directory outside the project tree fails with
    ``ModuleNotFoundError: No module named 'molmetal'`` even though the
    file is part of the molmetal package.  Idempotent.
    """
    rr = str(REPO_ROOT)
    if rr not in sys.path:
        sys.path.insert(0, rr)


def _check_pth_present() -> bool:
    """Return True iff the canonical venv .pth file is present and valid.

    Defence-in-depth check; the verifier's own bootstrap does NOT depend
    on the .pth file.  This flag is surfaced in the JSON output so a
    CI runner can quickly diagnose missing-env issues.
    """
    candidates = [
        MOLMETAL.parents[0]
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "molmetal_dev.pth",
        MOLMETAL.parents[0]
        / ".venv"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "molmetal_dev.pth",
        MOLMETAL.parents[0]
        / ".venv"
        / "lib"
        / "python3.13"
        / "site-packages"
        / "molmetal_dev.pth",
    ]
    for pth in candidates:
        if pth.is_file():
            try:
                contents = pth.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if contents and Path(contents).is_dir():
                return True
    return False


def run_check() -> dict:
    """Execute the import sweep and return the JSON-serialisable report."""
    _ensure_repo_root_on_path()

    failures: list[dict] = []
    successes: list[dict] = []
    by_group: dict[str, dict[str, bool]] = {
        g: {logical_id: False for logical_id in ids} for g, ids in GROUPS.items()
    }

    for logical_id, fqmn, description in SHIP_MODULES:
        ok, file, error, n_public = _safe_import(fqmn)
        group_key = logical_id.split("/", 1)[0]
        if group_key in by_group:
            by_group[group_key][logical_id] = ok
        entry = {
            "module": fqmn,
            "logical_id": logical_id,
            "description": description,
            "ok": ok,
            "file": file,
            "n_public": n_public,
        }
        if ok:
            successes.append(entry)
        else:
            entry["error"] = error
            failures.append(entry)

    n_total = len(SHIP_MODULES)
    n_passed = len(successes)
    n_failed = len(failures)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "timestamp": timestamp,
        "tool": "verify_r18_all_imports.py",
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "n_modules_total": n_total,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "all_ok": n_failed == 0,
        "pth_file_present": _check_pth_present(),
        "groups": GROUPS,
        "group_results": by_group,
        "failures": failures,
        "successes": successes,
    }


# ---------------------------------------------------------------------------
# Output renderers — text (default) and JSON-only.
# ---------------------------------------------------------------------------
def _render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("R18 ship-modules import verification")
    lines.append("=" * 72)
    lines.append(f"timestamp       : {report['timestamp']}")
    lines.append(f"python          : {report['python_version']}  ({report['python']})")
    lines.append(f"repo_root       : {report['repo_root']}")
    lines.append(f"pth file present: {report['pth_file_present']}")
    lines.append("")
    lines.append(
        f"summary         : {report['n_passed']}/{report['n_modules_total']}"
        f" passed, {report['n_failed']} failed"
    )
    lines.append("")
    lines.append("by group:")
    for g, results in report["group_results"].items():
        n_ok = sum(1 for v in results.values() if v)
        n_tot = len(results)
        lines.append(f"  {g:<24s}: {n_ok:>2d}/{n_tot:<2d}")
    lines.append("")

    if report["failures"]:
        lines.append("FAILURES:")
        for f in report["failures"]:
            lines.append(f"  - {f['module']}")
            lines.append(f"        {f['error']}")
        lines.append("")
    else:
        lines.append("all 32 ship modules import OK")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _write_json(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R18 ship-modules import verification",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="text-only check (exit 0 on all-ok, 1 on any failure); "
        "do NOT write JSON output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON-only output to stdout; no JSON file written",
    )
    args = parser.parse_args(argv)

    try:
        report = run_check()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"verifier itself failed: {type(exc).__name__}: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.check:
        print(_render_text(report))
    else:
        # default: text + JSON file
        print(_render_text(report))
        _write_json(report)
        print(f"JSON written: {REPORT_FILE}")

    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
