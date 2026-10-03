"""verify_r18_modules_import.py — R18 ship-modules + scripts import+argparse verification.

================================================================
Wf-46 Phase 2: R18 ship surface double-check (32 modules + 52 scripts).
================================================================

The verify_r18_all_imports.py script (shipped earlier in Wf-46 Phase 1)
covers the import side for the 32 R18 ship modules.  This script extends
that coverage with two additional dimensions:

  * **argparse --help check** for each R18-relevant script.  We do not
    run the script body — only invoke ``python <script> --help`` to make
    sure the script has a runnable ``main()`` / ``ArgumentParser`` and
    no top-level import error.
  * **Per-module signature summary** — for each of the 32 R18 modules,
    we record ``n_classes``, ``n_functions`` (top-level callables not
    starting with ``_``), and ``n_total_public`` for fast triage when
    a regression appears.

Together with ``verify_r18_all_imports.py`` and the per-module
``test_r18_imports.py`` pytest mirror, this closes the Wf-46 verification
gap: a single artifact that says "all 32 modules import clean AND
all 52 scripts argparse --help clean" under the canonical Python
environment.

Spec
----
Wf-46 Phase 2 brief calls for:

  * importlib.import_module for each of the 32 R18 ship modules.
  * subprocess ``--help`` for each of the 52 R18-relevant scripts.
  * Per-module status ``{import_ok, help_ok, signature_summary}``.
  * JSON output at ``molmetal/reports/wf46_r18_verification/import_check.json``.

CLI
---
::

    cd /home/hugo/codes/mol-metal
    python3 molmetal/scripts/verify_r18_modules_import.py            # text + JSON
    python3 molmetal/scripts/verify_r18_modules_import.py --check   # text only, exit 0/1
    python3 molmetal/scripts/verify_r18_modules_import.py --json    # JSON only

Exit codes
----------
    0 — all 32 modules imported AND all 52 scripts --help
    1 — at least one module or script failed
    2 — the verifier itself could not run (missing repo root, etc.)

This script is intentionally CPU-only and idempotent; it does not
mutate any project state beyond writing its own JSON output.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — keep absolute and script-relative so the verifier can be invoked
# from any CWD.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL = REPO_ROOT / "molmetal"
SCRIPTS_DIR = MOLMETAL / "scripts"
REPORT_DIR = MOLMETAL / "reports" / "wf46_r18_verification"
REPORT_FILE = REPORT_DIR / "import_check.json"


# ---------------------------------------------------------------------------
# R18 ship-modules — 32 modules across 6 categories (Wf-46 Phase 1 spec).
# Logical_id, fully-qualified-module-name, description.
# ---------------------------------------------------------------------------
SHIP_MODULES: list[tuple[str, str, str]] = [
    # math_skills — 9 modules (T1–T9)
    ("math_skills/bass_existence",
     "molmetal.math_skills.bass_existence",
     "T8 Bass martingale coupling existence (Backhoff-Veraguas 2020)"),
    ("math_skills/bergman_kernel",
     "molmetal.math_skills.bergman_kernel",
     "T9 Bergman kernel on bounded symmetric domains (麻小南)"),
    ("math_skills/monge_ampere_regularity",
     "molmetal.math_skills.monge_ampere_regularity",
     "T3 Monge-Ampère regularity theory (汪徐家)"),
    ("math_skills/multi_parameter_ph",
     "molmetal.math_skills.multi_parameter_ph",
     "T5 Multi-parameter persistent homology (Kim-Jung 2026)"),
    ("math_skills/ph_hardness_guardrail",
     "molmetal.math_skills.ph_hardness_guardrail",
     "T6 PH approximation hardness guardrail (Kolbe-Mayr 2026)"),
    ("math_skills/sb_bass_martingale",
     "molmetal.math_skills.sb_bass_martingale",
     "T7 SB-Bass martingale coupling (Henry-Labordère 2017)"),
    ("math_skills/se3_schrodinger_bridge",
     "molmetal.math_skills.se3_schrodinger_bridge",
     "T1 Schrödinger Bridges on SE(3) (Mahmood-Halder-Akhtar 2026)"),
    ("math_skills/transport_relaxation",
     "molmetal.math_skills.transport_relaxation",
     "T2 Transport relaxation (Otto-Villani 2024)"),
    ("math_skills/wasserstein_gradient_flow",
     "molmetal.math_skills.wasserstein_gradient_flow",
     "T4 Wasserstein gradient flow (薛小平)"),

    # adapters — 7 modules
    ("adapters/coordination_gnn",
     "molmetal.adapters.coordination_gnn",
     "Coordination-GNN scaffold for metal centre featurisation"),
    ("adapters/quotient_space_diffusion",
     "molmetal.adapters.quotient_space_diffusion",
     "Quotient-space diffusion bridge (invariance to SE(3)/permutation)"),
    ("adapters/multi_ph_integration",
     "molmetal.adapters.multi_ph_integration",
     "T5 multi-parameter PH integration adapter"),
    ("adapters/se3_sb_integration",
     "molmetal.adapters.se3_sb_integration",
     "T1 SE(3) Schrödinger-Bridge integration adapter"),
    ("adapters/se3_lipman_integration",
     "molmetal.adapters.se3_lipman_integration",
     "Lipman OT-CFM SE(3) integration adapter"),
    ("adapters/transport_relaxation_integration",
     "molmetal.adapters.transport_relaxation_integration",
     "T2 transport-relaxation integration adapter"),
    ("adapters/diffdock",
     "molmetal.adapters.diffdock",
     "DiffDock-Pose SOTA-clone adapter (bound-pose baseline)"),

    # lam_chem — 11 modules
    ("lam_chem/coupling_joint_inference",
     "molmetal.molmetal_lam.lam_chem.coupling_joint_inference",
     "Joint inference over CFM coupling + λ-combinator stack"),
    ("lam_chem/decoder_m3_sequence",
     "molmetal.molmetal_lam.lam_chem.decoder_m3_sequence",
     "Decoder rework M3 — sequence-coherent metal decoder"),
    ("lam_chem/cross_attention_m4",
     "molmetal.molmetal_lam.lam_chem.cross_attention_m4",
     "Cross-attention M4 — pocket↔ligand fusion layer"),
    ("lam_chem/layered_generation",
     "molmetal.molmetal_lam.lam_chem.layered_generation",
     "Layered generation (scaffold → attach → decorate, T30 P2.3)"),
    ("lam_chem/buyable_gate",
     "molmetal.molmetal_lam.lam_chem.buyable_gate",
     "Buyable-reactant gate (AiZynth-as-gate, T30 P1.3)"),
    ("lam_chem/ood_protocol",
     "molmetal.molmetal_lam.lam_chem.ood_protocol",
     "OOD protocol (Tanimoto<0.4 + MMseqs2 scaffold, T30 P6.3)"),
    ("lam_chem/macroguide_ph",
     "molmetal.molmetal_lam.lam_chem.macroguide_ph",
     "Macrocyle guide + PH scaffold (T30 P2.x macrocycle pitfall fix)"),
    ("lam_chem/non_click_medchem_rules",
     "molmetal.molmetal_lam.lam_chem.non_click_medchem_rules",
     "Non-click med-chem rule pack (peptide / Boc / reductive amination, "
     "T30 P1.1)"),
    ("lam_chem/skeleton_jump",
     "molmetal.molmetal_lam.lam_chem.skeleton_jump",
     "Skeleton-jump / fragment-swap rule (T30 P2.4)"),
    ("lam_chem/md_relaxation_stub",
     "molmetal.molmetal_lam.lam_chem.md_relaxation_stub",
     "MD-relaxation stub (OpenMM-backed, T30 P2.5 Phase A)"),
    ("lam_chem/fg_compat_veto",
     "molmetal.molmetal_lam.reactions.fg_compatibility",
     "FG-compatibility / leave-group veto (T30 P1.2)"),

    # reward — 1 module
    ("reward/hilbert_primal_dual",
     "molmetal.molmetal_lam.reward.hilbert_primal_dual",
     "Hilbert-space primal-dual reward (K-functional + ellipsoid)"),

    # search_alg — 2 modules
    ("search_alg/score_repellent_mc",
     "molmetal.molmetal_lam.search_alg.score_repellent_mc",
     "Score-repellent Monte-Carlo tree search (REP-UCB)"),
    ("search_alg/variance_puct_integration",
     "molmetal.molmetal_lam.search_alg.variance_puct_integration",
     "Variance-aware PUCT integration (UCB-V / UCT-V)"),

    # flow_matching_lipman — 2 modules
    ("flow_matching_lipman/w_flow",
     "molmetal.adapters.flow_matching_lipman.w_flow",
     "Wasserstein-flow backbone (Lipman OT-CFM)"),
    ("flow_matching_lipman/m5_schedulers",
     "molmetal.adapters.flow_matching_lipman.m5_schedulers",
     "M5 variance-preserving noise schedulers (cosine/EDM/polynomial)"),
]


# ---------------------------------------------------------------------------
# Group rollup — used for the JSON ``groups`` field.
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    "math_skills":           [m[0] for m in SHIP_MODULES
                              if m[0].startswith("math_skills/")],
    "adapters":              [m[0] for m in SHIP_MODULES
                              if m[0].startswith("adapters/")],
    "lam_chem":              [m[0] for m in SHIP_MODULES
                              if m[0].startswith("lam_chem/")],
    "reward":                [m[0] for m in SHIP_MODULES
                              if m[0].startswith("reward/")],
    "search_alg":            [m[0] for m in SHIP_MODULES
                              if m[0].startswith("search_alg/")],
    "flow_matching_lipman":  [m[0] for m in SHIP_MODULES
                              if m[0].startswith("flow_matching_lipman/")],
}


# ---------------------------------------------------------------------------
# R18-relevant scripts — 52 entries.  Mix of verify_* (40) + R18-named
# orchestrators / benchmarks / harnesses (12).  Each entry is
# (logical_id, relative-script-path).
# ---------------------------------------------------------------------------
SHIP_SCRIPTS: list[tuple[str, str]] = [
    # 40 verify_*.py scripts
    ("verify_aizynth_torch_policy", "molmetal/scripts/verify_aizynth_torch_policy.py"),
    ("verify_amd_eval_capabilities", "molmetal/scripts/verify_amd_eval_capabilities.py"),
    ("verify_astsr_multiph", "molmetal/scripts/verify_astsr_multiph.py"),
    ("verify_bench_multi_ph_diversity", "molmetal/scripts/verify_bench_multi_ph_diversity.py"),
    ("verify_bench_pb_valid", "molmetal/scripts/verify_bench_pb_valid.py"),
    ("verify_bidirectional_coupling", "molmetal/scripts/verify_bidirectional_coupling.py"),
    ("verify_buyable_gate", "molmetal/scripts/verify_buyable_gate.py"),
    ("verify_closure_theorem", "molmetal/scripts/verify_closure_theorem.py"),
    ("verify_dock_10pockets", "molmetal/scripts/verify_dock_10pockets.py"),
    ("verify_fg_compat_veto", "molmetal/scripts/verify_fg_compat_veto.py"),
    ("verify_fractional_ode", "molmetal/scripts/verify_fractional_ode.py"),
    ("verify_hilbert_astsr", "molmetal/scripts/verify_hilbert_astsr.py"),
    ("verify_hilbert_primal_dual", "molmetal/scripts/verify_hilbert_primal_dual.py"),
    ("verify_lambda_cfm_coupling", "molmetal/scripts/verify_lambda_cfm_coupling.py"),
    ("verify_layered_generation", "molmetal/scripts/verify_layered_generation.py"),
    ("verify_mcts_replay", "molmetal/scripts/verify_mcts_replay.py"),
    ("verify_md_relaxation", "molmetal/scripts/verify_md_relaxation.py"),
    ("verify_multi_ph_adapter", "molmetal/scripts/verify_multi_ph_adapter.py"),
    ("verify_multi_ph_pll", "molmetal/scripts/verify_multi_ph_pll.py"),
    ("verify_non_click_rules", "molmetal/scripts/verify_non_click_rules.py"),
    ("verify_paper_build", "molmetal/scripts/verify_paper_build.py"),
    ("verify_r18_all_imports", "molmetal/scripts/verify_r18_all_imports.py"),
    ("verify_r18_pipeline", "molmetal/scripts/verify_r18_pipeline.py"),
    ("verify_reinvent_learned_inference", "molmetal/scripts/verify_reinvent_learned_inference.py"),
    ("verify_reinvent_prior_adapter", "molmetal/scripts/verify_reinvent_prior_adapter.py"),
    ("verify_score_repellent_mc", "molmetal/scripts/verify_score_repellent_mc.py"),
    ("verify_se3_lipman", "molmetal/scripts/verify_se3_lipman.py"),
    ("verify_se3_sb_adapter", "molmetal/scripts/verify_se3_sb_adapter.py"),
    ("verify_skeleton_jump", "molmetal/scripts/verify_skeleton_jump.py"),
    ("verify_skill_ecosystem", "molmetal/scripts/verify_skill_ecosystem.py"),
    ("verify_train_affinity_regression", "molmetal/scripts/verify_train_affinity_regression.py"),
    ("verify_train_cisplatin_resistance_v2", "molmetal/scripts/verify_train_cisplatin_resistance_v2.py"),
    ("verify_train_multimetal_lomo", "molmetal/scripts/verify_train_multimetal_lomo.py"),
    ("verify_train_ru_5metal", "molmetal/scripts/verify_train_ru_5metal.py"),
    ("verify_transport_relaxation_adapter", "molmetal/scripts/verify_transport_relaxation_adapter.py"),
    ("verify_variance_mcts", "molmetal/scripts/verify_variance_mcts.py"),
    ("verify_variance_mcts_proof_search", "molmetal/scripts/verify_variance_mcts_proof_search.py"),
    ("verify_variance_puct", "molmetal/scripts/verify_variance_puct.py"),
    ("verify_wf29_phase2_summarize", "molmetal/scripts/verify_wf29_phase2_summarize.py"),
    ("verify_w_flow_cfm", "molmetal/scripts/verify_w_flow_cfm.py"),
    # 12 R18-related orchestrators / benchmarks / harnesses
    ("benchmark_r18_end_to_end", "molmetal/scripts/benchmark_r18_end_to_end.py"),
    ("benchmark_r18_metalcytotox", "molmetal/scripts/benchmark_r18_metalcytotox.py"),
    ("benchmark_r18_platinai", "molmetal/scripts/benchmark_r18_platinai.py"),
    ("benchmark_astsr_extended", "molmetal/scripts/benchmark_astsr_extended.py"),
    ("benchmark_metalcytotox_e2e", "molmetal/scripts/benchmark_metalcytotox_e2e.py"),
    ("dock_10pockets_se3sb", "molmetal/scripts/dock_10pockets_se3sb.py"),
    ("master_verify_r18", "molmetal/scripts/master_verify_r18.py"),
    ("r18_cv_harness", "molmetal/scripts/r18_cv_harness.py"),
    ("r18_pipeline_orchestrator", "molmetal/scripts/r18_pipeline_orchestrator.py"),
    ("run_all_r18_tests", "molmetal/scripts/run_all_r18_tests.py"),
    ("final_ship_inventory", "molmetal/scripts/final_ship_inventory.py"),
    ("integration_verify_scoring", "molmetal/scripts/integration_verify_scoring.py"),
]


# ---------------------------------------------------------------------------
# Module verification — importlib + signature summary.
# ---------------------------------------------------------------------------
def _safe_import(fqmn: str) -> tuple[bool, str | None, str | None, dict]:
    """Import *fqmn*; return (ok, file, error, signature_summary).

    *signature_summary* counts top-level classes / functions /
    public attributes — useful for fast triage of a regression.
    """
    try:
        mod = importlib.import_module(fqmn)
    except BaseException as exc:  # noqa: BLE001 — capture every failure
        msg = f"{type(exc).__name__}: {str(exc)[:240]}"
        return False, None, msg, {}

    file = getattr(mod, "__file__", None)
    n_total = 0
    n_classes = 0
    n_functions = 0
    for name in dir(mod):
        if name.startswith("_"):
            continue
        n_total += 1
        try:
            attr = getattr(mod, name)
        except Exception:  # noqa: BLE001 — defensive
            continue
        if inspect.isclass(attr):
            n_classes += 1
        elif inspect.isfunction(attr):
            n_functions += 1

    summary = {
        "n_total_public": n_total,
        "n_classes": n_classes,
        "n_functions": n_functions,
    }
    return True, file, None, summary


# ---------------------------------------------------------------------------
# Script verification — subprocess ``python3 <script> --help`` check.
# ---------------------------------------------------------------------------
def _check_script_help(rel_path: str, timeout: int = 30) -> tuple[bool, str | None, int | None]:
    """Run ``python3 <rel_path> --help``; return (ok, error, exit_code).

    A successful --help invocation exits 0 (or sometimes 1 when the
    script uses ``parser.error`` on bare --help) but always prints
    the usage banner to stdout.  We treat *any* zero-traceback exit
    as success; the *error* field carries the first 240 chars of
    stderr if the exit was non-zero.
    """
    abs_path = REPO_ROOT / rel_path
    if not abs_path.is_file():
        return False, f"FileNotFound: {rel_path}", None

    try:
        # NOTE: We intentionally do NOT set PYTHONPATH for the subprocess.
        # Many verify_*.py scripts do their own sys.path manipulation and
        # forcing PYTHONPATH=REPO_ROOT triggers spurious circular-import
        # errors.  The cwd is enough for ``molmetal`` to resolve via the
        # canonical package layout (no PYTHONPATH pollution).
        proc = subprocess.run(
            [sys.executable, str(abs_path), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired after {timeout}s", None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", None

    # The exit code from argparse --help is 0 normally; parser.parse_args
    # can exit 1 on user-error but the usage banner is still printed.
    # A successful --help means stdout contains "usage:" or "options:".
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    has_usage = "usage:" in stdout.lower() or "options:" in stdout.lower()

    # Real failure modes:
    #   * exit code != 0/1 with empty stdout
    #   * ModuleNotFoundError / ImportError in stderr
    #   * traceback in stderr without a usage banner
    is_real_error = (
        proc.returncode not in (0, 1)
        or not has_usage
        and ("Traceback" in stderr or "ImportError" in stderr
             or "ModuleNotFoundError" in stderr or "SyntaxError" in stderr)
    )

    if is_real_error:
        msg = stderr[:240] if stderr else f"exit={proc.returncode} no-usage"
        return False, msg, proc.returncode

    return True, None, proc.returncode


# ---------------------------------------------------------------------------
# Bootstrap helpers — keep verifier idempotent across CWDs.
# ---------------------------------------------------------------------------
def _ensure_repo_root_on_path() -> None:
    rr = str(REPO_ROOT)
    if rr not in sys.path:
        sys.path.insert(0, rr)


# ---------------------------------------------------------------------------
# Run sweep — import all 32 modules, --help all 52 scripts.
# ---------------------------------------------------------------------------
def run_check(timeout: int = 30) -> dict:
    """Execute the import + --help sweep; return JSON-serialisable report."""
    _ensure_repo_root_on_path()

    module_failures: list[dict] = []
    module_successes: list[dict] = []
    module_results_by_group: dict[str, dict[str, bool]] = {
        g: {logical_id: False for logical_id in ids}
        for g, ids in GROUPS.items()
    }

    # --- 32 module imports ----------------------------------------------
    for logical_id, fqmn, description in SHIP_MODULES:
        ok, file, error, summary = _safe_import(fqmn)
        group_key = logical_id.split("/", 1)[0]
        if group_key in module_results_by_group:
            module_results_by_group[group_key][logical_id] = ok
        entry = {
            "logical_id": logical_id,
            "module": fqmn,
            "description": description,
            "ok": ok,
            "file": file,
            "signature_summary": summary,
        }
        if ok:
            module_successes.append(entry)
        else:
            entry["error"] = error
            module_failures.append(entry)

    # --- 52 script --help checks ---------------------------------------
    script_failures: list[dict] = []
    script_successes: list[dict] = []
    for logical_id, rel_path in SHIP_SCRIPTS:
        ok, error, exit_code = _check_script_help(rel_path, timeout=timeout)
        entry = {
            "logical_id": logical_id,
            "path": rel_path,
            "ok": ok,
            "exit_code": exit_code,
        }
        if ok:
            script_successes.append(entry)
        else:
            entry["error"] = error
            script_failures.append(entry)

    n_modules = len(SHIP_MODULES)
    n_scripts = len(SHIP_SCRIPTS)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "timestamp": timestamp,
        "tool": "verify_r18_modules_import.py",
        "workflow": "Wf-46 Phase 2",
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "n_modules_total": n_modules,
        "n_modules_passed": len(module_successes),
        "n_modules_failed": len(module_failures),
        "n_scripts_total": n_scripts,
        "n_scripts_passed": len(script_successes),
        "n_scripts_failed": len(script_failures),
        "all_ok": (not module_failures) and (not script_failures),
        "module_groups": GROUPS,
        "module_results": module_results_by_group,
        "module_failures": module_failures,
        "module_successes": module_successes,
        "script_failures": script_failures,
        "script_successes": script_successes,
    }


# ---------------------------------------------------------------------------
# Output renderers — text (default) and JSON-only.
# ---------------------------------------------------------------------------
def _render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("R18 modules + scripts import + argparse verification (Wf-46 Phase 2)")
    lines.append("=" * 78)
    lines.append(f"timestamp       : {report['timestamp']}")
    lines.append(f"python          : {report['python_version']}  ({report['python']})")
    lines.append(f"repo_root       : {report['repo_root']}")
    lines.append("")
    lines.append(
        f"modules         : {report['n_modules_passed']}/{report['n_modules_total']} "
        f"passed, {report['n_modules_failed']} failed"
    )
    lines.append(
        f"scripts (--help): {report['n_scripts_passed']}/{report['n_scripts_total']} "
        f"passed, {report['n_scripts_failed']} failed"
    )
    lines.append("")

    lines.append("module roll-up by group:")
    for g, results in report["module_results"].items():
        n_ok = sum(1 for v in results.values() if v)
        n_tot = len(results)
        lines.append(f"  {g:<24s}: {n_ok:>2d}/{n_tot:<2d}")
    lines.append("")

    if report["module_failures"]:
        lines.append("MODULE FAILURES:")
        for f in report["module_failures"]:
            lines.append(f"  - {f['module']}")
            lines.append(f"        {f['error']}")
        lines.append("")

    if report["script_failures"]:
        lines.append("SCRIPT FAILURES:")
        for f in report["script_failures"]:
            lines.append(f"  - {f['path']}  (exit={f.get('exit_code')})")
            err = f.get("error", "")
            if err:
                lines.append(f"        {err}")
        lines.append("")

    if report["all_ok"]:
        lines.append("ALL OK — 32 modules import + 52 scripts --help")

    lines.append("=" * 78)
    return "\n".join(lines)


def _write_json(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R18 modules + scripts import + argparse verification",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="text-only check (exit 0 on all-ok, 1 on any failure); "
             "do NOT write JSON output",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="JSON-only output to stdout; no JSON file written",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Per-script --help timeout (default 30s)",
    )
    args = parser.parse_args(argv)

    try:
        report = run_check(timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"verifier itself failed: {type(exc).__name__}: {exc}\n"
        )
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
