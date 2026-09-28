"""collect_all_tests.py — read-only pytest collection + coverage report.

Background
----------
The R17/R18 ship waves delivered ~70 new test files across six families
(math_skills, math_skill adapters, lam_chem ship, CFM pipeline rescue +
fixes, P2 ML baselines, docking / pocket-macro).  Each ship module ships
with one or more parametrised pytest tests, but there is no single
artefact that says:

  * how many test files exist,
  * how many test functions/classes those files contain,
  * how many are explicitly marked ``skip`` / ``skipif`` / ``xfail``,
  * how each ship wave (R17, R18) contributes,
  * which tests target which module family.

This script closes that gap with a *read-only, AST-only* collector.  It
does **not** invoke ``pytest`` itself — that would require torch + numpy
+ RDKit + all of the project's GPU/CPU optional deps, which is
deliberately out of scope for a static-inventory script.  Instead it:

  1. globs ``molmetal/tests/test_*.py`` (the canonical pytest pattern),
  2. parses each file with ``ast`` to count ``test_*`` functions and
     ``Test*`` classes,
  3. greps for ``@pytest.mark.skip`` / ``skipif`` / ``xfail`` decorators
     to estimate the expected-skip / expected-fail population,
  4. classifies each file into one of the ship-wave families based on
     the canonical ``SHIP_MODULES`` table maintained in
     ``verify_r18_all_imports.py`` plus a curated filename table for the
     R17 + docking + P2 ML categories, and
  5. emits a JSON inventory + a Markdown summary at
     ``molmetal/reports/wf_final_verification/``.

CLI::

    cd /home/hugo/codes/mol-metal
    python3 molmetal/scripts/collect_all_tests.py            # text + JSON + MD
    python3 molmetal/scripts/collect_all_tests.py --json    # JSON only (stdout)
    python3 molmetal/scripts/collect_all_tests.py --md      # Markdown only (stdout)
    python3 molmetal/scripts/collect_all_tests.py --check   # text only, exit 0/1

Exit codes:
    0 — collection succeeded (always; no actual pytest is run).
    2 — collector itself could not run (missing repo root, etc.).

The script is CPU-only, AST-only, idempotent, and never mutates
anything outside its own output directory.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths — keep absolute and script-relative.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MOLMETAL = REPO_ROOT / "molmetal"
TESTS_DIR = MOLMETAL / "tests"
REPORT_DIR = MOLMETAL / "reports" / "wf_final_verification"
JSON_OUT = REPORT_DIR / "test_inventory.json"
MD_OUT = REPORT_DIR / "test_inventory.md"


# ---------------------------------------------------------------------------
# Canonical ship-module classification.
#
# This table mirrors verify_r18_all_imports.SHIP_MODULES plus the
# extensions needed by the test inventory (R17 + docking + P2 ML).
# Each entry is (category_id, test_filename_stem, ship_wave, family).
# ---------------------------------------------------------------------------
# Ship wave families:
#   r18_math_skills      — 9 mathematical skeleton modules (T1–T9)
#   r18_math_adapters    — integration / SOTA-clone adapters
#   r18_lam_chem         — 11 new lam_chem ship modules + fg-compat veto
#   r18_search_alg       — score_repellent + variance_puct
#   r18_reward           — hilbert_primal_dual
#   r18_flow_lipman      — w_flow + m5_schedulers
#   r18_cfm_rescue       — 5 cfm_rescue_* tests + 5 cfm_fix* tests
#   r17_legacy           — R17 vocabulary M15 + reproduce_R17
#   p2_ml_baselines      — Krasnov 2026 XGBoost/RF/Morgan FP reproductions
#   p2_ml_metal_hybrid   — multi-task metal hybrid training scripts
#   p2_ml_pretrain       — TMQM / AiZynth / retrosynthesis pretraining
#   docking_vina         — Vina + DiffDock + FlowDock + PoseBusters adapters
#   pitfall_guards       — ship rule registry + veto + guard JSON
#   layer_metrics        — goverened layer-metrics L1..L9
#   integration_e2e      — e2e + integration + design_loop + full_pipeline_demo
#   misc                 — everything else (deflex wireup phases, generate_*
#                          scripts, etc.)
# ---------------------------------------------------------------------------
R18_SHIP_TEST_STEMS: dict[str, tuple[str, str]] = {
    # math_skills T1..T9
    "bass_existence":                   ("r18_math_skills", "T8 Bass martingale coupling"),
    "bergman_kernel":                   ("r18_math_skills", "T9 Bergman kernel BSD"),
    "monge_ampere_regularity":          ("r18_math_skills", "T3 Monge-Ampere regularity"),
    "multi_parameter_ph":               ("r18_math_skills", "T5 multi-parameter PH"),
    "ph_hardness_guardrail":            ("r18_math_skills", "T6 PH hardness guardrail"),
    "sb_bass_martingale":               ("r18_math_skills", "T7 SB-Bass martingale"),
    "se3_schrodinger_bridge":           ("r18_math_skills", "T1 SE(3) Schrodinger bridge"),
    "transport_relaxation":             ("r18_math_skills", "T2 transport relaxation"),
    "wasserstein_gradient_flow":        ("r18_math_skills", "T4 Wasserstein gradient flow"),
    # math_skill adapters
    "coordination_gnn":                 ("r18_math_adapters", "Coordination-GNN scaffold"),
    "quotient_space_diffusion":         ("r18_math_adapters", "Quotient-space diffusion"),
    "multi_ph_integration":             ("r18_math_adapters", "T5 multi-PH integration"),
    "se3_sb_integration":               ("r18_math_adapters", "T1 SE(3)-SB integration"),
    "transport_relaxation_integration": ("r18_math_adapters", "T2 transport-relax integration"),
    "hilbert_primal_dual":              ("r18_reward", "Hilbert primal-dual reward"),
    "hilbert_score_repellent_integration": ("r18_math_adapters", "Hilbert + score-repellent"),
    "score_repellent_mc":               ("r18_search_alg", "Score-repellent MCTS"),
    "variance_puct_integration":        ("r18_search_alg", "Variance-aware PUCT"),
    "w_flow":                           ("r18_flow_lipman", "Wasserstein-flow backbone"),
    "m5_schedulers":                    ("r18_flow_lipman", "M5 noise schedulers"),
    # lam_chem ship modules
    "coupling_adapter":                 ("r18_lam_chem", "Coupling adapter"),
    "coupling_adapter_platinai5k":      ("r18_lam_chem", "PlatinAI-5k coupling adapter"),
    "lambda_astsr":                     ("r18_lam_chem", "Lambda ASTSR"),
    "lambda_combinators_expansion":     ("r18_lam_chem", "Lambda combinators expansion"),
    "lambda_measured_reward_dispatch":  ("r18_lam_chem", "Lambda measured-reward dispatch"),
    "lambda_reward_coupling":           ("r18_lam_chem", "Lambda reward coupling"),
    "lambda_cfm_integration":           ("r18_lam_chem", "Lambda-CFM integration"),
    "lambda_cfm_joint_inference":       ("r18_lam_chem", "Lambda-CFM joint inference"),
    "cross_attention_m4":               ("r18_lam_chem", "Cross-attention M4 fusion"),
    "cross_attention_fusion":           ("r18_lam_chem", "Cross-attention fusion (v0)"),
    "cross_attention_v3":               ("r18_lam_chem", "Cross-attention v3"),
    "decoder_m3_sequence":              ("r18_lam_chem", "Decoder M3 sequence"),
    "layered_generation":               ("r18_lam_chem", "Layered generation"),
    "buyable_gate":                     ("r18_lam_chem", "Buyable reactant gate"),
    "ood_protocol":                     ("r18_lam_chem", "OOD protocol scaffold"),
    "macroguide_ph":                    ("r18_lam_chem", "Macrocycle guide + PH"),
    "non_click_medchem_rules":          ("r18_lam_chem", "Non-click med-chem rules"),
    "skeleton_jump":                    ("r18_lam_chem", "Skeleton-jump fragment swap"),
    "md_relaxation_stub":               ("r18_lam_chem", "MD relaxation stub"),
    "pt_click_compat_smoke":            ("r18_lam_chem", "PT click compat smoke"),
    "pt_metal_ligand_exchange_known":   ("r18_lam_chem", "PT metal-ligand exchange known"),
    "click_rule_effect_size_study":     ("r18_lam_chem", "Click rule effect size study"),
    # CFM pipeline rescue + fixes (the 10 R18 ship CFM hardening tests)
    "cfm_rescue_bondhead_default":      ("r18_cfm_rescue", "CFM rescue: bondhead default"),
    "cfm_rescue_data_scale":            ("r18_cfm_rescue", "CFM rescue: data scale"),
    "cfm_rescue_decode_smoke":          ("r18_cfm_rescue", "CFM rescue: decode smoke"),
    "cfm_rescue_ode_solver":           ("r18_cfm_rescue", "CFM rescue: ODE solver"),
    "cfm_rescue_yuelbond":              ("r18_cfm_rescue", "CFM rescue: YuelBond"),
    "cfm_fix1_bond_head_default":       ("r18_cfm_rescue", "CFM fix1: bond-head default"),
    "cfm_fix2_midpoint_solver":         ("r18_cfm_rescue", "CFM fix2: midpoint solver"),
    "cfm_fix3_decode_smoke":            ("r18_cfm_rescue", "CFM fix3: decode smoke"),
    "cfm_p1_fixes":                     ("r18_cfm_rescue", "CFM P1 fixes aggregator"),
    "cfm_path_import":                  ("r18_cfm_rescue", "CFM path import guard"),
    # r18 import sweep mirror
    "r18_imports":                      ("r18_lam_chem", "R18 ship-modules import mirror"),
}

R17_SHIP_TEST_STEMS: dict[str, tuple[str, str]] = {
    "r17_legacy_compat":                ("r17_legacy", "R17 legacy compat canary"),
    "r17_paper_sweep_presets":          ("r17_legacy", "R17 paper-sweep presets"),
    "reproduce_R17":                    ("r17_legacy", "wf_reproduce_R17 end-to-end driver"),
    "atom_training_contract":           ("r17_legacy", "R17 vocabulary M15 regression pair"),
    "duplicate_tile_fix":               ("r17_legacy", "R17 duplicate-tile defensive fix"),
}

P2_ML_TEST_STEMS: dict[str, tuple[str, str]] = {
    # dmpnn
    "dmpnn":                            ("p2_ml_baselines", "D-MPNN baseline"),
    "dmpnn_attn":                       ("p2_ml_baselines", "D-MPNN attentive FP"),
    "dmpnn_baseline":                   ("p2_ml_baselines", "D-MPNN baseline (Krasnov 2026)"),
    "dmpnn_multitask":                  ("p2_ml_baselines", "D-MPNN multi-task Ru"),
    "dmpnn_scatter_wiring":             ("p2_ml_baselines", "D-MPNN scatter wiring"),
    "attentive_dmpnn_padding":          ("p2_ml_baselines", "Attentive D-MPNN padding"),
    "v4_multitask_dmpnn":               ("p2_ml_baselines", "V4 multi-task D-MPNN"),
    "baselines":                        ("p2_ml_baselines", "Baseline harness"),
    "baselines_lightgbm_lomo_temporal": ("p2_ml_baselines", "LightGBM LOMO temporal"),
    # metal_hybrid
    "metal_hybrid":                     ("p2_ml_metal_hybrid", "Metal hybrid V0"),
    "metal_hybrid_v3":                  ("p2_ml_metal_hybrid", "Metal hybrid V3"),
    "metal_hybrid_v4":                  ("p2_ml_metal_hybrid", "Metal hybrid V4"),
    "metal_smiles":                     ("p2_ml_metal_hybrid", "Metal SMILES parser"),
    "metal_ligand_exchange_matrix":     ("p2_ml_metal_hybrid", "Metal-ligand exchange matrix"),
    "metal_generator_adapter":          ("p2_ml_metal_hybrid", "Metal generator adapter"),
    "metal_geometry_known_pt":          ("p2_ml_metal_hybrid", "Metal geometry known-Pt"),
    "f2a_metal_ligand_exchange":        ("p2_ml_metal_hybrid", "F2A metal-ligand exchange"),
    "explicit_metal_prior_sampling":    ("p2_ml_metal_hybrid", "Explicit metal prior sampling"),
    # pretrain
    "tmqm_loading":                     ("p2_ml_pretrain", "TMQM loading"),
    "tmqm_pretrain":                    ("p2_ml_pretrain", "TMQM pretraining"),
    "tmqm_shape_bridge":                ("p2_ml_pretrain", "TMQM shape bridge"),
    "tmqm_wireup":                      ("p2_ml_pretrain", "TMQM wireup"),
    "aizynth_adapter":                  ("p2_ml_pretrain", "AiZynth adapter"),
    "aizynth_isolated":                 ("p2_ml_pretrain", "AiZynth isolated"),
    "aizynth_real_backend":             ("p2_ml_pretrain", "AiZynth real backend"),
    "aizynth_v4_api":                   ("p2_ml_pretrain", "AiZynth v4 API"),
    "retrosynthesis":                   ("p2_ml_pretrain", "Retrosynthesis smoke"),
}

DOCKING_TEST_STEMS: dict[str, tuple[str, str]] = {
    "vina_adapter":                     ("docking_vina", "Vina CPU adapter"),
    "vina_gpu_adapter":                 ("docking_vina", "Vina GPU adapter"),
    "vina_3engine_parity":              ("docking_vina", "Vina 3-engine parity"),
    "docking_adapters":                 ("docking_vina", "Docking adapters harness"),
    "docking_pose_chemistry":           ("docking_vina", "Docking pose chemistry"),
    "posebusters_adapter":              ("docking_vina", "PoseBusters adapter"),
    "posebusters_doc":                  ("docking_vina", "PoseBusters doc"),
    "posebusters_mmff94s_relax":        ("docking_vina", "PoseBusters MMFF94s relax"),
    "receptor_preparation_for_evaluation": ("docking_vina", "Receptor preparation"),
    "prepare_crossdocked_receptor":     ("docking_vina", "CrossDocked receptor prep"),
    "pocket_conditioned_lipman":        ("docking_vina", "Pocket-conditioned Lipman"),
    "pocket_docking_reward":            ("docking_vina", "Pocket docking reward"),
    "pocket_invariance_combined":       ("docking_vina", "Pocket invariance combined"),
    "pocket_macro_inference_path":      ("docking_vina", "Pocket macro inference"),
}

PITFALL_TEST_STEMS: dict[str, tuple[str, str]] = {
    "pitfall_guards":                   ("pitfall_guards", "Pitfall guard rules"),
    "pitfall_guard_json":               ("pitfall_guards", "Pitfall guard JSON"),
}

LAYER_METRICS_TEST_STEMS: dict[str, tuple[str, str]] = {
    "layer_metrics_l1_l3":              ("layer_metrics", "Layer metrics L1..L3"),
    "layer_metrics_l4_l6":              ("layer_metrics", "Layer metrics L4..L6"),
    "layer_metrics_l7_l8":              ("layer_metrics", "Layer metrics L7..L8"),
    "layer_metrics_l9_cross":           ("layer_metrics", "Layer metrics L9 cross-layer"),
}

INTEGRATION_TEST_STEMS: dict[str, tuple[str, str]] = {
    "e2e":                              ("integration_e2e", "End-to-end"),
    "integration":                      ("integration_e2e", "Integration smoke"),
    "design_loop":                      ("integration_e2e", "Design loop"),
    "full_pipeline_demo":               ("integration_e2e", "Full pipeline demo"),
    "hybrid":                           ("integration_e2e", "Hybrid atom+token"),
    "smoke_scripts":                    ("integration_e2e", "smoke_* script smoke"),
}

# ---------------------------------------------------------------------------
# Combined classification table — order matters (first match wins).
# ---------------------------------------------------------------------------
SHIP_CLASSIFICATION: list[tuple[str, dict[str, tuple[str, str]]]] = [
    ("R18 ship",     R18_SHIP_TEST_STEMS),
    ("R17 ship",     R17_SHIP_TEST_STEMS),
    ("P2 ML",        P2_ML_TEST_STEMS),
    ("Docking",      DOCKING_TEST_STEMS),
    ("Pitfall",      PITFALL_TEST_STEMS),
    ("Layer metrics",LAYER_METRICS_TEST_STEMS),
    ("Integration",  INTEGRATION_TEST_STEMS),
]


def classify(stem: str) -> tuple[str, str, str]:
    """Return (wave, category, description) for a test stem.

    The first matching table wins (R18 takes precedence over R17, etc.).
    Anything unmatched falls into ``misc`` with a short description.
    """
    for wave, table in SHIP_CLASSIFICATION:
        if stem in table:
            cat, desc = table[stem]
            return wave, cat, desc
    return "misc", "misc", "Uncategorized test (legacy / smoke)"


# ---------------------------------------------------------------------------
# AST helpers.
# ---------------------------------------------------------------------------
_TEST_FUNC_RE = re.compile(r"@pytest\.mark\.(skip|skipif|xfail)\b")


def count_tests_in_file(filepath: Path) -> dict:
    """Return counts of test functions / classes / markers in *filepath*.

    The function is read-only and AST-only; it never imports the module
    under test (so missing torch / numpy / RDKit do not break the
    inventory).  Syntax errors degrade gracefully — the file is recorded
    as ``parse_error`` and excluded from the test totals.
    """
    try:
        src = filepath.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "tests": 0,
            "classes": 0,
            "skip": 0,
            "skipif": 0,
            "xfail": 0,
            "parse_error": f"OSError: {exc}",
        }
    # Decorator markers (regex over source — robust to multiline).
    skip = len(re.findall(r"@pytest\.mark\.skip\b(?!\s*if)", src))
    skipif = len(re.findall(r"@pytest\.mark\.skipif\b", src))
    xfail = len(re.findall(r"@pytest\.mark\.xfail\b", src))
    # AST for test function / class counts.
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {
            "tests": 0,
            "classes": 0,
            "skip": skip,
            "skipif": skipif,
            "xfail": xfail,
            "parse_error": f"SyntaxError: {exc}",
        }

    n_funcs = 0
    n_classes = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                n_funcs += 1
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                n_classes += 1
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("test_"):
                            n_funcs += 1
    return {
        "tests": n_funcs,
        "classes": n_classes,
        "skip": skip,
        "skipif": skipif,
        "xfail": xfail,
        "parse_error": None,
    }


# ---------------------------------------------------------------------------
# Core collector.
# ---------------------------------------------------------------------------
def collect() -> dict:
    """Walk ``tests/test_*.py`` and produce the JSON inventory."""
    files = sorted(p for p in TESTS_DIR.glob("test_*.py") if p.is_file())
    entries = []
    by_category: dict[str, dict] = defaultdict(
        lambda: {"n_files": 0, "n_tests": 0, "n_skip": 0,
                 "n_skipif": 0, "n_xfail": 0, "files": []},
    )
    by_wave: dict[str, dict] = defaultdict(
        lambda: {"n_files": 0, "n_tests": 0, "n_skip": 0,
                 "n_skipif": 0, "n_xfail": 0},
    )

    total_tests = total_skip = total_skipif = total_xfail = 0
    parse_errors = 0

    for path in files:
        stem = path.stem[len("test_"):]  # strip leading "test_"
        wave, category, description = classify(stem)
        counts = count_tests_in_file(path)
        entry = {
            "file": str(path.relative_to(REPO_ROOT)),
            "stem": stem,
            "wave": wave,
            "category": category,
            "description": description,
            "n_tests": counts["tests"],
            "n_classes": counts["classes"],
            "n_skip": counts["skip"],
            "n_skipif": counts["skipif"],
            "n_xfail": counts["xfail"],
            "parse_error": counts["parse_error"],
        }
        entries.append(entry)

        by_category[category]["n_files"] += 1
        by_category[category]["n_tests"] += counts["tests"]
        by_category[category]["n_skip"] += counts["skip"]
        by_category[category]["n_skipif"] += counts["skipif"]
        by_category[category]["n_xfail"] += counts["xfail"]
        by_category[category]["files"].append(path.name)

        by_wave[wave]["n_files"] += 1
        by_wave[wave]["n_tests"] += counts["tests"]
        by_wave[wave]["n_skip"] += counts["skip"]
        by_wave[wave]["n_skipif"] += counts["skipif"]
        by_wave[wave]["n_xfail"] += counts["xfail"]

        total_tests += counts["tests"]
        total_skip += counts["skip"]
        total_skipif += counts["skipif"]
        total_xfail += counts["xfail"]
        if counts["parse_error"] is not None:
            parse_errors += 1

    # Stable ordering for JSON output.
    categories_sorted = sorted(by_category.items(),
                                key=lambda kv: (-kv[1]["n_tests"], kv[0]))
    waves_sorted = sorted(by_wave.items(),
                           key=lambda kv: (-kv[1]["n_tests"], kv[0]))

    # Top-level "categories" rollup — a shorter, spec-aligned view that
    # the paper dashboard can consume without knowing the internal
    # wave/category two-level hierarchy.  Each entry is {name: n_tests}.
    categories_summary: dict[str, int] = {
        "math_skills":      by_category.get("r18_math_skills", {}).get("n_tests", 0),
        "adapters":         by_category.get("r18_math_adapters", {}).get("n_tests", 0),
        "search_alg":       by_category.get("r18_search_alg", {}).get("n_tests", 0),
        "reward":           by_category.get("r18_reward", {}).get("n_tests", 0),
        "flow_lipman":      by_category.get("r18_flow_lipman", {}).get("n_tests", 0),
        "lam_chem":         by_category.get("r18_lam_chem", {}).get("n_tests", 0),
        "cfm_pipeline":     by_category.get("r18_cfm_rescue", {}).get("n_tests", 0),
        "lambda_cfm":       by_category.get("r18_lam_chem", {}).get("n_tests", 0),
        "pitfall":          by_category.get("pitfall_guards", {}).get("n_tests", 0),
        "integration":      by_category.get("integration_e2e", {}).get("n_tests", 0),
        "p2_ml":            (by_category.get("p2_ml_baselines", {}).get("n_tests", 0)
                             + by_category.get("p2_ml_metal_hybrid", {}).get("n_tests", 0)
                             + by_category.get("p2_ml_pretrain", {}).get("n_tests", 0)),
        "docking":          by_category.get("docking_vina", {}).get("n_tests", 0),
        "layer_metrics":    by_category.get("layer_metrics", {}).get("n_tests", 0),
        "r17_legacy":       by_category.get("r17_legacy", {}).get("n_tests", 0),
        "misc":             by_category.get("misc", {}).get("n_tests", 0),
    }

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_r18 = by_wave.get("R18 ship", {}).get("n_files", 0)
    n_r17 = by_wave.get("R17 ship", {}).get("n_files", 0)

    return {
        "timestamp": timestamp,
        "tool": "collect_all_tests.py",
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "mode": "static-collect (no pytest execution)",
        "tests_dir": str(TESTS_DIR.relative_to(REPO_ROOT)),
        "n_test_files": len(files),
        "n_tests": total_tests,
        "n_skip_markers": total_skip,
        "n_skipif_markers": total_skipif,
        "n_xfail_markers": total_xfail,
        "n_parse_errors": parse_errors,
        "r17_count": n_r17,
        "r18_count": n_r18,
        "categories": categories_summary,
        "by_wave": {w: by_wave[w] for w, _ in waves_sorted},
        "by_category": {c: by_category[c] for c, _ in categories_sorted},
        "files": entries,
    }


# ---------------------------------------------------------------------------
# Output renderers.
# ---------------------------------------------------------------------------
def _render_text(report: dict) -> str:
    L: list[str] = []
    L.append("=" * 72)
    L.append("R17 + R18 test inventory (read-only, AST-only)")
    L.append("=" * 72)
    L.append(f"timestamp  : {report['timestamp']}")
    L.append(f"python     : {report['python_version']}  ({report['python']})")
    L.append(f"repo_root  : {report['repo_root']}")
    L.append(f"mode       : {report['mode']}")
    L.append("")
    L.append(f"test files : {report['n_test_files']}")
    L.append(f"test funcs : {report['n_tests']}")
    L.append(f"@skip      : {report['n_skip_markers']}")
    L.append(f"@skipif    : {report['n_skipif_markers']}")
    L.append(f"@xfail     : {report['n_xfail_markers']}")
    L.append(f"parse errs : {report['n_parse_errors']}")
    L.append("")
    L.append(f"r17 files  : {report['r17_count']}")
    L.append(f"r18 files  : {report['r18_count']}")
    L.append("")

    L.append("by wave:")
    for wave, info in report["by_wave"].items():
        L.append(f"  {wave:<14s}  files={info['n_files']:>3d}"
                 f"  tests={info['n_tests']:>4d}"
                 f"  skip={info['n_skip']:>2d} skipif={info['n_skipif']:>2d}"
                 f"  xfail={info['n_xfail']:>2d}")
    L.append("")

    L.append("by category (top 15 by n_tests):")
    for cat, info in list(report["by_category"].items())[:15]:
        L.append(f"  {cat:<28s}  files={info['n_files']:>3d}"
                 f"  tests={info['n_tests']:>4d}"
                 f"  skip={info['n_skip']:>2d} skipif={info['n_skipif']:>2d}")
    L.append("")
    return "\n".join(L)


def _render_markdown(report: dict) -> str:
    L: list[str] = []
    L.append("# Test inventory — R17 + R18 ship waves")
    L.append("")
    L.append(f"Generated: `{report['timestamp']}`  ")
    L.append(f"Tool: `{report['tool']}`  ")
    L.append(f"Mode: **{report['mode']}** (no pytest execution; counts are AST-estimated).")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Metric | Count |")
    L.append("|---|---:|")
    L.append(f"| test files | {report['n_test_files']} |")
    L.append(f"| test functions | {report['n_tests']} |")
    L.append(f"| `@pytest.mark.skip` | {report['n_skip_markers']} |")
    L.append(f"| `@pytest.mark.skipif` | {report['n_skipif_markers']} |")
    L.append(f"| `@pytest.mark.xfail` | {report['n_xfail_markers']} |")
    L.append(f"| parse errors | {report['n_parse_errors']} |")
    L.append(f"| R17 ship files | {report['r17_count']} |")
    L.append(f"| R18 ship files | {report['r18_count']} |")
    L.append("")
    L.append("## By ship wave")
    L.append("")
    L.append("| Wave | Files | Tests | @skip | @skipif | @xfail |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for wave, info in report["by_wave"].items():
        L.append(
            f"| {wave} | {info['n_files']} | {info['n_tests']}"
            f" | {info['n_skip']} | {info['n_skipif']} | {info['n_xfail']} |"
        )
    L.append("")
    L.append("## By category")
    L.append("")
    L.append("| Category | Files | Tests | @skip | @skipif | @xfail |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for cat, info in report["by_category"].items():
        L.append(
            f"| {cat} | {info['n_files']} | {info['n_tests']}"
            f" | {info['n_skip']} | {info['n_skipif']} | {info['n_xfail']} |"
        )
    L.append("")

    L.append("## Categories rollup (spec-aligned)")
    L.append("")
    L.append(
        "A shorter, dashboard-friendly view that uses the category names\n"
        "from the wf_final_verification spec (math_skills, adapters, etc.).\n"
        "When a category maps onto multiple internal buckets, the counts\n"
        "are summed (e.g. `p2_ml` = `p2_ml_baselines` + `p2_ml_metal_hybrid`\n"
        "+ `p2_ml_pretrain`).\n"
    )
    L.append("")
    L.append("| Category | Tests |")
    L.append("|---|---:|")
    for cat, n in report["categories"].items():
        L.append(f"| {cat} | {n} |")
    L.append("")
    L.append("## Per-file details")
    L.append("")
    L.append("Each row is one `tests/test_*.py` file. The ``Wave`` column tags the")
    L.append("ship wave the file belongs to (R17 / R18 / P2 ML / Docking / etc.);")
    L.append("``Category`` is the finer-grained family within that wave.")
    L.append("")
    L.append("| File | Wave | Category | Tests | Classes | @skip | @skipif | @xfail |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for entry in report["files"]:
        L.append(
            f"| `{entry['file']}` | {entry['wave']} | {entry['category']}"
            f" | {entry['n_tests']} | {entry['n_classes']}"
            f" | {entry['n_skip']} | {entry['n_skipif']} | {entry['n_xfail']} |"
        )
    L.append("")
    L.append("## Ship-wave contributions")
    L.append("")
    L.append(
        "Each R17 / R18 ship module ships with one or more pytest tests.\n"
        "The mapping below links ship waves to the tests that guard them.\n"
        "Use it to verify that no shipped module is test-less before\n"
        "closing the ship wave.\n"
    )
    L.append("")
    L.append("| Wave | Category | Files contributing | Total tests |")
    L.append("|---|---|---:|---:|")
    # Aggregate per (wave, category)
    wave_cat: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"files": 0, "tests": 0},
    )
    for e in report["files"]:
        key = (e["wave"], e["category"])
        wave_cat[key]["files"] += 1
        wave_cat[key]["tests"] += e["n_tests"]
    for (wave, cat), info in sorted(wave_cat.items()):
        L.append(f"| {wave} | {cat} | {info['files']} | {info['tests']} |")
    L.append("")
    if report["n_parse_errors"]:
        L.append("## Parse errors")
        L.append("")
        for e in report["files"]:
            if e["parse_error"]:
                L.append(f"- `{e['file']}`: {e['parse_error']}")
        L.append("")
    L.append("## Notes")
    L.append("")
    L.append(
        "* This inventory is **static / AST-only** — no `pytest` was\n"
        "  invoked. Counts above are *expected* counts based on parsing\n"
        "  `test_*.py` files; actual pass/fail/skip numbers must come\n"
        "  from running `pytest --collect-only` on a populated env.\n"
        "* `@pytest.mark.skip` / `skipif` / `xfail` decorators are\n"
        "  detected via regex; a decorator split across multiple lines\n"
        "  is still counted (the regex is anchored to the marker, not\n"
        "  to line boundaries).\n"
        "* The classification table is curated in\n"
        "  `molmetal/scripts/collect_all_tests.py`; extend it in\n"
        "  lock-step with `verify_r18_all_imports.SHIP_MODULES` when\n"
        "  adding new ship modules.\n"
    )
    L.append("")
    return "\n".join(L)


def _write_outputs(report: dict, json_path: Path, md_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R17 + R18 ship test inventory (read-only AST collect).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit the JSON inventory to stdout; no files written",
    )
    parser.add_argument(
        "--md", action="store_true",
        help="emit the Markdown summary to stdout; no files written",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="text-only check; do not write JSON or Markdown files",
    )
    args = parser.parse_args(argv)

    try:
        report = collect()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"collector itself failed: {type(exc).__name__}: {exc}\n"
        )
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.md:
        print(_render_markdown(report))
    elif args.check:
        print(_render_text(report))
    else:
        _write_outputs(report, JSON_OUT, MD_OUT)
        print(_render_text(report))
        print()
        print(f"JSON written: {JSON_OUT}")
        print(f"Markdown written: {MD_OUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())