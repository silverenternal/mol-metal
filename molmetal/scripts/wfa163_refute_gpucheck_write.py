#!/usr/bin/env python
# Wf-A163 ADVERSARIAL refutation (refuter id: gpucheck) -- verdict writer.
# CPU-only. Uses the repo's atomic-write protocol (AGENTS.md §5.6).
"""Write molmetal/reports/wfa163_refute_gpucheck/verdict.{md,json}."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
OUT = ROOT / "molmetal/reports/wfa163_refute_gpucheck"

_spec = importlib.util.spec_from_file_location(
    "wf386a3_atomic_write", ROOT / "molmetal/scripts/wf386a3_atomic_write.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
atomic_write_file = _mod.atomic_write_file

# --------------------------------------------------------------------------
# One entry per SUBSTANTIVE CLAIM of branch G.
#   verdict: CONFIRMED | REFUTED | UNSURE
#   branch_number: the number the branch asserts
#   my_number: the number I measured INDEPENDENTLY
# --------------------------------------------------------------------------
ENTRIES = [
    {
        "id": "G-01",
        "claim": (
            "§2.1: 'torch.cuda.current_device() (8 sites) and torch.cuda.set_device() (2 sites) "
            "go through _lazy_init() ... 10 sites total; 0 of them at import time.'"
        ),
        "verdict": "REFUTED",
        "branch_number": 10,
        "my_number": 95,
        "confidence": "CONFIRMED",
        "severity": "HIGH",
        "evidence": (
            "The census's scanner recognises only 5 CUDA API names "
            "(molmetal/scripts/wfa163_gpu_tooling_census.py:96 PROBE_APIS = is_available, "
            "device_count, current_device, set_device, init). The installed torch has 25 "
            "torch.cuda.* functions that transitively reach _lazy_init() "
            "(.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:563-593 -> :591 "
            "torch._C._cuda_init()). My independent AST scan (ground truth derived FROM the "
            "installed torch source by transitive closure, not from a hand-written list) of the "
            "census's OWN scope prefixes (molmetal/, scripts/, tests/) finds 95 primary-context-"
            "creating call sites: synchronize 50, get_device_name 16, get_device_properties 12, "
            "get_device_capability 7, current_device 8, set_device 2. The census records 81 of "
            "those 95 sites as NO hit at all (census hits with api=synchronize / "
            "get_device_name / get_device_properties / get_device_capability = 0 each, in "
            "molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json). "
            "Ground truth per API in the installed torch: get_device_name -> get_device_properties "
            "-> _lazy_init() (torch/cuda/__init__.py:754, :787, :799); get_device_capability -> "
            "_lazy_init() (:769, :783); synchronize -> _lazy_init() (:1271, :1279); set_device -> "
            "torch._C._cuda_setDevice() (:739, :752). The report's §7.1 limitation uses the number "
            "10 as the census's basis for attributing the 550 MiB watchdog observation to a primary "
            "context; that basis covers 10 of 95 sites. The report discloses no API-table "
            "limitation (grep for synchronize|get_device_name|get_device_properties|"
            "get_device_capability|PROBE_APIS in report.md returns nothing). The sub-claim '0 of "
            "them at import time' IS correct: I measured 0 of 95 at import time."
        ),
        "why_it_matters": (
            "§2.1 is the load-bearing section for the report's central question (which sites "
            "allocate the ~500-600 MiB class). A census that enumerates 10 of 95 primary-context "
            "sites cannot support the attribution it makes, and the omission is undisclosed."
        ),
    },
    {
        "id": "G-02",
        "claim": "§2.1: '5 probes run at import time, all of them is_available' (5 listed file:line).",
        "verdict": "REFUTED",
        "branch_number": 5,
        "my_number": 6,
        "confidence": "CONFIRMED",
        "severity": "MEDIUM",
        "evidence": (
            "The census records the site molmetal/molmetal_lam/sbdd_env/protein_metal_binding.py:208 "
            "as a hit but tags it module_level=false, so it is excluded from the count of 5. The "
            "source is a @dataclass class-body field default "
            "('device: str = \"cuda\" if torch.cuda.is_available() else \"cpu\"'), and a dataclass "
            "field default is evaluated at CLASS-CREATION time = import time. The census's own "
            "scanner is the cause: _index_module_level_lines "
            "(molmetal/scripts/wfa163_gpu_tooling_census.py:372-383) skips ast.ClassDef spans, so "
            "class-body defaults are mis-flagged as not-module-level. I verified the runtime "
            "behaviour EMPIRICALLY (CPU-only, CUDA_VISIBLE_DEVICES=\"\"): importing "
            "molmetal.molmetal_lam.sbdd_env.protein_metal_binding calls torch.cuda.is_available() "
            "exactly 1 time during import (instrumented spy on torch.cuda.is_available). My "
            "corrected AST scan of the census's scope finds 6 import-time probes, not 5: the 5 "
            "listed plus protein_metal_binding.py:208. The report's §5.0 uncovered-paths table (U1, "
            "U2, U3) and §5.1 do not mention protein_metal_binding.py at all (grep returns "
            "nothing), so the 6th import-time site is neither counted nor disclosed as uncovered."
        ),
        "why_it_matters": (
            "The import-time probe count is a headline census number and the §5.0 uncovered-path "
            "table is what tells the main thread what to patch. A 6th import-time probe that is "
            "neither counted nor listed as uncovered is a real gap."
        ),
    },
    {
        "id": "G-03",
        "claim": (
            "§3: '28 child spawns in 18 files (all under molmetal/tests/) build a hand-made env "
            "dict that carries no CUDA_VISIBLE_DEVICES key. Such a child inherits NOTHING from the "
            "parent -- including the CPU lock.'  (also the U1 row and the diff's header comment)"
        ),
        "verdict": "REFUTED",
        "branch_number": 28,
        "my_number": 24,
        "confidence": "CONFIRMED",
        "severity": "MEDIUM",
        "evidence": (
            "I re-classified all 842 child spawns in scope with my own assignment-resolving AST "
            "provenance classifier (never the branch's scanner). Exactly 24 sites build a "
            "hand-made env dict carrying no CUDA_VISIBLE_DEVICES key, in 15 files -- not 28 in 18. "
            "Four of the census's 28 are NOT lock-droppers: (1) "
            "molmetal/tests/test_refactored_train_scripts.py:315 and (2) :392 build "
            "env = dict(__import__('os').environ), which DOES copy the parent's blank "
            "CUDA_VISIBLE_DEVICES, so the child inherits the lock; the census misclassifies them "
            "because _resolve_env_node's dict(...) branch does a literal substring test "
            "('os.environ' in _seg(...)), and the source text is ('os').environ, which does not "
            "contain the literal substring 'os.environ' "
            "(molmetal/scripts/wfa163_gpu_tooling_census.py:~660-670); (3) "
            "molmetal/tests/test_registry_generate_import_path.py:54 builds a dict comprehension "
            "{k: v for k, v in env.items() if k != 'PYTHONPATH'} over os.environ, which also "
            "carries the parent's blank; (4) molmetal/tests/test_wf432a2_w246_real_trainer.py:91 is "
            "a WRAPPER CALL SITE (_run_wrapper) whose implementation does "
            "full_env = os.environ.copy() at :25 and adds only PYTHONPATH -- the wrapper inherits "
            "the lock. My set is a strict subset of the census's 28: 24 shared, 4 census-only, "
            "0 mine-only. So the census OVER-COUNTS by 4 (17%) and its 'inherits NOTHING' phrasing "
            "is wrong for those 4."
        ),
        "why_it_matters": (
            "The count drives the patch's rationale and its header comment ('MEASURED 2026-11-01: "
            "28 child spawns ... inherit NOTHING'). The fix (choke point on subprocess.Popen.__init__) "
            "is still correct and still covers the real 24 -- but the artefact overstates the "
            "population and the mechanism statement is false for 4 of the 28."
        ),
    },
    {
        "id": "G-04",
        "claim": (
            "§4 (pre-flight repair): (1) no-subprocess short-circuit under the lock; (2) "
            "PYTORCH_NVML_BASED_CUDA_CHECK=1 env on the remaining path; (3) _device_authorisation_error "
            "refusal in BOTH consuming gates; (4) the -1-mask refusal."
        ),
        "verdict": "CONFIRMED",
        "branch_number": 4,
        "my_number": 4,
        "confidence": "CONFIRMED",
        "severity": "n/a",
        "evidence": (
            "NOT REFUTED. I verified all four against molmetal/scripts/pre_flight_check.py by "
            "EXECUTION under CUDA_VISIBLE_DEVICES=\"\" (CPU-only, torch never touched CUDA): "
            "(1) _session_cuda_is_locked() (pre_flight_check.py:13056) returns True and "
            "_smoke_device() (:13103) returns 'cpu' with torch NOT imported before or after "
            "('torch' in sys.modules == False both times) -- proving the no-subprocess short-circuit "
            "at :13136-13138. (2) The NVML env is at :13211 "
            "(env={**os.environ, 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}). (3) Both consuming gates "
            "call _device_authorisation_error (:13145) and return FAIL with the text "
            "'<error> (smoke NOT run)' -- D-MPNN gate at :13437-13444, chemprop gate at "
            ":17017-17024. (4) With CUDA_VISIBLE_DEVICES='-1', _visible_device_count_hint() (:13066) "
            "returns 0 (negative index excludes) and _device_authorisation_error('cuda:0') returns "
            "'CUDA_VISIBLE_DEVICES hides every device (visible=0) but the smoke was about to run on "
            "'cuda:0''. The report's line citations all check out."
        ),
        "why_it_matters": "The landed repair is correct; the branch's §4 is accurate and NOT REFUTED.",
    },
    {
        "id": "G-05",
        "claim": (
            "§3 (suspect b): 'a child inherits the lock ONLY IF THE PARENT HAD IT, which is why the "
            "exposure lives at the entry points that build a fresh env, not at the probes.'"
        ),
        "verdict": "CONFIRMED",
        "branch_number": None,
        "my_number": None,
        "confidence": "CONFIRMED",
        "severity": "n/a",
        "evidence": (
            "NOT REFUTED. The qualification is stated correctly and is the right physics: "
            "CUDA_VISIBLE_DEVICES is inherited through the environment, so a child spawned with no "
            "env= kwarg inherits the parent's blank. I confirmed the mechanism on disk: "
            "molmetal/molmetal_lam/sbdd_env/training_utils.py:99 "
            "ensure_cuda_visible_devices_default() sets CVD='0' only when the variable is UNSET "
            "(an explicit '' counts as set and is honoured verbatim), and "
            "molmetal/scripts/_cuda_default.py is the package-import-free transport. The report's "
            "shell-layer paragraph is also correct: 'CUDA_VISIBLE_DEVICES=0 cmd' is a per-command "
            "operator override, not a leak. I independently confirmed the launcher's "
            "'${VAR-default}' vs '${VAR:-default}' distinction: only 5 shell sites use the "
            "defeating ':-' form, and exactly ONE of them is a genuine session-lock defeater "
            "(scripts/launch_wf308c_todo220_gpu.sh:125), matching the branch's finding."
        ),
        "why_it_matters": "The answer to suspect (b) is correctly qualified; NOT REFUTED.",
    },
    {
        "id": "G-06",
        "claim": (
            "§1/§2: the probe census itself -- hits_total 2025, cuda_probe 274, and the per-API "
            "split {is_available 242, device_count 22, current_device 8, set_device 2}."
        ),
        "verdict": "CONFIRMED",
        "branch_number": 274,
        "my_number": 274,
        "confidence": "CONFIRMED",
        "severity": "n/a",
        "evidence": (
            "NOT REFUTED. My independent AST scan (torch.cuda.<api>(...) attribute form, scope "
            "molmetal/ + scripts/ + tests/, git ls-files tracked + untracked-not-ignored) returns "
            "exactly 274 with the identical per-API split {is_available 242, device_count 22, "
            "current_device 8, set_device 2}. The branch's probe-counting machinery is sound; its "
            "defect is the API TABLE (G-01), not the counting."
        ),
        "why_it_matters": "The census's counting method is sound; only its API coverage is short.",
    },
    {
        "id": "G-07",
        "claim": (
            "§1/§7.3: the scope prefixes are molmetal/, scripts/, tests/; 6339 files enumerated in "
            "scope; 3011 decoded."
        ),
        "verdict": "UNSURE",
        "branch_number": 6339,
        "my_number": 6380,
        "confidence": "PLAUSIBLE",
        "severity": "LOW",
        "evidence": (
            "I measured 6380 files enumerated and 2930 .py files in the same three scope prefixes "
            "on the CURRENT working tree (the branch's own snapshot records 6339 / 2915). The "
            "delta is concurrent Wf-A163 branches landing files during the run; the branch discloses "
            "this itself (§1 and §7.4 'The file counts are a snapshot'). I could not reconstruct the "
            "6339 figure exactly from the artefact, so the DENOMINATOR ITSELF is not refuted -- but "
            "it is not independently reproducible either, which is a property the report already "
            "acknowledges."
        ),
        "why_it_matters": "Minor; the drift is disclosed and explained.",
    },
    {
        "id": "G-08",
        "claim": (
            "§7.3: 'The scope prefixes are molmetal/, scripts/, tests/. Anything outside them "
            "(e.g. paper/, docs/, tools/ at the root) was not enumerated.'"
        ),
        "verdict": "REFUTED",
        "branch_number": 0,
        "my_number": 7,
        "confidence": "CONFIRMED",
        "severity": "MEDIUM",
        "evidence": (
            "The scope restriction is disclosed, but the disclosed examples are the benign ones. "
            "There are 7 primary-context-creating call sites in TRACKED, root-level files OUTSIDE "
            "the three scope prefixes, on real GPU paths: main.py:28 (get_device_name) and :29 "
            "(get_device_properties); models/_scatter.py:89 and :159 (get_device_capability); "
            "submission/paper_a/molmetal/__init__.py:95 (get_device_name); triton_kernels/backend.py:55 "
            "(get_device_capability) and :73 (get_device_name). I confirmed each is tracked "
            "(git ls-files) and that the census records 0 hits outside its scope prefixes. "
            "triton_kernels/tests/test_kernel_suite.py:12 is an additional import-time "
            "torch.cuda.is_available() probe outside the scope. The report names paper/, docs/ and "
            "tools/ as the excluded paths -- never models/, triton_kernels/, submission/ or main.py."
        ),
        "why_it_matters": (
            "The report's own limitation text points the reader at the harmless exclusions and "
            "away from the GPU paths that are actually excluded. models/_scatter.py:89 is a "
            "primary-context creator on a tracked module."
        ),
    },
    {
        "id": "G-09",
        "claim": (
            "§6 / suspect (d): the proposed diff applies cleanly WITHOUT being applied, and does not "
            "touch the read-only pre-flight path."
        ),
        "verdict": "CONFIRMED",
        "branch_number": None,
        "my_number": None,
        "confidence": "CONFIRMED",
        "severity": "n/a",
        "evidence": (
            "NOT REFUTED. 'git apply --check molmetal/reports/wfa163_gpu_tooling_census/"
            "patch_smoke_device_lock.diff' at the repo root exits 0 (CLEAN) on the current working "
            "tree, and the diff is NOT applied (the +++ headers list molmetal/tests/conftest.py, "
            "conftest.py, molmetal/scripts/infer_cfm.py -- none of which contain the patch's "
            "choke-point text in the working tree). The only occurrence of 'pre_flight_check' in the "
            "diff is a COMMENT at diff line 26 explaining that it is deliberately not touched; no "
            "hunk targets it. State note verified: the repo-root conftest.py is UNTRACKED "
            "(git ls-files conftest.py -> 0 results; git status shows '?? conftest.py'), and the "
            "branch's own STATE NOTE at diff lines 61-65 says so."
        ),
        "why_it_matters": "Suspect (d) is satisfied: the diff is applicable, unapplied, and off the read-only path.",
    },
    {
        "id": "G-10",
        "claim": (
            "Branch hygiene: did branch G write into a forbidden path (paper/, docs/, AGENTS.md, "
            "README.md, TODO/, configs/, metrics/, registry/, molmetal/molmetal_lam/, "
            "pre_flight_check.py, wf_a155_ligand_connected_rate.py, or a test_wfa* file it did not "
            "create)?"
        ),
        "verdict": "CONFIRMED",
        "branch_number": 0,
        "my_number": 0,
        "confidence": "CONFIRMED",
        "severity": "n/a",
        "evidence": (
            "NOT REFUTED -- no violation found. Branch G's own footprint is exactly three paths, all "
            "new and untracked: molmetal/scripts/wfa163_gpu_tooling_census.py, "
            "molmetal/tests/test_wfa163_gpu_tooling_census.py (its own test_wfa* file), and "
            "molmetal/reports/wfa163_gpu_tooling_census/. The forbidden-path check is confounded by "
            "PRE-EXISTING working-tree modifications from other waves (AGENTS.md, README.md, TODO/, "
            "configs/ and molmetal/scripts/pre_flight_check.py were already modified before this "
            "review; the pre-flight modification is Wf-A162's LANDED repair, which the branch "
            "READ-ONLY cites rather than edits). Branch G's diff does not target any of them. Its "
            "12 own regression tests pass: 'CUDA_VISIBLE_DEVICES=\"\" .venv/bin/python -m pytest "
            "molmetal/tests/test_wfa163_gpu_tooling_census.py -q' -> 12 passed."
        ),
        "why_it_matters": "No forbidden-path write by branch G.",
    },
]


def build_md() -> str:
    lines = []
    A = lines.append
    A("# Wf-A163 branch G (GPU-tooling census) -- ADVERSARIAL REFUTATION VERDICT")
    A("")
    A("Refuter id: `gpucheck`. Artefact: `molmetal/reports/wfa163_refute_gpucheck/verdict.md`.")
    A("Branch under review: `molmetal/reports/wfa163_gpu_tooling_census/report.md` + "
      "`gpu_tooling_census.json` + `patch_smoke_device_lock.diff`.")
    A("")
    A("All numbers below are **MEASURED** (2026-11-01) on the current working tree and every one "
      "traces to a path on disk. **0 GPU was used**: every command ran with "
      "`CUDA_VISIBLE_DEVICES=\"\"` and the single venv `.venv/bin/python`; torch never touched CUDA "
      "(the pre-flight verification below proves torch was never even IMPORTED on the locked path).")
    A("")
    A("## Scoreboard")
    A("")
    A("| id | claim | verdict | branch number | my number |")
    A("|---|---|---|---|---|")
    for e in ENTRIES:
        bn = "n/a" if e["branch_number"] is None else str(e["branch_number"])
        mn = "n/a" if e["my_number"] is None else str(e["my_number"])
        A(f"| {e['id']} | {e['claim'].split(':')[0]} | **{e['verdict']}** | {bn} | {mn} |")
    A("")
    refuted = [e for e in ENTRIES if e["verdict"] == "REFUTED"]
    confirmed = [e for e in ENTRIES if e["verdict"] == "CONFIRMED"]
    unsure = [e for e in ENTRIES if e["verdict"] == "UNSURE"]
    A(f"**{len(refuted)} REFUTED / {len(confirmed)} CONFIRMED (NOT REFUTED) / {len(unsure)} UNSURE** "
      f"out of {len(ENTRIES)} adjudicated claims.")
    A("")
    A("A refutation that FAILS is a finding: G-04, G-05, G-06, G-09 and G-10 are claims I set out "
      "to break and could not. They are recorded with the citation that saved them.")
    A("")
    A("## Refutations in detail")
    A("")
    for e in refuted:
        A(f"### {e['id']} -- {e['claim']}")
        A("")
        A(f"* **Verdict:** {e['verdict']} (severity {e['severity']})")
        A(f"* **Branch says:** {e['branch_number']}  |  **I measured:** {e['my_number']}")
        A(f"* **Evidence:** {e['evidence']}")
        A(f"* **Why it matters:** {e['why_it_matters']}")
        A("")
    A("## Claims NOT refuted (recorded with the citation that saved them)")
    A("")
    for e in confirmed + unsure:
        A(f"### {e['id']} -- {e['claim']}")
        A("")
        A(f"* **Verdict:** {e['verdict']}")
        A(f"* **Evidence:** {e['evidence']}")
        A("")
    A("## Answers to the workflow's four suspects")
    A("")
    A("* **(a) Is the census complete?** NO -- **REFUTED** (G-01, G-02, G-08). It misses 81 of the "
      "95 in-scope primary-context-creating sites because the scanner's `PROBE_APIS` table "
      "recognises 5 API names; it under-counts import-time probes as 5 when there are 6; and its "
      "scope prefixes exclude 7 tracked root-level primary-context sites (`models/_scatter.py:89`, "
      "`models/_scatter.py:159`, `triton_kernels/backend.py:55`, `triton_kernels/backend.py:73`, "
      "`main.py:28`, `main.py:29`, `submission/paper_a/molmetal/__init__.py:95`) while disclosing "
      "only `paper/`, `docs/`, `tools/`.")
    A("* **(b) Is the lock answer correctly qualified?** YES -- **NOT REFUTED** (G-05). The report "
      "states the parent-propagation condition verbatim and correctly.")
    A("* **(c) Did the branch verify the landed repair?** YES, and the repair is real -- **NOT "
      "REFUTED** (G-04). All four claims verified by execution, including that torch is never "
      "imported on the locked path.")
    A("* **(d) Does the diff apply cleanly and avoid the read-only path?** YES -- **NOT REFUTED** "
      "(G-09). `git apply --check` exits 0; the diff is unapplied; the sole `pre_flight_check` "
      "mention is a comment.")
    A("")
    A("## Forbidden-path check")
    A("")
    A("**No violation by branch G.** Its footprint is exactly "
      "`molmetal/scripts/wfa163_gpu_tooling_census.py`, "
      "`molmetal/tests/test_wfa163_gpu_tooling_census.py` and "
      "`molmetal/reports/wfa163_gpu_tooling_census/`. Pre-existing working-tree modifications to "
      "`AGENTS.md` / `README.md` / `TODO/` / `configs/` / `molmetal/scripts/pre_flight_check.py` "
      "come from other waves; the pre-flight one is Wf-A162's LANDED repair that branch G only "
      "reads.")
    A("")
    A("## Reproduction")
    A("")
    A("```")
    A("CUDA_VISIBLE_DEVICES=\"\" /home/hugo/codes/mol-metal/.venv/bin/python \\")
    A("  molmetal/scripts/wfa163_refute_gpucheck.py")
    A("```")
    A("")
    A("Writes `molmetal/reports/wfa163_refute_gpucheck/independent_scan.json` (raw measurements). "
      "The scanner derives the primary-context API set FROM the installed torch source by "
      "transitive closure over `_lazy_init()` -- it does not assume the branch's table is right or "
      "wrong -- and never imports torch, never touches CUDA, and never uses `os.walk` "
      "(`git ls-files` only).")
    A("")
    return "\n".join(lines)


def build_json() -> str:
    payload = {
        "wave": "Wf-A163",
        "branch": "G -- GPU-tooling census",
        "refuter_id": "gpucheck",
        "date": "2026-11-01",
        "gpu_used": False,
        "cuda_visible_devices": "",
        "python": "/home/hugo/codes/mol-metal/.venv/bin/python",
        "branch_artefacts": [
            "molmetal/reports/wfa163_gpu_tooling_census/report.md",
            "molmetal/reports/wfa163_gpu_tooling_census/gpu_tooling_census.json",
            "molmetal/reports/wfa163_gpu_tooling_census/patch_smoke_device_lock.diff",
            "molmetal/scripts/wfa163_gpu_tooling_census.py",
        ],
        "summary": {
            "claims_adjudicated": len(ENTRIES),
            "refuted": sum(1 for e in ENTRIES if e["verdict"] == "REFUTED"),
            "confirmed_not_refuted": sum(1 for e in ENTRIES if e["verdict"] == "CONFIRMED"),
            "unsure": sum(1 for e in ENTRIES if e["verdict"] == "UNSURE"),
        },
        "entries": ENTRIES,
        "suspects": {
            "a_census_complete": {
                "verdict": "REFUTED",
                "refs": ["G-01", "G-02", "G-08"],
            },
            "b_lock_answer_qualified": {
                "verdict": "CONFIRMED",
                "refs": ["G-05"],
            },
            "c_landed_repair_verified": {
                "verdict": "CONFIRMED",
                "refs": ["G-04"],
            },
            "d_diff_applies_and_avoids_readonly": {
                "verdict": "CONFIRMED",
                "refs": ["G-09"],
            },
        },
        "forbidden_path_violation": False,
        "branch_footprint": [
            "molmetal/scripts/wfa163_gpu_tooling_census.py",
            "molmetal/tests/test_wfa163_gpu_tooling_census.py",
            "molmetal/reports/wfa163_gpu_tooling_census/",
        ],
        "branch_own_tests": {
            "command": 'CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest '
                       "molmetal/tests/test_wfa163_gpu_tooling_census.py -q",
            "result": "12 passed",
        },
        "independent_measurements": {
            "primary_context_sites_in_scope": 95,
            "primary_context_sites_absent_from_census_hits": 81,
            "primary_context_sites_by_api": {
                "synchronize": 50,
                "get_device_name": 16,
                "get_device_properties": 12,
                "get_device_capability": 7,
                "current_device": 8,
                "set_device": 2,
            },
            "primary_context_sites_at_import_time": 0,
            "import_time_probe_sites_in_scope": 6,
            "import_time_probe_sites_branch_claimed": 5,
            "hand_made_no_cvd_child_spawns": 24,
            "hand_made_no_cvd_files": 15,
            "census_claimed_lock_droppers": 28,
            "census_droppers_that_actually_inherit_the_lock": 4,
            "probe_count_reproduced": 274,
            "probe_count_per_api": {
                "is_available": 242,
                "device_count": 22,
                "current_device": 8,
                "set_device": 2,
            },
            "out_of_scope_tracked_primary_context_sites": 7,
            "files_enumerated_in_scope": 6380,
        },
    }
    return json.dumps(payload, indent=2) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_file(OUT / "verdict.md", build_md())
    atomic_write_file(OUT / "verdict.json", build_json())
    for p in ("verdict.md", "verdict.json"):
        f = OUT / p
        print(f"wrote {f} ({f.stat().st_size} bytes)")
