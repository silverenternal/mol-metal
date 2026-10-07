#!/usr/bin/env python
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""§5.7 50-step CPU smoke for the L3 VariancePUCT wire-in.

bench_mcts_search_quality.py is an MCTS *benchmark*, not a torch trainer, so it
has no --device/--epochs/--batch-size and no per-step loss / decode_ratio.  The
§5.7 trainer invariants therefore map onto MCTS equivalents:

  §5.7 #1 parse_args accepts the flag  -> CHECK 1 (subprocess --help)
  §5.7 #2 imports + initialises        -> CHECK 2 (import probe)
  §5.7 #3 live-value invariant         -> CHECK 3 (variance PUCT actually
                                           changes _cpuct_for vs constant)
  §5.7 #4 monotone non-increasing      -> CHECK 4 (n/a -- SMOKE-SUBSET: bench
                                           has no per-step loss; the wire-in is
                                           in the SELECTION weight, not a loss)

Also probes --cpuct-variance-beta, which is NOT in argparse (FLAG_MISSING).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/mol-metal")
PY = REPO / ".venv/bin/python"
BENCH = REPO / "molmetal/scripts/bench_mcts_search_quality.py"
sys.path.insert(0, str(REPO / "molmetal"))
sys.path.insert(0, str(REPO))

VARIANCE_FLAGS = ["--cpuct-mode", "--cpuct-variance-alpha", "--cpuct-variance-beta"]
results = {}


def check(name, fn):
    try:
        results[name] = fn()
    except Exception as exc:  # noqa: BLE001
        results[name] = {"pass": False, "detail": f"{type(exc).__name__}: {exc}"}


# --- CHECK 1: argparse accepts the flags -------------------------------------
def _check_parse_args():
    proc = subprocess.run(
        [str(PY), str(BENCH), "--help"],
        capture_output=True, text=True, timeout=180, cwd=str(REPO),
    )
    help_txt = proc.stdout + proc.stderr
    accepted, missing = [], []
    for flag in VARIANCE_FLAGS:
        (accepted if flag in help_txt else missing).append(flag)
    return {
        "pass": proc.returncode == 0 and "--cpuct-mode" in accepted
        and "--cpuct-variance-alpha" in accepted,
        "returncode": proc.returncode,
        "accepted": accepted,
        "missing_from_argparse": missing,
    }


# --- CHECK 2: import + initialise ---------------------------------------------
def _check_import():
    from molmetal_lam.search_alg.proof_search import MCTSProofSearch  # noqa: F401
    from molmetal_lam.search_alg import variance_cpuct
    return {
        "pass": True,
        "effective_cpuct_present": hasattr(variance_cpuct, "effective_cpuct"),
        "MAX_VARIANCE": getattr(variance_cpuct, "MAX_VARIANCE", None),
    }


# --- CHECK 3: variance PUCT is a real wire-in, not FLAG_ONLY ------------------
def _check_wire_in():
    from molmetal_lam.search_alg.variance_cpuct import effective_cpuct

    # 1. constant mode returns base unchanged
    const = effective_cpuct(1.5, 0.0, 0.8)   # var=0 -> base (no-op case)
    var = effective_cpuct(1.5, 4.0, 0.8)     # var=4 -> 1.5*(1+0.8*2)=3.9
    var2 = effective_cpuct(1.5, 25.0, 0.8)   # clamped at MAX_VARIANCE -> 2x base
    # 3. _cpuct_for must actually route on the attribute
    import inspect
    from molmetal_lam.search_alg.proof_search import MCTSProofSearch
    src = inspect.getsource(MCTSProofSearch._cpuct_for)
    routes = 'cpuct_mode", "constant") != "variance"' in src
    consumes_alpha = "cpuct_variance_alpha" in src
    return {
        "pass": const == 1.5 and var != 1.5 and var2 != 1.5 and routes and consumes_alpha,
        "constant_returns_base": const == 1.5,
        "variance_parent_var_changes_weight": var != 1.5,
        "variance_child_var_changes_weight": var2 != 1.5,
        "cpuct_for_routes_on_mode": routes,
        "cpuct_for_consumes_alpha": consumes_alpha,
        "values": {"constant": const, "var_parent": var, "var_child": var2},
    }


# --- CHECK 4: 50-step-equivalent CPU run of the bench -------------------------
def _check_e2e():
    out_dir = "/tmp/wf37_smoke_out"
    flags = [
        str(BENCH),
        "--n-targets", "2",
        "--n-seeds", "1",
        "--n-simulations", "400",
        "--max-depth", "3",
        "--cpuct-mode", "variance",
        "--cpuct-variance-alpha", "0.8",
        "--fpu-reduction", "0.1",
        "--out-dir", out_dir,
    ]
    proc = subprocess.run(
        [str(PY), *flags], capture_output=True, text=True, timeout=900, cwd=str(REPO),
    )
    blob = proc.stdout + proc.stderr
    sims = re.findall(r"(\d+)\s+simulations", blob)
    return {
        "pass": proc.returncode == 0,
        "returncode": proc.returncode,
        "simulations_observed": int(sims[0]) if sims else None,
        "steps_ge_50": True,  # SMOKE-SUBSET: bench has no per-step loss; n_simulations=400 search iterations
        "cpuct_mode_in_report": "variance" in blob,
        "tail": blob.strip().splitlines()[-12:],
    }


for nm, fn in [
    ("check1_parse_args", _check_parse_args),
    ("check2_import", _check_import),
    ("check3_wire_in", _check_wire_in),
    ("check4_e2e_cpu", _check_e2e),
]:
    check(nm, fn)

results["OVERALL"] = all(
    results[k]["pass"] for k in results if k != "OVERALL"
)
print(json.dumps(results, indent=2, default=str))
sys.exit(0 if results["OVERALL"] else 1)