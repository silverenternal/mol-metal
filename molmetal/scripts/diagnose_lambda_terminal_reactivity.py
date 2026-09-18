"""Check actual MCTS leaves for reaction witnesses despite beta-NF termination.

Only observes existing search objects; no gates, chemistry, or search logic edited.
Each case has a 90-second subprocess deadline, with partial results retained.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal"), str(Path(__file__).parent)]
from diagnose_lambda_diversity import save, DEFAULT_OUTPUT


def worker(case_path, output):
    from molmetal.scripts import lambda_100pocket_sweep as runner
    source = json.loads(case_path.read_text())
    config = source["config"]
    instances = []
    base = runner.MCTSProofSearch

    class Observer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.leaf_oracle_call_top_k_only = False
            self.leaves = {}
            instances.append(self)

        def _collect_leaves(self, root, leaves_by_smi):
            # Keep actual tree objects even when the production registry returns
            # independent statistic snapshots rather than live node references.
            stack, visited = [root], set()
            while stack:
                node = stack.pop()
                if id(node) in visited:
                    continue
                visited.add(id(node))
                if node.children:
                    stack.extend(node.children)
                else:
                    self.leaves[id(node)] = node
            result = super()._collect_leaves(root, leaves_by_smi)
            return result

    runner.MCTSProofSearch = Observer
    result = runner.run_one_pocket("diagnostic_no_pocket", config["n_simulations"],
                                  config["max_depth"], 0, config["tile_library"]=="extended_204",
                                  use_fragment_pool=False, top_k=config["top_k"], patience=50,
                                  tile_library=config["tile_library"], click_rules=config["click_rules"],
                                  branching_target=config["branching_target"], seed=config["seed"],
                                  seed_strategy="click_tile", symbolic_prior=False, synthesis_oracle=False)
    search = instances[0]
    key = lambda s: runner._structure_key(runner._smi_of(s))
    report = {"status": "running", "source_case": str(case_path), "config": config,
              "rows": [], "scope": "Actual leaf state objects; post-search direct rule.reduce, stop at first new structure witness per leaf; max64leaves",
              "candidate_parity": [runner._structure_key(c["smiles"]) for c in result["candidates"]]
                                  ==source["final_selection"]["returned_structures"]}
    save(output, report)
    unique = {key(n.state): n for n in search.leaves.values()}
    report["total_unique_leaves"] = len(unique)
    for smi, node in sorted(unique.items())[:64]:
        row = {"smiles": smi, "beta_normal_form": node.state.is_beta_normal_form,
               "node_is_terminal": node.is_terminal, "attempts": 0, "nonmatches": 0,
               "exceptions": [], "reaction_witness": None}
        for name, rule in search.rules.items():
            for tile in search._resolve_expand_tile_pool():
                row["attempts"] += 1
                try:
                    products = rule.reduce((node.state, tile))
                except Exception as exc:
                    row["exceptions"].append({"rule": name, "tile": key(tile), "error": repr(exc)})
                    continue
                row["nonmatches"] += int(not products)
                new = sorted({key(p) for p in products if key(p) and key(p) not in {smi, key(tile)}})
                if new:
                    row["reaction_witness"] = {"rule": name, "partner": key(tile), "products": new}
                    break
            if row["reaction_witness"]:
                break
        report["rows"].append(row)
        save(output, report)
    report.update(status="completed", probed=len(report["rows"]),
                  terminal_but_reactive=sum(r["node_is_terminal"] and bool(r["reaction_witness"]) for r in report["rows"]))
    save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "terminal_reactivity")
    args = parser.parse_args()
    if args.case:
        try:
            worker(args.case, args.output)
        except Exception:
            partial = json.loads(args.output.read_text()) if args.output.exists() else {}
            partial.update(status="failed", error=traceback.format_exc())
            save(args.output, partial)
            raise
        return
    if list(args.output.glob("case_*.json")):
        parser.error("Output already contains evidence; choose a fresh --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for case in ("case_08", "case_20", "case_26"):
        path = args.output / f"{case}.json"
        with (args.output / f"{case}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, __file__, "--case", str(DEFAULT_OUTPUT / f"{case}.json"),
                                        "--output", str(path)], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
                partial = json.loads(path.read_text()) if path.exists() else {}
                partial.update(status="timeout")
                save(path, partial)
        print(case, process.returncode, flush=True)


if __name__ == "__main__":
    main()
