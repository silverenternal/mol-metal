"""Bounded, no-docking diagnostics of the existing Lambda search.

Run with project Python: uv run python molmetal/scripts/diagnose_lambda_diversity.py
Only this diagnostic process substitutes an observing subclass in the runner.
Search outputs, chemistry rules, type predicates and binding constraints are unchanged.
The optional external leaf docking oracle is explicitly disabled for this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "molmetal/reports/lambda_diversity_diagnostic_20260913"


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def worker(config, output):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "molmetal"))
    from molmetal.scripts import lambda_100pocket_sweep as runner
    from molmetal_lam.binding.types import typecheck
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    report = {"config": config, "status": "running", "stage": "imports_complete"}
    save(output, report)
    original_class = runner.MCTSProofSearch
    instances = []

    def key(state):
        return runner._structure_key(runner._smi_of(state))

    class ObservedSearch(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # This is a declared no-docking experiment, including discovered oracles.
            self.leaf_oracle_call_top_k_only = False
            self.observed_products = {}
            self.reduction_counts = Counter()
            self.expansions = []
            self.return_locals = {}
            instances.append(self)

        def _safe_reduce(self, rule, state, tile):
            products = super()._safe_reduce(rule, state, tile)
            self.reduction_counts[(rule.name, "calls")] += 1
            self.reduction_counts[(rule.name, "nonfires")] += int(not products)
            self.reduction_counts[(rule.name, "product_instances")] += len(products)
            for product in products:
                product_key = key(product)
                if product_key:
                    self.observed_products[product_key] = product
            return products

        def _expand(self, state):
            children = super()._expand(state)
            self.expansions.append({"parent": key(state), "product_instances": len(children),
                                    "products": sorted({key(c[0]) for c in children if key(c[0])}),
                                    "per_rule": dict(Counter(c[1] for c in children))})
            report.update(stage="search_expansion", expansions_completed=len(self.expansions),
                          unique_products_so_far=len(self.observed_products))
            save(output, report)
            return children

        def search(self, initial_state, max_depth=3):
            self.initial_state = initial_state
            target_code = original_class.search.__code__

            def capture(frame, event, arg):
                if event == "return" and frame.f_code is target_code:
                    self.return_locals = {
                        "candidates": list(frame.f_locals.get("candidates", [])),
                        "leaves_by_smi": dict(frame.f_locals.get("leaves_by_smi", {})),
                    }

            previous = sys.getprofile()
            sys.setprofile(capture)
            try:
                return super().search(initial_state, max_depth=max_depth)
            finally:
                sys.setprofile(previous)

    runner.MCTSProofSearch = ObservedSearch
    started = time.monotonic()
    result = runner.run_one_pocket(
        "diagnostic_no_pocket", config["n_simulations"], config["max_depth"], 0,
        config["tile_library"] == "extended_204", use_fragment_pool=False,
        top_k=config["top_k"], early_stop=True, patience=50,
        tile_library=config["tile_library"], click_rules=config["click_rules"],
        branching_target=config["branching_target"], seed=config["seed"],
        seed_strategy="click_tile", symbolic_prior=False, synthesis_oracle=False,
        docking_reward_config=None,
    )
    report.update(stage="search_finished", runner=result)
    save(output, report)
    if not instances or result.get("status") != "ok":
        report.update(status="failed", wall_seconds=time.monotonic()-started)
        save(output, report)
        return
    search = instances[0]
    seed_key = key(search.initial_state)
    gate_cache = {}

    def gate(state):
        state_key = key(state)
        if state_key not in gate_cache:
            typed = original_class._satisfies_predicates(search, state, search.target_predicates)
            binding = typecheck(state, search.binding_site, leaf_oracle_call=False, oracle=None)
            gate_cache[state_key] = {"typed": typed, "binding": bool(binding.success),
                                     "binding_reasons": binding.violated_constraints,
                                     "is_seed": state_key == seed_key}
        return gate_cache[state_key]

    def funnel(states):
        unique = {key(s): s for s in states if key(s) and key(s) != seed_key}
        typed = {k: s for k, s in unique.items() if gate(s)["typed"]}
        binding = {k: s for k, s in typed.items() if gate(s)["binding"]}
        return {"new_unique": len(unique), "typed_pass": len(typed),
                "typed_then_binding_pass": len(binding), "structures": sorted(unique),
                "accepted_structures": sorted(binding)}

    # Independent exhaustive ROOT enumeration, after the search, for true feasible
    # rule/partner counts. Exceptions remain distinct from legitimate nonmatches.
    root_rules = {}
    root_products = {}
    for name, rule in search.rules.items():
        row = {"attempts": 0, "nonmatching": 0, "exceptions": [], "feasible_pairs": 0,
               "product_instances": 0}
        states = []
        for tile in search._resolve_expand_tile_pool():
            row["attempts"] += 1
            try:
                products = rule.reduce((search.initial_state, tile))
            except Exception as exc:
                row["exceptions"].append({"tile": key(tile), "error": repr(exc)})
                continue
            row["nonmatching"] += int(not products)
            row["feasible_pairs"] += int(bool(products))
            row["product_instances"] += len(products)
            states.extend(products)
        root_products.update({key(s): s for s in states if key(s)})
        row["funnel"] = funnel(states)
        root_rules[name] = row
    leaves = [n.state for n in search.return_locals["leaves_by_smi"].values()]
    pre_top = [state for score, state in search.return_locals["candidates"]]
    returned_keys = [runner._structure_key(c["smiles"]) for c in result["candidates"]]
    depths = {}
    stack = [(search._root, 0)]
    seen_nodes = set()
    while stack:
        node, depth = stack.pop()
        if id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        depths.setdefault(str(depth), set()).add(key(node.state))
        stack.extend((child, depth+1) for child in node.children)
    scaffolds = set()
    for smi in returned_keys:
        mol = Chem.MolFromSmiles(smi)
        scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    report.update(
        status="completed", stage="complete", wall_seconds=time.monotonic()-started,
        root_enumeration={"per_rule": root_rules, "funnel": funnel(root_products.values())},
        search_observations={
            "scope": "all reduction calls, including expansion and rollout; overlapping with root enumeration",
            "per_rule": {name: {field: search.reduction_counts[(name, field)]
                        for field in ("calls", "nonfires", "product_instances")} for name in search.rules},
            "products_funnel": funnel(search.observed_products.values()),
            "expansions": search.expansions,
            "tree_unique_by_depth": {depth: sorted(values) for depth, values in depths.items()},
            "rollout_depth_hist": search.rollout_depth_hist,
        },
        final_selection={
            "leaf_registry_entries": len(leaves), "leaf_funnel": funnel(leaves),
            "pre_top_k_entries": len(pre_top), "pre_top_k_funnel": funnel(pre_top),
            "top_k": search.top_k, "truncated_entries": max(0, len(pre_top)-search.top_k),
            "returned_entries": len(returned_keys), "returned_unique": len(set(returned_keys)),
            "returned_structures": returned_keys, "returned_scaffolds": sorted(scaffolds),
            "scaffold_note": "Empty Murcko scaffold is retained for acyclic molecules.",
        },
        gate_verdicts=gate_cache,
        no_docking={"leaf_oracle_call_top_k_only": search.leaf_oracle_call_top_k_only,
                    "nfe_oracle": search.nfe_oracle, "docking_reward": result["docking_reward_report"]},
    )
    save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--worker-config", type=Path)
    args = parser.parse_args()
    if args.worker_config:
        config = json.loads(args.worker_config.read_text())
        try:
            worker(config, args.output)
        except Exception:
            partial = json.loads(args.output.read_text()) if args.output.exists() else {"config": config}
            partial.update(status="failed", error=traceback.format_exc())
            save(args.output, partial)
            raise
        return
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (args.output / "summary.json").exists():
        parser.error("Output already contains evidence; choose a fresh --output directory")
    cases = []
    # Actual N10 seeds/config first, then a complete 2 x 2 x 3 x 2 grid.
    for seed in (42, 0, 1234):
        cases.append(dict(tile_library="standard_12", click_rules="CuAAC", max_depth=1,
                          branching_target=12, seed=seed, n_simulations=4, top_k=10))
    for library, rules, depth, cap in itertools.product(
            ("standard_12", "extended_204"), ("CuAAC", "all_5"), (1, 2, 3), (60, 1020)):
        cases.append(dict(tile_library=library, click_rules=rules, max_depth=depth,
                          branching_target=cap, seed=42, n_simulations=4, top_k=10))
    if args.limit is not None:
        cases = cases[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), ROOT/"molmetal/scripts/lambda_100pocket_sweep.py",
             ROOT/"molmetal/molmetal_lam/search_alg/proof_search.py",
             ROOT/"molmetal/molmetal_lam/binding/types.py",
             ROOT/"molmetal/molmetal_lam/lam_chem/rules.py",
             ROOT/"molmetal/molmetal_lam/reactions/beta_reductions.py",
             ROOT/"molmetal/molmetal_lam/molecules/closed_term.py",
             ROOT/"molmetal/molmetal_lam/types/predicates.py",
             ROOT/"molmetal/molmetal_lam/scripts/_sweep_helpers.py",
             ROOT/"molmetal/molmetal_lam/search_alg/sweep_guidance.py",
             ROOT/"molmetal/molmetal_lam/tile_lib/library.py",
             ROOT/"molmetal/molmetal_lam/tile_lib/fragment_pool.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for source in files:
        snapshot = args.output / "source_snapshot" / source.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
    summary = {"protocol": "CPU RDKit, no pocket/ligand input, no docking/learned prior/synthesis gate; unchanged chemical/type/binding gates",
               "python": sys.version, "case_timeout_seconds": args.timeout,
               "runtime_sha256": hashes, "cases": []}
    for i, config in enumerate(cases):
        case_id = f"case_{i:02d}"
        output = args.output / f"{case_id}.json"
        config_path = args.output / f"{case_id}.config.json"
        save(config_path, config)
        start = time.monotonic()
        with (args.output / f"{case_id}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        "--worker-config", str(config_path), "--output", str(output)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            timed_out = False
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
        result = json.loads(output.read_text()) if output.exists() else {"config": config}
        if timed_out or process.returncode:
            result.update(status="timeout" if timed_out else "failed", process_returncode=process.returncode)
            save(output, result)
        row = {"case": case_id, "config": config, "status": result.get("status", "failed"),
               "process_wall_seconds": time.monotonic()-start}
        if result.get("status") == "completed":
            row.update(effective_branching=result["runner"]["search_config"]["branching_attempts"],
                       seed_smiles=result["runner"]["canonical_seed"],
                       root_funnel=result["root_enumeration"]["funnel"],
                       search_funnel=result["search_observations"]["products_funnel"],
                       final_selection=result["final_selection"])
        summary["cases"].append(row)
        save(args.output / "summary.json", summary)
        print(case_id, row["status"], round(row["process_wall_seconds"], 2), flush=True)
    summary["runtime_unchanged_during_run"] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
                                                for p, h in hashes.items())
    save(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
