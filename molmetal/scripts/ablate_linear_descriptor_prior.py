"""Small frozen-prior ablation using reagent-generated development data only.

No CrossDocked/test ligand files, receptor files or physical docking scores
are used. Development reagents exclude the complete evaluation tile library.
This measures algorithmic behavior and search reward, not binding quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'molmetal')]

from rdkit import Chem, rdBase
from molmetal.scripts import lambda_100pocket_sweep as runner
from molmetal_lam.search_alg import sweep_guidance as guidance


def key(term):
    return runner._structure_key(runner._smi_of(term))


def dependency_hashes():
    files = ['molmetal/scripts/lambda_100pocket_sweep.py',
             'molmetal/molmetal_lam/search_alg/sweep_guidance.py',
             'molmetal/molmetal_lam/search_alg/proof_search.py',
             'molmetal/molmetal_lam/reactions/beta_reductions.py']
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}


def scorer():
    reward = replace(runner._build_reward(), r_vina_proxy=runner._vina_proxy, w_vina_proxy=0.0)
    return runner.MCTSProofSearch(tile_library=[], rules={}, target_predicates=[runner.LIPINSKI],
        binding_site=runner.PROTEASE_GENERIC, reward=reward, use_fragment_pool=False,
        prior_refit_every=0)


def make_development_data(count, training_seed, evaluation_tiles):
    eval_keys = {runner._structure_key(t.smiles) for t in evaluation_tiles}
    reagents = {}
    for tile in runner.FRAGMENT_LIBRARY_200_TILES():
        term = runner.MoleculeClosedTerm.from_smiles(tile.smiles, embed_3d=False)
        identity = key(term)
        if identity and identity not in eval_keys:
            reagents[identity] = term
    rule = runner.REACTION_RULES['CuAAC']
    template = rule._rdkit_reaction_template()
    left = [t for t in reagents.values() if Chem.MolFromSmiles(key(t)).HasSubstructMatch(template.GetReactantTemplate(0))]
    right = [t for t in reagents.values() if Chem.MolFromSmiles(key(t)).HasSubstructMatch(template.GetReactantTemplate(1))]
    pairs = list(itertools.product(left, right))
    random.Random(training_seed).shuffle(pairs)
    products, provenance = {}, []
    attempts = 0
    for a, b in pairs:
        attempts += 1
        try:
            made = rule.reduce((a, b))
        except Exception:
            continue
        for product in made:
            identity = key(product)
            if not identity or identity in products or identity in {key(a), key(b)}:
                continue
            products[identity] = product
            provenance.append({'product': identity, 'rule': 'CuAAC',
                               'reactants': [key(a), key(b)], 'source': 'extended_tile_library',
                               'evaluation_reagents_excluded': True})
            if len(products) >= count:
                break
        if len(products) >= count:
            break
    observed = guidance.candidate_observations(list(products.values()), scorer(), 'reagent_development_partition')
    return observed, provenance, {'reaction_attempts': attempts, 'unique_products': len(products),
                                  'left_reagents': len(left), 'right_reagents': len(right)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'molmetal/reports/linear_prior_development_ablation_20260913')
    parser.add_argument('--development-count', type=int, default=32)
    parser.add_argument('--training-seed', type=int, default=20260913)
    parser.add_argument('--simulations', type=int, default=8)
    parser.add_argument('--depth', type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_hashes = dependency_hashes()
    started = time.monotonic()
    with rdBase.BlockLogs():
        eval_tiles = runner.build_tile_library(12, include_fragments=False)
        observations, provenance, development = make_development_data(args.development_count, args.training_seed, eval_tiles)
        state, _ = guidance.prepare_prior(None, seed=args.training_seed, mode='train', data_split='development')
        state, fit_report = guidance.update_prior(state, observations, mode='train', data_split='development', refit_every=1)
        if not fit_report['fit_performed']:
            raise RuntimeError(f'Development prior did not fit: {fit_report}')
        immutable_state = deepcopy(state)
        training_keys = {p['product'] for p in provenance}
        labels = scorer()
        records = []
        for seed in (42, 0, 1234):
            pair = []
            for enabled in (False, True):
                if dependency_hashes() != initial_hashes:
                    raise RuntimeError('Runtime dependencies changed during ablation')
                wall = time.monotonic()
                result = runner.run_one_pocket(str(args.output_dir / 'unconditioned_evaluation'),
                    args.simulations, args.depth, 0, False, tile_library='standard_12',
                    click_rules='CuAAC', seed_strategy='click_tile', seed=seed, early_stop=False,
                    symbolic_prior=enabled, prior_state=deepcopy(state) if enabled else None,
                    prior_mode='frozen', prior_data_split='test', synthesis_oracle=False)
                rewards = []
                compounds = []
                for candidate in result.get('candidates', []):
                    if not candidate.get('is_generated'):
                        continue
                    term = runner.MoleculeClosedTerm.from_smiles(candidate['smiles'], embed_3d=False)
                    rewards.append(labels.score_final(term))
                    compounds.append(key(term))
                if enabled and result.get('prior_state') != immutable_state:
                    raise AssertionError('Frozen prior state changed during evaluation')
                record = {'seed': seed, 'prior_enabled': enabled, 'status': result['status'],
                    'canonical_seed': result.get('canonical_seed'), 'chosen_tile': result.get('chosen_tile'),
                    'n_candidates': result.get('n_candidates', 0), 'n_generated': result.get('n_generated_candidates', 0),
                    'generated_smiles': compounds, 'reward_values': rewards,
                    'reward_mean': statistics.mean(rewards) if rewards else None,
                    'reward_best': max(rewards) if rewards else None,
                    'development_overlap': sorted(set(compounds) & training_keys),
                    'diagnostics': result.get('search_diagnostics', {}),
                    'prior_report': result.get('prior_report', {}),
                    'wall_seconds': time.monotonic() - wall}
                if enabled and record['prior_report'].get('puct_calls', 0) == 0:
                    raise AssertionError('Fitted prior never reached PUCT')
                pair.append(record)
                records.append(record)
            if pair[0]['canonical_seed'] != pair[1]['canonical_seed']:
                raise AssertionError('Paired off/on arms used different initialization')
    output = {'experiment': 'linear_descriptor_prior_development_then_frozen',
              'training_seed': args.training_seed, 'evaluation_seeds': [42, 0, 1234],
              'n_simulations': args.simulations, 'max_depth': args.depth,
              'data_protocol': 'disjoint development/evaluation reagent libraries; no reference ligand or docking labels',
              'development': development, 'fit_report': fit_report,
              'runtime_sha256': initial_hashes, 'records': records,
              'wall_seconds': time.monotonic() - started}
    for filename, data in [('development_observations.json', {'observations': observations, 'provenance': provenance,
          'training_seed': args.training_seed}), ('prior_state.json', state), ('results.json', output)]:
        (args.output_dir / filename).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    lines = ['# Linear descriptor prior: development fit and frozen off/on ablation', '',
        'This is a reagent-only algorithm check, not CrossDocked or physical binding evaluation. '
        'No test reference structures or docking scores were used as training inputs or labels.', '',
        f'Development: {len(observations)} real CuAAC product observations; training seed {args.training_seed}. '
        'Reagents present in the evaluation standard-12 library were excluded from development.', '',
        f'Frozen evaluation: seeds 42, 0, 1234; {args.simulations} simulations, depth {args.depth}; '
        'the same initialization is paired off/on within each seed. LIPINSKI and PROTEASE_GENERIC gates are unchanged.', '',
        '| seed | prior | PUCT calls | NFE reductions | candidates | generated | mean reward | best reward |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        def fmt(v): return 'unavailable' if v is None else f'{v:.6f}'
        lines.append(f"| {r['seed']} | {'on' if r['prior_enabled'] else 'off'} | "
            f"{r['prior_report'].get('puct_calls', 0)} | {r['diagnostics'].get('nfe_reductions', 0)} | "
            f"{r['n_candidates']} | {r['n_generated']} | {fmt(r['reward_mean'])} | {fmt(r['reward_best'])} |")
    lines += ['', 'The reward is the existing SA/QED descriptor-based search objective with type/binding '
        'bonuses; it is not experimental affinity. This backend is a Ridge linear descriptor model, '
        'not PySR symbolic discovery. All frozen state comparisons were exact and the on arms invoked '
        'the actual PUCT prior dispatcher.', '']
    improvements = []
    for off, on in zip(records[::2], records[1::2]):
        if off['reward_mean'] is not None and on['reward_mean'] is not None:
            improvements.append(on['reward_mean'] - off['reward_mean'])
    lines.append(f'Paired mean-reward deltas (on − off): {improvements}. '
                 'Zero or negative differences do not establish improvement; this small check does not support physical-quality claims.')
    lines.append(f"Generated molecule overlap with development products: {sum(len(r['development_overlap']) for r in records)}.")
    lines.append(f'Total elapsed time: {output["wall_seconds"]:.2f} seconds.')
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'output_dir': str(args.output_dir), 'development_observations': len(observations),
                      'paired_reward_deltas': improvements, 'wall_seconds': output['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
