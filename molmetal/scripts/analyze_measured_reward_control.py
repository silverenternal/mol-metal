"""Compare actual equal-limit oracle controls; distinguish selection from diversity."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def protocol(payload):
    result = deepcopy(payload['metadata'])
    result.pop('guidance_execution', None)
    search = result['search']
    reward = search['docking_reward_config']
    reward.pop('weight')
    reward.pop('output_dir')
    search['physical_config'].pop('output_dir')
    return result


def analyze(zero_path, feedback_path):
    zero, feedback = [json.loads(Path(p).read_text()) for p in (zero_path, feedback_path)]
    if protocol(zero) != protocol(feedback):
        raise ValueError('Treatments differ in more than reward weight and output paths')
    def keyed(payload):
        rows = payload['per_pocket']
        result = {(r['pocket_id'], r['seed']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate pocket/seed records')
        return result
    a, b = keyed(zero), keyed(feedback)
    if set(a) != set(b):
        raise ValueError('Different pocket/seed coverage')
    comparisons = []
    for key in a:
        arms = []
        for job, weight in ((a[key], 0), (b[key], .4)):
            reward = job['diagnostics']['docking_reward']
            if reward['weight'] != weight or reward['status'] != 'executed':
                raise ValueError('Each treatment must execute its explicitly declared measured oracle')
            execution = reward['execution']
            if execution['n_docked'] < 1:
                raise ValueError('No actual in-search docking')
            physical = job['physical']
            if physical['status'] != 'completed':
                raise ValueError('Physical evaluation incomplete; do not hide failed paired jobs')
            poses = physical['candidates']
            for pose in poses + [physical['reference']] + execution['candidates']:
                if pose.get('score_kcal_mol') is not None and not pose.get('gpu_execution', {}).get('gpu_verified'):
                    raise ValueError('Unverified GPU measurement')
            generated = job['diagnostics'].get('all_candidates', job['candidates'])
            arms.append({'weight': weight,
                         'generated_smiles': sorted({c['smiles'] for c in generated if c['is_generated']}),
                         'n_generated': job['n_generated_candidates'],
                         'n_docked': physical['summary']['n_docked'],
                         'n_pb_pass': physical['summary']['n_pb_pass'],
                         'top1_smiles': poses[0]['smiles'] if poses else None,
                         'top1_evaluation_score': poses[0]['score_kcal_mol'] if poses else None,
                         'mean_evaluation_score': mean([c['score_kcal_mol'] for c in poses]),
                         'search_oracle_attempts': execution['n_attempted'],
                         'search_oracle_cache_hits': execution['n_cache_hits'],
                         'search_oracle_budget_exhausted': execution['n_budget_exhausted'],
                         'worker_wall_seconds': job['wall_seconds']})
        off, on = arms
        comparisons.append({'pocket_id': key[0], 'seed': key[1], 'zero_weight': off, 'feedback': on,
                            'same_generated_set': off['generated_smiles'] == on['generated_smiles'],
                            'same_actual_oracle_attempts': off['search_oracle_attempts'] == on['search_oracle_attempts'],
                            'top1_changed': off['top1_smiles'] != on['top1_smiles'],
                            'top1_score_delta_feedback_minus_zero': on['top1_evaluation_score']-off['top1_evaluation_score'],
                            'mean_score_delta_feedback_minus_zero': on['mean_evaluation_score']-off['mean_evaluation_score']})
    pocket_means = [{'pocket_id': p, 'top1_paired_delta': mean([r['top1_score_delta_feedback_minus_zero'] for r in comparisons if r['pocket_id']==p])}
                    for p in sorted({r['pocket_id'] for r in comparisons})]
    return {'scope': 'bounded real docking feedback control; no full-pilot or SOTA claim',
            'source': [{'path':str(Path(p).resolve()), 'sha256':sha(p)} for p in (zero_path, feedback_path)],
            'analysis_source_sha256':sha(__file__),
            'n_pocket_seed_pairs':len(comparisons), 'n_pockets':len(pocket_means),
            'all_actual_oracle_attempts_equal':all(r['same_actual_oracle_attempts'] for r in comparisons),
            'all_generated_sets_equal':all(r['same_generated_set'] for r in comparisons),
            'top1_changed_pairs':sum(r['top1_changed'] for r in comparisons),
            'mean_top1_paired_delta':mean([r['top1_score_delta_feedback_minus_zero'] for r in comparisons]),
            'mean_all_pose_paired_delta':mean([r['mean_score_delta_feedback_minus_zero'] for r in comparisons]),
            'pocket_means':pocket_means, 'paired_results':comparisons,
            'limitations':[
                'Only two pockets and three nested seeds; no independent-six-sample significance claim or generalization conclusion.',
                'Weight zero still measures the oracle: same configured limits, chemistry, seeds and post-search protocol. Actual attempts are checked separately.',
                'Changed top-1 selection is distinct from generation diversity; unchanged generated sets do not prove expanded chemical exploration.',
                'Evaluation repeats docking separately from the in-search cached score, on the same engine; no independent physical affinity confirmation.',
                'Wall time includes evaluation and reflects concurrent host activity; no hardware speedup benchmark.',
                'Shared references/environment are listed in the snapshot manifest; this experiment does not retrain or load a learned generator.',
            ]}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zero', type=Path)
    parser.add_argument('feedback', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args=parser.parse_args()
    report=analyze(args.zero,args.feedback)
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Measured GPU reward control', '',
           f"{report['n_pockets']} pockets / {report['n_pocket_seed_pairs']} paired seed jobs.",
           f"Actual oracle attempt counts equal in every pair: {report['all_actual_oracle_attempts_equal']}.",
           f"Generated sets equal in every pair: {report['all_generated_sets_equal']}.",
           f"Top-1 changes: {report['top1_changed_pairs']}; mean top-1 score delta (feedback minus zero): {report['mean_top1_paired_delta']:.6f} kcal/mol.",
           f"Mean all-pose score delta: {report['mean_all_pose_paired_delta']:.6f} kcal/mol.", '',
           '| Pocket | Seed | Zero top-1 | Feedback top-1 | Delta | Same generated set |',
           '|---|---:|---:|---:|---:|---|']
    for r in report['paired_results']:
        lines.append(f"| {r['pocket_id']} | {r['seed']} | {r['zero_weight']['top1_evaluation_score']} | {r['feedback']['top1_evaluation_score']} | {r['top1_score_delta_feedback_minus_zero']:.3f} | {r['same_generated_set']} |")
    lines+=['', '## Limits', *['- '+x for x in report['limitations']]]
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('n_pocket_seed_pairs','all_actual_oracle_attempts_equal','all_generated_sets_equal','mean_top1_paired_delta')}))
