"""Summarize measured poses with explicit denominators and GPU diversity."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import statistics


def moments(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'std_sample': statistics.stdev(values) if len(values)>1 else None}


def build_report(payload, device='auto'):
    from molmetal.validation.gpu_molecular_metrics import molecular_diversity
    rows = payload['per_pocket']
    report = {'scope': 'bounded physical integration experiment, not SOTA protocol completion',
              'protocol': payload['metadata'], 'jobs': [], 'total_jobs': len(rows),
              'generation_scope': 'pocket-independent click generation, then per-pocket docking; no pocket-conditioned search reward',
              'novelty_vs_training_set': None,
              'novelty_note': 'Training fingerprints not supplied; new-to-seed does not prove training novelty.'}
    all_scores = []
    if payload.get('metadata', {}).get('search', {}).get('docking_reward_config') is not None:
        measured_jobs = sum(job.get('diagnostics', {}).get('docking_reward', {}).get('applied') is True for job in rows)
        report['generation_scope'] = f'Pocket-conditioned measured docking reward requested; {measured_jobs}/{len(rows)} jobs report actual in-search docking; post-search evaluation is separate'
    def generated_candidates(job):
        return [c for c in job.get('diagnostics', {}).get('all_candidates', job.get('candidates', []))
                if c.get('is_generated') is True]
    for job in rows:
        physical = job.get('physical', {})
        candidates = physical.get('candidates', [])
        scores = [c.get('score_kcal_mol') for c in candidates]
        all_scores += scores
        best_to_worst = sorted((c for c in candidates if c.get('score_kcal_mol') is not None), key=lambda c:c['score_kcal_mol'])
        summary = {'n_generated': job.get('n_generated_candidates', sum(c.get('is_generated') is True for c in job['candidates'])),
                   'n_selected': 0, 'n_docked': 0, 'n_pb_pass': 0,
                   'n_beats_redocked_reference': 0, 'n_triple_threshold_redocked_reference': 0,
                   'n_vina_below_minus8': 0, **physical.get('summary', {})}
        # A missing evaluator/import/crashed stage is still part of coverage.
        # Recover the search count even if no physical progress report exists.
        search_total = job.get('n_generated_candidates', len(generated_candidates(job)))
        if summary['n_generated'] < search_total:
            summary['physical_report_denominator'] = summary['n_generated']
            summary['n_generated'] = search_total
            summary['denominator_correction'] = 'Search total includes candidates rejected or not selected before physical evaluation'
        summary['pb_pass_rate_all_generated'] = summary['n_pb_pass']/summary['n_generated'] if summary['n_generated'] else None
        summary['n_reference_comparisons_available'] = sum(isinstance(c.get('beats_redocked_reference'), bool) for c in candidates)
        summary['n_reference_comparisons_unavailable'] = summary['n_generated'] - summary['n_reference_comparisons_available']
        reference_pb = (physical.get('reference') or {}).get('posebusters', {})
        report['jobs'].append({
            'pocket_id':job['pocket_id'], 'seed':job['seed'], 'search_status':job['status'],
            'physical_status':physical.get('status'), 'summary':summary,
            'reference_score':(physical.get('reference') or {}).get('score_kcal_mol'),
            'reference_posebusters':reference_pb,
            'vina':moments(scores),
            'descriptors':{key:moments([c.get('descriptors',{}).get(key) for c in candidates])
                           for key in ('sa','qed','logp','tpsa','rotatable_bonds','molecular_weight')},
            'diversity':molecular_diversity([c['smiles'] for c in generated_candidates(job)],device=device),
            'selection_counts':job.get('diagnostics', {}).get('selection_counts', {}),
            'best':best_to_worst[0] if best_to_worst else None,
            'worst':best_to_worst[-1] if best_to_worst else None,
            'pb_failures':dict(Counter(f for c in candidates for f in c.get('posebusters',{}).get('failures',[]))),
            'preparation_error':physical.get('receptor_preparation',{}).get('error'),
        })
    totals = {key:sum(j['summary'].get(key,0) for j in report['jobs'])
              for key in ('n_generated','n_selected','n_docked','n_pb_pass','n_beats_redocked_reference',
                          'n_triple_threshold_redocked_reference','n_vina_below_minus8',
                          'n_reference_comparisons_available','n_reference_comparisons_unavailable')}
    totals['vina'] = moments(all_scores)
    totals['generated_structure_diversity'] = molecular_diversity(
        [c['smiles'] for row in rows for c in generated_candidates(row)], device=device)
    totals['physical_status_counts'] = dict(Counter(j['physical_status'] for j in report['jobs']))
    totals['pb_pass_rate_all_generated'] = totals['n_pb_pass']/totals['n_generated'] if totals['n_generated'] else None
    totals['reference_posebusters_status_counts'] = dict(Counter(j['reference_posebusters'].get('status','unavailable') for j in report['jobs']))
    valid_reference_jobs = [j for j in report['jobs'] if j['reference_posebusters'].get('pb_valid') is True]
    totals['reference_pb_valid_subset'] = {
        'n_jobs':len(valid_reference_jobs),
        **{key:sum(j['summary'].get(key,0) for j in valid_reference_jobs) for key in (
            'n_generated','n_docked','n_beats_redocked_reference','n_triple_threshold_redocked_reference')},
    }
    gpu_records=[c.get('gpu_execution', {}) for job in rows
                 for c in (job.get('physical',{}).get('candidates',[]) +
                           ([job['physical']['reference']] if job.get('physical',{}).get('reference') else []))
                 if c.get('gpu_execution')]
    totals['gpu_execution']={
        'records':len(gpu_records), 'verified':sum(g.get('gpu_verified') is True for g in gpu_records),
        'kernel_trace_verified':sum(g.get('gpu_verified') is True and g.get('gpu_evidence')=='kernel_trace' for g in gpu_records),
        'devices':sorted({d for g in gpu_records for d in g.get('device_observed',[])}),
        'timing_note':'GPU event timestamps invalid on this driver; no speedup inferred from these experiments.'}
    report['aggregate'] = totals
    report['per_seed'] = []
    for seed in sorted({j['seed'] for j in report['jobs']}):
        subset = [j for j in report['jobs'] if j['seed']==seed]
        report['per_seed'].append({'seed':seed, 'n_jobs':len(subset),
                                  'pockets_with_vina':[j['pocket_id'] for j in subset if j['vina']['n']>0],
                                  'mean_pocket_vina':moments([j['vina']['mean'] for j in subset]),
                                  'mean_pocket_pb_rate_all_generated':moments([j['summary'].get('pb_pass_rate_all_generated') for j in subset])})
    common = set.intersection(*(set(s['pockets_with_vina']) for s in report['per_seed'])) if report['per_seed'] else set()
    report['common_pockets_across_seeds'] = sorted(common)
    report['common_pocket_seed_means'] = []
    for seed in sorted({j['seed'] for j in report['jobs']}):
        subset = [j for j in report['jobs'] if j['seed']==seed and j['pocket_id'] in common]
        report['common_pocket_seed_means'].append({
            'seed':seed, 'mean_pocket_vina':moments([j['vina']['mean'] for j in subset]),
            'mean_pocket_pb_rate_all_generated':moments([j['summary'].get('pb_pass_rate_all_generated') for j in subset])})
    report['seed_variation'] = {key:moments([s[key]['mean'] for s in report['common_pocket_seed_means']])
                               for key in ('mean_pocket_vina','mean_pocket_pb_rate_all_generated')}
    report['seed_variation_scope'] = 'Only the common pocket subset with measured Vina for every requested seed; coverage reported separately.'
    report['statistics_note'] = 'Sample SD, explicit finite n; no p-value against published aggregate numbers or claim of independent pocket/seed samples.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    payload=json.loads(args.input.read_text())
    report=build_report(payload,args.device)
    report['source_json']=str(args.input.resolve())
    report['source_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    report['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    total=report['aggregate']
    lines=['# Measured click-product physical evaluation','',
           '## Goal','Validate generated products on exact CrossDocked pairs with saved docking poses and strict PoseBusters checks.','',
           '## Outcome',f"{report['total_jobs']} jobs; {total['n_generated']} generated products; {total['n_docked']} docked; {total['n_pb_pass']} pass PoseBusters.",
           f"Unique generated structures across jobs: {total['generated_structure_diversity']['n_unique']}. Candidate instances repeated across pockets/seeds remain separate docking observations.",
           f"Physical job status counts: {total['physical_status_counts']}",
           f"Products beating a redocked reference: {total['n_beats_redocked_reference']}; triple threshold successes: {total['n_triple_threshold_redocked_reference']}.",
           f"Reference PoseBusters statuses: {total['reference_posebusters_status_counts']}; reference comparisons unavailable: {total['n_reference_comparisons_unavailable']}.",
           f"Reference-PB-passing subset: {total['reference_pb_valid_subset']}.",
           f"GPU execution evidence: {total['gpu_execution']}.",
           '', '| pocket | seed | physical status | docked/generated | Vina mean | PB pass/generated | reference PB |',
           '|---|---:|---|---:|---:|---:|---|']
    for j in report['jobs']:
        s=j['summary']; v=j['vina']['mean']; v=f'{v:.3f}' if v is not None else 'unavailable'
        lines.append(f"| {j['pocket_id']} | {j['seed']} | {j['physical_status']} | {s.get('n_docked',0)}/{s.get('n_generated',0)} | {v} | {s.get('n_pb_pass',0)}/{s.get('n_generated',0)} | {j['reference_posebusters'].get('status','unavailable')} |")
    lines += ['', '## Caveats', report['generation_scope']+'.', report['scope']+'.', report['novelty_note'],
              'Native Vina/QuickVina are CPU engines. QuickVina2-GPU uses OpenCL grid/search when selected; preparation, final refinement and PoseBusters retain CPU stages. Pairwise fingerprint similarity uses the recorded torch device.',
              'Unprepared receptors and missing poses remain in all-generated denominators. Descriptor/Vina means include only measured finite values and report n in JSON.',
              'Reference thresholds use redocked ligand scores, not co-crystal measured affinity. No significance test against cite-only means is performed.',
              'Some redocked reference poses can fail PB; the report retains their raw score comparisons and separately reports the reference-PB-passing subset. Unavailable comparisons are not observed failures.',
              report['seed_variation_scope'],
              '', '## Reproduce', f'Input: `{args.input}`; SHA256 `{report["source_sha256"]}`.',
              'The JSON includes exact generation/docking budgets, source/config hashes, per-job pose paths and GPU diversity backend.']
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    scores=[c['score_kcal_mol'] for j in payload['per_pocket'] for c in j.get('physical',{}).get('candidates',[]) if c.get('score_kcal_mol') is not None]
    if scores:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        ax.hist(scores,bins=min(15,max(3,len(scores)//3)),edgecolor='white')
        ax.set(xlabel='Docking score (kcal/mol)',ylabel='Measured generated poses',title='Bounded click-product experiment')
        fig.tight_layout()
        fig.savefig(args.output_prefix.with_suffix('.png'),dpi=180)
        plt.close(fig)
    print(json.dumps(total,allow_nan=False))


if __name__=='__main__':
    main()
