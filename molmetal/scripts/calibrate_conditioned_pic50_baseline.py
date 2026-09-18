"""New uncensored, assay-conditioned, scaffold-split GPU ridge baseline.

Historical pooled-target checkpoints are untouched. This baseline provides a
clean target/split control, not validation of anticancer efficacy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.model_selection import GroupShuffleSplit
    from scipy.stats import pearsonr, spearmanr
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('molmetal/reports/pic50_assay_audit_20260913/ru_uncensored_conditioned_rows.csv'))
    parser.add_argument('--cell-line',default='HeLa')
    parser.add_argument('--time-h',type=float,default=48.)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,default=Path('molmetal/reports/pic50_conditioned_baseline_20260913'))
    args=parser.parse_args()
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('This protocol requires an actual GPU; no silent CPU fit fallback')
    source=pd.read_csv(args.input)
    selected=source[source['Cell_line'].eq(args.cell_line)&source['Time(h)'].eq(args.time_h)].copy()
    if selected['dark_censored'].astype(str).str.lower().eq('true').any():
        raise ValueError('Censored rows must not become exact regression labels')
    selected['Counterion']=selected['Counterion'].fillna('')
    keys=['canonical_smiles','Counterion','Metal','Oxidation_state','Charge_complex']
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    records=[]; fingerprints=[]; rejected=[]
    with rdBase.BlockLogs():
        for key, group in selected.groupby(keys,dropna=False,sort=True):
            ligand,counterion,metal,oxidation,charge=key
            combined=ligand+('.'+counterion if counterion else '')
            molecule=Chem.MolFromSmiles(combined)
            parent=Chem.MolFromSmiles(ligand)
            if molecule is None or parent is None or not np.isfinite([oxidation,charge]).all():
                rejected.append({'source_rows':group['source_row'].astype(int).tolist(),'reason':'invalid_formulation_or_missing_metadata'})
                continue
            try:
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=parent,includeChirality=False)
                fingerprint=np.zeros(2048,dtype=np.float32)
                DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule),fingerprint)
            except Exception as exc:
                rejected.append({'source_rows':group['source_row'].astype(int).tolist(),'reason':str(exc)})
                continue
            records.append({'ligand_smiles':ligand,'counterion':counterion,'metal':metal,
                            'oxidation_state':float(oxidation),'charge':float(charge),
                            'scaffold':scaffold or '<acyclic>',
                            'pIC50':float(group['pIC50'].median()),'n_replicates':len(group),
                            'replicate_pic50_span':float(group['pIC50'].max()-group['pIC50'].min()),
                            'source_rows':group['source_row'].astype(int).tolist(),
                            'dois':sorted(group['DOI'].dropna().astype(str).unique().tolist())})
            fingerprints.append(np.concatenate([fingerprint,np.asarray([oxidation,charge],dtype=np.float32)]))
    if len(records)<30:
        raise ValueError('Insufficient conditioned cohort for this protocol')
    values=np.asarray([r['pIC50'] for r in records],dtype=np.float32)
    groups=np.asarray([r['scaffold'] for r in records])
    x=torch.as_tensor(np.stack(fingerprints),device=device)
    y=torch.as_tensor(values,device=device)
    def metrics(pred, truth):
        pred=np.asarray(pred,dtype=float); truth=np.asarray(truth,dtype=float)
        varied=np.std(pred)>0 and np.std(truth)>0
        return {'n':len(truth),'rmse':float(np.sqrt(np.mean((pred-truth)**2))),
                'mae':float(np.mean(np.abs(pred-truth))),
                'pearson':float(pearsonr(pred,truth).statistic) if varied else None,
                'spearman':float(spearmanr(pred,truth).statistic) if varied else None}
    results=[]
    for seed in (42,0,1234):
        train, rest=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(values,groups=groups))
        val_local,test_local=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(values[rest],groups=groups[rest]))
        val,test=rest[val_local],rest[test_local]
        splits={'train':train,'validation':val,'test':test}
        group_sets=[set(groups[index]) for index in splits.values()]
        if any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError('Scaffold leakage')
        # One ligand can have multiple salts/metadata; grouping by its scaffold
        # must keep all those formulations in a single partition.
        ligand_sets=[{records[i]['ligand_smiles'] for i in index} for index in splits.values()]
        if any(ligand_sets[i]&ligand_sets[j] for i in range(3) for j in range(i+1,3)):
            raise ValueError('Ligand leakage')
        xmean=x[train].mean(0); ymean=y[train].mean()
        xt=x[train]-xmean
        kernel=xt@xt.T
        trials=[]; candidates=[]
        for alpha in (1.,10.,100.,1000.):
            dual=torch.linalg.solve(kernel+alpha*torch.eye(len(train),device=device),y[train]-ymean)
            val_pred=((x[val]-xmean)@xt.T@dual+ymean).cpu().numpy()
            trial=metrics(val_pred,values[val]);trials.append({'alpha':alpha,'validation':trial})
            candidates.append(dual)
        best=min(range(len(trials)),key=lambda i:trials[i]['validation']['rmse'])
        test_pred=((x[test]-xmean)@xt.T@candidates[best]+ymean).cpu().numpy()
        results.append({'seed':seed,'split_indices':{k:v.tolist() for k,v in splits.items()},
                        'split_scaffold_counts':{k:len(set(groups[v])) for k,v in splits.items()},
                        'same_ligand_split_overlap':False,'same_scaffold_split_overlap':False,
                        'validation_trials':trials,'chosen_alpha':trials[best]['alpha'],
                        'test':metrics(test_pred,values[test]),
                        'train_mean_baseline_test':metrics(np.full(len(test),float(ymean)),values[test]),
                        'test_predictions':test_pred.tolist()})
    torch.cuda.synchronize(device)
    properties=torch.cuda.get_device_properties(device)
    report={'scope':'uncensored fixed-assay scaffold holdout baseline; not anticancer efficacy validation',
            'protocol':{'cell_line':args.cell_line,'time_h':args.time_h,'assay':'dark',
                        'target':'median pIC50 within identical ligand/counterion/metal/oxidation/charge and assay condition',
                        'fingerprints':'RDKit Morgan binary radius2/2048 on ligand + explicit counterion; oxidation and charge appended',
                        'model':'GPU dual ridge regression; training-only feature/target centering',
                        'split':'80/10/10 scaffold-group proportions; actual row sizes vary',
                        'selection':'largest uncensored unique-structure cell/time cohort from audit, not selected by test performance'},
            'input_csv':str(args.input.resolve()),'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'n_input_condition_rows':len(selected),'n_formulations':len(records),
            'n_scaffolds':len(set(groups)),'n_rejected_groups':len(rejected),'rejected':rejected,
            'cohort':records,'results':results,
            'execution':{'device':str(device),'gcn_architecture':getattr(properties,'gcnArchName',None),
                         'torch':torch.__version__,'hip':torch.version.hip,
                         'feature_tensor_device':str(x.device),'last_dual_tensor_device':str(candidates[-1].device)},
            'limits':['No comparison to old pooled/censored random-split metrics: targets and splits differ.',
                      'Scaffold proportions are group counts, not guaranteed row proportions; no test-driven rebalancing.',
                      'Known assay context is limited to available metadata; cross-study measurement differences remain.',
                      'This is one Ru/cell/time cohort; other metals, cell lines, temporal or publication holdouts remain unvalidated.']}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Conditioned Ru pIC50 baseline','',f"{args.cell_line}, {args.time_h:g}h, dark; {len(records)} formulations / {len(set(groups))} scaffolds.",'',
           '| Seed | Train/val/test rows | Ridge RMSE | Mean baseline RMSE | Pearson |','|---|---|---:|---:|---:|']
    for r in results:
        n='/'.join(str(len(v)) for v in r['split_indices'].values())
        lines.append(f"| {r['seed']} | {n} | {r['test']['rmse']:.4f} | {r['train_mean_baseline_test']['rmse']:.4f} | {r['test']['pearson']:.4f} |")
    lines+=['',*report['limits']]
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'n_formulations':len(records),'n_scaffolds':len(set(groups)),
                      'results':[{'seed':r['seed'],'test':r['test'],'mean_baseline':r['train_mean_baseline_test']} for r in results]}))


if __name__=='__main__':
    main()
