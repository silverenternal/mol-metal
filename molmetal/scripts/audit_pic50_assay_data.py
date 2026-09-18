"""Audit assay pooling and censoring before retraining a metal-activity model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def audit(source, output):
    import numpy as np
    import pandas as pd
    from rdkit import Chem, rdBase
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, low_memory=False)
    frame['source_row'] = np.arange(len(frame)) + 2
    raw_name, value_name = 'IC50_Dark(M*10^-6)', 'IC50_Dark_value'
    frame['dark_censored'] = frame[raw_name].fillna('').astype(str).str.contains(r'[<>≤≥]', regex=True)
    frame['numeric_dark_uM'] = pd.to_numeric(frame[value_name], errors='coerce')
    canonical = {}
    with rdBase.BlockLogs():
        for value in frame['SMILES_Ligands'].dropna().unique():
            mol = Chem.MolFromSmiles(str(value))
            canonical[value] = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None and mol.GetNumAtoms() else None
    frame['canonical_smiles'] = frame['SMILES_Ligands'].map(canonical)
    ru = frame[frame['Metal'].astype(str).eq('Ru')].copy()
    positive = ru[np.isfinite(ru['numeric_dark_uM']) & ru['numeric_dark_uM'].gt(0)].copy()
    valid = positive[positive['canonical_smiles'].notna()].copy()
    valid['pIC50'] = 6 - np.log10(valid['numeric_dark_uM'])
    by_molecule = valid.groupby('canonical_smiles', sort=True)
    compound_rows = []
    for smi, group in by_molecule:
        compound_rows.append({'canonical_smiles': smi, 'n_rows':len(group),
            'n_cell_lines':group['Cell_line'].nunique(), 'n_exposure_times':group['Time(h)'].nunique(),
            'n_dois':group['DOI'].nunique(), 'n_counterions':group['Counterion'].nunique(),
            'n_oxidation_states':group['Oxidation_state'].nunique(),
            'n_censored_rows':int(group['dark_censored'].sum()),
            'pic50_span':float(group['pIC50'].max()-group['pIC50'].min()),
            'first_last_pic50_difference':float(group['pIC50'].iloc[-1]-group['pIC50'].iloc[0]),
            'first_source_row':int(group['source_row'].iloc[0]),
            'last_source_row':int(group['source_row'].iloc[-1])})
    compounds = pd.DataFrame(compound_rows)
    first = valid.drop_duplicates('canonical_smiles', keep='first')
    eligible = valid[~valid['dark_censored'] & valid['Cell_line'].notna() & valid['Time(h)'].notna()].copy()
    groups=[]
    for (cell, duration), group in eligible.groupby(['Cell_line','Time(h)'], sort=True):
        groups.append({'cell_line':str(cell), 'time_h':float(duration), 'n_rows':len(group),
                       'n_unique_molecules':group['canonical_smiles'].nunique(), 'n_dois':group['DOI'].nunique()})
    groups.sort(key=lambda g:(-g['n_unique_molecules'],g['cell_line'],g['time_h']))
    selected=first
    if len(first)>2000:
        indices=np.random.RandomState(42).choice(len(first),size=2000,replace=False)
        selected=first.iloc[sorted(indices)]
    counts={'all_rows':len(frame), 'ru_rows':len(ru), 'ru_positive_numeric_dark_rows':len(positive),
            'ru_valid_structure_positive_rows':len(valid), 'ru_unique_structures':len(compounds),
            'ru_numeric_but_censored_rows':int(positive['dark_censored'].sum()),
            'legacy_first_per_molecule_censored':int(first['dark_censored'].sum()),
            'legacy_sample2000_censored':int(selected['dark_censored'].sum()),
            'molecules_measured_in_multiple_cell_lines':int(compounds['n_cell_lines'].gt(1).sum()),
            'molecules_measured_at_multiple_exposures':int(compounds['n_exposure_times'].gt(1).sum()),
            'molecules_with_multiple_counterions':int(compounds['n_counterions'].gt(1).sum()),
            'molecules_with_multiple_oxidation_states':int(compounds['n_oxidation_states'].gt(1).sum()),
            'molecules_with_pic50_span_gt1':int(compounds['pic50_span'].gt(1).sum()),
            'first_last_label_change_gt0_5':int(compounds['first_last_pic50_difference'].abs().gt(.5).sum()),
            'uncensored_condition_known_rows':len(eligible)}
    frame.loc[frame['dark_censored'], ['source_row','Metal',raw_name,value_name,'Cell_line','Time(h)','DOI']].to_csv(output/'censored_rows.csv',index=False)
    compounds.to_csv(output/'compound_assay_variation.csv',index=False)
    eligible.to_csv(output/'ru_uncensored_conditioned_rows.csv',index=False)
    report={'source_csv':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'counts':counts, 'largest_condition_cohorts':groups[:20],
            'limitations':[
                'Censor detection uses explicit comparison symbols in the raw dark-assay field; ambiguous free text still needs separate curation.',
                'Compound grouping mirrors legacy canonical ligand SMILES, not a claim that counterion/oxidation/assay context is interchangeable.',
                'Within-compound variation includes different cell lines/exposures and is not labelled replicate measurement noise.',
                'Eligible rows are an audit export, not a train/test split; do not randomly split repeated compounds or tune cohorts using test performance.',
                'This does not retroactively change any trained checkpoint or metric. Retraining needs a new protocol and new output prefix.',
            ]}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Ru pIC50 assay audit','', '| Quantity | Count |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in counts.items()]
    lines += ['', '## Largest uncensored cell-line/exposure cohorts', '| Cell line | Hours | Rows | Unique structures | DOIs |', '|---|---:|---:|---:|---:|']
    lines += [f"| {g['cell_line']} | {g['time_h']} | {g['n_rows']} | {g['n_unique_molecules']} | {g['n_dois']} |" for g in groups[:10]]
    lines += ['', '## Interpretation', 'The old first-row-per-SMILES target can depend on CSV ordering and assay context. Censored bounds cannot be treated as exact IC50 regression labels.', '', *report['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(counts))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='/mnt/storage/data/molmetal/MetalCytoToxDB.csv')
    parser.add_argument('--output',default='molmetal/reports/pic50_assay_audit_20260913')
    args=parser.parse_args()
    audit(args.source,args.output)
