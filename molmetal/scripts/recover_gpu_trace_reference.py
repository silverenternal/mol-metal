"""Recover saved GPU reference outputs rejected only by progress/trace framing.

Never changes the source experiment. Verifies original artifact hashes, device,
kernel completion, energy and chemical identity before generating derived
postprocessing output; no search or docking is repeated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(source: Path, output: Path):
    from rdkit import Chem
    from meeko import PDBQTMolecule, RDKitMolCreate
    from molmetal_lam.sbdd_env.vina_adapter import _parse_vina_result_energies
    from molmetal.scripts.evaluate_generated_poses import summarize
    from molmetal.validation.posebusters_runner import check_docked_pose
    import numpy as np
    if source.resolve()==output.resolve():
        raise ValueError('Derived recovery output must not overwrite original experiment')
    payload=json.loads(source.read_text())
    recoveries=[]
    for job in payload['per_pocket']:
        physical=job.get('physical',{})
        reference=physical.get('reference') or {}
        if not reference.get('error','').startswith('RuntimeError: No successful kernel2 OpenCL execution trace;'):
            continue
        execution=reference['gpu_execution']
        if execution['returncode']!=0:
            raise ValueError('Cannot recover an engine execution failure')
        log=Path(execution['log_path'])
        if digest(log)!=execution['log_sha256']:
            raise ValueError('Engine log changed after experiment')
        traces=re.findall(r'OPENCL_TRACE [^\r\n]*',log.read_text())
        device=execution['expected_device']
        for kernel in ('kernel1','kernel2'):
            if not any(f'kernel={kernel} device={device} enqueue=0 wait=0 status=0' in t for t in traces):
                raise ValueError('Complete device/kernel execution evidence is required')
        pdbqt=Path(execution['output_pdbqt'])
        text=pdbqt.read_text()
        energies=_parse_vina_result_energies(text)
        mols=RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(text,is_dlg=False),only_cluster_leads=False)
        if len(mols)!=1 or mols[0] is None or mols[0].GetNumConformers()!=len(energies):
            raise ValueError('Ambiguous saved pose/energy correspondence')
        if len(energies)!=1 or not np.isfinite(energies[0,0]):
            raise ValueError('This recovery supports exactly one finite saved reference pose')
        pose=Chem.RemoveHs(mols[0])
        if Chem.MolToSmiles(pose)!=Chem.MolToSmiles(Chem.MolFromSmiles(reference['smiles'])):
            raise ValueError('Saved reference chemistry differs from declared reference')
        if not np.isfinite(pose.GetConformer().GetPositions()).all():
            raise ValueError('Non-finite saved coordinates')
        energy=float(energies[0,0])
        pose_path=output.parent/(output.stem+'_recovered_reference.sdf')
        if recoveries:
            pose_path=output.parent/(output.stem+f'_recovered_reference_{len(recoveries)}.sdf')
        pose.SetDoubleProp('docking_score_kcal_mol',energy)
        with Chem.SDWriter(str(pose_path)) as writer:
            writer.write(pose)
        pb=check_docked_pose(pose,physical['receptor_preparation']['effective_pdb'])
        original_error=reference.pop('error')
        reference.update(status='docked',score_kcal_mol=energy,pose_sdf=str(pose_path.resolve()),
                         pose_sha256=digest(pose_path),returned_poses=1,selected_pose_index=0,
                         posebusters=pb,docking_seed=execution['seed'])
        execution.update(status='completed',gpu_verified=True,gpu_evidence='kernel_trace',kernel_trace=traces,
                         output_sha256=digest(pdbqt),original_parser_error=execution.pop('error'))
        physical['reference_posebusters_status']=pb['status']
        for candidate in physical['candidates']:
            score=candidate.get('score_kcal_mol')
            descriptors=candidate.get('descriptors',{})
            candidate['beats_redocked_reference']=score<energy if score is not None else None
            candidate['triple_threshold_redocked_reference']=(bool(score<energy and descriptors['sa']<4 and descriptors['qed']>.5)
                if score is not None and 'sa' in descriptors and 'qed' in descriptors else None)
        physical['summary']=summarize(physical['candidates'],job['n_generated_candidates'])
        complete=all(c.get('status')=='docked' and c.get('posebusters',{}).get('status') in ('passed','failed') for c in physical['candidates'])
        physical['status']='completed' if complete and pb['status'] in ('passed','failed') else 'partial_failure'
        recoveries.append({'pocket_id':job['pocket_id'],'seed':job['seed'],'original_error':original_error,
                           'log_sha256':digest(log),'pdbqt_sha256':digest(pdbqt),
                           'pdbqt_hash_capture':'recovery time; original adapter aborted before hashing the output',
                           'integrity_limit':'Original log hash is contemporaneous and verified. No contemporaneous output PDBQT hash exists, so unchanged output bytes since execution cannot be proved by a historical hash.',
                           'method':'parse complete OpenCL trace after stdout progress prefix; recover original saved pose, no new docking'})
    if not recoveries:
        raise ValueError('No reference with the specifically recoverable trace-framing error')
    payload['metadata']['derived_recovery']={'source_json':str(source.resolve()),'source_sha256':digest(source),
                                             'recovery_script_sha256':digest(__file__),'recoveries':recoveries}
    from molmetal.scripts.r4_c_full_sweep import PocketResult, aggregate
    payload['summary']=aggregate([PocketResult(**r) for r in payload['per_pocket']])
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'derived_json':str(output),'recoveries':len(recoveries)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    recover(args.source,args.output)
