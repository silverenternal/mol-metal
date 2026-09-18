"""Replay common real-data checkpoints to test integration resolution, no retraining."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.scripts.r10_cfg_real_crossdocked import build_context, read_ligand, decode_distance_graph, SizedGenerationConfig, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('molmetal/reports/r10_cfg_real_crossdocked_v2_train32_2000/report.json'))
    p.add_argument('--output',type=Path,default=Path('molmetal/reports/r10_cfg_integrator_v2_diagnostic.json'))
    args=p.parse_args();source=json.loads(args.source.read_text());scale=source['protocol']['position_scale']
    report={'source':str(args.source),'source_sha256':sha(args.source),'scope':'integration resolution diagnostic; no atom deletion/coordinate clamp','cells':[]}
    pair=source['tests'][0];pocket,center,_=build_context(pair['receptor_path'],read_ligand(pair['ligand_path']),scale)
    for ck in source['checkpoints']:
        adapter=LipmanFlowMatchingAdapter(hidden_dim=source['protocol']['hidden_dim'],n_layers=source['protocol']['n_layers'],max_atomic_number=100,metal_prior_weight=0,pocket_embed_scale=.1)
        adapter.setup('cuda:0');state=torch.load(ck['path'],map_location=adapter.device,weights_only=True)
        adapter.velocity_field.load_state_dict(state['velocity_field']);adapter.pocket_encoder.load_state_dict(state['pocket_encoder'])
        adapter._cfg_scale=1.
        for steps in [16,64,256]:
            cell={'seed':ck['seed'],'ode_steps':steps,'cfg':1.,'n_requested':8,'checkpoint_sha256':ck['sha256'],'trace':[]}
            original_forward=adapter.velocity_field.forward
            def observe(*args,**kwargs):
                output=original_forward(*args,**kwargs)
                x=args[0];v=output['vel'];cell['trace'].append({'t':float(torch.as_tensor(args[3]).flatten()[0]),
                    'x_absmax':float(x.abs().max()) if torch.isfinite(x).all() else 'nonfinite',
                    'v_absmax':float(v.abs().max()) if torch.isfinite(v).all() else 'nonfinite'})
                return output
            adapter.velocity_field.forward=observe
            try:
                generated=adapter.generate(pocket,SizedGenerationConfig(n_samples=8,n_steps=steps,seed=ck['seed']))
                cell['decode_status_counts']=dict(Counter(decode_distance_graph(m.atom_types,m.coords.numpy()*scale+center)[1] for m in generated))
                cell['status']='finite';cell['max_abs_coordinate']=max(float(m.coords.abs().max()) for m in generated)
            except Exception as exc:cell.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            finally:adapter.velocity_field.forward=original_forward
            report['cells'].append(cell);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print({k:v for k,v in cell.items() if k!='trace'},flush=True)

if __name__=='__main__':main()
