"""Real tmQM coordinate-control experiment for the explicit-edge prior primitive.

This is coordinate repair of known molecules, not a generative-model or docking
experiment. Original DFT coordinates evaluate recovery and never enter the
optimization objective. Full structures, including all non-donor atoms, persist.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csd_code(comment):
    return re.search(r"CSD_code\s*=\s*([^ |]+)", comment).group(1)


def parse_bo_neighbors(line):
    """Read one-based atom IDs from the explicit Wiberg neighbor triplets."""
    fields = line.split()
    if len(fields) < 3 or (len(fields) - 3) % 3:
        raise ValueError("Malformed tmQM Wiberg neighbor line")
    return int(fields[0]) - 1, [(int(fields[i+1]) - 1, float(fields[i+2])) for i in range(3, len(fields), 3)]


def load_cases(root: Path, n: int):
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    cases = {}
    # Keep an overcomplete candidate bank; the BO rule may reject weak or
    # additional donors. Final selection stays in original XYZ record order.
    with gzip.open(root / 'tmQM_X1.xyz.gz', 'rt') as stream:
        while len(cases) < n * 4:
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            count = int(line)
            comment = stream.readline().strip()
            rows = [stream.readline().split() for _ in range(count)]
            symbols = [row[0] for row in rows]
            if symbols.count('Pt') != 1 or not re.search(r'MND\s*=\s*4(?:\s|\|)', comment):
                continue
            cases[csd_code(comment)] = {'code': csd_code(comment), 'comment': comment,
                'symbols': symbols, 'atomic_numbers': [table.GetAtomicNumber(s) for s in symbols],
                'coords': [[float(v) for v in row[1:4]] for row in rows], 'metal_index': symbols.index('Pt')}
    with gzip.open(root / 'tmQM_X1.BO.gz', 'rt') as stream:
        current = None
        for line in stream:
            if line.startswith('CSD_code'):
                current = csd_code(line)
            elif current in cases and line.strip():
                atom, neighbors = parse_bo_neighbors(line)
                case = cases[current]
                if atom == case['metal_index']:
                    case['metal_bo_line'] = line.strip()
                    case['donors'] = [index for index, order in neighbors if order >= .3]
                case.setdefault('bo_edges', []).extend([atom, index, order] for index, order in neighbors if atom < index and order >= .3)
    selected = [case for case in cases.values() if len(case.get('donors', [])) == 4][:n]
    if len(selected) != n:
        raise ValueError(f'Only {len(selected)} eligible Pt CN4 records found; requested {n}')
    return selected


def coordinates_metrics(coords, reference, metal, donors, bo_edges):
    local = coords[[metal, *donors]]
    centered = local - local.mean(0)
    singular = torch.linalg.svdvals(centered)
    vectors = coords[donors] - coords[metal]
    ref_vectors = reference[donors] - reference[metal]
    unit = vectors / vectors.norm(dim=-1, keepdim=True)
    ref_unit = ref_vectors / ref_vectors.norm(dim=-1, keepdim=True)
    indices = torch.triu_indices(4, 4, 1, device=coords.device)
    angles = torch.acos((unit @ unit.T).clamp(-1, 1))[indices[0], indices[1]]
    ref_angles = torch.acos((ref_unit @ ref_unit.T).clamp(-1, 1))[indices[0], indices[1]]
    edges = torch.tensor([[a, b] for a, b, _ in bo_edges], device=coords.device)
    distances = (coords[edges[:, 0]] - coords[edges[:, 1]]).norm(dim=-1)
    ref_distances = (reference[edges[:, 0]] - reference[edges[:, 1]]).norm(dim=-1)
    return {'all_atom_coordinate_rmse_A_no_alignment': float((coords-reference).square().mean().sqrt()),
            'donor_vector_rmse_A': float((vectors-ref_vectors).square().mean().sqrt()),
            'coordination_plane_rms_A': float(singular[-1] / len(local)**.5),
            'donor_angle_mae_to_dft_degrees': float((angles-ref_angles).abs().mean()*180/torch.pi),
            'bo_edge_length_mae_to_dft_A': float((distances-ref_distances).abs().mean())}


def write_xyz(path, symbols, coords, comment):
    rows = [str(len(symbols)), comment]
    rows += [f'{symbol} {x:.8f} {y:.8f} {z:.8f}' for symbol, (x,y,z) in zip(symbols, coords.detach().cpu().tolist())]
    path.write_text('\n'.join(rows)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/mnt/storage/data/molmetal/tmQM'))
    p.add_argument('--output-dir', type=Path, default=Path('molmetal/reports/r10_tmqm_geometry_control'))
    p.add_argument('--n-cases', type=int, default=8)
    p.add_argument('--prior-implementation', choices=['primitive','production'], default='primitive')
    args = p.parse_args()
    from molmetal.molmetal_lam.priors.metal_geometry import GeometryPenalty, Geometry, MetalGeometryPrior
    if not torch.cuda.is_available():
        raise RuntimeError('Real ROCm GPU required for this protocol')
    device = torch.device('cuda:0')
    production_prior = MetalGeometryPrior(weight=1.)
    def penalty(coords, edge, mask, atoms):
        if args.prior_implementation == 'production':
            return production_prior.prior_loss(coords, atoms, mask.long()*2,
                geometry=Geometry.SQUARE_PLANAR, edge_index=edge)
        return GeometryPenalty(coords,edge,mask,atoms,geometry=Geometry.SQUARE_PLANAR,metal_atomic_number=78)
    started = time.monotonic()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.data_root, args.n_cases)
    report = {'status': 'running', 'date_utc': datetime.now(timezone.utc).isoformat(),
        'scope': f'real DFT geometry coordinate-repair control; {args.prior_implementation} prior energy; not production sampler or Vina evidence',
        'prior_implementation':args.prior_implementation,
        'sources': {str(args.data_root/name): sha(args.data_root/name) for name in ('tmQM_X1.xyz.gz','tmQM_X1.BO.gz')},
        'runtime_sha256': {str(path): sha(path) for path in (Path(__file__), Path('molmetal/molmetal_lam/priors/metal_geometry.py'))},
        'protocol': {'selection': 'first XYZ records with one Pt, metadata MND4, exactly four explicit Wiberg neighbors >=0.3',
                     'oxidation_state': 'not inferred; CN4 Pt is not independently proven Pt(II)',
                     'seeds': [42,0,1234], 'noise_std_A': .25, 'prior_weights': [0.,1.],
                     'steps': 50, 'lr': .025, 'objective': f'0.5*sum((coords-noisy_start)^2) + weight*{args.prior_implementation}_prior(explicit BO donor edges)',
                     'geometry': 'square_planar, ideal angles 90 and 180 degrees',
                     'dft_reference_used_in_objective': False, 'full_atoms_preserved': True,
                     'vina': None, 'vina_note': 'No receptor-paired Pt parameterization is validated; geometry repair is not affinity.'},
        'environment': {'device': str(device), 'gpu': torch.cuda.get_device_name(device), 'hip': torch.version.hip},
        'cases': cases, 'runs': []}
    for case in cases:
        target = torch.tensor(case['coords'], device=device)
        atoms = torch.tensor(case['atomic_numbers'], device=device)
        edge = torch.tensor([case['donors'], [case['metal_index']]*4], device=device)
        mask = torch.ones(4, dtype=torch.bool, device=device)
        write_xyz(out/f"{case['code']}_reference.xyz",case['symbols'],target,case['comment'])
        for seed in (42,0,1234):
            g = torch.Generator(device=device).manual_seed(seed)
            start = target + .25*torch.randn(target.shape, device=device, generator=g)
            for weight in (0.,1.):
                coords = start.clone().requires_grad_(True)
                optimizer = torch.optim.SGD([coords],lr=.025)
                for _ in range(50):
                    optimizer.zero_grad()
                    angular = penalty(coords,edge,mask,atoms)
                    objective = .5*(coords-start).square().sum() + weight*angular
                    objective.backward(); optimizer.step()
                angular = penalty(coords,edge,mask,atoms)
                path = out/f"{case['code']}_seed{seed}_prior{int(weight)}.xyz"
                write_xyz(path,case['symbols'],coords,f"{case['code']} seed={seed} prior={weight}; geometry repair, all atoms retained")
                report['runs'].append({'code':case['code'],'seed':seed,'prior_weight':weight,
                    'initial_coordinate_sha256':hashlib.sha256(start.detach().cpu().numpy().tobytes()).hexdigest(),
                    'prior_penalty_rad':float(angular.detach()),'xyz':str(path),'xyz_sha256':sha(path),
                    'metrics':coordinates_metrics(coords.detach(),target,case['metal_index'],case['donors'],case['bo_edges'])})
    report['paired_deltas'] = []
    for off,on in zip(report['runs'][::2],report['runs'][1::2]):
        assert off['initial_coordinate_sha256']==on['initial_coordinate_sha256']
        report['paired_deltas'].append({'code':off['code'],'seed':off['seed'],
            **{key:on['metrics'][key]-off['metrics'][key] for key in off['metrics']}})
    report['aggregate']={key:{'n':len(report['paired_deltas']),'delta_mean':statistics.mean(d[key] for d in report['paired_deltas']),
        'delta_std_sample':statistics.stdev(d[key] for d in report['paired_deltas']),
        'n_improved':sum(d[key]<0 for d in report['paired_deltas'])} for key in report['runs'][0]['metrics']}
    report.update(status='completed',elapsed_wall_s=time.monotonic()-started)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['aggregate'],indent=2))


if __name__=='__main__':
    main()
