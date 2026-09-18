from molmetal.scripts.summarize_physical_sweep import build_report, moments


def test_missing_evaluator_retains_generated_denominator():
    payload={'metadata':{},'per_pocket':[
        {'pocket_id':'one','seed':42,'status':'ok','n_generated_candidates':2,
         'candidates':[{'smiles':'CCO','is_generated':True},{'smiles':'CCN','is_generated':True}],
         'physical':{'status':'error','error':'optional import failed'}},
        {'pocket_id':'two','seed':42,'status':'ok','n_generated_candidates':1,
         'candidates':[{'smiles':'CCO','is_generated':True}],
         'physical':{'status':'completed','summary':{'n_generated':1,'n_selected':1,'n_docked':1,'n_pb_pass':1},
                     'candidates':[{'score_kcal_mol':-5.,'posebusters':{'pb_valid':True}}]}},
    ]}
    report=build_report(payload,device='cpu')
    assert report['aggregate']['n_generated']==3
    assert report['aggregate']['pb_pass_rate_all_generated']==1/3
    assert report['aggregate']['vina']['n']==1
    assert report['jobs'][0]['summary']['pb_pass_rate_all_generated']==0
    assert report['seed_variation']['mean_pocket_vina']['std_sample'] is None


def test_statistics_do_not_invent_values_for_unmeasured_inputs():
    assert moments([None,float('nan')])=={'n':0,'mean':None,'std_sample':None}
    assert moments([-5.])=={'n':1,'mean':-5.,'std_sample':None}


def test_seed_variation_uses_common_pockets_and_discloses_reference_pb():
    rows=[]
    for seed,pocket,score in [(0,'A',-10.),(0,'B',-2.),(1,'A',-10.),(1,'B',None)]:
        rows.append({'pocket_id':pocket,'seed':seed,'status':'ok','n_generated_candidates':1,
                     'candidates':[{'smiles':'CCO','is_generated':True}],
                     'physical':{'status':'partial_failure','summary':{'n_generated':1},
                                 'reference':{'posebusters':{'status':'failed','pb_valid':False}},
                                 'candidates':[{'score_kcal_mol':score}]}})
    report=build_report({'metadata':{},'per_pocket':rows},device='cpu')
    assert report['common_pockets_across_seeds']==['A']
    assert report['seed_variation']['mean_pocket_vina']['std_sample']==0.
    assert report['aggregate']['reference_posebusters_status_counts']=={'failed':4}
    assert report['aggregate']['reference_pb_valid_subset']['n_jobs']==0


def test_synthesis_rejections_remain_in_coverage_and_generated_diversity(tmp_path):
    from molmetal.scripts.evaluate_generated_poses import evaluate_candidates
    receptor, ligand = tmp_path / 'r.pdb', tmp_path / 'l.sdf'
    receptor.write_text('not needed when all products filtered')
    ligand.write_text('not needed when all products filtered')
    physical = evaluate_candidates([], str(receptor), str(ligand), str(tmp_path / 'poses'),
                                   total_generated=2)
    assert physical['status'] == 'no_selected_candidates'
    assert physical['summary']['n_generated'] == 2
    assert physical['summary']['pb_pass_rate_all_generated'] == 0
    all_candidates = [{'smiles': s, 'is_generated': True} for s in ['CCO', 'CCN']]
    # Also repair an older evaluator's smaller retained-only denominator.
    physical['summary']['n_generated'] = 0
    report = build_report({'metadata': {}, 'per_pocket': [{
        'pocket_id': 'heldout', 'seed': 42, 'status': 'no_candidates',
        'n_generated_candidates': 2, 'candidates': [],
        'diagnostics': {'all_candidates': all_candidates}, 'physical': physical,
    }]}, device='cpu')
    assert report['aggregate']['n_generated'] == 2
    assert report['aggregate']['pb_pass_rate_all_generated'] == 0
    assert report['aggregate']['generated_structure_diversity']['n_unique'] == 2
