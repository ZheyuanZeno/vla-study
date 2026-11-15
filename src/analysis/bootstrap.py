"""Task-stratified paired bootstrap; one resampled unit contains seven conditions."""
from collections import defaultdict
import numpy as np

CONDITIONS=('Baseline','T1','T2','T3','B1','B2','B3')
VALID={'success','valid_task_failure','task_timeout'}


def blocks_from_rows(rows,phase='formal',expected_tasks=4,allow_synthetic=False):
    selected=[r for r in rows if r.get('study_phase')==phase]
    if not selected: raise ValueError(f'No {phase} data; pilot is never pooled into formal data')
    if any(r.get('data_origin')=='synthetic' for r in selected) and not allow_synthetic:
        raise ValueError('Synthetic data is not allowed in real reports')
    revisions={r['checkpoint_revision'] for r in selected}
    if len(revisions)!=1: raise ValueError('Mixed checkpoint revisions')
    units=defaultdict(dict); infrastructure=[]
    for row in selected:
        task=(row['suite'],row['task_id']); key=(task,row['state_id'])
        if row['condition'] not in CONDITIONS: raise ValueError('Unknown condition')
        # Keep incomplete blocks visible even if the only attempt was infrastructure failure.
        unit=units[key]
        if row['status']=='infrastructure_failure':
            infrastructure.append(row); continue
        if row['status'] not in VALID: raise ValueError('Unknown outcome status')
        if type(row['success']) is not bool: raise ValueError('Valid outcomes require Boolean success')
        if (row['status']=='success') != row['success']: raise ValueError('Inconsistent success/status')
        if row['condition'] in unit: raise ValueError('Duplicate completed condition in paired block')
        unit[row['condition']]=row
    blocks=defaultdict(list)
    for (task,state),unit in sorted(units.items()):
        if set(unit)!=set(CONDITIONS):
            raise ValueError(f'Incomplete block {task}/{state}: missing {set(CONDITIONS)-set(unit)}')
        state_hashes={r['state_sha256'] for r in unit.values()}
        if len(state_hashes)!=1: raise ValueError('Paired conditions used different initial states')
        blocks[task].append([int(unit[c]['success']) for c in CONDITIONS])
    if len(blocks)!=expected_tasks: raise ValueError(f'Expected {expected_tasks} tasks, got {len(blocks)}')
    return {f'{suite}:{task}':np.asarray(values,dtype=np.float64) for (suite,task),values in blocks.items()},infrastructure


def _interval(samples):
    return np.percentile(samples,[2.5,97.5],axis=0).T.tolist()


def summarize(blocks,n_bootstrap=5000,seed=20261115):
    if n_bootstrap<1: raise ValueError('Need positive bootstrap count')
    rng=np.random.default_rng(seed)
    per_task={}; task_boot=[]; task_points=[]
    for task,matrix in sorted(blocks.items()):
        if matrix.ndim!=2 or matrix.shape[1]!=7 or len(matrix)==0 or not np.isin(matrix,[0,1]).all():
            raise ValueError('Expected a nonempty binary matrix of complete seven-condition blocks')
        indices=rng.integers(0,len(matrix),size=(n_bootstrap,len(matrix)))
        boot=matrix[indices].mean(axis=1)
        point=matrix.mean(axis=0)
        gap=point[4:7]-point[1:4]
        boot_gap=boot[:,4:7]-boot[:,1:4]
        per_task[task]={'n_blocks':len(matrix),'success_rates':dict(zip(CONDITIONS,point.tolist())),
            'gap_small_medium_large':gap.tolist(),'gap_ci95_pointwise':_interval(boot_gap),
            'large_gap_minus_small_gap':float(gap[2]-gap[0]),
            'large_minus_small_ci95':_interval((boot_gap[:,2]-boot_gap[:,0])[:,None])[0]}
        task_points.append(point); task_boot.append(boot)
    average=np.mean(task_points,axis=0); boot_average=np.mean(task_boot,axis=0)
    gaps=average[4:7]-average[1:4]; gap_boot=boot_average[:,4:7]-boot_average[:,1:4]
    return {'bootstrap_replicates':n_bootstrap,'bootstrap_seed':seed,'resampling_unit':'seven-condition initial-state block within each task',
        'task_weighting':'equal','interval_type':'pointwise 95% percentile (not simultaneous)',
        'conditions':list(CONDITIONS),'per_task':per_task,
        'equal_task_average':{'success_rates':dict(zip(CONDITIONS,average.tolist())),
            'success_rate_ci95_pointwise':dict(zip(CONDITIONS,_interval(boot_average))),
            'gap_small_medium_large':gaps.tolist(),'gap_ci95_pointwise':_interval(gap_boot),
            'large_gap_minus_small_gap':float(gaps[2]-gaps[0]),
            'large_minus_small_ci95':_interval((gap_boot[:,2]-gap_boot[:,0])[:,None])[0]}}
