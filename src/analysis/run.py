"""Analyze stored real results; no simulator or policy import/download required."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.analysis.bootstrap import blocks_from_rows,summarize


def plots(report,output):
    levels=np.arange(3)
    tasks=report['per_task']
    def panel(ax,values,title):
        sr=values['success_rates']
        ax.plot(levels,[sr[f'T{i}'] for i in (1,2,3)],'o-',label='Target')
        ax.plot(levels,[sr[f'B{i}'] for i in (1,2,3)],'s-',label='Background')
        ax.axhline(sr['Baseline'],color='black',linestyle='--',label='Baseline')
        ax.set(xticks=levels,xticklabels=['Small','Medium','Large'],ylim=(-.03,1.03),ylabel='Success rate',title=title)
        ax.legend(); ax.grid(alpha=.2)
    fig,ax=plt.subplots(figsize=(6,4)); panel(ax,report['equal_task_average'],'Equal task average')
    fig.tight_layout(); fig.savefig(output/'severity_success_rates.png',dpi=220); plt.close(fig)
    fig,axes=plt.subplots(len(tasks),1,figsize=(6,3.4*len(tasks)),squeeze=False)
    for ax,(task,values) in zip(axes[:,0],tasks.items()): panel(ax,values,task)
    fig.tight_layout(); fig.savefig(output/'per_task_success_rates.png',dpi=220); plt.close(fig)
    avg=report['equal_task_average']; point=np.asarray(avg['gap_small_medium_large']); ci=np.asarray(avg['gap_ci95_pointwise'])
    fig,ax=plt.subplots(figsize=(6,4)); ax.errorbar(levels,point,yerr=np.maximum(0,np.vstack((point-ci[:,0],ci[:,1]-point))),fmt='o',capsize=5)
    ax.axhline(0,color='black',linewidth=1); ax.set(xticks=levels,xticklabels=['Small','Medium','Large'],ylabel='SR background - SR target',title='Pointwise 95% paired bootstrap intervals')
    fig.tight_layout(); fig.savefig(output/'paired_gaps.png',dpi=220); plt.close(fig)
    names=list(avg['success_rates']); fig,ax=plt.subplots(figsize=(7,4)); ax.bar(names,[avg['success_rates'][n] for n in names]); ax.set(ylim=(0,1.03),ylabel='Success rate',title='Baseline and occlusion conditions')
    fig.tight_layout(); fig.savefig(output/'baseline_comparison.png',dpi=220); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',nargs='+',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--phase',choices=['formal','pilot'],default='formal')
    parser.add_argument('--expected-tasks',type=int,default=4); parser.add_argument('--seed',type=int,default=20261115)
    args=parser.parse_args()
    if args.phase=='formal' and args.expected_tasks!=4: raise ValueError('Formal analysis requires exactly four tasks')
    rows=[]
    for file in args.input:
        rows.extend(json.loads(line) for line in Path(file).read_text().splitlines() if line.strip())
    blocks,failures=blocks_from_rows(rows,phase=args.phase,expected_tasks=args.expected_tasks)
    report=summarize(blocks,n_bootstrap=5000,seed=args.seed)
    report.update(study_phase=args.phase,data_origin='real_rollout',input_files=args.input,infrastructure_attempts=len(failures))
    if args.phase=='pilot': report['warning']='Pilot only. Not a formal estimate; single-block intervals are degenerate and uninformative.'
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    (output/'statistics.json').write_text(json.dumps(report,indent=2))
    plots(report,output)
    print(f'Saved {args.phase} analysis: {output}')


if __name__=='__main__': main()
