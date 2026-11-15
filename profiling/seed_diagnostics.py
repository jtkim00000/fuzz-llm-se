"""All-point descriptive diagnostics; never creates counterfactual training labels."""
import argparse
import json
from pathlib import Path


def main(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(line) for line in data.read_text().splitlines() if line]
    pairs={}
    for row in rows:
        pairs.setdefault(row['logical_request_id'],{})[row['worker_id']]=row
    assert len(rows)==288 and len(pairs)==144
    out.mkdir(exist_ok=False)
    diagnostics=[]
    for key,by_worker in sorted(pairs.items()):
        assert set(by_worker)=={'bf16','fp8'}
        diagnostics.append({'logical_request_id':key,'workers':by_worker,
            'descriptive_delta_ms':{t:by_worker['fp8']['labels'][t]-by_worker['bf16']['labels'][t]
                                    for t in ('e2e_ms','ttft_ms')},
            'warning':'Observed at different dispatch times and queue states; not a counterfactual label.'})
    (out/'paired-evidence.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in diagnostics))
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    fields=[('input_tokens_estimate','Input tokens'),('output_limit','Requested output cap'),('queued_requests','Waiting requests before dispatch')]
    for axrow,target in zip(axes,['e2e_ms','ttft_ms']):
        for ax,(feature,label) in zip(axrow,fields):
            for worker,color in [('bf16','#2166ac'),('fp8','#d6604d')]:
                subset=[r for r in rows if r['worker_id']==worker]
                ax.scatter([r['features'][feature] for r in subset],[r['labels'][target]/1000 for r in subset],
                           alpha=.5,s=14,label=f'{worker}: n={len(subset)}',color=color)
            ax.set(xlabel=label,ylabel=target.replace('_ms','')+' (s)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Controlled seed observations: all 288 points; no outlier removal')
    fig.savefig(out/'observations.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,target in zip(axes,['e2e_ms','ttft_ms']):
        ax.axhline(0,color='gray',linewidth=1)
        ax.scatter([p['workers']['bf16']['features']['input_tokens_estimate'] for p in diagnostics],
                   [p['descriptive_delta_ms'][target]/1000 for p in diagnostics],s=16,alpha=.6)
        ax.set(xlabel='BF16 input tokens',ylabel='FP8 minus BF16 '+target.replace('_ms','')+' (s)')
        ax.grid(alpha=.2)
    fig.suptitle('144 matched prompts; different queue states/times, not counterfactuals')
    fig.savefig(out/'paired-differences.png',dpi=160);plt.close(fig)
    print('Saved all-point and paired diagnostics:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.data,a.out)
