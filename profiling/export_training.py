#!/usr/bin/env python3
"""One row per request/candidate; unselected counterfactual labels stay null."""
import argparse,json
from pathlib import Path
from collect_training import rows,dump,sha

def export(runs,out):
    if out.exists():raise ValueError('Export exists')
    out.mkdir(parents=True);count=0;seen_groups={};sources={}
    with (out/'training.jsonl').open('w') as f:
        for run in runs:
            provenance=json.loads((run/'provenance.json').read_text());sources[str(run)]=sha(run/'requests.jsonl')
            for r in rows(run/'requests.jsonl'):
                group=r['group_id'];split=r['split']
                if group in seen_groups and seen_groups[group]!=split:raise ValueError('Cross-run session split leakage')
                seen_groups[group]=split
                for c in r['candidates']:
                    selected=c['worker_id']==r['selected_worker']
                    row={'run_id':str(run),'request_id':r['request_id'],'session_id':r['session_id'],'group_id':group,'split':split,
                         'worker_id':c['worker_id'],'configuration_hash':r['configuration_hashes'][c['worker_id']],
                         'configuration_verified':r['candidate_worker_manifests'][c['worker_id']].get('effective_configuration_verified',False),
                         'trace_sha256':provenance['trace_sha256'],'selected':selected,'eligible':c['eligible'],
                         'features':c['features'],'labels':r['labels'] if selected else None,
                         'predictions':c.get('predictions',{}),'predicted_e2e_ms':c['prediction'],'arrival_ns':r['arrival_ns'],'dispatch_ns':r['dispatch_ns'],
                         'snapshot_ns':c['snapshot']['sample_completed_ns'],'missing_metrics':c['snapshot']['missing_metrics'],
                         'client_schedule_delay_ms':r['client_schedule_delay_ms']}
                    if row['snapshot_ns']>r['decision_ns']:raise ValueError('Future-state leakage')
                    f.write(json.dumps(row)+'\n');count+=1
    dump(out/'manifest.json',{'rows':count,'sources':sources,'sha256':sha(out/'training.jsonl'),
        'split_groups':{s:sorted(k for k,v in seen_groups.items() if v==s) for s in ['train','validation','test']},
        'labels_policy':'Only selected worker observed. Actual output/cached tokens and timings are labels, never features.'})
    return count

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(export(a.runs,a.out))
