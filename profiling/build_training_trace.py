#!/usr/bin/env python3
"""Create saved arrival schedules with session-group splits from finalized suites."""
import argparse,hashlib,json,random
from pathlib import Path
from collect_training import sha,rows,dump

def split_for(group):
    value=int(hashlib.sha256(('split-v1:'+group).encode()).hexdigest()[:8],16)%100
    return 'train' if value<70 else 'validation' if value<85 else 'test'

def build(suites,out,kind='short',arrival='concurrency',rate=1.0,seed=42,multi_turn=False,slo_ttft_ms=None,slo_e2e_ms=None,allow_preview=False,partition='all'):
    if out.exists():raise ValueError('Trace output exists')
    if rate<=0:raise ValueError('Rate must be positive')
    rng=random.Random(seed);sessions=[];source_hashes={};offset=0
    for si,suite in enumerate(suites):
        m=json.loads((suite/'dataset-manifest.json').read_text())
        for name,digest in m['files'].items():
            if sha(suite/name)!=digest:raise ValueError('Suite hash mismatch: '+name)
        if m['token_count_mode']=='character-preview' and not allow_preview:raise ValueError('Finalize token-sized inputs first')
        source_hashes[str(suite)]=sha(suite/'dataset-manifest.json')
        source=rows(suite/('accuracy-prompts.jsonl' if kind=='accuracy' else f'performance-{kind}.jsonl'))
        expected={r['task_id']:r['expected'] for r in rows(suite/'accuracy-answers.jsonl')}
        for row in source:
            base=row.get('session_id',row.get('task_id'));uid=f'suite{si}:{base}'
            turn={'request_id':uid+':0','text':row['text'],'output_limit':row['output_length'],'streaming':True,
                  'category':row.get('category'),'target_position':row.get('position'),'structured_output':False,'thinking_enabled':None,'priority':0,'slo_ttft_ms':slo_ttft_ms,'slo_e2e_ms':slo_e2e_ms}
            if kind=='accuracy':turn['expected']=expected[base]
            turns=[turn]
            if multi_turn:
                if kind=='accuracy':raise ValueError('Use separately designed multi-turn correctness traces')
                turns.append({**turn,'request_id':uid+':1','text':'Using the records from our previous turn, describe a robust indexing strategy and its validation checks.','output_limit':1024,'delay_after_previous_ms':100})
            if arrival=='constant' and sessions:offset+=1/rate
            elif arrival=='poisson' and sessions:offset+=rng.expovariate(rate)
            elif arrival=='bursty':offset=int(len(sessions)/10)*10/rate
            sessions.append({'session_id':uid,'group_id':'group:'+base.rsplit('-',1)[0]+':'+str(int(base.rsplit('-',1)[-1])//10),'split':split_for('group:'+base.rsplit('-',1)[0]+':'+str(int(base.rsplit('-',1)[-1])//10)),
                'arrival_offset_seconds':offset,'arrival_pattern':arrival,'history_mode':'live' if multi_turn else 'single','turns':turns})
    if partition!='all':sessions=[s for s in sessions if s['split']==partition]
    if not sessions:raise ValueError('No sessions in requested partition; generate more groups')
    offset=0;rng=random.Random(seed)
    for index,session in enumerate(sessions):
        if arrival=='constant':offset=index/rate
        elif arrival=='poisson' and index:offset+=rng.expovariate(rate)
        elif arrival=='bursty':offset=(index//10)*10/rate
        session['arrival_offset_seconds']=offset
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(r)+'\n' for r in sessions))
    dump(out.with_suffix('.manifest.json'),{'trace_sha256':sha(out),'source_suites':source_hashes,'seed':seed,
        'kind':kind,'arrival_pattern':arrival,'rate':rate,'multi_turn':multi_turn,'slo_ttft_ms':slo_ttft_ms,'slo_e2e_ms':slo_e2e_ms,
        'split_rule':'sha256(split-v1:group_id), 70/15/15; preserve across contexts, repeats and policies',
        'partition':partition,'sessions':len(sessions),'split_counts':{s:sum(r['split']==s for r in sessions) for s in ['train','validation','test']}})
    return sessions

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suites',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=['short','long','accuracy'],default='short');p.add_argument('--arrival',choices=['concurrency','constant','poisson','bursty'],default='concurrency')
    p.add_argument('--split',choices=['all','train','validation','test'],default='all');p.add_argument('--rate',type=float,default=1);p.add_argument('--seed',type=int,default=42);p.add_argument('--multi-turn',action='store_true')
    p.add_argument('--slo-ttft-ms',type=float);p.add_argument('--slo-e2e-ms',type=float);p.add_argument('--allow-preview',action='store_true');a=p.parse_args()
    print(len(build(a.suites,a.out,a.kind,a.arrival,a.rate,a.seed,a.multi_turn,a.slo_ttft_ms,a.slo_e2e_ms,a.allow_preview,a.split)))
