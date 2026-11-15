#!/usr/bin/env python3
"""Execute explicit cell schedules, stop on measured limits, export and optionally train."""
import argparse,copy,json
from pathlib import Path
from collect_training import run,dump
from export_training import export

def campaign(spec,out):
    if out.exists():raise ValueError('Campaign output exists')
    out.mkdir(parents=True);dump(out/'campaign.json',spec);runs=[];summary=[]
    for ei,experiment in enumerate(spec['experiments']):
        config=json.loads(Path(experiment['config']).read_text())
        stopped=False
        for ci,cell in enumerate(experiment['cells']):
            if stopped:break
            for rep in range(cell.get('repetitions',3)):
                cfg=copy.deepcopy(config);cfg['concurrency']=cell['concurrency']
                directory=out/f'experiment-{ei}-cell-{ci}-rep-{rep}'
                result=run(cfg,Path(cell['trace']),directory,experiment.get('tokenizer'))
                runs.append(directory);summary.append({'experiment':ei,'cell':ci,'rep':rep,**result})
                stop=spec.get('stop_sweep',{})
                error_rate=result['failures']/max(1,result['attempted'])
                stopped=(error_rate>stop.get('max_error_fraction',1) or
                    result['client_delay_p95_ms'] is not None and result['client_delay_p95_ms']>stop.get('max_client_delay_ms',float('inf')) or
                    result['slo_attainment'] is not None and result['slo_attainment']<stop.get('min_slo_attainment',0))
                dump(out/'campaign-summary.json',{'runs':summary,'stopped_current_experiment':stopped})
                if stopped:break
    export(runs,out/'training-export')
    if spec.get('train_after_collection',False):
        from train_regression import train
        train(out/'training-export/training.jsonl',out/'model')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(campaign(json.loads(a.spec.read_text()),a.out),indent=2))
