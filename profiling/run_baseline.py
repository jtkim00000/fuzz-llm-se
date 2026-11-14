#!/usr/bin/env python3
"""Run a preflight, performance cell, or strict retrieval accuracy baseline."""
import argparse, datetime, hashlib, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
from grade import grade

def read_rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def dump(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','performance','accuracy'])
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--url',required=True);p.add_argument('--model',required=True)
    p.add_argument('--configuration',choices=['stock','bf16','fp8','gateway'],required=True)
    p.add_argument('--worker-manifest',type=Path,required=True)
    p.add_argument('--allow-preview',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--workload',choices=['short','long'],default='short');p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--tokenizer');p.add_argument('--aiperf',default='aiperf');p.add_argument('--metrics-url')
    p.add_argument('--output-limit-field',choices=['max_tokens','max_completion_tokens'],default='max_tokens')
    a=p.parse_args()
    if a.concurrency<1: p.error('concurrency must be positive')
    if a.limit is not None and a.limit<1: p.error('limit must be positive')
    if a.out.exists(): p.error('Output exists: choose a unique run directory')
    if not a.url.startswith(('http://','https://')): p.error('URL must start with http:// or https://')
    suite=json.loads((a.suite/'dataset-manifest.json').read_text())
    for name,digest in suite['files'].items():
        if sha(a.suite/name)!=digest: p.error('Dataset hash mismatch: '+name)
    if suite['token_count_mode']=='character-preview' and not a.allow_preview:
        p.error('Regenerate with pinned tokenizer, or explicitly use --allow-preview for smoke tests')
    worker=json.loads(a.worker_manifest.read_text())
    if a.mode=='performance' and not a.tokenizer and not a.allow_preview: p.error('Performance requires --tokenizer')
    # Credentials for the simple accuracy client are environment-only, never saved.
    key=os.environ.get('BENCHMARK_API_KEY')
    if a.mode=='performance' and key:
        p.error('Authenticated AIPerf needs the venue-approved credential integration; do not put credentials in saved commands')
    a.out.mkdir(parents=True)
    dump(a.out/'worker-manifest.json',worker)
    dump(a.out/'dataset-manifest.json',suite)
    manifest={'mode':a.mode,'configuration_label':a.configuration,'effective_configuration_verified':worker.get('effective_configuration_verified',False),'url':a.url,'model':a.model,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'suite_manifest_sha256':sha(a.suite/'dataset-manifest.json'),'cache_condition':'unverified','status':'running','client_script_sha256':sha(Path(__file__)),'grader_sha256':sha(Path(__file__).with_name('grade.py')),'requested_output_limit_field':a.output_limit_field,'sampling':{'temperature':0},'client_python':sys.version}
    dump(a.out/'manifest.json',manifest)
    def request(path,payload=None):
        headers={'Content-Type':'application/json'}
        if key: headers['Authorization']='Bearer '+key
        req=urllib.request.Request(a.url.rstrip('/')+path,data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
        with urllib.request.urlopen(req,timeout=300) as response: return json.load(response)
    try:
        if a.mode=='preflight':
            models=request('/v1/models');dump(a.out/'models.json',models)
            if a.model not in [r['id'] for r in models.get('data',[])]: raise ValueError('Served model ID not present in /v1/models')
            sample=request('/v1/chat/completions',{'model':a.model,'messages':[{'role':'user','content':'Reply with the word ready.'}],a.output_limit_field:64,'temperature':0,'stream':False})
            dump(a.out/'sample-response.json',sample)
            if not sample.get('choices'): raise ValueError('Missing completion choices')
            print('Model discovery and non-streaming completion passed. Verify streaming with the performance pilot.')
        elif a.mode=='performance':
            trace=a.suite/f'performance-{a.workload}.jsonl'
            rows=read_rows(trace)
            if a.limit:
                rows=rows[:a.limit]; trace=a.out/'pilot-input.jsonl'
                trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            cmd=[a.aiperf,'profile','--model',a.model,'--url',a.url,'--endpoint-type','chat','--streaming','--input-file',str(trace.resolve()),'--custom-dataset-type','single_turn','--tokenizer',a.tokenizer or 'builtin','--use-server-token-count','--concurrency',str(a.concurrency),'--request-count',str(len(rows)),'--random-seed','42','--artifact-dir',str((a.out/'aiperf').resolve()),'--ui','none']
            if a.output_limit_field=='max_tokens':cmd+=['--use-legacy-max-tokens']
            if a.metrics_url:cmd+=['--server-metrics',a.metrics_url]
            dump(a.out/'command.json',cmd)
            manifest['trace_sha256']=sha(trace)
            with (a.out/'aiperf-console.log').open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'AIPerf failed with exit code {result.returncode}; see aiperf-console.log')
            print('Performance artifacts:',a.out/'aiperf')
        else:
            tasks=read_rows(a.suite/'accuracy-prompts.jsonl')
            expected={r['task_id']:r['expected'] for r in read_rows(a.suite/'accuracy-answers.jsonl')}
            if a.limit:tasks=tasks[:a.limit]
            results=[]
            with (a.out/'responses.jsonl').open('w') as raw:
                for task in tasks:
                    started=time.perf_counter()
                    record={'task_id':task['task_id'],'category':task['category'],'position':task['position']}
                    try:
                        response=request('/v1/chat/completions',{'model':a.model,'messages':[{'role':'user','content':task['text']}],a.output_limit_field:task['output_length'],'temperature':0,'stream':False})
                        choice=response['choices'][0]; text=choice['message'].get('content') or ''
                        record.update(response=response,**grade(text,expected[task['task_id']]),error=False)
                        record['truncated']=choice.get('finish_reason')=='length'
                        if record['truncated']:record['passed']=False
                    except Exception as exc:
                        # Exception class only: avoid recording auth-bearing URLs/errors.
                        record.update(error=True,error_type=type(exc).__name__,passed=False,json_valid=False,truncated=False)
                    record['client_elapsed_seconds']=time.perf_counter()-started
                    raw.write(json.dumps(record)+'\n');raw.flush();results.append(record)
            summary={'attempted':len(results),'passed':sum(r['passed'] for r in results),'errors':sum(r['error'] for r in results),'truncated':sum(r['truncated'] for r in results),'json_valid':sum(r['json_valid'] for r in results)}
            summary['pass_rate']=summary['passed']/summary['attempted']
            for group in ['category','position']:
                summary['by_'+group]={value:{'attempted':sum(r[group]==value for r in results),'passed':sum(r[group]==value and r['passed'] for r in results)} for value in sorted({r[group] for r in results})}
            dump(a.out/'accuracy-summary.json',summary);print(json.dumps(summary,indent=2))
        manifest['status']='completed'
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        manifest['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(a.out/'manifest.json',manifest)

if __name__=='__main__':main()
