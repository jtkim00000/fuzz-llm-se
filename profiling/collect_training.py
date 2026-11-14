#!/usr/bin/env python3
"""Supplement AIPerf with timestamped candidate states and observed outcomes."""
import argparse,copy,hashlib,json,os,socket,statistics,threading,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from telemetry import Collector,headers
from features import feature_row,predict,in_domain
from grade import grade

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def percentile(values,q):
    values=sorted(v for v in values if v is not None)
    if not values:return None
    point=(len(values)-1)*q;lo=int(point);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(point-lo)

def completion(worker,messages,turn,timeout):
    """Timestamp SSE events. Exact ITL percentiles remain unavailable without token timestamps."""
    start=time.perf_counter();result={'success':False,'http_status':None,'timeout':False,
        'error_type':None,'truncated':False,'ttft_ms':None,'first_visible_token_ms':None,
        'itl_p50_ms':None,'itl_p95_ms':None,'itl_p99_ms':None,'queue_ms':None,
        'server_processing_ms':None,'network_overhead_ms':None,'oom':None,'cache_allocation_failure':None,
        'output_tokens':None,'actual_input_tokens':None,'actual_cached_tokens':None,'accuracy_pass':None}
    body={'model':worker['model'],'messages':messages,'stream':turn.get('streaming',True),
          worker.get('output_limit_field','max_tokens'):turn['output_limit'],'temperature':0}
    # Modes are explicit in the trace and preserved as pre-dispatch features.
    for k in ['response_format','tools','tool_choice','chat_template_kwargs']:
        if k in turn:body[k]=turn[k]
    if body['stream']:body['stream_options']={'include_usage':True}
    chunks=[];content='';reasoning='';usage={};finish=None;events=[]
    try:
        req=urllib.request.Request(worker['url'].rstrip('/')+'/v1/chat/completions',
             data=json.dumps(body).encode(),headers={'Content-Type':'application/json',
             'X-Request-ID':turn['request_id'],'X-Session-ID':turn['session_id'],**headers(worker)})
        with urllib.request.urlopen(req,timeout=timeout) as response:
            result['http_status']=response.status
            def consume(event):
                nonlocal usage,finish,content,reasoning
                events.append(event)
                if event.get('usage'):usage=event['usage']
                for choice in event.get('choices',[]):
                    delta=choice.get('delta',choice.get('message',{}))
                    text=delta.get('content') or '';think=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if text or think or delta.get('tool_calls'):
                        t=(time.perf_counter()-start)*1000
                        if result['ttft_ms'] is None:result['ttft_ms']=t
                        if text and result['first_visible_token_ms'] is None:result['first_visible_token_ms']=t
                        chunks.append({'elapsed_ms':t,'content':text,'reasoning':think})
                        content+=text;reasoning+=think
                    if choice.get('finish_reason') is not None:finish=choice['finish_reason']
            if body['stream']:
                buffer=[]
                for line in response:
                    if time.perf_counter()-start>timeout:raise TimeoutError('overall deadline exceeded')
                    line=line.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'):buffer.append(line[5:].lstrip())
                    elif not line and buffer:
                        text='\n'.join(buffer);buffer=[]
                        if text=='[DONE]':break
                        consume(json.loads(text))
                if buffer and '\n'.join(buffer)!='[DONE]':consume(json.loads('\n'.join(buffer)))
            else:consume(json.load(response))
        result['success']=finish is not None
        if not result['success']:result['error_type']='incomplete_response'
    except urllib.error.HTTPError as e:result.update(http_status=e.code,error_type='HTTPError')
    except Exception as e:
        result['error_type']=type(e).__name__
        result['timeout']=isinstance(e,(TimeoutError,socket.timeout)) or isinstance(getattr(e,'reason',None),(TimeoutError,socket.timeout))
    result['e2e_ms']=(time.perf_counter()-start)*1000
    result.update(output_tokens=usage.get('completion_tokens'),actual_input_tokens=usage.get('prompt_tokens'),
                  actual_cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                  truncated=finish=='length',finish_reason=finish)
    result['label_sources']={}
    for label,spec in worker.get('response_label_paths',{}).items():
        if label not in ['queue_ms','server_processing_ms','server_prefill_ms','server_decode_ms','network_overhead_ms','itl_p50_ms','itl_p95_ms','itl_p99_ms','oom','cache_allocation_failure']:raise ValueError('Unsupported server label mapping')
        for event in reversed(events):
            value=event
            try:
                for key in spec['path'].split('.'):value=value[key]
                if isinstance(value,(int,float)):
                    result[label]=value*spec.get('scale',1);result['label_sources'][label]=spec;break
            except (KeyError,TypeError):pass
    gaps=[b['elapsed_ms']-a['elapsed_ms'] for a,b in zip(chunks,chunks[1:])]
    result.update(stream_chunk_gap_p50_ms=percentile(gaps,.5),stream_chunk_gap_p95_ms=percentile(gaps,.95),
                  stream_chunk_gap_p99_ms=percentile(gaps,.99))
    if body['stream'] and len(chunks)>1 and isinstance(result['output_tokens'],int) and result['output_tokens']>1:
        result['tpot_estimate_ms']=(chunks[-1]['elapsed_ms']-chunks[0]['elapsed_ms'])/(result['output_tokens']-1)
    else:result['tpot_estimate_ms']=None
    result['output_tokens_per_second']=None if result['output_tokens'] is None else result['output_tokens']/(result['e2e_ms']/1000)
    if not body['stream']:
        result['ttft_ms']=None;result['first_visible_token_ms']=None
    if 'expected' in turn:
        result.update(grade(content,turn['expected']))
        result['accuracy_pass']=result['passed'] and result['success'] and not result['truncated']
    checks=[]
    for field,limit in [('ttft_ms',turn.get('slo_ttft_ms')),('e2e_ms',turn.get('slo_e2e_ms'))]:
        if limit is not None:checks.append(None if result[field] is None else result[field]<=limit)
    result['slo_met']=False if not result['success'] and checks else (None if not checks or None in checks else all(checks))
    return result,{'events':events,'chunks':chunks,'content':content,'reasoning':reasoning}

class Router:
    def __init__(self,workers,policy,limits,model=None):
        self.workers=workers;self.policy=policy;self.limits=limits;self.model=model
        self.lock=threading.Lock();self.outstanding={w['id']:0 for w in workers};self.affinity={};self.index=0
    def choose(self,session,request,snapshots):
        now=time.time_ns();candidates=[]
        for w in self.workers:
            snap=snapshots[w['id']];age=(now-snap['sample_completed_ns'])/1e6
            features=feature_row(request,snap,w,self.outstanding[w['id']],now)
            reason=None
            if not w.get('enabled',True):reason='disabled'
            elif snap['health'] is False:reason='unhealthy'
            elif self.limits.get('require_quality_approval') and not w.get('quality_approved',False):reason='quality_not_approved'
            for metric,limit_key in [('kv_cache_usage','max_kv_cache_usage'),('queued_requests','max_queued_requests')]:
                limit=self.limits.get(limit_key);value=features[metric]
                if limit is not None:
                    if value is None or age>self.limits.get('max_state_age_ms',3000):reason='required_state_unavailable'
                    elif value>=limit:reason='overloaded'
            predictions={target:predict(features,self.model,target) for target in ['e2e_ms','ttft_ms','slo_met','success']} if self.model and in_domain(features,self.model,w) else {}
            candidates.append({'predictions':predictions,'worker_id':w['id'],'features':features,'snapshot':snap,'eligible':reason is None,'exclusion':reason,
                               'prediction':predictions.get('e2e_ms')})
        eligible=[c for c in candidates if c['eligible']]
        bound=self.affinity.get(session)
        if bound:
            chosen=next((c for c in eligible if c['worker_id']==bound),None);reason='session_affinity' if chosen else 'affinity_worker_unavailable'
        elif not eligible:chosen=None;reason='all_workers_unavailable_or_overloaded'
        else:
            reason=self.policy
            if self.policy=='single':chosen=eligible[0]
            elif self.policy=='round_robin':chosen=eligible[self.index%len(eligible)];self.index+=1
            elif self.policy=='least_queue' and all(c['features']['queued_requests'] is not None for c in eligible):
                chosen=min(eligible,key=lambda c:(c['features']['queued_requests'],c['features']['client_outstanding_requests']))
            elif self.policy=='regression' and all(c['prediction'] is not None for c in eligible):chosen=min(eligible,key=lambda c:c['prediction'])
            else:
                chosen=min(eligible,key=lambda c:c['features']['client_outstanding_requests']);reason='least_outstanding_fallback' if self.policy!='least_outstanding' else self.policy
            self.affinity[session]=chosen['worker_id']
        if chosen:self.outstanding[chosen['worker_id']]+=1
        return chosen,candidates,reason,now

def run(config,trace,out,tokenizer=None):
    if out.exists():raise ValueError('Output exists')
    sessions=rows(trace);workers=copy.deepcopy(config['workers'])
    if config.get('concurrency',1)<1 or config.get('sample_interval_seconds',1)<=0:raise ValueError('Concurrency and sampling interval must be positive')
    if config.get('policy','single') not in ['single','round_robin','least_outstanding','least_queue','regression']:raise ValueError('Unknown routing policy')
    if config.get('policy','single')=='single' and len(workers)!=1:raise ValueError('single policy requires one worker')
    if len({w['id'] for w in workers})!=len(workers):raise ValueError('Duplicate worker ID')
    groups={}
    if not sessions:raise ValueError('Empty trace')
    if any(t.get('output_limit',0)<=0 for s in sessions for t in s['turns']):raise ValueError('Output limits must be positive')
    for session in sessions:
        group=session['group_id'];split=session['split']
        if group in groups and groups[group]!=split:raise ValueError('Session-group split leakage')
        groups[group]=split
    request_ids=[t['request_id'] for s in sessions for t in s['turns']]
    if len(request_ids)!=len(set(request_ids)):raise ValueError('Duplicate request IDs')
    for w in workers:
        w['manifest']=json.loads(Path(w['manifest_path']).read_text())
        w['configuration_hash']=hashlib.sha256(json.dumps(w['manifest'],sort_keys=True).encode()).hexdigest()
        from urllib.parse import urlsplit
        for url in [w['url'],w.get('metrics_url'),w.get('health_url')]+[s['url'] for s in w.get('metric_sources',[])]:
            if url and urlsplit(url).scheme not in ['http','https']:raise ValueError('HTTP(S) endpoints required')
            if url and (urlsplit(url).username or urlsplit(url).query or urlsplit(url).fragment):raise ValueError('Credential-bearing/query URLs are not permitted')
    tok=None
    if tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True,trust_remote_code=False)
    if not tok and not config.get('allow_preview',False):raise ValueError('Matching tokenizer required, or explicitly allow preview')
    out.mkdir(parents=True)
    safeconfig=copy.deepcopy(config);safeconfig['workers']=workers
    dump(out/'run-config.json',safeconfig)
    dump(out/'provenance.json',{'trace_sha256':sha(trace),'collector_sha256':sha(__file__),
         'features_sha256':sha(Path(__file__).with_name('features.py')),'code_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'tokenizer':tokenizer,
         'cache_condition':config.get('cache_condition','unverified'),'client_hostname':socket.gethostname(),
         'started_ns':time.time_ns(),'timing_notes':'SSE chunks are not guaranteed single tokens; ITL percentiles remain null.'})
    (out/'trace.jsonl').write_bytes(Path(trace).read_bytes())
    collector=Collector(workers,out/'telemetry.jsonl',config.get('sample_interval_seconds',1),config.get('metrics_timeout_seconds',2))
    model=json.loads(Path(config['model_path']).read_text()) if config.get('model_path') else None
    router=Router(workers,config.get('policy','single'),config.get('limits',{}),model)
    lock=threading.Lock();results=[];started=time.perf_counter();epoch=time.time_ns()
    def execute(session):
        history=[]
        for index,raw in enumerate(session['turns']):
            if index:time.sleep(raw.get('delay_after_previous_ms',0)/1000)
            arrival_ns=epoch+int(session.get('arrival_offset_seconds',0)*1e9) if not index else time.time_ns()
            if session.get('arrival_pattern')=='concurrency':arrival_ns=time.time_ns()
            turn={**raw,'session_id':session['session_id'],'previous_turns':index,'streaming':raw.get('streaming',True)}
            turn['tool_call_mode']=bool(raw.get('tools'))
            turn['structured_output']=bool(raw.get('response_format'))
            if 'enable_thinking' in raw.get('chat_template_kwargs',{}):turn['thinking_enabled']=raw['chat_template_kwargs']['enable_thinking']
            messages=raw.get('messages') or history+[{'role':'user','content':raw['text']}]
            if tok:
                try:
                    rendered=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,**raw.get('chat_template_kwargs',{}))
                    # Newer Transformers returns BatchEncoding here; len() would
                    # count mapping keys rather than prompt tokens.
                    token_ids=rendered['input_ids'] if hasattr(rendered,'keys') else rendered
                    turn['input_tokens_estimate']=len(token_ids)
                except Exception as e:raise ValueError('Tokenizer/chat template failed; fix before collection') from e
            else:turn['input_tokens_estimate']=None
            turn['estimated_total_tokens']=None if turn['input_tokens_estimate'] is None else turn['input_tokens_estimate']+turn.get('expected_output_tokens',turn['output_limit'])
            with router.lock:
                chosen,candidates,reason,decision_ns=router.choose(session['session_id'],turn,collector.snapshot())
            if chosen:
                worker=next(w for w in workers if w['id']==chosen['worker_id'])
                dispatch_ns=time.time_ns()
                selected_snapshot=chosen['snapshot']
                state_sample_ns=selected_snapshot.get('sample_completed_ns')
                dispatch_state={
                    'running_requests_at_dispatch': selected_snapshot.get('state',{}).get('running_requests'),
                    'waiting_requests_at_dispatch': selected_snapshot.get('state',{}).get('queued_requests'),
                    'state_sample_completed_ns': state_sample_ns,
                    'state_sample_age_ms': None if state_sample_ns is None else (dispatch_ns-state_sample_ns)/1e6,
                }
                # Persist this before opening the request: it is the durable record of
                # an attempted dispatch even if the client or server fails mid-stream.
                with lock:
                    dispatches.write(json.dumps({'request_id':turn['request_id'],'session_id':session['session_id'],
                        'selected_worker':worker['id'],'dispatch_ns':dispatch_ns,'input_tokens_estimate':turn['input_tokens_estimate'],
                        'requested_output_token_limit':turn['output_limit'],**dispatch_state})+'\n');dispatches.flush()
                try:labels,response=completion(worker,messages,turn,config.get('request_timeout_seconds',300))
                finally:
                    with router.lock:router.outstanding[worker['id']]-=1
                history=messages+[{'role':'assistant','content':response['content']}]
            else:
                dispatch_ns=time.time_ns();labels={'success':False,'http_status':None,'error_type':reason,'slo_met':False if turn.get('slo_ttft_ms') or turn.get('slo_e2e_ms') else None,'accuracy_pass':None}
                response=None
                dispatch_state={'running_requests_at_dispatch':None,'waiting_requests_at_dispatch':None,
                                'state_sample_completed_ns':None,'state_sample_age_ms':None}
                with lock:
                    dispatches.write(json.dumps({'request_id':turn['request_id'],'session_id':session['session_id'],
                        'selected_worker':None,'dispatch_ns':dispatch_ns,'input_tokens_estimate':turn['input_tokens_estimate'],
                        'requested_output_token_limit':turn['output_limit'],**dispatch_state})+'\n');dispatches.flush()
            record={'request_id':turn['request_id'],'session_id':session['session_id'],'group_id':session['group_id'],'split':session['split'],
                    'turn_index':index,'task_category':raw.get('category'),'target_position':raw.get('target_position'),'arrival_ns':arrival_ns,'decision_ns':decision_ns,'dispatch_ns':dispatch_ns,'completed_ns':time.time_ns(),
                    'client_schedule_delay_ms':max(0,(dispatch_ns-arrival_ns)/1e6),'configured_thinking_mode':raw.get('chat_template_kwargs'), 'tool_call_mode':turn['tool_call_mode'],'routing_reason':reason,'candidates':candidates,
                    'selected_worker':chosen['worker_id'] if chosen else None,'session_migrated':False,'labels':labels,
                    'candidate_worker_manifests':{w['id']:w['manifest'] for w in workers},
                    'configuration_hashes':{w['id']:w['configuration_hash'] for w in workers},**dispatch_state}
            if labels.get('e2e_ms') is not None:record['arrival_to_completion_ms']=(record['completed_ns']-arrival_ns)/1e6
            with lock:
                responses.write(json.dumps({'request_id':turn['request_id'],'response':response})+'\n');responses.flush()
                log.write(json.dumps(record)+'\n');log.flush();results.append(record)
            if not labels['success']:break
    try:
        collector.start()
        started=time.perf_counter();epoch=time.time_ns()
        with (out/'requests.jsonl').open('w') as log,(out/'responses.jsonl').open('w') as responses,(out/'dispatch-ledger.jsonl').open('w') as dispatches:
            with ThreadPoolExecutor(max_workers=config.get('concurrency',1)) as pool:
                futures=[]
                for session in sorted(sessions,key=lambda s:s.get('arrival_offset_seconds',0)):
                    delay=session.get('arrival_offset_seconds',0)-(time.perf_counter()-started)
                    if delay>0:time.sleep(delay)
                    futures.append(pool.submit(execute,session))
                for f in futures:f.result()
    finally:collector.close()
    duration=time.perf_counter()-started
    observed=[r['labels'] for r in results];success=sum(x['success'] for x in observed)
    summary={'attempted':len(results),'planned_requests':len(request_ids),'not_attempted_after_session_failure':len(request_ids)-len(results),
             'successes':success,'failures':len(results)-success,'duration_seconds':duration,'completed_requests_per_second':success/duration,
             'output_tokens_per_second':sum(x.get('output_tokens') or 0 for x in observed if x['success'])/duration,
             'client_delay_p95_ms':percentile([r['client_schedule_delay_ms'] for r in results],.95),
             'session_migrations':0,'unavailable_or_overload_selections':sum(r['selected_worker'] is None for r in results)}
    for field in ['ttft_ms','e2e_ms']:
        for q in [.5,.95,.99]:summary[f'{field}_p{int(q*100)}']=percentile([x.get(field) for x in observed if x['success']],q)
    slo=[x['slo_met'] for x in observed if x.get('slo_met') is not None]
    quality=[x['accuracy_pass'] for x in observed if x.get('accuracy_pass') is not None]
    summary.update(slo_attainment=None if not slo else sum(slo)/len(slo),accuracy_pass_rate=None if not quality else sum(quality)/len(quality),
                   goodput=None if not slo else sum(slo)/duration)
    summary['accuracy_by_category']={}
    for category in sorted({r['task_category'] for r in results if r['task_category']}):
        rs=[r['labels'] for r in results if r['task_category']==category and r['labels'].get('accuracy_pass') is not None]
        summary['accuracy_by_category'][category]={'n':len(rs),'passed':sum(r['accuracy_pass'] for r in rs)}
    dump(out/'summary.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tokenizer');a=p.parse_args()
    print(json.dumps(run(json.loads(a.config.read_text()),a.trace,a.out,a.tokenizer),indent=2))
