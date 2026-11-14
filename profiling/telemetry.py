"""Timestamped, explicitly mapped Prometheus snapshots; never fabricate missing data."""
import copy,math,threading,time,urllib.request,json,os
from prometheus_client.parser import text_string_to_metric_families

STATE_FIELDS=['active_requests','running_requests','queued_requests','running_sequences','running_tokens',
'kv_cache_usage','kv_cache_free_blocks','prefix_cache_hit_rate','recent_request_rate',
'recent_token_rate','uptime_seconds','recent_failures','time_since_last_request_seconds',
'gpu_utilization','memory_bandwidth_utilization','gpu_memory_bytes','system_memory_bytes',
'cpu_utilization','network_rtt_ms']

def headers(worker):
    key=os.environ.get(worker.get('api_key_env','BENCHMARK_API_KEY'))
    return {'Authorization':'Bearer '+key} if key else {}

def parse_metrics(text,mapping):
    samples=[s for family in text_string_to_metric_families(text) for s in family.samples]
    values={name:None for name in STATE_FIELDS};observations={}
    for key,spec in mapping.items():
        if spec is None: continue
        matches=[s for s in samples if s.name==spec['metric'] and
                 all(s.labels.get(k)==str(v) for k,v in spec.get('labels',{}).items()) and math.isfinite(s.value)]
        if matches:
            nums=[s.value for s in matches]
            op=spec.get('aggregation','sum')
            value={'sum':sum,'mean':lambda x:sum(x)/len(x),'max':max,'min':min}[op](nums)
            values[key]=value*spec.get('scale',1)
            observations[key]=[{'metric':s.name,'labels':s.labels,'value':s.value} for s in matches]
    return values,observations

class Collector:
    def __init__(self,workers,out,interval=1.0,timeout=2.0):
        self.workers=workers;self.interval=interval;self.timeout=timeout
        self.lock=threading.Lock();self.latest={};self.stop_event=threading.Event()
        self.log=open(out,'w');self.threads=[];self.previous={}
    def sample(self,w):
        snap={'worker_id':w['id'],'sample_started_ns':time.time_ns(),'state':dict.fromkeys(STATE_FIELDS),
              'observations':{},'health':None,'scrape_error':None}
        sources=([{'url':w['metrics_url'],'metric_map':w.get('metric_map',{})}] if w.get('metrics_url') else [])+w.get('metric_sources',[])
        snap['source_errors']=[];snap['source_timings']=[]
        for source in sources:
            try:
                t=time.perf_counter()
                req=urllib.request.Request(source['url'],headers=headers(w))
                with urllib.request.urlopen(req,timeout=self.timeout) as r:text=r.read().decode()
                snap['source_timings'].append({'url':source['url'],'http_elapsed_ms':(time.perf_counter()-t)*1000})
                values,observations=parse_metrics(text,source.get('metric_map',{}))
                snap['state'].update({k:v for k,v in values.items() if v is not None})
                snap['observations'].update(observations)
            except Exception as e:snap['source_errors'].append({'url':source['url'],'error_type':type(e).__name__})
        snap['scrape_error']=None if sources and not snap['source_errors'] else ('not_configured' if not sources else 'partial_or_failed_scrape')
        if w.get('health_url'):
            try:
                with urllib.request.urlopen(urllib.request.Request(w['health_url'],headers=headers(w)),timeout=self.timeout) as r:
                    snap['health']=r.status==200
            except Exception:snap['health']=False
        for field,spec in {k:v for source in sources for k,v in source.get('metric_map',{}).items()}.items():
            if spec and spec.get('kind')=='age_seconds' and snap['state'].get(field) is not None:
                snap['state'][field]=max(0,time.time()-snap['state'][field])
            if spec and spec.get('kind')=='counter_rate':
                value=snap['state'].get(field);now=time.time_ns();old=self.previous.get((w['id'],field))
                self.previous[(w['id'],field)]=(now,value)
                snap['state'][field]=None if value is None or old is None or old[1] is None or value<old[1] else (value-old[1])/((now-old[0])/1e9)
        snap['sample_completed_ns']=time.time_ns()
        snap['missing_metrics']=[k for k,v in snap['state'].items() if v is None]
        with self.lock:
            self.latest[w['id']]=snap;self.log.write(json.dumps(snap)+'\n');self.log.flush()
    def loop(self,w):
        while not self.stop_event.is_set():
            self.sample(w);self.stop_event.wait(self.interval)
    def start(self):
        for w in self.workers:
            self.sample(w)
            t=threading.Thread(target=self.loop,args=(w,),daemon=True);t.start();self.threads.append(t)
    def snapshot(self):
        with self.lock:return copy.deepcopy(self.latest)
    def close(self):
        self.stop_event.set()
        for t in self.threads:t.join()
        self.log.close()
