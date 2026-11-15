"""Explicit shadow-only setup. Never promotes the seed model into serving."""
import asyncio
import json
import time
from pathlib import Path

import httpx

from profiling.features import REQUEST_FEATURES
from profiling.telemetry import parse_metrics
from .app import Worker, create_app
from .policy import load_model


class SnapshotProvider:
    def __init__(self, workers, *, transport=None):
        self.workers = workers
        self.transport = transport

    async def __call__(self, request, body):
        payload = json.loads(body)
        features = dict.fromkeys(REQUEST_FEATURES)
        cap = payload.get('max_completion_tokens',payload.get('max_tokens'))
        if not isinstance(cap,int) or isinstance(cap,bool) or cap <= 0:
            raise ValueError('Explicit output budget required for prediction')
        features.update(output_limit=cap, streaming=bool(payload.get('stream',False)),
                        tool_call_mode=bool(payload.get('tools')),
                        structured_output=bool(payload.get('response_format')),
                        thinking_enabled=payload.get('chat_template_kwargs',{}).get('enable_thinking'))
        async with httpx.AsyncClient(timeout=3,trust_env=False,transport=self.transport) as client:
            async def sample(worker_id, info):
                started=time.time_ns()
                snapshot={'health':False,'state':{},'sample_started_ns':started,
                          'source_timings':[]}
                origin=info['url'].rstrip('/')
                # No caller credential is forwarded. This setup supports private
                # cluster routes; protected backends need explicit secret wiring.
                count=None
                try:
                    health, metrics, tokens = await asyncio.gather(
                        client.get(origin+'/health'),client.get(origin+'/metrics'),
                        client.post(origin+'/tokenize',json={k:v for k,v in {
                            'model':payload.get('model'),'messages':payload.get('messages'),
                            'tools':payload.get('tools'),'chat_template_kwargs':payload.get('chat_template_kwargs'),
                            'add_generation_prompt':True}.items() if v is not None}))
                    snapshot['health']=health.status_code==200
                    metrics.raise_for_status();tokens.raise_for_status()
                    snapshot['state'],_=parse_metrics(metrics.text,info['metric_map'])
                    count=tokens.json()['count']
                    if not isinstance(count,int) or isinstance(count,bool) or count<0:
                        raise ValueError('Invalid token count')
                except (httpx.HTTPError,ValueError,KeyError):
                    snapshot['health']=False
                snapshot['sample_completed_ns']=time.time_ns()
                snapshot['source_timings']=[{'url':origin+'/metrics',
                    'http_elapsed_ms':(snapshot['sample_completed_ns']-started)/1e6}]
                return worker_id, {'snapshot':snapshot,'manifest':info['manifest'],
                    'configuration_hash':info['configuration_hash'],
                    'quality_approved':info.get('quality_approved') is True}, count
            samples=await asyncio.gather(*(sample(w,i) for w,i in self.workers.items()))
        counts={count for _,_,count in samples if count is not None}
        if len(counts)!=1:
            raise ValueError('Missing or inconsistent pinned-tokenizer counts')
        features['input_tokens_estimate']=counts.pop()
        features['estimated_total_tokens']=features['input_tokens_estimate']+cap
        # Cached tokens, previous turns, priority and SLOs remain unknown unless
        # a separately validated pre-dispatch source supplies them.
        return features,{w:c for w,c,_ in samples}


def create_shadow_app(config_path, model_path):
    config=json.loads(Path(config_path).read_text())
    if config.get('mode')!='shadow':
        raise ValueError('Only shadow mode is supported')
    try:
        artifact=load_model(model_path)
    except (OSError,ValueError,KeyError,TypeError,OverflowError):
        artifact=None
    workers=config['workers']
    return create_app([Worker(w,i['url']) for w,i in workers.items()],
        policy_provider=SnapshotProvider(workers),policy_model=artifact)
