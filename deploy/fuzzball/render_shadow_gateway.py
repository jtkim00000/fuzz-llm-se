"""Package an isolated CPU shadow canary. Never replaces existing services."""
import argparse
import hashlib
import json
from pathlib import Path


def render(repo, model_dir):
    sources={}
    for package,names in {'gateway':['__init__.py','app.py','policy.py','shadow.py'],
                          'profiling':['features.py','schema.py','telemetry.py']}.items():
        for name in names:
            sources[f'{package}/{name}']=(repo/package/name).read_text()
    sources['profiling/__init__.py']=''
    candidates={}
    for name,ip,workflow in [('node-a','192.168.99.12','b0245156-566a-4796-b5cc-1207514b4c95'),
                             ('node-b','192.168.99.13','cb9eae51-9223-4efc-a3c3-21cd49128559')]:
        manifest={'effective_configuration_verified':True,'effective_kv_dtype':'float16',
                  'model_revision':'482ca0f3832238542f8f5295dde86b5f22711d80',
                  'max_model_len':8192,'fuzzball_workflow_id':workflow,
                  'evidence':'Workflow specification plus EngineCore startup log checked 2026-11-15'}
        candidates[name]={'url':f'http://{ip}:18027','manifest':manifest,
                         'configuration_hash':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                         'quality_approved':False,'metric_map':{
                             k:{'metric':v} for k,v in [('running_requests','vllm:num_requests_running'),
                             ('queued_requests','vllm:num_requests_waiting'),('kv_cache_usage','vllm:kv_cache_usage_perc')]}}
    config={'mode':'shadow','workers':candidates}
    script='''#!/bin/sh
set -eu
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/tmp/router-canary')
for name,content in json.loads(Path('/opt/source.json').read_text()).items():
    target=p/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(content)
PY
export PYTHONPATH=/tmp/router-canary
exec python3 - <<'PY'
import logging,uvicorn
from gateway.shadow import create_shadow_app
logging.basicConfig(level=logging.INFO,format='%(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
app=create_shadow_app('/opt/config.json','/opt/model.json')
uvicorn.run(app,host='0.0.0.0',port=19051,workers=1)
PY
'''
    smoke='''#!/bin/sh
set -eu
python3 - <<'PY'
import urllib.request,json,time
base='http://gateway:19051'
def get(path):
    return json.load(urllib.request.urlopen(base+path,timeout=10))
model=get('/v1/models')['data'][0]['id']
turns=[]
for session,stream in [('seed-shadow-a',False),('seed-shadow-b',False),('seed-shadow-a',True),('seed-shadow-b',True)]:
    body={'model':model,'messages':[{'role':'user','content':'Reply with OK.'}],
          'max_tokens':8,'stream':stream,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json','X-Session-ID':session})
    with urllib.request.urlopen(req,timeout=180) as response:
        content=response.read()
        if stream: assert b'data: [DONE]' in content
        else: assert json.loads(content)['choices']
        turns.append({'session':session,'worker':response.headers['X-Routed-Node'],
                      'policy':response.headers['X-Policy'],'stream':stream,'status':response.status})
assert turns[0]['worker']==turns[2]['worker']
assert turns[1]['worker']==turns[3]['worker']
assert turns[0]['worker']!=turns[1]['worker']
report={'passed':True,'mode':'shadow','model':model,'turns':turns,
        'note':'Both live workers are FP16-KV; this is functional shadow-gate validation, not a precision benefit test.'}
open('/report/shadow-smoke.json','w').write(json.dumps(report,indent=2))
print(json.dumps(report))
PY
'''
    image={'uri':'docker://vllm/vllm-openai:v0.28.0'}
    return {'version':'v4','files':{'source.json':json.dumps(sources),'config.json':json.dumps(config),
            'model.json':(model_dir/'model.json').read_text()},
        'volumes':{'report':{'use':'team2','size':'100MiB','egress':[{'source':{'uri':'file://shadow-smoke.json'},
             'destination':{'uri':'fb://group/predictive-router/seed-2026-11-15-v1/shadow-smoke.json'}}]}},
        'services':{'gateway':{'persist':True,'image':image,'files':{f'/opt/{n}':f'file://{n}' for n in ['source.json','config.json','model.json']},
           'script':script,'resource':{'cpu':{'cores':1},'memory':{'size':'2GiB'}},
           'network':{'ports':[{'name':'api','port':19051,'protocol':'tcp'}],
            'endpoints':[{'name':'api','port-name':'api','protocol':'http','type':'subdomain','scope':'group'}]},
           'readiness-probe':{'http-get':{'path':'/health','port':19051},'initial-delay-seconds':5,'period-seconds':5,'failure-threshold':30}}},
        'jobs':{'smoke':{'image':image,'script':smoke,'depends-on':[{'name':'gateway','status':'RUNNING'}],
            'resource':{'cpu':{'cores':1},'memory':{'size':'1GiB'}},'mounts':{'/report':{'volume':'report'}},
            'policy':{'timeout':{'execute':'15m'}}}}}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    with a.out.open('x') as f:json.dump(render(repo,a.model_dir),f)
    print('Wrote isolated shadow-only canary:',a.out)
