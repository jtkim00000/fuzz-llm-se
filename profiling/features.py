"""Explicit pre-dispatch feature allowlist and portable forest inference."""
import math
try:
    from .telemetry import STATE_FIELDS
except ImportError:
    from telemetry import STATE_FIELDS
REQUEST_FEATURES=['estimated_cached_tokens','input_tokens_estimate','output_limit','estimated_total_tokens','previous_turns',
                 'streaming','tool_call_mode','structured_output','thinking_enabled','priority','slo_ttft_ms','slo_e2e_ms']
FEATURES=REQUEST_FEATURES+['precision_fp8','precision_bf16','client_outstanding_requests','state_age_ms']+STATE_FIELDS

def feature_row(request,state,worker,outstanding,now_ns):
    values={k:request.get(k) for k in REQUEST_FEATURES}
    if values['thinking_enabled'] is None and isinstance(worker['manifest'].get('thinking_mode'),bool):
        values['thinking_enabled']=worker['manifest']['thinking_mode']
    precision=worker['manifest'].get('effective_kv_dtype')
    values.update(precision_fp8=None if precision is None else int('fp8' in precision.lower()),
                  precision_bf16=None if precision is None else int(precision.lower() in ['bfloat16','bf16']),
                  client_outstanding_requests=outstanding,
                  state_age_ms=(now_ns-state['sample_completed_ns'])/1e6)
    values.update(state['state'])
    return {k:values.get(k) for k in FEATURES}

def vector(features,artifact):
    x=[]
    for name in artifact['features']:
        value=features.get(name);missing=value is None
        x.append(float(artifact['medians'][name] if missing else value))
        x.append(float(missing))
    return x

def predict(features,artifact,target):
    model=artifact['models'].get(target)
    if not model:return None
    x=vector(features,artifact)
    if 'constant' in model:return model['constant']
    predictions=[]
    for tree in model['trees']:
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        predictions.append(tree['value'][node])
    return sum(predictions)/len(predictions)

def in_domain(features,artifact,worker):
    if worker.get('configuration_hash') not in artifact.get('configuration_hashes',[]):return False
    for key,bounds in artifact['bounds'].items():
        value=features.get(key)
        if value is None and not artifact.get('missing_allowed',{}).get(key,False):return False
        if value is not None and not bounds[0]<=value<=bounds[1]:return False
    return True
