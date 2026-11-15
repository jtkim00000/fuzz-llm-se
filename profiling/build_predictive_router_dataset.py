#!/usr/bin/env python3
"""Prepare leakage-safe per-worker data and matched-pair diagnostics for routing."""
import argparse, hashlib, json
from pathlib import Path

# Keep this import-free so a data-preparation job does not need Prometheus or
# scikit-learn merely to validate raw JSON artifacts. Keep in sync with
# features.py; the trainer performs a second strict allowlist validation.
FEATURES=['estimated_cached_tokens','input_tokens_estimate','output_limit','estimated_total_tokens','previous_turns',
          'streaming','tool_call_mode','structured_output','thinking_enabled','priority','slo_ttft_ms','slo_e2e_ms',
          'precision_fp8','precision_bf16','client_outstanding_requests','state_age_ms',
          'active_requests','running_requests','queued_requests','running_sequences','running_tokens','kv_cache_usage',
          'kv_cache_free_blocks','prefix_cache_hit_rate','recent_request_rate','recent_token_rate','uptime_seconds',
          'recent_failures','time_since_last_request_seconds','gpu_utilization','memory_bandwidth_utilization',
          'gpu_memory_bytes','system_memory_bytes','cpu_utilization','network_rtt_ms']


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def split_for(group):
    bucket=int(hashlib.sha256(('predictive-router-v1:'+group).encode()).hexdigest()[:8],16)%100
    return 'train' if bucket < 70 else 'validation' if bucket < 85 else 'test'


def build(run_dirs, out):
    if out.exists(): raise ValueError('Output already exists')
    samples=[]; paired={}; sources={}
    for run_dir in run_dirs:
        run_dir=Path(run_dir); requests={r['request_id']:r for r in rows(run_dir/'requests.jsonl')}
        trace={t['request_id']:{**t,'session_group':s.get('group_id',s['session_id'])} for s in rows(run_dir/'trace.jsonl') for t in s['turns']}
        sources[str(run_dir)]={'requests_sha256':hashlib.sha256((run_dir/'requests.jsonl').read_bytes()).hexdigest(),
                               'trace_sha256':hashlib.sha256((run_dir/'trace.jsonl').read_bytes()).hexdigest()}
        for request_id,request in requests.items():
            turn=trace[request_id]; chosen=request.get('selected_worker')
            candidate=next((c for c in request['candidates'] if c['worker_id']==chosen),None)
            if candidate is None: continue
            features={key:candidate['features'].get(key) for key in FEATURES}
            labels=request['labels']
            worker_manifest=request['candidate_worker_manifests'][chosen]
            # One prompt across both precisions AND concurrency blocks stays together.
            # Real multi-turn sessions inherit their enclosing session group.
            group=turn['session_group']
            split=split_for(group)
            sample={'request_id':request_id,'logical_request_id':turn.get('logical_request_id',request_id),
                'session_id':request['session_id'],'group_id':group,'split':split,'worker_id':chosen,
                'configuration_hash':request['configuration_hashes'][chosen],
                'configuration_verified':worker_manifest.get('effective_configuration_verified',False),
                'selected':True,'features':features,
                'labels':{'ttft_ms':labels.get('first_visible_token_ms'),'e2e_ms':labels.get('e2e_ms'),
                    'success':labels.get('success'),'slo_met':labels.get('slo_met'),
                    'output_tokens':labels.get('output_tokens'),'actual_input_tokens':labels.get('actual_input_tokens'),
                    'error_type':labels.get('error_type'),'finish_reason':labels.get('finish_reason')},
                'pre_dispatch_snapshot_ns':candidate['snapshot'].get('sample_completed_ns'),
                'dispatch_ns':request['dispatch_ns'],'completion_ns':request['completed_ns']}
            if not isinstance(sample['pre_dispatch_snapshot_ns'],int) or sample['pre_dispatch_snapshot_ns'] > sample['dispatch_ns']:
                raise ValueError('Future telemetry state detected')
            samples.append(sample)
            paired.setdefault(sample['logical_request_id'],{})[chosen]=sample
    if len(samples) < 2: raise ValueError('No selected observations')
    out.mkdir(parents=True)
    (out/'training.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in samples))
    pairs=[]
    for logical_id,by_worker in sorted(paired.items()):
        pairs.append({'logical_request_id':logical_id,'workers':{w:{'request_id':r['request_id'],'split':r['split'],
            'ttft_ms':r['labels']['ttft_ms'],'e2e_ms':r['labels']['e2e_ms'],'success':r['labels']['success'],
            'output_tokens':r['labels']['output_tokens']} for w,r in by_worker.items()}})
    (out/'matched-pairs.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in pairs))
    counts={split:sum(r['split']==split for r in samples) for split in ('train','validation','test')}
    (out/'manifest.json').write_text(json.dumps({'schema_version':1,'sources':sources,'rows':len(samples),
        'matched_logical_requests':len(pairs),'split_counts':counts,
        'features_online_allowlist':FEATURES,
        'labels_not_online_features':['ttft_ms','e2e_ms','success','slo_met','output_tokens','actual_input_tokens','error_type','finish_reason'],
        'warning':'Controlled seed data from one workload family, not counterfactual queue labels. Pipeline validation only; held-out mixed-gateway traffic and independent quality evidence are required before promotion.'},indent=2)+'\n')
    return counts


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.runs,a.out),indent=2))
