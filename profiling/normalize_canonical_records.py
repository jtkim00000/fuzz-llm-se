#!/usr/bin/env python3
"""Make one analysis-safe canonical row per attempted profiling request."""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path


def iso(ns):
    return None if ns is None else datetime.fromtimestamp(ns/1e9,tz=timezone.utc).isoformat().replace('+00:00','Z')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    requests=[json.loads(x) for x in (a.run_dir/'requests.jsonl').read_text().splitlines() if x]
    trace={t['request_id']:(s,t) for s in [json.loads(x) for x in (a.run_dir/'trace.jsonl').read_text().splitlines() if x] for t in s['turns']}
    out=[]
    for row in requests:
        session,turn=trace[row['request_id']]; labels=row['labels']; missing=[]
        values={
            'input_tokens': next((c['features'].get('input_tokens_estimate') for c in row['candidates'] if c['worker_id']==row['selected_worker']),None),
            'requested_output_token_limit':turn['output_limit'],
            'running_requests_at_dispatch':row.get('running_requests_at_dispatch'),
            'waiting_requests_at_dispatch':row.get('waiting_requests_at_dispatch'),
            'ttft_ms':labels.get('first_visible_token_ms'), 'output_tokens':labels.get('output_tokens'),
            'e2e_latency_ms':labels.get('e2e_ms'), 'dispatch_timestamp_utc':iso(row.get('dispatch_ns')),
            'completion_timestamp_utc':iso(row.get('completed_ns')), 'state_sample_completed_timestamp_utc':iso(row.get('state_sample_completed_ns')),
            'state_sample_age_ms':row.get('state_sample_age_ms')}
        for key in ['input_tokens','running_requests_at_dispatch','waiting_requests_at_dispatch','ttft_ms','output_tokens','e2e_latency_ms','state_sample_completed_timestamp_utc','state_sample_age_ms']:
            if values[key] is None: missing.append(key)
        out.append({'run_id':a.run_id,'request_id':row['request_id'],'logical_request_id':turn.get('logical_request_id',row['request_id']),
            'session_id':row['session_id'],'group_id':row['group_id'],'turn_index':row['turn_index'],'worker_id':row.get('selected_worker'),
            'gateway_routed_worker':labels.get('gateway_routed_worker'),
            **values,'input_token_count_source':'local_pinned_tokenizer_chat_template','server_input_tokens':labels.get('actual_input_tokens'),
            'output_token_count_source':'server_stream_usage' if labels.get('output_tokens') is not None else 'missing',
            'status':'succeeded' if labels.get('success') else 'failed','finish_reason':labels.get('finish_reason'),
            'error_type':labels.get('error_type'),'http_status':labels.get('http_status'),'missing_fields':missing,
            'cache_condition':'unverified','trace_category':turn.get('category')})
    (a.run_dir/'canonical-requests.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in out))
    summary={'run_id':a.run_id,'attempts':len(out),'succeeded':sum(x['status']=='succeeded' for x in out),
        'failed':sum(x['status']=='failed' for x in out),'field_missing_counts':{}}
    for row in out:
        for field in row['missing_fields']: summary['field_missing_counts'][field]=summary['field_missing_counts'].get(field,0)+1
    (a.run_dir/'canonical-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
