"""Lightweight shared feature schema; no telemetry dependencies."""
STATE_FIELDS=['active_requests','running_requests','queued_requests','running_sequences','running_tokens',
'kv_cache_usage','kv_cache_free_blocks','prefix_cache_hit_rate','recent_request_rate',
'recent_token_rate','uptime_seconds','recent_failures','time_since_last_request_seconds',
'gpu_utilization','memory_bandwidth_utilization','gpu_memory_bytes','system_memory_bytes',
'cpu_utilization','network_rtt_ms']
