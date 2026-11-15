"""Pure, fail-closed new-session routing. No HTTP, retries, or cache migration."""
import json
import math
from pathlib import Path

from profiling.features import FEATURES, feature_row, in_domain, predict


def load_model(path):
    artifact = json.loads(Path(path).read_text())
    if artifact.get('schema_version') != 1:
        raise ValueError('Unsupported model schema')
    names = artifact.get('features', [])
    if not names or len(set(names)) != len(names) or set(names)-set(FEATURES):
        raise ValueError('Invalid feature allowlist')
    if not artifact.get('configuration_hashes'):
        raise ValueError('Missing configuration allowlist')
    for name in names:
        bounds = artifact['bounds'][name]
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
            raise ValueError('Invalid feature bounds')
        if not math.isfinite(artifact['medians'][name]):
            raise ValueError('Invalid imputation')
    for target in ('e2e_ms', 'ttft_ms'):
        model = artifact['models'][target]
        if 'constant' in model:
            if not math.isfinite(model['constant']) or model['constant'] < 0:
                raise ValueError('Invalid constant')
            continue
        if not model.get('trees'):
            raise ValueError('Missing trees')
        for tree in model['trees']:
            n = len(tree['left'])
            if not n or any(len(tree[k]) != n for k in ('right', 'feature', 'threshold', 'value')):
                raise ValueError('Malformed tree')
            stack, seen = [0], set()
            while stack:
                i = stack.pop()
                if i in seen or not 0 <= i < n:
                    raise ValueError('Cyclic or invalid tree')
                seen.add(i)
                if not all(math.isfinite(tree[k][i]) for k in ('threshold', 'value')):
                    raise ValueError('Nonfinite tree')
                if tree['left'][i] == -1:
                    if tree['right'][i] != -1 or tree['value'][i] < 0:
                        raise ValueError('Invalid leaf')
                else:
                    if not 0 <= tree['feature'][i] < 2*len(names):
                        raise ValueError('Invalid feature index')
                    stack.extend([tree['left'][i], tree['right'][i]])
    return artifact


def choose_new_session(request_features, candidate_snapshots, model, *, now_ns,
                       max_age_ms=1000, max_outstanding=8, max_kv_usage=0.95,
                       required_metrics=('running_requests', 'queued_requests', 'kv_cache_usage')):
    eligible, exclusions, vectors, predictions = {}, {}, {}, {}
    for worker_id, candidate in sorted(candidate_snapshots.items()):
        snapshot = candidate.get('snapshot', {})
        manifest = candidate.get('manifest', {})
        completed = snapshot.get('sample_completed_ns')
        state = snapshot.get('state', {})
        reason = None
        if snapshot.get('health') is not True:
            reason = 'unhealthy'
        elif not isinstance(completed, int) or completed > now_ns or (now_ns-completed)/1e6 > max_age_ms:
            reason = 'missing_future_or_stale_snapshot'
        elif not manifest.get('effective_configuration_verified'):
            reason = 'unverified_configuration'
        elif model is not None and candidate.get('configuration_hash') not in model['configuration_hashes']:
            reason = 'unknown_configuration'
        elif candidate.get('quality_approved') is not True:
            reason = 'quality_unapproved'
        elif any(not isinstance(state.get(k), (int, float)) or not math.isfinite(state[k]) or state[k] < 0 for k in required_metrics):
            reason = 'missing_invalid_telemetry'
        elif not isinstance(candidate.get('outstanding'), int) or candidate['outstanding'] < 0:
            reason = 'missing_outstanding'
        elif candidate['outstanding'] >= max_outstanding or state['kv_cache_usage'] >= max_kv_usage:
            reason = 'overloaded'
        elif not isinstance(request_features.get('estimated_total_tokens'), (int, float)) or request_features['estimated_total_tokens'] > manifest.get('max_model_len', 0):
            reason = 'context_limit_or_missing_token_count'
        if reason:
            exclusions[worker_id] = reason
            continue
        features = feature_row(request_features, snapshot, {'manifest':manifest}, candidate['outstanding'], now_ns)
        if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)) for v in features.values()):
            exclusions[worker_id] = 'invalid_features'
            continue
        eligible[worker_id] = candidate
        vectors[worker_id] = features
    snapshot_evidence = {w:{'sample_started_ns':c.get('snapshot',{}).get('sample_started_ns'),
        'sample_completed_ns':c.get('snapshot',{}).get('sample_completed_ns'),
        'source_timings':c.get('snapshot',{}).get('source_timings',[]),
        'health':c.get('snapshot',{}).get('health'),
        'configuration_hash':c.get('configuration_hash')} for w,c in candidate_snapshots.items()}
    result = {'worker_id':None, 'policy':'rejected', 'predictions':predictions,
              'snapshot_evidence':snapshot_evidence,'decision_ns':now_ns,
              'fallback_reason':None, 'exclusions':exclusions, 'features':vectors,
              'model_version':None if model is None else model.get('model_version', model.get('training_data_sha256'))}
    if not eligible:
        result['fallback_reason'] = 'no_safe_candidate'
        return result
    fallback = 'model_unavailable' if model is None else None
    for worker_id, candidate in eligible.items():
        if model is None:
            break
        try:
            if not in_domain(vectors[worker_id], model, candidate):
                fallback = 'out_of_domain_or_required_feature_missing'
                break
            values = {t:predict(vectors[worker_id], model, t) for t in ('e2e_ms', 'ttft_ms')}
            if any(v is None or not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError('Invalid prediction')
            predictions[worker_id] = values
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
            fallback = 'invalid_prediction'
            break
    if fallback is None:
        allowed = [w for w in eligible if all(request_features.get(slo) is None or predictions[w][target] <= request_features[slo]
                   for slo, target in [('slo_ttft_ms','ttft_ms'), ('slo_e2e_ms','e2e_ms')])]
        if not allowed:
            result['fallback_reason'] = 'predicted_slo_violation'
            return result
        result.update(worker_id=min(allowed,key=lambda w:(predictions[w]['e2e_ms'],w)), policy='regression')
    else:
        result.update(worker_id=min(eligible,key=lambda w:(eligible[w]['outstanding'],w)),
                      policy='least_outstanding', fallback_reason=fallback)
    return result
