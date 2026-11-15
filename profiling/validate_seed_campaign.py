"""Fail closed on seed-campaign completeness, provenance, and paired traces."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate(root):
    manifests = {p: read(root/p/'traces/manifest.json') for p in ('bf16', 'fp8')}
    assert manifests['bf16'] == manifests['fp8'], 'Trace manifests differ'
    sources = {}
    counts = {}
    for precision, expected in [('bf16', 'bfloat16'), ('fp8', 'fp8')]:
        worker = read(root/precision/'manifests/worker.json')
        assert worker['effective_configuration_verified'] is True
        assert worker['effective_kv_dtype'] == expected
        for concurrency in (1, 4):
            for repetition in (1, 2, 3):
                block = f'c{concurrency}-r{repetition}'
                path = root/precision/block
                canonical = rows(path/'canonical-requests.jsonl')
                raw = rows(path/'requests.jsonl')
                assert len(canonical) == len(raw) == 24, str(path)
                assert len({r['request_id'] for r in raw}) == 24
                assert all(r['status'] == 'succeeded' and r['output_tokens'] in (32, 256) for r in canonical)
                assert all(r['labels']['success'] for r in raw)
                config = read(path/'run-config.json')
                assert config['workers'][0]['manifest'] == worker
                trace = path/'trace.jsonl'
                # Git's Windows checkout may convert tracked text to CRLF.
                assert hashlib.sha256(trace.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == read(path/'provenance.json')['trace_sha256']
                other = root/('fp8' if precision == 'bf16' else 'bf16')/block/'trace.jsonl'
                assert trace.read_bytes() == other.read_bytes(), block
                for r in raw:
                    selected = next(c for c in r['candidates'] if c['worker_id'] == r['selected_worker'])
                    assert selected['snapshot']['sample_completed_ns'] <= r['dispatch_ns']
                counts[f'{precision}/{block}'] = 24
                for source in path.iterdir():
                    if source.is_file():
                        sources[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sum(counts.values()) == 288
    return {'validated': True, 'rows': 288, 'blocks': counts,
            'source_sha256': manifests['bf16']['source_sha256'], 'source_files': sources,
            'notes': ['Different attention backends: BF16 FLASH_ATTN, FP8 FLASHINFER.',
                      'Prefix reuse minimized; cold cache not verified.',
                      'Paired observations are descriptive, not counterfactual labels.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    with args.out.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps({'validated': True, 'rows': result['rows'], 'blocks': len(result['blocks'])}))
