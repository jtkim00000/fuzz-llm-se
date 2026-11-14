#!/usr/bin/env python3
import json, sys
from pathlib import Path
from grade import grade

suite=Path(sys.argv[1])
answers={r['task_id']:r['expected'] for r in map(json.loads,(suite/'accuracy-answers.jsonl').read_text().splitlines())}
for line in (suite/'accuracy-prompts.jsonl').read_text().splitlines():
    task=json.loads(line)
    body=task['text'].split('BEGIN RECORDS\n',1)[1].split('\nEND RECORDS',1)[0]
    lookup={r['id']:r for r in map(json.loads,body.splitlines())}
    answer=answers[task['task_id']]
    assert all(lookup[r['id']]==r for r in answer['records'])
    assert grade(json.dumps(answer),answer)['passed']
    assert not grade('{"records":[]}',answer)['passed']
assert len(answers)==20
print('20 source-derived answer keys verified against prompts; correct/incorrect controls passed.')
