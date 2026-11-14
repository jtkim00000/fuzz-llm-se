"""Offline checks of provenance, answer keys, and failure accounting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_suite
import run_baseline
from grade import grade

class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / 'source.tsv'
        cls.source.write_text(''.join(
            f'R-TEST-{i}\tPathway {i} with validation and boundary text\tSpecies example\n'
            for i in range(800)))
        cls.suite = cls.root / 'suite'
        build_suite.build(cls.source, cls.suite, count=3)
        cls.worker = cls.root / 'worker.json'
        cls.worker.write_text('{"effective_configuration_verified": false}')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_strict_json_controls(self):
        expected = {'records':[{'id':'R-1','name':'Example','species':'Species'}]}
        self.assertTrue(grade(json.dumps(expected,indent=2),expected)['passed'])
        for text in ['{"records":[]}', '{"records":[],"records":[]}',
                     '```json\n'+json.dumps(expected)+'\n```',
                     json.dumps(expected)+' explanation']:
            self.assertFalse(grade(text,expected)['passed'])

    def test_answers_are_present_in_context(self):
        answers = {r['task_id']:r['expected'] for r in run_baseline.read_rows(self.suite/'accuracy-answers.jsonl')}
        tasks = run_baseline.read_rows(self.suite/'accuracy-prompts.jsonl')
        self.assertEqual(len(tasks),20)
        self.assertEqual(len({r['task_id'] for r in tasks}),20)
        self.assertEqual({r['position'] for r in tasks},{'beginning','middle','end'})
        for task in tasks:
            body = task['text'].split('BEGIN RECORDS\n',1)[1].split('\nEND RECORDS',1)[0]
            lookup = {r['id']:r for r in map(json.loads,body.splitlines())}
            for record in answers[task['task_id']]['records']:
                self.assertEqual(lookup[record['id']],record)

    def test_hashes_and_reproducibility(self):
        other = self.root / 'repeat'
        build_suite.build(self.source,other,count=3)
        manifest = json.loads((self.suite/'dataset-manifest.json').read_text())
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((self.suite/name).read_bytes()).hexdigest(),digest)
            self.assertEqual((self.suite/name).read_bytes(),(other/name).read_bytes())
        with self.assertRaises(ValueError):
            build_suite.build(self.source,self.suite,count=3)

    def test_accuracy_failure_accounting(self):
        answers = run_baseline.read_rows(self.suite/'accuracy-answers.jsonl')
        calls = iter(range(4))
        class Response:
            def __init__(self,value): self.value=value
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self): return json.dumps(self.value).encode()
        def fake_request(req,timeout):
            i=next(calls)
            payload=json.loads(req.data)
            self.assertEqual(set(payload),{'model','messages','max_tokens','temperature','stream'})
            if i==2: raise TimeoutError()
            content=json.dumps(answers[i]['expected']) if i in (0,3) else '{"records":[]}'
            return Response({'choices':[{'message':{'content':content},'finish_reason':'length' if i==3 else 'stop'}]})
        out=self.root/'mock-results'
        argv=['run_baseline.py','accuracy','--suite',str(self.suite),'--out',str(out),
              '--url','http://mock.invalid','--model','test','--configuration','stock',
              '--worker-manifest',str(self.worker),'--allow-preview','--limit','4']
        with patch.object(sys,'argv',argv), patch.dict('os.environ',{'BENCHMARK_API_KEY':''}), \
             patch.object(run_baseline.urllib.request,'urlopen',fake_request), contextlib.redirect_stdout(io.StringIO()):
            run_baseline.main()
        summary=json.loads((out/'accuracy-summary.json').read_text())
        self.assertEqual((summary['attempted'],summary['passed'],summary['errors'],summary['truncated']),(4,1,1,1))
        self.assertEqual(summary['pass_rate'],.25)

if __name__=='__main__': unittest.main()
