"""Offline and localhost mock tests; no real inference or accuracy claims."""
import copy,json,sys,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grade import grade
from telemetry import parse_metrics
from features import FEATURES,predict,in_domain
from collect_training import run,Router
from export_training import export
from build_training_trace import split_for,build as build_trace
from build_suite import build as build_suite

class UnitTests(unittest.TestCase):
    def test_detailed_quality_diagnostics(self):
        expected={'records':[{'id':'a'},{'id':'b'}]}
        score=grade('prefix '+json.dumps(expected)+' suffix',expected)
        self.assertFalse(score['passed']);self.assertTrue(score['extra_prose_present']);self.assertTrue(score['field_correctness'])
        score=grade('{"records":[{"id":"b"},{"id":"a"}]}',expected)
        self.assertTrue(score['field_correctness']);self.assertFalse(score['array_order_correctness'])
    def test_mapping_missing_and_replica_filter(self):
        text='vllm:num_requests_waiting{replica="a"} 3\nvllm:num_requests_waiting{replica="b"} 8\n'
        values,obs=parse_metrics(text,{'queued_requests':{'metric':'vllm:num_requests_waiting','labels':{'replica':'a'}}})
        self.assertEqual(values['queued_requests'],3);self.assertIsNone(values['gpu_utilization'])
        self.assertEqual(obs['queued_requests'][0]['labels'],{'replica':'a'})
    def test_affinity_health_and_overload(self):
        workers=[{'id':x,'manifest':{'effective_kv_dtype':'bfloat16'}} for x in ['a','b']]
        snapshots={x:{'sample_completed_ns':time.time_ns(),'state':{},'health':True} for x in ['a','b']}
        router=Router(workers,'round_robin',{})
        first=router.choose('s',{},snapshots)[0]['worker_id']
        self.assertEqual(router.choose('s',{},snapshots)[0]['worker_id'],first)
        snapshots[first]['health']=False
        chosen,_,reason,_=router.choose('s',{},snapshots)
        self.assertIsNone(chosen);self.assertEqual(reason,'affinity_worker_unavailable')
        snapshots[first]['health']=True
        router=Router(workers,'least_outstanding',{'max_kv_cache_usage':.9})
        self.assertIsNone(router.choose('x',{},snapshots)[0])
    def test_group_splits_stable(self):
        self.assertEqual(split_for('session-a'),split_for('session-a'))
        self.assertEqual({split_for(str(i)) for i in range(100)},{'train','validation','test'})

    def test_bursts_are_partitioned_as_whole_groups(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'source.tsv'
            source.write_text(''.join(f'R-{i}\tLong pathway name with validation record {i}\tSpecies example\n' for i in range(800)))
            suite=root/'suite';build_suite(source,suite,count=20)
            sessions=build_trace([suite],root/'all.jsonl',arrival='bursty',allow_preview=True)
            for offset in [0,10]:
                self.assertEqual(len({s['group_id'] for s in sessions[offset:offset+10]}),1)
                self.assertEqual(len({s['split'] for s in sessions[offset:offset+10]}),1)
            partition=sessions[0]['split']
            selected=build_trace([suite],root/'partition.jsonl',arrival='bursty',allow_preview=True,partition=partition)
            self.assertTrue(all(s['split']==partition for s in selected))

class MockPipelineTests(unittest.TestCase):
    def test_stream_collection_export(self):
        class Server(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                self.send_response(200);self.end_headers()
                self.wfile.write(b'vllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.2\n' if self.path=='/metrics' else b'ok')
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for part in ['{"records":','[]}']:
                    event={'choices':[{'delta':{'content':part},'finish_reason':None}]}
                    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush()
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':80,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':0}}})+'\n\ndata: [DONE]\n\n').encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Server)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);manifest=root/'worker.json';manifest.write_text('{"effective_configuration_verified":true,"effective_kv_dtype":"bfloat16"}')
                url=f'http://127.0.0.1:{server.server_port}'
                w={'id':'a','url':url,'model':'mock','manifest_path':str(manifest),'metrics_url':url+'/metrics','health_url':url+'/health',
                   'metric_map':{'queued_requests':{'metric':'vllm:num_requests_waiting'},'kv_cache_usage':{'metric':'vllm:kv_cache_usage_perc'}}}
                cfg={'workers':[w,{**w,'id':'b'}],'policy':'round_robin','allow_preview':True,'sample_interval_seconds':.05,'concurrency':2}
                trace=root/'trace.jsonl'
                sessions=[{'session_id':str(i),'group_id':str(i),'split':['train','validation','test'][i%3],
                           'turns':[{'request_id':str(i),'text':'Extract records','output_limit':32,'expected':{'records':[]},'slo_e2e_ms':5000}]} for i in range(6)]
                trace.write_text(''.join(json.dumps(s)+'\n' for s in sessions))
                summary=run(cfg,trace,root/'run')
                self.assertEqual(summary['successes'],6);self.assertEqual(summary['accuracy_pass_rate'],1)
                self.assertEqual(export([root/'run'],root/'export'),12)
                result=[json.loads(x) for x in (root/'export/training.jsonl').read_text().splitlines()]
                for r in result:
                    self.assertLessEqual(r['snapshot_ns'],r['dispatch_ns'])
                    self.assertNotIn('actual_input_tokens',r['features'])
                    if r['selected']:
                        self.assertTrue(r['labels']['success']);self.assertIsNone(r['labels']['itl_p95_ms'])
                        self.assertIsNone(r['features']['gpu_utilization'])
                        self.assertEqual(r['labels']['actual_input_tokens'],80)
                    else:self.assertIsNone(r['labels'])
        finally:server.shutdown();server.server_close();thread.join()

class TrainingTests(unittest.TestCase):
    def test_grouped_training_portable_inference_and_leak_rejection(self):
        try:import sklearn
        except ImportError:self.skipTest('Install requirements-training.txt for regression tests')
        from train_regression import train
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'training.jsonl';rs=[]
            for i in range(45):
                features=dict.fromkeys(FEATURES)
                features.update(input_tokens_estimate=100+i*10,output_limit=256,precision_bf16=1,precision_fp8=0,queued_requests=i%4)
                rs.append({'group_id':f'g{i}','request_id':str(i),'worker_id':'w','split':['train','validation','test'][i%3],
                  'selected':True,'configuration_verified':True,'configuration_hash':'config1','features':features,
                  'labels':{'e2e_ms':100+i*10,'ttft_ms':30+i,'success':i%5!=0,'slo_met':i%4!=0}})
            data.write_text(''.join(json.dumps(r)+'\n' for r in rs))
            report=train(data,root/'model')
            artifact=json.loads((root/'model/model.json').read_text())
            self.assertGreater(predict(rs[3]['features'],artifact,'e2e_ms'),0)
            self.assertEqual(report['splits']['test']['groups'],15)
            missing=dict(rs[3]['features']);missing['input_tokens_estimate']=None
            self.assertFalse(in_domain(missing,artifact,{'configuration_hash':'config1'}))
            rs[0]['features']['actual_e2e_ms']=10
            data.write_text(''.join(json.dumps(r)+'\n' for r in rs))
            with self.assertRaisesRegex(ValueError,'feature'):train(data,root/'leak')
            del rs[0]['features']['actual_e2e_ms']
            for r in rs:r['run_id']='same-run'
            data.write_text(''.join(json.dumps(r)+'\n' for r in rs))
            with self.assertRaisesRegex(ValueError,'Run-level'):train(data,root/'run-leak')
