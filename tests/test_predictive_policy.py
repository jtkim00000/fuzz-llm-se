import copy
import json
from pathlib import Path
import tempfile
import unittest

from gateway.policy import choose_new_session, load_model


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000_000
        self.request = {'estimated_total_tokens':100, 'input_tokens_estimate':68, 'output_limit':32}
        self.candidates = {w:{'manifest':{'effective_configuration_verified':True,
            'effective_kv_dtype':w,'max_model_len':8192}, 'configuration_hash':w,
            'quality_approved':True,'outstanding':n,'snapshot':{'health':True,
            'sample_completed_ns':self.now-100_000,'state':{'running_requests':0,
            'queued_requests':0,'kv_cache_usage':0.1}}} for w,n in [('bf16',1),('fp8',0)]}
        self.model={'schema_version':1,'features':['precision_fp8'],'medians':{'precision_fp8':0},
            'bounds':{'precision_fp8':[0,1]},'missing_allowed':{'precision_fp8':False},
            'configuration_hashes':['bf16','fp8'],'models':{
            'e2e_ms':{'trees':[{'left':[1,-1,-1],'right':[2,-1,-1],'feature':[0,-2,-2],
                'threshold':[.5,-2,-2],'value':[0,10,20]}]},'ttft_ms':{'constant':2}}}

    def choose(self, model=True):
        return choose_new_session(self.request,self.candidates,self.model if model else None,now_ns=self.now)

    def test_minimum_e2e_and_safe_fallback(self):
        self.assertEqual(self.choose()['worker_id'],'bf16')
        self.assertEqual(self.choose(False)['worker_id'],'fp8')
        self.assertEqual(self.choose(False)['policy'],'least_outstanding')

    def test_exclusions_never_fallback_to_unsafe_workers(self):
        cases=[('quality_approved',False),('configuration_hash','changed'),('outstanding',8)]
        for key,value in cases:
            with self.subTest(key=key):
                candidates=copy.deepcopy(self.candidates)
                for c in self.candidates.values(): c[key]=value
                self.assertEqual(self.choose()['policy'],'rejected')
                self.candidates=candidates
        for c in self.candidates.values(): c['snapshot']['sample_completed_ns']=self.now+1
        self.assertEqual(self.choose()['policy'],'rejected')

    def test_domain_and_slo(self):
        self.model['bounds']['precision_fp8']=[0,0]
        self.assertEqual(self.choose()['policy'],'least_outstanding')
        self.model['bounds']['precision_fp8']=[0,1]
        self.request['slo_ttft_ms']=1
        self.assertEqual(self.choose()['policy'],'rejected')

    def test_artifact_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.json'
            p.write_text(json.dumps(self.model))
            self.assertEqual(load_model(p)['features'],['precision_fp8'])
            self.model['models']['e2e_ms']['trees'][0]['left'][0]=0
            p.write_text(json.dumps(self.model))
            with self.assertRaises(ValueError): load_model(p)

    def test_missing_metrics_and_nonfinite_feature(self):
        for c in self.candidates.values(): c['snapshot']['state'].pop('queued_requests')
        self.assertEqual(self.choose()['policy'],'rejected')


if __name__=='__main__': unittest.main()
