#!/usr/bin/env python3
"""Train a small forest with a fixed pre-dispatch allowlist and grouped splits."""
import argparse,json,statistics
from pathlib import Path
from collect_training import rows,dump,sha
from features import FEATURES,predict,vector

def train(data,out,allow_unverified=False,min_groups=3):
    import sklearn
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    if out.exists():raise ValueError('Model directory exists')
    allrows=rows(data);groups={};run_splits={}
    for row in allrows:
        if row['split'] not in ['train','validation','test']:raise ValueError('Invalid split')
        if row.get('run_id'):
            if row['run_id'] in run_splits and run_splits[row['run_id']]!=row['split']:raise ValueError('Run-level split leakage; replay partitioned traces in separate runs')
            run_splits[row['run_id']]=row['split']
        group=row['group_id']
        if group in groups and groups[group]!=row['split']:raise ValueError('Session-group split leakage')
        groups[group]=row['split']
        if set(row['features'])-set(FEATURES):raise ValueError('Unrecognized/possibly leaked feature')
    eligible=[r for r in allrows if r['selected'] and r['labels'] is not None and (r['configuration_verified'] or allow_unverified)]
    parts={s:[r for r in eligible if r['split']==s] for s in ['train','validation','test']}
    for s,rs in parts.items():
        if len({r['group_id'] for r in rs})<min_groups:raise ValueError(f'Insufficient independent {s} groups; collect more traces')
    features=[f for f in FEATURES if any(r['features'].get(f) is not None for r in parts['train'])]
    if not features:raise ValueError('No measured input features')
    artifact={'schema_version':1,'features':features,'medians':{},'bounds':{},'models':{},'missing_allowed':{},
              'configuration_hashes':sorted({r['configuration_hash'] for r in parts['train']}),'training_data_sha256':sha(data),
              'sklearn_version':sklearn.__version__,'allow_unverified':allow_unverified,'seed':42,
              'notes':['Probabilities are uncalibrated forest estimates.','Quality gate is separate from latency prediction.','Report errors and abstain/fallback outside training configuration/ranges.']}
    for key in features:
        values=[float(r['features'][key]) for r in parts['train'] if r['features'].get(key) is not None]
        artifact['missing_allowed'][key]=any(r['features'].get(key) is None for r in parts['train'])
        artifact['medians'][key]=statistics.median(values);artifact['bounds'][key]=[min(values),max(values)]
    report={'split_method':'complete session groups with stable hash assignments','warnings':[],'splits':{s:{'rows':len(rs),'groups':len({r['group_id'] for r in rs})} for s,rs in parts.items()},'targets':{}}
    predictions=[]
    for target,classifier in [('ttft_ms',False),('e2e_ms',False),('slo_met',True),('success',True)]:
        filtered={s:[r for r in rs if r['labels'].get(target) is not None and (classifier or r['labels'].get('success'))] for s,rs in parts.items()}
        tr=filtered['train']
        if len(tr)<5:report['targets'][target]={'status':'unavailable_insufficient_labels'};continue
        x=[vector(r['features'],artifact) for r in tr];y=[float(r['labels'][target]) for r in tr]
        if classifier and len(set(y))==1:
            artifact['models'][target]={'constant':y[0]};status='constant_single_class_no_reliability_validation'
        else:
            cls=RandomForestClassifier if classifier else RandomForestRegressor
            model=cls(n_estimators=64,max_depth=8,min_samples_leaf=3,random_state=42,n_jobs=1).fit(x,y)
            trees=[]
            for estimator in model.estimators_:
                t=estimator.tree_;values=[]
                for value in t.value:
                    array=value[0]
                    values.append(float(array[list(model.classes_).index(1.0)]/sum(array)) if classifier else float(array[0]))
                trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),
                              'threshold':t.threshold.tolist(),'value':values})
            artifact['models'][target]={'trees':trees};status='trained'
        results={'status':status,'feature_importance_note':'Compare to held-out baselines; do not tune on test data.'}
        if classifier and len(set(y))==1:report['warnings'].append(f'{target}: no outcome variation; do not claim a useful reliability classifier')
        baseline=statistics.mean(y)
        for split in ['validation','test']:
            rs=filtered[split]
            errors=[];baseline_errors=[];by_precision={}
            for r in rs:
                pred=predict(r['features'],artifact,target);actual=float(r['labels'][target]);err=(pred-actual)**2 if classifier else abs(pred-actual)
                errors.append(err);baseline_errors.append((baseline-actual)**2 if classifier else abs(baseline-actual))
                precision='fp8' if r['features'].get('precision_fp8') else 'other_or_unknown'
                by_precision.setdefault(precision,[]).append(err)
                predictions.append({'split':split,'target':target,'request_id':r['request_id'],'worker_id':r['worker_id'],
                                    'group_id':r['group_id'],'actual':actual,'predicted':pred,
                                    'context_tokens':r['features'].get('input_tokens_estimate'),'queued_requests':r['features'].get('queued_requests')})
            results[split]={'n':len(rs),'brier_score' if classifier else 'mae_ms':None if not errors else statistics.mean(errors),
                           'constant_baseline_error':None if not errors else statistics.mean(baseline_errors),
                           'by_precision_error':{k:statistics.mean(v) for k,v in by_precision.items()}}
        report['targets'][target]=results
    if not artifact['models'].get('e2e_ms'):raise ValueError('No end-to-end model could be trained')
    out.mkdir(parents=True);dump(out/'model.json',artifact);dump(out/'evaluation.json',report)
    (out/'heldout-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-unverified',action='store_true');a=p.parse_args();print(json.dumps(train(a.data,a.out,a.allow_unverified),indent=2))
