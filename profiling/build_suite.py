#!/usr/bin/env python3
"""Prepare a versioned retrieval/extraction suite from a local Reactome TSV."""
import argparse, csv, hashlib, json, random
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))

def build(source, out, tokenizer=None, target=8192, count=100, seed=42, namespace="reactome-v1"):
    if out.exists():
        raise ValueError('Output exists; choose a new version directory')
    records=[]
    seen=set()
    with source.open(newline='') as f:
        for line, row in enumerate(csv.reader(f, delimiter='\t'),1):
            if len(row)!=3 or not all(row) or row[0] in seen:
                raise ValueError(f'Invalid or duplicate source row {line}')
            seen.add(row[0]); records.append(dict(zip(['id','name','species'],row)))
    if len(records)<500:
        raise ValueError('Need at least 500 valid unique source records')
    tok=None
    if tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True,trust_remote_code=False)
    measure=(lambda text: len(tok.encode(text,add_special_tokens=False))) if tok else len
    budget=target-256 if tok else target*4
    short=[]; long=[]; prompts=[]; answers=[]; labels=[]
    for i in range(count+20):
        rng=random.Random(f'{namespace}:{seed}:{i}')
        selected=rng.sample(records,min(12000,len(records)))
        lines=[json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in selected]
        lo,hi=1,len(lines)
        while lo<hi:
            mid=(lo+hi+1)//2
            if measure('\n'.join(lines[:mid]))<=budget: lo=mid
            else: hi=mid-1
        selected=selected[:lo]; body='\n'.join(lines[:lo])
        if measure(body)>budget or measure(body)<budget*.8:
            raise ValueError('Source cannot fill the requested context budget')
        uid=f'{namespace}-{seed}-{i:04d}'
        context=f'Input identifier: {uid}\nUse only these supplied database records. Each line is a JSON record.\nBEGIN RECORDS\n{body}\nEND RECORDS\n'
        if i<count:
            for rows,task,cap in [(short,'Summarize the types of records and identify three examples, citing their IDs. Keep the answer concise.',256),(long,'Write a detailed technical report on organizing and validating this data. Include example record IDs, a proposed JSON schema, validation rules, and pseudocode for indexing and querying it.',1024)]:
                text=context+'TASK: '+task
                rows.append({'session_id':uid,'text':text,'output_length':cap,'extra':{'temperature':0}})
                labels.append({'task_id':uid,'kind':'short' if cap==256 else 'long','input_text_tokens':measure(text) if tok else None,'input_characters':len(text)})
        else:
            j=i-count; position=['beginning','middle','end'][j%3]
            index={'beginning':int(len(selected)*.1),'middle':int(len(selected)*.5),'end':int(len(selected)*.9)}[position]
            n=1 if j<10 else 3
            targets=[selected[(index+k)%len(selected)] for k in range(n)]
            ids=[r['id'] for r in targets]
            question='Return the records with these IDs in the stated order: '+json.dumps(ids)+'. Return ONLY a JSON object with exactly one key "records", whose value is an array of objects with exactly the string keys "id", "name", "species". Copy field values exactly. No explanation or Markdown.'
            task={'task_id':uid,'category':'single_record' if n==1 else 'three_records','position':position,'text':context+'TASK: '+question,'output_length':2048}
            prompts.append(task)
            answers.append({'task_id':uid,'expected':{'records':targets}})
            labels.append({'task_id':uid,'kind':'accuracy','position':position,'target_record_indices':[index+k for k in range(n)],'context_record_count':len(selected),'input_text_tokens':measure(task['text']) if tok else None,'input_characters':len(task['text'])})
    out.mkdir(parents=True)
    write_jsonl(out/'performance-short.jsonl',short);write_jsonl(out/'performance-long.jsonl',long)
    write_jsonl(out/'accuracy-prompts.jsonl',prompts);write_jsonl(out/'accuracy-answers.jsonl',answers)
    write_jsonl(out/'labels.jsonl',labels)
    (out/'SOURCE.txt').write_text('Source: Reactome Pathways, https://reactome.org/download/current/ReactomePathways.txt\nLicense: https://reactome.org/license\nCustom retrieval/extraction benchmark; not a standard public benchmark score.\n')
    sidecar=source.with_name('SOURCE.txt')
    if sidecar.exists(): (out/'original-SOURCE.txt').write_bytes(sidecar.read_bytes())
    manifest={'suite_version':'reactome-baseline-v1','source_path':str(source),'source_sha256':sha(source),'seed':seed,'namespace':namespace,'generator_sha256':sha(__file__),'tokenizer_path':tokenizer,'token_count_mode':'matching-tokenizer-text-only' if tok else 'character-preview','input_target':target,'chat_template_overhead_included':False,'performance_requests':count,'accuracy_tasks':20,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'limitations':['No coding or multi-turn accuracy evaluation.','Record positions are relative to record count, not exact tokens.','Token lengths exclude server-added template and prompts.','Output caps do not guarantee generated length.','Cold-cache status must be measured.','20 accuracy tasks are a pilot, not a strong noninferiority test.']}
    (out/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/datasets/bio/reactome/ReactomePathways.txt'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tokenizer');p.add_argument('--tokens',type=int,default=8192);p.add_argument('--count',type=int,default=100);p.add_argument('--seed',type=int,default=42);p.add_argument('--namespace',default='reactome-v1')
    a=p.parse_args()
    if a.tokens<1024: p.error('Use at least 1024 target tokens')
    m=build(a.source,a.out,a.tokenizer,a.tokens,a.count,a.seed,a.namespace)
    print(json.dumps({'directory':str(a.out),'mode':m['token_count_mode'],'performance_requests':a.count,'accuracy_tasks':20}))
