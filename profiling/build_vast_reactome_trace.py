#!/usr/bin/env python3
"""Build a deterministic, token-verified long-context VAST Reactome trace."""
import argparse, hashlib, json, random
from pathlib import Path
from transformers import AutoTokenizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    result=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields=line.split('\t')
        if len(fields) >= 2 and fields[0].strip() and fields[1].strip():
            ident,name=fields[0].strip(),fields[1].strip()
            # The source can include a malformed expanded line.  It is not a
            # pathway-name record suitable for a bounded-context serving trace.
            if len(ident) <= 128 and len(name) <= 512:
                result.append((ident,name))
    if len(result) < 100:
        raise ValueError('Reactome source did not contain 100 usable records')
    return result


def prompt(index, target, source, minimum_tokens, maximum_tokens, tokenizer):
    """Use a rotating, character-bounded VAST slice before exact token verification."""
    lines=[]; chars=0; cursor=(index * 37) % len(source)
    # Character bounding avoids a malformed raw data line causing unbounded
    # tokenizer work.  Names are still copied from VAST, just at a useful index
    # display length; target answers retain the full name.
    while chars < 19000:
        ident, name=source[cursor % len(source)]
        line=f'{ident}\t{name[:120]}'
        lines.append(line); chars+=len(line)+1; cursor+=1
    while lines:
        text=(
            'You are auditing a VAST-provided Reactome pathway index. The tab-separated '
            'records below map a stable pathway identifier to its pathway name. Use only this '
            'index. Return exactly one JSON object with this schema: '
            '{"records":[{"id":"<target id>","name":"<exact pathway name>"}]}.\n\n'
            'INDEX:\n'+'\n'.join(lines)+f'\n\nTARGET ID: {target[0]}\n')
        rendered=tokenizer.apply_chat_template([{'role':'user','content':text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        token_ids=rendered['input_ids'] if hasattr(rendered,'keys') else rendered
        count=len(token_ids)
        if minimum_tokens <= count <= maximum_tokens:
            return text,count
        if count > maximum_tokens:
            lines=lines[:max(1,int(len(lines)*maximum_tokens/count*0.98))]
            continue
        # The original source is text-heavy; this guard makes an unexpectedly
        # token-sparse tokenizer output a visible error rather than looping.
        raise ValueError(f'VAST context too short after bounded construction for request {index}: {count}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261114);p.add_argument('--min-input-tokens',type=int,default=5200)
    p.add_argument('--max-input-tokens',type=int,default=6200);p.add_argument('--output-limit',type=int,default=128)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Trace output already exists')
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    source=records(a.source); rng=random.Random(a.seed); selected=rng.sample(source,a.count)
    sessions=[]; token_counts=[]
    for index,target in enumerate(selected):
        text,count=prompt(index,target,source,a.min_input_tokens,a.max_input_tokens,tok)
        logical_id=f'vast-reactome-{index:03d}'
        expected={'records':[{'id':target[0],'name':target[1]}]}
        sessions.append({'session_id':logical_id,'group_id':'group:'+logical_id,'split':'train',
            'arrival_offset_seconds':0,'arrival_pattern':'concurrency','history_mode':'single',
            'turns':[{'request_id':logical_id+':0','logical_request_id':logical_id,'text':text,
                'output_limit':a.output_limit,'streaming':True,'category':'vast_reactome_long_context_retrieval',
                'target_position':'rotating_index','expected':expected,
                'chat_template_kwargs':{'enable_thinking':False}}]})
        token_counts.append(count)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in sessions))
    manifest={'dataset':'VAST ReactomePathways.txt','source_path':str(a.source),'source_sha256':sha256(a.source),
        'trace_sha256':sha256(a.out),'count':a.count,'seed':a.seed,'tokenizer':a.tokenizer,
        'chat_template':'tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)',
        'input_tokens_min':min(token_counts),'input_tokens_max':max(token_counts),'input_tokens_mean':sum(token_counts)/len(token_counts),
        'requested_output_token_limit':a.output_limit,
        'workload_note':'Long, distinct VAST Reactome retrieval contexts exercise attention KV capacity and bandwidth; this trace does not claim multi-turn prefix-cache reuse.'}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
