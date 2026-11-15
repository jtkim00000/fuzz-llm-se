#!/usr/bin/env python3
"""Freeze the 30-minute matched VAST matrix into c1 and c4 traces."""
import argparse, hashlib, json, random
from pathlib import Path
from transformers import AutoTokenizer

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def token_count(tok, messages):
    value=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    return len(value['input_ids'] if hasattr(value,'keys') else value)
def source_rows(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        parts=line.split('\t')
        if len(parts)>=2 and 0<len(parts[0])<=128 and 0<len(parts[1])<=512: rows.append((parts[0],parts[1]))
    if len(rows)<100: raise ValueError('insufficient bounded Reactome records')
    return rows
def make_prompt(tok, rows, start, target, nonce):
    lines=[]; cursor=start
    def render():
        return (f'NONCE: {nonce}\nYou are given a VAST Reactome pathway index. Create a detailed structured catalogue '
                'of the supplied records. Continue listing records and their exact pathway names until your response limit. '
                'Do not discuss the instruction.\n\nINDEX:\n'+'\n'.join(lines))
    while True:
        ident,name=rows[cursor%len(rows)]; lines.append(f'{ident}\t{name}') ; cursor+=1
        if len(lines)%16: continue
        text=render()
        count=token_count(tok,[{'role':'user','content':text}])
        if count>=target*.98:
            while count>target*1.02 and len(lines)>1:
                lines.pop(); text=render(); count=token_count(tok,[{'role':'user','content':text}])
            return text,count
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--lengths',nargs='+',type=int,default=[512,2048,4096,7168]);a=p.parse_args()
    if a.out.exists(): raise ValueError('output exists')
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False); rows=source_rows(a.source); a.out.mkdir(parents=True)
    manifest={'campaign':'tomorrow-30min-bf16-fp8','source_sha256':sha(a.source),'tokenizer':a.tokenizer,'length_targets':a.lengths,'output_targets':[32,256],'concurrencies':[1,4],'prefix_cache_condition':'prefix-reuse-minimized, not verified cold','blocks':{}}
    for concurrency in (1,4):
        sessions=[]; seed=2026111500+concurrency; rng=random.Random(seed)
        cells=[(length,out) for length in a.lengths for out in (32,256)]; rng.shuffle(cells)
        for pos,(length,out) in enumerate(cells):
            logical=f'compact-vast-l{length}-o{out}-c{concurrency}'; nonce=f'cp30-{logical}'
            text,count=make_prompt(tok,rows,(length*17+out*31+concurrency*101)%len(rows),length,nonce)
            turn={'request_id':logical+':0','logical_request_id':logical,'text':text,'output_limit':out,'min_tokens':out,'ignore_eos':True,'streaming':True,'category':'vast_reactome_controlled_catalogue','input_bucket':length,'requested_generated_tokens':out,'concurrency_block':concurrency,'block_seed':seed,'block_position':pos,'nonce':nonce,'chat_template_kwargs':{'enable_thinking':False}}
            sessions.append({'session_id':logical,'group_id':'group:'+logical,'split':'test','arrival_offset_seconds':0,'arrival_pattern':'concurrency','history_mode':'single','turns':[turn]})
            manifest['blocks'].setdefault(str(concurrency),[]).append({'logical_request_id':logical,'actual_input_tokens':count,'requested_output_tokens':out,'position':pos,'nonce':nonce})
        path=a.out/f'c{concurrency}.jsonl';path.write_text(''.join(json.dumps(s,separators=(',',':'))+'\n' for s in sessions));manifest['blocks'][str(concurrency)+'_trace_sha256']=sha(path)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
