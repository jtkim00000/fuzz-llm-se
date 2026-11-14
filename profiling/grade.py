#!/usr/bin/env python3
"""Strict scoring, with diagnostic labels independent of overall correctness."""
import json

def no_duplicates(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key]=value
    return result

def grade(text, expected):
    result=dict(json_valid=False,field_correctness=False,array_order_correctness=False,
                extra_prose_present=False,passed=False)
    if not isinstance(text,str): return result
    decoder=json.JSONDecoder(object_pairs_hook=no_duplicates,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    try:
        actual=decoder.decode(text);result['json_valid']=True
    except ValueError:
        try:
            start=text.index('{');actual,end=decoder.raw_decode(text,start)
            result['extra_prose_present']=bool(text[:start].strip() or text[end:].strip())
        except ValueError: return result
    if isinstance(actual,dict) and set(actual)=={'records'} and isinstance(actual['records'],list):
        records=actual['records']; refs=expected['records']
        if all(isinstance(r,dict) for r in records):
            result['array_order_correctness']=[r.get('id') for r in records]==[r['id'] for r in refs]
            canonical=lambda rows: sorted(json.dumps(r,sort_keys=True) for r in rows)
            result['field_correctness']=canonical(records)==canonical(refs)
    result['passed']=result['json_valid'] and actual==expected
    return result

if __name__=='__main__':
    expected={'records':[{'id':'x','name':'Name','species':'Species'}]}
    assert grade(json.dumps(expected),expected)['passed']
    assert not grade('{"records":[]}',expected)['passed']
    assert not grade('{"records":[],"records":[]}',expected)['json_valid']
    print('Strict grader controls passed.')
