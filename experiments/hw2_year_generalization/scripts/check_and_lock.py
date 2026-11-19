from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from .common import ROOT, REPO, cfg, classify, delta_attr, normalize, occurs, parse, read_jsonl, run_dir, semantic_parse, sha, stable_hash, write_json
from .statistics import base_effects, estimate


def checks():
    valid = ['{"value":null}', '{"value":"1998"}', '{"value":"null"}']
    invalid = ['{"value":1998}', '{"value":true}', '{"value":""}', '{"value":"..."}',
               '{"value":"1","value":"2"}', 'null', '```json\n{"value":null}\n```',
               '{"value":null,"why":"missing"}', '{"value":NaN}', 'answer: {"value":null}']
    assert all(parse(r)['valid'] for r in valid)
    assert not any(parse(r)['valid'] for r in invalid)
    assert normalize('１９９８') == '1998'
    assert occurs('1998', 'Year: 1998.') and not occurs('1998', '119980')
    for alias in cfg()['analysis']['abstention_aliases'] + ['N/A', 'NULL', 'Not provided']:
        p = parse(json.dumps({'value':alias}))
        assert p['normalized_value'] is not None and semantic_parse(p)['normalized_value'] is None
    for raw in invalid + ['{"value":"1998"}', '{"value":"not known"}']:
        assert semantic_parse(parse(raw)) == parse(raw)
    request = {'family':'MISSING','state':'MISSING','v1':1998,'v2':2004,'target_value':1977,
               'messages':[{'content':'founding year 1998, audit year 2004, note 1950. null'}]}
    expected = {'{"value":null}':'ABSTAIN','{"value":"1998"}':'DONOR_MATCH',
                '{"value":"2004"}':'DONOR_MATCH','{"value":"1950"}':'OTHER_IN_CONTEXT',
                '{"value":"1900"}':'CONTEXT_ABSENT','{"value":1998}':'INVALID',
                '{"value":"null"}':'OTHER_IN_CONTEXT'}
    for raw, category in expected.items():
        assert classify(request,parse(raw)) == category
    assert classify({**request,'state':'PRESENT','family':'PRESENT'},parse('{"value":"1977"}')) == 'TARGET_CORRECT'
    assert delta_attr(parse('{"value":"1998"}'),parse('{"value":"2004"}'),1998,2004) == 1
    assert delta_attr(parse('{"value":"2004"}'),parse('{"value":"1998"}'),1998,2004) == -1
    assert delta_attr(parse('{"value":1998}'),parse('{"value":null}'),1998,2004) == 0
    # Known complete factorial: Middle abstains, Adjacent follows the attribute.
    fake=[]
    for bid in ('b0','b1'):
        for template in ('KV','PROSE','BULLET'):
            for order in ('AB','BA'):
                for placement in ('MIDDLE','ADJACENT'):
                    for assignment in ('S1','S2'):
                        value = None if placement=='MIDDLE' else str(1998 if assignment=='S1' else 2004)
                        p=parse(json.dumps({'value':value}))
                        fake.append({**request,'base_id':bid,'template':template,'order':order,
                                     'placement':placement,'assignment':assignment,'parse':p,'category':classify(request,p)})
    for v in base_effects(fake):
        assert v['delta_attr']==.5 and v['delta_C']==1 and v['delta_U']==0 and v['delta_type']==1
    assert estimate([1.,1.],np.array([[0,0],[1,1],[0,1]])) == {'estimate':1.,'ci95_low':1.,'ci95_high':1.}
    for row in fake:
        if row['placement']=='ADJACENT' and row['template']!='KV':
            row['parse']=parse('{"value":"1900"}'); row['category']='CONTEXT_ABSENT'
    for v in base_effects(fake):
        assert np.isclose(v['delta_C'],1/3) and np.isclose(v['delta_type'],-1/3)
    pool={p['paragraph_id']:p['text'] for p in read_jsonl(ROOT/'source/filtered_paragraphs.jsonl')}
    for bg in read_jsonl(ROOT/'data/backgrounds.jsonl'):
        for key in ('prefix','suffix','solo_suffix'):
            for part in bg[key]:
                assert pool[part['paragraph_id']].startswith(part['text'])
                assert not re.search(r'\d|\b(founding|audit|review|salary|budget|revenue|usd)\b',part['text'],re.I)
    names=set(); triples=set(); all_ids=set(); counts_by_split={}
    for split in ('dev','main'):
        bases=read_jsonl(ROOT/f'data/{split}_bases.jsonl')
        assert len(bases)==cfg()['data'][split+'_bases']
        assert sum(b['v1']<b['v2'] for b in bases)==len(bases)//2
        for b in bases:
            years=[b[k] for k in ('v1','v2','target_value')]
            assert len(set(years))==3 and all(1900<=y<=2025 for y in years)
            triple=tuple(sorted(years)); assert triple not in triples; triples.add(triple)
            for role in ('target','donor1','donor2'):
                assert b[role+'_name'] not in names; names.add(b[role+'_name'])
        rs=read_jsonl(ROOT/f'data/{split}_requests.jsonl')
        assert len(rs)==cfg()['data'][split+'_calls']
        index={(r['base_id'],r['template'],r['family'],r['assignment'],r['order'],r['placement']):r for r in rs}
        assert len(index)==len(rs)
        for r in rs:
            assert r['request_id'] not in all_ids; all_ids.add(r['request_id'])
            assert r['input_hash']==stable_hash(r['messages'])
            user=r['messages'][1]['content']; docs=re.findall(r'<document>(.*?)</document>',user,re.S)
            assert r['target_name'] in docs[0]
            assert (str(r['target_value']) in docs[0])==(r['state']=='PRESENT')
            assert user.endswith(f"Question: What is the recorded founding year for {r['target_name']}?")
            assert abs(r['input_token_count']-2000)<=64
            for i,(donor,field) in enumerate((('donor1','v1'),('donor2','v2'))):
                if r['family'] in ('T0','T1'):
                    assert r[donor+'_name'] not in user and len(docs)==1
                else:
                    doc=next(d for d in docs if r[donor+'_name'] in d)
                    label={'S1':('founding year','audit year'),'S2':('audit year','founding year'),'N0':('audit year','review year')}[r['assignment']][i]
                    assert label in doc and str(r[field]) in doc
                    assert user.count(str(r[field]))==1 and len(docs)==3
            if r['placement']=='MIDDLE':
                a=index[(r['base_id'],r['template'],r['family'],r['assignment'],r['order'],'ADJACENT')]
                assert .45<=r['donor_midpoint_fraction']<=.55 and a['donor_query_distance']<=2
                assert r['input_token_count']==a['input_token_count'] and r['token_spans']['question']==a['token_spans']['question']
                assert r['block_multiset_hash']==a['block_multiset_hash']
            if r['assignment']=='S1':
                s2=index[(r['base_id'],r['template'],r['family'],'S2',r['order'],r['placement'])]
                assert r['input_token_count']==s2['input_token_count']
                assert all(r['token_spans'][k]==s2['token_spans'][k] for k in ('v1','v2'))
                mask=lambda x:re.sub(r'founding year|audit year','ATTR',x)
                assert mask(user)==mask(s2['messages'][1]['content'])
        assert set(Counter((r['base_id'],r['template']) for r in rs).values())=={22}
        counts_by_split[split]=len(rs)
    # Independently inspect the recorded prior base files for name reuse.
    audit=json.loads((ROOT/'data/data_audit.json').read_text())
    def prior_names(obj):
        if isinstance(obj,dict):
            for key,val in obj.items():
                if isinstance(val,str) and ('company' in key or 'organization' in key or key.endswith('_name')):
                    yield val
                else:
                    yield from prior_names(val)
        elif isinstance(obj,list):
            for val in obj:
                yield from prior_names(val)
    for filename in audit['prior_base_files_checked']:
        assert names.isdisjoint(prior_names(read_jsonl(REPO/filename)))
    return {'pass':True,'parser_cases':len(valid)+len(invalid),'taxonomy_cases':len(expected)+1,
            'semantic_aliases_and_invalid_nonrepair':True,'known_positive_negative_factorial':True,
            'actual_request_hashes_and_pair_geometry':True,'corpus_excerpt_provenance':True,
            'names_disjoint_from_prior_and_across_splits':True,'unique_year_triples':len(triples),
            'requests_checked':counts_by_split}


def main():
    if (run_dir()/'protocol_lock.json').exists():
        raise RuntimeError('Lock already exists; refuse replacement')
    if any(run_dir().glob('*_raw.jsonl')):
        raise RuntimeError('Refuse prospective lock after outputs exist')
    assert json.loads((ROOT/'data/data_audit.json').read_text())['status']=='PASS'
    result=checks(); write_json(ROOT/'data/static_checks.json',result)
    write_json(ROOT/'source/design_references.json',{name:sha(REPO/name) for name in
               ('22_When_Missing_Fields_Borrow_.pdf','2026_2_NLP_HW_Guidelines.pdf')})
    files=[ROOT/name for name in ('config.json','experiment_spec.md','requirements_frozen.txt','requirements_minimal.txt')]
    files+=sorted((ROOT/'scripts').glob('*.py'))
    for directory in ('source','data','reference'):
        files+=sorted(p for p in (ROOT/directory).iterdir() if p.is_file())
    lock={'status':'LOCKED','locked_utc':datetime.now(timezone.utc).isoformat(),
          'before_any_development_or_main_model_output':True,
          'hashes':{str(p.relative_to(ROOT)):sha(p) for p in files},
          'hypotheses':cfg()['analysis'],'model':cfg()['model'],'generation':cfg()['generation']}
    write_json(run_dir()/'protocol_lock.json',lock)
    print(json.dumps({'checks':result,'locked_files':len(files)},indent=2))


if __name__=='__main__':
    main()
