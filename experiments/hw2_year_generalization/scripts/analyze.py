from __future__ import annotations

import copy
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from .common import ROOT, cfg, classify, offline, parse, read_jsonl, run_dir, semantic_parse, sha, verify_lock, write_json, write_jsonl
from .statistics import CATEGORIES, base_effects, csv_file, estimate

METRICS = ('delta_attr', 'delta_C', 'delta_U', 'delta_type', 'forward_switch', 'reverse_switch')


def main():
    offline(); verify_lock(); c = cfg(); out = ROOT/'results'
    pilot = json.loads((run_dir()/'pilot_audit.json').read_text())
    if not pilot['main_allowed']:
        raise RuntimeError('Main capability gate failed')
    requests = read_jsonl(ROOT/'data/main_requests.jsonl')
    raw = read_jsonl(run_dir()/'main_raw.jsonl')
    by_id = {r['request_id']:r for r in raw}
    if len(raw)!=len(by_id) or len(raw)!=c['data']['main_calls'] or set(by_id)!={r['request_id'] for r in requests}:
        raise RuntimeError('Incomplete/duplicate/unexpected main records')
    strict = []
    for req in requests:
        rec = by_id[req['request_id']]; p = parse(rec['raw_output'])
        if p != rec['parse'] or rec['input_hash'] != req['input_hash'] or rec['system_error'] is not None:
            raise RuntimeError({'raw_replay_failure':req['request_id']})
        strict.append({**{k:v for k,v in req.items() if k!='messages'},**rec,
                       'category':classify(req,p),'correct':p['valid'] and p['normalized_value']==req['gold_value']})
    write_jsonl(out/'request_level.jsonl',strict)
    bids=sorted({r['base_id'] for r in strict})
    assert len(bids)==96 and len({r['setup_hash'] for r in raw})==1
    indices=np.random.default_rng(c['analysis']['bootstrap_seed']).integers(0,len(bids),size=(c['analysis']['bootstrap_repetitions'],len(bids)))
    np.save(out/'bootstrap_indices.npy',indices)
    write_json(out/'bootstrap_manifest.json',{'base_ids':bids,'seed':c['analysis']['bootstrap_seed'],
               'repetitions':len(indices),'unit':'base, retaining every template/condition','sha256':sha(out/'bootstrap_indices.npy')})
    policies={'STRICT':strict}
    for policy,aliases in [('SEMANTIC',set(c['analysis']['abstention_aliases'])),('NULL_NA',{'null','na'})]:
        rs=copy.deepcopy(strict)
        for r in rs:
            if r['parse']['valid'] and r['parse']['normalized_value'] in aliases:
                r['parse']={**r['parse'],'value':None,'normalized_value':None}
                r['category']='ABSTAIN'; r['correct']=r['state']=='MISSING'
        policies[policy]=rs
    contrasts=[]; rates=[]; control_rows=[]; decisions={}; base_rows=[]
    scopes=[('ALL',None,None)]+[(f'template={t}',[t],None) for t in c['data']['templates']]+[(f'order={o}',None,[o]) for o in c['data']['orders']]+[(f'without_template={t}',[x for x in c['data']['templates'] if x!=t],None) for t in c['data']['templates']]
    for policy,rs in policies.items():
        for scope,templates,orders in (scopes if policy!='NULL_NA' else scopes[:1]):
            vals=base_effects(rs,templates,orders)
            assert [v['base_id'] for v in vals]==bids
            if scope=='ALL':
                base_rows += [{'policy':policy,**v} for v in vals]
            for metric in METRICS:
                contrasts.append({'policy':policy,'subset':scope,'metric':metric,
                                  'role':'PRIMARY' if policy=='STRICT' and scope=='ALL' and metric in ('delta_attr','delta_C','delta_type') else 'PRESPECIFIED_SECONDARY',
                                  **estimate([v[metric] for v in vals],indices),'n_base':len(vals)})
        all_est={r['metric']:r for r in contrasts if r['policy']==policy and r['subset']=='ALL'}
        decisions[policy]={'h1_supported':all_est['delta_attr']['ci95_low']>0,
                           'h2_supported':all_est['delta_C']['ci95_low']>0 and all_est['delta_type']['ci95_low']>0,
                           'effects':all_est}
        for family in ('MISSING','N0','PRESENT','T0','T1'):
            all_family=[r for r in rs if r['family']==family]
            correct_base=[np.mean([r['correct'] for r in all_family if r['base_id']==bid]) for bid in bids]
            control_rows.append({'policy':policy,'family':family,'correct_n':sum(r['correct'] for r in all_family),
                                 'n':len(all_family),**estimate(correct_base,indices)})
            for placement in (('MIDDLE','ADJACENT') if family in ('MISSING','N0','PRESENT') else (None,)):
                sub=[r for r in all_family if r['placement']==placement]
                for cat in CATEGORIES+(('TARGET_CORRECT',) if family in ('PRESENT','T1') else ()):
                    per_base=[np.mean([r['category']==cat for r in sub if r['base_id']==bid]) for bid in bids]
                    rates.append({'policy':policy,'family':family,'placement':placement,'category':cat,
                                  'count':sum(r['category']==cat for r in sub),'n':len(sub),**estimate(per_base,indices)})
    csv_file(out/'contrasts.csv',contrasts); csv_file(out/'base_effects.csv',base_rows)
    csv_file(out/'outcome_rates.csv',rates); csv_file(out/'control_accuracy.csv',control_rows)
    # Exact cell counts and abstention renderings remain visible, independent of policy.
    cells=defaultdict(list); notation=defaultdict(Counter)
    for r in strict:
        cells[tuple(r[k] for k in ('family','template','assignment','order','placement'))].append(r)
        p=r['parse']
        label=('INVALID' if not p['valid'] else 'LITERAL_NULL' if p['value'] is None else
               'QUOTED_NULL' if p['normalized_value']=='null' else 'OTHER_ALIAS' if p['normalized_value'] in c['analysis']['abstention_aliases'] else 'OTHER_VALUE')
        notation[(r['family'],r['placement'])][label]+=1
    cell_rows=[]
    for key,rs in sorted(cells.items(),key=lambda kv:str(kv[0])):
        counts=Counter(r['category'] for r in rs)
        cell_rows.append({**dict(zip(('family','template','assignment','order','placement'),key)),
                          'n':len(rs),'correct_n':sum(r['correct'] for r in rs),
                          **{cat+'_n':counts[cat] for cat in CATEGORIES+('TARGET_CORRECT',)}})
    csv_file(out/'condition_counts.csv',cell_rows)
    notation_rows=[{'family':key[0],'placement':key[1],**dict(counts),'n':sum(counts.values())} for key,counts in notation.items()]
    csv_file(out/'abstention_notation.csv',notation_rows)
    absent=[r for r in strict if r['family']=='MISSING' and r['category']=='CONTEXT_ABSENT']
    write_json(out/'context_absent_cases.json',[{k:r[k] for k in ('request_id','placement','raw_output','parse')} for r in absent])
    old=json.loads((ROOT/'reference/salary_summary.json').read_text())
    old_sens=list(csv.DictReader((ROOT/'reference/salary_notation_sensitivity.csv').open()))
    comparison=[]
    for metric in ('delta_attr','delta_C','delta_U','delta_type'):
        for attribute,policy,entry in [('Salary','STRICT',old['primary'][metric]),
                                        ('Year','STRICT',decisions['STRICT']['effects'][metric]),
                                        ('Salary','NULL_NA',next(r for r in old_sens if r['variant']=='quoted_null_and_na_as_abstain' and r['metric']==metric)),
                                        ('Year','NULL_NA',decisions['NULL_NA']['effects'][metric])]:
            comparison.append({'attribute':attribute,'policy':policy,'metric':metric,
                               **{k:float(entry[k]) for k in ('estimate','ci95_low','ci95_high')},
                               'interpretation':'separate task datasets; no paired cross-task causal contrast'})
    csv_file(out/'salary_year_comparison.csv',comparison)
    runtime=json.loads((run_dir()/'main_runtime.json').read_text())
    summary={'status':'COMPLETE','n_base':len(bids),'main_calls':len(strict),'primary_calls':2304,
             'decisions':decisions,'controls':control_rows,'pilot':pilot,'runtime':runtime,
             'strict_valid':sum(r['parse']['valid'] for r in strict),'truncated':sum(r['truncated'] for r in strict),
             'context_absent_primary':len(absent),'context_absent_numeric':sum(bool(re.fullmatch(r'\d+',r['parse']['normalized_value'])) for r in absent),
             'context_absent_sparse_under_20':len(absent)<20,
             'zero_event_base_any_event_95_upper':1-.05**(1/len(bids)),
             'quoted_null_primary':sum(r['family']=='MISSING' and r['parse']['valid'] and r['parse']['normalized_value']=='null' for r in strict),
             'protocol_unchanged':True,'parser_replay_mismatches':0,
             'cross_task_comparison_is_descriptive':True}
    write_json(out/'summary.json',summary)
    examples=[]
    req_map={r['request_id']:r for r in requests}
    for cat in CATEGORIES:
        candidates=sorted((r for r in strict if r['family']=='MISSING' and r['category']==cat),key=lambda r:r['request_id'])
        if candidates:
            r=candidates[0]; examples.append({**r,'messages':req_map[r['request_id']]['messages']})
    write_json(out/'representative_cases.json',examples)
    figures(rates,comparison)
    report(summary,rates,contrasts,comparison,notation_rows,examples)
    verify_lock()
    files=[p for folder in (out,ROOT/'figures',run_dir()) for p in folder.iterdir() if p.is_file()]
    files += [ROOT/'final_report_ko.md',ROOT/'reproduce.md',ROOT/'ai_usage_log.md']
    write_json(ROOT/'completion_manifest.json',{'status':'COMPLETE','run_id':c['run_id'],
               'completed_utc':datetime.now(timezone.utc).isoformat(),'main_calls':len(strict),'primary_calls':2304,
               'h1_supported':decisions['STRICT']['h1_supported'],'h2_supported':decisions['STRICT']['h2_supported'],
               'protocol_lock_sha256':sha(run_dir()/'protocol_lock.json'),
               'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    print(json.dumps({'status':'COMPLETE','main_calls':len(strict),'decisions':decisions,'controls':control_rows,
                      'context_absent_primary':len(absent)},ensure_ascii=False,indent=2))


def figures(rates,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for ax,policy in zip(axes,('STRICT','SEMANTIC')):
        bottom=np.zeros(2)
        for category,color in zip(CATEGORIES,('#7d9b76','#db7949','#baaa80','#826ba1','#a5acb0')):
            vals=np.array([next(r['estimate'] for r in rates if r['policy']==policy and r['family']=='MISSING' and r['placement']==p and r['category']==category) for p in ('MIDDLE','ADJACENT')])*100
            ax.bar([0,1],vals,bottom=bottom,width=.58,color=color,label=category.replace('_',' ')); bottom+=vals
        ax.set(xticks=[0,1],xticklabels=['Middle','Adjacent'],ylim=(0,100),ylabel='Planned requests (%)',title='Founding year: '+policy.lower())
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(ROOT/'figures/outcome_rates.png',dpi=200); fig.savefig(ROOT/'figures/outcome_rates.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,metric in zip(axes,('delta_attr','delta_C','delta_type')):
        for i,attribute in enumerate(('Salary','Year')):
            r=next(r for r in comparison if r['attribute']==attribute and r['policy']=='STRICT' and r['metric']==metric)
            ax.errorbar(i,r['estimate']*100,yerr=[[100*(r['estimate']-r['ci95_low'])],[100*(r['ci95_high']-r['estimate'])]],fmt='o',capsize=5,color=('#365f7d','#bc6a44')[i])
        ax.axhline(0,linestyle='--',color='gray',linewidth=1)
        ax.set(xticks=[0,1],xticklabels=['Salary','Founding year'],ylabel='Percentage points (95% CI)',title=metric); ax.margins(x=.3)
    fig.suptitle('Separate-task replication, strict evaluation')
    fig.savefig(ROOT/'figures/salary_year_comparison.png',dpi=200); fig.savefig(ROOT/'figures/salary_year_comparison.pdf'); plt.close(fig)


def report(s,rates,contrasts,comparison,notation,examples):
    def eff(entry):
        return f"{100*entry['estimate']:+.2f} [{100*entry['ci95_low']:+.2f}, {100*entry['ci95_high']:+.2f}]"
    def decision(policy,h):
        return '지지' if s['decisions'][policy][h+'_supported'] else '지지되지 않음'
    lines=['# HW2 두 번째 속성 재현: 설립연도 / 감사연도','',
           '이 문서는 실제 실행과 분석을 보존하는 연구 기록이다. 제출용 학생 작성 원고가 아니다.','',
           '## 결과와 판정','',
           f"신규 96개 base × 세 template의 본실험·대조군 6,336회를 실제 실행했다. 독립 개발 12개 base는 792회이며 본 통계에서 제외했다. 엄격 평가에서 H1은 **{decision('STRICT','h1')}**, H2는 **{decision('STRICT','h2')}**다.",'',
           '| 평가 규칙 | delta_attr | delta_C | delta_U | delta_type | H1 | H2 |',
           '|---|---:|---:|---:|---:|---|---|']
    for policy in ('STRICT','SEMANTIC','NULL_NA'):
        d=s['decisions'][policy]
        lines.append('| '+policy+' | '+' | '.join(eff(d['effects'][m]) for m in ('delta_attr','delta_C','delta_U','delta_type'))+f" | {decision(policy,'h1')} | {decision(policy,'h2')} |")
    lines += ['', '단위는 %p이며 대괄호는 95% paired base-bootstrap CI다. H1은 delta_attr 하한>0, H2는 delta_C와 delta_type 하한이 모두>0일 때만 지지한다. 0 포함은 방향 미확립, 전 구간 음수는 예측 방향에 반하는 근거다. STRICT가 일차 분석이고 나머지는 실행 전 고정한 보조 분석이다.',
              '', '## 기존 Salary와 비교','',
              '| 속성 | delta_attr | delta_C | delta_type |','|---|---:|---:|---:|']
    for attr in ('Salary','Year'):
        vals=[next(r for r in comparison if r['attribute']==attr and r['policy']=='STRICT' and r['metric']==m) for m in ('delta_attr','delta_C','delta_type')]
        lines.append('| '+attr+' | '+' | '.join(eff(v) for v in vals)+' |')
    lines += ['', 'Salary 결과는 기존 동결 자료를 참조했다. 새 조직·연도·질문·필드 문구·값 token 길이·배경 일부가 함께 달라졌으므로 두 행의 차이를 속성 정체성만의 인과효과로 해석하지 않는다. 두 과제의 base는 서로 대응되지 않으며 cross-task paired 검정은 하지 않았다. 공통 null/N/A 정책 비교도 salary_year_comparison.csv에 별도 보존했다.',
              '', '## 데이터와 실행 설정','',
              '- 모델: Qwen/Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. FP16, SDPA, 공식 chat template, 비제약 greedy, repetition penalty 1.1, max 96 new tokens.',
              '- 각 base에 1900–2025 범위의 서로 다른 세 연도와 신규 조직 세 개를 배정했다. 전체 base의 연도 triple은 중복되지 않지만 개별 연도는 다른 base에서 반복될 수 있다.',
              '- S1/S2는 founding year/audit year 라벨만 교환하고 숫자·조직·문서 순서를 유지한다. N0는 audit year/review year이며 target의 founding year가 없다.',
              '- Target은 처음, 질문은 마지막이다. 약 2,000토큰의 같은 텍스트 블록을 재배열하며, Middle/Adjacent 총 길이와 질문 위치는 정확히 같다. S1/S2의 총 길이와 숫자 token span도 같다.',
              '- 공개 Gutenberg ebook 11의 기존 원문을 재사용하고 필터·발췌·원문 hash를 보존했다. Template별 배경 발췌도 달라지므로 template 차이는 순수 문구 효과가 아니다.',
              '- 생성 전에 코드·조건·평가·데이터·이전 Salary 참조 자료를 해시 잠금했다. 결과로 prompt나 사례를 선택하지 않았다.',
              '', '## 기본 능력 대조군','',
              '| 조건 | Strict 정답 / 전체 | Strict 정확도 | Semantic 정확도 |','|---|---:|---:|---:|']
    for family in ('T0','T1','PRESENT','N0'):
        a=next(r for r in s['controls'] if r['policy']=='STRICT' and r['family']==family)
        b=next(r for r in s['controls'] if r['policy']=='SEMANTIC' and r['family']==family)
        lines.append(f"| {family} | {a['correct_n']}/{a['n']} | {100*a['estimate']:.2f}% | {100*b['estimate']:.2f}% |")
    lines += ['', 'T0는 target-only missing, T1은 target-only present, PRESENT는 정답이 있는 다문서 조건, N0는 질문 속성이 없는 두 donor 조건이다. 개발 gate는 strict-valid≥95%, T0 의미상 기권≥90%, T1 strict 정확도≥90%로 미리 정했다. Donor 효과나 가설 유의성은 진입 기준으로 쓰지 않았다.',
              '', '## 출력 유형','', '| 평가 | 배치 | 기권 | Donor 일치 | 기타 문맥 내 | 문맥 외 | Invalid |','|---|---|---:|---:|---:|---:|---:|']
    for policy in ('STRICT','SEMANTIC'):
        for placement in ('MIDDLE','ADJACENT'):
            rs=[next(r for r in rates if r['policy']==policy and r['family']=='MISSING' and r['placement']==placement and r['category']==cat) for cat in CATEGORIES]
            lines.append('| '+policy+' | '+placement+' | '+' | '.join(f"{r['count']}/{r['n']} ({100*r['estimate']:.2f}%)" for r in rs)+' |')
    lines += ['', f"Primary에서 문자열 null은 {s['quoted_null_primary']}건이다. Strict에서는 JSON null과 구분하며, null이 system 지시문에 등장하므로 문자열 null은 OTHER_IN_CONTEXT로 분류될 수 있다. 이 범주를 실제 내용의 복사로 일괄 해석하지 않는다.",
              'SEMANTIC은 사전 목록의 유효 문자열만 기권으로 읽는다: `'+', '.join(cfg()['analysis']['abstention_aliases'])+'`. NULL_NA는 null/na만 허용해 이전 Salary 민감도와 정책을 맞춘다. 숫자 JSON이나 잘못된 JSON은 어느 규칙에서도 자동 보정하지 않는다.',
              f"Strict context-absent는 {s['context_absent_primary']}건, 그중 숫자 문자열은 {s['context_absent_numeric']}건이다. 숫자와 비수치 기권 표현 등은 context_absent_cases.json에서 구분할 수 있다.",
              '', '## Template와 순서','', '| 부분집합 | delta_attr | delta_C | delta_type |','|---|---:|---:|---:|']
    for scope in dict.fromkeys(r['subset'] for r in contrasts if r['policy']=='STRICT' and r['subset']!='ALL'):
        vals=[next(r for r in contrasts if r['policy']=='STRICT' and r['subset']==scope and r['metric']==m) for m in ('delta_attr','delta_C','delta_type')]
        lines.append('| '+scope+' | '+' | '.join(eff(v) for v in vals)+' |')
    f=s['decisions']['STRICT']['effects']['forward_switch']; b=s['decisions']['STRICT']['effects']['reverse_switch']
    lines += ['', f"완전한 S1=v1→S2=v2 전환은 {100*f['estimate']:.2f}% [{100*f['ci95_low']:.2f}, {100*f['ci95_high']:.2f}], 반대 전환은 {100*b['estimate']:.2f}%다. 이는 delta_attr와 다르다. 부분집합 CI는 개별 보조 구간이며 동시 검정이나 새로운 주 가설이 아니다.",
              '', '## 무결성·자원·해석 범위','',
              f"- Strict-valid {s['strict_valid']}/{s['main_calls']}, truncation {s['truncated']}, 시스템 오류 0, 원출력 재파싱 불일치 0.",
              '- 96개 base를 단위로 2,000회 공통 paired bootstrap, seed 2026110203. 같은 base의 모든 조건과 template를 함께 재표집했다.',
              f"- 0-event bootstrap [0,0]은 발생 확률 0의 증명이 아니다. 96개 base에서 한 번도 발생하지 않은 경우 base당 any-event 확률의 단측 95% 상한은 {100*s['zero_event_base_any_event_95_upper']:.2f}%다.",
              f"- Strict context-absent 20건 미만 여부: {s['context_absent_sparse_under_20']}. 희소한 경우 유형 간 차이 해석에 한계가 있다.",
              f"- 실제 GPU {s['runtime']['gpu']}, main generation {s['runtime']['generation_seconds']:.1f}초, peak allocated/reserved {s['runtime']['peak_allocated_bytes']/2**30:.2f}/{s['runtime']['peak_reserved_bytes']/2**30:.2f} GiB.",
              '- 이 결과는 이 모델·연도 속성·세 template·단일 배경 말뭉치·길이의 행동 결과다. 내부 binding 회로나 모든 속성/모델/실문서로의 일반화는 검정하지 않았다.',
              '- 길이를 맞춰도 donor 절대 위치와 질문까지 거리는 함께 변하므로 독립 위치 메커니즘을 분리한 결과는 아니다.',
              '', '## 대표 원출력','', '각 primary 범주에서 request ID 순으로 첫 사례를 선택했다. 전체 입력은 results/representative_cases.json에 있다.','']
    for r in examples:
        lines.append(f"- `{r['request_id']}` / {r['category']}: `{r['raw_output'].replace(chr(10),' ')}`")
    lines += ['', '## 산출물','', '- results/summary.json, contrasts.csv, outcome_rates.csv, control_accuracy.csv, salary_year_comparison.csv',
              '- results/abstention_notation.csv, context_absent_cases.json, representative_cases.json',
              '- figures/outcome_rates.png 및 PDF, figures/salary_year_comparison.png 및 PDF',
              '- data/*_requests.jsonl, runs/run_20261102_founding_year/*_raw.jsonl, protocol_lock.json',
              '- reproduce.md, ai_usage_log.md, completion_manifest.json','']
    (ROOT/'final_report_ko.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
