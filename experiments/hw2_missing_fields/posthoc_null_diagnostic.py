"""Post-hoc abstention-notation sensitivity; never replaces locked primary metrics."""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np

from experiments.hw2_missing_fields.scripts.analyze import base_effects, csv_file, estimate
from experiments.hw2_missing_fields.scripts.common import ROOT, offline, read_jsonl, sha, verify_lock, write_json


def main():
    offline(); verify_lock()
    rows = read_jsonl(ROOT / 'results/request_level.jsonl')
    indices = np.load(ROOT / 'results/bootstrap_indices.npy')
    primary = base_effects(rows)
    variants = [('strict_primary', set()), ('quoted_null_as_abstain', {'null'}),
                ('quoted_null_and_na_as_abstain', {'null', 'na'})]
    contrasts, outcomes, checks = [], [], {}
    diagnostic_rows = None
    for name, aliases in variants:
        rs = copy.deepcopy(rows)
        changed = 0
        for row in rs:
            if row['parse']['valid'] and row['parse']['normalized_value'] in aliases:
                row['parse']['value'] = None
                row['parse']['normalized_value'] = None
                row['category'] = 'ABSTAIN'
                row['correct'] = row['state'] == 'MISSING'
                changed += 1
        effects = base_effects(rs)
        for metric in ('delta_attr', 'delta_C', 'delta_U', 'delta_type'):
            contrasts.append({'variant': name, 'metric': metric,
                              'status': 'REGISTERED_PRIMARY' if name == 'strict_primary' else 'POST_HOC_SENSITIVITY',
                              **estimate([r[metric] for r in effects], indices)})
        if name == 'quoted_null_as_abstain':
            for a, b in zip(primary, effects):
                assert a['base_id'] == b['base_id']
                for metric in ('delta_attr', 'delta_C', 'delta_U', 'delta_type'):
                    assert a[metric] == b[metric]
            diagnostic_rows = rs
        checks[name] = {'changed_outputs_all_families': changed}
        for placement in ('MIDDLE', 'ADJACENT'):
            subset = [r for r in rs if r['family'] == 'MISSING' and r['placement'] == placement]
            counts = Counter(r['category'] for r in subset)
            for category in ('ABSTAIN', 'DONOR_MATCH', 'OTHER_IN_CONTEXT', 'CONTEXT_ABSENT', 'INVALID'):
                outcomes.append({'variant': name, 'placement': placement, 'category': category,
                                 'count': counts[category], 'n': len(subset), 'rate': counts[category]/len(subset)})
    csv_file(ROOT / 'results/abstention_notation_sensitivity.csv', contrasts)
    csv_file(ROOT / 'results/abstention_notation_outcomes.csv', outcomes)
    missing = [r for r in rows if r['family'] == 'MISSING']
    absent = [r for r in missing if r['category'] == 'CONTEXT_ABSENT']
    detail = {'status': 'POST_HOC_DIAGNOSTIC', 'primary_changed': False,
              'quoted_null_primary_n': sum(r['parse']['valid'] and r['parse']['normalized_value']=='null' for r in missing),
              'other_in_context_primary_n': sum(r['category']=='OTHER_IN_CONTEXT' for r in missing),
              'context_absent_primary': [{'request_id':r['request_id'],'raw_output':r['raw_output'],'placement':r['placement']} for r in absent],
              'checks': checks, 'quoted_null_reclassification_preserves_all_primary_base_effects': True,
              'ci_scope': 'Same 2,000 base-cluster bootstrap resamples as primary; post-hoc only.'}
    write_json(ROOT / 'results/abstention_notation_audit.json', detail)

    # A companion to the strict-outcome plot, clearly marked as a sensitivity view.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cats = ('ABSTAIN','DONOR_MATCH','OTHER_IN_CONTEXT','CONTEXT_ABSENT','INVALID')
    fig, ax = plt.subplots(figsize=(7,4.5), constrained_layout=True)
    bottom = np.zeros(2)
    for cat,color in zip(cats, ['#7d9b76','#db7949','#baaa80','#826ba1','#a5acb0']):
        vals = [next(r['rate'] for r in outcomes if r['variant']=='quoted_null_as_abstain' and r['placement']==p and r['category']==cat)*100 for p in ('MIDDLE','ADJACENT')]
        ax.bar([0,1],vals,bottom=bottom,width=.58,color=color,label=cat.replace('_',' '))
        bottom += vals
    ax.set(xticks=[0,1],xticklabels=['Middle','Adjacent'],ylabel='All planned requests (%)',ylim=(0,100),
           title='Post-hoc sensitivity: quoted "null" counted as abstention')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=8)
    fig.savefig(ROOT/'figures/abstention_notation_sensitivity.png',dpi=200)
    fig.savefig(ROOT/'figures/abstention_notation_sensitivity.pdf'); plt.close(fig)

    path = ROOT/'final_report_ko.md'
    original = ROOT/'audit/primary_report_before_null_diagnostic.md'
    if not original.exists():
        original.write_text(path.read_text())
    text = path.read_text().split('\n## 사후 진단: 기권 표기의 영향')[0]
    text = text.replace('개발 792회와 본실험', '동일한 독립 개발 12개 base를 표기 수정 전후 각 792회(총 1,584회), 본실험')
    note = ('**표기 해석 주의:** 아래 `OTHER_IN_CONTEXT` 1,571건 중 1,570건은 실제 내용값 복사가 아니라 문자열 `"null"`이다. '
            'JSON literal `null`과 구분한 사전 규칙 때문에 이 범주에 들어갔다. 표기 민감도와 해석은 마지막 사후 진단에 별도로 제시한다.\n\n')
    anchor = '## Primary 조건의 출력 유형\n\n'
    if note not in text:
        text = text.replace(anchor, anchor + note)
    # Rates are percentages, unlike differences expressed in percentage points.
    text = text.replace('전환율은 +0.87%p [+0.26, +1.65]', '전환율은 0.87% [0.26, 1.65]')
    text = text.replace('반대 전환율은 +0.00%p [+0.00, +0.00]', '반대 전환율은 0.00% [0.00, 0.00]')
    extra = ['','## 사후 진단: 기권 표기의 영향','',
             '원출력 점검에서 primary의 OTHER_IN_CONTEXT 1,571건 중 1,570건이 문자열 `"null"`이고, 나머지 1건만 조직명이었다. '
             '모든 입력의 system 지시문에 null이 있으므로 사전 분류에서 문자열 null이 문맥 내 값에 해당한다. '
             '따라서 이 범주의 큰 비율을 donor 외 내용의 대규모 복사나 일반적인 사실 환각으로 설명하면 안 된다.', '',
             '사전 엄격 분류는 유지했다. 별도의 사후 분석에서 문자열 null을 기권으로 바꾸면 다음과 같다.', '',
             '| 배치 | Literal null | 문자열 null | 둘을 합친 기권 | Donor-copy |',
             '|---|---:|---:|---:|---:|',
             '| Middle | 100/1152 | 948/1152 | 1048/1152 (90.97%) | 102/1152 (8.85%) |',
             '| Adjacent | 243/1152 | 622/1152 | 865/1152 (75.09%) | 281/1152 (24.39%) |', '',
             '**이 재분류에서는 모든 base의 delta_attr, delta_C, delta_U, delta_type가 정확히 같았다.** '
             '따라서 H1/H2의 점추정치·신뢰구간·판정은 변하지 않는다. 반면 기권율 해석은 달라진다. '
             'Strict literal-null 비율만 보면 Adjacent에서 기권이 늘지만, 문자열 null까지 포함하면 기권은 줄어든다.', '',
             'Context-absent 7건 중 6건은 숫자이고 1건은 `N/A`다. 이를 포함해 추가로 기권 별칭 처리한 분석 역시 사후 민감도로만 제시한다.', '',
             '| 처리 | delta_attr | delta_C | delta_U | delta_type |', '|---|---:|---:|---:|---:|']
    for name,_ in variants:
        vals=[next(r for r in contrasts if r['variant']==name and r['metric']==m) for m in ('delta_attr','delta_C','delta_U','delta_type')]
        extra.append('| '+name+' | '+' | '.join(f"{r['estimate']*100:+.2f} [{r['ci95_low']*100:+.2f}, {r['ci95_high']*100:+.2f}]" for r in vals)+' |')
    extra += ['', '표의 단위는 %p다. N/A 민감도는 탐색적이며 주 분석을 대체하지 않는다. 문맥 외 실제 숫자는 적으므로 '
              'H2를 일반적인 환각 메커니즘 차이로 확대하지 않는다. 완전한 양방향 Salary-following 전환은 10/1152쌍(0.87%)으로 드물고, '
              'H1의 16.36%p는 평균 선택률 대비이다.', '',
              'Template별로 길이를 맞추기 위한 배경 발췌도 달라진다. Template 부분집합 차이를 순수한 문구 변화의 인과효과로 해석하지 않는다. '
              'PROSE 단독의 delta_C/delta_type CI 하한은 0이므로 전체 지지를 모든 template에서의 유의한 지지로 표현하지 않는다.', '',
              '관련 파일: `results/abstention_notation_audit.json`, `results/abstention_notation_sensitivity.csv`, '
              '`results/abstention_notation_outcomes.csv`, `figures/abstention_notation_sensitivity.png`.', '']
    path.write_text(text.rstrip()+'\n'+'\n'.join(extra))
    # Add the diagnostic provenance without altering the frozen protocol or primary results.
    manifest_path=ROOT/'completion_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['posthoc_diagnostic']='Quoted null and N/A notation; registered primary outcomes unchanged.'
    tracked=[path,original,ROOT/'posthoc_null_diagnostic.py',ROOT/'reproduce.md',ROOT/'ai_usage_log.md']
    tracked += list((ROOT/'results').glob('abstention_notation*')) + list((ROOT/'figures').glob('abstention_notation*'))
    for p in tracked:
        manifest['artifacts'][str(p.relative_to(ROOT))]=sha(p)
    write_json(manifest_path,manifest)
    print(json.dumps(detail,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
