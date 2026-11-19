from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from .analyze import base_effects, estimate
from .common import ROOT, REPO, cfg, classify, delta_attr, normalize, occurs, parse, read_jsonl, run_dir, sha, write_json


def checks():
    # Invalid outputs must not become abstentions or numeric strings by coercion.
    valid = ['{"value": null}', '{"value":"123456"}', '{"value":"123,456"}', '{"value":"null"}']
    invalid = ['{"value":123456}', '{"value":true}', '{"value":""}', '{"value":"..."}',
               '{"value":"1","value":"2"}', 'null', '```json\n{"value":null}\n```',
               '{"value":null,"why":"missing"}', '{"value":NaN}', 'answer: {"value":null}']
    assert all(parse(r)["valid"] for r in valid)
    assert not any(parse(r)["valid"] for r in invalid)
    assert parse('{"value":"null"}')["normalized_value"] == "null"
    assert normalize('$123,456') == '123456'
    assert occurs('123456', 'Amount: USD 123,456.')
    assert not occurs('123456', 'Amount: 91234567.')
    request = {"family": "MISSING", "state": "MISSING", "v1": 123456, "v2": 654321,
               "target_value": 777777, "messages": [{"content": "salary 123456, budget 654321, note 987654. null"}]}
    expected = {'{"value":null}': 'ABSTAIN', '{"value":"123456"}': 'DONOR_MATCH',
                '{"value":"654321"}': 'DONOR_MATCH', '{"value":"987654"}': 'OTHER_IN_CONTEXT',
                '{"value":"333333"}': 'CONTEXT_ABSENT', '{"value":123456}': 'INVALID',
                '{"value":"null"}': 'OTHER_IN_CONTEXT'}
    for raw, category in expected.items():
        assert classify(request, parse(raw)) == category
    present = {**request, "family": "PRESENT", "state": "PRESENT"}
    assert classify(present, parse('{"value":"777777"}')) == 'TARGET_CORRECT'
    assert delta_attr(parse('{"value":"123456"}'), parse('{"value":"654321"}'), 123456, 654321) == 1
    assert delta_attr(parse('{"value":"654321"}'), parse('{"value":"123456"}'), 123456, 654321) == -1
    assert delta_attr(parse('{"value":123456}'), parse('{"value":null}'), 123456, 654321) == 0
    # A known factorial: Middle abstains; Adjacent follows labels. Thus H1=.5,
    # delta_C=1, delta_U=0, delta_type=1; each base retains every template.
    fake = []
    for bid in ('b0', 'b1'):
        for template in ('KV', 'PROSE', 'BULLET'):
            for order in ('AB', 'BA'):
                for placement in ('MIDDLE', 'ADJACENT'):
                    for assignment in ('S1', 'S2'):
                        value = None if placement == 'MIDDLE' else str(123456 if assignment == 'S1' else 654321)
                        parsed = parse(json.dumps({'value': value}))
                        row = {**request, 'base_id': bid, 'template': template, 'order': order,
                               'placement': placement, 'assignment': assignment, 'parse': parsed}
                        row['category'] = classify(request, parsed)
                        fake.append(row)
    vals = base_effects(fake)
    for v in vals:
        assert v['delta_attr'] == .5 and v['delta_C'] == 1 and v['delta_U'] == 0 and v['delta_type'] == 1
    assert estimate([1., 1.], np.array([[0, 0], [1, 1], [0, 1]])) == {'estimate': 1., 'ci95_low': 1., 'ci95_high': 1.}
    # Higher context-absent rates can overturn H2 even if donor copying increases.
    for row in fake:
        if row['placement'] == 'ADJACENT' and row['template'] != 'KV':
            row['parse'] = parse('{"value":"333333"}')
            row['category'] = 'CONTEXT_ABSENT'
    for v in base_effects(fake):
        assert np.isclose(v['delta_C'], 1/3) and np.isclose(v['delta_type'], -1/3)
    # Independent structural checks on the generated requests and corpus excerpts.
    pool = {p['paragraph_id']: p['text'] for p in read_jsonl(ROOT / 'source/filtered_paragraphs.jsonl')}
    for bg in read_jsonl(ROOT / 'data/backgrounds.jsonl'):
        for key in ('prefix', 'suffix', 'solo_suffix'):
            for part in bg[key]:
                assert pool[part['paragraph_id']].startswith(part['text'])
    for split in ('dev', 'main'):
        rs = read_jsonl(ROOT / f'data/{split}_requests.jsonl')
        assert len(rs) == cfg()['data'][split + '_calls']
        for row in rs:
            user = row['messages'][1]['content']
            first_document = user.split('<document>')[1].split('</document>')[0]
            assert row['target_name'] in first_document
            assert (str(row['target_value']) in first_document) == (row['state'] == 'PRESENT')
            assert user.endswith(f"Question: What is the recorded salary for {row['target_name']}?")
            for donor, field in [('donor1', 'v1'), ('donor2', 'v2')]:
                if row['family'] in ('T0', 'T1'):
                    assert row[donor + '_name'] not in user
                else:
                    doc = next(s.split('</document>')[0] for s in user.split('<document>')[1:] if row[donor + '_name'] in s.split('</document>')[0])
                    expected_label = ({'S1': ('salary', 'budget'), 'S2': ('budget', 'salary'), 'N0': ('budget', 'revenue')}[row['assignment']][0 if donor == 'donor1' else 1])
                    assert expected_label in doc.lower() and str(row[field]) in doc
        counts = Counter((r['base_id'], r['template']) for r in rs)
        assert set(counts.values()) == {22}
    archive = ROOT / 'audit/protocol_v1_before_currency_format_fix.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            old_config = json.loads(z.read('config.json'))
            new_config = cfg()
            assert {k:v for k,v in old_config.items() if k != 'run_id'} == {k:v for k,v in new_config.items() if k != 'run_id'}
            for split in ('dev', 'main'):
                old_bases = [json.loads(line) for line in z.read(f'data/{split}_bases.jsonl').decode().splitlines()]
                assert old_bases == read_jsonl(ROOT / f'data/{split}_bases.jsonl')
                old_requests = {r['request_id']:r for r in (json.loads(line) for line in z.read(f'data/{split}_requests.jsonl').decode().splitlines())}
                for row in read_jsonl(ROOT / f'data/{split}_requests.jsonl'):
                    old = old_requests[row['request_id']]
                    old_docs = re.findall(r'<document>.*?</document>', old['messages'][1]['content'], flags=re.S)
                    new_docs = re.findall(r'<document>.*?</document>', row['messages'][1]['content'], flags=re.S)
                    assert [re.sub(r'USD (\d+)', r'$\1', doc) for doc in old_docs] == new_docs
                    assert old['messages'][0] == row['messages'][0]
    return {'pass': True, 'parser_cases': len(valid) + len(invalid),
            'taxonomy_cases': len(expected) + 1, 'known_factorial_positive_and_negative_checks': True,
            'source_provenance_and_target_donor_mapping_checks': True,
            'v1_v2_only_document_currency_notation_changed': archive.exists()}


def main():
    if (run_dir() / 'protocol_lock.json').exists():
        raise RuntimeError('Lock already exists; refuse replacement')
    if json.loads((ROOT / 'data/data_audit.json').read_text())['status'] != 'PASS':
        raise RuntimeError('Dataset audit failure')
    result = checks(); write_json(ROOT / 'data/static_checks.json', result)
    write_json(ROOT / 'source/design_references.json', {
        name: sha(REPO / name) for name in ('22_When_Missing_Fields_Borrow_.pdf', '2026_2_NLP_HW_Guidelines.pdf')})
    files = [ROOT / 'config.json', ROOT / 'experiment_spec.md', ROOT / 'requirements_frozen.txt']
    files += sorted((ROOT / 'scripts').glob('*.py'))
    for directory in ('source', 'data'):
        files += sorted(p for p in (ROOT / directory).iterdir() if p.is_file())
    files += sorted((ROOT / 'audit').glob('*.zip'))
    lock = {'status': 'LOCKED', 'locked_utc': datetime.now(timezone.utc).isoformat(),
            'before_any_development_or_main_model_output': True,
            'hashes': {str(p.relative_to(ROOT)): sha(p) for p in files},
            'hypotheses': cfg()['analysis'], 'model': cfg()['model'], 'generation': cfg()['generation']}
    write_json(run_dir() / 'protocol_lock.json', lock)
    print(json.dumps({'checks': result, 'locked_files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
