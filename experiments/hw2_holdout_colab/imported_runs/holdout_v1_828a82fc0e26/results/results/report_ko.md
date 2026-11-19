# HW2 Target-only 비교와 고정 holdout 평가

개발 480개 입력과 새 holdout 1,152개 입력을 별도로 평가했다. Holdout은 독립 family 16개, task당 8개다.
각 family의 두 표현·세 문서 형식·두 placement·여섯 조건은 대응 관측이며 독립 표본 수를 늘리지 않는다.
모든 방법과 prompt·parser 사전·평가 정책은 모델 실행 전에 protocol_lock.json으로 고정했다. 추가 학습 없음.

주 비교는 ID + constrained gate와 Target-only-routed 숫자 출력 + 동일 gate다. 둘 다 target에 숫자가 없으면 CPU null로 종료하고, 후보를 선택할 때의 prompt/출력 인터페이스가 다르다. 검증기는 같다. CPU 생략을 하지 않은 일반 Target-only+gate도 별도 보고한다.
아래는 invalid를 오류로 유지한 decision/semantic 평가다. Strict와 fail-closed service 결과는 metrics.csv에 별도로 있다.

## development

| 방법 | Salary 정답 | Year 정답 |
|---|---:|---:|
| standard | 190/240 | 128/240 |
| target_only | 216/240 | 166/240 |
| target_only_presence | 216/240 | 196/240 |
| id_selection | 240/240 | 206/240 |
| standard_gate_constrained | 240/240 | 236/240 |
| target_only_gate_constrained | 240/240 | 236/240 |
| id_gate_constrained | 240/240 | 236/240 |
| target_only_routed_gate_constrained | 240/240 | 236/240 |
| field_parser | 240/240 | 240/240 |
| alias_field_parser | 240/240 | 240/240 |
| always_abstain | 120/240 | 120/240 |

| 방법 | 캐시 반영 고유 모델 요청 | 입력당 논리적 생성 시간 평균(초) |
|---|---:|---:|
| target_only | 240 | 0.3292 |
| id_selection | 168 | 0.0861 |
| target_only_gate_constrained | 409 | 0.4001 |
| target_only_routed_gate_constrained | 322 | 0.3268 |
| id_gate_constrained | 305 | 0.1437 |
| standard_gate_constrained | 692 | 0.8280 |
| alias_field_parser | 0 | 0.0000 |

## holdout_canonical

| 방법 | Salary 정답 | Year 정답 |
|---|---:|---:|
| standard | 200/288 | 157/288 |
| target_only | 240/288 | 188/288 |
| target_only_presence | 240/288 | 208/288 |
| id_selection | 288/288 | 226/288 |
| standard_gate_constrained | 287/288 | 286/288 |
| target_only_gate_constrained | 288/288 | 286/288 |
| id_gate_constrained | 288/288 | 286/288 |
| target_only_routed_gate_constrained | 288/288 | 286/288 |
| field_parser | 288/288 | 288/288 |
| alias_field_parser | 288/288 | 288/288 |
| always_abstain | 144/288 | 144/288 |

| 방법 | 캐시 반영 고유 모델 요청 | 입력당 논리적 생성 시간 평균(초) |
|---|---:|---:|
| target_only | 288 | 0.3407 |
| id_selection | 240 | 0.0970 |
| target_only_gate_constrained | 506 | 0.4195 |
| target_only_routed_gate_constrained | 448 | 0.3779 |
| id_gate_constrained | 415 | 0.1598 |
| standard_gate_constrained | 839 | 0.8292 |
| alias_field_parser | 0 | 0.0000 |

## holdout_paraphrase

| 방법 | Salary 정답 | Year 정답 |
|---|---:|---:|
| standard | 196/288 | 181/288 |
| target_only | 288/288 | 234/288 |
| target_only_presence | 288/288 | 250/288 |
| id_selection | 288/288 | 260/288 |
| standard_gate_constrained | 287/288 | 286/288 |
| target_only_gate_constrained | 288/288 | 286/288 |
| id_gate_constrained | 288/288 | 286/288 |
| target_only_routed_gate_constrained | 288/288 | 286/288 |
| field_parser | 144/288 | 144/288 |
| alias_field_parser | 288/288 | 288/288 |
| always_abstain | 144/288 | 144/288 |

| 방법 | 캐시 반영 고유 모델 요청 | 입력당 논리적 생성 시간 평균(초) |
|---|---:|---:|
| target_only | 288 | 0.3108 |
| id_selection | 240 | 0.0950 |
| target_only_gate_constrained | 459 | 0.3731 |
| target_only_routed_gate_constrained | 403 | 0.3333 |
| id_gate_constrained | 398 | 0.1526 |
| standard_gate_constrained | 830 | 0.8101 |
| alias_field_parser | 0 | 0.0000 |

## 해석 기준

paired_contrasts.csv의 ID gate − target-only-routed gate를 주 비교로 읽고, 일반 target-only gate와의 차이도 함께 확인한다. 정확도·PRESENT 손실·MISSING false-fill을 task와 표현별로 본다. 차이가 0인 관측을 통계적 동등성이나 비열등성의 증명으로 읽지 않는다.
paired_style_shift.csv는 동일 family에서 paraphrase − canonical의 정확도 차이를 직접 bootstrap했다. 독립 표본 CI를 빼서 구한 구간이 아니다.
Literal parser와 더 강한 alias_field_parser를 함께 보고한다. 후자는 생성에 사용한 모든 긍정 속성 표현을 아는 schema-aware baseline이다. 이를 빼고 표현 변형에서 literal parser를 이겼다는 주장만 하지 않는다.
모델 생성 비용은 캐시 반영 unique와 입력마다 같은 요청을 다시 센 logical을 모두 저장했다. 과거/새 생성 시간을 구분하며 CPU·모델 로드·종단 latency는 포함하지 않는다. 서로 겹치는 방법·split 비용을 합산하지 않는다.
0건 CI를 위험 0으로 해석하지 않는다. Zero-event CSV는 family에 사건이 하나라도 있을 위험의 상한이다.
이 평가는 같은 합성 생성기의 새 개체·값 조합과 미리 선언한 문구 변화에 대한 검증이다. 실제 문서·다른 모델·새 알고리즘의 신규성을 증명하지 않는다.
결과를 보고 prompt를 수정하면 이 holdout은 새 개발 자료가 된다. 보고서는 자동으로 최적 방법을 선택하거나 기존 규칙을 바꾸지 않는다.
