# HW1 기반 누락 필드·속성 교환·길이 통제 위치 비교 — 실행 보고서

이 문서는 실제 실행 결과와 해석을 보존하는 연구 기록이다. 학생 제출용 완성 원고가 아니다.

## 실행과 판정

새 base 96개를 각각 세 template로 표현하고, 동일한 독립 개발 12개 base를 표기 수정 전후 각 792회(총 1,584회), 본실험 6,336회를 실제 실행했다. 본실험 중 H1/H2의 primary factorial은 2,304회다.
사전 기준에 따른 H1은 **지지**, H2는 **지지**이다.

| 지표 | 추정치 및 95% paired base-bootstrap CI |
|---|---:|
| delta_attr | +16.36%p [+14.02, +18.79] |
| delta_C | +15.54%p [+12.93, +18.32] |
| delta_U | +0.43%p [+0.00, +0.95] |
| delta_type | +15.10%p [+12.41, +17.97] |

H1은 delta_attr의 CI 하한이 0보다 커야 한다. H2는 delta_C와 delta_type의 CI 하한이 **모두** 0보다 커야 한다. CI가 0을 포함하면 예측 방향이 확립되지 않은 것이며, 전 구간 음수이면 해당 방향에 반하는 근거다.

## 제출한 HW1과 구현의 대응

- Target 문서는 항상 처음, 질문은 마지막이다. Middle의 donor block 중점은 전체 chat-template 입력의 45–55%에 있고 Adjacent는 질문 바로 앞에 있다.
- 같은 문서·중립 prefix·중립 suffix를 재배열했으며 placement pair와 label-swap pair의 토큰 수를 정확히 맞췄다.
- 같은 96개 base에 세 template를 모두 적용했다. 표본 수는 2,304가 아니라 96개 독립 base다.
- 문자열/null JSON을 비제약 greedy로 생성했다. Invalid도 전체 분모에 포함했다. 정수/null grammar 또는 donor enum은 사용하지 않았다.
- 모델 revision은 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, FP16, SDPA, repetition_penalty 1.1이다.
- 계획의 Colab T4 대신 로컬 RTX A6000 한 장을 사용했다. 모델·질문·가설은 동일 연구 범위이며 실제 하드웨어 차이를 공개한다.
- 중립 문단은 [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11)의 영어 소설 발췌다. 숫자·조작 속성어·개체명 오염을 검사했고 출처·원문·paragraph ID·hash를 보존했다.

## Primary 조건의 출력 유형

**표기 해석 주의:** 아래 `OTHER_IN_CONTEXT` 1,571건 중 1,570건은 실제 내용값 복사가 아니라 문자열 `"null"`이다. JSON literal `null`과 구분한 사전 규칙 때문에 이 범주에 들어갔다. 표기 민감도와 해석은 마지막 사후 진단에 별도로 제시한다.

| 배치 | 기권 | Donor 일치 | 다른 문맥 내 값 | 문맥 외 값 | Invalid |
|---|---:|---:|---:|---:|---:|
| MIDDLE | 100/1152 (8.68%) | 102/1152 (8.85%) | 949/1152 (82.38%) | 1/1152 (0.09%) | 0/1152 (0.00%) |
| ADJACENT | 243/1152 (21.09%) | 281/1152 (24.39%) | 622/1152 (53.99%) | 6/1152 (0.52%) | 0/1152 (0.00%) |

Donor 일치는 질문 속성과 연결된 후보뿐 아니라 두 donor 후보 중 어느 쪽이든 정확히 일치한 출력이다. Context-absent는 donor가 아니라는 이유만으로 판정하지 않고 정규화한 전체 입력에서 부재를 확인했다.
완전한 S1=v1→S2=v2 전환율은 0.87% [0.26, 1.65], 반대 전환율은 0.00% [0.00, 0.00]다. 이는 delta_attr와 다른 지표다.

## 대조군

| 조건 | 정답 / 전체 | 정확도 및 95% CI |
|---|---:|---:|
| T0 | 288/288 | 100.00% [100.00, 100.00] |
| T1 | 288/288 | 100.00% [100.00, 100.00] |
| PRESENT | 2275/2304 | 98.74% [97.70, 99.52] |
| N0 | 1139/1152 | 98.87% [98.09, 99.57] |

T0는 donor 없는 missing target의 기권, T1은 donor 없는 정답 추출, PRESENT는 두 donor가 있는 상황의 target 정답 추출, N0는 질문 속성이 없는 donor 두 개를 둔 기권이다. Target-present의 정답은 context-absent나 false-fill로 세지 않았다.
기본 대조군이 낮으면 donor 오염만으로 전체 실패를 설명할 수 없다. 이런 실패를 이유로 사례를 제외하거나 프롬프트를 변경하지 않았다.

## Template·donor order 민감도

| 부분집합 | delta_attr | delta_C | delta_type |
|---|---:|---:|---:|
| template=KV | +15.36 [+12.63, +18.23] | +23.96 [+20.57, +27.86] | +23.44 [+19.79, +27.34] |
| template=PROSE | +11.85 [+9.11, +14.71] | +3.91 [+0.00, +7.81] | +3.91 [+0.00, +7.81] |
| template=BULLET | +21.88 [+18.88, +25.13] | +18.75 [+14.58, +23.18] | +17.97 [+13.54, +22.66] |
| order=AB | +15.45 [+12.59, +18.40] | +14.58 [+11.63, +17.88] | +14.06 [+10.94, +17.36] |
| order=BA | +17.27 [+14.24, +20.40] | +16.49 [+13.02, +20.31] | +16.15 [+12.50, +20.31] |
| without_template=KV | +16.86 [+14.32, +19.47] | +11.33 [+8.20, +14.58] | +10.94 [+7.81, +14.32] |
| without_template=PROSE | +18.62 [+16.08, +21.35] | +21.35 [+18.23, +24.61] | +20.70 [+17.32, +24.09] |
| without_template=BULLET | +13.61 [+11.26, +16.08] | +13.93 [+11.19, +16.93] | +13.67 [+10.81, +16.67] |

단위는 %p다. 부분집합/leave-one-template-out은 사전 지정 보조 분석이며 CI는 개별 구간이다. 세 template를 모든 가능한 문서 양식의 무작위 표본으로 간주하지 않는다.

## 품질·통계·재현성

- Strict-valid 6335/6336, truncation 0, 시스템 오류 0, raw 재파싱 불일치 0.
- Invalid 유형: `{"malformed_json": 1}`.
- Bootstrap 2,000회, seed 2026103103. 모든 대비에 같은 base index matrix를 사용했다. base별 모든 template·조건을 함께 재표집했다.
- Primary context-absent 관측은 7건이다. 20건 미만이므로 희소 사건으로 해석에 주의해야 한다.
- 0건의 bootstrap [0,0]은 모집단 발생률 0의 증명이 아니다. 96개 독립 base에서 한 번도 발생하지 않은 사건의 base당 any-event 확률에 대한 단측 95% 상한은 약 3.07%다. 요청 단위 상한과는 다르다.
- 본실험 순수 generation 611.2초, peak allocated/reserved 5.87/9.97 GiB.
- 첫 v1 개발 792회에서 모든 JSON은 유효했지만, 정답에 USD 단어를 붙여 복사해 T1 엄격 정확도가 0/36이었다. 숫자 선택과 단위 표기를 분리하기 위해 문서 표기만 USD 123456→$123456으로 한 번 수정했다. 기존 기호 정규화는 유지했다. 원래 데이터·코드·lock·출력은 audit/protocol_v1_before_currency_format_fix.zip과 이전 run 폴더에 보존했다.
- V2는 본실험 출력 확인 전에 별도로 SHA-256 잠금했고 개발 재실행 및 main 이후에도 일치했다. 가설·후보 값·개체·생성 설정·평가식은 바꾸지 않았다. 개발 표기를 한 번 수정한 사실을 사전 무수정 실행처럼 표현하지 않는다.

## 대표 원출력

각 primary 오류 유형에서 request ID가 가장 앞선 사례를 기계적으로 선택했다. 전체 입력은 results/representative_cases.json에 있다.

- `HW2M000_PROSE_MISSING_S2_AB_ADJACENT` / ABSTAIN: `{"value": null}`
- `HW2M000_BULLET_MISSING_S1_AB_ADJACENT` / DONOR_MATCH: `{"value": "104782"}`
- `HW2M000_BULLET_MISSING_S1_AB_MIDDLE` / OTHER_IN_CONTEXT: `{"value": "null"}`
- `HW2M014_BULLET_MISSING_S2_BA_ADJACENT` / CONTEXT_ABSENT: `{"value": "23456"}`

## 해석 범위와 HW2 작성

이 연구는 unsupported target의 출력 선택이 속성 연결 및 길이가 통제된 배치에 민감한지를 검정한다. 내부 binding 회로, attention head, 또는 모든 모델·속성·실문서에 대한 일반화는 검정하지 않았다.
위치 조작은 donor 절대 위치와 질문까지의 거리를 함께 바꾼다. 전체 길이를 고정해도 두 요인을 독립적으로 식별하는 설계는 아니다. 합성 영어 salary 한 속성, 단일 모델, 단일 길이, 단일 공개 소설 배경과 세 template에 한정된다.
이전 position_balanced_donor의 target-last·길이 변화·제약 생성과 직접 같은 실험으로 취급하지 않는다. 기존 결과와 차이가 나면 이 여러 설계 차이 중 어느 하나의 인과효과로 단정할 수 없다.
HW2 본문은 두 primary 가설의 결과, 대조군, 부분집합 분석과 오류 사례를 중심으로 작성한다. 다음 단계의 출력 제약 강건성 및 속성/모델 일반화는 이번 실행 범위에 포함되지 않았다.

## 산출물

- results/contrasts.csv, outcome_rates.csv, condition_counts.csv, base_effects.csv
- figures/outcome_rates.png 및 .pdf, hypothesis_contrasts.png 및 .pdf
- data/*_requests.jsonl, runs/run_20261031_hw1_main_v2/*_raw.jsonl
- source/source_manifest.json, data/data_audit.json, results/bootstrap_manifest.json
- reproduce.md, ai_usage_log.md, completion_manifest.json

## 사후 진단: 기권 표기의 영향

원출력 점검에서 primary의 OTHER_IN_CONTEXT 1,571건 중 1,570건이 문자열 `"null"`이고, 나머지 1건만 조직명이었다. 모든 입력의 system 지시문에 null이 있으므로 사전 분류에서 문자열 null이 문맥 내 값에 해당한다. 따라서 이 범주의 큰 비율을 donor 외 내용의 대규모 복사나 일반적인 사실 환각으로 설명하면 안 된다.

사전 엄격 분류는 유지했다. 별도의 사후 분석에서 문자열 null을 기권으로 바꾸면 다음과 같다.

| 배치 | Literal null | 문자열 null | 둘을 합친 기권 | Donor-copy |
|---|---:|---:|---:|---:|
| Middle | 100/1152 | 948/1152 | 1048/1152 (90.97%) | 102/1152 (8.85%) |
| Adjacent | 243/1152 | 622/1152 | 865/1152 (75.09%) | 281/1152 (24.39%) |

**이 재분류에서는 모든 base의 delta_attr, delta_C, delta_U, delta_type가 정확히 같았다.** 따라서 H1/H2의 점추정치·신뢰구간·판정은 변하지 않는다. 반면 기권율 해석은 달라진다. Strict literal-null 비율만 보면 Adjacent에서 기권이 늘지만, 문자열 null까지 포함하면 기권은 줄어든다.

Context-absent 7건 중 6건은 숫자이고 1건은 `N/A`다. 이를 포함해 추가로 기권 별칭 처리한 분석 역시 사후 민감도로만 제시한다.

| 처리 | delta_attr | delta_C | delta_U | delta_type |
|---|---:|---:|---:|---:|
| strict_primary | +16.36 [+14.02, +18.79] | +15.54 [+12.93, +18.32] | +0.43 [+0.00, +0.95] | +15.10 [+12.41, +17.97] |
| quoted_null_as_abstain | +16.36 [+14.02, +18.79] | +15.54 [+12.93, +18.32] | +0.43 [+0.00, +0.95] | +15.10 [+12.41, +17.97] |
| quoted_null_and_na_as_abstain | +16.36 [+14.02, +18.79] | +15.54 [+12.93, +18.32] | +0.52 [+0.09, +1.04] | +15.02 [+12.33, +17.80] |

표의 단위는 %p다. N/A 민감도는 탐색적이며 주 분석을 대체하지 않는다. 문맥 외 실제 숫자는 적으므로 H2를 일반적인 환각 메커니즘 차이로 확대하지 않는다. 완전한 양방향 Salary-following 전환은 10/1152쌍(0.87%)으로 드물고, H1의 16.36%p는 평균 선택률 대비이다.

Template별로 길이를 맞추기 위한 배경 발췌도 달라진다. Template 부분집합 차이를 순수한 문구 변화의 인과효과로 해석하지 않는다. PROSE 단독의 delta_C/delta_type CI 하한은 0이므로 전체 지지를 모든 template에서의 유의한 지지로 표현하지 않는다.

관련 파일: `results/abstention_notation_audit.json`, `results/abstention_notation_sensitivity.csv`, `results/abstention_notation_outcomes.csv`, `figures/abstention_notation_sensitivity.png`.
