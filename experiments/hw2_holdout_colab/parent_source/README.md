# HW2 관계 검증 보완 실험

`hw2_relation_gate_colab.ipynb`를 **웹 Colab에 업로드 → T4 GPU → Run All**로 실행한다. 코드·기존 원출력·자료가 모두 내장되어 ZIP을 다시 업로드할 필요가 없다. 모델은 Qwen2.5-1.5B-Instruct 하나이며 추가 학습은 없다. 기존 pointer 노트북과 결과는 수정하지 않는다.

기본값 `REUSE_PARENT=True`는 원래 480개 개발 입력의 검증된 원출력을 재사용한다. 관계 검증의 두 후보 표현을 숫자로 통일하고, 같은 prompt는 공유하여 GPU 호출을 줄인다. `call_plan.json`에 정확한 추가 요청 수를 먼저 표시한다. 과거 실행과 Python/Torch/CUDA/GPU/Transformers/tokenizers/chat template가 다르면 모델 로드 전에 중단한다. 이때 설정 셀의 `REUSE_PARENT=False`로 바꾸고 Run All 하면 별도 `_fresh` 폴더에서 비교군까지 새로 실행한다. 서로 다른 환경의 출력을 합치지 않는다. Fresh 모드의 호출 수는 후보가 달라질 수 있어 사전 추정이며 최대 4,000회다.

Drive를 쓰려면 설정 셀의 두 mount 줄을 해제한다. Drive가 연결되지 않았으면 `/content/hw2_relation_gate`에 저장하므로 완료 후 마지막 ZIP을 반드시 다운로드한다. 중간에 끊기면 같은 코드·설정·환경과 RUN_DIR에서 Run All 한다. 체크포인트가 있는 요청은 다시 생성하지 않는다.

## 고정한 비교

| 파이프라인 | 후보 선택 | 검증 |
|---|---|---|
| Standard | 전체 문맥에서 JSON 숫자/null | 없음 |
| ID selection | target 문서 숫자의 ID 또는 NONE | 없음 |
| Standard + gate (free/constrained) | Standard와 동일 | target–질문 속성–후보 관계 검증 |
| ID + gate (free/constrained) | ID selection과 동일 | 위와 동일한 검증기 |

검증의 prompt와 두 라벨 `SUPPORTED` / `UNSUPPORTED`는 같다. Free는 원래 자유 생성이며 `SUPPORTING`은 invalid다. Constrained는 `prefix_allowed_tokens_fn`으로 매 단계 허용 토큰을 제한하며 라벨 뒤 EOS까지 강제한다. 사후 라벨 치환이 아니다. 후보가 valid null/NONE이면 그대로 반환하고 검증을 호출하지 않는다. 두 검증기의 입력 후보는 모두 통화 기호 없는 정수 문자열이다. 승인되면 원래 후보 답변을 유지한다.

Presence, target-only + presence, any-change, presence→any-change, 명시적 field parser, always-null도 유지한다. 두 새 검증 방법의 차이는 같은 검증기를 사용하는 조건에서 평가한다. Parser가 강한 이 합성 자료에서 ID 방식의 독립적인 이득이 없을 가능성도 그대로 보고한다.

별도 진단은 두 과제의 모든 KV numeric-missing 16개 base와 각각의 PRESENT 대응 자료를 사용한다. 32입력 × (숫자 앞 ID / 항목 앞 ID / 숫자 앞 ID+항목 대시) = 96요청이다. 세 형식의 지시문은 동일하다. Other-field의 PRESENT는 같은 숫자의 라벨만 요청 속성으로 바꿔 만든다. Multi PRESENT는 기존 대응 입력을 쓴다. 이 진단은 본 평가 480개에 합산하지 않는다.

## 결과

`results/report_ko.md`, `metrics.csv`, `paired_contrasts.csv`, `zero_event_base_bounds.csv`, `diagnostic_metrics.csv`, `diagnostic_paired_contrasts.csv`, `id_gate_changed_cases.json`, `figures/*.png/pdf`를 만든다. 각 원출력의 메시지·token IDs·시간·판정·제약 경로를 저장하고 전체 판정을 재생해 검증한다. Invalid는 주 평가에서 오류로 유지하며 fail-closed service 평가는 별도로 제시한다. CI는 task별 base 단위 paired bootstrap 2,000회다. 형식/배치 복제본을 독립 표본으로 세지 않는다.

`method_costs.json`의 방법별 요청에는 과거 재사용 요청과 새 요청의 비용이 함께 있다. 공유 요청을 합산하면 안 된다. 실제 새 GPU 생성량은 `_current_physical_total`, 모델 로드·세션 시간·메모리는 `runtime_sessions.json`을 본다. 과거 전체 추론 시간을 새 실행 시간으로 보고하지 않는다.

최종 `relation_gate_dev_v1_results.zip`(fresh면 `relation_gate_dev_v1_fresh_results.zip`)에는 자료, 원출력, 분석, 재현 소스와 해시 manifest를 넣는다. VS Code로 다운로드하면 저장 경로를 `/home/cv/SSD0/jang/nlp/` 아래로 지정한다.

## 범위와 재현

이 방법은 과거 Year 오류를 확인한 뒤 설계한 **개발 실험**이다. 전체 480개를 독립 holdout으로 부르지 않는다. 긍정적 결과가 나와도 규칙을 고정한 뒤 새 개체·숫자와 속성 표현을 바꾼 holdout이 필요하다. Routing과 target-only의 효과를 ID 인터페이스만의 효과로 주장하지 않는다. 새 알고리즘의 신규성도 이 실험만으로 확정하지 않는다.

기존 ZIP SHA256: `b3bbf04043bf13b68eac370b7542bb5da610fd726633c8af1dbada02ea65e27f`. Snapshot에서 중복된 메시지만 제거했다. 준비 시 frozen upstream 코드로 정확한 메시지를 복원하고, 원래 1,622개 record hash·prompt key·token IDs·입력 길이와 480개 판정을 검증한다. 출처는 `source/provenance.json`에 있다. 배경 텍스트는 [Project Gutenberg의 Alice](https://www.gutenberg.org/ebooks/11)이다.

CPU 검증: `python -m unittest discover -s experiments/hw2_relation_gate_colab/tests -t .`. 노트북 생성: `python -m experiments.hw2_relation_gate_colab.build_notebook`. 모델 추론은 Colab에서만 허용한다. Constrained decoding은 [Transformers generate 문서](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)의 prefix callback을 사용한다.
