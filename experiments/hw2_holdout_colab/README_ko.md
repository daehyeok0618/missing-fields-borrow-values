# HW2 — Target-only ablation and frozen synthetic holdout

`hw2_holdout_colab.ipynb`를 **웹 Colab에 업로드 → T4 GPU → Run All**로 실행한다. 코드·기존 원출력을 내장했으므로 추가 자료 업로드는 없다. Qwen2.5-1.5B-Instruct 하나로 추론하며 학습은 없다. 기존의 실행 노트북과 ZIP은 수정하지 않는다.

## 확인할 두 질문

1. **같은 짧은 target 문맥에서 ID 선택이 직접 숫자 JSON 출력보다 기여하는가?** 기존 480개 개발 입력에 Target-only 숫자 출력 + 동일 constrained gate를 추가한다. 기존 target-only 원출력과 이미 실행한 gate 원출력을 재사용하면 새 검증은 21회다. 이 비교는 후보 선택 prompt/출력 인터페이스의 차이이며 순수한 ID 표식 하나의 효과는 아니다.
2. **고정한 방법이 새 입력과 정해진 문구 변화에서도 작동하는가?** 모델 출력을 보기 전에 새 entity/fact family 16개(task당 8개)를 만든다. 모든 family에 basic MISSING/PRESENT, collision PRESENT, other-field MISSING, multi PRESENT/MISSING을 넣는다. 6조건 × 3형식 × 2placement × 기존/변형 표현 × 16family = **1,152입력**이다. 개발 입력과 합쳐 하나의 성과로 보고하지 않는다.

방법은 Standard, Target-only JSON, Target-only+Presence, ID/NONE, 세 후보 선택기에 같은 constrained relation gate를 붙인 조합, 기존 literal parser, 고정 alias parser, always-null이다. Gate prompt·ID selector·JSON prompts·숫자 정규화·null/invalid 정책은 이전 실행과 동일하게 유지한다. Verifier는 `SUPPORTED`/`UNSUPPORTED` 뒤 EOS만 허용하는 실제 토큰 제약 생성이다.

**주 비교에는 `target_only_routed_gate_constrained`를 사용한다.** 이 방법과 ID+gate 모두 target에 숫자가 없으면 모델을 호출하지 않고 CPU null로 종료한다. 숫자가 있으면 일반 Target-only+gate와 같다. 일반 Target-only+gate도 함께 저장해 CPU 생략 효과를 구분한다. 이 추가 비교군은 같은 생성 결과를 공유하므로 물리적 요청을 늘리지 않는다. 비용 하네스는 여러 방법을 같이 실행하므로, routed 방법의 독립 비용에서는 생략할 요청도 다른 baseline을 위해 실제 생성할 수 있다.

## Holdout의 범위와 공정한 parser 비교

새 이름과 salary 값은 과거 inventory와 개발 입력에서 제외한다. Year는 유한한 범위라 개별 연도는 반복될 수 있지만 연도 쌍/세 값 조합은 과거 inventory·개발 입력과 겹치지 않게 생성한다. 조건·형식·placement·표현은 같은 family를 공유하므로 독립 표본으로 세지 않는다. 두 표현은 개체·관계·숫자·배경을 공유하고 속성 표현과 질문 문구가 바뀐다. 같은 표현 안의 두 placement는 길이와 질문 위치를 맞춘다. 표현 간에는 문구 길이 차이가 있다.

변형 표현과 사전은 `protocol.py`에서 사전 선언한다. 예: salary → annual pay/yearly compensation/annual remuneration, founding year → year of establishment/establishment year/year of founding. **Alias parser에는 생성에 사용한 모든 긍정 속성 표현을 제공한다.** 이는 schema-aware baseline이며 실제 범용 언어 이해 parser라는 주장은 하지 않는다. 단어가 다르다는 이유로 literal parser만 약해진 결과를 LLM 방법의 우월성으로 사용하지 않는다.

입력·정답·소스·프롬프트·제약 경로와 평가 정책은 `protocol_lock.json`으로 모델 로드 전에 잠근다. 결과를 보고 prompt를 수정하면 이 holdout은 다음 실험의 개발 자료가 된다. 같은 합성 생성기의 새 family와 지정 문구 변화에 대한 평가이며 자연 문서·다른 모델·알고리즘 신규성의 검증은 아니다. 과거 모델 출력이나 gold가 새 방법의 입력에 들어가지 않는다.

## 실행과 재개

기본 `REUSE_PARENT=True`는 과거와 Python/Torch/CUDA/GPU/Transformers/tokenizers/chat template가 동일한 경우에만 개발 출력의 재사용을 허용한다. 불일치 오류가 나오면 설정 셀에서 `REUSE_PARENT=False`로 바꾸고 Run All 한다. 별도 `_fresh` 폴더에서 개발 비교군도 다시 실행한다. Holdout 요청은 어떤 경우에도 과거 출력에 적중하면 오류로 중단한다.

CPU 준비 후 `call_plan.json`을 출력한다. 재사용 시 개발 추가 gate는 21회, holdout 후보 생성의 고유 요청은 2,208회이며 gate를 포함한 holdout 상한은 4,416회다. 합산 새 요청 상한은 4,437회, 전부 재실행하면 상한 6,192회다. Gate는 후보가 숫자인 경우에만 호출하고 동일 요청을 공유하므로 실제 요청은 상한보다 적을 수 있다. 이 수치는 실행 시간 측정값이 아니다. 긴 전체 문맥 기준선도 새로 생성하므로 이전 보완 실험보다 오래 걸린다.

Drive 보관을 원하면 설정 셀의 mount 두 줄을 해제한다. 연결되지 않았으면 결과는 `/content/hw2_holdout/holdout_v1`에 있다. 중단되면 같은 환경·코드·설정·RUN_DIR에서 Run All 한다. 저장된 원출력은 다시 생성하지 않는다. Invalid나 오답을 이유로 재시도하지 않는다. 완료 후 `holdout_v1_results.zip`(fresh면 `holdout_v1_fresh_results.zip`)을 다운로드한다. VS Code 다운로드 경로는 `/home/cv/SSD0/jang/nlp/` 아래로 지정한다.

## 분석 파일

- `results/report_ko.md`: 개발·기존 표현 holdout·변형 표현 holdout의 별도 결과.
- `metrics.csv`, `paired_contrasts.csv`: ID+gate 대 Target-only-routed+같은 gate를 주 비교로 사용한다. 일반 Target-only+gate와의 차이도 보존한다. Task/조건별 정확도, PRESENT 손실, MISSING false-fill, invalid, coverage를 Decision/service와 strict/semantic 정책으로 별도 저장한다.
- `paired_style_shift.csv`: 같은 family의 변형 표현−기존 표현 정확도 차이에 대한 paired bootstrap.
- `zero_event_base_bounds.csv`: 사건 0건을 위험 0으로 읽지 않기 위한 family-any-event 상한. 비열등성 검정은 아니다.
- `costs.csv`, `method_costs.json`: split별 캐시 반영 고유 요청과 입력마다 반복해서 센 논리적 요청/시간/토큰. 재사용과 새 생성을 구분한다. CPU·모델 로드·종단 latency를 제외하며 서로 겹치는 수치를 합산하지 않는다.
- `paired_examples.json`: 방법 간 승패 범주의 전체 ID와 고정 RNG의 균등 예시. 결과에 유리한 사례만 수작업 선택하지 않는다.
- `figures/holdout_accuracy.{png,pdf}`, `holdout_conditions.{png,pdf}`: 논문용 그림.

Base 단위 task-stratified paired bootstrap 2,000회다. 두 표현을 합친 표에서도 독립 family는 16개다. 과제·조건별 family는 8개라 CI가 거칠 수 있다. 관측된 정답 수 동률을 통계적 동등성으로 해석하지 않는다. 분석은 결과를 보고 자동으로 방법을 선택하지 않는다.

## 재현과 출처

원출력 및 소스 snapshot은 `source/provenance.json`에 기록한 두 완료 ZIP에 연결된다. CPU에서 기존 pointer 480개와 gate 480개 판정, 2,000개 과거 원출력을 복원·검증한 후 재사용한다. 추론은 Colab guard 뒤에서만 허용한다. Alice 배경은 [Project Gutenberg](https://www.gutenberg.org/ebooks/11)의 공개 자료다.

CPU 단위 검사: `python -m unittest discover -s experiments/hw2_holdout_colab/tests -t .`. 전체 소프트웨어 검사: `HF_HUB_OFFLINE=1 python -m experiments.hw2_holdout_colab.tests.cpu_smoke --cache-dir .cache/huggingface/hub`. 후자는 명시적인 테스트 응답만 임시 디렉터리에 생성하며 Qwen 성능을 측정하지 않는다.

노트북 생성: `python -m experiments.hw2_holdout_colab.build_notebook`. 실행 출력이 있는 노트북은 덮어쓰지 않는다.
