"""Create a self-contained clean notebook without overwriting executed notebooks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

from .common import HERE, source_paths


def cell(kind, source, identifier):
    obj = {"cell_type": kind, "id": identifier, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        obj.update(execution_count=None, outputs=[])
    return obj


def build(output=None):
    target = Path(output) if output else HERE / "hw2_holdout_colab.ipynb"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if any(c.get("outputs") or c.get("execution_count") is not None for c in existing["cells"] if c["cell_type"] == "code"):
            raise RuntimeError("Refusing to overwrite an executed notebook. Choose --output NEW.ipynb.")
    sources = {"experiments/__init__.py": '"""Research experiments."""\n'}
    # Decode bytes directly so CRLF and a UTF-8 BOM survive the embedded round trip.
    sources.update({"experiments/hw2_holdout_colab/" + str(p.relative_to(HERE)): p.read_bytes().decode("utf-8") for p in source_paths()})
    encoded = base64.b64encode(zlib.compress(json.dumps(sources, ensure_ascii=False).encode(), 9)).decode()
    bundle = hashlib.sha256(json.dumps(sources, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    guard = '''import platform, subprocess, sys
from pathlib import Path
if platform.system() != "Linux" or not Path("/content").is_dir():
    raise RuntimeError("웹 Colab에서 T4 GPU를 선택하세요. 로컬 모델 실행은 차단됩니다.")
import google.colab
print("Python:", sys.version.split()[0])
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout)
'''
    bootstrap = '''import base64, hashlib, json, sys, zlib
from pathlib import Path, PurePosixPath
CODE_ROOT = Path("/content/hw2_holdout_code")
SOURCES = json.loads(zlib.decompress(base64.b64decode(''' + repr(encoded) + ''')))
EXPECTED_BUNDLE = ''' + repr(bundle) + '''
if hashlib.sha256(json.dumps(SOURCES, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != EXPECTED_BUNDLE:
    raise RuntimeError("Embedded source integrity failure")
for relative, content in SOURCES.items():
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError("Unsafe embedded path")
    path = CODE_ROOT / relative
    if not path.resolve().is_relative_to(CODE_ROOT.resolve()):
        raise RuntimeError("Embedded path escapes code root")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))
sys.path.insert(0, str(CODE_ROOT))
package = sys.modules.get("experiments")
if package is not None:
    if not hasattr(package, "__path__"):
        raise RuntimeError("Incompatible experiments module; restart the kernel")
    folder = str(CODE_ROOT / "experiments")
    package.__path__ = [folder] + [p for p in package.__path__ if p != folder]
for name in list(sys.modules):
    if name == "experiments.hw2_holdout_colab" or name.startswith("experiments.hw2_holdout_colab."):
        del sys.modules[name]
print("Embedded source ready:", CODE_ROOT, "|", EXPECTED_BUNDLE[:12])
'''
    install = '''import importlib, subprocess, sys
from importlib import metadata
for package, expected in (("transformers", "4.57.6"), ("tokenizers", "0.22.2")):
    if package in sys.modules and metadata.version(package) != expected:
        raise RuntimeError(package + " 버전이 이미 다르게 로드됐습니다. 커널 재시작 후 Run All 하세요.")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r",
               str(CODE_ROOT / "experiments/hw2_holdout_colab/requirements-colab.txt")], check=True)
importlib.invalidate_caches()
print("Transformers:", metadata.version("transformers"), "| tokenizers:", metadata.version("tokenizers"))
'''
    settings = '''# Drive에 보관하려면 아래 두 줄의 주석을 해제하세요.
# from google.colab import drive
# drive.mount("/content/drive")
REUSE_PARENT = True  # 환경이 다르다는 오류가 나오면 False로 변경 후 Run All
DRIVE = Path("/content/drive/MyDrive")
RESULT_BASE = DRIVE / "hw2_holdout" if DRIVE.is_dir() else Path("/content/hw2_holdout")
RUN_NAME = "holdout_v1" + ("" if REUSE_PARENT else "_fresh")
RUN_DIR = RESULT_BASE / RUN_NAME
MODEL_CACHE = Path("/content/hw2_value_tracking_model_cache")
print("RESULTS:", RUN_DIR)
print("Reuse parent outputs:", REUSE_PARENT, "| Drive persistent:", DRIVE.is_dir())
'''
    tests = '''import unittest
suite = unittest.defaultTestLoader.discover(str(CODE_ROOT / "experiments/hw2_holdout_colab/tests"),
                                          pattern="test_*.py", top_level_dir=str(CODE_ROOT))
test_result = unittest.TextTestRunner(verbosity=1).run(suite)
if not test_result.wasSuccessful():
    raise RuntimeError("CPU checks failed; inference was not started")
'''
    prepare = '''from experiments.hw2_holdout_colab.runner import prepare
prepared = prepare(RUN_DIR, cache_dir=MODEL_CACHE, reuse_parent=REUSE_PARENT)
print(json.dumps(json.loads((RUN_DIR / "call_plan.json").read_text()), ensure_ascii=False, indent=2))
'''
    preflight = '''from experiments.hw2_holdout_colab.engine import preflight
_, runtime = preflight(prepared["tok"], prepared["history"].setup, REUSE_PARENT)
print("Runtime check passed; no model loaded yet.")
print(json.dumps(runtime, indent=2))
'''
    run = '''from experiments.hw2_holdout_colab.runner import run, archive
try:
    summary = run(RUN_DIR, cache_dir=MODEL_CACHE, reuse_parent=REUSE_PARENT, prepared=prepared)
finally:
    if RUN_DIR.exists():
        result_zip = archive(RUN_DIR)
'''
    report = '''from IPython.display import Markdown, Image, display
status = json.loads((RUN_DIR / "status.json").read_text()) if (RUN_DIR / "status.json").exists() else {}
if status.get("status") == "COMPLETE":
    display(Markdown((RUN_DIR / "results/report_ko.md").read_text(encoding="utf-8")))
    display(Image(filename=str(RUN_DIR / "figures/holdout_accuracy.png")))
    display(Image(filename=str(RUN_DIR / "figures/holdout_conditions.png")))
else:
    print("추론·감사 완료 후 결과가 생성됩니다. status.json:", status)
'''
    download = '''from experiments.hw2_holdout_colab.runner import archive
result_zip = archive(RUN_DIR)
print("RESULT_ZIP:", result_zip)
print("VS Code 저장 위치 예: /home/cv/SSD0/jang/nlp/" + Path(result_zip).name)
try:
    from google.colab import files
    files.download(result_zip)
except Exception as exc:
    print("Colab 왼쪽 파일 패널에서 위 ZIP을 다운로드하세요:", type(exc).__name__)
'''
    cells = [cell("markdown", "# HW2 — Target-only 비교군과 고정 holdout 평가\n\n**웹 Colab에 이 ipynb 업로드 → T4 GPU → Run All. Qwen2.5-1.5B 하나, 추가 학습 없음.**\n\n코드·과거 원출력이 내장되어 추가 파일 업로드가 필요 없습니다. ① 개발 480입력에 **Target-only 숫자 출력 + 기존과 동일한 제약 관계 검증기**를 추가합니다. ② 방법을 고정한 뒤 새 개체·숫자 조합의 **holdout 1,152입력**에서 ID/Target-only/Standard와 각 gate 조합을 평가합니다.\n\nHoldout은 **새 family 16개(task당 8개)**의 6조건 × 3문서 형식 × 2placement × 기존/변형 표현입니다. 1,152개가 독립 family인 것은 아닙니다. 두 표현은 같은 사실·배경을 공유합니다. 기존 literal parser와 모든 선언된 속성 표현을 아는 강한 schema-aware parser를 함께 평가합니다. 개발/holdout 결과는 별도 표와 통계로 저장합니다.\n\n**새 holdout 모델 출력은 이 노트북에서 처음 생성합니다.** 전체 문맥 추론을 새로 실행하므로 이전 378회 실험보다 오래 걸립니다. CPU 준비 후 정확한 개발 추가 요청과 holdout 요청 상한을 표시합니다. 실행 전에 코드·자료·정답·prompt·parser를 잠그고, 결과를 보고 방법을 자동 변경하지 않습니다. 같은 합성 분포와 사전 선언한 문구 변화에 대한 평가이며 자연 문서·다른 모델의 일반화나 알고리즘 신규성을 확정하지 않습니다.\n\n기본 `REUSE_PARENT=True`는 기록된 실행 환경이 같을 때 개발 원출력만 재사용합니다. 불일치 오류라면 설정 셀에서 False로 바꾸고 Run All 하세요. 개발 비교군도 새 폴더에서 다시 실행합니다. Holdout에는 과거 원출력을 사용하지 않습니다. Drive 미연결 시 마지막 ZIP을 내려받으세요. 기존 실행 노트북·ZIP은 보존합니다.\n", "gate-intro"),
             cell("markdown", "주 비교에는 **Target-only-routed + gate**를 사용합니다. ID 방식과 똑같이 target에 숫자가 없으면 CPU에서 null을 반환하고 모델을 호출하지 않습니다. 일반 Target-only + gate도 함께 저장해 호출 생략 효과를 구분합니다. 이 비교군은 같은 원출력을 공유하므로 새 GPU 요청을 늘리지 않습니다.\n", "holdout-matched-control"),
             cell("code", guard, "gate-colab-guard"), cell("code", bootstrap, "gate-embedded-source"),
             cell("code", install, "gate-install"), cell("code", settings, "gate-settings"), cell("code", tests, "gate-cpu-tests"),
             cell("code", prepare, "gate-prepare"), cell("code", preflight, "gate-runtime-check"),
             cell("markdown", "## 추론 실행\n\n아래에서만 Qwen을 로드합니다. 입력을 먼저 모두 고정했으며, holdout 요청 상한은 실제 소요 시간이나 정확도 예측이 아닙니다. 동일 요청은 공유하고 invalid나 오답 때문에 재시도하지 않습니다. 중단 후 같은 코드·환경·설정에서 다시 실행하면 체크포인트를 사용합니다. 실제 새 생성량·시간·메모리와 입력당 논리적 비용도 저장합니다.\n", "gate-run-note"),
             cell("code", run, "gate-run"), cell("code", report, "gate-report"), cell("code", download, "gate-download")]
    for c in cells:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), c["id"], "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
                             "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}}}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Built:", target, "| bytes:", target.stat().st_size, "| bundle:", bundle[:12])
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    build(parser.parse_args().output)
