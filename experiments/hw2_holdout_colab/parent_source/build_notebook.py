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
    target = Path(output) if output else HERE / "hw2_relation_gate_colab.ipynb"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if any(c.get("outputs") or c.get("execution_count") is not None for c in existing["cells"] if c["cell_type"] == "code"):
            raise RuntimeError("Refusing to overwrite an executed notebook. Choose --output NEW.ipynb.")
    sources = {"experiments/__init__.py": '"""Research experiments."""\n'}
    sources.update({"experiments/hw2_relation_gate_colab/" + str(p.relative_to(HERE)): p.read_text(encoding="utf-8") for p in source_paths()})
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
CODE_ROOT = Path("/content/hw2_relation_gate_code")
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
    if name == "experiments.hw2_relation_gate_colab" or name.startswith("experiments.hw2_relation_gate_colab."):
        del sys.modules[name]
print("Embedded source ready:", CODE_ROOT, "|", EXPECTED_BUNDLE[:12])
'''
    install = '''import importlib, subprocess, sys
from importlib import metadata
for package, expected in (("transformers", "4.57.6"), ("tokenizers", "0.22.2")):
    if package in sys.modules and metadata.version(package) != expected:
        raise RuntimeError(package + " 버전이 이미 다르게 로드됐습니다. 커널 재시작 후 Run All 하세요.")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r",
               str(CODE_ROOT / "experiments/hw2_relation_gate_colab/requirements-colab.txt")], check=True)
importlib.invalidate_caches()
print("Transformers:", metadata.version("transformers"), "| tokenizers:", metadata.version("tokenizers"))
'''
    settings = '''# Drive에 보관하려면 아래 두 줄의 주석을 해제하세요.
# from google.colab import drive
# drive.mount("/content/drive")
REUSE_PARENT = True  # 환경이 다르다는 오류가 나오면 False로 변경 후 Run All
DRIVE = Path("/content/drive/MyDrive")
RESULT_BASE = DRIVE / "hw2_relation_gate" if DRIVE.is_dir() else Path("/content/hw2_relation_gate")
RUN_NAME = "relation_gate_dev_v1" + ("" if REUSE_PARENT else "_fresh")
RUN_DIR = RESULT_BASE / RUN_NAME
MODEL_CACHE = Path("/content/hw2_value_tracking_model_cache")
print("RESULTS:", RUN_DIR)
print("Reuse parent outputs:", REUSE_PARENT, "| Drive persistent:", DRIVE.is_dir())
'''
    tests = '''import unittest
suite = unittest.defaultTestLoader.discover(str(CODE_ROOT / "experiments/hw2_relation_gate_colab/tests"),
                                          pattern="test_*.py", top_level_dir=str(CODE_ROOT))
test_result = unittest.TextTestRunner(verbosity=1).run(suite)
if not test_result.wasSuccessful():
    raise RuntimeError("CPU checks failed; inference was not started")
'''
    prepare = '''from experiments.hw2_relation_gate_colab.runner import prepare
prepared = prepare(RUN_DIR, cache_dir=MODEL_CACHE, reuse_parent=REUSE_PARENT)
print(json.dumps(json.loads((RUN_DIR / "call_plan.json").read_text()), ensure_ascii=False, indent=2))
'''
    preflight = '''from experiments.hw2_relation_gate_colab.engine import preflight
_, runtime = preflight(prepared["tok"], prepared["parent"].setup, REUSE_PARENT)
print("Runtime check passed; no model loaded yet.")
print(json.dumps(runtime, indent=2))
'''
    run = '''from experiments.hw2_relation_gate_colab.runner import run, archive
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
    display(Image(filename=str(RUN_DIR / "figures/condition_accuracy.png")))
    display(Image(filename=str(RUN_DIR / "figures/representation_diagnostic.png")))
else:
    print("추론·감사 완료 후 결과가 생성됩니다. status.json:", status)
'''
    download = '''from experiments.hw2_relation_gate_colab.runner import archive
result_zip = archive(RUN_DIR)
print("RESULT_ZIP:", result_zip)
print("VS Code 저장 위치 예: /home/cv/SSD0/jang/nlp/" + Path(result_zip).name)
try:
    from google.colab import files
    files.download(result_zip)
except Exception as exc:
    print("Colab 왼쪽 파일 패널에서 위 ZIP을 다운로드하세요:", type(exc).__name__)
'''
    cells = [cell("markdown", "# HW2 — ID 선택 + 관계 검증 보완 실험\n\n**웹 Colab에 이 ipynb 업로드 → T4 GPU → Run All. 추가 학습 없음.**\n\n기존 원출력과 코드·자료가 내장되어 추가 파일 업로드가 필요 없습니다. 기존 480개 개발 입력에서 **ID 선택 vs Standard+같은 검증기 vs ID+검증기**를 비교합니다. 같은 prompt의 자유 생성과 `SUPPORTED / UNSUPPORTED`만 허용하는 제약 생성을 구분합니다. 두 방식의 후보 숫자 표기도 통일합니다.\n\n별도로 숫자 ID 위치·항목 ID 위치·대시 표시를 비교하는 96회 대응 진단을 실행합니다. **독립 holdout이 아닌 개발 실험**입니다. 기존 노트북·ZIP은 수정하지 않습니다.\n\n기본값은 환경이 같은 경우 과거 출력을 재사용해 호출을 줄입니다. 환경 불일치 오류가 나오면 설정 셀에서 `REUSE_PARENT=False`로 바꾸고 Run All 하세요. 이 경우 새 폴더에서 모든 비교군을 다시 실행하며 더 오래 걸립니다. Drive 미연결 시 마지막 ZIP을 내려받으세요.\n", "gate-intro"),
             cell("code", guard, "gate-colab-guard"), cell("code", bootstrap, "gate-embedded-source"),
             cell("code", install, "gate-install"), cell("code", settings, "gate-settings"), cell("code", tests, "gate-cpu-tests"),
             cell("code", prepare, "gate-prepare"), cell("code", preflight, "gate-runtime-check"),
             cell("markdown", "## 추론 실행\n\n아래에서만 Qwen을 로드합니다. 위에 표시된 호출 수는 사전 계산이며 실행 시간은 실제 측정합니다. 같은 요청은 공유하고, invalid나 오답 때문에 재시도하지 않습니다. 중단 후 같은 환경·설정에서 다시 실행하면 체크포인트를 사용합니다. 제약 생성의 라벨 판정 자체가 틀릴 가능성도 측정 대상입니다.\n", "gate-run-note"),
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
