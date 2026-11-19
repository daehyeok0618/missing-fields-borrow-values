"""Build a self-contained Colab notebook that re-runs the locked Salary and Founding Year
experiments on a Colab Tesla T4.

The notebook embeds the exact protocol-locked files of both experiments (byte-identical,
verified against each protocol_lock.json), restores them at the original absolute paths
(/home/cv/SSD0/jang/nlp/experiments/...), and then runs the UNMODIFIED scripts:
    infer --split dev  ->  infer --split main  ->  analyze
for each task. Nothing in the locked protocol is changed; only the hardware differs.

Usage (from /home/cv/SSD0/jang/nlp):
    .venv-exception-loss/bin/python experiments/hw2_t4_rerun_colab/build_notebook.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                      # /home/cv/SSD0/jang/nlp/experiments
BASE = EXP.parent                      # /home/cv/SSD0/jang/nlp
NOTEBOOK = HERE / "hw2_t4_rerun_colab.ipynb"
if "--out" in sys.argv:                 # build elsewhere (e.g. for testing) without touching NOTEBOOK
    NOTEBOOK = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()

TASKS = {
    "salary": {"dir": "hw2_missing_fields", "run_id": "run_20261031_hw1_main_v2"},
    "year": {"dir": "hw2_year_generalization", "run_id": "run_20261102_founding_year"},
}
# analyze.py records these documents in completion_manifest.json, so they must exist.
EXTRA_DOCS = ("reproduce.md", "ai_usage_log.md")
TRANSFORMERS = "4.57.6"   # same pins as the Colab T4 probe run; token counts verified identical
TOKENIZERS = "0.22.2"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect():
    """Return {relative path under experiments/: bytes} and per-task lock hashes."""
    files, locks = {}, {}
    for key, task in TASKS.items():
        root = EXP / task["dir"]
        lock_rel = f"runs/{task['run_id']}/protocol_lock.json"
        lock = json.loads((root / lock_rel).read_text())
        cfg = json.loads((root / "config.json").read_text())
        if cfg["run_id"] != task["run_id"]:
            raise RuntimeError(f"{key}: run_id mismatch")
        for rel, expected in lock["hashes"].items():
            data = (root / rel).read_bytes()
            if sha256(data) != expected:
                raise RuntimeError(f"{key}: locked file changed locally: {rel}")
            files[f"{task['dir']}/{rel}"] = data
        files[f"{task['dir']}/{lock_rel}"] = (root / lock_rel).read_bytes()
        for doc in EXTRA_DOCS:
            files[f"{task['dir']}/{doc}"] = (root / doc).read_bytes()
        locks[key] = {"dir": task["dir"], "run_id": task["run_id"], "hashes": lock["hashes"],
                      "model_id": cfg["model"]["id"], "revision": cfg["model"]["revision"],
                      "cache_dir": cfg["model"]["cache_dir"]}
    return files, locks


def make_bundle(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tf:
        for rel in sorted(files):
            info = tarfile.TarInfo(rel)
            info.size = len(files[rel]); info.mtime = 0; info.mode = 0o644
            tf.addfile(info, io.BytesIO(files[rel]))
    return buf.getvalue()


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build():
    if NOTEBOOK.exists():
        old = json.loads(NOTEBOOK.read_text())
        if any(c.get("outputs") for c in old["cells"] if c["cell_type"] == "code"):
            raise RuntimeError("Refusing to overwrite an executed notebook")
    files, locks = collect()
    bundle = make_bundle(files)
    bundle_b64 = base64.b64encode(bundle).decode()
    bundle_sha = sha256(bundle)
    revisions = {v["revision"] for v in locks.values()}
    caches = {v["cache_dir"] for v in locks.values()}
    if len(revisions) != 1 or len(caches) != 1:
        raise RuntimeError("Tasks disagree on model revision or cache")
    revision, cache_dir = revisions.pop(), caches.pop()

    intro = f"""
# Re-run the Salary and Founding Year experiments on a Colab T4

Open this notebook in Google Colab, select **Runtime > Change runtime type > T4 GPU**, and choose **Run all**.

- The notebook embeds the {len(files)} protocol-locked files of both experiments (inputs, scripts, configuration), checks each against the SHA-256 in its `protocol_lock.json`, and stops if any differs.
- It then runs the unmodified scripts for each task: `infer --split dev` → `infer --split main` → `analyze`.
- Model: `Qwen/Qwen2.5-1.5B-Instruct@{revision[:7]}`, FP16, greedy decoding; no training.
- Libraries: transformers {TRANSFORMERS}, tokenizers {TOKENIZERS}. Before generation, the notebook checks that all 14,256 inputs have the locked token counts.
- The main split took about 45 minutes per task on a T4. If the development split projects more than two hours for the main split, the main split is not run (a rule fixed before the experiment).
- When a task finishes, a results ZIP is downloaded. Set `USE_DRIVE = True` below to keep progress on Google Drive and resume after a disconnect; outputs that already exist are never regenerated.
- Files are restored under `/home/cv/SSD0/jang/nlp/experiments/`, the path that the locked configuration files point to.
"""
    settings = """
# ---- 설정 ----
USE_DRIVE = False                              # True: 진행 상황을 Google Drive에 저장(끊겨도 이어서 실행)
DRIVE_DIR = "/content/drive/MyDrive/hw2_t4_rerun"
RUN_TASKS = ["salary", "year"]                 # 한 과제만 돌리려면 하나만 남기세요
"""
    guard = """
import platform, subprocess, sys
from pathlib import Path
if platform.system() != "Linux" or not Path("/content").is_dir():
    raise RuntimeError("Colab 런타임에서만 실행하세요.")
import google.colab  # noqa: F401  (Colab이 아니면 여기서 실패)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True, check=True).stdout.strip()
print("GPU:", gpu)
if "T4" not in gpu:
    raise RuntimeError("T4 GPU가 아닙니다. 런타임 > 런타임 유형 변경 > T4 GPU를 선택하세요.")
print("Python:", sys.version.split()[0])
"""
    install = f"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers=={TRANSFORMERS}", "tokenizers=={TOKENIZERS}",
                "accelerate>=1.0,<2", "safetensors>=0.4,<1"], check=True)
print(subprocess.run([sys.executable, "-c",
      "import torch, transformers, tokenizers, numpy, matplotlib; "
      "print('torch', torch.__version__, 'cuda', torch.version.cuda); "
      "print('transformers', transformers.__version__, 'tokenizers', tokenizers.__version__); "
      "print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"],
      capture_output=True, text=True, check=True).stdout)
"""
    restore = f"""
import base64, hashlib, io, json, os, shutil, tarfile
from pathlib import Path
BASE = Path("/home/cv/SSD0/jang/nlp")          # 원래 경로 그대로(설정 파일의 cache_dir와 모듈 경로가 이 위치를 가정)
EXP = BASE / "experiments"
TASKS = {json.dumps({k: {"dir": v["dir"], "run_id": v["run_id"]} for k, v in locks.items()})}
LOCK_HASHES = {json.dumps({k: v["hashes"] for k, v in locks.items()})}
BUNDLE_SHA = "{bundle_sha}"
BUNDLE = base64.b64decode("{bundle_b64}")
if hashlib.sha256(BUNDLE).hexdigest() != BUNDLE_SHA:
    raise RuntimeError("내장 파일 묶음이 손상되었습니다")
sha = lambda b: hashlib.sha256(b).hexdigest()
with tarfile.open(fileobj=io.BytesIO(BUNDLE), mode="r:xz") as tf:
    for member in tf.getmembers():
        rel = Path(member.name)
        if rel.is_absolute() or ".." in rel.parts or not member.isfile():
            raise RuntimeError("허용되지 않는 경로: " + member.name)
        data = tf.extractfile(member).read()
        dest = EXP / rel
        if dest.exists():
            if sha(dest.read_bytes()) != sha(data):
                raise RuntimeError("이미 있는 파일이 내장본과 다릅니다(덮어쓰지 않음): " + str(dest))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
for key, t in TASKS.items():
    root = EXP / t["dir"]
    run = root / "runs" / t["run_id"]
    if USE_DRIVE and not run.is_symlink():
        target = Path(DRIVE_DIR) / t["dir"] / t["run_id"]
        target.mkdir(parents=True, exist_ok=True)
        lock = run / "protocol_lock.json"
        if (target / "protocol_lock.json").exists():
            if sha((target / "protocol_lock.json").read_bytes()) != sha(lock.read_bytes()):
                raise RuntimeError("Drive의 protocol_lock이 다릅니다: " + str(target))
        else:
            shutil.copy2(lock, target / "protocol_lock.json")
        shutil.rmtree(run)
        run.symlink_to(target, target_is_directory=True)
    bad = [p for p, h in LOCK_HASHES[key].items() if sha((root / p).read_bytes()) != h]
    if bad:
        raise RuntimeError({{key: bad}})
    # analyze.py saves plots to figures/ without creating it; the tar bundle has no empty folders.
    (root / "figures").mkdir(exist_ok=True)
    print(key, "| 잠긴 파일", len(LOCK_HASHES[key]), "개 일치 | run 폴더:", run.resolve(),
          "| 기존 출력:", sorted(p.name for p in run.resolve().iterdir()))
"""
    model = f"""
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision="{revision}",
                         cache_dir="{cache_dir}")
if not path.rstrip("/").endswith("{revision}"):
    raise RuntimeError("모델 revision 불일치: " + path)
print("모델 준비 완료:", path)
"""
    preflight = """
# 전체 입력 14,256개의 토큰 수가 잠긴 값과 같은지 먼저 확인(다르면 생성 전에 중단)
import subprocess, sys
CHECK = r'''
import json, sys
sys.path.insert(0, "/home/cv/SSD0/jang/nlp")
for d in ("hw2_missing_fields", "hw2_year_generalization"):
    common = __import__(f"experiments.{d}.scripts.common", fromlist=["x"])
    tok = common.tokenizer()
    for split in ("dev", "main"):
        rows = common.read_jsonl(common.ROOT / f"data/{split}_requests.jsonl")
        enc = tok([common.render(r["messages"], tok) for r in rows], add_special_tokens=False)["input_ids"]
        bad = [r["request_id"] for r, ids in zip(rows, enc) if len(ids) != r["input_token_count"]]
        print(d, split, len(rows), "inputs, token-count mismatches:", len(bad), bad[:3], flush=True)
        if bad:
            sys.exit(1)
'''
r = subprocess.run([sys.executable, "-c", CHECK], cwd="/home/cv/SSD0/jang/nlp", text=True,
                   capture_output=True)
print(r.stdout); print(r.stderr[-2000:] if r.returncode else "")
if r.returncode:
    raise RuntimeError("토큰 수가 달라졌습니다. 실행하지 말고 결과를 공유해 주세요.")
"""
    helpers = """
import csv, json, subprocess, sys, zipfile
from pathlib import Path

def run(module, *args):
    cmd = [sys.executable, "-m", module, *args]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"실패: {module} {args} (출력은 run 폴더에 보존됨)")

def package(key):
    t = TASKS[key]; root = EXP / t["dir"]; run_dir = (root / "runs" / t["run_id"]).resolve()
    out = Path(f"/content/hw2_t4_{key}_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, arc in ((run_dir, f"runs/{t['run_id']}"), (root / "results", "results"), (root / "figures", "figures")):
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p, f"{t['dir']}/{arc}/{p.relative_to(folder)}")
        for name in ("final_report_ko.md", "completion_manifest.json"):
            if (root / name).exists():
                z.write(root / name, f"{t['dir']}/{name}")
    print("결과 ZIP:", out, round(out.stat().st_size / 1e6, 1), "MB")
    if USE_DRIVE:
        import shutil; shutil.copy2(out, Path(DRIVE_DIR) / out.name)
    from google.colab import files
    files.download(str(out))

def finished(key, split):
    # A finished split must not be re-run: infer.py would overwrite {split}_runtime.json
    # (and pilot_audit.json for dev) with the timings of a run that generated nothing.
    t = TASKS[key]; p = (EXP / t["dir"] / "runs" / t["run_id"]).resolve() / f"{split}_runtime.json"
    if not p.exists():
        return False
    rt = json.loads(p.read_text())
    return rt.get("status") == "COMPLETE" and rt.get("completed") == rt.get("requests")

def run_task(key):
    module = f"experiments.{TASKS[key]['dir']}.scripts"
    for split in ("dev", "main"):
        if finished(key, split):
            print(f"{key} {split}: 이미 완료됨 → 추론을 다시 실행하지 않음(실행 시간 기록 보존)", flush=True)
        else:
            run(module + ".infer", "--split", split, "--device", "cuda:0")
    run(module + ".analyze")
    package(key)
"""
    salary = """
if "salary" in RUN_TASKS:
    run_task("salary")
"""
    year = """
if "year" in RUN_TASKS:
    run_task("year")
"""
    summary = """
# 요약: 주요 대비(%p, 95% CI)와 실행 환경
for key in RUN_TASKS:
    root = EXP / TASKS[key]["dir"]
    rows = list(csv.DictReader((root / "results/contrasts.csv").open()))
    rows = [r for r in rows if r["subset"] == "ALL" and r.get("policy", "STRICT") == "STRICT"
            and r["metric"] in ("delta_attr", "delta_C", "delta_U", "delta_type")]
    rt = json.loads(((root / "runs" / TASKS[key]["run_id"]).resolve() / "main_runtime.json").read_text())
    print(f"== {key} | GPU {rt['gpu']} | main {rt['process_seconds']/60:.1f} min | valid {rt['strict_valid']}/{rt['requests']}")
    for r in rows:
        print(f"  {r['metric']:<11} {100*float(r['estimate']):+7.2f}  [{100*float(r['ci95_low']):+.2f}, {100*float(r['ci95_high']):+.2f}]")
"""
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": [cell("markdown", intro), cell("code", settings), cell("code", guard), cell("code", install),
                    cell("code", restore), cell("code", model), cell("code", preflight), cell("code", helpers),
                    cell("markdown", "## Salary (dev → main → analyze → ZIP download)"), cell("code", salary),
                    cell("markdown", "## Founding Year (dev → main → analyze → ZIP download)"), cell("code", year),
                    cell("markdown", "## Summary"), cell("code", summary)]}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    self_test(bundle, locks)
    print(json.dumps({"notebook": str(NOTEBOOK), "size_MB": round(NOTEBOOK.stat().st_size / 1e6, 2),
                      "embedded_files": len(files), "bundle_sha256": bundle_sha}, indent=1))


def self_test(bundle, locks):
    """Extract the embedded bundle into a temp dir and re-verify every locked hash."""
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:xz") as tf:
            tf.extractall(tmp, filter="data")
        for key, lock in locks.items():
            root = Path(tmp) / lock["dir"]
            bad = [p for p, h in lock["hashes"].items() if sha256((root / p).read_bytes()) != h]
            extra = sorted(p.name for p in (root / "runs" / lock["run_id"]).iterdir())
            if bad or extra != ["protocol_lock.json"]:
                raise RuntimeError({key: {"bad": bad, "run_dir": extra}})
            print(f"self-test {key}: {len(lock['hashes'])} locked files OK, run dir = {extra}")


if __name__ == "__main__":
    sys.exit(build())
