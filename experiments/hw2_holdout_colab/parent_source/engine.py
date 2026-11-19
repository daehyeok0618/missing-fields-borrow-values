"""Colab-only model access, exact request caching, constrained decoding and cost refs."""
import copy
from collections import defaultdict
import platform
import time
from pathlib import Path

from .common import digest, now, read_json, write_json
from .core import label_sequences, prefix_callback
from .protocol import METHODS, PROTOCOL

RUNTIME_KEYS = ("python", "torch", "transformers", "tokenizers", "cuda_build", "gpu", "capability", "chat_template_hash")


def colab_guard():
    if platform.system() != "Linux" or not Path("/content").is_dir():
        raise RuntimeError("Colab GPU 커널을 선택하세요. 로컬 모델 추론은 차단됩니다.")
    import google.colab  # noqa: F401
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Colab에서 T4 GPU를 활성화하세요.")
    return torch


def runtime_info(torch, tok):
    from importlib import metadata
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": metadata.version("transformers"), "tokenizers": metadata.version("tokenizers"),
            "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "chat_template_hash": digest(tok.chat_template)}


def compatibility(current, parent):
    return {k: {"current": current.get(k), "parent": parent.get(k)} for k in RUNTIME_KEYS if current.get(k) != parent.get(k)}


def preflight(tok, parent_setup, reuse_parent):
    torch = colab_guard()
    info = runtime_info(torch, tok)
    if info["transformers"] != PROTOCOL["transformers"] or info["tokenizers"] != PROTOCOL["tokenizers"]:
        raise RuntimeError("의존성 셀을 실행하세요. 이미 다른 버전이 import됐다면 커널 재시작이 필요합니다.")
    mismatch = compatibility(info, parent_setup)
    if reuse_parent and mismatch:
        raise RuntimeError("기존 출력 재사용 환경이 다릅니다: " + str(mismatch)
                           + "\n설정 셀의 REUSE_PARENT=False로 바꾸고 Run All: 새 폴더에서 비교군을 전부 재실행합니다.")
    return torch, info


def query_key(messages, maximum):
    return digest({"messages": messages, "maximum": maximum})


def request_key(messages, maximum, mode, paths, setup_hash):
    return digest({"messages": messages, "maximum": maximum, "mode": mode,
                   "label_sequences": paths if mode == "constrained" else None, "setup": setup_hash})


class Generator:
    def __init__(self, model, tok, root, setup, torch, parent, replay_only=False):
        self.model, self.tok, self.root, self.setup, self.torch = model, tok, Path(root), setup, torch
        self.setup_hash, self.paths = digest(setup), label_sequences(tok)
        self.parent = parent
        self.parent_queries = {query_key(r["messages"], r["max_new_tokens"]): r for r in parent.records.values()}
        self.records, self.used, self.last_used = {}, defaultdict(set), defaultdict(set)
        self.seen_current = set()
        self.new_calls, self.replay_only = 0, replay_only
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        for path in sorted((self.root / "checkpoints").glob("*.json")):
            row = read_json(path)
            self.verify(row, path.stem)
            self.records[path.stem] = row

    def reference(self, row):
        return ("parent:" if row["setup_hash"] == self.parent.setup_hash else "current:") + row["key"]

    def resolve(self, ref):
        kind, key = ref.split(":", 1)
        if kind not in ("parent", "current"):
            raise RuntimeError("Unknown checkpoint reference")
        if kind == "parent" and not self.setup["reuse_parent"]:
            raise RuntimeError("Refresh mode cannot use historical outputs")
        return (self.parent.records if kind == "parent" else self.records)[key]

    def verify(self, row, key):
        if row.get("key") != key or row.get("setup_hash") != self.setup_hash:
            raise RuntimeError("Checkpoint environment mismatch")
        if row.get("record_hash") != digest({k: v for k, v in row.items() if k != "record_hash"}):
            raise RuntimeError("Checkpoint integrity failure")
        mode = row["mode"]
        if mode not in ("free", "constrained") or request_key(row["messages"], row["max_new_tokens"], mode, self.paths, self.setup_hash) != key:
            raise RuntimeError("Checkpoint request mismatch")
        if row.get("label_sequences") != (self.paths if mode == "constrained" else None):
            raise RuntimeError("Checkpoint grammar mismatch")
        if mode == "constrained" and (row["generated_token_ids"] not in self.paths.values() or row["truncated"]):
            raise RuntimeError("Constrained decoder failed to produce a complete legal label")

    def generate(self, messages, stage, max_new_tokens=None, mode="free"):
        maximum = max_new_tokens or PROTOCOL["generation"]["max_new_tokens"]
        if mode not in ("free", "constrained"):
            raise ValueError(mode)
        old_query = query_key(messages, maximum)
        if self.setup["reuse_parent"] and mode == "free" and old_query in self.parent_queries:
            return self.parent_queries[old_query]
        key = request_key(messages, maximum, mode, self.paths, self.setup_hash)
        if key in self.records:
            self.seen_current.add(key)
            return self.records[key]
        if self.replay_only:
            raise RuntimeError("Replay is missing a raw response: " + stage)
        limit = PROTOCOL["max_calls_reuse" if self.setup["reuse_parent"] else "max_calls_refresh"]
        if len(self.records) >= limit:
            raise RuntimeError("Fixed call budget exhausted")
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        input_length = int(inputs.input_ids.shape[1])
        if input_length > PROTOCOL["max_prompt_tokens"]:
            raise RuntimeError("Prompt exceeds budget; refusing silent truncation")
        config = copy.deepcopy(self.model.generation_config)
        for name, value in PROTOCOL["generation"].items():
            setattr(config, name, value)
        config.max_new_tokens, config.pad_token_id = maximum, self.tok.pad_token_id
        config.temperature = config.top_p = config.top_k = None
        extra = {}
        if mode == "constrained":
            config.eos_token_id = self.tok.eos_token_id
            extra["prefix_allowed_tokens_fn"] = prefix_callback(self.paths, input_length)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=config, **extra)
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        ids = output[0, input_length:].tolist()
        eos_ids = set(config.eos_token_id if isinstance(config.eos_token_id, (tuple, list)) else [config.eos_token_id])
        end = next((i for i, token in enumerate(ids) if token in eos_ids), None)
        if end is not None:
            ids = ids[:end + 1]
        row = {"key": key, "setup_hash": self.setup_hash, "stage": stage, "messages": messages,
               "mode": mode, "label_sequences": self.paths if mode == "constrained" else None,
               "max_new_tokens": maximum, "generated_token_ids": ids,
               "raw_output": self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
               "input_tokens": input_length, "output_tokens": len(ids), "generation_seconds": seconds,
               "truncated": end is None and len(ids) >= maximum, "finished_utc": now(), "system_error": None}
        row["record_hash"] = digest(row)
        # Preserve even an unexpected decoder failure for inspection; never silently retry it.
        write_json(self.root / "checkpoints" / (key + ".json"), row)
        self.records[key] = row
        self.seen_current.add(key)
        self.verify(row, key)
        self.new_calls += 1
        if self.new_calls == 1 or self.new_calls % 20 == 0:
            print("New generations:", self.new_calls, "|", stage, "|", mode, flush=True)
        return row

    def attribute(self, method, *rows):
        refs = {self.reference(r) for r in rows}
        self.used[method].update(refs)
        self.last_used[method].update(refs)

    def costs(self):
        def total(refs):
            rows = [self.resolve(ref) for ref in sorted(refs)]
            return {"calls": len(rows), "parent_reused_calls": sum(r.startswith("parent:") for r in refs),
                    "current_calls": sum(r.startswith("current:") for r in refs),
                    "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
                    "generation_seconds": sum(r["generation_seconds"] for r in rows),
                    "truncated": sum(r["truncated"] for r in rows)}
        costs = {m: total(self.used[m]) for m in METHODS}
        costs["_diagnostic"] = total(self.used["diagnostic"])
        costs["_current_physical_total"] = total({"current:" + k for k in self.records})
        costs["_method_refs"] = {m: sorted(self.used[m]) for m in METHODS + ("diagnostic",)}
        costs["_note"] = "Standalone unique request dependencies include historical time when reused; do not sum methods or interpret as new end-to-end latency. Current physical total excludes parent inference."
        return costs
