from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone

import torch
import transformers
from transformers import AutoModelForCausalLM

from .common import ROOT, cfg, offline, parse, read_jsonl, render, run_dir, semantic_parse, stable_hash, tokenizer, verify_lock, write_json, write_jsonl


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "main"), required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    lock = verify_lock(); c = cfg(); offline()
    if args.split == "main":
        pilot = json.loads((run_dir() / "pilot_audit.json").read_text())
        if not pilot["main_allowed"]:
            raise RuntimeError("Development execution/capability gate did not pass; preserve diagnostic and stop")
    requests = read_jsonl(ROOT / f"data/{args.split}_requests.jsonl")
    request_map = {r["request_id"]: r for r in requests}
    expected = c["data"][args.split + "_calls"]
    if len(requests) != len(request_map) or len(requests) != expected:
        raise RuntimeError("Unexpected requests")
    output_path = run_dir() / f"{args.split}_raw.jsonl"
    prior = read_jsonl(output_path); existing = {r["request_id"]: r for r in prior}
    if len(prior) != len(existing) or set(existing) - set(request_map):
        raise RuntimeError("Unexpected checkpoint rows")
    for rid, row in existing.items():
        if row["input_hash"] != request_map[rid]["input_hash"] or parse(row["raw_output"]) != row["parse"]:
            raise RuntimeError("Checkpoint input/parse mismatch")
    tok = tokenizer(); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(20261102)
    torch.backends.cuda.matmul.allow_tf32 = False
    m = c["model"]; started = now(); process_tick = time.perf_counter(); tick = process_tick
    model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], cache_dir=m["cache_dir"],
            local_files_only=True, torch_dtype=torch.float16, attn_implementation=m["attention"],
            low_cpu_mem_usage=True).eval().to(args.device)
    load_seconds = time.perf_counter() - tick
    gen = copy.deepcopy(model.generation_config)
    for key, value in c["generation"].items():
        setattr(gen, key, value)
    gen.temperature = gen.top_p = gen.top_k = None
    gen.pad_token_id = tok.pad_token_id
    setup = {"model": m, "generation": gen.to_dict(), "chat_template": tok.chat_template,
             "torch": torch.__version__, "transformers": transformers.__version__,
             "dtype": "float16", "constraint": "none", "protocol_hashes": stable_hash(lock["hashes"])}
    setup_hash = stable_hash(setup)
    if any(row["setup_hash"] != setup_hash for row in prior):
        raise RuntimeError("Checkpoint setup mismatch")
    write_json(run_dir() / f"{args.split}_setup.json", setup)
    # Keep batch membership fixed when resuming to avoid numerical differences.
    ordered = sorted(requests, key=lambda r: (r["input_token_count"], r["request_id"]))
    batch_size = m["batch_size"]
    eos = gen.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    inference_seconds = 0.0; new_count = 0
    runtime_path = run_dir() / f"{args.split}_runtime.json"
    previous_runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    process_id = now().replace(":", "-")
    try:
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            if all(r["request_id"] in existing for r in batch):
                continue
            if any(r["request_id"] in existing for r in batch):
                raise RuntimeError("Partial batch checkpoint: preserve and repair explicitly before resuming")
            prompts = [render(r["messages"], tok) for r in batch]
            inputs = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(args.device)
            if inputs.attention_mask.sum(1).tolist() != [r["input_token_count"] for r in batch]:
                raise RuntimeError("Token count changed")
            torch.cuda.synchronize(); tick = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=gen)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - tick
            inference_seconds += elapsed
            generated = output[:, inputs.input_ids.shape[1]:]
            records = []
            for request, all_ids in zip(batch, generated.tolist()):
                first_eos = next((j for j, token in enumerate(all_ids) if token in eos_ids), None)
                ids = all_ids if first_eos is None else all_ids[:first_eos + 1]
                raw = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                records.append({"request_id": request["request_id"], "base_id": request["base_id"],
                                "input_hash": request["input_hash"], "setup_hash": setup_hash,
                                "raw_output": raw, "generated_token_ids": ids, "parse": parse(raw),
                                "input_token_count": request["input_token_count"], "output_token_count": len(ids),
                                "truncated": first_eos is None and len(ids) >= gen.max_new_tokens,
                                "system_error": None, "retry_count": 0, "batch_index": start // batch_size,
                                "batch_seconds": elapsed, "batch_size": len(batch), "finished_utc": now()})
            write_jsonl(output_path, records, append=True)
            existing.update({r["request_id"]: r for r in records}); new_count += len(records)
            if start % (10 * batch_size) == 0 or start + batch_size >= len(ordered):
                print(json.dumps({"split": args.split, "complete": len(existing), "total": expected,
                                  "generation_seconds": round(inference_seconds, 2)}), flush=True)
                write_json(run_dir() / f"{args.split}_progress.json", {"completed": len(existing), "total": expected,
                            "updated_utc": now(), "generation_seconds_this_process": inference_seconds})
    except Exception as exc:
        write_json(run_dir() / f"incident_{args.split}_{process_id}.json", {
            "error": repr(exc), "completed": len(existing), "at_utc": now(), "output_preserved": True})
        raise
    verify_lock()
    rows = list(existing.values())
    runtime = {"split": args.split, "status": "COMPLETE", "started_utc": started, "finished_utc": now(),
               "requests": expected, "completed": len(rows), "new_requests": new_count,
               "model_load_seconds": load_seconds, "generation_seconds_this_process": inference_seconds,
               "generation_seconds": sum({r['batch_index']: r['batch_seconds'] for r in rows}.values()),
               "process_seconds": time.perf_counter() - process_tick,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "gpu": torch.cuda.get_device_name(), "device": args.device, "gpu_total_bytes": torch.cuda.get_device_properties(torch.device(args.device)).total_memory,
               "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
               "cuda_build": torch.version.cuda, "setup_hash": setup_hash,
               "strict_valid": sum(r["parse"]["valid"] for r in rows),
               "parse_errors": dict(Counter(r["parse"]["error"] for r in rows if not r["parse"]["valid"])),
               "truncated": sum(r["truncated"] for r in rows), "system_errors": 0,
               "input_tokens": sum(r["input_token_count"] for r in rows), "output_tokens": sum(r["output_token_count"] for r in rows)}
    write_json(runtime_path, runtime)
    if args.split == "dev":
        controls = {}
        for family in ("T0", "T1", "PRESENT"):
            selected = [r for r in requests if r["family"] == family]
            correct = sum(existing[r["request_id"]]["parse"]["valid"] and
                          existing[r["request_id"]]["parse"]["normalized_value"] == r["gold_value"] for r in selected)
            sem_correct = sum(existing[r['request_id']]['parse']['valid'] and
                              semantic_parse(existing[r['request_id']]['parse'])['normalized_value'] == r['gold_value'] for r in selected)
            controls[family] = {"correct": correct, "n": len(selected), "accuracy": correct / len(selected),
                                "semantic_correct": sem_correct, "semantic_accuracy": sem_correct / len(selected)}
        projected = runtime['generation_seconds'] / len(rows) * c["data"]["main_calls"]
        technical = len(rows) == expected and runtime['strict_valid']/len(rows) >= c['gates']['min_strict_valid'] and runtime['truncated'] == 0
        capability = (controls['T0']['semantic_accuracy'] >= c['gates']['min_t0_semantic_accuracy'] and
                      controls['T1']['accuracy'] >= c['gates']['min_t1_accuracy'])
        pilot = {"technical_pass": technical, "capability_pass": capability,
                 "main_allowed": technical and capability and projected <= c['gates']['max_projected_run_seconds'],
                 "complete": len(rows), "strict_valid": runtime["strict_valid"], "truncated": runtime["truncated"],
                 "controls": controls, "projected_main_seconds": projected, "under_two_hours": projected <= 7200,
                 "behavioral_effects_not_used_for_selection": True, "protocol_changed": False,
                 "note": "T0 semantic and T1 strict capability gates fixed before inference; no selection by donor effects, and no automatic repair after failure."}
        write_json(run_dir() / "pilot_audit.json", pilot)
        print(json.dumps(pilot, indent=2), flush=True)
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    main()
