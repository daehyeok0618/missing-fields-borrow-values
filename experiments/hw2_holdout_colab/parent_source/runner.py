"""Prepare on CPU, run only in Colab, resume, replay and package results."""
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import zipfile

from . import core, parent
from .common import HERE, digest, file_hash, now, read_json, read_jsonl, source_hashes, source_paths, write_json, write_jsonl
from .engine import Generator, preflight, query_key
from .protocol import METHODS, PROTOCOL
from .upstream import core as old

VARIANTS = ("numeric", "clause", "numeric_dash")
META = ("id", "base_id", "task", "group", "template", "placement")


def call_plan(tok, samples, decisions, replay, diagnostics):
    historical = {query_key(r["messages"], r["max_new_tokens"]) for r in replay.records.values()}
    gates, new_free, constrained, diagnostic = set(), set(), set(), set()
    max_tokens = 0

    def check(messages):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tok.encode(prompt, add_special_tokens=False))

    for sample, decision in zip(samples, decisions):
        for method in ("standard", "id_selection"):
            candidate = core.canonical_candidate(decision["methods"][method])
            if candidate is None:
                continue
            messages = old.relation_messages(old.public_view(sample), candidate)
            key = query_key(messages, PROTOCOL["relation_max_new_tokens"])
            if key not in gates:
                max_tokens = max(max_tokens, check(messages))
            gates.add(key)
            constrained.add(key)
            if key not in historical:
                new_free.add(key)
    for sample in diagnostics:
        for variant in VARIANTS:
            messages, _ = core.diagnostic_messages(old.public_view(sample), variant)
            key = query_key(messages, PROTOCOL["pointer_max_new_tokens"])
            diagnostic.add(key)
            max_tokens = max(max_tokens, check(messages))
            if key in historical or key in gates:
                raise RuntimeError("Unexpected diagnostic/primary request overlap")
    total = len(new_free) + len(constrained) + len(diagnostic)
    if total > PROTOCOL["max_calls_reuse"] or max_tokens > PROTOCOL["max_prompt_tokens"]:
        raise RuntimeError("Prepared requests exceed fixed budget")
    return {"parent_decisions": len(decisions), "parent_raw_records_authenticated": len(replay.records),
            "unique_normalized_gate_prompts": len(gates), "free_gate_parent_hits": len(gates) - len(new_free),
            "new_free_gate_calls": len(new_free), "new_constrained_gate_calls": len(constrained),
            "diagnostic_requests": len(diagnostics) * len(VARIANTS), "diagnostic_unique_calls": len(diagnostic),
            "reuse_new_calls_exact": total, "maximum_new_prompt_tokens": max_tokens,
            "refresh_calls_estimate_from_parent_outputs": len(replay.records) + total,
            "note": "Reuse counts are exact request counts, not measured time. Refresh candidates may differ; hard cap is 4000. All data are development."}


def check_lock(root, lock):
    if source_hashes() != lock["material"]["sources"]:
        raise RuntimeError("Source changed; use a new RUN_DIR")
    for name, expected in lock["files"].items():
        if not (root / name).is_file() or file_hash(root / name) != expected:
            raise RuntimeError("Locked input changed: " + name)


def prepare(run_dir, cache_dir="/content/hw2_value_tracking_model_cache", reuse_parent=True, local_files_only=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PROTOCOL["model"], revision=PROTOCOL["revision"],
                                        cache_dir=str(cache_dir), local_files_only=local_files_only, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    samples, gold, decisions, replay, deps = parent.restore(tok)
    diagnostics, diagnostic_gold = parent.diagnostic_dataset(samples, gold)
    plan = call_plan(tok, samples, decisions, replay, diagnostics)
    material = {"protocol": PROTOCOL, "sources": source_hashes(), "samples": digest(samples), "gold": digest(gold),
                "diagnostics": digest(diagnostics), "diagnostic_gold": digest(diagnostic_gold),
                "reuse_parent": bool(reuse_parent), "chat_template": digest(tok.chat_template), "label_sequences": core.label_sequences(tok)}
    signature, root = digest(material), Path(run_dir)
    if (root / "protocol_lock.json").exists():
        lock = read_json(root / "protocol_lock.json")
        if signature != lock["signature"]:
            raise RuntimeError("RUN_DIR has different settings/code. Use a new directory.")
        check_lock(root, lock)
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("RUN_DIR has files without protocol_lock.json")
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "protocol.json", PROTOCOL)
        write_jsonl(root / "public_samples.jsonl", samples)
        write_json(root / "gold.json", gold)
        write_jsonl(root / "diagnostic_samples.jsonl", diagnostics)
        write_json(root / "diagnostic_gold.json", diagnostic_gold)
        write_json(root / "call_plan.json", plan)
        write_json(root / "parent_replay_audit.json", {"status": "PASS", "decisions_replayed": 480,
                   "raw_records": len(replay.records), "original_hashes_verified": True,
                   "input_lengths_and_output_token_ids_verified": True, "new_model_calls": 0})
        names = ("protocol.json", "public_samples.jsonl", "gold.json", "diagnostic_samples.jsonl",
                 "diagnostic_gold.json", "call_plan.json", "parent_replay_audit.json")
        lock = {"signature": signature, "material": material, "files": {n: file_hash(root / n) for n in names}, "created_utc": now()}
        write_json(root / "protocol_lock.json", lock)
    print("CPU preparation complete. New calls with reuse:", plan["reuse_new_calls_exact"], "| no model loaded", flush=True)
    return {"tok": tok, "samples": samples, "gold": gold, "parent_decisions": decisions, "parent": replay,
            "parent_dependencies": deps, "diagnostics": diagnostics, "diagnostic_gold": diagnostic_gold,
            "lock": lock, "reuse_parent": bool(reuse_parent), "root": root}


def baseline_for(sample, index, prepared, generator):
    if prepared["reuse_parent"]:
        return prepared["parent_decisions"][index], prepared["parent_dependencies"][sample["id"]]
    generator.last_used = defaultdict(set)
    baseline = old.decide(old.public_view(sample), generator)
    return baseline, {m: sorted(generator.last_used[m]) for m in METHODS if m in baseline["methods"]}


def primary_record(sample, index, prepared, generator):
    baseline, deps = baseline_for(sample, index, prepared, generator)
    return {k: sample[k] for k in META} | core.decide(old.public_view(sample), baseline, generator, deps)


def diagnostic_record(sample, variant, generator):
    messages, spans = core.diagnostic_messages(old.public_view(sample), variant)
    row = generator.generate(messages, "diagnostic_" + variant, PROTOCOL["pointer_max_new_tokens"])
    parsed, status, selected = old.parse_selection(row["raw_output"], spans, row["truncated"])
    generator.attribute("diagnostic", row)
    return {"id": sample["id"], "base_id": sample["base_id"], "task": sample["task"],
            "family": sample["diagnostic_family"], "condition": sample["diagnostic_condition"], "variant": variant,
            "decision": parsed, "selection_status": status, "selected": selected, "ref": generator.reference(row)}


def audit_saved_run(prepared, setup):
    root, tok = prepared["root"], prepared["tok"]
    check_lock(root, prepared["lock"])
    replay = Generator(None, tok, root, setup, None, prepared["parent"], replay_only=True)
    saved = read_jsonl(root / "decisions.jsonl")
    diag_saved = read_jsonl(root / "diagnostic_decisions.jsonl")
    if len(saved) != len(prepared["samples"]) or len(diag_saved) != PROTOCOL["expected_diagnostic_requests"]:
        raise RuntimeError("Partial run cannot pass the final audit")
    for i, sample in enumerate(prepared["samples"]):
        if digest(primary_record(sample, i, prepared, replay)) != digest(saved[i]):
            raise RuntimeError("Primary decision replay mismatch")
    index = 0
    for sample in prepared["diagnostics"]:
        for variant in VARIANTS:
            if digest(diagnostic_record(sample, variant, replay)) != digest(diag_saved[index]):
                raise RuntimeError("Diagnostic decision replay mismatch")
            index += 1
    if replay.costs() != read_json(root / "method_costs.json"):
        raise RuntimeError("Method cost attribution replay mismatch")
    if replay.seen_current != set(replay.records):
        raise RuntimeError("Unreferenced current raw checkpoint detected")
    for row in replay.records.values():
        if tok.decode(row["generated_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) != row["raw_output"]:
            raise RuntimeError("New output token decode mismatch")
        prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        if len(tok.encode(prompt, add_special_tokens=False)) != row["input_tokens"]:
            raise RuntimeError("New input token count mismatch")
    write_json(root / "replay_audit.json", {"status": "PASS", "primary_replayed": len(saved), "diagnostic_replayed": index,
               "current_checkpoints": len(replay.records), "grammar_and_hashes_verified": True,
               "token_ids_and_input_lengths_verified": True, "costs_replayed": True,
               "no_orphan_current_checkpoints": True, "new_model_calls": 0})
    print("New raw-output replay audit: PASS", flush=True)


def run(run_dir, cache_dir="/content/hw2_value_tracking_model_cache", reuse_parent=True, prepared=None):
    prepared = prepared or prepare(run_dir, cache_dir, reuse_parent)
    root, tok = prepared["root"], prepared["tok"]
    if root.resolve() != Path(run_dir).resolve() or prepared["reuse_parent"] != bool(reuse_parent):
        raise RuntimeError("Prepared inputs do not match requested run")
    check_lock(root, prepared["lock"])
    # All checks precede weight loading. CPU prepare and audit never load a model.
    torch, runtime = preflight(tok, prepared["parent"].setup, reuse_parent)
    setup = runtime | {"protocol_signature": prepared["lock"]["signature"], "protocol": PROTOCOL,
                       "reuse_parent": bool(reuse_parent), "label_sequences": core.label_sequences(tok)}
    if (root / "setup.json").exists() and read_json(root / "setup.json") != setup:
        raise RuntimeError("Runtime changed; use a new RUN_DIR")
    if not (root / "setup.json").exists() and any((root / "checkpoints").glob("*.json")):
        raise RuntimeError("Checkpoints found without setup")
    write_json(root / "setup.json", setup)
    write_json(root / "status.json", {"status": "RUNNING", "started_utc": now()})
    from transformers import AutoModelForCausalLM
    torch.manual_seed(PROTOCOL["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, generator = None, None
    decisions, diagnostics = [], []
    try:
        model = AutoModelForCausalLM.from_pretrained(PROTOCOL["model"], revision=PROTOCOL["revision"],
                    cache_dir=str(cache_dir), torch_dtype=torch.float16, attn_implementation=PROTOCOL["attention"],
                    low_cpu_mem_usage=True, trust_remote_code=False).eval().to("cuda:0")
        loaded = time.perf_counter() - started
        generator = Generator(model, tok, root, setup, torch, prepared["parent"])
        for i, sample in enumerate(prepared["samples"]):
            decisions.append(primary_record(sample, i, prepared, generator))
            if i == 0 or (i + 1) % 16 == 0 or i + 1 == len(prepared["samples"]):
                write_jsonl(root / "decisions.jsonl", decisions)
                write_json(root / "method_costs.json", generator.costs())
                write_json(root / "progress.json", {"phase": "primary", "complete": i + 1, "total": 480,
                           "new_calls_this_session": generator.new_calls, "updated_utc": now()})
                print("Primary inputs:", i + 1, "/ 480 | new calls:", generator.new_calls, flush=True)
        for sample in prepared["diagnostics"]:
            for variant in VARIANTS:
                diagnostics.append(diagnostic_record(sample, variant, generator))
            write_jsonl(root / "diagnostic_decisions.jsonl", diagnostics)
            print("Diagnostic requests:", len(diagnostics), "/ 96", flush=True)
        costs = generator.costs()
        write_json(root / "method_costs.json", costs)
        sessions = read_json(root / "runtime_sessions.json") if (root / "runtime_sessions.json").exists() else []
        sessions.append({"finished_utc": now(), "model_load_seconds": loaded,
                         "session_seconds": time.perf_counter() - started, "new_calls": generator.new_calls,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        write_json(root / "runtime_sessions.json", sessions)
        write_json(root / "status.json", {"status": "INFERENCE_COMPLETE"})
    except BaseException as exc:
        write_jsonl(root / "decisions.jsonl", decisions)
        write_jsonl(root / "diagnostic_decisions.jsonl", diagnostics)
        if generator is not None:
            write_json(root / "method_costs.json", generator.costs())
        write_json(root / "status.json", {"status": "INTERRUPTED", "complete": len(decisions),
                   "diagnostic_complete": len(diagnostics), "error_type": type(exc).__name__, "error": str(exc), "at_utc": now()})
        raise
    finally:
        if generator is not None:
            generator.model = None
        del model
        torch.cuda.empty_cache()
    audit_saved_run(prepared, setup)
    from .analysis import analyze
    summary = analyze(root)
    write_json(root / "status.json", {"status": "COMPLETE", "finished_utc": now()})
    return summary


def archive(run_dir, destination=None):
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    target = Path(destination) if destination else root.parent / (root.name + "_results.zip")
    if root.resolve() in target.resolve().parents or root.resolve() == target.resolve():
        raise ValueError("ZIP must be outside RUN_DIR")
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json" and not p.name.endswith(".tmp")]
    write_json(root / "artifact_manifest.json", {"files": {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}, "sources": source_hashes()})
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as package:
        for path in sorted(files + [root / "artifact_manifest.json"]):
            package.write(path, "results/" + str(path.relative_to(root)))
        for path in source_paths():
            package.write(path, "reproduce/experiments/hw2_relation_gate_colab/" + str(path.relative_to(HERE)))
        package.writestr("reproduce/experiments/__init__.py", '"""Reproducible experiments."""\n')
    os.replace(temporary, target)
    print("RESULT_ZIP:", target, flush=True)
    return str(target)
