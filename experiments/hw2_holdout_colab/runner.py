"""CPU freeze, Colab inference, checkpoint resume, replay, and self-contained ZIP."""
import os
from pathlib import Path
import time
import zipfile

from . import core, data, history
from .common import HERE, digest, file_hash, now, read_json, read_jsonl, source_hashes, source_paths, write_json, write_jsonl
from .engine import Generator, method_costs, preflight
from .protocol import METHODS, PROTOCOL

META = ("id", "base_id", "split", "task", "group", "template", "placement")


def call_plan(tok, restored, holdout):
    past = restored["history"]
    new_dev = set()
    for sample, previous in zip(restored["samples"], restored["pointer_decisions"]):
        view = core.old.public_view(sample)
        for name in ("standard", "target_only", "id_selection"):
            candidate = core.canonical_candidate(previous["methods"][name])
            if candidate is not None:
                messages = core.old.relation_messages(view, candidate)
                if past.lookup(messages, PROTOCOL["relation_max_new_tokens"], "constrained") is None:
                    new_dev.add(history.query_key(messages, PROTOCOL["relation_max_new_tokens"], "constrained", past.paths))
    baseline = set()
    max_prompt = 0
    holdout_history_hits = 0
    for sample in holdout:
        view = core.old.public_view(sample)
        pointer_messages, spans = core.old.pointer_messages(view)
        requests = [(core.old.make_messages(view), 96), (core.old.target_messages(view), 96)]
        if spans:
            requests.append((pointer_messages, PROTOCOL["pointer_max_new_tokens"]))
        for messages, maximum in requests:
            key = history.query_key(messages, maximum)
            if key not in baseline:
                rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                max_prompt = max(max_prompt, len(tok.encode(rendered, add_special_tokens=False)))
                holdout_history_hits += past.lookup(messages, maximum) is not None
                baseline.add(key)
    if holdout_history_hits or max_prompt > PROTOCOL["max_prompt_tokens"]:
        raise RuntimeError("Holdout overlap or prompt budget violation")
    # Each unique candidate-generation request can require at most one new gate request.
    holdout_upper = 2 * len(baseline)
    dev_baseline = set()
    for sample in restored["samples"]:
        view = core.old.public_view(sample)
        pm, spans = core.old.pointer_messages(view)
        requests = [(core.old.make_messages(view), 96), (core.old.target_messages(view), 96)]
        if spans:
            requests.append((pm, 16))
        dev_baseline.update(history.query_key(msg, maximum) for msg, maximum in requests)
    if len(new_dev) + holdout_upper > PROTOCOL["max_calls_reuse"] or 2 * len(dev_baseline) + holdout_upper > PROTOCOL["max_calls_refresh"]:
        raise RuntimeError("Protocol call cap is smaller than the structural upper bound")
    return {"development_inputs": 480, "new_development_gate_calls_exact_if_reuse": len(new_dev),
            "holdout_inputs": len(holdout), "holdout_bases": PROTOCOL["expected_holdout_bases"],
            "holdout_unique_candidate_calls": len(baseline), "holdout_total_calls_upper_bound": holdout_upper,
            "reuse_total_new_calls_upper_bound": len(new_dev) + holdout_upper,
            "refresh_total_new_calls_upper_bound": 2 * len(dev_baseline) + holdout_upper,
            "maximum_candidate_prompt_tokens": max_prompt, "holdout_history_request_hits": holdout_history_hits,
            "note": "Development reuse count is exact. Holdout gate candidates are unknown until inference; counts are conservative bounds, not runtime estimates."}


def check_lock(root, lock):
    if source_hashes() != lock["material"]["sources"]:
        raise RuntimeError("Source changed after freeze; choose a new RUN_DIR")
    for name, value in lock["files"].items():
        if not (root / name).is_file() or file_hash(root / name) != value:
            raise RuntimeError("Locked input changed: " + name)


def prepare(run_dir, cache_dir="/content/hw2_value_tracking_model_cache", reuse_parent=True, local_files_only=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PROTOCOL["model"], revision=PROTOCOL["revision"], cache_dir=str(cache_dir),
                                        local_files_only=local_files_only, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    restored = history.restore(tok)
    heldout, held_gold, held_audit = data.build_holdout(tok, restored["samples"])
    development = [{k: v for k, v in s.items() if k != "messages"} | {"split": "development"} for s in restored["samples"]]
    samples, gold = development + heldout, restored["gold"] | held_gold
    if len({s["id"] for s in samples}) != len(samples) or len(gold) != len(samples):
        raise RuntimeError("Duplicate evaluation IDs")
    plan = call_plan(tok, restored, heldout)
    material = {"protocol": PROTOCOL, "sources": source_hashes(), "samples": digest(samples), "gold": digest(gold),
                "reuse_parent": bool(reuse_parent), "chat_template": digest(tok.chat_template),
                "label_sequences": core.label_sequences(tok)}
    signature, root = digest(material), Path(run_dir)
    if (root / "protocol_lock.json").exists():
        lock = read_json(root / "protocol_lock.json")
        if lock["signature"] != signature:
            raise RuntimeError("Different data/code/settings in RUN_DIR; choose a fresh directory")
        check_lock(root, lock)
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("RUN_DIR has files without a protocol lock")
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "protocol.json", PROTOCOL)
        write_jsonl(root / "public_samples.jsonl", samples)
        write_json(root / "gold.json", gold)
        write_json(root / "holdout_data_audit.json", held_audit)
        write_json(root / "historical_replay_audit.json", restored["audit"])
        write_json(root / "call_plan.json", plan)
        names = ("protocol.json", "public_samples.jsonl", "gold.json", "holdout_data_audit.json", "historical_replay_audit.json", "call_plan.json")
        lock = {"signature": signature, "material": material, "files": {n: file_hash(root / n) for n in names}, "frozen_utc": now()}
        write_json(root / "protocol_lock.json", lock)
    print("Inputs and rules frozen before model loading:", len(development), "development +", len(heldout), "holdout |", signature[:12], flush=True)
    return {"root": root, "tok": tok, "samples": samples, "gold": gold, "restored": restored,
            "history": restored["history"], "lock": lock, "reuse_parent": bool(reuse_parent), "plan": plan}


def record_for(sample, generator):
    result = {k: sample[k] for k in META} | core.decide(core.old.public_view(sample), generator)
    if sample["split"].startswith("holdout"):
        if any(not ref.startswith("current:") for refs in result["dependencies"].values() for ref in refs):
            raise RuntimeError("Historical model output was used for a heldout input")
    return result


def audit_saved_run(prepared, setup):
    root, tok = prepared["root"], prepared["tok"]
    check_lock(root, prepared["lock"])
    replay = Generator(None, tok, root, setup, None, prepared["history"], replay_only=True)
    saved = read_jsonl(root / "decisions.jsonl")
    if len(saved) != len(prepared["samples"]):
        raise RuntimeError("Cannot audit partial run as complete")
    for sample, row in zip(prepared["samples"], saved):
        if digest(record_for(sample, replay)) != digest(row):
            raise RuntimeError("Decision replay mismatch: " + sample["id"])
    if replay.seen_current != set(replay.records):
        raise RuntimeError("Unused current raw checkpoint")
    if method_costs(saved, replay) != read_json(root / "method_costs.json"):
        raise RuntimeError("Split-specific cost replay mismatch")
    for row in replay.records.values():
        if tok.decode(row["generated_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) != row["raw_output"]:
            raise RuntimeError("Output token decode mismatch")
        rendered = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        if len(tok.encode(rendered, add_special_tokens=False)) != row["input_tokens"] or row["output_tokens"] != len(row["generated_token_ids"]):
            raise RuntimeError("Token length mismatch")
    write_json(root / "replay_audit.json", {"status": "PASS", "decisions_replayed": len(saved),
               "development_inputs": 480, "heldout_inputs": PROTOCOL["expected_holdout_inputs"],
               "current_raw_records": len(replay.records), "tokens_and_grammar_verified": True,
               "costs_and_no_orphans_verified": True, "holdout_never_used_historical_outputs": True,
               "new_model_calls": 0})
    print("Final raw-output replay: PASS", flush=True)


def run(run_dir, cache_dir="/content/hw2_value_tracking_model_cache", reuse_parent=True, prepared=None):
    prepared = prepared or prepare(run_dir, cache_dir, reuse_parent)
    root, tok = prepared["root"], prepared["tok"]
    if root.resolve() != Path(run_dir).resolve() or prepared["reuse_parent"] != bool(reuse_parent):
        raise RuntimeError("Prepared settings mismatch")
    check_lock(root, prepared["lock"])
    torch, runtime = preflight(tok, prepared["history"].setup, reuse_parent)
    setup = runtime | {"protocol_signature": prepared["lock"]["signature"], "protocol": PROTOCOL,
                       "reuse_parent": bool(reuse_parent), "label_sequences": core.label_sequences(tok)}
    if (root / "setup.json").exists() and read_json(root / "setup.json") != setup:
        raise RuntimeError("Runtime changed; choose a fresh RUN_DIR")
    if not (root / "setup.json").exists() and any((root / "checkpoints").glob("*.json")):
        raise RuntimeError("Checkpoints exist without setup")
    write_json(root / "setup.json", setup)
    write_json(root / "status.json", {"status": "RUNNING", "started_utc": now()})
    from transformers import AutoModelForCausalLM
    torch.manual_seed(PROTOCOL["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, generator, records = None, None, []
    try:
        model = AutoModelForCausalLM.from_pretrained(PROTOCOL["model"], revision=PROTOCOL["revision"], cache_dir=str(cache_dir),
                    torch_dtype=torch.float16, attn_implementation=PROTOCOL["attention"], low_cpu_mem_usage=True, trust_remote_code=False).eval().to("cuda:0")
        load_seconds = time.perf_counter() - started
        generator = Generator(model, tok, root, setup, torch, prepared["history"])
        for i, sample in enumerate(prepared["samples"]):
            records.append(record_for(sample, generator))
            if i == 0 or (i + 1) % 24 == 0 or i + 1 == len(prepared["samples"]):
                write_jsonl(root / "decisions.jsonl", records)
                write_json(root / "progress.json", {"complete": i + 1, "total": len(prepared["samples"]),
                           "split": sample["split"], "current_unique_calls": len(generator.records), "new_calls_this_session": generator.new_calls, "updated_utc": now()})
                print("Inputs", i + 1, "/", len(prepared["samples"]), "|", sample["split"], "| new calls", generator.new_calls, flush=True)
            if (i + 1) % 240 == 0:
                check_lock(root, prepared["lock"])
        write_json(root / "method_costs.json", method_costs(records, generator))
        sessions = read_json(root / "runtime_sessions.json") if (root / "runtime_sessions.json").exists() else []
        sessions.append({"finished_utc": now(), "model_load_seconds": load_seconds, "session_seconds": time.perf_counter() - started,
                         "new_calls": generator.new_calls, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        write_json(root / "runtime_sessions.json", sessions)
        write_json(root / "status.json", {"status": "INFERENCE_COMPLETE"})
    except BaseException as exc:
        write_jsonl(root / "decisions.jsonl", records)
        if generator is not None:
            write_json(root / "method_costs.json", method_costs(records, generator))
        write_json(root / "status.json", {"status": "INTERRUPTED", "complete": len(records), "error_type": type(exc).__name__, "error": str(exc), "at_utc": now()})
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
            package.write(path, "reproduce/experiments/hw2_holdout_colab/" + str(path.relative_to(HERE)))
        package.writestr("reproduce/experiments/__init__.py", '"""Reproducible experiments."""\n')
    os.replace(temporary, target)
    print("RESULT_ZIP:", target, flush=True)
    return str(target)
