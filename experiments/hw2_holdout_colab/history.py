"""Authenticate and replay both completed runs; no model weights are used."""
from collections import defaultdict

from .common import HERE, digest, file_hash, read_json, read_jsonl
from .parent_source import parent as pointer_parent, core as gate_core, engine as gate_engine
from .core import old, label_sequences
from .protocol import PROTOCOL


def query_key(messages, maximum, mode="free", paths=None):
    return digest({"messages": messages, "maximum": maximum, "mode": mode,
                   "label_sequences": paths if mode == "constrained" else None})


def authenticate_sources():
    info = read_json(HERE / "source/provenance.json")
    for name, expected in info["snapshot_files"].items():
        if file_hash(HERE / "source" / name) != expected:
            raise RuntimeError("Historical snapshot changed: " + name)
    for name, expected in info["parent_source_hashes"].items():
        if file_hash(HERE / "parent_source" / name) != expected:
            raise RuntimeError("Frozen historical code changed: " + name)
    return info


class History:
    def __init__(self, tok, pointer, gate_rows, gate_setup):
        self.tok, self.setup, self.paths = tok, gate_setup, label_sequences(tok)
        self.pointer_hash, self.gate_hash = pointer.setup_hash, digest(gate_setup)
        self.records = {"pointer:" + k: v for k, v in pointer.records.items()}
        self.records.update({"gate:" + k: v for k, v in gate_rows.items()})
        self.queries, self.by_identity = {}, {}
        for ref, row in self.records.items():
            mode = row.get("mode", "free")
            key = query_key(row["messages"], row["max_new_tokens"], mode, self.paths)
            if key in self.queries:
                other = self.records[self.queries[key]]
                if other["raw_output"] != row["raw_output"] or other["generated_token_ids"] != row["generated_token_ids"]:
                    raise RuntimeError("Conflicting historical outputs for the exact request")
            else:
                self.queries[key] = ref
            self.by_identity[(row["setup_hash"], row["key"])] = ref

    def lookup(self, messages, maximum, mode="free"):
        ref = self.queries.get(query_key(messages, maximum, mode, self.paths))
        return None if ref is None else self.records[ref]

    def reference(self, row):
        return self.by_identity[(row["setup_hash"], row["key"])]


class GateReplay:
    """Use the old pipeline's own reference convention to replay its stored decisions."""
    def __init__(self, history):
        self.history, self.tok, self.used = history, history.tok, defaultdict(set)

    def generate(self, messages, stage, max_new_tokens=None, mode="free"):
        row = self.history.lookup(messages, max_new_tokens or PROTOCOL["generation"]["max_new_tokens"], mode)
        if row is None:
            raise RuntimeError("Historical gate replay missing: " + stage)
        return row

    def reference(self, row):
        return ("parent:" if row["setup_hash"] == self.history.pointer_hash else "current:") + row["key"]


def restore(tok):
    authenticate_sources()
    samples, gold, pointer_decisions, pointer, dependencies = pointer_parent.restore(tok)
    setup = read_json(HERE / "source/gate_setup.json")
    gate_rows = read_json(HERE / "source/gate_checkpoints.json")
    lock = read_json(HERE / "source/gate_protocol_lock.json")
    manifest = read_json(HERE / "source/gate_artifact_manifest.json")
    if digest(lock["material"]) != lock["signature"] or setup["protocol_signature"] != lock["signature"]:
        raise RuntimeError("Historical gate lock failed")
    if manifest["sources"] != lock["material"]["sources"] or manifest["sources"] != read_json(HERE / "source/provenance.json")["parent_source_hashes"]:
        raise RuntimeError("Historical source identity failed")
    if setup["protocol"] != gate_core.PROTOCOL or not setup["reuse_parent"]:
        raise RuntimeError("Unexpected historical protocol")
    if gate_engine.compatibility(setup, pointer.setup):
        raise RuntimeError("Historical runtimes differ")
    # Reuse the exact historical request/hash verifier without creating or writing files.
    verifier = object.__new__(gate_engine.Generator)
    verifier.setup_hash, verifier.paths = digest(setup), label_sequences(tok)
    for key, row in gate_rows.items():
        verifier.verify(row, key)
        if row["system_error"] is not None or row["truncated"]:
            raise RuntimeError("Incomplete historical generation")
        if tok.decode(row["generated_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) != row["raw_output"]:
            raise RuntimeError("Historical gate output token mismatch")
        prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        if len(tok.encode(prompt, add_special_tokens=False)) != row["input_tokens"]:
            raise RuntimeError("Historical gate input length mismatch")
    history = History(tok, pointer, gate_rows, setup)
    replay = GateReplay(history)
    decisions = read_jsonl(HERE / "source/gate_decisions.jsonl")
    for i, (sample, saved) in enumerate(zip(samples, decisions)):
        actual = {k: sample[k] for k in ("id", "base_id", "task", "group", "template", "placement")}
        actual.update(gate_core.decide(old.public_view(sample), pointer_decisions[i], replay, dependencies[sample["id"]]))
        if digest(actual) != digest(saved):
            raise RuntimeError("Historical gate decision replay mismatch")
    if len(decisions) != 480 or len(gate_rows) != 378:
        raise RuntimeError("Historical run is incomplete")
    print("Historical CPU audit: 480 pointer + 480 gate decisions,", len(history.records), "raw records.", flush=True)
    return {"history": history, "samples": samples, "gold": gold, "pointer_decisions": pointer_decisions,
            "gate_decisions": decisions,
            "audit": {"status": "PASS", "pointer_decisions": 480, "gate_decisions": 480,
                      "pointer_raw_records": len(pointer.records), "gate_raw_records": len(gate_rows),
                      "hashes_token_ids_and_input_lengths_verified": True, "new_model_calls": 0}}
