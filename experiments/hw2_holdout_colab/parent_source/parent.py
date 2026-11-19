"""Restore and authenticate historical raw records without loading model weights."""
import copy
from collections import defaultdict

from .common import HERE, digest, file_hash, read_json, read_jsonl
from .protocol import METHODS, PROTOCOL
from .upstream import core as old


def authenticate_sources():
    info = read_json(HERE / "source/provenance.json")
    for name, expected in info["snapshot_files"].items():
        if file_hash(HERE / "source" / name) != expected:
            raise RuntimeError("Parent snapshot changed: " + name)
    for name, expected in info["upstream_source_hashes"].items():
        if file_hash(HERE / "upstream" / name) != expected:
            raise RuntimeError("Frozen parent code changed: " + name)
    return info


class ParentReplay:
    def __init__(self, tok):
        authenticate_sources()
        self.tok = tok
        self.setup = read_json(HERE / "source/parent_setup.json")
        self.setup_hash = digest(self.setup)
        self.compact = read_json(HERE / "source/parent_checkpoints_compact.json")
        self.records, self.used = {}, defaultdict(set)

    def generate(self, messages, stage, max_new_tokens=None):
        maximum = max_new_tokens or PROTOCOL["generation"]["max_new_tokens"]
        key = digest({"messages": messages, "maximum": maximum, "setup": self.setup_hash})
        if key not in self.compact:
            raise RuntimeError("Missing historical raw request: " + stage)
        if key not in self.records:
            row = copy.deepcopy(self.compact[key])
            row["messages"] = messages
            if row["key"] != key or row["setup_hash"] != self.setup_hash or row["record_hash"] != digest({k: v for k, v in row.items() if k != "record_hash"}):
                raise RuntimeError("Historical raw record hash failed")
            self.records[key] = row
        return self.records[key]

    def attribute(self, method, *rows):
        self.used[method].update("parent:" + r["key"] for r in rows)


def restore(tok, verify_tokens=True):
    replay = ParentReplay(tok)
    lock = read_json(HERE / "source/parent_lock.json")
    samples = read_jsonl(HERE / "source/parent_samples_compact.jsonl")
    for sample in samples:
        sample["messages"] = old.make_messages(old.public_view(sample))
    gold = read_json(HERE / "source/parent_gold.json")
    decisions = read_jsonl(HERE / "source/parent_decisions.jsonl")
    if digest(samples) != lock["material"]["samples"] or digest(gold) != lock["material"]["gold"]:
        raise RuntimeError("Historical data reconstruction failed")
    if digest(tok.chat_template) != replay.setup["chat_template_hash"]:
        raise RuntimeError("Historical tokenizer template differs")
    dependencies = {}
    for i, (sample, saved) in enumerate(zip(samples, decisions)):
        replay.used.clear()
        actual = {k: sample[k] for k in ("id", "base_id", "task", "group", "template", "placement")}
        actual.update(old.decide(old.public_view(sample), replay))
        if digest(actual) != digest(saved):
            raise RuntimeError("Historical decision replay mismatch: " + sample["id"])
        dependencies[sample["id"]] = {m: sorted(replay.used[m]) for m in METHODS if m in saved["methods"]}
        if i == 0 or (i + 1) % 120 == 0:
            print("Parent CPU replay:", i + 1, "/", len(samples), flush=True)
    if len(samples) != 480 or len(decisions) != 480 or len(replay.records) != len(replay.compact):
        raise RuntimeError("Incomplete parent replay")
    if verify_tokens:
        for row in replay.records.values():
            if tok.decode(row["generated_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) != row["raw_output"]:
                raise RuntimeError("Historical output token mismatch")
            prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            if len(tok.encode(prompt, add_special_tokens=False)) != row["input_tokens"]:
                raise RuntimeError("Historical input token count mismatch")
    return samples, gold, decisions, replay, dependencies


def diagnostic_dataset(samples, gold):
    """All KV families, one placement, paired missing/present; labels remain separate."""
    rows, labels = [], {}
    by_key = {(s["base_id"], s["group"], s["template"], s["placement"]): s for s in samples}
    for sample in samples:
        if sample["template"] != "KV" or sample["placement"] != "MIDDLE" or sample["group"] not in ("otherfield_missing", "multi_missing"):
            continue
        for condition in ("missing", "present"):
            row = copy.deepcopy(sample)
            if condition == "present" and sample["group"] == "multi_missing":
                row = copy.deepcopy(by_key[(sample["base_id"], "multi_present", "KV", "MIDDLE")])
                value = str(gold[row["id"]]["value"])
            elif condition == "present":
                target = next(d for d in row["docs"] if d["id"] == row["target_id"])
                lines = target["text"].splitlines(keepends=True)
                indices = [i for i, line in enumerate(lines) if old.numeric_spans(line)]
                if len(indices) != 1:
                    raise RuntimeError("Expected one other-field numeric record")
                index = indices[0]
                head, sep, tail = lines[index].partition(":")
                if not sep:
                    raise RuntimeError("Expected KV label separator")
                label = "Recorded salary" if row["task"] == "salary" else "Recorded founding year"
                lines[index] = label + sep + tail
                target["text"] = "".join(lines)
                value = old.numeric_spans(target["text"])[0]["value"]
            else:
                value = None
            row["id"] = sample["base_id"] + "_" + sample["group"] + "_" + condition
            # Diagnostic metadata is never included in method messages.
            row["diagnostic_condition"] = condition
            row["diagnostic_family"] = sample["group"]
            row.pop("messages", None)
            labels[row["id"]] = {"value": value}
            rows.append(row)
    if len(rows) != PROTOCOL["expected_diagnostic_cases"] or len(labels) != len(rows):
        raise RuntimeError("Diagnostic size mismatch")
    return rows, labels
