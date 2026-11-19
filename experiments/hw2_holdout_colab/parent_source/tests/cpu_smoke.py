"""Full pipeline smoke test with explicit fixture outputs, never model inference.

All generated fixture files live in a temporary directory and are discarded.
They are software checks, not measurements of the proposed method.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from .. import core
from ..analysis import analyze
from ..common import digest, file_hash, read_json, write_json, write_jsonl
from ..engine import Generator, query_key, request_key
from ..protocol import PROTOCOL
from ..runner import VARIANTS, archive, audit_saved_run, diagnostic_record, prepare, primary_record
from ..upstream import core as old


def fixture_row(generator, messages, maximum, mode):
    """Fixed hash-based fixture labels; does not receive or read gold."""
    key = request_key(messages, maximum, mode, generator.paths, generator.setup_hash)
    number = int(key[:8], 16)
    if maximum == PROTOCOL["pointer_max_new_tokens"]:
        raw = ("NONE", "n01", "INVALID")[number % 3]
    elif mode == "constrained":
        raw = ("SUPPORTED", "UNSUPPORTED")[number % 2]
    else:
        raw = ("SUPPORTED", "UNSUPPORTED", "SUPPORTING")[number % 3]
    ids = generator.tok.encode(raw, add_special_tokens=False) + [generator.tok.eos_token_id]
    prompt = generator.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    row = {"key": key, "setup_hash": generator.setup_hash, "stage": "EXPLICIT_CPU_TEST_FIXTURE",
           "messages": messages, "mode": mode, "label_sequences": generator.paths if mode == "constrained" else None,
           "max_new_tokens": maximum, "generated_token_ids": ids, "raw_output": raw,
           "input_tokens": len(generator.tok.encode(prompt, add_special_tokens=False)),
           "output_tokens": len(ids), "generation_seconds": 0.0, "truncated": False,
           "finished_utc": "CPU_FIXTURE_NOT_AN_EXPERIMENT", "system_error": None}
    row["record_hash"] = digest(row)
    generator.verify(row, key)
    return row


def seed_new_fixtures(generator, prepared):
    parent_queries = {query_key(r["messages"], r["max_new_tokens"]) for r in prepared["parent"].records.values()}
    requests = {}
    for sample, baseline in zip(prepared["samples"], prepared["parent_decisions"]):
        for method in ("standard", "id_selection"):
            candidate = core.canonical_candidate(baseline["methods"][method])
            if candidate is None:
                continue
            messages = old.relation_messages(old.public_view(sample), candidate)
            maximum = PROTOCOL["relation_max_new_tokens"]
            for mode in ("free", "constrained"):
                if mode == "free" and query_key(messages, maximum) in parent_queries:
                    continue
                requests[request_key(messages, maximum, mode, generator.paths, generator.setup_hash)] = (messages, maximum, mode)
    for sample in prepared["diagnostics"]:
        for variant in VARIANTS:
            messages, _ = core.diagnostic_messages(old.public_view(sample), variant)
            maximum = PROTOCOL["pointer_max_new_tokens"]
            requests[request_key(messages, maximum, "free", generator.paths, generator.setup_hash)] = (messages, maximum, "free")
    for key, (messages, maximum, mode) in requests.items():
        row = fixture_row(generator, messages, maximum, mode)
        write_json(generator.root / "checkpoints" / (key + ".json"), row)
    if not prepared["reuse_parent"]:
        # Historical outputs become explicit refresh-path fixtures under new request keys.
        # Production refresh does not perform this copying; it runs the model.
        for source in prepared["parent"].records.values():
            row = copy.deepcopy(source)
            row.pop("record_hash")
            row["key"] = request_key(row["messages"], row["max_new_tokens"], "free", generator.paths, generator.setup_hash)
            row["setup_hash"], row["mode"], row["label_sequences"] = generator.setup_hash, "free", None
            row["stage"], row["generation_seconds"] = "EXPLICIT_REFRESH_TEST_FIXTURE", 0.0
            row["record_hash"] = digest(row)
            generator.verify(row, row["key"])
            write_json(generator.root / "checkpoints" / (row["key"] + ".json"), row)
    return len(list((generator.root / "checkpoints").glob("*.json")))


def smoke(cache_dir, notebook=None, reuse_parent=True):
    with tempfile.TemporaryDirectory(prefix="hw2_relation_gate_CPU_FIXTURE_") as directory:
        root = Path(directory) / "run"
        prepared = prepare(root, cache_dir=cache_dir, local_files_only=True, reuse_parent=reuse_parent)
        setup = {"reuse_parent": reuse_parent, "protocol_signature": prepared["lock"]["signature"], "fixture_only": True}
        write_json(root / "setup.json", setup)
        generator = Generator(None, prepared["tok"], root, setup, None, prepared["parent"], replay_only=True)
        count = seed_new_fixtures(generator, prepared)
        plan = read_json(root / "call_plan.json")
        assert count == plan["reuse_new_calls_exact" if reuse_parent else "refresh_calls_estimate_from_parent_outputs"]
        # Production request lookup, parsing, dependency tracking, resume and replay are used unchanged.
        resumed = Generator(None, prepared["tok"], root, setup, None, prepared["parent"], replay_only=True)
        rows = [primary_record(s, i, prepared, resumed) for i, s in enumerate(prepared["samples"])]
        diag = [diagnostic_record(s, v, resumed) for s in prepared["diagnostics"] for v in VARIANTS]
        write_jsonl(root / "decisions.jsonl", rows)
        write_jsonl(root / "diagnostic_decisions.jsonl", diag)
        write_json(root / "method_costs.json", resumed.costs())
        audit_saved_run(prepared, setup)
        assert resumed.new_calls == 0
        assert resumed.costs()["_current_physical_total"]["calls"] == count
        if not reuse_parent:
            assert all(c["parent_reused_calls"] == 0 for m, c in resumed.costs().items() if not m.startswith("_"))
        summary = analyze(root, bootstrap_repetitions=PROTOCOL["bootstrap_repetitions"])
        assert summary["primary_inputs"] == 480 and summary["diagnostic_requests"] == 96
        # Verify deterministic reanalysis for all tables, using the same fixed resampling seed.
        csv_before = {p.name: file_hash(p) for p in (root / "results").glob("*.csv")}
        analyze(root, bootstrap_repetitions=PROTOCOL["bootstrap_repetitions"])
        assert csv_before == {p.name: file_hash(p) for p in (root / "results").glob("*.csv")}
        package = archive(root)
        with zipfile.ZipFile(package) as zip_file:
            manifest = read_json(root / "artifact_manifest.json")
            for name, expected in manifest["files"].items():
                assert hashlib.sha256(zip_file.read("results/" + name)).hexdigest() == expected
        if notebook is not None:
            data = read_json(notebook)
            bootstrap = next("".join(c["source"]) for c in data["cells"] if c["id"] == "gate-embedded-source")
            sandbox = Path(directory) / "embedded"
            # The real bootstrap is executed after changing only its output location.
            bootstrap = bootstrap.replace('Path("/content/hw2_relation_gate_code")', "Path(" + repr(str(sandbox)) + ")")
            namespace = {}
            exec(compile(bootstrap, "embedded-cpu-smoke", "exec"), namespace)
            for name, content in namespace["SOURCES"].items():
                assert (sandbox / name).read_bytes() == content.encode("utf-8")
            code = (sandbox / "experiments/hw2_relation_gate_colab/core.py").read_text()
            compile(code, "embedded-core", "exec")
            for c in data["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), c["id"], "exec")
            # Run the actual embedded test cell with an unrelated legacy module present.
            import sys
            from types import ModuleType
            sys.modules.setdefault("test_analysis", ModuleType("test_analysis"))
            test_cell = next("".join(c["source"]) for c in data["cells"] if c["id"] == "gate-cpu-tests")
            exec(compile(test_cell, "embedded-test-cell", "exec"), namespace)
        result = {"status": "PASS", "fixture_outputs_only": True, "new_model_inference": False, "reuse_parent": reuse_parent,
                  "parent_real_raw_records_verified": len(prepared["parent"].records), "primary_fixture_replayed": len(rows),
                  "diagnostic_fixture_replayed": len(diag), "current_fixture_checkpoints": count,
                  "bootstrap_repetitions": PROTOCOL["bootstrap_repetitions"], "deterministic_tables": True,
                  "archive_manifest_verified": True, "notebook_bootstrap_checked": notebook is not None}
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/huggingface/hub")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    smoke(args.cache_dir, args.notebook, reuse_parent=not args.fresh)
