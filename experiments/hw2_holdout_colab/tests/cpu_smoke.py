"""Full software check with explicit temporary fixtures; never Qwen inference."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from .. import core, data
from ..common import digest, file_hash, read_json, write_json, write_jsonl
from ..engine import Generator, method_costs, request_key
from ..protocol import METHODS, PROTOCOL
from ..runner import archive, audit_saved_run, prepare, record_for
from ..analysis import analyze


class FixtureGenerator(Generator):
    """Supply known test strings through the production cache/replay paths."""
    def generate(self, messages, stage, max_new_tokens=None, mode="free"):
        maximum = max_new_tokens or PROTOCOL["generation"]["max_new_tokens"]
        key = request_key(messages, maximum, mode, self.paths, self.setup_hash)
        if not (self.setup["reuse_parent"] and self.parent.lookup(messages, maximum, mode) is not None) and key not in self.records:
            number = int(key[:8], 16)
            if mode == "constrained":
                raw = ("SUPPORTED", "UNSUPPORTED")[number % 2]
            elif maximum == PROTOCOL["pointer_max_new_tokens"]:
                raw = ("n01", "NONE", "INVALID")[number % 3]
            else:
                spans = core.old.numeric_spans(messages[-1]["content"])
                options = [None] + [s["value"] for s in spans] + ["999999"]
                raw = json.dumps({"value": options[number % len(options)]}) if number % 13 else "INVALID_JSON"
            ids = self.tok.encode(raw, add_special_tokens=False) + [self.tok.eos_token_id]
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            row = {"key": key, "setup_hash": self.setup_hash, "stage": "EXPLICIT_CPU_FIXTURE_" + stage,
                   "messages": messages, "mode": mode, "max_new_tokens": maximum,
                   "label_sequences": self.paths if mode == "constrained" else None,
                   "generated_token_ids": ids, "raw_output": raw, "input_tokens": len(self.tok.encode(prompt, add_special_tokens=False)),
                   "output_tokens": len(ids), "generation_seconds": .001 * len(ids), "truncated": False,
                   "finished_utc": "CPU_FIXTURE_NOT_A_RESEARCH_RESULT", "system_error": None}
            row["record_hash"] = digest(row)
            self.verify(row, key)
            self.records[key] = row
            write_json(self.root / "checkpoints" / (key + ".json"), row)
        return super().generate(messages, stage, max_new_tokens, mode)


def check_dataset(prepared):
    holdout = [s for s in prepared["samples"] if s["split"].startswith("holdout_")]
    dev = [s for s in prepared["samples"] if s["split"] == "development"]
    regenerated, regenerated_gold, _ = data.build_holdout(prepared["tok"], prepared["restored"]["samples"])
    assert regenerated == holdout
    assert regenerated_gold == {s["id"]: prepared["gold"][s["id"]] for s in holdout}
    old_names, old_amounts, old_triples, old_pairs = data.inventory(dev)
    from itertools import combinations
    families = read_json(prepared["root"] / "holdout_data_audit.json")["family_inventory"]
    for family in families:
        assert not {n.casefold() for n in family["entities"]} & old_names
        if family["task"] == "salary":
            assert not set(family["values"]) & old_amounts
        else:
            assert tuple(sorted(family["values"])) not in old_triples
            assert not {tuple(sorted(c)) for c in combinations(family["values"], 2)} & old_pairs
    by_pair = defaultdict(dict)
    for s in holdout:
        view = core.old.public_view(s)
        assert not set(view) & {"gold", "group", "split", "base_id"}
        assert core.alias_field_parser(view)["value"] == prepared["gold"][s["id"]]["value"]
        by_pair[(s["base_id"], s["group"], s["template"], s["placement"])][s["split"]] = s
    for pair in by_pair.values():
        a, b = pair["holdout_canonical"], pair["holdout_paraphrase"]
        assert a["prefix"] == b["prefix"] and a["suffix"] == b["suffix"]
        assert [d["id"] for d in a["docs"]] == [d["id"] for d in b["docs"]]
        assert [[x["value"] for x in core.old.numeric_spans(d["text"])] for d in a["docs"]] == [[x["value"] for x in core.old.numeric_spans(d["text"])] for d in b["docs"]]
        assert prepared["gold"][a["id"]]["value"] == prepared["gold"][b["id"]]["value"]


def smoke(cache_dir, notebook=None, reuse_parent=True, output=None):
    with tempfile.TemporaryDirectory(prefix="hw2_holdout_CPU_FIXTURE_") as directory:
        root = Path(directory) / "run"
        prepared = prepare(root, cache_dir=cache_dir, reuse_parent=reuse_parent, local_files_only=True)
        check_dataset(prepared)
        setup = {"reuse_parent": reuse_parent, "protocol_signature": prepared["lock"]["signature"], "fixture_only": True}
        write_json(root / "setup.json", setup)
        fixture = FixtureGenerator(None, prepared["tok"], root, setup, None, prepared["history"], replay_only=True)
        records = [record_for(s, fixture) for s in prepared["samples"]]
        write_jsonl(root / "decisions.jsonl", records)
        write_json(root / "method_costs.json", method_costs(records, fixture))
        assert len(fixture.records) <= prepared["plan"]["reuse_total_new_calls_upper_bound" if reuse_parent else "refresh_total_new_calls_upper_bound"]
        resumed = Generator(None, prepared["tok"], root, setup, None, prepared["history"], replay_only=True)
        assert records == [record_for(s, resumed) for s in prepared["samples"]]
        audit_saved_run(prepared, setup)
        assert resumed.new_calls == 0
        result = analyze(root)
        assert result["holdout_inputs"] == 1152 and result["holdout_independent_bases"] == 16
        hashes = {p.name: file_hash(p) for p in (root / "results").glob("*.csv")}
        analyze(root)
        assert hashes == {p.name: file_hash(p) for p in (root / "results").glob("*.csv")}
        package = archive(root)
        with zipfile.ZipFile(package) as z:
            for name, value in read_json(root / "artifact_manifest.json")["files"].items():
                assert hashlib.sha256(z.read("results/" + name)).hexdigest() == value
        if notebook is not None:
            nb = read_json(notebook)
            source_cell = next("".join(c["source"]) for c in nb["cells"] if c["id"] == "gate-embedded-source")
            sandbox = Path(directory) / "embedded"
            source_cell = source_cell.replace('Path("/content/hw2_holdout_code")', "Path(" + repr(str(sandbox)) + ")")
            env = {}
            exec(compile(source_cell, "embedded-source", "exec"), env)
            for rel, content in env["SOURCES"].items():
                assert (sandbox / rel).read_bytes() == content.encode("utf-8")
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), c["id"], "exec")
            import sys
            from types import ModuleType
            sys.modules.setdefault("test_analysis", ModuleType("test_analysis"))
            tests = next("".join(c["source"]) for c in nb["cells"] if c["id"] == "gate-cpu-tests")
            exec(compile(tests, "embedded-tests", "exec"), env)
        result = {"status": "PASS_CPU_ONLY", "new_model_inference_executed": False, "performance_measured": False,
                  "fixture_outputs_only": True, "reuse_parent": reuse_parent, "historical_raw_records_verified": 2000,
                  "dataset_regenerated_identically": True, "historical_inventory_disjointness_verified": True,
                  "decisions_replayed": len(records), "current_fixture_records": len(fixture.records),
                  "bootstrap_repetitions": 2000, "deterministic_csv_reanalysis": True,
                  "archive_manifest_verified": True, "notebook_bootstrap_and_test_cell_verified": notebook is not None}
        if output is not None:
            write_json(output, result)
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/huggingface/hub")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    smoke(args.cache_dir, args.notebook, reuse_parent=not args.fresh, output=args.output)
