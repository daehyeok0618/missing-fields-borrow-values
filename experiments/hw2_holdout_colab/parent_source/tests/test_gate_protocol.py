import copy
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from .. import core, parent
from ..common import digest, read_json, write_json
from ..engine import Generator, compatibility, query_key, request_key
from ..protocol import METHODS, PROTOCOL
from ..runner import archive
from ..upstream import core as old


class ToyTokenizer:
    eos_token_id = 9

    def encode(self, label, add_special_tokens=False):
        return {"SUPPORTED": [1, 3], "UNSUPPORTED": [2, 3]}[label]

    def decode(self, ids, **kwargs):
        return {(1, 3): "SUPPORTED", (2, 3): "UNSUPPORTED"}[tuple(ids)]


def parsed(value):
    return {"valid": True, "value": value, "error": None}


def view():
    return {"target_id": "Example A", "docs": [{"id": "Example A", "text": "Organization: Example A\nRecorded closure year: 1984\nRecorded renovation year: 2001"}],
            "question": "What is the founding year of Example A?", "task": "year", "prefix": "", "suffix": "", "placement": "MIDDLE"}


class GateProtocolTests(unittest.TestCase):
    def test_every_allowed_token_path_is_a_label_with_eos(self):
        paths = core.label_sequences(ToyTokenizer())
        frontier = [[]]
        finished = []
        while frontier:
            suffix = frontier.pop()
            if suffix and suffix[-1] == 9:
                finished.append(suffix)
                continue
            frontier.extend(suffix + [token] for token in core.allowed_next(paths, suffix))
        self.assertCountEqual(finished, [[1, 3, 9], [2, 3, 9]])
        with self.assertRaises(RuntimeError):
            core.allowed_next(paths, [5])

    def test_actual_transformers_logits_processor_masks_illegal_tokens(self):
        import torch
        from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
        paths = core.label_sequences(ToyTokenizer())
        callback = core.prefix_callback(paths, 2)
        processor = PrefixConstrainedLogitsProcessor(callback, num_beams=1)
        scores = torch.zeros((1, 12))
        scores[0, 5] = 100  # An illegal answer would win without masking.
        first = processor(torch.tensor([[10, 11]]), scores)
        self.assertEqual(torch.where(torch.isfinite(first[0]))[0].tolist(), [1, 2])
        second = processor(torch.tensor([[10, 11, 1]]), scores)
        self.assertEqual(torch.where(torch.isfinite(second[0]))[0].tolist(), [3])
        last = processor(torch.tensor([[10, 11, 1, 3]]), scores)
        self.assertEqual(torch.where(torch.isfinite(last[0]))[0].tolist(), [9])

    def test_same_prompt_free_and_constrained_are_distinct_requests(self):
        paths = core.label_sequences(ToyTokenizer())
        messages = old.relation_messages(view(), parsed("1984"))
        self.assertNotEqual(request_key(messages, 12, "free", paths, "setup"), request_key(messages, 12, "constrained", paths, "setup"))
        self.assertEqual(core.canonical_candidate(parsed("$72,000")), parsed("72000"))

    def test_verdicts_invalid_and_null_remain_distinct(self):
        self.assertEqual(old.relation_decision(parsed("1984"), "SUPPORTED")[0], parsed("1984"))
        self.assertEqual(old.relation_decision(parsed("1984"), "UNSUPPORTED")[0], parsed(None))
        self.assertFalse(old.relation_decision(parsed("1984"), "SUPPORTING")[0]["valid"])
        self.assertFalse(old.relation_decision(parsed("1984"), "SUPPORTED", True)[0]["valid"])

    def test_both_gates_normalize_the_same_candidate_and_skip_none(self):
        class Fake:
            def __init__(self):
                self.used, self.calls = defaultdict(set), []

            def generate(self, messages, stage, maximum, mode="free"):
                self.calls.append((messages, mode))
                return {"key": "x", "raw_output": "UNSUPPORTED", "truncated": False}

            def reference(self, row):
                return "current:x"
        baseline = {"methods": {m: parsed("$1984" if m == "standard" else "1984") for m in METHODS if "_gate_" not in m},
                    "selection": {}, "raw_keys": {}}
        fake = Fake()
        result = core.decide(view(), baseline, fake, {"standard": ["parent:s"], "id_selection": ["parent:i"]})
        self.assertEqual(fake.calls[0][0], fake.calls[2][0])
        self.assertEqual(fake.used["id_gate_constrained"], {"parent:i", "current:x"})
        self.assertEqual(result["methods"]["id_gate_constrained"], parsed(None))
        baseline["methods"]["id_selection"] = parsed(None)
        fake = Fake()
        result = core.decide(view(), baseline, fake, {"standard": [], "id_selection": []})
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(result["gates"]["id_gate_free"]["status"], "skipped_non_numeric")

    def test_representation_changes_are_matched_and_do_not_use_gold(self):
        original = view()
        messages = {}
        for variant in ("numeric", "clause", "numeric_dash"):
            messages[variant], spans = core.diagnostic_messages(original, variant)
            self.assertEqual([s["value"] for s in spans], ["1984", "2001"])
        self.assertEqual(len({digest(m[0]) for m in messages.values()}), 1)
        a = messages["numeric"][1]["content"]
        b = messages["numeric_dash"][1]["content"]
        self.assertEqual(b.replace("- Recorded", "Recorded"), a)
        poisoned = copy.deepcopy(original) | {"gold": "9999", "support_spans": [999], "group": "x"}
        self.assertEqual(core.diagnostic_messages(poisoned, "clause"), core.diagnostic_messages(original, "clause"))

    def test_runtime_drift_is_detected(self):
        self.assertEqual(compatibility({"gpu": "Tesla T4"}, {"gpu": "Tesla T4"}), {})
        self.assertIn("torch", compatibility({"torch": "new"}, {"torch": "old"}))

    def test_parent_snapshot_has_not_changed(self):
        self.assertIn("parent_zip_sha256", parent.authenticate_sources())

    def test_archive_and_raw_record_tampering(self):
        tok = ToyTokenizer()
        frozen = SimpleNamespace(records={}, setup_hash="parent")
        setup = {"reuse_parent": False}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            generator = Generator(None, tok, root, setup, None, frozen, replay_only=True)
            messages = [{"role": "user", "content": "test"}]
            with self.assertRaises(RuntimeError):
                generator.generate(messages, "cannot infer in replay", 12)
            row = {"key": request_key(messages, 12, "constrained", generator.paths, generator.setup_hash),
                   "setup_hash": generator.setup_hash, "messages": messages, "mode": "constrained",
                   "max_new_tokens": 12, "label_sequences": generator.paths, "generated_token_ids": [1, 3, 9],
                   "truncated": False}
            row["record_hash"] = digest(row)
            generator.verify(row, row["key"])
            row["generated_token_ids"] = [5, 9]
            row["record_hash"] = digest({k: v for k, v in row.items() if k != "record_hash"})
            with self.assertRaises(RuntimeError):
                generator.verify(row, row["key"])
            write_json(root / "test.json", {"status": "test"})
            result = archive(root)
            self.assertTrue(Path(result).is_file())
            self.assertIn("test.json", read_json(root / "artifact_manifest.json")["files"])
            with self.assertRaises(ValueError):
                archive(root, root / "bad.zip")


if __name__ == "__main__":
    unittest.main()
