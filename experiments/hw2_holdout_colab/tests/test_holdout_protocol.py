import copy
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from .. import core, data, history
from ..common import digest
from ..engine import Generator, compatibility, method_costs, request_key
from ..protocol import ALIASES, GROUPS, METHODS, PROTOCOL
from ..runner import record_for


def parsed(value):
    return {"valid": True, "value": value, "error": None}


def view(task="salary", phrase="budget", value="128527"):
    return {"target_id": "Example A", "docs": [{"id": "Example A", "text": "Organization: Example A\nRecorded " + phrase + ": " + value}],
            "question": "Question: What is the recorded " + ("salary" if task == "salary" else "founding year") + " for Example A?",
            "task": task, "prefix": "", "suffix": "", "placement": "MIDDLE"}


class ToyTokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return {"SUPPORTED": [1, 3], "UNSUPPORTED": [2, 3]}[text]

    def decode(self, ids, **kwargs):
        return {(1, 3): "SUPPORTED", (2, 3): "UNSUPPORTED"}[tuple(ids)]


class FakeRequests:
    def __init__(self, prefix="current:", verdict="UNSUPPORTED"):
        self.calls, self.prefix, self.verdict = [], prefix, verdict

    def generate(self, messages, stage, maximum=None, mode="free"):
        if stage == "standard":
            raw = '{"value":"$128527"}'
        elif stage == "target_only":
            raw = '{"value":"128527"}'
        elif stage == "id_selection":
            raw = "n01"
        else:
            raw = self.verdict
        self.calls.append((stage, messages, mode))
        return {"key": digest([messages, maximum, mode]), "raw_output": raw, "truncated": False}

    def reference(self, row):
        return self.prefix + row["key"]


class HoldoutProtocolTests(unittest.TestCase):
    def test_methods_and_token_mask_remain_frozen(self):
        from ..parent_source.protocol import PROTOCOL as previous
        for key in ("model", "revision", "generation", "dtype", "attention", "pointer_max_new_tokens", "relation_max_new_tokens"):
            self.assertEqual(PROTOCOL[key], previous[key])
        self.assertEqual(json.loads(json.dumps(PROTOCOL)), PROTOCOL)
        tok = ToyTokenizer()
        paths = core.label_sequences(tok)
        import torch
        from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
        mask = PrefixConstrainedLogitsProcessor(core.prefix_callback(paths, 2), num_beams=1)
        scores = torch.zeros((1, 12)); scores[0, 5] = 100
        self.assertEqual(torch.where(torch.isfinite(mask(torch.tensor([[10, 11]]), scores)[0]))[0].tolist(), [1, 2])
        self.assertEqual(core.allowed_next(paths, [1]), [3])
        self.assertEqual(core.allowed_next(paths, [1, 3]), [9])
        with self.assertRaises(RuntimeError):
            core.allowed_next(paths, [5])

    def test_three_candidates_receive_identical_verifier(self):
        fake = FakeRequests()
        result = core.decide(view(), fake)
        gates = [(msg, mode) for stage, msg, mode in fake.calls if "gate" in stage]
        self.assertEqual(len(gates), 3)
        self.assertTrue(all(pair == gates[0] for pair in gates))
        self.assertEqual(gates[0][1], "constrained")
        self.assertEqual(len(result["dependencies"]["target_only_gate_constrained"]), 2)
        self.assertEqual(result["methods"]["id_gate_constrained"], parsed(None))

    def test_gold_and_split_metadata_cannot_change_requests(self):
        original = view()
        poisoned = copy.deepcopy(original) | {"gold": "9999", "group": "present", "split": "holdout_paraphrase"}
        a, b = FakeRequests(), FakeRequests()
        self.assertEqual(core.decide(original, a), core.decide(poisoned, b))
        self.assertEqual(a.calls, b.calls)

    def test_invalid_gate_is_not_valid_abstention(self):
        result = core.decide(view(), FakeRequests(verdict="SUPPORTING"))
        self.assertFalse(result["methods"]["target_only_gate_constrained"]["valid"])
        self.assertEqual(result["served_methods"]["target_only_gate_constrained"], parsed(None))

    def test_primary_control_matches_id_no_candidate_shortcut(self):
        sample = view()
        sample["docs"][0]["text"] = "Organization: Example A\nActivity: acoustic research"
        result = core.decide(sample, FakeRequests())
        for method in ("id_selection", "id_gate_constrained", "target_only_routed_gate_constrained"):
            self.assertEqual(result["methods"][method], parsed(None))
            self.assertEqual(result["dependencies"][method], [])
        self.assertEqual(result["gates"]["target_only_routed_gate_constrained"]["status"], "no_candidate_cpu")

    def test_schema_aware_parser_covers_all_declared_aliases(self):
        for task in ("salary", "year"):
            for phrase in ALIASES[task]["requested"]:
                self.assertEqual(core.alias_field_parser(view(task, phrase)), parsed("128527"))
            for kind in ("competing", "third"):
                for phrase in ALIASES[task][kind]:
                    self.assertEqual(core.alias_field_parser(view(task, phrase)), parsed(None))

    def test_group_construction_and_balanced_multi_slot(self):
        values = ["1935", "1958", "1997"]
        for slot in (0, 1):
            present, answer = data.facts_for("multi_present", values, slot)
            missing, absent = data.facts_for("multi_missing", values, slot)
            self.assertEqual([v for _, v in present], [v for _, v in missing])
            self.assertEqual(answer, values[slot])
            self.assertIsNone(absent)
            self.assertEqual(present[slot][0], "requested")
            self.assertNotIn("requested", [k for k, _ in missing])
        self.assertEqual(len(GROUPS), 6)
        self.assertEqual(PROTOCOL["expected_holdout_inputs"], PROTOCOL["expected_holdout_bases"] * 6 * 3 * 2 * 2)

    def test_holdout_rejects_historical_model_dependencies(self):
        sample = view() | {"id": "HLD_TEST", "base_id": "HLD", "split": "holdout_canonical", "template": "KV", "group": "otherfield_missing"}
        with self.assertRaises(RuntimeError):
            record_for(sample, FakeRequests(prefix="pointer:"))

    def test_history_mode_identity_and_runtime_check(self):
        self.assertNotEqual(history.query_key([], 12, "free"), history.query_key([], 12, "constrained", {"x": [1]}))
        self.assertIn("torch", compatibility({"torch": "a"}, {"torch": "b"}))
        self.assertIn("parent_zip_sha256", history.authenticate_sources())

    def test_costs_keep_logical_duplicates_and_separate_splits(self):
        historical = SimpleNamespace(records={}, queries={})
        with tempfile.TemporaryDirectory() as tmp:
            generator = Generator(None, ToyTokenizer(), tmp, {"reuse_parent": False}, None, historical, replay_only=True)
            generator.records["a"] = {"input_tokens": 10, "output_tokens": 3, "generation_seconds": .25, "truncated": False}
            row = {"split": "holdout_canonical", "dependencies": {m: (["current:a"] if m == "id_selection" else []) for m in METHODS}}
            costs = method_costs([row, row], generator)
            self.assertEqual(costs["holdout_canonical"]["id_selection"]["unique"]["calls"], 1)
            self.assertEqual(costs["holdout_canonical"]["id_selection"]["logical"]["calls"], 2)
            self.assertEqual(costs["development"]["id_selection"]["n_inputs"], 0)


if __name__ == "__main__":
    unittest.main()
