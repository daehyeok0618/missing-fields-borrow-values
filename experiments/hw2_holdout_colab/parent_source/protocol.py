"""Fixed before observing the follow-up generations. All inputs are development data."""
METHODS = ("standard", "presence_gate", "target_only_presence", "id_selection",
           "standard_gate_free", "id_gate_free", "standard_gate_constrained", "id_gate_constrained",
           "any_change", "presence_any_change", "field_parser", "always_abstain")
GROUPS = ("basic_missing", "basic_present", "collision_present", "otherfield_missing",
          "multi_present", "multi_missing")
PROTOCOL = {
    "version": "hw2-relation-gate-development-v1",
    "status": "development, selected after parent error analysis; not independent confirmation",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "transformers": "4.57.6", "tokenizers": "0.22.2",
    "seed": 2026111001, "bootstrap_seed": 2026111003, "bootstrap_repetitions": 2000,
    "dtype": "float16", "attention": "sdpa", "batch_size": 1,
    "generation": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                   "repetition_penalty": 1.1, "max_new_tokens": 96, "use_cache": True},
    "relation_max_new_tokens": 12, "pointer_max_new_tokens": 16,
    "max_prompt_tokens": 2304, "max_calls_reuse": 500, "max_calls_refresh": 4000,
    "expected_samples": 480, "expected_bases": 48, "expected_diagnostic_cases": 32,
    "expected_diagnostic_requests": 96, "methods": list(METHODS),
    "candidate_normalization": "Both Standard and ID gate prompts use canonical integer strings; original accepted answer is retained.",
    "gate": "Same frozen parent relation prompt. SUPPORTED retains candidate; UNSUPPORTED returns null. No alias mapping or retry.",
    "constrained": "prefix_allowed_tokens_fn masks every generation step to tokenized SUPPORTED or UNSUPPORTED followed by tokenizer EOS.",
    "invalid": "Primary decision counts invalid as error; separate fail-closed service analysis maps invalid to null.",
    "reuse": "Only exact parent free-generation requests in an identical recorded runtime; strict compatibility before model load. REUSE_PARENT=False reruns all baselines.",
    "diagnostic": "Both tasks, all 16 KV numeric-missing families paired with present counterparts. Same instruction; numeric-ID vs clause-ID vs numeric-ID plus dash. Separate from primary 480.",
    "scope": "Independent entity documents, provided target routing; synthetic explicit labels; no training, no inference-time gold.",
}
