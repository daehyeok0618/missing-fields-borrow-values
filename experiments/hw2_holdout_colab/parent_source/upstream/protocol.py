"""Settings fixed before the new ID-selection generations are observed."""
PROTOCOL = {
    "version": "hw2-pointer-development-v1",
    "status": "development: reused 384 input cases plus 96 new multi-candidate cases; not confirmatory",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "transformers": "4.57.6", "seed": 2026111001, "bootstrap_seed": 2026111002,
    "dtype": "float16", "attention": "sdpa", "batch_size": 1,
    "generation": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                   "repetition_penalty": 1.1, "max_new_tokens": 96, "use_cache": True},
    "pointer_max_new_tokens": 16, "relation_max_new_tokens": 12,
    "expected_samples": 480, "expected_bases": 48,
    "new_bases_per_task": 4, "bootstrap_repetitions": 2000,
    "max_prompt_tokens": 2304, "max_total_generation_calls": 2880,
    "methods": ["standard", "presence_gate", "target_only", "target_only_presence",
                "id_selection", "relation_verifier", "field_parser", "value_tracking",
                "any_change", "presence_any_change", "always_abstain"],
    "pointer_scope": "target document only; all numeric occurrences, exact ID or NONE; unconstrained greedy",
    "pointer_no_candidate": "valid null without model call; reported separately",
    "invalid_policy": "strict decision remains INVALID; a separate service-output analysis maps invalid to null",
    "probe_policy": "legacy exact numeric interventions; invalid response is neither change nor tracking",
    "reuse": "Reuse source documents/labels, not historical model outputs; rerun all model baselines",
    "scope": "independent entity documents; no cross-document answer derivation",
}

GROUPS = ("basic_missing", "basic_present", "collision_present", "otherfield_missing",
          "multi_present", "multi_missing")
