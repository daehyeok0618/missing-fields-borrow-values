"""Methods receive public documents/questions only, never evaluation labels."""
import copy
import re

from . import legacy_core as legacy
from .protocol import PROTOCOL

NULL = {"valid": True, "value": None, "error": None}
public_view = legacy.public_view
numeric_spans = legacy.numeric_spans
canonical_number = legacy.canonical_number
parse = legacy.parse
abstains = legacy.abstains
make_messages = legacy.make_messages
target_messages = legacy.target_messages


def target_text(view):
    return next(d["text"] for d in view["docs"] if d["id"] == view["target_id"])


def presence(parsed, text):
    number = canonical_number(parsed["value"]) if parsed["valid"] else None
    reject = number is not None and number not in {span["value"] for span in numeric_spans(text)}
    return (dict(NULL) if reject else copy.deepcopy(parsed)), reject


def answer_equal(a, b):
    if not a["valid"] or not b["valid"]:
        return False
    if abstains(a, "semantic") and abstains(b, "semantic"):
        return True
    na, nb = canonical_number(a["value"]), canonical_number(b["value"])
    return na == nb if na is not None and nb is not None else a["value"] == b["value"]


def occurrences(text):
    return [{"id": f"n{i:02d}", **span} for i, span in enumerate(numeric_spans(text), 1)]


def pointer_messages(view):
    text = target_text(view)
    spans = occurrences(text)
    marked = text
    for span in reversed(spans):
        marked = marked[:span["start"]] + "[" + span["id"] + "]" + marked[span["start"]:]
    instruction = (
        "Select a numeric occurrence that explicitly states the field asked in the question "
        "for the target organization. Each bracketed ID marks the numeric occurrence immediately after it. "
        "A number describing another field does not answer the question. "
        "If the requested field is not explicitly stated, select NONE. "
        "Return exactly one listed ID or NONE. Do not return a number, JSON, explanation, or Markdown."
    )
    body = ("Target organization: " + view["target_id"] + "\n\nTarget document:\n" + marked
            + "\n\n" + view["question"] + "\nAllowed outputs: "
            + ", ".join([s["id"] for s in spans] + ["NONE"]))
    return [{"role": "system", "content": instruction}, {"role": "user", "content": body}], spans


def parse_selection(raw, spans, truncated=False):
    choice = raw.strip()
    if truncated:
        return {"valid": False, "value": None, "error": "selection_truncated"}, "invalid", None
    if choice == "NONE":
        return dict(NULL), "model_none", None
    selected = next((s for s in spans if s["id"] == choice), None)
    if selected is None:
        return {"valid": False, "value": None, "error": "invalid_selection"}, "invalid", None
    return {"valid": True, "value": selected["value"], "error": None}, "selected", selected


def relation_messages(view, candidate):
    instruction = (
        "Decide whether the target document explicitly assigns the candidate value to the target "
        "organization for the field asked in the question. A value for another field is not support. "
        "Return SUPPORTED only if this exact relationship is stated. Otherwise return UNSUPPORTED. "
        "Return exactly one of these two words without JSON, evidence, explanation, or Markdown."
    )
    body = ("Target organization: " + view["target_id"] + "\n\nTarget document:\n" + target_text(view)
            + "\n\n" + view["question"] + "\nCandidate value: " + str(candidate["value"]))
    return [{"role": "system", "content": instruction}, {"role": "user", "content": body}]


def relation_decision(candidate, raw, truncated=False):
    choice = raw.strip()
    if not truncated and choice == "SUPPORTED":
        return copy.deepcopy(candidate), "supported"
    if not truncated and choice == "UNSUPPORTED":
        return dict(NULL), "unsupported"
    return {"valid": False, "value": None, "error": "invalid_relation_verdict"}, "invalid"


def field_parser(view):
    # Transparent lexical co-occurrence baseline, not a general entailment parser.
    label = r"\bsalary\b" if view["task"] == "salary" else r"\bfounding\s+year\b"
    values = set()
    for clause in re.split(r"[\n.!?;]", target_text(view)):
        if re.search(label, clause, re.I) and not re.search(r"\b(not|unknown|missing|unavailable)\b", clause, re.I):
            values.update(s["value"] for s in numeric_spans(clause))
    if len(values) != 1:
        return dict(NULL)
    return {"valid": True, "value": next(iter(values)), "error": None}


def decide(view, generator):
    """Shared inference and separate cost attribution; no gold or group arguments."""
    initial = generator.generate(make_messages(view), "standard")
    original = parse(initial["raw_output"])
    methods = {m: copy.deepcopy(original) for m in PROTOCOL["methods"]}
    keys = {"standard": initial["key"]}
    for method in ("standard", "presence_gate", "relation_verifier", "value_tracking", "any_change", "presence_any_change"):
        generator.attribute(method, initial)
    methods["presence_gate"], early_reject = presence(original, target_text(view))
    direct = generator.generate(target_messages(view), "target_only")
    methods["target_only"] = parse(direct["raw_output"])
    methods["target_only_presence"], _ = presence(methods["target_only"], target_text(view))
    generator.attribute("target_only", direct)
    generator.attribute("target_only_presence", direct)
    keys["target_only"] = direct["key"]

    messages, spans = pointer_messages(view)
    selection_status, selected = "no_candidate", None
    selection_raw = None
    if spans:
        row = generator.generate(messages, "id_selection", PROTOCOL["pointer_max_new_tokens"])
        methods["id_selection"], selection_status, selected = parse_selection(row["raw_output"], spans, row.get("truncated", False))
        selection_raw = row["raw_output"]
        keys["id_selection"] = row["key"]
        generator.attribute("id_selection", row)
    else:
        methods["id_selection"] = dict(NULL)

    relation_status = "not_numeric_initial"
    if original["valid"] and canonical_number(original["value"]) is not None:
        row = generator.generate(relation_messages(view, original), "relation_verifier", PROTOCOL["relation_max_new_tokens"])
        methods["relation_verifier"], relation_status = relation_decision(original, row["raw_output"], row.get("truncated", False))
        keys["relation_verifier"] = row["key"]
        generator.attribute("relation_verifier", row)

    probes = legacy.prepare_probes(view, original, generator.tok)
    probe_values = []
    if probes["eligible"]:
        probe_rows = [generator.generate(messages, f"value_probe_{i+1}") for i, messages in enumerate(probes["messages"])]
        probe_values = [parse(row["raw_output"]) for row in probe_rows]
        tracking = all(p["valid"] and canonical_number(p["value"]) == z
                       for p, z in zip(probe_values, probes["replacements"]))
        changed = any(p["valid"] and not answer_equal(original, p) for p in probe_values)
        if tracking:
            methods["value_tracking"] = dict(NULL)
        if changed:
            methods["any_change"] = dict(NULL)
        for method in ("value_tracking", "any_change"):
            generator.attribute(method, *probe_rows)
        if not early_reject:
            generator.attribute("presence_any_change", *probe_rows)
        keys["probes"] = [row["key"] for row in probe_rows]
    methods["presence_any_change"] = dict(NULL) if early_reject else copy.deepcopy(methods["any_change"])
    methods["field_parser"] = field_parser(view)
    methods["always_abstain"] = dict(NULL)
    # A service can fail closed, but primary analysis retains invalid decisions.
    served = {m: copy.deepcopy(p) if p["valid"] else dict(NULL) for m, p in methods.items()}
    return {"methods": methods, "served_methods": served, "raw_keys": keys,
            "eligible": probes["eligible"], "probe_reason": probes["reason"],
            "replacements": probes["replacements"], "probe_values": probe_values,
            "probe_audit": probes["audit"],
            "selection": {"status": selection_status, "raw": selection_raw, "candidates": spans, "selected": selected},
            "relation_status": relation_status, "presence_early_reject": early_reject}
