"""Public-input methods; evaluation labels are never passed to these functions."""
import copy

from .protocol import METHODS, PROTOCOL
from .upstream import core as old

NULL = old.NULL
abstains = old.abstains
canonical_number = old.canonical_number


def canonical_candidate(parsed):
    value = canonical_number(parsed["value"]) if parsed["valid"] else None
    return None if value is None else {"valid": True, "value": value, "error": None}


def label_sequences(tok):
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer EOS is required")
    paths = {label: tok.encode(label, add_special_tokens=False) + [tok.eos_token_id]
             for label in ("SUPPORTED", "UNSUPPORTED")}
    for label, ids in paths.items():
        if len(ids) >= PROTOCOL["relation_max_new_tokens"] or tok.decode(ids[:-1], clean_up_tokenization_spaces=False) != label:
            raise ValueError("Cannot represent the exact verdict labels")
    return paths


def allowed_next(paths, suffix):
    allowed = sorted({ids[len(suffix)] for ids in paths.values()
                      if len(suffix) < len(ids) and ids[:len(suffix)] == list(suffix)})
    if not allowed:
        raise RuntimeError("Generated sequence left the fixed label grammar")
    return allowed


def prefix_callback(paths, prompt_length):
    def callback(batch_id, input_ids):
        if batch_id != 0:
            raise RuntimeError("This protocol requires batch size one")
        return allowed_next(paths, input_ids.tolist()[prompt_length:])
    return callback


def decide(view, baseline, generator, dependencies):
    methods = {m: copy.deepcopy(baseline["methods"][m]) for m in METHODS if m in baseline["methods"]}
    gates = {}
    for source, prefix in (("standard", "standard"), ("id_selection", "id")):
        candidate = baseline["methods"][source]
        normalized = canonical_candidate(candidate)
        for mode in ("free", "constrained"):
            method = prefix + "_gate_" + mode
            refs = list(dependencies[source])
            if normalized is None:
                methods[method] = copy.deepcopy(candidate)
                gates[method] = {"status": "skipped_non_numeric", "ref": None, "candidate": None}
            else:
                row = generator.generate(old.relation_messages(view, normalized), method,
                                         PROTOCOL["relation_max_new_tokens"], mode=mode)
                methods[method], status = old.relation_decision(candidate, row["raw_output"], row.get("truncated", False))
                refs.append(generator.reference(row))
                gates[method] = {"status": status, "ref": generator.reference(row), "candidate": normalized["value"]}
            generator.used[method].update(refs)
    for method in METHODS:
        if method in dependencies:
            generator.used[method].update(dependencies[method])
    if set(methods) != set(METHODS):
        raise RuntimeError("Incomplete method outputs")
    served = {m: copy.deepcopy(p) if p["valid"] else dict(NULL) for m, p in methods.items()}
    return {"methods": methods, "served_methods": served, "gates": gates,
            "baseline_selection": baseline["selection"], "baseline_raw_keys": baseline["raw_keys"]}


DIAGNOSTIC_INSTRUCTION = (
    "Select the ID of the numeric fact that explicitly states the field asked in the question "
    "for the target organization. Each bracketed ID identifies one numeric fact. "
    "A number describing another field does not answer the question. "
    "If the requested field is not explicitly stated, select NONE. "
    "Return exactly one listed ID or NONE. Do not return a number, JSON, explanation, or Markdown."
)


def diagnostic_messages(view, variant):
    """Only ID location or dash prefix changes; instruction/header/question are identical."""
    if variant not in ("numeric", "clause", "numeric_dash"):
        raise ValueError(variant)
    text = old.target_text(view)
    spans = old.occurrences(text)
    lines = text.splitlines(keepends=True)
    marked, offset = [], 0
    for line in lines:
        local = [s for s in spans if offset <= s["start"] < offset + len(line)]
        if len(local) > 1:
            raise ValueError("Diagnostic expects one numeric occurrence per record")
        if local:
            s = local[0]
            insert = 0 if variant == "clause" else s["start"] - offset
            line = line[:insert] + "[" + s["id"] + "]" + line[insert:]
            if variant == "numeric_dash":
                line = "- " + line
        marked.append(line)
        offset += len(lines[len(marked) - 1])
    body = ("Target organization: " + view["target_id"] + "\n\nTarget document:\n" + "".join(marked)
            + "\n\n" + view["question"] + "\nAllowed outputs: " + ", ".join([s["id"] for s in spans] + ["NONE"]))
    return [{"role": "system", "content": DIAGNOSTIC_INSTRUCTION}, {"role": "user", "content": body}], spans
