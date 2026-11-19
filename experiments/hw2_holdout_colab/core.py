"""Methods operate on public documents/questions, without gold or split metadata."""
import copy
import re

from .parent_source import core as frozen
from .parent_source.upstream import core as old
from .protocol import ALIASES, METHODS, PROTOCOL

NULL = old.NULL
canonical_candidate = frozen.canonical_candidate
canonical_number = old.canonical_number
abstains = old.abstains
label_sequences = frozen.label_sequences
prefix_callback = frozen.prefix_callback
allowed_next = frozen.allowed_next


def alias_field_parser(view):
    """Stronger disclosed schema-aware lexical baseline; uses every declared alias."""
    pattern = r"\b(?:" + "|".join(re.escape(s) for s in ALIASES[view["task"]]["requested"]) + r")\b"
    values = set()
    for clause in re.split(r"[\n.!?;]", old.target_text(view)):
        if re.search(pattern, clause, re.I) and not re.search(r"\b(not|unknown|missing|unavailable)\b", clause, re.I):
            values.update(s["value"] for s in old.numeric_spans(clause))
    return dict(NULL) if len(values) != 1 else {"valid": True, "value": next(iter(values)), "error": None}


def decide(view, generator):
    methods, deps, gates, raw_keys = {}, {}, {}, {}
    standard = generator.generate(old.make_messages(view), "standard")
    direct = generator.generate(old.target_messages(view), "target_only")
    for name, row in (("standard", standard), ("target_only", direct)):
        methods[name] = old.parse(row["raw_output"])
        deps[name], raw_keys[name] = [generator.reference(row)], generator.reference(row)
    methods["target_only_presence"], _ = old.presence(methods["target_only"], old.target_text(view))
    deps["target_only_presence"] = list(deps["target_only"])
    messages, spans = old.pointer_messages(view)
    selection = {"status": "no_candidate", "raw": None, "candidates": spans, "selected": None}
    methods["id_selection"], deps["id_selection"] = dict(NULL), []
    if spans:
        row = generator.generate(messages, "id_selection", PROTOCOL["pointer_max_new_tokens"])
        methods["id_selection"], selection["status"], selection["selected"] = old.parse_selection(row["raw_output"], spans, row["truncated"])
        selection["raw"] = row["raw_output"]
        deps["id_selection"], raw_keys["id_selection"] = [generator.reference(row)], generator.reference(row)
    for source, name in (("standard", "standard_gate_constrained"), ("target_only", "target_only_gate_constrained"),
                         ("id_selection", "id_gate_constrained")):
        candidate = methods[source]
        normalized = canonical_candidate(candidate)
        deps[name] = list(deps[source])
        if normalized is None:
            methods[name] = copy.deepcopy(candidate)
            gates[name] = {"status": "skipped_non_numeric", "ref": None, "candidate": None}
        else:
            row = generator.generate(old.relation_messages(view, normalized), name, PROTOCOL["relation_max_new_tokens"], mode="constrained")
            methods[name], status = old.relation_decision(candidate, row["raw_output"], row["truncated"])
            deps[name].append(generator.reference(row))
            gates[name] = {"status": status, "ref": generator.reference(row), "candidate": normalized["value"]}
    routed = "target_only_routed_gate_constrained"
    if spans:
        methods[routed] = copy.deepcopy(methods["target_only_gate_constrained"])
        deps[routed] = list(deps["target_only_gate_constrained"])
        gates[routed] = copy.deepcopy(gates["target_only_gate_constrained"])
    else:
        methods[routed], deps[routed] = dict(NULL), []
        gates[routed] = {"status": "no_candidate_cpu", "ref": None, "candidate": None}
    methods["field_parser"] = old.field_parser(view)
    methods["alias_field_parser"] = alias_field_parser(view)
    methods["always_abstain"] = dict(NULL)
    for name in METHODS:
        deps.setdefault(name, [])
    assert set(methods) == set(METHODS)
    served = {m: copy.deepcopy(p) if p["valid"] else dict(NULL) for m, p in methods.items()}
    return {"methods": methods, "served_methods": served, "dependencies": deps, "selection": selection,
            "gates": gates, "raw_refs": raw_keys}
