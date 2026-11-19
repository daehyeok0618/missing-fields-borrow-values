"""Data and label-free methods for the Colab value-tracking development pilot.

This module has no model inference or file I/O. Methods consume ``public_view``
only; hidden values and experimental group labels belong to the evaluator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter

SEED = 2026110901
TEMPLATES = ("KV", "PROSE", "BULLET")
ALIASES = frozenset(("null", "na", "n/a", "none", "unknown", "not specified",
                     "not provided", "not available", "not stated", "unspecified"))
SYSTEMS = {
    "salary": 'Extract information only about the organization named in the question. '
              'Use only explicitly stated facts about that organization in the provided documents. '
              'A fact about a different organization is not evidence about the requested organization. '
              'If no document states the requested salary for the named organization, return exactly '
              '{"value": null}. If it is stated, return a JSON object with exactly one key, "value", '
              'whose value is a JSON string containing the stated amount in USD as digits without '
              'currency symbols or separators. Do not infer, estimate, or invent a missing amount. '
              'Return only the JSON object, without explanation or Markdown.',
    "year": 'Extract information only about the organization named in the question. '
            'Use only explicitly stated facts about that organization in the provided documents. '
            'A fact about a different organization is not evidence about the requested organization. '
            'If no document states the requested founding year for the named organization, return exactly '
            '{"value": null}. If it is stated, return a JSON object with exactly one key, "value", '
            'whose value is a JSON string containing the stated year as digits. '
            'Do not infer, estimate, or invent a missing year. '
            'Return only the JSON object, without explanation or Markdown.',
}
FORBIDDEN_BACKGROUND = re.compile(r"\d|\b(salary|budget|founding|audit|review|revenue|usd)\b", re.I)
NUMBER = re.compile(r"(?<![\w.,/+\-−])(?:\d{1,3}(?:,\d{3})+|\d+)(?!\w|[/+\-−]|[.,]\d)")


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw.strip(), object_pairs_hook=_unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_json_constant")))


def parse(raw):
    """Strict original protocol: one value key, nonempty string or literal null."""
    try:
        obj = _json(raw)
        if not isinstance(obj, dict) or set(obj) != {"value"}:
            raise ValueError("schema")
        value = obj["value"]
        if value is not None and (not isinstance(value, str) or not any(c.isalnum() for c in value)):
            raise ValueError("value_type_or_empty")
        return {"valid": True, "value": value, "error": None}
    except (ValueError, TypeError, AttributeError) as exc:
        return {"valid": False, "value": None,
                "error": "malformed_json" if isinstance(exc, json.JSONDecodeError) else str(exc)}


def abstains(parsed, policy="strict"):
    if policy not in ("strict", "semantic"):
        raise ValueError("unknown_policy")
    if not parsed["valid"]:
        return False
    value = parsed["value"]
    if value is None:
        return True
    return policy == "semantic" and " ".join(unicodedata.normalize("NFKC", value).casefold().split()) in ALIASES


def canonical_number(value):
    """Numeric matching is whole-value, and never removes arbitrary prose."""
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).strip()
    match = re.fullmatch(r"(?:(?:USD\s*)?\$?\s*)((?:\d{1,3}(?:,\d{3})+)|\d+)", text, re.I)
    return str(int(match.group(1).replace(",", ""))) if match else None


def numeric_spans(text):
    """Return original digit/comma spans, excluding embedded IDs and decimals."""
    return [{"start": m.start(), "end": m.end(), "text": m.group(),
             "value": str(int(m.group().replace(",", "")))} for m in NUMBER.finditer(text)]


def same_number(parsed, value):
    return bool(parsed["valid"] and canonical_number(parsed["value"]) == canonical_number(str(value)))


def public_view(sample):
    keys = ("target_id", "docs", "prefix", "suffix", "question", "task", "placement")
    return {key: copy.deepcopy(sample[key]) for key in keys}


def _document(doc):
    return '<document id="' + doc["id"] + '">\n' + doc["text"] + "\n</document>"


def make_messages(view, question=None):
    target = next(d for d in view["docs"] if d["id"] == view["target_id"])
    others = [d for d in view["docs"] if d["id"] != view["target_id"]]
    pre = "<background>\n" + view["prefix"] + "\n</background>"
    post = "<background>\n" + view["suffix"] + "\n</background>"
    donor_block = "\n\n".join(_document(d) for d in others)
    tail = [donor_block, post] if view["placement"] == "MIDDLE" else [post, donor_block]
    content = "Documents and background:\n\n" + "\n\n".join([_document(target), pre] + tail)
    content += "\n\n" + (question if question is not None else view["question"])
    return [{"role": "system", "content": SYSTEMS[view["task"]]}, {"role": "user", "content": content}]


def target_messages(view):
    """Practical target-only baseline: remove donors AND neutral filler."""
    target = next(d for d in view["docs"] if d["id"] == view["target_id"])
    content = "Documents and background:\n\n" + _document(target) + "\n\n" + view["question"]
    return [{"role": "system", "content": SYSTEMS[view["task"]]}, {"role": "user", "content": content}]


def question_variants(view):
    attribute = "salary" if view["task"] == "salary" else "founding year"
    return [f"Question: According to the documents, what {attribute} is explicitly recorded for {view['target_id']}?",
            f"Question: For {view['target_id']}, give its recorded {attribute}, or null if it is not stated."]


def _render(messages, tok):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _count(text, tok):
    return len(tok.encode(text, add_special_tokens=False))


def _span(prompt, needle, tok):
    start = prompt.index(needle)
    end = start + len(needle)
    encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    indices = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > start and a < end]
    if not indices:
        raise ValueError("empty_token_span")
    return [indices[0], indices[-1] + 1]


def _signature(view, tok):
    prompt = _render(make_messages(view), tok)
    return _count(prompt, tok), _span(prompt, view["question"], tok)


def prepare_probes(view, parsed, tok):
    """Create two interventions using public input and initial prediction only."""
    result = {"eligible": False, "reason": None, "replacements": [], "messages": [], "audit": {}}
    candidate = canonical_number(parsed["value"]) if parsed["valid"] else None
    if candidate is None:
        result["reason"] = "initial_not_valid_numeric"
        return result
    locations = []
    for index, doc in enumerate(view["docs"]):
        if doc["id"] != view["target_id"]:
            locations.extend((index, s) for s in numeric_spans(doc["text"]) if s["value"] == candidate)
    if not locations:
        result["reason"] = "no_non_target_match"
        return result
    original = make_messages(view)
    visible = "\n".join(m["content"] for m in original)
    visible_numbers = {s["value"] for s in numeric_spans(visible)}
    lo, hi = (1900, 2025) if view["task"] == "year" else (100001, 199998)
    if not lo <= int(candidate) <= hi:
        result["reason"] = "candidate_outside_task_domain"
        return result
    rng = random.Random(int(hashlib.sha256(visible.encode()).hexdigest()[:16], 16) ^ SEED)
    signature = _signature(view, tok)
    target = next(d for d in view["docs"] if d["id"] == view["target_id"])
    audits = []
    tried = set()
    for _ in range(256):
        replacement = str(rng.randint(lo, hi))
        if replacement in tried or replacement in visible_numbers or len(replacement) != len(candidate):
            continue
        tried.add(replacement)
        edited = copy.deepcopy(view)
        by_doc = {}
        for index, span in locations:
            by_doc.setdefault(index, []).append(span)
        spans_ok = True
        for index, spans in by_doc.items():
            text = view["docs"][index]["text"]
            for span in sorted(spans, key=lambda x: x["start"], reverse=True):
                rendered = f"{int(replacement):,}" if "," in span["text"] else replacement
                if _count(span["text"], tok) != _count(rendered, tok):
                    spans_ok = False
                text = text[:span["start"]] + rendered + text[span["end"]:]
            edited["docs"][index]["text"] = text
        if not spans_ok or _signature(edited, tok) != signature:
            continue
        target_after = next(d for d in edited["docs"] if d["id"] == view["target_id"])
        if target_after != target or edited["question"] != view["question"]:
            raise AssertionError("protected_input_changed")
        result["replacements"].append(replacement)
        result["messages"].append(make_messages(edited))
        audits.append({"target_unchanged": True, "query_unchanged": True,
                       "same_input_tokens": True, "same_question_token_span": True,
                       "changed_occurrences": len(locations)})
        if len(result["replacements"]) == 2:
            result.update(eligible=True, reason="two_valid_interventions")
            result["audit"] = {"input_tokens": signature[0], "question_token_span": signature[1],
                               "locations": [{"doc_id": view["docs"][i]["id"], **s} for i, s in locations],
                               "interventions": audits, "candidate_attempt_limit": 256}
            return result
    result.update(reason="no_two_token_matched_replacements", replacements=[], messages=[])
    result["audit"] = {"valid_candidates_found": len(audits), "candidate_attempt_limit": 256}
    return result


def verifier_messages(view, parsed):
    target = next(d for d in view["docs"] if d["id"] == view["target_id"])
    instruction = ('Verify the candidate using only the target document. Return exactly '
                   '{"answerable": true, "evidence": "an exact supporting span copied from the target document"} '
                   'if that document explicitly assigns the requested attribute and candidate value to the target. '
                   'Otherwise return exactly {"answerable": false, "evidence": null}. '
                   'A different field containing the same number is not support. Return JSON only.')
    body = _document(target) + "\n\n" + view["question"] + "\nCandidate: " + json.dumps(parsed["value"])
    return [{"role": "system", "content": instruction}, {"role": "user", "content": body}]


def evidence_accept(view, parsed, raw_verifier):
    """Exact-span/attribute co-occurrence heuristic, not semantic entailment."""
    candidate = canonical_number(parsed["value"]) if parsed["valid"] else None
    if candidate is None:
        return False, "initial_not_valid_numeric"
    try:
        obj = _json(raw_verifier)
        if not isinstance(obj, dict) or set(obj) != {"answerable", "evidence"} or type(obj["answerable"]) is not bool:
            return False, "invalid_verifier_schema"
        if obj["answerable"] is not True:
            return False, "verifier_rejected"
        evidence = obj["evidence"]
        target = next(d for d in view["docs"] if d["id"] == view["target_id"])
        if not isinstance(evidence, str) or not evidence or evidence not in target["text"]:
            return False, "not_exact_target_span"
        label = r"\bsalary\b" if view["task"] == "salary" else r"\bfounding\s+year\b"
        for clause in re.split(r"[\n.!?;]", evidence):
            if re.search(label, clause, re.I) and candidate in {s["value"] for s in numeric_spans(clause)}:
                return True, "exact_target_span_attribute_value_cooccurrence"
        return False, "missing_local_attribute_value_pair"
    except (ValueError, TypeError, AttributeError):
        return False, "invalid_verifier_json"


def _pool(corpus_text):
    body = re.split(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n", corpus_text, maxsplit=1)
    body = body[-1]
    body = re.split(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", body, maxsplit=1)[0]
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body)]
    result = [(i, p) for i, p in enumerate(paragraphs) if len(p) >= 80 and not FORBIDDEN_BACKGROUND.search(p)]
    if len(result) < 50:
        raise ValueError("background_corpus_too_small")
    return result


def _take(pool, tok, budget, rng, used):
    candidates = [p for p in pool if p[0] not in used]
    rng.shuffle(candidates)
    chunks, provenance = [], []
    for pid, text in candidates:
        combined = "\n\n".join(chunks + [text])
        if _count(combined, tok) <= budget:
            chunks.append(text); provenance.append(pid); used.add(pid)
            continue
        words = text.split()
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _count("\n\n".join(chunks + [" ".join(words[:mid])]), tok) <= budget:
                lo = mid
            else:
                hi = mid - 1
        if lo:
            chunks.append(" ".join(words[:lo])); provenance.append(pid); used.add(pid)
        break
    result = "\n\n".join(chunks)
    if abs(_count(result, tok) - budget) > 12:
        raise ValueError("background_budget_not_met")
    return result, provenance


def _body(name, template, label=None, value=None, task="salary"):
    amount = "$" + str(value) if task == "salary" else str(value)
    if template == "KV":
        text = f"Organization: {name}\nActivity: acoustic research\nRegion: northern district"
        return text + (f"\nRecorded {label}: {amount}" if label else "")
    if template == "PROSE":
        text = f"{name} specializes in acoustic research and operates in the northern district."
        return text + (f" Its recorded {label.lower()} is {amount}." if label else "")
    text = f"Organization memo: {name}\n- Activity: acoustic research\n- Region: northern district"
    return text + (f"\n- Recorded {label}: {amount}" if label else "")


def build_dataset(tok, corpus_text, prior_inventory=None):
    """Generate 384 inputs and separate evaluator labels, deterministically."""
    inventory = prior_inventory or {}
    old_names = {str(x).casefold() for x in inventory.get("names", [])}
    old_values = {str(x) for x in inventory.get("salary_values", inventory.get("values", []))}
    old_triples = {tuple(sorted(int(v) for v in x)) for x in inventory.get("year_tuples", [])}
    used_names, used_values, used_triples = set(old_names), set(old_values), set(old_triples)
    pool = _pool(corpus_text)
    rng = random.Random(SEED)
    syllables = ["bel", "cor", "del", "fen", "gil", "hor", "jor", "kel", "lum", "mer", "nel", "por", "quil", "rin", "sel", "tor", "vel", "wyn", "xel", "zor"]
    samples, gold, provenance = [], {}, []
    def new_name():
        for _ in range(10000):
            name = "".join(rng.sample(syllables, 3)).capitalize() + " " + "".join(rng.sample(syllables, 3)).capitalize() + " Institute"
            if name.casefold() not in used_names:
                used_names.add(name.casefold())
                return name
        raise ValueError("name_space_exhausted")
    for task in ("salary", "year"):
        for bi in range(20):
            base_id = f"VTP_{task.upper()}_{bi:03d}"
            names = [new_name() for _ in range(3)]
            if task == "salary":
                values = []
                while len(values) < 3:
                    value = str(rng.randint(100001, 199998))
                    if value not in used_values:
                        used_values.add(value); values.append(value)
            else:
                while True:
                    values = [str(v) for v in rng.sample(range(1900, 2026), 3)]
                    triple = tuple(sorted(map(int, values)))
                    if triple not in used_triples:
                        used_triples.add(triple); break
            v1, v2, target_value = values
            groups = ("basic_missing", "basic_present") if bi < 12 else (("collision_present",) if bi < 16 else ("otherfield_missing",))
            label, competing = ("salary", "budget") if task == "salary" else ("founding year", "audit year")
            for template in TEMPLATES:
                for group in groups:
                    target_label = label if group in ("basic_present", "collision_present") else (competing if group == "otherfield_missing" else None)
                    tv = v1 if group in ("collision_present", "otherfield_missing") else target_value
                    docs = [{"id": names[0], "text": _body(names[0], template, target_label, tv, task)},
                            {"id": names[1], "text": _body(names[1], template, label, v1, task)},
                            {"id": names[2], "text": _body(names[2], template, competing, v2, task)}]
                    question = f"Question: What is the recorded {label} for {names[0]}?"
                    view = {"target_id": names[0], "docs": docs, "prefix": "", "suffix": "",
                            "question": question, "task": task, "placement": "MIDDLE"}
                    empty_prompt = _render(make_messages(view), tok)
                    ds = _span(empty_prompt, _document(docs[1]), tok)
                    de = _span(empty_prompt, _document(docs[2]), tok)
                    midpoint = (ds[0] + de[1]) / 2
                    budget_prefix = int(1000 - midpoint)
                    bg_rng = random.Random(int(hashlib.sha256(f"{base_id}:{template}:{group}".encode()).hexdigest()[:16], 16))
                    used = set()
                    view["prefix"], pre_ids = _take(pool, tok, budget_prefix, bg_rng, used)
                    prefix_prompt = _render(make_messages(view), tok)
                    suffix_budget = 2000 - _count(prefix_prompt, tok)
                    view["suffix"], post_ids = _take(pool, tok, suffix_budget, bg_rng, used)
                    provenance.append({"base_id": base_id, "template": template, "group": group,
                                       "prefix_paragraphs": pre_ids, "suffix_paragraphs": post_ids})
                    pair = []
                    for placement in ("MIDDLE", "ADJACENT"):
                        view["placement"] = placement
                        messages = make_messages(view)
                        prompt = _render(messages, tok)
                        n = _count(prompt, tok)
                        rid = f"{base_id}_{template}_{group}_{placement}"
                        sample = {"id": rid, "base_id": base_id, "template": template, "group": group,
                                  **copy.deepcopy(view), "messages": messages, "input_tokens": n}
                        if abs(n - 2000) > 64:
                            raise ValueError((rid, "not_2000_tokens", n))
                        if placement == "MIDDLE":
                            a = _span(prompt, _document(docs[1]), tok)[0]
                            b = _span(prompt, _document(docs[2]), tok)[1]
                            if not .45 <= (a + b) / (2 * n) <= .55:
                                raise ValueError((rid, "middle_position"))
                        pair.append((n, _span(prompt, question, tok)))
                        samples.append(sample)
                        gold[rid] = {"id": rid, "base_id": base_id, "task": task, "template": template,
                                     "placement": placement, "group": group,
                                     "value": tv if group in ("basic_present", "collision_present") else None,
                                     "non_target_values": [v1, v2]}
                    if pair[0] != pair[1]:
                        raise ValueError((base_id, template, group, "placement_not_token_matched", pair))
    if len(samples) != 384 or len({s["base_id"] for s in samples}) != 40 or len(gold) != 384:
        raise AssertionError("dataset_counts")
    audit = {"status": "PASS", "seed": SEED, "n_inputs": len(samples), "n_bases": 40,
             "group_counts": dict(Counter(s["group"] for s in samples)),
             "task_counts": dict(Counter(s["task"] for s in samples)),
             "input_token_min": min(s["input_tokens"] for s in samples),
             "input_token_max": max(s["input_tokens"] for s in samples),
             "placement_pairs_token_matched": True, "middle_bounds": [.45, .55],
             "background_sha256": hashlib.sha256(corpus_text.encode()).hexdigest(),
             "background_filtered_paragraphs": len(pool), "background_provenance": provenance,
             "prior_names_checked": len(old_names), "prior_salary_values_checked": len(old_values),
             "prior_year_tuples_checked": len(old_triples),
             "year_individual_values_may_repeat": True,
             "method_input_excludes_labels": True,
             "target_only_removes_neutral_filler": True}
    return samples, gold, audit
