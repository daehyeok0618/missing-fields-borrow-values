"""Generate the full prespecified heldout factorial without reading model predictions."""
import copy
from collections import Counter
import hashlib
from itertools import combinations
import random

from .common import HERE, digest, read_json
from .core import old
from .parent_source.upstream import legacy_core as legacy
from .protocol import ALIASES, GROUPS, PROTOCOL


def inventory(development):
    prior = read_json(HERE / "source/prior_inventory.json")
    names = {s.casefold() for s in prior["names"]}
    amounts = set(map(str, prior["salary_values"]))
    triples = {tuple(sorted(map(str, t))) for t in prior["year_tuples"]}
    for s in development:
        names.update(d["id"].casefold() for d in s["docs"])
        if s["task"] == "salary":
            amounts.update(x["value"] for d in s["docs"] for x in old.numeric_spans(d["text"]))
    years_by_base = {}
    for s in development:
        if s["task"] == "year":
            years_by_base.setdefault(s["base_id"], set()).update(x["value"] for d in s["docs"] for x in old.numeric_spans(d["text"]))
    pairs = {tuple(sorted(c)) for triple in triples for c in combinations(triple, 2)}
    for values in years_by_base.values():
        pairs.update(tuple(sorted(c)) for c in combinations(values, 2))
        triples.update(tuple(sorted(c)) for c in combinations(values, 3))
    return names, amounts, triples, pairs


def body(name, template, facts, task):
    text = legacy._body(name, template, task=task)
    for phrase, value in facts:
        amount = "$" + str(value) if task == "salary" else str(value)
        if template == "PROSE":
            text += " Its recorded " + phrase + " is " + amount + "."
        else:
            text += ("\n- Recorded " if template == "BULLET" else "\nRecorded ") + phrase + ": " + amount
    return text


def phrase(task, kind, style, index):
    options = ALIASES[task][kind]
    return options[0] if style == "canonical" else options[1 + index % (len(options) - 1)]


def question(task, name, style):
    if style == "canonical":
        field = "salary" if task == "salary" else "founding year"
        return "Question: What is the recorded " + field + " for " + name + "?"
    return ("Question: What amount is stated as annual pay for " + name + "?" if task == "salary"
            else "Question: In which year was " + name + " founded?")


def facts_for(group, values, slot):
    if group == "basic_missing":
        return [], None
    if group == "basic_present":
        return [("requested", values[2])], values[2]
    if group == "collision_present":
        return [("requested", values[0])], values[0]
    if group == "otherfield_missing":
        return [("competing", values[0])], None
    labels = ["competing", "competing"]
    labels[slot] = "requested" if group == "multi_present" else "third"
    return list(zip(labels, values[:2])), values[slot] if group == "multi_present" else None


def build_holdout(tok, development):
    old_names, old_amounts, old_triples, old_pairs = inventory(development)
    used_names, used_amounts, used_triples, used_pairs = set(old_names), set(old_amounts), set(old_triples), set(old_pairs)
    rng = random.Random(PROTOCOL["data_seed"])
    syllables = ("bel", "cor", "del", "fen", "gil", "hor", "jor", "kel", "lum", "mer", "nel", "por", "quil", "rin", "sel", "tor", "vel", "wyn", "xel", "zor")
    pool = legacy._pool((HERE / "source/alice_gutenberg_11.txt").read_text(encoding="utf-8-sig"))
    samples, gold, family_meta = [], {}, []

    def new_name():
        for _ in range(10000):
            result = "".join(rng.sample(syllables, 3)).capitalize() + " " + "".join(rng.sample(syllables, 3)).capitalize() + " Institute"
            if result.casefold() not in used_names:
                used_names.add(result.casefold())
                return result
        raise RuntimeError("Name space exhausted")

    for task in ("salary", "year"):
        for index in range(PROTOCOL["holdout_bases_per_task"]):
            base = "HLD_" + task.upper() + "_" + str(index).zfill(3)
            names = [new_name() for _ in range(3)]
            if task == "salary":
                values = []
                while len(values) < 3:
                    value = str(rng.randint(100001, 199998))
                    if value not in used_amounts:
                        used_amounts.add(value); values.append(value)
            else:
                for _ in range(100000):
                    values = list(map(str, rng.sample(range(1900, 2026), 3)))
                    triple = tuple(sorted(values))
                    pairs = {tuple(sorted(c)) for c in combinations(values, 2)}
                    if triple not in used_triples and not (pairs & used_pairs):
                        used_triples.add(triple); used_pairs.update(pairs)
                        break
                else:
                    raise RuntimeError("Year combination exclusion failed")
            slot = index % 2
            family_meta.append({"base_id": base, "task": task, "entities": names, "values": values, "requested_multi_slot": slot})
            for template_index, template in enumerate(legacy.TEMPLATES):
                for group in GROUPS:
                    facts, answer = facts_for(group, values, slot)
                    views = {}
                    for style in ("canonical", "paraphrase"):
                        def render(facts):
                            return [(phrase(task, kind, style, index + template_index), value) for kind, value in facts]
                        docs = [{"id": names[0], "text": body(names[0], template, render(facts), task)},
                                {"id": names[1], "text": body(names[1], template, render([("requested", values[0])]), task)},
                                {"id": names[2], "text": body(names[2], template, render([("competing", values[1])]), task)}]
                        views[style] = {"target_id": names[0], "docs": docs, "prefix": "", "suffix": "",
                                        "question": question(task, names[0], style), "task": task, "placement": "MIDDLE"}
                    view = views["canonical"]
                    prompt = legacy._render(old.make_messages(view), tok)
                    a = legacy._span(prompt, legacy._document(view["docs"][1]), tok)[0]
                    b = legacy._span(prompt, legacy._document(view["docs"][2]), tok)[1]
                    background_rng = random.Random(int(hashlib.sha256((base + template + group).encode()).hexdigest()[:16], 16))
                    used = set()
                    prefix, _ = legacy._take(pool, tok, int(1000 - (a + b) / 2), background_rng, used)
                    view["prefix"] = prefix
                    suffix, _ = legacy._take(pool, tok, 2000 - legacy._count(legacy._render(old.make_messages(view), tok), tok), background_rng, used)
                    for style, view in views.items():
                        view["prefix"], view["suffix"] = prefix, suffix
                        placement_check = []
                        for placement in ("MIDDLE", "ADJACENT"):
                            view["placement"] = placement
                            identifier = base + "_" + style + "_" + template + "_" + group + "_" + placement
                            messages = old.make_messages(view)
                            prompt = legacy._render(messages, tok)
                            n = legacy._count(prompt, tok)
                            if abs(n - 2000) > 64 or n > PROTOCOL["max_prompt_tokens"]:
                                raise RuntimeError((identifier, "Unexpected prompt length", n))
                            placement_check.append((n, legacy._span(prompt, view["question"], tok)))
                            spans = old.numeric_spans(old.target_text(view))
                            expected_spans = 0 if group == "basic_missing" else (2 if group.startswith("multi_") else 1)
                            if len(spans) != expected_spans:
                                raise RuntimeError("Unexpected target numeric occurrence count")
                            support = [{"start": s["start"], "end": s["end"]} for s in spans if answer is not None and s["value"] == answer]
                            if (answer is not None) != bool(support):
                                raise RuntimeError("Support construction failed")
                            samples.append({"id": identifier, "base_id": base, "split": "holdout_" + style,
                                            "group": group, "template": template, **copy.deepcopy(view),
                                            "input_tokens": n})
                            gold[identifier] = {"value": answer, "support_spans": support, "non_target_values": values[:2]}
                        if placement_check[0] != placement_check[1]:
                            raise RuntimeError("Placement pair length or question position changed")
    if len(samples) != PROTOCOL["expected_holdout_inputs"] or len(gold) != len(samples):
        raise RuntimeError("Holdout size mismatch")
    # All six cases and both styles remain in each independent base during resampling.
    counts = Counter((s["task"], s["split"], s["group"]) for s in samples)
    fresh_names = {n.casefold() for family in family_meta for n in family["entities"]}
    fresh_amounts = {v for family in family_meta if family["task"] == "salary" for v in family["values"]}
    fresh_triples = {tuple(sorted(family["values"])) for family in family_meta if family["task"] == "year"}
    fresh_pairs = {tuple(sorted(pair)) for triple in fresh_triples for pair in combinations(triple, 2)}
    overlap = {"entities": len(fresh_names & old_names), "salary_values": len(fresh_amounts & old_amounts),
               "year_pairs": len(fresh_pairs & old_pairs), "year_triples": len(fresh_triples & old_triples)}
    if any(overlap.values()) or len(fresh_names) != 3 * len(family_meta):
        raise RuntimeError("Holdout exclusion or entity uniqueness failed")
    audit = {"status": "PASS", "n_inputs": len(samples), "n_base": len(family_meta),
             "counts": {"/".join(k): v for k, v in counts.items()}, "family_inventory": family_meta,
             "overlap_counts_computed": overlap,
             "entity_overlap_with_history": 0, "salary_value_overlap_with_history": 0,
             "year_pair_and_triple_overlap_with_history": 0, "individual_years_can_repeat": True,
             "placement_length_and_question_matched": True, "styles_share_facts_and_background": True,
             "prompt_length_min": min(s["input_tokens"] for s in samples), "prompt_length_max": max(s["input_tokens"] for s in samples),
             "samples_digest": digest(samples), "gold_digest": digest(gold), "model_outputs_used_in_generation": False,
             "scope": PROTOCOL["holdout_scope"]}
    return samples, gold, audit
