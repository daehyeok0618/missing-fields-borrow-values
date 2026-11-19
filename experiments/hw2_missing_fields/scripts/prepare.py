from __future__ import annotations

import itertools
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone

from .common import ROOT, REPO, cfg, render, sha, stable_hash, tokenizer, write_json, write_jsonl

LABELS = {"S1": ("Salary", "Budget"), "S2": ("Budget", "Salary"), "N0": ("Budget", "Revenue")}


def public_paragraphs():
    path = ROOT / "source/alice_gutenberg_11.txt"
    text = path.read_text(encoding="utf-8-sig")
    body = re.split(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n", text, maxsplit=1)[1]
    body = re.split(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", body, maxsplit=1)[0]
    pool = []
    for i, paragraph in enumerate(re.split(r"\n\s*\n", body)):
        paragraph = " ".join(paragraph.split())
        if len(paragraph) < 80 or re.search(r"\d|\b(salary|budget|revenue|usd)\b", paragraph, re.I):
            continue
        pool.append({"paragraph_id": i, "text": paragraph})
    if len(pool) < 100:
        raise RuntimeError({"corpus_too_small": len(pool)})
    write_json(ROOT / "source/source_manifest.json", {
        "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll",
        "catalog_url": "https://www.gutenberg.org/ebooks/11",
        "download_url": "https://www.gutenberg.org/ebooks/11.txt.utf-8",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha(path),
        "filtered_paragraphs": len(pool), "filter": ">=80 characters, no digits or whole-word salary/budget/revenue/USD",
        "transform": "collapse whitespace; final excerpt may stop at a word boundary"})
    write_jsonl(ROOT / "source/filtered_paragraphs.jsonl", pool)
    return pool


def take_text(pool, tok, rng, budget, used):
    candidates = [p for p in pool if p["paragraph_id"] not in used]
    rng.shuffle(candidates)
    chunks, provenance = [], []
    for p in candidates:
        available = p["text"]
        joined = "\n\n".join(chunks + [available])
        if len(tok.encode(joined, add_special_tokens=False)) <= budget:
            chunks.append(available)
            provenance.append({"paragraph_id": p["paragraph_id"], "text": available, "cut": False})
            used.add(p["paragraph_id"])
            continue
        words = available.split()
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            fragment = " ".join(words[:mid])
            n = len(tok.encode("\n\n".join(chunks + [fragment]), add_special_tokens=False))
            if n <= budget:
                lo = mid
            else:
                hi = mid - 1
        if lo:
            fragment = " ".join(words[:lo])
            chunks.append(fragment)
            provenance.append({"paragraph_id": p["paragraph_id"], "text": fragment, "cut": lo < len(words)})
            used.add(p["paragraph_id"])
        break
    text = "\n\n".join(chunks)
    if abs(len(tok.encode(text, add_special_tokens=False)) - budget) > 8:
        raise RuntimeError("neutral excerpt did not approach token budget")
    return text, provenance


def prior_values_names():
    values, names, files = set(), set(), []
    folders = [p for p in (REPO / "experiments").iterdir() if p.is_dir() and p != ROOT and
               any(w in p.name for w in ("donor", "missing_field", "attribute", "value_exposure"))]
    def walk(obj, key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, key)
        elif isinstance(obj, int) and not isinstance(obj, bool) and 10000 <= obj <= 999999:
            values.add(obj)
        elif isinstance(obj, str) and ("company" in key or "organization" in key):
            names.add(obj)
    for folder in folders:
        for path in sorted((folder / "data").glob("*base*.jsonl")):
            files.append(str(path.relative_to(REPO)))
            with path.open() as f:
                for line in f:
                    walk(json.loads(line))
    return values, names, files


def document(base, role, template, label=None, value=None):
    name, domain, region = base[role + "_name"], base[role + "_domain"], base[role + "_region"]
    if template == "KV":
        body = f"Organization: {name}\nActivity: {domain}\nRegion: {region}"
        if label:
            body += f"\n{label}: ${value}"
    elif template == "PROSE":
        body = f"{name} specializes in {domain} and operates in the {region}."
        if label:
            body += f" Its recorded {label.lower()} is ${value}."
    else:
        body = f"Organization memo: {name}\n- Activity: {domain}\n- Region: {region}"
        if label:
            body += f"\n- {label}: ${value}"
    return f"<document>\n{body}\n</document>"


def messages(base, template, family, assignment, order, placement, prefix, suffix, instruction):
    present = family in ("PRESENT", "T1")
    target = document(base, "target", template, "Salary" if present else None,
                      base["target_value"] if present else None)
    pre = f"<background>\n{prefix}\n</background>"
    post = f"<background>\n{suffix}\n</background>"
    blocks = [target, pre]
    donors = []
    if family not in ("T0", "T1"):
        labels = LABELS[assignment]
        d1 = document(base, "donor1", template, labels[0], base["v1"])
        d2 = document(base, "donor2", template, labels[1], base["v2"])
        donors = [d1, d2]
        block = "\n\n".join(donors if order == "AB" else donors[::-1])
        blocks += [block, post] if placement == "MIDDLE" else [post, block]
    else:
        blocks.append(post)
    question = f"Question: What is the recorded salary for {base['target_name']}?"
    user = "Documents and background:\n\n" + "\n\n".join(blocks) + "\n\n" + question
    return [{"role": "system", "content": instruction}, {"role": "user", "content": user}], target, donors, question


def spans(prompt, needles, tok):
    encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    result = {}
    for name, needle in needles.items():
        start = prompt.index(needle)
        end = start + len(needle)
        indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
        result[name] = [indices[0], indices[-1] + 1]
    return len(encoded["input_ids"]), result


def main():
    if (ROOT / "runs" / cfg()["run_id"] / "protocol_lock.json").exists():
        raise RuntimeError("Refusing to regenerate locked data")
    c = cfg(); tok = tokenizer(); pool = public_paragraphs()
    old_values, old_names, old_files = prior_values_names()
    used_values, used_names = set(old_values), set(old_names)
    domains = ["acoustic research", "botanical imaging", "cultural archiving", "water monitoring", "fabric testing", "map conservation"]
    regions = ["northern district", "southern district", "eastern district", "western district"]
    syllables = ["bel", "cor", "del", "fen", "gil", "hor", "jor", "kel", "lum", "mer", "nel", "por", "quil", "rin", "sel", "tor", "vel", "wyn", "xel", "zor"]
    all_requests, all_bases, backgrounds = {}, {}, []
    for split in ("dev", "main"):
        rng = random.Random(c["data"][split + "_seed"])
        bases, requests = [], []
        for i in range(c["data"][split + "_bases"]):
            base = {"base_id": f"HW2{'D' if split == 'dev' else 'M'}{i:03d}", "split": split,
                    "data_seed": c["data"][split + "_seed"], "index": i}
            for role in ("target", "donor1", "donor2"):
                while True:
                    name = "".join(rng.sample(syllables, 3)).capitalize() + " " + "".join(rng.sample(syllables, 2)).capitalize() + " Institute"
                    if name not in used_names:
                        break
                used_names.add(name)
                base[role + "_name"] = name
                base[role + "_domain"] = rng.choice(domains)
                base[role + "_region"] = rng.choice(regions)
            amounts = []
            while len(amounts) < 3:
                value = rng.randrange(100001, 199999)
                if value not in used_values:
                    used_values.add(value); amounts.append(value)
            low, high = sorted(amounts[:2])
            base["v1"], base["v2"] = (low, high) if i % 2 == 0 else (high, low)
            base["target_value"] = amounts[2]
            bases.append(base)
            for ti, template in enumerate(c["data"]["templates"]):
                bg_rng = random.Random(c["data"][split + "_seed"] * 10000 + i * 7 + ti)
                empty, target, donors, question = messages(base, template, "MISSING", "S1", "AB", "MIDDLE", "", "", c["system_instruction"])
                core_n, core_spans = spans(render(empty, tok), {"d1": donors[0], "d2": donors[1]}, tok)
                midpoint = (core_spans["d1"][0] + core_spans["d2"][1]) / 2
                prefix_budget = round(c["data"]["target_tokens"] * .5 - midpoint)
                suffix_budget = c["data"]["target_tokens"] - core_n - prefix_budget
                used = set()
                prefix, prefix_source = take_text(pool, tok, bg_rng, prefix_budget, used)
                suffix, suffix_source = take_text(pool, tok, bg_rng, suffix_budget, used)
                empty_t0, _, _, _ = messages(base, template, "T0", None, None, None, prefix, "", c["system_instruction"])
                solo_budget = c["data"]["target_tokens"] - len(tok.encode(render(empty_t0, tok), add_special_tokens=False))
                solo, solo_source = take_text(pool, tok, bg_rng, solo_budget, set(p["paragraph_id"] for p in prefix_source))
                bg_id = base["base_id"] + "_" + template
                backgrounds.append({"background_id": bg_id, "prefix": prefix_source, "suffix": suffix_source, "solo_suffix": solo_source})
                conditions = [(family, assn, order, placement) for family in ("MISSING", "PRESENT")
                              for assn, order, placement in itertools.product(("S1", "S2"), c["data"]["orders"], c["data"]["placements"])]
                conditions += [("N0", "N0", o, p) for o, p in itertools.product(c["data"]["orders"], c["data"]["placements"])]
                conditions += [("T0", None, None, None), ("T1", None, None, None)]
                for family, assignment, order, placement in conditions:
                    msg, target, donors, question = messages(base, template, family, assignment, order, placement,
                                                            prefix, solo if family in ("T0", "T1") else suffix, c["system_instruction"])
                    prompt = render(msg, tok)
                    needles = {"target": target, "question": question}
                    if donors:
                        needles.update({"donor1": donors[0], "donor2": donors[1], "v1": str(base["v1"]), "v2": str(base["v2"])})
                    count, token_spans = spans(prompt, needles, tok)
                    condition = "_".join(x for x in (family, assignment, order, placement) if x)
                    state = "PRESENT" if family in ("PRESENT", "T1") else "MISSING"
                    request = {**base, "request_id": f"{bg_id}_{condition}", "template": template, "family": family,
                               "assignment": assignment, "order": order, "placement": placement, "state": state,
                               "gold_value": str(base["target_value"]) if state == "PRESENT" else None,
                               "background_id": bg_id, "messages": msg, "input_hash": stable_hash(msg),
                               "input_token_count": count, "token_spans": token_spans,
                               "block_multiset_hash": stable_hash(sorted([target] + donors + [prefix, solo if family in ("T0", "T1") else suffix]))}
                    if donors:
                        start = min(token_spans[d][0] for d in ("donor1", "donor2"))
                        end = max(token_spans[d][1] for d in ("donor1", "donor2"))
                        request.update(donor_midpoint_fraction=(start + end) / (2 * count),
                                       donor_query_distance=token_spans["question"][0] - end)
                    requests.append(request)
            if i % 12 == 0:
                print(f"prepare {split}: {i+1}/{c['data'][split+'_bases']}", flush=True)
        all_bases[split], all_requests[split] = bases, requests
        write_jsonl(ROOT / f"data/{split}_bases.jsonl", bases)
        write_jsonl(ROOT / f"data/{split}_requests.jsonl", requests)
    write_jsonl(ROOT / "data/backgrounds.jsonl", backgrounds)
    checks = {}; lengths = {}; pair_counts = Counter()
    for split, requests in all_requests.items():
        index = {(r["base_id"], r["template"], r["family"], r["assignment"], r["order"], r["placement"]): r for r in requests}
        checks[split + "_counts"] = len(requests) == c["data"][split + "_calls"] == len({r["request_id"] for r in requests})
        checks[split + "_templates_crossed"] = all(sum(r["base_id"] == b["base_id"] for r in requests) == 66 for b in all_bases[split])
        errors = []
        for r in requests:
            n = r["input_token_count"]; ts = r["token_spans"]
            if abs(n - 2000) > c["data"]["length_tolerance"]:
                errors.append((r["request_id"], "length", n))
            if r["family"] not in ("T0", "T1"):
                if not ts["target"][1] <= min(ts["donor1"][0], ts["donor2"][0]):
                    errors.append((r["request_id"], "target_not_first"))
                for key in ("v1", "v2"):
                    if "\n".join(m["content"] for m in r["messages"]).count(str(r[key])) != 1:
                        errors.append((r["request_id"], "candidate_occurrence"))
                if r["placement"] == "MIDDLE":
                    if not .45 <= r["donor_midpoint_fraction"] <= .55:
                        errors.append((r["request_id"], "middle_fraction"))
                    a = index[(r["base_id"], r["template"], r["family"], r["assignment"], r["order"], "ADJACENT")]
                    if n != a["input_token_count"] or r["block_multiset_hash"] != a["block_multiset_hash"] or ts["question"] != a["token_spans"]["question"]:
                        errors.append((r["request_id"], "placement_pair"))
                    pair_counts["placement"] += 1
                else:
                    if r["donor_query_distance"] > 2:
                        errors.append((r["request_id"], "not_adjacent", r["donor_query_distance"]))
                if r["assignment"] == "S1":
                    s2 = index[(r["base_id"], r["template"], r["family"], "S2", r["order"], r["placement"])]
                    p1, p2 = (x["messages"][1]["content"] for x in (r, s2))
                    labelmask = lambda s: re.sub(r"\b(salary|budget)\b", "ATTR", s, flags=re.I)
                    if n != s2["input_token_count"] or any(ts[k] != s2["token_spans"][k] for k in ("v1", "v2")) or labelmask(p1) != labelmask(p2):
                        errors.append((r["request_id"], "assignment_pair"))
                    pair_counts["assignment"] += 1
            if r["state"] == "MISSING" and str(r["target_value"]) in r["messages"][1]["content"]:
                errors.append((r["request_id"], "hidden_target_leaked"))
        checks[split + "_structural"] = not errors
        lengths[split] = {"min": min(r["input_token_count"] for r in requests), "max": max(r["input_token_count"] for r in requests),
                          "mean": sum(r["input_token_count"] for r in requests)/len(requests), "errors": errors[:30]}
    bg_errors = []
    base_by_id = {b["base_id"]: b for bs in all_bases.values() for b in bs}
    for bg in backgrounds:
        b = base_by_id[bg["background_id"].rsplit("_", 1)[0]]
        for parts in (bg["prefix"], bg["suffix"], bg["solo_suffix"]):
            text = "\n\n".join(p["text"] for p in parts)
            if re.search(r"\d|\b(salary|budget|revenue|usd)\b", text, re.I) or any(b[role + "_name"].casefold() in text.casefold() for role in ("target", "donor1", "donor2")):
                bg_errors.append(bg["background_id"])
    checks["background_clean"] = not bg_errors
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "lengths": lengths,
             "pair_counts": dict(pair_counts), "prior_base_files_checked": old_files,
             "prior_values_excluded": len(old_values), "prior_names_excluded": len(old_names),
             "dev_main_values_and_names_disjoint": True,
             "counts": {s: {"bases": len(all_bases[s]), "requests": len(all_requests[s]),
                             "families": dict(Counter(r["family"] for r in all_requests[s]))} for s in all_requests}}
    write_json(ROOT / "data/data_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise RuntimeError("Dataset audit failed; no model inference permitted")


if __name__ == "__main__":
    main()
