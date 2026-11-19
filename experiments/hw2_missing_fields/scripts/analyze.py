from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from .common import ROOT, cfg, classify, delta_attr, offline, parse, read_jsonl, run_dir, sha, verify_lock, write_json, write_jsonl

CATEGORIES = ("ABSTAIN", "DONOR_MATCH", "OTHER_IN_CONTEXT", "CONTEXT_ABSENT", "INVALID")


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def estimate(vector, indices):
    vector = np.asarray(vector, dtype=float)
    draws = vector[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return {"estimate": float(vector.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def base_effects(rows, templates=None, orders=None):
    rows = [r for r in rows if r["family"] == "MISSING" and
            (templates is None or r["template"] in templates) and (orders is None or r["order"] in orders)]
    bases = sorted({r["base_id"] for r in rows})
    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base_id"]].append(r)
    output = []
    for bid in bases:
        rs = by_base[bid]
        pairs = defaultdict(dict)
        for r in rs:
            pairs[(r["template"], r["order"], r["placement"])][r["assignment"]] = r
        attr, forward, reverse = [], [], []
        for pair in pairs.values():
            a, b = pair["S1"], pair["S2"]
            attr.append(delta_attr(a["parse"], b["parse"], a["v1"], a["v2"]))
            av = a["parse"]["normalized_value"] if a["parse"]["valid"] else None
            bv = b["parse"]["normalized_value"] if b["parse"]["valid"] else None
            forward.append(av == str(a["v1"]) and bv == str(a["v2"]))
            reverse.append(av == str(a["v2"]) and bv == str(a["v1"]))
        effect = {"base_id": bid, "delta_attr": float(np.mean(attr)),
                  "forward_switch": float(np.mean(forward)), "reverse_switch": float(np.mean(reverse))}
        for placement in ("MIDDLE", "ADJACENT"):
            subset = [r for r in rs if r["placement"] == placement]
            for short, category in (("C", "DONOR_MATCH"), ("U", "CONTEXT_ABSENT"), ("I", "INVALID"), ("A", "ABSTAIN")):
                effect[f"{short}_{placement}"] = float(np.mean([r["category"] == category for r in subset]))
        effect["delta_C"] = effect["C_ADJACENT"] - effect["C_MIDDLE"]
        effect["delta_U"] = effect["U_ADJACENT"] - effect["U_MIDDLE"]
        effect["delta_type"] = effect["delta_C"] - effect["delta_U"]
        output.append(effect)
    return output


def main():
    offline(); verify_lock(); c = cfg(); out = ROOT / "results"
    requests = read_jsonl(ROOT / "data/main_requests.jsonl")
    raw = read_jsonl(run_dir() / "main_raw.jsonl")
    raws = {r["request_id"]: r for r in raw}
    if len(raw) != len(raws) or set(raws) != {r["request_id"] for r in requests}:
        raise RuntimeError("Incomplete or duplicate outputs")
    rows = []; replay_errors = []
    for request in requests:
        record = raws[request["request_id"]]
        parsed = parse(record["raw_output"])
        if parsed != record["parse"] or record["input_hash"] != request["input_hash"]:
            replay_errors.append(request["request_id"])
        category = classify(request, parsed)
        correct = parsed["valid"] and parsed["normalized_value"] == request["gold_value"]
        rows.append({**{k: v for k, v in request.items() if k not in ("messages",)}, **record,
                     "category": category, "correct": correct})
    if replay_errors or any(r["system_error"] is not None for r in rows):
        raise RuntimeError({"replay_errors": replay_errors})
    write_jsonl(out / "request_level.jsonl", rows)
    base_ids = sorted({r["base_id"] for r in rows})
    if len(base_ids) != c["data"]["main_bases"]:
        raise RuntimeError("Unexpected number of bases")
    indices = np.random.default_rng(c["analysis"]["bootstrap_seed"]).integers(
        0, len(base_ids), size=(c["analysis"]["bootstrap_repetitions"], len(base_ids)))
    np.save(out / "bootstrap_indices.npy", indices)
    write_json(out / "bootstrap_manifest.json", {"seed": c["analysis"]["bootstrap_seed"], "repetitions": len(indices),
               "base_ids": base_ids, "unit": "base; keep all renderings/conditions together", "sha256": sha(out / "bootstrap_indices.npy")})
    effects = base_effects(rows)
    csv_file(out / "base_effects.csv", effects)
    contrasts = []
    for subset, templates, orders in [("ALL", None, None)] + [(f"template={t}", [t], None) for t in c["data"]["templates"]] + \
        [(f"order={o}", None, [o]) for o in c["data"]["orders"]] + \
        [(f"without_template={t}", [a for a in c["data"]["templates"] if a != t], None) for t in c["data"]["templates"]]:
        vals = base_effects(rows, templates, orders)
        if [v["base_id"] for v in vals] != base_ids:
            raise RuntimeError("Subgroup base identity mismatch")
        for metric in ("delta_attr", "delta_C", "delta_U", "delta_type", "forward_switch", "reverse_switch"):
            contrasts.append({"subset": subset, "metric": metric,
                              "role": "PRIMARY" if subset == "ALL" and metric in ("delta_attr", "delta_C", "delta_type") else "SECONDARY",
                              **estimate([v[metric] for v in vals], indices), "n_base": len(vals), "bootstrap_repetitions": len(indices)})
    csv_file(out / "contrasts.csv", contrasts)
    primary = {r["metric"]: r for r in contrasts if r["subset"] == "ALL"}
    h1 = primary["delta_attr"]["ci95_low"] > 0
    h2 = primary["delta_C"]["ci95_low"] > 0 and primary["delta_type"]["ci95_low"] > 0
    # Counts for every factorial cell, and base-cluster CI for pooled outcomes.
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("family", "template", "assignment", "order", "placement"))].append(row)
    cells = []
    for key, rs in sorted(groups.items(), key=lambda item: str(item[0])):
        counts = Counter(r["category"] for r in rs)
        cells.append({**dict(zip(("family", "template", "assignment", "order", "placement"), key)),
                      "n": len(rs), "correct_n": sum(r["correct"] for r in rs),
                      **{cat + "_n": counts[cat] for cat in CATEGORIES + ("TARGET_CORRECT",)}})
    csv_file(out / "condition_counts.csv", cells)
    summary_rows = []
    for family in ("MISSING", "N0", "PRESENT", "T0", "T1"):
        for placement in (("MIDDLE", "ADJACENT") if family in ("MISSING", "N0", "PRESENT") else (None,)):
            rs = [r for r in rows if r["family"] == family and r["placement"] == placement]
            categories = CATEGORIES + (("TARGET_CORRECT",) if family in ("PRESENT", "T1") else ())
            for category in categories:
                per_base = [np.mean([r["category"] == category for r in rs if r["base_id"] == bid]) for bid in base_ids]
                summary_rows.append({"family": family, "placement": placement, "category": category,
                                     "n": len(rs), "count": sum(r["category"] == category for r in rs),
                                     **estimate(per_base, indices)})
    csv_file(out / "outcome_rates.csv", summary_rows)
    controls = {}
    for family in ("T0", "T1", "PRESENT", "N0"):
        rs = [r for r in rows if r["family"] == family]
        per_base = [np.mean([r["correct"] for r in rs if r["base_id"] == bid]) for bid in base_ids]
        controls[family] = {"correct": sum(r["correct"] for r in rs), "n": len(rs), **estimate(per_base, indices)}
    missing = [r for r in rows if r["family"] == "MISSING"]
    u_count = sum(r["category"] == "CONTEXT_ABSENT" for r in missing)
    zero_notes = []
    for result in summary_rows:
        if result["count"] == 0:
            zero_notes.append({"family": result["family"], "placement": result["placement"], "category": result["category"],
                               "zero_event_bases": len(base_ids), "base_any_event_one_sided_95_upper": 1 - .05 ** (1 / len(base_ids)),
                               "note": "Upper bound is for at least one event in a base cluster, not per-request risk."})
    runtime = json.loads((run_dir() / "main_runtime.json").read_text())
    summary = {"status": "COMPLETE", "n_base": len(base_ids), "n_requests": len(rows), "primary_requests": len(missing),
               "h1_supported": h1, "h2_supported": h2, "primary": primary, "controls": controls,
               "strict_valid": sum(r["parse"]["valid"] for r in rows), "truncated": sum(r["truncated"] for r in rows),
               "invalid_types": dict(Counter(r["parse"]["error"] for r in rows if not r["parse"]["valid"])),
               "context_absent_count_primary": u_count, "context_absent_sparse_under_20": u_count < 20,
               "zero_event_notes": zero_notes, "runtime": runtime, "parser_replay_mismatches": 0,
               "protocol_unchanged": True, "new_model_inference": True}
    write_json(out / "summary.json", summary)
    # Deterministic examples: first request ID per category, no effect-based case selection.
    examples = []
    for category in CATEGORIES:
        rs = sorted((r for r in missing if r["category"] == category), key=lambda r: r["request_id"])
        if rs:
            r = rs[0]
            request = next(q for q in requests if q["request_id"] == r["request_id"])
            examples.append({**r, "messages": request["messages"]})
    write_json(out / "representative_cases.json", examples)
    figures(summary_rows, contrasts)
    report(summary, summary_rows, contrasts, examples)
    verify_lock()
    artifacts = [p for folder in (out, ROOT / "figures", run_dir()) for p in folder.iterdir() if p.is_file()]
    artifacts += [ROOT / "final_report_ko.md", ROOT / "reproduce.md", ROOT / "ai_usage_log.md"]
    write_json(ROOT / "completion_manifest.json", {"status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "run_id": c["run_id"], "counts": {"main": len(rows), "primary": len(missing), "bases": len(base_ids)},
               "h1_supported": h1, "h2_supported": h2, "protocol_lock_sha256": sha(run_dir() / "protocol_lock.json"),
               "artifacts": {str(p.relative_to(ROOT)): sha(p) for p in artifacts}})
    print(json.dumps({k: summary[k] for k in ("status", "n_requests", "h1_supported", "h2_supported", "strict_valid", "controls", "primary")}, indent=2))


def figures(outcomes, contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#7d9b76", "#db7949", "#baaa80", "#826ba1", "#a5acb0"]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bottom = np.zeros(2)
    for category, color in zip(CATEGORIES, colors):
        vals = [next(r["estimate"] for r in outcomes if r["family"] == "MISSING" and r["placement"] == p and r["category"] == category) for p in ("MIDDLE", "ADJACENT")]
        ax.bar([0, 1], np.array(vals)*100, bottom=bottom, color=color, label=category.replace("_", " "), width=.58)
        bottom += np.array(vals)*100
    ax.set(xticks=[0, 1], xticklabels=["Middle", "Adjacent"], ylabel="All planned requests (%)", ylim=(0, 100), title="Missing target salary: output categories")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=8)
    fig.savefig(ROOT / "figures/outcome_rates.png", dpi=200); fig.savefig(ROOT / "figures/outcome_rates.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metrics, title in ((axes[0], ["delta_attr"], "Attribute assignment (H1)"),
                                (axes[1], ["delta_C", "delta_U", "delta_type"], "Adjacent minus middle (H2)")):
        rs = [next(r for r in contrasts if r["subset"] == "ALL" and r["metric"] == m) for m in metrics]
        y = np.array([r["estimate"] for r in rs])*100
        errs = np.array([[r["estimate"]-r["ci95_low"] for r in rs], [r["ci95_high"]-r["estimate"] for r in rs]])*100
        ax.errorbar(range(len(rs)), y, yerr=errs, fmt="o", capsize=5, color="#365f7d")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(xticks=list(range(len(rs))), xticklabels=metrics, ylabel="Percentage points (95% CI)", title=title)
        ax.margins(x=.4)
    fig.savefig(ROOT / "figures/hypothesis_contrasts.png", dpi=200); fig.savefig(ROOT / "figures/hypothesis_contrasts.pdf"); plt.close(fig)


def report(s, rates, contrasts, examples):
    def effect(m):
        r = s["primary"][m]
        return f"{100*r['estimate']:+.2f}%p [{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}]"
    lines = ["# HW1 기반 누락 필드·속성 교환·길이 통제 위치 비교 — 실행 보고서", "",
             "이 문서는 실제 실행 결과와 해석을 보존하는 연구 기록이다. 학생 제출용 완성 원고가 아니다.", "",
             "## 실행과 판정", "",
             f"새 base 96개를 각각 세 template로 표현하고, 개발 792회와 본실험 {s['n_requests']:,}회를 실제 실행했다. 본실험 중 H1/H2의 primary factorial은 2,304회다.",
             f"사전 기준에 따른 H1은 **{'지지' if s['h1_supported'] else '지지되지 않음'}**, H2는 **{'지지' if s['h2_supported'] else '지지되지 않음'}**이다.", "",
             "| 지표 | 추정치 및 95% paired base-bootstrap CI |", "|---|---:|"]
    for m in ("delta_attr", "delta_C", "delta_U", "delta_type"):
        lines.append(f"| {m} | {effect(m)} |")
    lines += ["", "H1은 delta_attr의 CI 하한이 0보다 커야 한다. H2는 delta_C와 delta_type의 CI 하한이 **모두** 0보다 커야 한다. CI가 0을 포함하면 예측 방향이 확립되지 않은 것이며, 전 구간 음수이면 해당 방향에 반하는 근거다.",
              "", "## 제출한 HW1과 구현의 대응", "",
              "- Target 문서는 항상 처음, 질문은 마지막이다. Middle의 donor block 중점은 전체 chat-template 입력의 45–55%에 있고 Adjacent는 질문 바로 앞에 있다.",
              "- 같은 문서·중립 prefix·중립 suffix를 재배열했으며 placement pair와 label-swap pair의 토큰 수를 정확히 맞췄다.",
              "- 같은 96개 base에 세 template를 모두 적용했다. 표본 수는 2,304가 아니라 96개 독립 base다.",
              "- 문자열/null JSON을 비제약 greedy로 생성했다. Invalid도 전체 분모에 포함했다. 정수/null grammar 또는 donor enum은 사용하지 않았다.",
              "- 모델 revision은 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, FP16, SDPA, repetition_penalty 1.1이다.",
              "- 계획의 Colab T4 대신 로컬 RTX A6000 한 장을 사용했다. 모델·질문·가설은 동일 연구 범위이며 실제 하드웨어 차이를 공개한다.",
              "- 중립 문단은 [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11)의 영어 소설 발췌다. 숫자·조작 속성어·개체명 오염을 검사했고 출처·원문·paragraph ID·hash를 보존했다.",
              "", "## Primary 조건의 출력 유형", "", "| 배치 | 기권 | Donor 일치 | 다른 문맥 내 값 | 문맥 외 값 | Invalid |", "|---|---:|---:|---:|---:|---:|"]
    for p in ("MIDDLE", "ADJACENT"):
        rs = [next(r for r in rates if r["family"] == "MISSING" and r["placement"] == p and r["category"] == cat) for cat in CATEGORIES]
        lines.append("| " + p + " | " + " | ".join(f"{r['count']}/{r['n']} ({100*r['estimate']:.2f}%)" for r in rs) + " |")
    lines += ["", "Donor 일치는 질문 속성과 연결된 후보뿐 아니라 두 donor 후보 중 어느 쪽이든 정확히 일치한 출력이다. Context-absent는 donor가 아니라는 이유만으로 판정하지 않고 정규화한 전체 입력에서 부재를 확인했다.",
              f"완전한 S1=v1→S2=v2 전환율은 {effect('forward_switch')}, 반대 전환율은 {effect('reverse_switch')}다. 이는 delta_attr와 다른 지표다.",
              "", "## 대조군", "", "| 조건 | 정답 / 전체 | 정확도 및 95% CI |", "|---|---:|---:|"]
    for family, r in s["controls"].items():
        lines.append(f"| {family} | {r['correct']}/{r['n']} | {100*r['estimate']:.2f}% [{100*r['ci95_low']:.2f}, {100*r['ci95_high']:.2f}] |")
    lines += ["", "T0는 donor 없는 missing target의 기권, T1은 donor 없는 정답 추출, PRESENT는 두 donor가 있는 상황의 target 정답 추출, N0는 질문 속성이 없는 donor 두 개를 둔 기권이다. Target-present의 정답은 context-absent나 false-fill로 세지 않았다.",
              "기본 대조군이 낮으면 donor 오염만으로 전체 실패를 설명할 수 없다. 이런 실패를 이유로 사례를 제외하거나 프롬프트를 변경하지 않았다.",
              "", "## Template·donor order 민감도", "", "| 부분집합 | delta_attr | delta_C | delta_type |", "|---|---:|---:|---:|"]
    for subset in dict.fromkeys(r["subset"] for r in contrasts if r["subset"] != "ALL"):
        vals = [next(r for r in contrasts if r["subset"] == subset and r["metric"] == m) for m in ("delta_attr", "delta_C", "delta_type")]
        lines.append("| " + subset + " | " + " | ".join(f"{100*r['estimate']:+.2f} [{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}]" for r in vals) + " |")
    lines += ["", "단위는 %p다. 부분집합/leave-one-template-out은 사전 지정 보조 분석이며 CI는 개별 구간이다. 세 template를 모든 가능한 문서 양식의 무작위 표본으로 간주하지 않는다.",
              "", "## 품질·통계·재현성", "",
              f"- Strict-valid {s['strict_valid']}/{s['n_requests']}, truncation {s['truncated']}, 시스템 오류 0, raw 재파싱 불일치 0.",
              f"- Invalid 유형: `{json.dumps(s['invalid_types'], ensure_ascii=False)}`.",
              "- Bootstrap 2,000회, seed 2026103103. 모든 대비에 같은 base index matrix를 사용했다. base별 모든 template·조건을 함께 재표집했다.",
              f"- Primary context-absent 관측은 {s['context_absent_count_primary']}건이다. " + ("20건 미만이므로 희소 사건으로 해석에 주의해야 한다." if s['context_absent_sparse_under_20'] else "유형별 실제 건수와 신뢰구간을 함께 보고했다."),
              "- 0건의 bootstrap [0,0]은 모집단 발생률 0의 증명이 아니다. 96개 독립 base에서 한 번도 발생하지 않은 사건의 base당 any-event 확률에 대한 단측 95% 상한은 약 3.07%다. 요청 단위 상한과는 다르다.",
              f"- 본실험 순수 generation {s['runtime']['generation_seconds']:.1f}초, peak allocated/reserved {s['runtime']['peak_allocated_bytes']/2**30:.2f}/{s['runtime']['peak_reserved_bytes']/2**30:.2f} GiB.",
              "- 첫 v1 개발 792회에서 모든 JSON은 유효했지만, 정답에 USD 단어를 붙여 복사해 T1 엄격 정확도가 0/36이었다. 숫자 선택과 단위 표기를 분리하기 위해 문서 표기만 USD 123456→$123456으로 한 번 수정했다. 기존 기호 정규화는 유지했다. 원래 데이터·코드·lock·출력은 audit/protocol_v1_before_currency_format_fix.zip과 이전 run 폴더에 보존했다.",
              "- V2는 본실험 출력 확인 전에 별도로 SHA-256 잠금했고 개발 재실행 및 main 이후에도 일치했다. 가설·후보 값·개체·생성 설정·평가식은 바꾸지 않았다. 개발 표기를 한 번 수정한 사실을 사전 무수정 실행처럼 표현하지 않는다.",
              "", "## 대표 원출력", "", "각 primary 오류 유형에서 request ID가 가장 앞선 사례를 기계적으로 선택했다. 전체 입력은 results/representative_cases.json에 있다.", ""]
    for r in examples:
        lines += [f"- `{r['request_id']}` / {r['category']}: `{r['raw_output'].replace(chr(10), ' ')}`"]
    lines += ["", "## 해석 범위와 HW2 작성", "",
              "이 연구는 unsupported target의 출력 선택이 속성 연결 및 길이가 통제된 배치에 민감한지를 검정한다. 내부 binding 회로, attention head, 또는 모든 모델·속성·실문서에 대한 일반화는 검정하지 않았다.",
              "위치 조작은 donor 절대 위치와 질문까지의 거리를 함께 바꾼다. 전체 길이를 고정해도 두 요인을 독립적으로 식별하는 설계는 아니다. 합성 영어 salary 한 속성, 단일 모델, 단일 길이, 단일 공개 소설 배경과 세 template에 한정된다.",
              "이전 position_balanced_donor의 target-last·길이 변화·제약 생성과 직접 같은 실험으로 취급하지 않는다. 기존 결과와 차이가 나면 이 여러 설계 차이 중 어느 하나의 인과효과로 단정할 수 없다.",
              "HW2 본문은 두 primary 가설의 결과, 대조군, 부분집합 분석과 오류 사례를 중심으로 작성한다. 다음 단계의 출력 제약 강건성 및 속성/모델 일반화는 이번 실행 범위에 포함되지 않았다.",
              "", "## 산출물", "",
              "- results/contrasts.csv, outcome_rates.csv, condition_counts.csv, base_effects.csv",
              "- figures/outcome_rates.png 및 .pdf, hypothesis_contrasts.png 및 .pdf",
              "- data/*_requests.jsonl, runs/run_20261031_hw1_main_v2/*_raw.jsonl",
              "- source/source_manifest.json, data/data_audit.json, results/bootstrap_manifest.json",
              "- reproduce.md, ai_usage_log.md, completion_manifest.json", ""]
    (ROOT / "final_report_ko.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
