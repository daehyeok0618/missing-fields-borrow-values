"""Separate development/holdout evaluation and paired wording-shift analysis."""
from collections import Counter, defaultdict
from pathlib import Path
import random

from . import core
from .common import read_json, read_jsonl, write_json
from .protocol import GROUPS, METHODS, PROTOCOL, SPLITS
from .statistics import correct, summarize, weights_for, write_csv


def paired_style(records, gold, repetitions):
    import numpy as np
    metrics = []
    for task in ("salary", "year"):
        for group in GROUPS + ("all_conditions",):
            chosen = [r for r in records if r["task"] == task and r["split"].startswith("holdout_")
                      and (group == "all_conditions" or r["group"] == group)]
            bases = sorted({r["base_id"] for r in chosen})
            weights = weights_for([(task, b) for b in bases], repetitions, PROTOCOL["bootstrap_seed"])
            for method in METHODS:
                by_style = {}
                for split in ("holdout_canonical", "holdout_paraphrase"):
                    counts = np.array([[sum(correct(r["methods"][method], gold[r["id"]], "semantic")
                                            for r in chosen if r["split"] == split and r["base_id"] == base),
                                        sum(r["split"] == split and r["base_id"] == base for r in chosen)] for base in bases])
                    resampled = weights @ counts
                    by_style[split] = (float(counts[:, 0].sum() / counts[:, 1].sum()), resampled[:, 0] / resampled[:, 1])
                a, b = by_style["holdout_paraphrase"], by_style["holdout_canonical"]
                lo, hi = np.percentile(a[1] - b[1], [2.5, 97.5])
                metrics.append({"task": task, "group": group, "method": method, "n_base": len(bases),
                                "difference_paraphrase_minus_canonical": a[0] - b[0], "ci_low": float(lo), "ci_high": float(hi)})
    return metrics


def analyze(run_dir, bootstrap_repetitions=None):
    root = Path(run_dir)
    if read_json(root / "replay_audit.json")["status"] != "PASS":
        raise RuntimeError("Raw decision replay must pass before reporting")
    records, gold = read_jsonl(root / "decisions.jsonl"), read_json(root / "gold.json")
    samples = {r["id"]: r for r in read_jsonl(root / "public_samples.jsonl")}
    expected = PROTOCOL["expected_development_inputs"] + PROTOCOL["expected_holdout_inputs"]
    if len(records) != expected or len({r["id"] for r in records}) != len(records):
        raise RuntimeError("Incomplete evaluation")
    if {r["id"] for r in records} != set(samples) or set(samples) != set(gold):
        raise RuntimeError("Evaluation identity mismatch")
    out = root / "results"
    out.mkdir(exist_ok=True)
    reps = bootstrap_repetitions or PROTOCOL["bootstrap_repetitions"]
    metrics, contrasts, bounds, tables = [], [], [], {}
    for split in SPLITS + ("holdout_all",):
        for task in ("salary", "year"):
            for group in GROUPS + ("all_conditions",):
                selected = [r for r in records if r["task"] == task and (r["split"] == split or (split == "holdout_all" and r["split"].startswith("holdout_")))
                            and (group == "all_conditions" or r["group"] == group)]
                for mode in ("decision", "service"):
                    for policy in ("strict", "semantic"):
                        m, c, b, a = summarize(selected, gold, task, group, reps, mode, policy)
                        metrics.extend({"split": split, **r} for r in m)
                        contrasts.extend({"split": split, **r} for r in c)
                        bounds.extend({"split": split, **r} for r in b)
                        if mode == "decision" and policy == "semantic":
                            tables[split + "/" + task + "/" + group] = a
    styles = paired_style(records, gold, reps)
    costs = read_json(root / "method_costs.json")
    cost_rows, agreement = [], {}
    for split in SPLITS + ("holdout_all",):
        chosen = [r for r in records if r["split"] == split or (split == "holdout_all" and r["split"].startswith("holdout_"))]
        agreement[split] = {"n": len(chosen), "id_vs_target_only_routed_gate_same_normalized_prediction": sum(
            core.old.answer_equal(r["methods"]["id_gate_constrained"], r["methods"]["target_only_routed_gate_constrained"]) for r in chosen)}
        for method in METHODS:
            for cache_mode in ("unique", "logical"):
                row = costs[split][method]
                cost_rows.append({"split": split, "method": method, "cost_mode": cache_mode,
                                  "n_inputs": row["n_inputs"], **row[cache_mode],
                                  "generation_seconds_per_input": row[cache_mode]["generation_seconds"] / max(row["n_inputs"], 1)})
    # Save complete disagreements; uniform examples use a fixed RNG, never a hand-picked win.
    rng = random.Random(PROTOCOL["data_seed"] + 17)
    examples = {}
    for split in SPLITS:
        for task in ("salary", "year"):
            buckets = defaultdict(list)
            for r in records:
                if r["split"] != split or r["task"] != task:
                    continue
                a = correct(r["methods"]["id_gate_constrained"], gold[r["id"]], "semantic")
                b = correct(r["methods"]["target_only_routed_gate_constrained"], gold[r["id"]], "semantic")
                bucket = "both_correct" if a and b else "id_only_correct" if a else "target_only_routed_correct" if b else "both_wrong"
                buckets[bucket].append(r)
            for bucket, entries in buckets.items():
                key = split + "/" + task + "/" + bucket
                examples[key] = {"n": len(entries), "all_ids": [r["id"] for r in entries], "examples": [
                    {"id": r["id"], "base_id": r["base_id"], "group": r["group"], "gold": gold[r["id"]]["value"],
                     "target": core.old.target_text(core.old.public_view(samples[r["id"]])), "question": samples[r["id"]]["question"],
                     "methods": r["methods"], "gates": r["gates"], "selection": r["selection"]}
                    for r in rng.sample(entries, min(3, len(entries)))]}
    summary = {"status": "COMPLETE", "development_inputs": 480, "holdout_inputs": 1152,
               "holdout_independent_bases": 16, "holdout_bases_per_task": 8, "bootstrap_repetitions": reps,
               "scope": PROTOCOL["holdout_scope"], "decision_semantic_counts": tables, "agreement": agreement,
               "bootstrap_unit": "task-stratified independent family; all conditions, templates, placements, and styles retained together",
               "reuse_parent": read_json(root / "setup.json")["reuse_parent"]}
    write_json(out / "summary.json", summary)
    write_json(out / "paired_examples.json", examples)
    for name, rows in (("metrics.csv", metrics), ("paired_contrasts.csv", contrasts), ("zero_event_base_bounds.csv", bounds),
                       ("paired_style_shift.csv", styles), ("costs.csv", cost_rows)):
        write_csv(out / name, rows)
    lines = ["# HW2 Target-only 비교와 고정 holdout 평가", "",
             "개발 480개 입력과 새 holdout 1,152개 입력을 별도로 평가했다. Holdout은 독립 family 16개, task당 8개다.",
             "각 family의 두 표현·세 문서 형식·두 placement·여섯 조건은 대응 관측이며 독립 표본 수를 늘리지 않는다.",
             "모든 방법과 prompt·parser 사전·평가 정책은 모델 실행 전에 protocol_lock.json으로 고정했다. 추가 학습 없음.", "",
             "주 비교는 ID + constrained gate와 Target-only-routed 숫자 출력 + 동일 gate다. 둘 다 target에 숫자가 없으면 CPU null로 종료하고, 후보를 선택할 때의 prompt/출력 인터페이스가 다르다. 검증기는 같다. CPU 생략을 하지 않은 일반 Target-only+gate도 별도 보고한다.",
             "아래는 invalid를 오류로 유지한 decision/semantic 평가다. Strict와 fail-closed service 결과는 metrics.csv에 별도로 있다.", ""]
    for split in SPLITS:
        lines += ["## " + split, "", "| 방법 | Salary 정답 | Year 정답 |", "|---|---:|---:|"]
        for method in METHODS:
            a, b = (tables[split + "/" + task + "/all_conditions"][method] for task in ("salary", "year"))
            lines.append("| " + method + " | " + str(a["correct"]) + "/" + str(a["n"]) + " | " + str(b["correct"]) + "/" + str(b["n"]) + " |")
        lines += ["", "| 방법 | 캐시 반영 고유 모델 요청 | 입력당 논리적 생성 시간 평균(초) |", "|---|---:|---:|"]
        for method in ("target_only", "id_selection", "target_only_gate_constrained", "target_only_routed_gate_constrained", "id_gate_constrained", "standard_gate_constrained", "alias_field_parser"):
            c = costs[split][method]
            lines.append(f"| {method} | {c['unique']['calls']} | {c['logical']['generation_seconds']/c['n_inputs']:.4f} |")
        lines.append("")
    lines += ["## 해석 기준", "",
              "paired_contrasts.csv의 ID gate − target-only-routed gate를 주 비교로 읽고, 일반 target-only gate와의 차이도 함께 확인한다. 정확도·PRESENT 손실·MISSING false-fill을 task와 표현별로 본다. 차이가 0인 관측을 통계적 동등성이나 비열등성의 증명으로 읽지 않는다.",
              "paired_style_shift.csv는 동일 family에서 paraphrase − canonical의 정확도 차이를 직접 bootstrap했다. 독립 표본 CI를 빼서 구한 구간이 아니다.",
              "Literal parser와 더 강한 alias_field_parser를 함께 보고한다. 후자는 생성에 사용한 모든 긍정 속성 표현을 아는 schema-aware baseline이다. 이를 빼고 표현 변형에서 literal parser를 이겼다는 주장만 하지 않는다.",
              "모델 생성 비용은 캐시 반영 unique와 입력마다 같은 요청을 다시 센 logical을 모두 저장했다. 과거/새 생성 시간을 구분하며 CPU·모델 로드·종단 latency는 포함하지 않는다. 서로 겹치는 방법·split 비용을 합산하지 않는다.",
              "0건 CI를 위험 0으로 해석하지 않는다. Zero-event CSV는 family에 사건이 하나라도 있을 위험의 상한이다.",
              "이 평가는 같은 합성 생성기의 새 개체·값 조합과 미리 선언한 문구 변화에 대한 검증이다. 실제 문서·다른 모델·새 알고리즘의 신규성을 증명하지 않는다.",
              "결과를 보고 prompt를 수정하면 이 holdout은 새 개발 자료가 된다. 보고서는 자동으로 최적 방법을 선택하거나 기존 규칙을 바꾸지 않는다.", ""]
    (out / "report_ko.md").write_text("\n".join(lines), encoding="utf-8")
    figures(root, tables)
    print("Separate split tables, paired contrasts, examples and figures saved:", out, flush=True)
    return summary


def figures(root, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    methods = ("target_only", "id_selection", "target_only_routed_gate_constrained", "id_gate_constrained", "standard_gate_constrained", "field_parser", "alias_field_parser")
    names = ("Target-only JSON", "Occurrence ID / NONE", "Target-only routed + gate", "ID + gate", "Standard + gate", "Literal parser", "Schema-aware parser")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(methods))
    for ax, task in zip(axes, ("salary", "year")):
        for shift, split in ((-.19, "holdout_canonical"), (.19, "holdout_paraphrase")):
            values = [100 * tables[split + "/" + task + "/all_conditions"][m]["correct"] / tables[split + "/" + task + "/all_conditions"][m]["n"] for m in methods]
            ax.bar(x + shift, values, width=.38, label=split.removeprefix("holdout_"))
        ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(task.capitalize() + " — frozen synthetic holdout")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Decision accuracy (%)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / ("holdout_accuracy." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True)
    selected = ("target_only_routed_gate_constrained", "id_gate_constrained", "alias_field_parser")
    for ax, task in zip(axes, ("salary", "year")):
        vals = np.array([[100 * tables["holdout_paraphrase/" + task + "/" + g][m]["correct"] / tables["holdout_paraphrase/" + task + "/" + g][m]["n"] for g in GROUPS] for m in selected])
        ax.imshow(vals, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for i, method in enumerate(selected):
            for j, group in enumerate(GROUPS):
                a = tables["holdout_paraphrase/" + task + "/" + group][method]
                ax.text(j, i, str(a["correct"]) + "/" + str(a["n"]), ha="center", va="center", color="white" if vals[i, j] > 55 else "black", fontsize=8)
        ax.set_xticks(range(6), [g.replace("_", "\n") for g in GROUPS], fontsize=7)
        ax.set_yticks(range(3), ("Target-only routed + gate", "ID + gate", "Schema-aware parser"), fontsize=9)
        ax.set_title(task.capitalize() + " — paraphrase holdout")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / ("holdout_conditions." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
