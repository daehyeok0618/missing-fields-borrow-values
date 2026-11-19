"""CPU reporting: strict/fail-closed, paired base inference and separate diagnostics."""
from collections import Counter
from pathlib import Path

from .common import read_json, read_jsonl, write_json
from .protocol import GROUPS, METHODS, PROTOCOL
from .statistics import correct, summarize, write_csv, weights_for
from . import core


def diagnostic_analysis(records, gold, repetitions):
    import numpy as np
    table, contrasts = [], []
    for task in ("salary", "year"):
        for family in ("otherfield_missing", "multi_missing", "all_families"):
            for condition in ("missing", "present"):
                selected = [r for r in records if r["task"] == task and r["condition"] == condition
                            and (family == "all_families" or r["family"] == family)]
                bases = sorted({r["base_id"] for r in selected})
                weights = weights_for([(task, b) for b in bases], repetitions, PROTOCOL["bootstrap_seed"])
                data = {}
                for variant in ("numeric", "clause", "numeric_dash"):
                    chosen = [r for r in selected if r["variant"] == variant]
                    by_base = {r["base_id"]: r for r in chosen}
                    if len(by_base) != len(bases) or len(chosen) != len(bases):
                        raise RuntimeError("Diagnostic pairing incomplete")
                    flags = np.array([int(correct(by_base[b]["decision"], gold[by_base[b]["id"]], "semantic")) for b in bases])
                    draws = weights @ flags / len(bases)
                    data[variant] = flags, draws
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    table.append({"task": task, "family": family, "condition": condition, "variant": variant,
                                  "n_base": len(bases), "correct": int(flags.sum()), "n": len(flags),
                                  "accuracy": float(flags.mean()), "ci_low": float(lo), "ci_high": float(hi),
                                  "invalid": sum(not r["decision"]["valid"] for r in chosen),
                                  "model_none": sum(r["selection_status"] == "model_none" for r in chosen)})
                for variant in ("clause", "numeric_dash"):
                    flags, draws = data[variant]
                    base_flags, base_draws = data["numeric"]
                    lo, hi = np.percentile(draws - base_draws, [2.5, 97.5])
                    contrasts.append({"task": task, "family": family, "condition": condition,
                                      "variant": variant, "reference": "numeric", "n_base": len(bases),
                                      "difference_accuracy": float((flags - base_flags).mean()),
                                      "ci_low": float(lo), "ci_high": float(hi),
                                      "improved": int(((flags == 1) & (base_flags == 0)).sum()),
                                      "regressed": int(((flags == 0) & (base_flags == 1)).sum())})
    return table, contrasts


def analyze(run_dir, bootstrap_repetitions=None):
    root = Path(run_dir)
    if read_json(root / "replay_audit.json")["status"] != "PASS":
        raise RuntimeError("Raw-output audit must pass before reporting")
    records, gold = read_jsonl(root / "decisions.jsonl"), read_json(root / "gold.json")
    if len(records) != 480 or len({r["id"] for r in records}) != 480 or set(gold) != {r["id"] for r in records}:
        raise RuntimeError("Incomplete primary evaluation")
    out = root / "results"
    out.mkdir(exist_ok=True)
    reps = bootstrap_repetitions or PROTOCOL["bootstrap_repetitions"]
    metrics, contrasts, bounds, tables = [], [], [], {}
    for task in ("salary", "year"):
        for group in GROUPS + ("all_conditions",):
            selected = [r for r in records if r["task"] == task and (group == "all_conditions" or r["group"] == group)]
            for mode in ("decision", "service"):
                for policy in ("strict", "semantic"):
                    m, c, b, a = summarize(selected, gold, task, group, reps, mode, policy)
                    metrics.extend(m); contrasts.extend(c); bounds.extend(b)
                    if mode == "decision" and policy == "semantic":
                        tables[task + "/" + group] = a
    diag, diag_contrasts = diagnostic_analysis(read_jsonl(root / "diagnostic_decisions.jsonl"),
                                               read_json(root / "diagnostic_gold.json"), reps)
    costs = read_json(root / "method_costs.json")
    gate_status = {}
    regressions, recoveries = [], []
    for task in ("salary", "year"):
        for method in ("standard_gate_free", "id_gate_free", "standard_gate_constrained", "id_gate_constrained"):
            gate_status[task + "/" + method] = dict(Counter(r["gates"][method]["status"] for r in records if r["task"] == task))
    for r in records:
        before = correct(r["methods"]["id_selection"], gold[r["id"]], "semantic")
        after = correct(r["methods"]["id_gate_constrained"], gold[r["id"]], "semantic")
        if before != after:
            (regressions if before else recoveries).append({"id": r["id"], "task": r["task"], "group": r["group"],
                "gold": gold[r["id"]]["value"], "before": r["methods"]["id_selection"],
                "after": r["methods"]["id_gate_constrained"], "gate": r["gates"]["id_gate_constrained"]})
    summary = {"status": "COMPLETE", "study_status": PROTOCOL["status"], "reuse_parent": read_json(root / "setup.json")["reuse_parent"],
               "primary_inputs": 480, "primary_bases": 48, "diagnostic_requests": 96, "diagnostic_bases": 16,
               "bootstrap_unit": "independent base within task; renderings, conditions and placements retained together",
               "bootstrap_repetitions": reps, "decision_semantic_counts": tables, "gate_status": gate_status,
               "id_gate_recovered": len(recoveries), "id_gate_regressed": len(regressions)}
    write_json(out / "summary.json", summary)
    write_json(out / "id_gate_changed_cases.json", {"recovered": recoveries, "regressed": regressions})
    for name, rows in (("metrics.csv", metrics), ("paired_contrasts.csv", contrasts), ("zero_event_base_bounds.csv", bounds),
                       ("diagnostic_metrics.csv", diag), ("diagnostic_paired_contrasts.csv", diag_contrasts)):
        write_csv(out / name, rows)
    lines = ["# HW2 관계 검증 보완 개발 실험", "", "기존 480개 입력 / 48개 base. 별도 표현 진단 32개 대응 입력 / 16개 base × 3형식 = 96요청.",
             "학습 없음. 과거 오류를 보고 설계한 개발 실험이므로 독립 holdout 성과가 아니다.", "",
             "동일한 관계 검증 prompt에서 후보 값은 두 파이프라인 모두 숫자 문자열로 통일했다. Free는 자유 생성, constrained는 매 토큰에서 두 라벨만 허용한다.",
             "아래는 invalid를 오류로 유지한 decision/semantic 정답 수다. Service의 invalid→null과 strict null 평가는 별도 CSV에 있다.", ""]
    for task in ("salary", "year"):
        lines += ["## " + task, "", "| 방법 | " + " | ".join(GROUPS) + " |", "|---|" + "---:|" * len(GROUPS)]
        for method in METHODS:
            lines.append("| " + method + " | " + " | ".join(str(tables[task + "/" + g][method]["correct"]) + "/"
                           + str(tables[task + "/" + g][method]["n"]) for g in GROUPS) + " |")
        lines.append("")
    lines += ["## 판정 형식과 오답", "", f"ID + constrained 검증의 회복 {len(recoveries)}건, 퇴행 {len(regressions)}건. 개별 사례는 id_gate_changed_cases.json에 있다.",
              "기존 SUPPORTING 출력은 SUPPORTED로 재해석하지 않는다. Free의 invalid와 constrained의 유효 라벨 오판을 구분해야 한다.",
              "진단의 numeric/clause/numeric_dash는 동일 지시문을 사용하며 ID 위치 또는 대시만 바꾼다. 기존 ID prompt와 지시문이 달라 기존 대비 순수 ID 위치 효과로 읽지 않는다.", "",
              "## 비용", "", "| 방법 | 고유 요청 | 기존 재사용 | 현재 요청 | 기록된 생성 시간(초) |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        c = costs[method]
        lines.append(f"| {method} | {c['calls']} | {c['parent_reused_calls']} | {c['current_calls']} | {c['generation_seconds']:.2f} |")
    current = costs["_current_physical_total"]
    lines += ["", f"이번 실행에서 생성한 고유 요청은 {current['calls']}개, 저장된 생성 시간 합계 {current['generation_seconds']:.2f}초다. 모델 로드·CPU 분석 시간은 제외된다.",
              "방법별 시간에는 재사용된 과거 생성 시간이 포함된다. 서로 공유되는 요청이 있으므로 방법별 수치를 합산하지 않는다. 실행 세션 시간·메모리는 runtime_sessions.json에 있다.", "",
              "## 해석 기준", "", "Year otherfield_missing/multi_missing 오류 감소와 PRESENT 정답 보존을 함께 확인한다.",
              "ID + gate와 Standard + 같은 gate를 비교해야 ID 선택의 추가 효과를 평가할 수 있다. Parser도 포함해 명시적 라벨 합성 자료의 한계를 확인한다.",
              "paired_contrasts.csv는 method−reference이며 정확도는 높을수록, false-fill/invalid는 낮을수록 좋다.",
              "Base 단위 paired bootstrap을 사용한다. 소규모 조건은 base가 4개뿐이며, 0건의 [0,0] CI를 위험 0이나 비열등성으로 해석하지 않는다.",
              "zero_event_base_bounds.csv의 상한은 base에서 사건이 하나라도 생길 위험이며 입력 평균 손실률 상한과 다르다.",
              "이 개발 결과로 규칙을 고른 뒤 독립 숫자·개체와 paraphrase holdout에 고정 평가해야 일반화 주장을 할 수 있다.", ""]
    (out / "report_ko.md").write_text("\n".join(lines), encoding="utf-8")
    figures(root, tables, diag)
    print("Reports and figures saved:", out, flush=True)
    return summary


def figures(root, tables, diagnostic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out = root / "figures"
    out.mkdir(exist_ok=True)
    selected_methods = ("standard", "id_selection", "standard_gate_free", "id_gate_free", "standard_gate_constrained", "id_gate_constrained", "any_change", "field_parser")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, task in zip(axes, ("salary", "year")):
        values = np.array([[tables[task + "/" + g][m]["correct"] / tables[task + "/" + g][m]["n"] * 100
                            for g in GROUPS] for m in selected_methods])
        im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        for i in range(len(selected_methods)):
            for j in range(len(GROUPS)):
                a = tables[task + "/" + GROUPS[j]][selected_methods[i]]
                ax.text(j, i, str(a["correct"]) + "/" + str(a["n"]), ha="center", va="center", color="white" if values[i, j] > 65 else "black", fontsize=9)
        ax.set_xticks(range(len(GROUPS)), [g.replace("_", "\n") for g in GROUPS], fontsize=8)
        ax.set_yticks(range(len(selected_methods)), selected_methods, fontsize=8)
        ax.set_title(task.capitalize() + " — development")
    fig.colorbar(im, ax=axes, label="Decision accuracy (%)", shrink=.8)
    fig.subplots_adjust(left=.19, right=.86, bottom=.18, wspace=.15)
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("condition_accuracy." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, ("salary", "year")):
        for shift, condition in ((-.18, "missing"), (.18, "present")):
            rows = [r for r in diagnostic if r["task"] == task and r["family"] == "all_families" and r["condition"] == condition]
            ax.bar(np.arange(3) + shift, [100 * r["accuracy"] for r in rows], width=.36, label=condition)
        ax.set_xticks(range(3), ["Numeric ID", "Clause ID", "Numeric ID + dash"])
        ax.set_ylim(0, 105)
        ax.set_title(task.capitalize() + " — paired diagnostic")
        ax.legend()
    axes[0].set_ylabel("Decision accuracy (%)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("representation_diagnostic." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
