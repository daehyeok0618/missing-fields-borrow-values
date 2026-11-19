# Salary task

Protocol-locked code, configuration, data, raw model outputs, and results of the Salary task.

- `experiment_spec.md`, `config.json`: design, model, decoding, and seeds, fixed before any model output.
- `data/`: base examples, background text, and the 6,336 main and 792 development requests (`*.jsonl.gz`; run `python restore_data.py` from the repository root to restore them).
- `scripts/`: `prepare` (data), `check_and_lock` (protocol lock), `infer` (model), `analyze` (contrasts, bootstrap, figures).
- `runs/run_20261031_hw1_main_v2/`: protocol lock and the Colab T4 raw outputs (`dev_raw.jsonl`, `main_raw.jsonl`) with runtime records.
- `results/`, `figures/`: outputs of `analyze`; `posthoc_null_diagnostic.py` adds the post hoc notation check (`results/abstention_notation_*`).

See `reproduce.md` and the repository README.
