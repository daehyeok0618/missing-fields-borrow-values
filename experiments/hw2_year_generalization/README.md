# Founding Year task

Protocol-locked code, configuration, data, raw model outputs, and results of the Founding Year task.
The structure follows the Salary task (`../hw2_missing_fields`).

- `experiment_spec.md`, `config.json`: design, model, decoding, seeds, and the nine abstention strings, fixed before any model output.
- `data/`: base examples, background text, and the 6,336 main and 792 development requests (`*.jsonl.gz`; run `python restore_data.py` from the repository root to restore them).
- `scripts/`: `prepare`, `check_and_lock`, `infer`, `analyze`, `statistics`.
- `reference/`: locked Salary summary files used by the descriptive cross-task comparison in `analyze` (not reported in the paper).
- `runs/run_20261102_founding_year/`: protocol lock and the Colab T4 raw outputs with runtime records.
- `results/`, `figures/`: outputs of `analyze`.

See `reproduce.md` and the repository README.
