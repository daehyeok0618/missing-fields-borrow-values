# Reproduction

Run all commands from the repository root.

## Recompute the results from the released raw outputs (CPU)

```bash
python restore_data.py                                   # once: restores data/*_requests.jsonl, checks the lock
python -m experiments.hw2_year_generalization.scripts.analyze
```

`analyze` verifies every file in the protocol lock, re-parses every raw output in
`runs/*/main_raw.jsonl`, and rewrites `results/` and `figures/` with the same 2,000 paired
bootstrap draws (fixed seed in `config.json`).

## Re-run the model

The model outputs were generated on a Google Colab Tesla T4 with
`experiments/hw2_t4_rerun_colab/hw2_t4_rerun_colab.ipynb`, which restores these files and runs

```bash
python -m experiments.hw2_year_generalization.scripts.infer --split dev --device cuda:0
python -m experiments.hw2_year_generalization.scripts.infer --split main --device cuda:0
python -m experiments.hw2_year_generalization.scripts.analyze
```

`infer` resumes by request ID and never regenerates a request that already has an output, so
re-run it in a copy without the existing `runs/*/dev_raw.jsonl` and `main_raw.jsonl`.
`prepare` and `check_and_lock` refuse to overwrite an existing protocol lock.
