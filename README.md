# When Missing Fields Borrow Values

Code, data, raw model outputs, and analysis scripts for the paper
*When Missing Fields Borrow Values: Diagnosing Cross-Document Contamination in Information Extraction*
(CSE4057 term paper, HW2).

All model outputs were generated with `Qwen/Qwen2.5-1.5B-Instruct` on a Google Colab Tesla T4.
Every number in the paper can be recomputed on a CPU from the released raw outputs, and the model runs
can be repeated on a Colab T4 with the included notebooks.

## Contents

| Path | Contents |
|---|---|
| `experiments/hw2_missing_fields/` | Salary task: protocol-locked code, configuration, and data; raw outputs (`runs/`); results |
| `experiments/hw2_year_generalization/` | Founding Year task, same structure |
| `experiments/hw2_holdout_colab/` | Route–Propose–Verify probe: code, Colab notebook, holdout outputs |
| `experiments/hw2_t4_rerun_colab/` | Colab notebook that runs both main experiments on a T4 |
| `paper/` | Scripts that draw Figures 1–3 and print every table and Analysis number of the paper |
| `restore_data.py` | Restores the compressed request files and checks all protocol-locked files |

## Environment

- **Model runs:** Google Colab, Tesla T4 (16 GB); Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0),
  Transformers 4.57.6, tokenizers 0.22.2. Model `Qwen/Qwen2.5-1.5B-Instruct` at revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP16, greedy decoding, no training.
  The notebooks install the pinned libraries.
- **Reanalysis (CPU only, no model or GPU):** Python 3.10 or later (checked with 3.10 and 3.13) with
  `requirements-cpu.txt` (NumPy 2.2.6, Matplotlib 3.10.9).

## Reproduce the tables and figures (CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
python restore_data.py
```

Then, from the repository root:

| Paper item | Command |
|---|---|
| Table 2 (main results) | `python -m experiments.hw2_missing_fields.scripts.analyze`<br>`python -m experiments.hw2_year_generalization.scripts.analyze`<br>`python paper/analysis/paper_tables.py main` |
| Table 3 (probe) | `python -c "from experiments.hw2_holdout_colab.analysis import analyze; analyze('experiments/hw2_holdout_colab/imported_runs/holdout_v1_828a82fc0e26/results')"`<br>`python paper/analysis/paper_tables.py probe` |
| Table 5 (Appendix B, outcomes) | `python paper/analysis/paper_tables.py outcomes` |
| Numbers in Section 6 | `python -m experiments.hw2_missing_fields.posthoc_null_diagnostic`<br>`python paper/analysis/paper_tables.py analysis` |
| Table 6 (Appendix C, sampled outputs) | `python paper/analysis/sample_examples.py` |
| Figure 2 | `python paper/figures/make_fig2.py` |
| Figure 3 | `python paper/figures/make_fig3.py` |
| Figure 1 | `pip install python-pptx` then `python paper/figures/make_fig1.py fig1.pptx fig1.png` |

The two `analyze` commands verify every protocol-locked file, re-parse every raw output, and rewrite
`results/` with the same bootstrap draws. `paper_tables.py` prints each table and checks every value
against the number printed in the paper; it stops with an error if any value differs.

We ran all of these steps in a fresh clone with a new virtual environment (Python 3.13 and the pinned
packages). Each step took under five seconds. All regenerated result files (CSV, JSON, JSONL) were
byte-identical to the released ones, and `paper_tables.py` confirmed all 52 values. Figure files are
redrawn with the same content, but their bytes can differ (embedded metadata and font subsets).

## Re-run the model (Colab T4)

- **Main experiments (Tables 2 and 5, Figures 2 and 3):** open
  `experiments/hw2_t4_rerun_colab/hw2_t4_rerun_colab.ipynb` in Google Colab, select a T4 GPU, and
  choose Run all. It restores the locked files, runs `infer` (development and main splits) and
  `analyze` for each task without modification, and downloads a ZIP of the outputs. The main split
  took about 45 minutes per task. Outside the notebook, pass `--device cuda:0` to `infer`
  (see `experiments/*/reproduce.md`).
- **Probe (Table 3):** run `experiments/hw2_holdout_colab/hw2_holdout_colab.ipynb` on a Colab T4
  (see `experiments/hw2_holdout_colab/README.md`).

## Seeds

Decoding is greedy, so generation uses no sampling seed.

| Purpose | Seed |
|---|---|
| Salary data (development, main) | 2026103101, 2026103102 |
| Founding Year data (development, main) | 2026110201, 2026110202 |
| Probe holdout data | 2026111011 |
| Bootstrap (Salary, Founding Year, probe) | 2026103103, 2026110203, 2026111012 |
| Sampled outputs in Appendix C | 20261116 |

## Protocol locks

Before any model output, the inputs, prompts, scripts, and configuration of each experiment were
fixed by SHA-256 in `runs/<run_id>/protocol_lock.json`. The scripts check these hashes and stop if a
locked file has changed, so locked files must not be edited.

## Data

All entities and values are synthetic. Background paragraphs come from *Alice's Adventures in
Wonderland* (Project Gutenberg eBook #11, public domain).
