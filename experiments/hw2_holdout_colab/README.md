# Route–Propose–Verify probe (holdout)

Code, Colab notebook, and outputs of the probe in Sections 3.5 and 6.4 of the paper (Table 3).
The holdout has 16 new base examples (8 per task) in 6 conditions × 3 templates × 2 placements × 2
wordings = 1,152 inputs. Prompts, rules, gold labels, and the bootstrap seed (2026111012) were locked
by hash before the first holdout generation; no output was regenerated.

- `protocol.py`, `data.py`, `core.py`, `engine.py`, `runner.py`, `history.py`: holdout generation, methods, and inference.
- `analysis.py`, `statistics.py`: metrics and paired bootstrap.
- `parent_source/`, `source/`: frozen code and development outputs of the earlier probe stages that the holdout reuses.
- `imported_runs/holdout_v1_828a82fc0e26/results/`: the Colab T4 holdout run (`decisions.jsonl`, `gold.json`, `protocol_lock.json`, checkpoints) and its analysis (`results/metrics.csv`).
- `hw2_holdout_colab.ipynb`: self-contained notebook that ran the holdout on a Colab T4.
- `README_ko.md`: the original protocol note (Korean).

## Recompute Table 3 (CPU)

From the repository root:

```bash
python -c "from experiments.hw2_holdout_colab.analysis import analyze; analyze('experiments/hw2_holdout_colab/imported_runs/holdout_v1_828a82fc0e26/results')"
python paper/analysis/paper_tables.py probe
```

## Re-run on a Colab T4

Upload `hw2_holdout_colab.ipynb` to Google Colab, select a T4 GPU, and choose Run all. The notebook
embeds the code and the development outputs it reuses. With `REUSE_PARENT = True` (default) it reuses
development outputs only if Python, PyTorch, CUDA, GPU, Transformers, tokenizers, and the chat template
match the original run; otherwise set `REUSE_PARENT = False` to regenerate them as well. Holdout
requests are always generated anew. At the end it downloads `holdout_v1_results.zip`.

Unit tests (CPU; one test also needs PyTorch): `python -m unittest discover -s experiments/hw2_holdout_colab/tests -t .`
