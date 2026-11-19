# Colab T4 notebook for the main experiments

`hw2_t4_rerun_colab.ipynb` runs the Salary and Founding Year experiments on a Google Colab Tesla T4:
it restores the protocol-locked files of both tasks byte for byte, checks them against the locks, and
runs `infer --split dev`, `infer --split main`, and `analyze` without modification. This is how the
released raw outputs in `experiments/*/runs/` were produced. Each task downloads a ZIP with its
`runs/`, `results/`, and `figures/` folders, which can be compared with the released files.

`build_notebook.py` rebuilds the notebook from the files in `experiments/` (run `python restore_data.py`
first): `python experiments/hw2_t4_rerun_colab/build_notebook.py`.
