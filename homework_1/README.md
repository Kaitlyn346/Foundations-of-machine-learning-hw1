# Homework 1

Breast-cancer classification with autodiff (PyTorch vs. logistic regression) and ALE regression on `mcs_ds_edited_iter_shuffled.csv` (linear, SVM, decision tree, neural net).

## Setup

```bash
cd homework_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the repo root (with the venv active):

```bash
jupyter nbconvert --execute --to notebook --output-dir /tmp homework_1/breast_cancer_autodiff_classification.ipynb
```

The notebook reads `mcs_ds_edited_iter_shuffled.csv` via a relative path, and nbconvert runs it from the notebook's own folder.
