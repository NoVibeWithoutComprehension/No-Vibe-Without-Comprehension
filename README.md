# No-Vibe-Without-Comprehension

Repository with the merged Study 0 / Study 0+ dataset and modeling scripts for
predicting code comprehension from physiological, code, and participant
metadata.

The current scripts add the generalization protocols needed for the workflow:

- `LOSO`: Leave-One-Subject-Out, using `vol` as the held-out group.
- `LOTO`: Leave-One-Task-Out, using `task` as the held-out group.
- `LOSATO`: Leave-One-Subject-And-Task-Out, using the `vol` + `task`
  observation as the held-out group.

## Repository Structure

```text
Data/
  combined_study_data.csv
results/
  Modeling_Results_with_Confidence_Intervals.csv
  train-test balance.png
scripts/
  dataset_summary.py
  run_generalization_cv.py
requirements.txt
README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Dataset Check

```bash
python3 scripts/dataset_summary.py
```

Expected dataset shape for the current merged CSV:

- 145 rows
- 43 subjects
- 7 tasks
- 145 unique subject-task observations

## Generalization Experiments

Run all three protocols with the default feature set and classifiers:

```bash
python3 scripts/run_generalization_cv.py --strategy all
```

Run each protocol separately:

```bash
python3 scripts/run_generalization_cv.py --strategy loso
python3 scripts/run_generalization_cv.py --strategy loto
python3 scripts/run_generalization_cv.py --strategy losato
```

By default, `entendimento >= 4.0` is encoded as the positive comprehension
class. Override this threshold when matching a specific paper/rebuttal
definition:

```bash
python3 scripts/run_generalization_cv.py --strategy all --threshold 3.5
```

Feature subsets can be evaluated independently:

```bash
python3 scripts/run_generalization_cv.py --strategy all --feature-set eyetracking
python3 scripts/run_generalization_cv.py --strategy all --feature-set eeg
python3 scripts/run_generalization_cv.py --strategy all --feature-set hrv
python3 scripts/run_generalization_cv.py --strategy all --feature-set code
python3 scripts/run_generalization_cv.py --strategy all --feature-set demographics
```

Outputs are written to `results/generalization_cv/`:

- `*_fold_metrics.csv`: per-fold metrics
- `*_predictions.csv`: held-out predictions
- `*_summary.csv`: model summaries with 95% confidence intervals across folds

The script uses scikit-learn `Pipeline` objects so imputation, scaling, and
one-hot encoding are fitted only on each training fold.
