#!/usr/bin/env python3
"""Run LOSO, LOTO, and LOSATO validation on the merged comprehension dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python dependency. Install the repository requirements first:\n"
        "  python3 -m pip install -r requirements.txt"
    ) from exc


IDENTIFIER_COLUMNS = {"vol", "task"}

FEATURE_GROUPS = {
    "eyetracking": [
        "fixation",
        "saccade",
        "pupil",
        "dispersion",
        "scan",
        "spatial",
        "amplitude",
        "reactivity",
        "lf_power",
        "hf_power",
        "lf_hf_ratio",
    ],
    "eeg": [
        "Delta_",
        "Theta_",
        "Alpha_",
        "Beta_",
        "Gamma_",
        "AlphaPeak",
        "AlphaTheta",
        "Brainbeat",
        "Index2",
    ],
    "hrv": ["HRV_"],
    "code": ["LOC", "Vg", "Hdiff", "Heff", "CCSonar"],
    "demographics": [
        "Age",
        "Gender",
        "Education/Job",
        "Approximate number of lines",
        "Number of years",
        "Expertise",
    ],
}


@dataclass(frozen=True)
class Fold:
    label: str
    train_index: np.ndarray
    test_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate comprehension classifiers with Leave-One-Subject-Out "
            "(LOSO), Leave-One-Task-Out (LOTO), or Leave-One-Subject-And-Task-Out "
            "(LOSATO) validation."
        )
    )
    parser.add_argument("--data", default="Data/combined_study_data.csv")
    parser.add_argument("--output-dir", default="results/generalization_cv")
    parser.add_argument(
        "--strategy",
        choices=["loso", "loto", "losato", "all"],
        default="all",
    )
    parser.add_argument("--target", default="entendimento")
    parser.add_argument("--subject-column", default="vol")
    parser.add_argument("--task-column", default="task")
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Comprehension score >= threshold is encoded as the positive class.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", *FEATURE_GROUPS.keys()],
        default="all",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf", "svm", "knn", "gb", "nb", "dummy"],
        choices=["logreg", "rf", "svm", "knn", "gb", "nb", "dummy"],
    )
    parser.add_argument("--random-state", type=int, default=121)
    return parser.parse_args()


def select_feature_columns(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    blocked = {args.target, args.subject_column, args.task_column} | IDENTIFIER_COLUMNS
    candidate_columns = [column for column in df.columns if column not in blocked]
    if args.feature_set == "all":
        return candidate_columns

    tokens = FEATURE_GROUPS[args.feature_set]
    selected = [
        column
        for column in candidate_columns
        if any(token.lower() in column.lower() for token in tokens)
    ]
    if not selected:
        raise SystemExit(f"No columns matched feature set {args.feature_set!r}")
    return selected


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def make_models(random_state: int) -> dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm": SVC(kernel="rbf", class_weight="balanced", probability=True),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gb": GradientBoostingClassifier(random_state=random_state),
        "nb": GaussianNB(),
        "dummy": DummyClassifier(strategy="most_frequent"),
    }


def make_folds(df: pd.DataFrame, strategy: str, subject: str, task: str) -> list[Fold]:
    if strategy == "loso":
        groups = df[subject].astype(str)
    elif strategy == "loto":
        groups = df[task].astype(str)
    elif strategy == "losato":
        groups = df[subject].astype(str) + "__" + df[task].astype(str)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    splitter = LeaveOneGroupOut()
    folds = []
    for train_index, test_index in splitter.split(df, groups=groups):
        label = str(groups.iloc[test_index[0]])
        folds.append(Fold(label, train_index, test_index))
    return folds


def predict_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return np.asarray(scores, dtype=float)
    return model.predict(x_test).astype(float)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    has_both_classes = len(np.unique(y_true)) == 2
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
        if has_both_classes
        else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    row["roc_auc"] = roc_auc_score(y_true, y_score) if has_both_classes else np.nan
    return row


def confidence_interval(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    if len(values) <= 1:
        return (np.nan, np.nan)
    mean = values.mean()
    margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return (mean - margin, mean + margin)


def run_strategy(
    df: pd.DataFrame,
    args: argparse.Namespace,
    strategy: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[feature_columns].copy().replace([np.inf, -np.inf], np.nan)
    y = (df[args.target] >= args.threshold).astype(int).to_numpy()
    preprocessor = make_preprocessor(x)
    models = make_models(args.random_state)
    folds = make_folds(df, strategy, args.subject_column, args.task_column)

    fold_rows = []
    prediction_rows = []
    for model_name in args.models:
        estimator = Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("model", clone(models[model_name])),
            ]
        )
        for fold in folds:
            x_train = x.iloc[fold.train_index]
            x_test = x.iloc[fold.test_index]
            y_train = y[fold.train_index]
            y_test = y[fold.test_index]

            estimator.fit(x_train, y_train)
            y_pred = estimator.predict(x_test)
            y_score = predict_scores(estimator, x_test)

            metrics = metric_row(y_test, y_pred, y_score)
            fold_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "feature_set": args.feature_set,
                    "fold": fold.label,
                    "n_train": len(fold.train_index),
                    "n_test": len(fold.test_index),
                    **metrics,
                }
            )
            for row_index, true_value, pred_value, score in zip(
                fold.test_index, y_test, y_pred, y_score
            ):
                prediction_rows.append(
                    {
                        "strategy": strategy,
                        "model": model_name,
                        "feature_set": args.feature_set,
                        "row_index": int(row_index),
                        args.subject_column: df.iloc[row_index][args.subject_column],
                        args.task_column: df.iloc[row_index][args.task_column],
                        "target_raw": df.iloc[row_index][args.target],
                        "target_binary": int(true_value),
                        "prediction": int(pred_value),
                        "score": float(score),
                        "fold": fold.label,
                    }
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    rows = []
    for (strategy, model, feature_set), group in fold_metrics.groupby(
        ["strategy", "model", "feature_set"]
    ):
        row = {
            "strategy": strategy,
            "model": model,
            "feature_set": feature_set,
            "folds": len(group),
            "mean_n_test": group["n_test"].mean(),
        }
        for metric in metric_columns:
            low, high = confidence_interval(group[metric])
            row[metric] = group[metric].mean()
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["strategy", "model", "feature_set"])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    for column in [args.subject_column, args.task_column, args.target]:
        if column not in df.columns:
            raise SystemExit(f"Missing required column: {column}")

    feature_columns = select_feature_columns(df, args)
    strategies = ["loso", "loto", "losato"] if args.strategy == "all" else [args.strategy]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []
    all_predictions = []
    for strategy in strategies:
        fold_metrics, predictions = run_strategy(df, args, strategy, feature_columns)
        all_fold_metrics.append(fold_metrics)
        all_predictions.append(predictions)

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize(fold_metrics)

    stem = f"{args.strategy}_{args.feature_set}_threshold_{args.threshold:g}"
    fold_metrics.to_csv(output_dir / f"{stem}_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / f"{stem}_predictions.csv", index=False)
    summary.to_csv(output_dir / f"{stem}_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
