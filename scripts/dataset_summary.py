#!/usr/bin/env python3
"""Summarize the merged Study 0/Study 0+ dataset used by the paper scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data/combined_study_data.csv")
    parser.add_argument("--subject-column", default="vol")
    parser.add_argument("--task-column", default="task")
    parser.add_argument("--target", default="entendimento")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    required = [args.subject_column, args.task_column, args.target]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    print(f"dataset: {Path(args.data)}")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print(f"subjects: {df[args.subject_column].nunique()}")
    print(f"tasks: {df[args.task_column].nunique()}")
    print(
        "subject-task observations: "
        f"{df[[args.subject_column, args.task_column]].drop_duplicates().shape[0]}"
    )
    print()
    print("Rows per task:")
    print(
        df.groupby(args.task_column)
        .agg(
            rows=(args.subject_column, "size"),
            subjects=(args.subject_column, "nunique"),
            mean_target=(args.target, "mean"),
            sd_target=(args.target, "std"),
        )
        .to_string()
    )
    print()
    print("Rows per subject:")
    print(df[args.subject_column].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
