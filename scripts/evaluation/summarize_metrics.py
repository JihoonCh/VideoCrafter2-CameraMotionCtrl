"""Aggregate per-video CSVs from evaluate_temporal_metrics.py into the paper's Table 2."""
import argparse
import csv
import glob
import os

import numpy as np

COLUMNS = ["warping_error_mean", "temporal_consistency_mean", "valid_ratio_mean"]


def load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return np.array([[float(row[c]) for c in COLUMNS] for row in csv.DictReader(handle)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=str, default="metrics")
    args = parser.parse_args()

    print(f"{'file':32s} {'n':>4s} {'WE':>8s} {'TC':>8s} {'Valid':>8s}")
    for method in ["base", "opt"]:
        stacked = []
        for path in sorted(glob.glob(os.path.join(args.metrics_dir, f"{method}_*_metrics.csv"))):
            values = load(path)
            stacked.append(values)
            we, tc, valid = values.mean(0)
            print(f"{os.path.basename(path):32s} {len(values):4d} {we:8.4f} {tc:8.4f} {valid:8.4f}")
        if stacked:
            values = np.concatenate(stacked)
            we, tc, valid = values.mean(0)
            print(f"{'[' + method + '] all motions':32s} {len(values):4d} {we:8.4f} {tc:8.4f} {valid:8.4f}\n")


if __name__ == "__main__":
    main()
