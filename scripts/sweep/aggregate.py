#!/usr/bin/env python
"""Aggregate sweep results into a CSV.

Usage:
    python scripts/sweep/aggregate.py \
        --root output/sweep --dataset stanford_cars \
        --output output/sweep/stanford_cars_summary.csv

Parses accuracy from log files under each method/epoch run directory.
Skips runs where no accuracy line is found (prints warning to stderr).
"""
import argparse
import csv
import os
import re
import sys


def find_accuracy(log_path):
    """Parse accuracy value from a log file.

    Looks for lines matching the pattern  ``* accuracy: XX.XX%``.
    Returns float or None.
    """
    pat = re.compile(r"accuracy:\s*(\d+\.?\d*)%")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


def main():
    p = argparse.ArgumentParser(description="Aggregate sweep results to CSV")
    p.add_argument("--root", required=True, help="Root sweep output directory")
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. stanford_cars)")
    p.add_argument("--output", required=True, help="Output CSV path")
    args = p.parse_args()

    dataset_dir = os.path.join(args.root, args.dataset)
    if not os.path.isdir(dataset_dir):
        print(f"Error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    seed_pat = re.compile(r"seed(\d+)")
    run_pat = re.compile(r"(baseline|maml)_ft(\d+)")

    for seed_name in sorted(os.listdir(dataset_dir)):
        seed_path = os.path.join(dataset_dir, seed_name)
        if not os.path.isdir(seed_path):
            continue
        sm = seed_pat.match(seed_name)
        if not sm:
            continue
        seed = sm.group(1)

        for run_name in sorted(os.listdir(seed_path)):
            run_path = os.path.join(seed_path, run_name)
            if not os.path.isdir(run_path):
                continue
            rm = run_pat.match(run_name)
            if not rm:
                continue
            method = rm.group(1)
            ft_epoch = rm.group(2)

            # Walk the run directory tree looking for an accuracy line
            acc = None
            for dirpath, _, filenames in os.walk(run_path):
                for fn in filenames:
                    fp = os.path.join(dirpath, fn)
                    result = find_accuracy(fp)
                    if result is not None:
                        acc = result
                        break
                if acc is not None:
                    break

            if acc is not None:
                rows.append({
                    "dataset": args.dataset,
                    "seed": seed,
                    "method": method,
                    "ft_epoch": ft_epoch,
                    "accuracy": acc,
                })
            else:
                print(
                    f"Warning: no accuracy found in {run_path}",
                    file=sys.stderr,
                )

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "seed", "method", "ft_epoch", "accuracy"]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
