#!/usr/bin/env python3
"""
Paired Bootstrap Test & Permutation Test for lm-evaluation-harness results.

Usage:
    python significance_test.py \
        --model_a_dir /path/to/modelA/results/ \
        --model_b_dir /path/to/modelB/results/ \
        --tasks arc_easy,arc_challenge,hellaswag,sciq,piqa \
        --metric acc \
        --test_type both \
        --n_resamples 10000 \
        --alpha 0.05 \
        --correction bonferroni

The script reads the per-sample JSONL files produced by `lm_eval --log_samples`
and performs paired bootstrap / permutation tests between two models.

Output format of --log_samples:
    Each task produces a file: samples_{task_name}_{date}.jsonl
    Each line is a JSON object with fields: doc_id, acc, acc_norm, ...
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load_samples(result_dir, task_name, metric):
    """
    Load per-sample metric values from JSONL files.
    Returns dict: {doc_id: metric_value}
    """
    pattern = os.path.join(result_dir, "**", f"samples_{task_name}_*.jsonl")
    files = glob.glob(pattern, recursive=True)
    if not files:
        pattern = os.path.join(result_dir, f"samples_{task_name}_*.jsonl")
        files = glob.glob(pattern)
    if not files:
        return None

    # Use the most recent file if multiple exist
    files.sort(key=os.path.getmtime)
    filepath = files[-1]

    samples = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record["doc_id"]
            if metric not in record:
                raise ValueError(
                    f"Metric '{metric}' not found in sample for task '{task_name}'. "
                    f"Available metrics: {record.get('metrics', [])}"
                )
            samples[doc_id] = record[metric]
    return samples


def align_samples(samples_a, samples_b):
    """
    Align two sample dicts by doc_id.
    Returns two numpy arrays of the same length, ordered by doc_id.
    """
    common_ids = sorted(set(samples_a.keys()) & set(samples_b.keys()))
    if len(common_ids) == 0:
        raise ValueError("No common doc_ids found between model A and model B.")

    n_a = len(samples_a)
    n_b = len(samples_b)
    n_common = len(common_ids)
    if n_common < min(n_a, n_b):
        print(
            f"  Warning: Only {n_common}/{n_a} (A) and {n_common}/{n_b} (B) "
            f"samples matched by doc_id."
        )

    arr_a = np.array([samples_a[did] for did in common_ids], dtype=np.float64)
    arr_b = np.array([samples_b[did] for did in common_ids], dtype=np.float64)
    return arr_a, arr_b


def paired_bootstrap_test(arr_a, arr_b, n_resamples=10000, rng=None):
    """
    Paired bootstrap test (two-sided).
    Returns: observed_diff, ci_lower, ci_upper, p_value
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(arr_a)
    observed_diff = np.mean(arr_a) - np.mean(arr_b)

    bootstrap_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        bootstrap_diffs[i] = np.mean(arr_a[idx]) - np.mean(arr_b[idx])

    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    # Two-sided p-value: center the bootstrap distribution, then count
    # how often the centered distribution produces a diff as extreme as observed
    centered_diffs = bootstrap_diffs - observed_diff
    p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))

    return observed_diff, ci_lower, ci_upper, p_value


def permutation_test(arr_a, arr_b, n_resamples=10000, rng=None):
    """
    Paired permutation test (two-sided).
    For each sample, randomly swap A/B labels.
    Returns: observed_diff, p_value
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(arr_a)
    observed_diff = np.mean(arr_a) - np.mean(arr_b)
    diff_per_sample = arr_a - arr_b

    count_extreme = 0
    for _ in range(n_resamples):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(diff_per_sample * signs)
        if np.abs(perm_diff) >= np.abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_resamples
    return observed_diff, p_value


def holm_bonferroni_correction(p_values, alpha=0.05):
    """
    Holm-Bonferroni correction for multiple comparisons.
    Returns list of (corrected_p, is_significant) in original order.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * m
    max_corrected = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_p = min(p * (m - rank), 1.0)
        adjusted_p = max(adjusted_p, max_corrected)
        max_corrected = adjusted_p
        corrected[orig_idx] = (adjusted_p, adjusted_p < alpha)
    return corrected


def bonferroni_correction(p_values, alpha=0.05):
    """
    Bonferroni correction for multiple comparisons.
    """
    m = len(p_values)
    return [(min(p * m, 1.0), min(p * m, 1.0) < alpha) for p in p_values]


def main():
    parser = argparse.ArgumentParser(
        description="Paired significance tests for lm-evaluation-harness results"
    )
    parser.add_argument(
        "--model_a_dir", required=True,
        help="Directory containing model A's evaluation results (with --log_samples output)",
    )
    parser.add_argument(
        "--model_b_dir", required=True,
        help="Directory containing model B's evaluation results (with --log_samples output)",
    )
    parser.add_argument(
        "--tasks", required=True,
        help="Comma-separated list of tasks to compare (e.g., arc_easy,hellaswag)",
    )
    parser.add_argument(
        "--metric", default="acc",
        help="Metric to compare (default: acc). Other options: acc_norm, exact_match",
    )
    parser.add_argument(
        "--test_type", choices=["bootstrap", "permutation", "both"], default="both",
        help="Type of significance test to run (default: both)",
    )
    parser.add_argument(
        "--n_resamples", type=int, default=10000,
        help="Number of bootstrap/permutation resamples (default: 10000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--correction", choices=["bonferroni", "holm", "none"], default="holm",
        help="Multiple comparison correction method (default: holm)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results as JSON (optional)",
    )
    parser.add_argument(
        "--model_a_name", default="Model A",
        help="Display name for model A",
    )
    parser.add_argument(
        "--model_b_name", default="Model B",
        help="Display name for model B",
    )

    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    rng = np.random.default_rng(args.seed)

    results = []
    raw_p_values = []

    print(f"\nLoading per-sample results...")
    print(f"  Model A: {args.model_a_dir}")
    print(f"  Model B: {args.model_b_dir}")
    print(f"  Metric : {args.metric}")
    print()

    for task in tasks:
        samples_a = load_samples(args.model_a_dir, task, args.metric)
        samples_b = load_samples(args.model_b_dir, task, args.metric)

        if samples_a is None:
            print(f"  [SKIP] {task}: No sample file found for Model A")
            continue
        if samples_b is None:
            print(f"  [SKIP] {task}: No sample file found for Model B")
            continue

        arr_a, arr_b = align_samples(samples_a, samples_b)
        n_samples = len(arr_a)
        mean_a = np.mean(arr_a)
        mean_b = np.mean(arr_b)

        entry = {
            "task": task,
            "n_samples": n_samples,
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
        }

        if args.test_type in ("bootstrap", "both"):
            obs_diff, ci_lo, ci_hi, p_boot = paired_bootstrap_test(
                arr_a, arr_b, n_resamples=args.n_resamples, rng=rng
            )
            entry["bootstrap"] = {
                "observed_diff": float(obs_diff),
                "ci_95_lower": float(ci_lo),
                "ci_95_upper": float(ci_hi),
                "p_value": float(p_boot),
            }

        if args.test_type in ("permutation", "both"):
            obs_diff, p_perm = permutation_test(
                arr_a, arr_b, n_resamples=args.n_resamples, rng=rng
            )
            entry["permutation"] = {
                "observed_diff": float(obs_diff),
                "p_value": float(p_perm),
            }

        # Use bootstrap p-value for correction if available, else permutation
        if "bootstrap" in entry:
            raw_p_values.append(entry["bootstrap"]["p_value"])
        elif "permutation" in entry:
            raw_p_values.append(entry["permutation"]["p_value"])

        results.append(entry)

    if not results:
        print("No tasks could be compared. Exiting.")
        sys.exit(1)

    # Apply multiple comparison correction
    if args.correction == "bonferroni":
        corrected = bonferroni_correction(raw_p_values, args.alpha)
    elif args.correction == "holm":
        corrected = holm_bonferroni_correction(raw_p_values, args.alpha)
    else:
        corrected = [(p, p < args.alpha) for p in raw_p_values]

    for i, entry in enumerate(results):
        entry["corrected_p_value"] = float(corrected[i][0])
        entry["significant"] = bool(corrected[i][1])

    # Print results table
    correction_label = {
        "bonferroni": "Bonferroni",
        "holm": "Holm-Bonferroni",
        "none": "None",
    }[args.correction]

    print("=" * 105)
    print(
        f"Significance Testing Results "
        f"(alpha={args.alpha}, correction={correction_label}, "
        f"n_resamples={args.n_resamples})"
    )
    print(f"  {args.model_a_name} vs {args.model_b_name}")
    print("=" * 105)

    if args.test_type in ("bootstrap", "both"):
        header = (
            f"{'Task':<18} {'N':>6} "
            f"{'A':>8} {'B':>8} {'Diff':>8} "
            f"{'95% CI':>18} {'p-val':>8} {'p-corr':>8} {'Sig?':>5}"
        )
        print(header)
        print("-" * 105)

        for entry in results:
            bs = entry["bootstrap"]
            sig_marker = "*" if entry["significant"] else ""
            ci_str = f"[{bs['ci_95_lower']:+.4f}, {bs['ci_95_upper']:+.4f}]"
            print(
                f"{entry['task']:<18} {entry['n_samples']:>6} "
                f"{entry['mean_a']:>8.4f} {entry['mean_b']:>8.4f} "
                f"{bs['observed_diff']:>+8.4f} "
                f"{ci_str:>18} "
                f"{bs['p_value']:>8.4f} "
                f"{entry['corrected_p_value']:>8.4f} "
                f"{sig_marker:>5}"
            )
        print()

    if args.test_type in ("permutation", "both"):
        print(f"{'Task':<18} {'N':>6} {'Diff':>8} {'p-val(perm)':>12} {'p-corr':>8} {'Sig?':>5}")
        print("-" * 65)
        for entry in results:
            pm = entry["permutation"]
            sig_marker = "*" if entry["significant"] else ""
            print(
                f"{entry['task']:<18} {entry['n_samples']:>6} "
                f"{pm['observed_diff']:>+8.4f} "
                f"{pm['p_value']:>12.4f} "
                f"{entry['corrected_p_value']:>8.4f} "
                f"{sig_marker:>5}"
            )
        print()

    print("=" * 105)
    print(f"* = significant at alpha={args.alpha} after {correction_label} correction")
    print()

    # Save JSON output
    if args.output:
        output_data = {
            "config": {
                "model_a_dir": args.model_a_dir,
                "model_b_dir": args.model_b_dir,
                "model_a_name": args.model_a_name,
                "model_b_name": args.model_b_name,
                "metric": args.metric,
                "test_type": args.test_type,
                "n_resamples": args.n_resamples,
                "alpha": args.alpha,
                "correction": args.correction,
                "seed": args.seed,
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
