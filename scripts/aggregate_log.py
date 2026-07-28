#!/usr/bin/env python3

# This code was created with the help of generative artificial intelligence.

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict


CSV_FILES = [
    "oned_block.csv",
    "oned_cyclic.csv",
    "oned_block_cudaaware.csv",
]

BASE_METRICS = [
    "total_execution_time",
    "total_gflops",
]


def get_num_ranks(data_dir):
    """
    data_dir is expected to be .../r1, .../r2 or .../r4
    """
    return int(data_dir.name[1:])


def build_metrics(num_ranks):
    metrics = BASE_METRICS.copy()

    for rank in range(num_ranks):
        metrics.extend([
            f"{rank}_nnz",
            f"{rank}_kernel_execution_time",
            f"{rank}_kernel_gflops",
        ])

    return metrics


def read_metric(csv_path, metric_name):
    """
    Reads a metric grouped by matrix_id from a CSV file.
    """
    values = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            matrix_id = row["matrix_id"]

            metric_value = float(row[metric_name])

            if metric_name.endswith("_gflops"):
                metric_value *= 1000.0

            values[matrix_id].append(metric_value)

    return values


def compute_statistics(samples):
    """
    Computes:
    - mean
    - min
    - max
    - MSE (against the mean)
    - standard deviation
    """
    n = len(samples)

    if n == 0:
        return None

    mean = sum(samples) / n
    minimum = min(samples)
    maximum = max(samples)

    mse = sum((x - mean) ** 2 for x in samples) / n
    stddev = math.sqrt(mse)

    return {
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "mse": mse,
        "stddev": stddev,
    }


def process_method(data_dir, output_dir, filename, metric_name):
    """
    Processes all CSV files with the same method name
    across all subfolders for a given metric.
    """
    aggregated = defaultdict(list)

    csv_paths = list(data_dir.rglob(filename))

    for csv_path in csv_paths:
        method_values = read_metric(csv_path, metric_name)

        for matrix_id, values in method_values.items():
            aggregated[matrix_id].extend(values)

    # Create metric output directory
    output_metric_name = metric_name.replace("gflops", "mflops")
    metric_output_dir = output_dir / output_metric_name
    metric_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = metric_output_dir / filename

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "matrix_id",
            f"mean_{output_metric_name}",
            f"min_{output_metric_name}",
            f"max_{output_metric_name}",
            f"mse_{output_metric_name}",
            f"stddev_{output_metric_name}",
        ])

        for matrix_id in sorted(aggregated.keys()):
            stats = compute_statistics(aggregated[matrix_id])

            writer.writerow([
                matrix_id,
                f"{stats['mean']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                f"{stats['mse']:.6f}",
                f"{stats['stddev']:.6f}",
            ])

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SpMV benchmark CSV files."
    )

    parser.add_argument(
        "data_dir",
        help="Path to input data directory"
    )

    parser.add_argument(
        "output_dir",
        help="Path to output directory"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    num_ranks = get_num_ranks(data_dir)
    metrics = build_metrics(num_ranks)

    for metric_name in metrics:
        for filename in CSV_FILES:
            process_method(
                data_dir,
                output_dir,
                filename,
                metric_name
            )


if __name__ == "__main__":
    main()