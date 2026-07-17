#!/usr/bin/env python3

# This code was created with the help of generative artificial intelligence.

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict


CSV_FILES = [
    "csr_adaptive.csv",
    "csr_stream.csv",
    "csr_vector.csv",
    "cusparse.csv",
]

METRICS = [
    "total_execution_time",
    "kernel_execution_time",
    "total_gflops",
    "kernel_gflops",
    "preprocessing_time_percentage",
]


def read_metric(csv_path, metric_name):
    """
    Reads a metric grouped by matrix_id from a CSV file.
    """
    values = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            matrix_id = row["matrix_id"]

            if metric_name == "preprocessing_time_percentage":
                total_time = float(row["total_execution_time"])
                kernel_time = float(row["kernel_execution_time"])

                metric_value = (
                    (total_time - kernel_time) / total_time * 100.0
                )

            else:
                metric_value = float(row[metric_name])

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
    metric_output_dir = output_dir / metric_name
    metric_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = metric_output_dir / filename

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "matrix_id",
            f"mean_{metric_name}",
            f"min_{metric_name}",
            f"max_{metric_name}",
            f"mse_{metric_name}",
            f"stddev_{metric_name}",
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


def print_kernel_execution_speedups(output_dir):
    """
    Prints speedups relative to cuSPARSE using:
        speedup = cusparse_time / method_time
    """

    metric_dir = output_dir / "kernel_execution_time"

    methods = {
        "CSR-Vector": "csr_vector.csv",
        "CSR-Stream": "csr_stream.csv",
        "CSR-Adaptive": "csr_adaptive.csv",
    }

    cusparse_file = metric_dir / "cusparse.csv"

    # Read cuSPARSE means
    cusparse_times = {}

    with open(cusparse_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            matrix_id = row["matrix_id"]
            mean_time = float(row["mean_kernel_execution_time"])

            cusparse_times[matrix_id] = mean_time

    print("\n=== Kernel Execution Time Speedups (vs cuSPARSE) ===\n")

    for method_name, filename in methods.items():
        print(f"{method_name}:")

        method_file = metric_dir / filename

        with open(method_file, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                matrix_id = row["matrix_id"]

                method_time = float(
                    row["mean_kernel_execution_time"]
                )

                cusparse_time = cusparse_times[matrix_id]

                speedup = cusparse_time / method_time

                print(
                    f"  {matrix_id:15s} "
                    f"{speedup:.3f}x"
                )

        print()


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

    for metric_name in METRICS:
        for filename in CSV_FILES:
            process_method(
                data_dir,
                output_dir,
                filename,
                metric_name
            )

    print_kernel_execution_speedups(output_dir)

if __name__ == "__main__":
    main()