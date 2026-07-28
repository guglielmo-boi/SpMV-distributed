#!/usr/bin/env python3

# This code was created with the help of generative artificial intelligence.

import argparse
import csv
from pathlib import Path


CSV_FILES = [
    "oned_block.csv",
    "oned_cyclic.csv",
    "oned_block_cudaaware.csv",
]

CONFIGURATIONS = [
    "r1",
    "r2",
    "r4",
]

DATASETS = {
    "dataset_strong": [
        "bcircuit",
        "bcsstk14",
        "boyd2",
        "memplus",
        "scircuit",
    ],
    "dataset_weak": [
        "net25",
        "net50",
        "net75",
        "net100",
        "net125",
        "net150",
    ],
}


def read_execution_times(csv_path):
    values = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            values[row["matrix_id"]] = float(
                row["mean_total_execution_time"]
            )

    return values


def generate_speedup(results_dir):
    speedup_dir = results_dir / "speedup"
    speedup_dir.mkdir(parents=True, exist_ok=True)

    for filename in CSV_FILES:

        execution_times = {}

        for config in CONFIGURATIONS:
            csv_path = (
                results_dir /
                config /
                "total_execution_time" /
                filename
            )

            execution_times[config] = read_execution_times(csv_path)

        output_path = speedup_dir / filename

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "matrix_id",
                "r1",
                "r2",
                "r4",
            ])

            for matrix_id in sorted(execution_times["r1"].keys()):

                baseline = execution_times["r1"][matrix_id]

                writer.writerow([
                    matrix_id,
                    f"{baseline / execution_times['r1'][matrix_id]:.6f}",
                    f"{baseline / execution_times['r2'][matrix_id]:.6f}",
                    f"{baseline / execution_times['r4'][matrix_id]:.6f}",
                ])

        print(f"Generated: {output_path}")


def generate_speedup_average(results_dir, dataset_name):
    speedup_dir = results_dir / "speedup"

    output_file = speedup_dir / f"average_{dataset_name}.csv"

    implementations = {
        "oned_block": "oned_block.csv",
        "oned_cyclic": "oned_cyclic.csv",
        "oned_block_cudaaware": "oned_block_cudaaware.csv",
    }

    matrices = DATASETS[dataset_name]

    rows = []

    for gpus in ["r1", "r2", "r4"]:

        row = [
            {"r1": 1, "r2": 2, "r4": 4}[gpus]
        ]

        for _, filename in implementations.items():

            csv_path = speedup_dir / filename

            values = []

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)

                for r in reader:
                    if r["matrix_id"] in matrices:
                        values.append(float(r[gpus]))

            row.append(sum(values) / len(values))

        row.append({"r1": 1.0, "r2": 2.0, "r4": 4.0}[gpus])

        rows.append(row)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "gpus",
            "oned_block",
            "oned_cyclic",
            "oned_block_cudaaware",
            "ideal",
        ])

        for row in rows:
            writer.writerow([f"{v:.3f}" if isinstance(v, float) else v for v in row])

    print(f"Generated: {output_file}")


def generate_efficiency(results_dir):
    efficiency_dir = results_dir / "efficiency"
    efficiency_dir.mkdir(parents=True, exist_ok=True)

    processes = {
        "r1": 1,
        "r2": 2,
        "r4": 4,
    }

    for filename in CSV_FILES:

        execution_times = {}

        for config in CONFIGURATIONS:
            csv_path = (
                results_dir /
                config /
                "total_execution_time" /
                filename
            )

            execution_times[config] = read_execution_times(csv_path)

        output_path = efficiency_dir / filename

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "matrix_id",
                "r1",
                "r2",
                "r4",
            ])

            for matrix_id in sorted(execution_times["r1"].keys()):

                baseline = execution_times["r1"][matrix_id]

                row = [matrix_id]

                for config in CONFIGURATIONS:
                    speedup = (
                        baseline /
                        execution_times[config][matrix_id]
                    )

                    efficiency = speedup / processes[config]

                    row.append(f"{efficiency:.6f}")

                writer.writerow(row)

        print(f"Generated: {output_path}")


def generate_efficiency_average(results_dir, dataset_name):
    efficiency_dir = results_dir / "efficiency"

    output_file = efficiency_dir / f"average_{dataset_name}.csv"

    implementations = {
        "oned_block": "oned_block.csv",
        "oned_cyclic": "oned_cyclic.csv",
        "oned_block_cudaaware": "oned_block_cudaaware.csv",
    }

    matrices = DATASETS[dataset_name]

    rows = []

    for gpus in ["r1", "r2", "r4"]:

        row = [
            {"r1": 1, "r2": 2, "r4": 4}[gpus]
        ]

        for _, filename in implementations.items():

            csv_path = efficiency_dir / filename

            values = []

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)

                for r in reader:
                    if r["matrix_id"] in matrices:
                        values.append(float(r[gpus]))

            row.append(sum(values) / len(values))

        row.append({"r1": 1.0, "r2": 1.0, "r4": 1.0}[gpus])

        rows.append(row)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "gpus",
            "oned_block",
            "oned_cyclic",
            "oned_block_cudaaware",
            "ideal",
        ])

        for row in rows:
            writer.writerow([f"{v:.3f}" if isinstance(v, float) else v for v in row])

    print(f"Generated: {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "results_dir",
        help="Directory containing r1/, r2/ and r4/"
    )

    args = parser.parse_args()

    generate_speedup(Path(args.results_dir))
    generate_speedup_average(Path(args.results_dir), "dataset_strong")
    generate_speedup_average(Path(args.results_dir), "dataset_weak")
    
    generate_efficiency(Path(args.results_dir))
    generate_efficiency_average(Path(args.results_dir), "dataset_strong")
    generate_efficiency_average(Path(args.results_dir), "dataset_weak")

if __name__ == "__main__":
    main()