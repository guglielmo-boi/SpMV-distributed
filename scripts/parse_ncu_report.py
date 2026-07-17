#!/usr/bin/env python3

# This code was created with the help of generative artificial intelligence.

import argparse
import csv
from pathlib import Path
from collections import defaultdict

MATRICES = [
    "ASIC_680ks",
    "FullChip",
    "Ga41As41H72",
    "Rucci1",
    "Si41Ge41H72",
    "boyd2",
    "eu-2005",
    "ldoor",
    "rajat31",
    "webbase-1M",
]

METRICS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "total_bandwidth_pct",
    "dram__bytes.sum": "dram_bytes_mb",
    "l1tex__t_bytes.sum": "l1_bytes",
    "lts__t_bytes.sum": "l2_bytes",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
}

IMPLEMENTATIONS = [
    "CSR-Vector",
    "CSR-Stream",
    "CSR-Adaptive(Stream)",
    "CSR-Adaptive(Vector)",
    "cuSPARSE",
]


def to_float(value: str) -> float:
    value = value.replace(",", "").strip()

    if value.lower() in ("n/a", ""):
        return 0.0

    return float(value)


def kb_to_mb(value_kb: float) -> float:
    return value_kb / 1024.0


def read_report(csv_path: Path):
    rows = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 15:
                continue

            metric_name = row[-3].strip()

            if metric_name not in METRICS:
                continue

            metric_value = to_float(row[-1])

            rows.append({
                "kernel": row[4].strip(),
                "metric": metric_name,
                "value": metric_value,
            })

    return rows


def build_blocks(rows):
    block_size = len(METRICS)

    grouped = []

    for i in range(0, len(rows), block_size):
        block_rows = rows[i:i + block_size]

        if len(block_rows) != block_size:
            continue

        grouped.append(block_rows)

    return grouped


def aggregate_blocks(blocks):
    results = defaultdict(dict)

    for idx, block in enumerate(blocks):
        matrix_idx = idx // 5
        impl_idx = idx % 5

        if matrix_idx >= len(MATRICES):
            break

        matrix = MATRICES[matrix_idx]
        impl = IMPLEMENTATIONS[impl_idx]

        metrics = {}

        for row in block:
            metrics[row["metric"]] = row["value"]

        results[matrix][impl] = metrics

    return results


def make_output_dir(base_output_dir: Path):
    ncu_dir = base_output_dir / "ncu"
    ncu_dir.mkdir(parents=True, exist_ok=True)
    return ncu_dir


def write_single_impl_csv(output_path, impl_name, data):
    fieldnames = [
        "matrix",
        "total_bandwidth_pct",
        "dram_bytes_mb",
        "cache_bytes_mb",
        "sm_throughput_pct",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for matrix in MATRICES:
            if impl_name not in data[matrix]:
                continue

            metrics = data[matrix][impl_name]

            l1_mb = kb_to_mb(
                metrics["l1tex__t_bytes.sum"]
            )

            l2_mb = metrics["lts__t_bytes.sum"]

            cache_mb = l1_mb + l2_mb

            row = {
                "matrix": matrix,
                "total_bandwidth_pct":
                    f"{metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed']:.6f}",

                "dram_bytes_mb":
                    f"{metrics['dram__bytes.sum']:.6f}",

                "cache_bytes_mb":
                    f"{cache_mb:.6f}",

                "sm_throughput_pct":
                    f"{metrics['sm__throughput.avg.pct_of_peak_sustained_elapsed']:.6f}",
            }

            writer.writerow(row)


def write_adaptive_csv(output_path, data):
    fieldnames = [
        "matrix",
        "total_bandwidth_pct",
        "dram_bytes_mb",
        "cache_bytes_mb",
        "sm_throughput_pct",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for matrix in MATRICES:
            stream = data[matrix]["CSR-Adaptive(Stream)"]
            vector = data[matrix]["CSR-Adaptive(Vector)"]

            total_bandwidth = (
                stream["dram__throughput.avg.pct_of_peak_sustained_elapsed"] +
                vector["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
            ) / 2.0

            sm_throughput = (
                stream["sm__throughput.avg.pct_of_peak_sustained_elapsed"] +
                vector["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
            ) / 2.0

            dram_bytes = (
                stream["dram__bytes.sum"] +
                vector["dram__bytes.sum"]
            )

            l1_mb = kb_to_mb(
                stream["l1tex__t_bytes.sum"] +
                vector["l1tex__t_bytes.sum"]
            )

            l2_mb = (
                stream["lts__t_bytes.sum"] +
                vector["lts__t_bytes.sum"]
            )

            cache_mb = l1_mb + l2_mb

            row = {
                "matrix": matrix,
                "total_bandwidth_pct": f"{total_bandwidth:.6f}",
                "dram_bytes_mb": f"{dram_bytes:.6f}",
                "cache_bytes_mb": f"{cache_mb:.6f}",
                "sm_throughput_pct": f"{sm_throughput:.6f}",
            }

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Nsight Compute metrics into per-implementation CSV files."
    )

    parser.add_argument(
        "report_csv",
        type=Path,
        help="Path to the Nsight Compute CSV report"
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Base output directory"
    )

    args = parser.parse_args()

    output_dir = make_output_dir(args.output_dir)

    rows = read_report(args.report_csv)

    blocks = build_blocks(rows)

    data = aggregate_blocks(blocks)

    write_single_impl_csv(
        output_dir / "csr_vector.csv",
        "CSR-Vector",
        data
    )

    write_single_impl_csv(
        output_dir / "csr_stream.csv",
        "CSR-Stream",
        data
    )

    write_single_impl_csv(
        output_dir / "cusparse.csv",
        "cuSPARSE",
        data
    )

    write_adaptive_csv(
        output_dir / "csr_adaptive.csv",
        data
    )

    print(f"Generated files inside: {output_dir}")


if __name__ == "__main__":
    main()