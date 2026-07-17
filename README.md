# SpMV CSR-Based Methods Investigation on GPU

This project investigates different GPU implementations of Sparse Matrix-Vector Multiplication (SpMV) using the compressed sparse row (CSR) sparse matrix format.

* CSR-Vector
* CSR-Stream
* CSR-Adaptive
* cuSPARSE (baseline)

The project evaluates kernel execution time, GFLOPS, memory traffic, Streaming Multiprocessor throughput, and bandwidth utilization across several sparse matrices from the SuiteSparse collection. To reproduce the results download the following matrices in the data folder (.mtx format).

## Dataset
```text
ASIC_680ks FullChip Ga41As41H72 Rucci1 Si41Ge41H72 boyd2 eu-2005 ldoor rajat31 webbase-1M
```

## Build

```bash
cmake -B build
cmake --build build
```

## Run Benchmarks

```bash
sbatch ./scripts/benchmark.sh
```

## Run Tests

```bash
sbatch ./scripts/test.sh
```

## Run NVIDIA Nsight Compute
```bash
sbatch ./scripts/ncu.sh
```

## Repository Structure

```text
include/      Header files
src/          Source code
tests/        SpMV implementation tests
scripts/      SLURM and Python scripts
data/         Sparse matrices dataset
results/      Benchmark results
log/          Benchmark logs
references/   Papers and reference material
external/     External dependencies
```
