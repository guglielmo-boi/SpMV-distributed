# SpMV CSR-Based Methods Investigation on GPU

This project investigates different GPU implementations of Sparse Matrix-Vector Multiplication (SpMV) using the compressed sparse row (CSR) sparse matrix format.

* CSR-Vector
* CSR-Stream
* CSR-Adaptive
* cuSPARSE (baseline)

The project evaluates kernel execution time, GFLOPS, memory traffic, Streaming Multiprocessor throughput, and bandwidth utilization across several sparse matrices from the SuiteSparse collection. To reproduce the results download the following matrices in the data folder (.mtx format).

## Dataset
```text
bcircuit bcsstk14 boyd2 memplus scircuit net25 net50 net75 net100 net125 net150
```

## Build

```bash
sbatch ./scripts/build.sh
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
tests/        SpMV and MPI SpMV tests
scripts/      SLURM and Python scripts
data/         Sparse matrices dataset
results/      Benchmark results
log/          Benchmark logs
references/   Papers and reference material
external/     External dependencies
```
