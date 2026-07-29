# Distributed SpMV on Multi-GPU CUDA-Aware Systems

This project investigates multi-GPU implementations of Sparse Matrix-Vector Multiplication (SpMV) with different partitioning strategies.

* 1D Block Partitioning
* 1D Cyclic Partitioning
* 1D Block Partitioning with CUDA-aware MPI

The project evaluates execution time, FLOPS, speedup, and efficiency across several sparse matrices from the SuiteSparse collection. To reproduce the results [download](https://drive.google.com/drive/folders/1igGDWuASZ_pCRLqSsePVie3KiZ3GETQa?usp=sharing) the following matrices in the data folder (.mtx format).

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
