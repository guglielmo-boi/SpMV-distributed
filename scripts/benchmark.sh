#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --job-name=spmv-benchmark
#SBATCH --output=spmv-benchmark-%j.out
#SBATCH --error=spmv-benchmark-%j.err

module load CUDA/12.1.1
module load OpenMPI/4.1.5-GCC-12.3.0
module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1

export OMPI_MCA_opal_cuda_support=true

mkdir -p "${SLURM_SUBMIT_DIR}/log"

mpirun -np $SLURM_NTASKS "${SLURM_SUBMIT_DIR}/bin/spmv" "${SLURM_SUBMIT_DIR}/data" "${SLURM_SUBMIT_DIR}/log"