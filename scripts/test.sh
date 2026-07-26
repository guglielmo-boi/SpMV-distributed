#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --job-name=spmv-test
#SBATCH --output=spmv-test-%j.out
#SBATCH --error=spmv-test-%j.err

module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1
module load OpenMPI/4.1.5-GCC-12.3.0

export OMPI_MCA_opal_cuda_support=true

set -e

BUILD_DIR="${SLURM_SUBMIT_DIR}/build"
echo "=== Starting Build Process ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SLURM_SUBMIT_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel "${SLURM_CPUS_PER_TASK}"

echo "=== Build Completed Successfully ==="

ctest --output-on-failure