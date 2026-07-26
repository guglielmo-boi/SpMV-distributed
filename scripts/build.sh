#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --job-name=spmv-build
#SBATCH --output=spmv-build-%j.out
#SBATCH --error=spmv-build-%j.err

module load CUDA/12.1.1
module load OpenMPI/4.1.5-GCC-12.3.0
module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1

export OMPI_MCA_opal_cuda_support=true

BUILD_DIR="${SLURM_SUBMIT_DIR}/build"
BIN_DIR="${SLURM_SUBMIT_DIR}/bin"

rm -rf "${BUILD_DIR}" "${BIN_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}"

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed. Exiting job."
    exit 1
fi