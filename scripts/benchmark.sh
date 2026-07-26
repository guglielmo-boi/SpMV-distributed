#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --job-name=spmv-benchmark
#SBATCH --array=1-1
#SBATCH --output=spmv-benchmark-%A_%a.out
#SBATCH --error=spmv-benchmark-%A_%a.err

module load UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1
module load OpenMPI/4.1.5-GCC-12.3.0

export OMPI_MCA_opal_cuda_support=true

BUILD_DIR="${SLURM_SUBMIT_DIR}/build"

echo "=== Starting Build Process ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || exit 1
cmake "${SLURM_SUBMIT_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel "${SLURM_CPUS_PER_TASK}"

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed. Exiting job."
    exit 1
fi

echo "=== Build Completed Successfully ==="

cd "${SLURM_SUBMIT_DIR}" || exit 1

mkdir -p "${SLURM_SUBMIT_DIR}/log"

mpirun -np $SLURM_NTASKS ./bin/spmv "${SLURM_SUBMIT_DIR}/data" "${SLURM_SUBMIT_DIR}/log"