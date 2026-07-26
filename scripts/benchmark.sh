#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --job-name=spmv-benchmark
#SBATCH --array=1-10
#SBATCH --output=spmv-benchmark-%A_%a.out
#SBATCH --error=spmv-benchmark-%A_%a.err

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

export LD_LIBRARY_PATH=$SLURM_SUBMIT_DIR/external/ucx/lib:$LD_LIBRARY_PATH
export UCX_TLS=rc,cuda_copy,cuda_ipc
export OMPI_MCA_opal_cuda_support=true

mpirun -np $SLURM_NTASKS ./bin/spmv $SLURM_SUBMIT_DIR/data $SLURM_SUBMIT_DIR/log