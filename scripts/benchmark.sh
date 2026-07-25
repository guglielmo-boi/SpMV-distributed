#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --job-name=spmv-benchmark
#SBATCH --output=spmv-benchmark-%j.out
#SBATCH --error=spmv-benchmark-%j.err

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

export LD_LIBRARY_PATH=$SLURM_SUBMIT_DIR/external/ucx/lib:$LD_LIBRARY_PATH
export UCX_TLS=all
export OMPI_MCA_opal_cuda_support=true mpirun -np $SLURM_NTASKS ./bin/spmv $SLURM_SUBMIT_DIR/data $SLURM_SUBMIT_DIR/log

mpirun -np $SLURM_NTASKS ./bin/spmv $SLURM_SUBMIT_DIR/data $SLURM_SUBMIT_DIR/log