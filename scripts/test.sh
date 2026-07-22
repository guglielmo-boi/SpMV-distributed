#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --job-name=spmv-test
#SBATCH --output=spmv-test-%j.out
#SBATCH --error=spmv-test-%j.err

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2  

export LD_LIBRARY_PATH=$SLURM_SUBMIT_DIR/external/ucx-install/lib:$LD_LIBRARY_PATH

set -e

cd $SLURM_SUBMIT_DIR/build

ctest --output-on-failure