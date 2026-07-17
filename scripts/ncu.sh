#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

#SBATCH --job-name=spmv-ncu
#SBATCH --array=1-10
#SBATCH --output=spmv-ncu-%A_%a.out
#SBATCH --error=spmv-ncu-%A_%a.err

module load CUDA/12.1.1

export TMPDIR=$HOME/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMPDIR

REPORT_NAME=ncu-report-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

ncu \
    --target-processes all \
    --metrics \
        dram__bytes.sum,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        l1tex__t_bytes.sum,\
        lts__t_bytes.sum,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --export ${REPORT_NAME} \
    ./bin/spmv \
    $SLURM_SUBMIT_DIR/data \
    $SLURM_SUBMIT_DIR/log

ncu \
    --import ${REPORT_NAME}.ncu-rep \
    --page details \
    --csv \
    > ${REPORT_NAME}.csv 2>&1