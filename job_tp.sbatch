#!/bin/bash
#SBATCH --job-name=TDA_tp
#SBATCH --account=<account>
#SBATCH --partition=gpuA40x4-preempt,gpuA100x4-preempt,gpuA100x8,gpuH200x8,gpuA40x4,gpuA100x4,gpuMI100x8
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1 --mem=100g
#SBATCH --time=04:00:00 --requeue
#SBATCH --output=$HOME/tda_imc/logs/%x_%j.out
#SBATCH --error=$HOME/tda_imc/logs/%x_%j.out
case "${SLURM_JOB_NAME:-}" in TDA_*) ;; *) scontrol update jobid=$SLURM_JOB_ID JobName=TDA_${SLURM_JOB_NAME} >/dev/null 2>&1 || true ;; esac
source $HOME/tda_imc/code/act.sh
cd $HOME/tda_imc/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u runlist.py "$1" "${2:-30}"
echo JOB_EXIT=$?
