#!/bin/bash
# SLURM script for a multi-step job on a Compute Canada cluster. 
#SBATCH --account=def-hsajjad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000M
#SBATCH --time=24:00:00
#SBATCH --job-name=training_baseline_model
#SBATCH --output=/home/sher4672/Desktop/robustqa/output/robustqa-%j.out
#SBATCH --mail-user=sh545346@dal.ca
#SBATCH --mail-type=FAIL
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
module load python/3.8
source ~/env_robustqa/bin/activate
cd /home/sher4672/Desktop/robustqa
python train.py --do-train --eval-every 2000 --run-name baseline
echo "Job finished with exit code $? at: `date`"
