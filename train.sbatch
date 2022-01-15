#!/bin/bash
#SBATCH --job-name="reproj"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/yugay/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/yugay/logs/slurm-%j.out

source /usr/stud/yugay/anaconda3/etc/profile.d/conda.sh
conda activate mots_tracker

srun python scripts/run_tracker.py --cp configs/reproj_tracker_config.yaml
