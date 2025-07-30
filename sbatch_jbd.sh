#!/bin/bash
#SBATCH --job-name=dec
#SBATCH --partition=hpc-high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --output=./slurm/jbd/slurm-%j.out
#SBATCH --error=./slurm/jbd/slurm-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=amirh.farzam@gmail.com

# Activate your environment if needed
source ~/jupyter-env/bin/activate

cd /mnt/home/amir/framingdecomp/framingDecomp/

# Forward all arguments to the python script
python jailbreak_detect_nsp.py "$@" 