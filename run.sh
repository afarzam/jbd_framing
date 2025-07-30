#!/bin/bash
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1 
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=jailbreak
#SBATCH -t 24:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=a.farzam@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /hpc/group/sapirolab/af304/jailbreak/slurm/slurm_%j.out
#SBATCH -e /hpc/group/sapirolab/af304/jailbreak/slurm/slurm_%j.err
#SBATCH --exclude dcc-youlab-gpu-[01-57],dcc-brownlab-gpu-[01-04],dcc-brunellab-gpu-[01-04],dcc-carin-gpu-[01-12],dcc-gehmlab-gpu-[01-50]

# Set up paths
PROJECT_DIR="/hpc/group/sapirolab/af304/jailbreak"
CONFIG_DIR="${PROJECT_DIR}/configs"

# Activate the environment and run the make commands
cd ${PROJECT_DIR}

# Setup environment
make setup
venv/bin/pip install ipykernel

# Run the main jailbreak experiment with default config
make run_jailbreak config_path=${CONFIG_DIR}/standard_gpt2_harmful_qa.json 