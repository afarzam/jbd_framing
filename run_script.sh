#!/bin/bash
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1 
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=%x
#SBATCH -t 24:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=a.farzam@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /hpc/group/sapirolab/af304/jailbreak/slurm/slurm_%x_%j.out
#SBATCH -e /hpc/group/sapirolab/af304/jailbreak/slurm/slurm_%x_%j.err
#SBATCH --exclude dcc-youlab-gpu-[01-57],dcc-brownlab-gpu-[01-04],dcc-brunellab-gpu-[01-04],dcc-carin-gpu-[01-12],dcc-gehmlab-gpu-[01-50]

# Parse command line arguments
RUN_COMMAND=$1
EXPERIMENT_TYPE=$2
MODEL=$3
DATASET=$4

# Set up paths
PROJECT_DIR="/hpc/group/sapirolab/af304/jailbreak"
CONFIG_DIR="${PROJECT_DIR}/configs"

# Activate the environment and run the make commands
cd ${PROJECT_DIR}

# Setup environment
make setup
venv/bin/pip install ipykernel

# Run the specified experiment
case ${RUN_COMMAND} in
    "jailbreak")
        make run_jailbreak config_path=${CONFIG_DIR}/${EXPERIMENT_TYPE}_${MODEL}_${DATASET}.json
        ;;
    "detector")
        make run_detector config_path=${CONFIG_DIR}/${EXPERIMENT_TYPE}_${MODEL}_${DATASET}.json
        ;;
    "align")
        make run_align config_path=${CONFIG_DIR}/${EXPERIMENT_TYPE}_${MODEL}_${DATASET}.json
        ;;
    "eval")
        make run_eval config_path=${CONFIG_DIR}/${EXPERIMENT_TYPE}_${MODEL}_${DATASET}.json
        ;;
    "process")
        make run_process config_path=${CONFIG_DIR}/${EXPERIMENT_TYPE}_${MODEL}_${DATASET}.json
        ;;
    *)
        echo "Invalid run command. Available commands: jailbreak, detector, align, eval, process"
        exit 1
        ;;
esac 