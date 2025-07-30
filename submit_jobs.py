import os
import subprocess
from datetime import datetime

# Experiment configurations
run_commands = ['jailbreak', 'detector', 'align', 'eval', 'process']

config_type_dict = {
    'jailbreak': 'standard',
    'detector': 'detector',
    'align': 'alignment',
    'eval': 'evaluation',
    'process': 'processing'
}

job_name_base_dict = {
    'jailbreak': 'jb',
    'detector': 'det',
    'align': 'align',
    'eval': 'eval',
    'process': 'proc'
}

# Model and dataset configurations
models = ['gpt2', 'gpt2-medium', 'gpt2-large']
datasets = ['harmful_qa', 'advbench', 'custom']

workdir = '/hpc/group/sapirolab/af304/jailbreak/'

# Get the current date and time for the filename
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
output_filename = f"job_ids/job_ids_{timestamp}.txt"

# Create job_ids directory if it doesn't exist
os.makedirs("job_ids", exist_ok=True)

# Open the file to save job IDs
with open(output_filename, 'w') as file:
    for run_command in run_commands:
        config_type = config_type_dict[run_command]
        job_name_base = job_name_base_dict[run_command]
        
        for model in models:
            for dataset in datasets:
                job_name = f"jailbreak_{model}_{job_name_base}_{dataset}"
                current_command = f"sbatch -J {job_name} {workdir}/run_script.sh {run_command} {config_type} {model} {dataset}"
                
                # Run the command and capture the output
                result = subprocess.run(current_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout.strip()
                
                # Extract the job ID from the output
                if result.returncode == 0:
                    job_id = output.split()[-1]
                    print(f"Submitted job {job_name} with ID {job_id}")
                    # Write the job ID to the file
                    file.write(f"{job_name}: {job_id}\n")
                else:
                    print(f"Failed to submit job {job_name}. Error: {result.stderr}")

print(f"Job IDs saved to {output_filename}") 