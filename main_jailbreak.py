#!/usr/bin/env python3
import argparse
import os
import yaml
import logging
import datetime
import uuid
import pickle
from pathlib import Path

from utils.data_utils import load_dataset
from utils.model_utils import load_model
from experiments.jailbreak_experiment import JailbreakExperiment

def setup_logging(log_dir: str, unique_id: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment_{unique_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config_with_id(config: dict, output_dir: str, unique_id: str):
    """Save configuration with unique ID"""
    config_path = os.path.join(output_dir, f"config_{unique_id}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Run LLM jailbreaking experiments')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment configuration file')
    args = parser.parse_args()

    # Generate unique ID for this run
    unique_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4()}"
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory with unique ID
    output_dir = os.path.join(config['experiment']['output_dir'], unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config with unique ID
    config_path = save_config_with_id(config, output_dir, unique_id)
    
    # Setup logging
    logger = setup_logging(os.path.join(output_dir, "logs"), unique_id)
    logger.info(f"Starting experiment with unique ID: {unique_id}")
    logger.info(f"Loaded configuration from {args.config}")

    
    dataset = load_dataset(config['dataset']['name'])
    logger.info(f"Loaded dataset: {config['dataset']['name']}")

    # Load model
    model = load_model(config['model']['name'])
    logger.info(f"Loaded model: {config['model']['name']}")

    # Initialize and run experiment
    experiment = JailbreakExperiment(
        model=model,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        unique_id=unique_id
    )
    
    results = experiment.run()
    logger.info("Experiment completed successfully")

    # Save results with unique ID
    results_path = os.path.join(output_dir, f'results_{unique_id}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {results_path}")



if __name__ == "__main__":
    main() 