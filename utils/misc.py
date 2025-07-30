import os
import json
import time
import random
import numpy as np
import torch
import yaml
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path


import sys
import io
import logging
import scipy.stats
import numpy as np
import torch

import numpy as np
import os
import random



class EmptyPI(Exception):
    """Exception raised when a persistence interval is empty."""

    def __init__(self, message="Persistence interval is empty"):
        self.message = message
        super().__init__(self.message)


def set_seed(seed=42):
    """
    Seed everything

    Parameters:
    - seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)  # Python random module.
    os.environ["PYTHONHASHSEED"] = str(seed)  # Environment variable that controls the hashing algorithm.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)  # PyTorch for CPU operations.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch for CUDA operations.
        torch.cuda.manual_seed_all(seed)  # PyTorch for all CUDA devices.
    
    # The following two settings are recommended for deterministic behavior during convolutions,
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    

# ==== Cell: [Dataset & dataloader] ====

from collections import defaultdict
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

class DualPairDataset(Dataset):
    """
    Returns (sample_a, sample_b, pair_type)
      pair_type = 0 → same-goal / diff-frame  (from varyF)
      pair_type = 1 → same-frame / diff-goal  (from varyG)
    """
    def __init__(self, samples, stratified_capping=True):
        self.samples = samples
        self.goal_pairs, self.frame_pairs = [], []

        by_goal_F  = defaultdict(list)
        by_frame_G = defaultdict(list)

        for idx, s in enumerate(samples):
            if s["split"] == "varyF":  by_goal_F [s["goal_index"]   ].append(idx)
            else:                      by_frame_G[s["framing_index"]].append(idx)

        for lst in by_goal_F.values():
            self.goal_pairs  += [(a,b,0) for a in lst for b in lst if a<b]
        for lst in by_frame_G.values():
            self.frame_pairs += [(a,b,1) for a in lst for b in lst if a<b]

        # --- stratified capping ---------------------------------
        # this improved the performance a bit
        if stratified_capping:
            cap = int(np.median([len(v) for v in by_goal_F.values()]))
            for g, lst in by_goal_F.items():
                if len(lst) > cap:               # down-sample heavy goals
                    by_goal_F[g] = random.sample(lst, cap)
        # --------------------------------------------------------------

        self.all_pairs = self.goal_pairs + self.frame_pairs

    def __len__(self): return len(self.all_pairs)
    def __getitem__(self, k): return self.all_pairs[k]

def collate_dual(batch, all_samples) -> Tuple[list,str,str,torch.Tensor]:
    """
    batch → (texts, goal_ids, frame_ids, pair_types)
    """
    texts, gid, fid, ptype = [], [], [], []
    for a,b,t in batch:
        sa, sb = all_samples[a], all_samples[b]
        texts.extend([sa["text"], sb["text"]])
        gid.extend([sa["goal_cid"], sb["goal_cid"]])
        fid.extend([sa["framing_index"], sb["framing_index"]])
        ptype.append(t)
    return (texts,
            torch.tensor(gid),
            torch.tensor(fid),
            torch.tensor(ptype))





def setup_logging(log_dir: str, name: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def timeit(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_device() -> str:
    """Get available device (cuda or cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB" 




def print_example_goal_prompt(entry, width=80):
    def insert_newlines(s):
        out = []
        last = 0
        while last < len(s):
            next_chunk = s[last:last+width]
            # If there's already a newline in the chunk, split at the first one
            nl_pos = next_chunk.find('\n')
            if nl_pos != -1:
                out.append(s[last:last+nl_pos+1])
                last += nl_pos + 1
            else:
                out.append(next_chunk + '\n')
                last += width
        return ''.join(out)

    print(f"Goal:\n{insert_newlines(entry['goal'])}\nPrompt:\n{insert_newlines(entry['prompt'])}")


def reconstruct_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    original_config_path: str,
    current_config: Dict[str, Any],
    dataset_size: int,
    logger: logging.Logger
) -> int:
    """
    Reconstruct optimizer and scheduler state when loading weights-only checkpoints.
    
    This function estimates the training progress and advances the optimizer and scheduler
    to the appropriate step to maintain training continuity.
    
    Args:
        optimizer: The optimizer to reconstruct state for
        scheduler: The learning rate scheduler to advance (can be None)
        original_config_path: Path to the original training config file
        current_config: Current training configuration
        dataset_size: Size of the training dataset
        logger: Logger instance for output messages
        
    Returns:
        int: The number of completed training steps
        
    Raises:
        FileNotFoundError: If original config file doesn't exist
        Exception: If reconstruction fails
    """
    try:
        # Load original config
        with open(original_config_path, "r") as f:
            original_config = yaml.safe_load(f)
        
        # Extract training parameters from original config
        original_epochs = original_config.get("training", {}).get("num_epochs", 
                                                                 current_config["training"]["num_epochs"])
        original_batch_size = original_config.get("training", {}).get("batch_size", 
                                                                     current_config["training"]["batch_size"])
        original_grad_accum = original_config.get("training", {}).get("grad_accum_steps", 
                                                                     current_config["training"]["grad_accum_steps"])
        
        # Calculate completed training steps
        steps_per_epoch = dataset_size // (original_batch_size * original_grad_accum)
        completed_steps = steps_per_epoch * original_epochs
        
        logger.info("Reconstructing optimizer state from original config: %s", original_config_path)
        logger.info("Estimated completed steps: %d (epochs: %d, steps/epoch: %d)", 
                   completed_steps, original_epochs, steps_per_epoch)
        
        # Reconstruct optimizer state by setting step count
        # Note: This is a simplified approach - for more accurate reconstruction,
        # we would need to replay actual gradients
        for param_group in optimizer.param_groups:
            if 'step' in param_group:
                param_group['step'] = completed_steps
        
        logger.info("Optimizer state reconstructed with step count: %d", completed_steps)
        
        # Advance scheduler to the same step if provided
        if scheduler is not None:
            for _ in range(completed_steps):
                scheduler.step()
            logger.info("Scheduler advanced to step: %d", completed_steps)
        
        return completed_steps
        
    except FileNotFoundError:
        logger.error("Original config file not found: %s", original_config_path)
        raise
    except Exception as e:
        logger.warning("Failed to reconstruct optimizer state: %s", str(e))
        logger.info("Continuing with fresh optimizer state")
        raise