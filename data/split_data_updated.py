#!/usr/bin/env python3
"""
Updated data splitting script with support for three strategies:
1. Random split: shuffle and randomly split data
2. Goal-based split: split by goal_index (original method)
3. Category-based split: split by category
"""

import os
import sys
import torch
import numpy as np
import yaml
import json

# Set working directory to project root
os.chdir("/mnt/home/amir/framingdecomp/framingDecomp")
sys.path.append('.')

def load_jsonl(path):
    """Load data from JSONL file"""
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def split_data_random(data, train_ratio=0.8):
    """Split data randomly into train and test sets"""
    data_copy = data.copy()
    np.random.shuffle(data_copy)
    split_pt = int(train_ratio * len(data_copy))
    return data_copy[:split_pt], data_copy[split_pt:]

def mask_by_goal(items, goal_set):
    """Filter items by goal_index"""
    return [e for e in items if e["goal_index"] in goal_set]

def mask_by_category(items, category_set):
    """Filter items by category"""
    return [e for e in items if e.get("category", e.get("Category", "unknown")) in category_set]

def get_output_paths(base_path, split_strategy):
    """Generate output paths based on split strategy"""
    base_dir = '/'.join(base_path.split('/')[:-1])
    filename = base_path.split('/')[-1]
    
    if split_strategy == 'random':
        train_path = f"{base_dir}/train/{filename}"
        test_path = f"{base_dir}/test/{filename}"
    elif split_strategy == 'goal':
        train_path = f"{base_dir}/id/{filename}"
        test_path = f"{base_dir}/ood/{filename}"
    elif split_strategy == 'category':
        train_path = f"{base_dir}/id_category/{filename}"
        test_path = f"{base_dir}/ood_category/{filename}"
    
    return train_path, test_path

def main():
    # Configuration
    SPLIT_STRATEGY = 'goal'  # Change this to 'random', 'goal', or 'category'
    TRAIN_RATIO = 0.8  # 80% for train, 20% for test
    
    print(f"Using split strategy: {SPLIT_STRATEGY}")
    print(f"Train ratio: {TRAIN_RATIO}")
    
    # Load configuration
    with open('configs/decomposer3.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # Load data
    DATA_PATH_varyF = config['data']['input_path_varyFraming']
    DATA_PATH_varyG = config['data']['input_path_varyGoal']
    DATA_PATH_varyF_benign = config['data']['input_path_varyFraming_benign']
    DATA_PATH_varyG_benign = config['data']['input_path_varyGoal_benign']
    
    raw_data_varyF = load_jsonl(DATA_PATH_varyF)
    raw_data_varyG = load_jsonl(DATA_PATH_varyG)
    raw_data_varyF_benign = load_jsonl(DATA_PATH_varyF_benign)
    raw_data_varyG_benign = load_jsonl(DATA_PATH_varyG_benign)
    
    # Mark benign / jailbreak flags
    benign = raw_data_varyF_benign + raw_data_varyG_benign
    jailbrks = raw_data_varyF + raw_data_varyG
    print(f"{len(benign)=}   {len(jailbrks)=}")
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Perform splitting based on strategy
    if SPLIT_STRATEGY == 'random':
        # Random split: shuffle all data and split randomly
        ben_train_varyF, ben_test_varyF = split_data_random(raw_data_varyF_benign, TRAIN_RATIO)
        ben_train_varyG, ben_test_varyG = split_data_random(raw_data_varyG_benign, TRAIN_RATIO)
        jb_train_varyF, jb_test_varyF = split_data_random(raw_data_varyF, TRAIN_RATIO)
        jb_train_varyG, jb_test_varyG = split_data_random(raw_data_varyG, TRAIN_RATIO)
        
    elif SPLIT_STRATEGY == 'goal':
        # Goal-based split: split by goal_index (original method)
        all_goals = sorted({e["goal_index"] for e in benign})
        np.random.shuffle(all_goals)
        split_pt = int(TRAIN_RATIO * len(all_goals))
        ID_GOALS = set(all_goals[:split_pt])
        OOD_GOALS = set(all_goals[split_pt:])
        
        ben_train_varyF = mask_by_goal(raw_data_varyF_benign, ID_GOALS)
        ben_train_varyG = mask_by_goal(raw_data_varyG_benign, ID_GOALS)
        ben_test_varyF = mask_by_goal(raw_data_varyF_benign, OOD_GOALS)
        ben_test_varyG = mask_by_goal(raw_data_varyG_benign, OOD_GOALS)
        jb_train_varyF = mask_by_goal(raw_data_varyF, ID_GOALS)
        jb_train_varyG = mask_by_goal(raw_data_varyG, ID_GOALS)
        jb_test_varyF = mask_by_goal(raw_data_varyF, OOD_GOALS)
        jb_test_varyG = mask_by_goal(raw_data_varyG, OOD_GOALS)
        
    elif SPLIT_STRATEGY == 'category':
        # Category-based split: split by category
        all_categories = sorted({e.get("category", e.get("Category", "unknown")) for e in benign})
        np.random.shuffle(all_categories)
        split_pt = int(TRAIN_RATIO * len(all_categories))
        ID_CATEGORIES = set(all_categories[:split_pt])
        OOD_CATEGORIES = set(all_categories[split_pt:])
        
        ben_train_varyF = mask_by_category(raw_data_varyF_benign, ID_CATEGORIES)
        ben_train_varyG = mask_by_category(raw_data_varyG_benign, ID_CATEGORIES)
        ben_test_varyF = mask_by_category(raw_data_varyF_benign, OOD_CATEGORIES)
        ben_test_varyG = mask_by_category(raw_data_varyG_benign, OOD_CATEGORIES)
        jb_train_varyF = mask_by_category(raw_data_varyF, ID_CATEGORIES)
        jb_train_varyG = mask_by_category(raw_data_varyG, ID_CATEGORIES)
        jb_test_varyF = mask_by_category(raw_data_varyF, OOD_CATEGORIES)
        jb_test_varyG = mask_by_category(raw_data_varyG, OOD_CATEGORIES)
        
    else:
        raise ValueError(f"Unknown split strategy: {SPLIT_STRATEGY}. Use 'random', 'goal', or 'category'")
    
    # Print split statistics
    print(f"\nSplit statistics:")
    print(f"Benign varyF - Train: {len(ben_train_varyF)}, Test: {len(ben_test_varyF)}")
    print(f"Benign varyG - Train: {len(ben_train_varyG)}, Test: {len(ben_test_varyG)}")
    print(f"Jailbreak varyF - Train: {len(jb_train_varyF)}, Test: {len(jb_test_varyF)}")
    print(f"Jailbreak varyG - Train: {len(jb_train_varyG)}, Test: {len(jb_test_varyG)}")
    
    # Generate output paths
    paths_varyF_benign = get_output_paths(DATA_PATH_varyF_benign, SPLIT_STRATEGY)
    paths_varyG_benign = get_output_paths(DATA_PATH_varyG_benign, SPLIT_STRATEGY)
    paths_varyF = get_output_paths(DATA_PATH_varyF, SPLIT_STRATEGY)
    paths_varyG = get_output_paths(DATA_PATH_varyG, SPLIT_STRATEGY)
    
    # Unpack paths
    out_path_varyF_benign_train, out_path_varyF_benign_test = paths_varyF_benign
    out_path_varyG_benign_train, out_path_varyG_benign_test = paths_varyG_benign
    out_path_varyF_train, out_path_varyF_test = paths_varyF
    out_path_varyG_train, out_path_varyG_test = paths_varyG
    
    # Create directories if they don't exist
    for path in [out_path_varyF_benign_train, out_path_varyF_benign_test, 
                 out_path_varyG_benign_train, out_path_varyG_benign_test,
                 out_path_varyF_train, out_path_varyF_test,
                 out_path_varyG_train, out_path_varyG_test]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare data and paths for writing
    train_and_test_data = [ben_train_varyF, ben_train_varyG, 
                           ben_test_varyF, ben_test_varyG,
                           jb_train_varyF, jb_train_varyG, 
                           jb_test_varyF, jb_test_varyG]
    
    out_paths = [out_path_varyF_benign_train, out_path_varyG_benign_train,
                 out_path_varyF_benign_test, out_path_varyG_benign_test,
                 out_path_varyF_train, out_path_varyG_train,
                 out_path_varyF_test, out_path_varyG_test]
    
    # Write data to files
    for idx, out_path in enumerate(out_paths):
        print(f"Writing {len(train_and_test_data[idx])} samples to {out_path}")
        
        # Write header with model and input data info
        with open(out_path, "w") as f:
            f.write(f"# model_name: gpt4.1\n")
            f.write(f"# split_strategy: {SPLIT_STRATEGY}\n")
            f.write(f"# train_ratio: {TRAIN_RATIO}\n")
        
        # Write data
        for e in train_and_test_data[idx]:
            with open(out_path, "a") as fout:
                fout.write(json.dumps(e) + "\n")
    
    print(f"\nData splitting completed using {SPLIT_STRATEGY} strategy!")
    print("\nGenerated files:")
    for path in out_paths:
        print(f"  {path}")

if __name__ == "__main__":
    main() 