import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from transformers import PreTrainedTokenizer

# --- more imports ---
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from models.encoder       import HFEncoder          # wraps the frozen LLM you loaded
from models.decomposer    import LinearDecomposer   # or NonlinearDecomposer
from train_test.decomposer_training import train_decomposer
from utils.losses        import l2_recon, orth_penalty, info_nce  # optional reuse



class GoalPairDataset(Dataset):
    """
    Returns *pairs of framings* for the same goal so that
    anchor = item[0] , positive = item[1].
    Each __getitem__ gives (dict_anchor , dict_pos).
    """
    def __init__(self, samples: list[dict]):
        buckets = defaultdict(list)
        for s in samples:
            buckets[s['goal_index']].append(s)

        self.pairs = []
        for goal, lst in buckets.items():
            if len(lst) < 2:
                continue
            # simple deterministic pairing: (0,1) (2,3) ...
            lst = sorted(lst, key=lambda x: x['framing_index'])
            for i in range(0, len(lst)-1, 2):
                self.pairs.append((lst[i], lst[i+1]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
    
    
def collate_anchor_pos(batch):
    """
    → texts  (list[str]  length = 2*B)
      gid    (LongTensor)   goal id duplicated for anchor/pos
      glbl   (dummy – keep API the same)
      fid    framing indices
    The *train_decomposer* loop will automatically pick
    anchor  = v[0::2] ,  pos = v[1::2]
    """
    texts, gid, glbl, fid = [], [], [], []
    for anchor, pos in batch:
        for item in (anchor, pos):
            texts.append(item['text'])
            gid.append(item['goal_index'])
            glbl.append(item['goal_index'])          # placeholder (not used)
            fid.append(item['framing_index'])
    return texts, torch.tensor(gid), torch.tensor(glbl), torch.tensor(fid)




def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load dataset from various sources (local file, Hugging Face, etc.)
    
    Args:
        dataset_path: Path to dataset or Hugging Face dataset name
        
    Returns:
        DataFrame containing the dataset
    """
    if os.path.exists(dataset_path):
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif dataset_path.endswith('.csv'):
            return pd.read_csv(dataset_path)
    else:
        # Try loading from Hugging Face
        try:
            dataset = load_dataset(dataset_path)
            return pd.DataFrame(dataset['train'])
        except Exception as e:
            raise ValueError(f"Could not load dataset from {dataset_path}: {str(e)}")

def load_jsonl_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a JSONL file
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        DataFrame containing the dataset
    """
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)

def load_paraphrase_dataset(data_dir: str, model_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load paraphrase dataset from the populated artifacts directory
    
    Args:
        data_dir: Path to the populated artifacts directory
        model_name: Optional model name to filter by (e.g., 'gpt-4-0125-preview')
        
    Returns:
        DataFrame containing the paraphrase dataset
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
        
    # Find all JSONL files
    if model_name:
        files = [data_dir / f"{model_name}.jsonl"]
    else:
        files = list(data_dir.glob("*.jsonl"))
        
    if not files:
        raise ValueError(f"No JSONL files found in {data_dir}")
        
    # Load and combine all files
    dfs = []
    for f in files:
        df = load_jsonl_dataset(str(f))
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def group_by_goal(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group dataset by goal
    
    Args:
        df: DataFrame containing the paraphrase dataset
        
    Returns:
        Dictionary mapping goals to their corresponding DataFrames
    """
    return {goal: group for goal, group in df.groupby('goal')}

def get_goal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each goal
    
    Args:
        df: DataFrame containing the paraphrase dataset
        
    Returns:
        DataFrame with statistics for each goal
    """
    stats = []
    for goal, group in df.groupby('goal'):
        total = len(group)
        successful = group['jailbroken'].sum()
        success_rate = successful / total if total > 0 else 0
        
        stats.append({
            'goal': goal,
            'goal_index': group['goal_index'].iloc[0],
            'total_prompts': total,
            'successful_prompts': successful,
            'success_rate': success_rate,
            'num_framings': group['framing_index'].nunique()
        })
    
    return pd.DataFrame(stats)

def filter_by_success_rate(df: pd.DataFrame, min_rate: float = 0.0, max_rate: float = 1.0) -> pd.DataFrame:
    """
    Filter dataset by success rate of goals
    
    Args:
        df: DataFrame containing the paraphrase dataset
        min_rate: Minimum success rate to include
        max_rate: Maximum success rate to include
        
    Returns:
        Filtered DataFrame
    """
    stats = get_goal_stats(df)
    valid_goals = stats[
        (stats['success_rate'] >= min_rate) & 
        (stats['success_rate'] <= max_rate)
    ]['goal'].tolist()
    
    return df[df['goal'].isin(valid_goals)]

def preprocess_prompts(prompts: List[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Preprocess prompts for model input
    
    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer for the model
        
    Returns:
        Dictionary containing tokenized inputs
    """
    return tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, by_goal: bool = True) -> tuple:
    """
    Split dataset into train and test sets
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio of data to use for training
        by_goal: If True, split by goals to ensure all framings of a goal stay together
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if by_goal:
        # Get unique goals
        goals = df['goal'].unique()
        train_size = int(len(goals) * train_ratio)
        
        # Split goals
        train_goals = goals[:train_size]
        test_goals = goals[train_size:]
        
        # Split dataset
        train_df = df[df['goal'].isin(train_goals)]
        test_df = df[df['goal'].isin(test_goals)]
    else:
        # Simple random split
        train_size = int(len(df) * train_ratio)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
    return train_df, test_df

def save_dataset(df: pd.DataFrame, output_path: str):
    """
    Save dataset to file
    
    Args:
        df: DataFrame to save
        output_path: Path to save the dataset
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith('.jsonl'):
        with open(output_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
    elif output_path.endswith('.json'):
        df.to_json(output_path, orient='records', indent=2)
    elif output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use .jsonl, .json, or .csv")

def get_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each model
    
    Args:
        df: DataFrame containing the paraphrase dataset
        
    Returns:
        DataFrame with statistics for each model
    """
    stats = []
    for model, group in df.groupby('model'):
        total = len(group)
        successful = group['jailbroken'].sum()
        success_rate = successful / total if total > 0 else 0
        
        stats.append({
            'model': model,
            'total_prompts': total,
            'successful_prompts': successful,
            'success_rate': success_rate,
            'num_goals': group['goal'].nunique(),
            'num_framings': group['framing_index'].nunique()
        })
    
    return pd.DataFrame(stats)

def load_jailbreak_prompts(prompts_path: str) -> List[str]:
    """
    Load jailbreaking prompts from file
    
    Args:
        prompts_path: Path to prompts file
        
    Returns:
        List of prompt strings
    """
    with open(prompts_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def save_jailbreak_prompts(prompts: List[str], output_path: str):
    """
    Save jailbreaking prompts to file
    
    Args:
        prompts: List of prompt strings
        output_path: Path to save the prompts
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n") 