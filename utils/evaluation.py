from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import json

logger = logging.getLogger(__name__)

def calculate_metrics(
    predictions: List[float],
    targets: List[float],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        predictions: List of prediction scores
        targets: List of target labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary predictions
    binary_preds = [1 if p >= threshold else 0 for p in predictions]
    
    metrics = {
        "accuracy": accuracy_score(targets, binary_preds),
        "precision": precision_score(targets, binary_preds),
        "recall": recall_score(targets, binary_preds),
        "f1": f1_score(targets, binary_preds)
    }
    
    return metrics

def evaluate_attack_success(
    results: List[Dict[str, Union[str, float]]]
) -> Dict[str, float]:
    """
    Evaluate attack success metrics
    
    Args:
        results: List of attack results
        
    Returns:
        Dictionary of evaluation metrics
    """
    success_rates = []
    response_times = []
    
    for result in results:
        success_rates.append(result["success"])
        if "response_time" in result:
            response_times.append(result["response_time"])
    
    metrics = {
        "success_rate": np.mean(success_rates),
        "success_rate_std": np.std(success_rates),
        "avg_response_time": np.mean(response_times) if response_times else 0.0
    }
    
    return metrics

def analyze_attack_patterns(
    results: List[Dict[str, Union[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze patterns in successful attacks
    
    Args:
        results: List of attack results
        
    Returns:
        Dictionary of pattern analysis results
    """
    pattern_analysis = {}
    
    # Group results by attack type
    for result in results:
        attack_type = result["attack_type"]
        if attack_type not in pattern_analysis:
            pattern_analysis[attack_type] = {
                "count": 0,
                "success_count": 0,
                "success_rate": 0.0
            }
        
        pattern_analysis[attack_type]["count"] += 1
        if result["success"] > 0:
            pattern_analysis[attack_type]["success_count"] += 1
    
    # Calculate success rates
    for attack_type in pattern_analysis:
        pattern_analysis[attack_type]["success_rate"] = (
            pattern_analysis[attack_type]["success_count"] /
            pattern_analysis[attack_type]["count"]
        )
    
    return pattern_analysis

def generate_evaluation_report(
    results: List[Dict[str, Union[str, float]]],
    output_path: str
) -> None:
    """
    Generate comprehensive evaluation report
    
    Args:
        results: List of attack results
        output_path: Path to save the report
    """
    # Calculate overall metrics
    metrics = evaluate_attack_success(results)
    
    # Analyze attack patterns
    patterns = analyze_attack_patterns(results)
    
    # Generate report
    report = {
        "overall_metrics": metrics,
        "attack_patterns": patterns,
        "total_attempts": len(results),
        "successful_attempts": sum(1 for r in results if r["success"] > 0)
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_path}")
    
    return report 