"""
Module for evaluating generated summaries.

This script handles:
1. Loading generated summaries
2. Evaluating summaries using various metrics
3. Saving evaluation results

Input: Generated summaries (JSON)
Output: Evaluation metrics saved in JSON format
"""

import json
import os
from typing import Dict, Any
from evaluation import Evaluator
from config import RESULTS_DIR

def evaluate_summaries():
    """Evaluate all generated summaries using various metrics."""
    # 1. Load summaries
    with open(os.path.join(RESULTS_DIR, 'summaries.json'), 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    # 2. Initialize evaluator
    evaluator = Evaluator()
    
    # 3. Evaluate summaries
    metrics = {}
    
    for review_id, review_data in summaries.items():
        metrics[review_id] = {}
        
        # Evaluate each model
        for model_name in ['t5', 'bart']:
            if model_name not in review_data:
                continue
                
            metrics[review_id][model_name] = {}
            
            # Evaluate each prompt type
            for prompt_name, summary in review_data[model_name].items():
                metric = evaluator.evaluate_summary(
                    original=review_data['original_text'],
                    summary=summary,
                    reference=review_data['gold_summary']
                )
                metrics[review_id][model_name][prompt_name] = metric
    
    # 4. Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'automatic_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to {os.path.join(RESULTS_DIR, 'automatic_metrics.json')}")

if __name__ == "__main__":
    evaluate_summaries() 