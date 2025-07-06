"""
Module for generating human evaluation templates.

This script handles:
1. Loading generated summaries
2. Selecting samples for human evaluation
3. Generating evaluation templates
4. Saving evaluation files

Input: Generated summaries (JSON)
Output: Human evaluation templates in Markdown format
"""

import json
import os
from typing import Dict, Any
from data_loader import DataLoader
from human_evaluation import HumanEvaluation
from config import DATA_PATH, RESULTS_DIR

def generate_human_eval():
    """Generate human evaluation templates for selected samples."""
    # 1. Load summaries
    with open(os.path.join(RESULTS_DIR, 'summaries.json'), 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    # 2. Select samples for human evaluation
    data_loader = DataLoader(DATA_PATH)
    selected_samples = data_loader.select_samples_for_human_eval()
    
    # 3. Generate evaluation templates
    human_eval = HumanEvaluation()
    human_eval.generate_evaluation_files(selected_samples, summaries)

if __name__ == "__main__":
    generate_human_eval() 