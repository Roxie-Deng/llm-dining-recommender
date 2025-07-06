"""
Module for generating summaries using different models and prompts.

This script handles:
1. Loading the review data
2. Initializing models
3. Generating summaries for each review using different models and prompts
4. Saving the generated summaries

Input: Restaurant review data (JSON)
Output: Generated summaries saved in JSON format
"""

import json
import os
from typing import Dict, Any
from data_loader import DataLoader
from models import SummaryModel
from config import MODEL_NAMES, MODEL_PARAMS, PROMPT_TEMPLATES, DATA_PATH, RESULTS_DIR

def generate_summaries():
    """Generate summaries for all reviews using different models and prompts."""
    # 1. Load data
    data_loader = DataLoader(DATA_PATH)
    data = data_loader.load_data()
    
    # 2. Initialize models
    models = {
        't5': SummaryModel(MODEL_NAMES['t5']),
        'bart': SummaryModel(MODEL_NAMES['bart'])
    }
    
    # 3. Generate summaries
    summaries = {}
    
    for review in data:
        review_id = review['business_id']
        summaries[review_id] = {
            'original_text': review['original_text'],
            'gold_summary': review['gold_summary']
        }
        
        # Generate summaries for each model and prompt
        for model_name, model in models.items():
            for prompt_name, prompt in PROMPT_TEMPLATES.items():
                summary = model.generate_summary(
                    review['original_text'],
                    prompt,
                    MODEL_PARAMS
                )
                
                # Store summary
                if model_name not in summaries[review_id]:
                    summaries[review_id][model_name] = {}
                summaries[review_id][model_name][prompt_name] = summary
    
    # 4. Save summaries
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'summaries.json'), 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)

if __name__ == "__main__":
    generate_summaries() 