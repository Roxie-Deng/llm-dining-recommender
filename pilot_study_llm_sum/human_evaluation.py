"""
Human evaluation module for restaurant review summarization experiment.

This script handles:
- Generating evaluation templates for human assessment
- Creating comparison tables
- Formatting output for human evaluation

Input: Model outputs and reference summaries
Output: Formatted evaluation templates
"""

from typing import Dict, Any, List
import os

class HumanEvaluation:
    def __init__(self, output_dir: str = 'results/human_evaluation'):
        """
        Initialize the human evaluation module.
        
        Args:
            output_dir (str): Directory to save evaluation files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def format_comparison(self, review_id: str, original: str, gold: str, 
                         model_outputs: Dict[str, Dict[str, str]]) -> str:
        """
        Format comparison for human evaluation.
        
        Args:
            review_id (str): Review identifier
            original (str): Original review text
            gold (str): Reference summary
            model_outputs (Dict[str, Dict[str, str]]): Model outputs for different prompts
            
        Returns:
            str: Formatted comparison text
        """
        template = f"""# Review {review_id}

## Original Review
{original}

## Reference Summary
{gold}

## Model Outputs

### 1. Basic Prompt
**T5 Output:**
{model_outputs.get('basic', {}).get('t5', '[No Data]')}

**BART Output:**
{model_outputs.get('basic', {}).get('bart', '[No Data]')}

### 2. Zero-shot Prompt
**T5 Output:**
{model_outputs.get('zero_shot', {}).get('t5', '[No Data]')}

**BART Output:**
{model_outputs.get('zero_shot', {}).get('bart', '[No Data]')}

### 3. Few-shot Prompt
**T5 Output:**
{model_outputs.get('few_shot', {}).get('t5', '[No Data]')}

**BART Output:**
{model_outputs.get('few_shot', {}).get('bart', '[No Data]')}

### 4. Chain-of-Thought Prompt
**T5 Output:**
{model_outputs.get('cot', {}).get('t5', '[No Data]')}

**BART Output:**
{model_outputs.get('cot', {}).get('bart', '[No Data]')}

### 5. Role-based CoT Prompt
**T5 Output:**
{model_outputs.get('role_cot', {}).get('t5', '[No Data]')}

**BART Output:**
{model_outputs.get('role_cot', {}).get('bart', '[No Data]')}

## Evaluation Table

| Model/Prompt | Completeness | Accuracy | Balance | Fluency | Conciseness | Overall | Notes |
|--------------|--------------|----------|---------|---------|-------------|---------|-------|
| T5-Basic     |              |          |         |         |             |         |       |
| BART-Basic   |              |          |         |         |             |         |       |
| T5-Zero      |              |          |         |         |             |         |       |
| BART-Zero    |              |          |         |         |             |         |       |
| T5-Few       |              |          |         |         |             |         |       |
| BART-Few     |              |          |         |         |             |         |       |
| T5-CoT       |              |          |         |         |             |         |       |
| BART-CoT     |              |          |         |         |             |         |       |
| T5-Role      |              |          |         |         |             |         |       |
| BART-Role    |              |          |         |         |             |         |       |

## Evaluation Guidelines

1. Completeness (1-5):
   - 5: Contains all important information
   - 3: Contains most important information
   - 1: Missing important information

2. Accuracy (1-5):
   - 5: Completely accurate
   - 3: Mostly accurate with minor errors
   - 1: Contains significant errors

3. Balance (1-5):
   - 5: Perfectly balanced
   - 3: Somewhat balanced
   - 1: Heavily biased

4. Fluency (1-5):
   - 5: Very fluent
   - 3: Somewhat fluent
   - 1: Not fluent

5. Conciseness (1-5):
   - 5: Very concise
   - 3: Somewhat concise
   - 1: Too verbose
"""
        return template
    
    def generate_evaluation_files(self, samples: List[Dict[str, Any]], 
                                model_outputs: Dict[str, Dict[str, Dict[str, str]]]):
        """
        Generate evaluation files for human assessment.
        
        Args:
            samples (List[Dict[str, Any]]): Selected samples for evaluation
            model_outputs (Dict[str, Dict[str, Dict[str, str]]]): Model outputs for all samples
        """
        for sample in samples:
            review_id = sample['business_id']
            output = self.format_comparison(
                review_id,
                sample['original_text'],
                sample['gold_summary'],
                model_outputs[review_id]
            )
            
            # Save to file
            output_path = os.path.join(self.output_dir, f'review_{review_id}.md')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output) 