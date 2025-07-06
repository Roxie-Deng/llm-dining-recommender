"""
Model handler for restaurant review summarization experiment.

This script handles:
- Loading and initializing models
- Generating summaries using different prompts
- Basic model utilities

Input: Text data and prompt templates
Output: Generated summaries
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, Any
import torch

class SummaryModel:
    def __init__(self, model_name: str, device: str = None):
        """Initialize the summary model."""
        import torch
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def generate_summary(self, text: str, prompt_template: str, params: Dict[str, Any]) -> str:
        """
        Generate summary using the model.
        
        Args:
            text (str): Input text to summarize
            prompt_template (str): Prompt template to use
            params (Dict[str, Any]): Generation parameters
            
        Returns:
            str: Generated summary
        """
        if not self.model or not self.tokenizer:
            self.load_model()
            
        # Prepare input
        input_text = prompt_template.replace('[REVIEW_TEXT_HERE]', text)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        # Generate summary
        outputs = self.model.generate(
            **inputs,
            max_length=params['max_length'],
            min_length=params['min_length'],
            num_beams=params['num_beams'],
            temperature=params['temperature']
        )
        
        # Decode and return
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary 