"""
Data loader for restaurant review summarization experiment.

This script handles:
- Loading the review data from JSON file
- Selecting samples for human evaluation
- Basic text preprocessing

Input: JSON file containing restaurant reviews and their gold summaries
Output: Processed data ready for model input
"""

import json
import random
import nltk
from typing import Dict, List, Any

class DataLoader:
    def __init__(self, json_path: str):
        """
        Initialize the data loader.
        
        Args:
            json_path (str): Path to the JSON file containing review data
        """
        self.json_path = json_path
        self.data = None
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load and preprocess the review data.
        
        Returns:
            List[Dict[str, Any]]: List of processed review data
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data
    
    def select_samples_for_human_eval(self, num_samples: int = 3) -> List[Dict[str, Any]]:
        """
        Select samples for human evaluation.
        
        Args:
            num_samples (int): Number of samples to select
            
        Returns:
            List[Dict[str, Any]]: Selected samples for human evaluation
        """
        if not self.data:
            self.load_data()
            
        # Select samples based on review length and content diversity
        selected_samples = []
        for review in self.data:
            # Add selection criteria here
            if len(review['original_text'].split()) > 100:  # Example criterion
                selected_samples.append(review)
                
        return random.sample(selected_samples, min(num_samples, len(selected_samples)))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text with the following steps:
        1. Merge line breaks into spaces
        2. Split into sentences
        3. Remove extra spaces
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        text = text.replace('\n', ' ')
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        sentences = nltk.sent_tokenize(text)
        
        sentences = [' '.join(s.split()) for s in sentences]
        
        return ' '.join(sentences) 