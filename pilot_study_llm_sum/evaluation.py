"""
Evaluation module for restaurant review summarization experiment.

This script handles:
- Calculating ROUGE scores
- Calculating BLEU scores
- Analyzing summary length
- Checking for repetitions
- Analyzing sentiment balance

Input: Generated summaries and reference summaries
Output: Evaluation metrics
"""

import evaluate
from nltk.translate.bleu_score import sentence_bleu
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, Any, List
import nltk
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

class Evaluator:
    def __init__(self):
        """Initialize the evaluator with required metrics."""
        self.rouge = evaluate.load('rouge')
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('punkt')
            nltk.download('vader_lexicon')
        
        self.sia = SentimentIntensityAnalyzer()
        
    def calculate_rouge(self, pred: str, gold: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            pred (str): Predicted summary
            gold (str): Reference summary
            
        Returns:
            Dict[str, float]: ROUGE scores
        """
        results = self.rouge.compute(
            predictions=[pred],
            references=[gold]
        )
        return {k: v for k, v in results.items() if k in ['rouge1', 'rougeL']}
    
    def calculate_bleu(self, pred: str, gold: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            pred (str): Predicted summary
            gold (str): Reference summary
            
        Returns:
            float: BLEU score
        """
        pred_tokens = pred.split()
        gold_tokens = gold.split()
        return sentence_bleu([gold_tokens], pred_tokens)
    
    def calculate_length_ratio(self, pred: str, gold: str) -> float:
        """
        Calculate length ratio between prediction and reference.
        
        Args:
            pred (str): Predicted summary
            gold (str): Reference summary
            
        Returns:
            float: Length ratio
        """
        pred_len = len(pred.split())
        gold_len = len(gold.split())
        return pred_len / gold_len if gold_len > 0 else 0
    
    def calculate_repetition_rate(self, text: str, n: int = 2) -> float:
        """
        Calculate n-gram repetition rate.
        
        Args:
            text (str): Input text
            n (int): n-gram size
            
        Returns:
            float: Repetition rate
        """
        words = text.split()
        if len(words) < n:
            return 0.0
            
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return 0.0
            
        ngram_counts = Counter(ngrams)
        repeated = sum(count - 1 for count in ngram_counts.values())
        return repeated / len(ngrams)
    
    def calculate_sentiment_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment metrics for a text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Sentiment metrics
        """
        # VADER sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        return {
            'positive_ratio': vader_scores['pos'],
            'negative_ratio': vader_scores['neg'],
            'neutral_ratio': vader_scores['neu'],
            'compound_score': vader_scores['compound'],
            'textblob_score': textblob_score
        }
        
    def calculate_sentiment_consistency(self, original: str, summary: str) -> Dict[str, float]:
        """
        Calculate sentiment consistency between original text and summary.
        
        Args:
            original (str): Original text
            summary (str): Generated summary
            
        Returns:
            Dict[str, float]: Sentiment consistency metrics
        """
        original_sentiment = self.calculate_sentiment_metrics(original)
        summary_sentiment = self.calculate_sentiment_metrics(summary)
        
        # Calculate KL divergence between sentiment distributions
        def kl_divergence(p, q):
            p = np.array([p['positive_ratio'], p['negative_ratio'], p['neutral_ratio']])
            q = np.array([q['positive_ratio'], q['negative_ratio'], q['neutral_ratio']])
            # Add small epsilon to avoid log(0)
            p = np.clip(p, 1e-10, 1)
            q = np.clip(q, 1e-10, 1)
            return np.sum(p * np.log(p / q))
        
        # Calculate sentiment consistency metrics
        kl_div = kl_divergence(original_sentiment, summary_sentiment)
        compound_diff = abs(original_sentiment['compound_score'] - summary_sentiment['compound_score'])
        polarity_consistency = 1.0 if (original_sentiment['compound_score'] * summary_sentiment['compound_score']) >= 0 else 0.0
        
        return {
            'kl_divergence': kl_div,
            'compound_score_diff': compound_diff,
            'polarity_consistency': polarity_consistency,
            'original_sentiment': original_sentiment,
            'summary_sentiment': summary_sentiment
        }
        
    def evaluate_summary(self, original: str, summary: str, reference: str) -> Dict[str, Any]:
        """
        Evaluate a generated summary against reference summary.
        
        Args:
            original (str): Original review text
            summary (str): Generated summary
            reference (str): Reference summary
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge(summary, reference)
        
        # Calculate BLEU score
        bleu_score = self.calculate_bleu(summary, reference)
        
        # Calculate length ratio
        length_ratio = self.calculate_length_ratio(summary, reference)
        
        # Calculate repetition rate
        repetition_rate = self.calculate_repetition_rate(summary)
        
        # Calculate sentiment metrics
        sentiment_metrics = self.calculate_sentiment_metrics(summary)
        
        # Calculate sentiment consistency
        sentiment_consistency = self.calculate_sentiment_consistency(original, summary)
        
        return {
            'rouge1': rouge_scores['rouge1'],
            'rougeL': rouge_scores['rougeL'],
            'bleu': bleu_score,
            'length_ratio': length_ratio,
            'repetition_rate': repetition_rate,
            'sentiment': sentiment_metrics,
            'sentiment_consistency': sentiment_consistency
        } 