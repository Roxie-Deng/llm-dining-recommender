"""
Analysis script for evaluating model and prompt strategy performance.

Input:
    - automatic_metrics.json: Contains evaluation metrics for different models and prompt strategies
    - Format: {
        "sample_id": {
            "model_name": {
                "prompt_strategy": {
                    "rouge1": float,
                    "rougeL": float,
                    "length_ratio": float,
                    "repetition_rate": float,
                    "sentiment": {...},
                    "sentiment_consistency": {...}
                }
            }
        }
    }

Output:
    - analysis_results.json: Comprehensive analysis of model and prompt performance
    - Format: {
        "statistical_analysis": {...},
        "overall_summary": {...},
        "sample_analysis": {...}
    }
    - visualizations/: Directory containing various performance plots

This script analyzes the performance of different models (T5, BART) and prompt strategies
(basic, zero-shot, few-shot, role few-shot, role zero-shot, cot, role cot) across multiple
samples. It provides statistical analysis, identifies best combinations, and generates
visualizations for better understanding of the results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

class ModelAnalyzer:
    def __init__(self, metrics_file: str):
        """Initialize the analyzer with metrics data."""
        self.metrics_file = metrics_file
        self.metrics_data = self._load_metrics()
        self.models = ["t5", "bart"]
        self.prompts = [
            "basic", "zero_shot", "few_shot", "role_few_shot",
            "role_zero_shot", "cot", "role_cot"
        ]
        self.samples = list(self.metrics_data.keys())
        
    def _load_metrics(self) -> Dict:
        """Load metrics data from JSON file."""
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_sentiment_consistency_summary(self, metrics: Dict) -> float:
        """Calculate a summary score for sentiment consistency.
        
        The summary score is a weighted combination of:
        - polarity_consistency (weight: 0.4): Most important as it indicates if the overall sentiment direction is maintained
        - compound_score_diff (weight: 0.3): Second most important as it measures the intensity difference
        - kl_divergence (weight: 0.3): Least important as it's more sensitive to distribution changes
        
        Returns a score between 0 and 1, where 1 indicates perfect consistency.
        """
        sentiment_consistency = metrics["sentiment_consistency"]
        
        # Normalize kl_divergence (lower is better)
        kl_score = 1 / (1 + sentiment_consistency["kl_divergence"])
        
        # Normalize compound_score_diff (lower is better)
        compound_score = 1 - min(1, abs(sentiment_consistency["compound_score_diff"]))
        
        # polarity_consistency is already between 0 and 1
        polarity_score = sentiment_consistency["polarity_consistency"]
        
        # Calculate weighted sum
        summary_score = (
            0.4 * polarity_score +
            0.3 * compound_score +
            0.3 * kl_score
        )
        
        return summary_score
    
    def calculate_statistical_analysis(self) -> Dict:
        """Calculate statistical analysis of model and prompt performance."""
        stats = {
            "model_consistency": {},
            "prompt_consistency": {},
            "sentiment_consistency": {
                "model_consistency": {},
                "prompt_consistency": {},
                "best_combinations": {},
                "summary_scores": {
                    "model": {},
                    "prompt": {},
                    "best_combinations": []
                }
            }
        }
        
        # Model consistency analysis
        for model in self.models:
            model_metrics = {
                "rouge1": [],
                "rougeL": [],
                "length_ratio": [],
                "repetition_rate": []
            }
            
            # Add sentiment consistency metrics
            sentiment_metrics = {
                "kl_divergence": [],
                "compound_score_diff": [],
                "polarity_consistency": []
            }
            
            for sample in self.samples:
                for prompt in self.prompts:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        model_metrics["rouge1"].append(metrics["rouge1"])
                        model_metrics["rougeL"].append(metrics["rougeL"])
                        model_metrics["length_ratio"].append(metrics["length_ratio"])
                        model_metrics["repetition_rate"].append(metrics["repetition_rate"])
                        
                        # Add sentiment consistency metrics
                        sentiment_consistency = metrics["sentiment_consistency"]
                        sentiment_metrics["kl_divergence"].append(sentiment_consistency["kl_divergence"])
                        sentiment_metrics["compound_score_diff"].append(sentiment_consistency["compound_score_diff"])
                        sentiment_metrics["polarity_consistency"].append(sentiment_consistency["polarity_consistency"])
            
            stats["model_consistency"][model] = {
                metric: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric, values in model_metrics.items()
            }
            
            # Add sentiment consistency stats for model
            stats["sentiment_consistency"]["model_consistency"][model] = {
                metric: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric, values in sentiment_metrics.items()
            }
        
        # Prompt consistency analysis
        for prompt in self.prompts:
            prompt_metrics = {
                "rouge1": [],
                "rougeL": [],
                "length_ratio": [],
                "repetition_rate": []
            }
            
            # Add sentiment consistency metrics
            sentiment_metrics = {
                "kl_divergence": [],
                "compound_score_diff": [],
                "polarity_consistency": []
            }
            
            for sample in self.samples:
                for model in self.models:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        prompt_metrics["rouge1"].append(metrics["rouge1"])
                        prompt_metrics["rougeL"].append(metrics["rougeL"])
                        prompt_metrics["length_ratio"].append(metrics["length_ratio"])
                        prompt_metrics["repetition_rate"].append(metrics["repetition_rate"])
                        
                        # Add sentiment consistency metrics
                        sentiment_consistency = metrics["sentiment_consistency"]
                        sentiment_metrics["kl_divergence"].append(sentiment_consistency["kl_divergence"])
                        sentiment_metrics["compound_score_diff"].append(sentiment_consistency["compound_score_diff"])
                        sentiment_metrics["polarity_consistency"].append(sentiment_consistency["polarity_consistency"])
            
            stats["prompt_consistency"][prompt] = {
                metric: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric, values in prompt_metrics.items()
            }
            
            # Add sentiment consistency stats for prompt
            stats["sentiment_consistency"]["prompt_consistency"][prompt] = {
                metric: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric, values in sentiment_metrics.items()
            }
        
        # Calculate summary scores for models
        model_summary_scores = {model: [] for model in self.models}
        for model in self.models:
            for prompt in self.prompts:
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        summary_score = self.calculate_sentiment_consistency_summary(metrics)
                        model_summary_scores[model].append(summary_score)
        
        stats["sentiment_consistency"]["summary_scores"]["model"] = {
            model: {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
            for model, scores in model_summary_scores.items()
        }
        
        # Calculate summary scores for prompts
        prompt_summary_scores = {prompt: [] for prompt in self.prompts}
        for prompt in self.prompts:
            for model in self.models:
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        summary_score = self.calculate_sentiment_consistency_summary(metrics)
                        prompt_summary_scores[prompt].append(summary_score)
        
        stats["sentiment_consistency"]["summary_scores"]["prompt"] = {
            prompt: {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
            for prompt, scores in prompt_summary_scores.items()
        }
        
        # Calculate summary scores for best combinations
        best_combinations = self.find_best_combinations()
        best_summary_scores = []
        for sample, combo in best_combinations["samples"].items():
            summary_score = self.calculate_sentiment_consistency_summary(combo["metrics"])
            best_summary_scores.append({
                "sample": sample,
                "model": combo["model"],
                "prompt": combo["prompt"],
                "score": summary_score
            })
        
        stats["sentiment_consistency"]["summary_scores"]["best_combinations"] = best_summary_scores
        
        return stats
    
    def find_best_combinations(self) -> Dict:
        """Find best model-prompt combinations for each sample and overall."""
        best_combinations = {
            "overall": self._find_overall_best(),
            "samples": {}
        }
        
        for sample in self.samples:
            best_combinations["samples"][sample] = self._find_sample_best(sample)
        
        return best_combinations
    
    def _find_overall_best(self) -> Dict:
        """Find the best overall model-prompt combination."""
        best_score = -1
        best_combination = None
        
        for model in self.models:
            for prompt in self.prompts:
                scores = []
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        # Combined score: 0.4*rouge1 + 0.3*rougeL + 0.2*(1-repetition_rate) + 0.1*(1-abs(1-length_ratio))
                        score = (
                            0.4 * metrics["rouge1"] +
                            0.3 * metrics["rougeL"] +
                            0.2 * (1 - metrics["repetition_rate"]) +
                            0.1 * (1 - abs(1 - metrics["length_ratio"]))
                        )
                        scores.append(score)
                
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_combination = {
                            "model": model,
                            "prompt": prompt,
                            "score": best_score
                        }
        
        return best_combination
    
    def _find_sample_best(self, sample: str) -> Dict:
        """Find the best model-prompt combination for a specific sample."""
        best_score = -1
        best_combination = None
        
        for model in self.models:
            for prompt in self.prompts:
                if prompt in self.metrics_data[sample][model]:
                    metrics = self.metrics_data[sample][model][prompt]
                    score = (
                        0.4 * metrics["rouge1"] +
                        0.3 * metrics["rougeL"] +
                        0.2 * (1 - metrics["repetition_rate"]) +
                        0.1 * (1 - abs(1 - metrics["length_ratio"]))
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_combination = {
                            "model": model,
                            "prompt": prompt,
                            "score": best_score,
                            "metrics": metrics
                        }
        
        return best_combination
    
    def generate_visualizations(self, output_dir: str):
        """Generate visualizations for model and prompt performance."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Model comparison plot
        self._plot_model_comparison(output_path / "model_comparison.png")
        
        # Prompt strategy comparison plot
        self._plot_prompt_comparison(output_path / "prompt_comparison.png")
        
        # Sentiment analysis plot
        self._plot_sentiment_analysis(output_path / "sentiment_analysis.png")
        
        # Sentiment consistency plot
        self._plot_sentiment_consistency(output_path / "sentiment_consistency.png")
    
    def _plot_model_comparison(self, output_file: Path):
        """Generate model comparison visualization."""
        plt.figure(figsize=(12, 6))
        
        data = []
        for model in self.models:
            for prompt in self.prompts:
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        data.append({
                            "model": model,
                            "prompt": prompt,
                            "rouge1": metrics["rouge1"],
                            "rougeL": metrics["rougeL"],
                            "length_ratio": metrics["length_ratio"],
                            "repetition_rate": metrics["repetition_rate"]
                        })
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df, x="model", y="rouge1")
        plt.title("Model Comparison - ROUGE-1 Scores")
        plt.savefig(output_file)
        plt.close()
    
    def _plot_prompt_comparison(self, output_file: Path):
        """Generate prompt strategy comparison visualization."""
        plt.figure(figsize=(12, 6))
        
        data = []
        for model in self.models:
            for prompt in self.prompts:
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        data.append({
                            "model": model,
                            "prompt": prompt,
                            "rouge1": metrics["rouge1"],
                            "rougeL": metrics["rougeL"],
                            "length_ratio": metrics["length_ratio"],
                            "repetition_rate": metrics["repetition_rate"]
                        })
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df, x="prompt", y="rouge1", hue="model")
        plt.title("Prompt Strategy Comparison - ROUGE-1 Scores")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def _plot_sentiment_analysis(self, output_file: Path):
        """Generate sentiment analysis visualization."""
        plt.figure(figsize=(12, 6))
        
        data = []
        for model in self.models:
            for prompt in self.prompts:
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        sentiment = metrics["sentiment"]
                        data.append({
                            "model": model,
                            "prompt": prompt,
                            "positive": sentiment["positive_ratio"],
                            "negative": sentiment["negative_ratio"],
                            "neutral": sentiment["neutral_ratio"]
                        })
        
        df = pd.DataFrame(data)
        df_melted = pd.melt(df, id_vars=["model", "prompt"], 
                           value_vars=["positive", "negative", "neutral"],
                           var_name="sentiment", value_name="ratio")
        
        sns.boxplot(data=df_melted, x="model", y="ratio", hue="sentiment")
        plt.title("Sentiment Distribution by Model")
        plt.savefig(output_file)
        plt.close()
    
    def _plot_sentiment_consistency(self, output_file: Path):
        """Generate sentiment consistency visualization."""
        # Create a figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        data = []
        for model in self.models:
            for prompt in self.prompts:
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        sentiment_consistency = metrics["sentiment_consistency"]
                        data.append({
                            "model": model,
                            "prompt": prompt,
                            "kl_divergence": sentiment_consistency["kl_divergence"],
                            "compound_score_diff": sentiment_consistency["compound_score_diff"],
                            "polarity_consistency": sentiment_consistency["polarity_consistency"]
                        })
        
        df = pd.DataFrame(data)
        
        # Plot 1: Model comparison
        df_melted = pd.melt(df, id_vars=["model"], 
                           value_vars=["kl_divergence", "compound_score_diff", "polarity_consistency"],
                           var_name="metric", value_name="value")
        sns.boxplot(data=df_melted, x="model", y="value", hue="metric", ax=ax1)
        ax1.set_title("Sentiment Consistency by Model")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Prompt comparison
        df_melted = pd.melt(df, id_vars=["prompt"], 
                           value_vars=["kl_divergence", "compound_score_diff", "polarity_consistency"],
                           var_name="metric", value_name="value")
        sns.boxplot(data=df_melted, x="prompt", y="value", hue="metric", ax=ax2)
        ax2.set_title("Sentiment Consistency by Prompt")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Best combinations
        best_combinations = self.find_best_combinations()
        best_data = []
        for sample, combo in best_combinations["samples"].items():
            metrics = combo["metrics"]
            sentiment_consistency = metrics["sentiment_consistency"]
            best_data.append({
                "sample": sample,
                "model": combo["model"],
                "prompt": combo["prompt"],
                "kl_divergence": sentiment_consistency["kl_divergence"],
                "compound_score_diff": sentiment_consistency["compound_score_diff"],
                "polarity_consistency": sentiment_consistency["polarity_consistency"]
            })
        
        df_best = pd.DataFrame(best_data)
        df_best_melted = pd.melt(df_best, id_vars=["model", "prompt"], 
                                value_vars=["kl_divergence", "compound_score_diff", "polarity_consistency"],
                                var_name="metric", value_name="value")
        sns.boxplot(data=df_best_melted, x="model", y="value", hue="metric", ax=ax3)
        ax3.set_title("Sentiment Consistency of Best Combinations")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Summary scores
        summary_data = []
        for model in self.models:
            for prompt in self.prompts:
                for sample in self.samples:
                    if prompt in self.metrics_data[sample][model]:
                        metrics = self.metrics_data[sample][model][prompt]
                        summary_score = self.calculate_sentiment_consistency_summary(metrics)
                        summary_data.append({
                            "model": model,
                            "prompt": prompt,
                            "summary_score": summary_score
                        })
        
        df_summary = pd.DataFrame(summary_data)
        sns.boxplot(data=df_summary, x="model", y="summary_score", ax=ax4)
        ax4.set_title("Sentiment Consistency Summary Scores")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def run_analysis(self, output_dir: str):
        """Run complete analysis and save results."""
        # Calculate statistics
        stats = self.calculate_statistical_analysis()
        
        # Find best combinations
        best_combinations = self.find_best_combinations()
        
        # Generate visualizations
        self.generate_visualizations(output_dir)
        
        # Save analysis results
        results = {
            "statistical_analysis": stats,
            "best_combinations": best_combinations
        }
        
        with open(Path(output_dir) / "analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

def main():
    """Main function to run the analysis."""
    metrics_file = "results/automatic_metrics.json"
    output_dir = "results/analysis"
    
    analyzer = ModelAnalyzer(metrics_file)
    analyzer.run_analysis(output_dir)

if __name__ == "__main__":
    main() 