"""
Main script for restaurant review summarization experiment.

This script provides a unified entry point for the experiment with the following steps:
1. Generate summaries using different models and prompts
2. Evaluate the generated summaries
3. Generate human evaluation templates
4. Analyze the evaluation results

Usage:
    python main.py --step [all|generate|evaluate|human_eval|analyze]

Input: Command line arguments specifying which steps to run
Output: Results in the specified output directory
"""

import argparse
import os
from generate_summaries import generate_summaries
from evaluate_summaries import evaluate_summaries
from generate_human_eval import generate_human_eval
from analysis import ModelAnalyzer
from config import RESULTS_DIR

def main():
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(description='Restaurant review summarization experiment')
    parser.add_argument('--step', 
                       choices=['all', 'generate', 'evaluate', 'human_eval', 'analyze'],
                       default='all',
                       help='Which step(s) to run')
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run selected steps
    if args.step in ['all', 'generate']:
        print("Generating summaries...")
        generate_summaries()
    
    if args.step in ['all', 'evaluate']:
        print("Evaluating summaries...")
        evaluate_summaries()
    
    if args.step in ['all', 'human_eval']:
        print("Generating human evaluation templates...")
        generate_human_eval()
    
    if args.step in ['all', 'analyze']:
        print("Analyzing evaluation results...")
        metrics_file = os.path.join(RESULTS_DIR, "automatic_metrics.json")
        output_dir = os.path.join(RESULTS_DIR, "analysis")
        analyzer = ModelAnalyzer(metrics_file)
        analyzer.run_analysis(output_dir)

if __name__ == "__main__":
    main() 