#!/usr/bin/env python3
"""
Test script to run summarization on a small sample in Colab
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import torch
from feature_engineering.summarizer import ReviewSummarizer
import yaml

def test_summarization_sample():
    """Test summarization on a small sample of data"""
    
    print("Testing summarization on small sample...")
    print("=" * 50)
    
    # Check GPU status
    print("GPU Status:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"Initial memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Load config
    with open("configs/data_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    summarizer_cfg = config.get('summarization', {})
    
    # Load sample data (first 5 records)
    print("\nLoading sample data...")
    df = pd.read_json("data/engineered/with_review/extracted_features_with_text.json", encoding='utf-8')
    sample_df = df.head(5).copy()  # Take first 5 records
    print(f"Loaded {len(sample_df)} sample records")
    
    # Initialize summarizer with aggressive parameters
    print("\nInitializing summarizer with aggressive parameters...")
    summarizer = ReviewSummarizer(
        model_name=summarizer_cfg.get('model', 'facebook/bart-large-cnn'),
        device=summarizer_cfg.get('device', 'cuda'),
        prompt=summarizer_cfg.get('prompt'),
        max_length=summarizer_cfg.get('max_length', 512),
        min_length=summarizer_cfg.get('min_length', 50),
        num_beams=summarizer_cfg.get('num_beams', 4),
        temperature=summarizer_cfg.get('temperature', 0.7),
        batch_size=summarizer_cfg.get('batch_size', 1)
    )
    
    print(f"Summarizer parameters:")
    print(f"  max_length: {summarizer.max_length}")
    print(f"  min_length: {summarizer.min_length}")
    print(f"  num_beams: {summarizer.num_beams}")
    print(f"  temperature: {summarizer.temperature}")
    print(f"  batch_size: {summarizer.batch_size}")
    
    # Process sample texts
    texts = sample_df['review_tip'].fillna("").tolist()
    print(f"\nProcessing {len(texts)} sample texts...")
    
    # Generate summaries
    summaries = summarizer.summarize_batch(texts, chunk_size=5)  # Small chunk for testing
    sample_df['review_tip_summary'] = summaries
    
    # Display results
    print("\n" + "=" * 50)
    print("SAMPLE RESULTS:")
    print("=" * 50)
    
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"\n--- Sample {i} ---")
        print(f"Business: {row['name']}")
        print(f"Stars: {row['stars']}")
        print(f"Categories: {row['categories']}")
        
        # Show original text (truncated)
        original_text = row['review_tip'][:200] + "..." if len(row['review_tip']) > 200 else row['review_tip']
        print(f"\nOriginal text (first 200 chars):")
        print(f"{original_text}")
        
        # Show summary
        print(f"\nSummary:")
        print(f"{row['review_tip_summary']}")
        
        # Show summary stats
        summary_length = len(row['review_tip_summary'])
        original_length = len(row['review_tip'])
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        print(f"\nSummary stats:")
        print(f"  Original length: {original_length} chars")
        print(f"  Summary length: {summary_length} chars")
        print(f"  Compression ratio: {compression_ratio:.2%}")
        
        print("-" * 40)
    
    # Final GPU status
    if torch.cuda.is_available():
        print(f"\nFinal GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    print("\nTest completed successfully!")
    print("You can now evaluate the summary quality and decide if to proceed with full dataset.")

if __name__ == "__main__":
    test_summarization_sample() 