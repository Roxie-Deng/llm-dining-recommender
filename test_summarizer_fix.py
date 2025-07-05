#!/usr/bin/env python3
"""
Test script to verify the summarizer fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering.summarizer import ReviewSummarizer

def test_summarizer_fix():
    """Test the summarizer with fixed parameters"""
    
    print("Testing ReviewSummarizer with fixed parameters...")
    
    # Test with very conservative parameters
    summarizer = ReviewSummarizer(
        device="cpu",        # Use CPU to avoid CUDA issues
        max_length=64,       # Very short
        min_length=10,       # Very short
        num_beams=1,         # No beam search
        temperature=0.3,     # Very low temperature
        batch_size=1         # Single batch
    )
    
    print(f"Parameters:")
    print(f"  device: {summarizer.device}")
    print(f"  max_length: {summarizer.max_length}")
    print(f"  min_length: {summarizer.min_length}")
    print(f"  num_beams: {summarizer.num_beams}")
    print(f"  temperature: {summarizer.temperature}")
    print(f"  batch_size: {summarizer.batch_size}")
    
    # Test with simple, safe texts
    test_texts = [
        "This restaurant has good food and friendly service. The prices are reasonable.",
        "The food was okay but the service was slow. The atmosphere was nice.",
        "",  # Empty text
        "Great place!"  # Very short text
    ]
    
    print(f"\nTesting with {len(test_texts)} texts...")
    
    try:
        summaries = summarizer.summarize_batch(test_texts)
        print("✅ Summarization completed successfully!")
        print(f"Generated {len(summaries)} summaries")
        
        for i, (text, summary) in enumerate(zip(test_texts, summaries)):
            print(f"\nText {i+1}: {text[:50]}...")
            print(f"Summary {i+1}: {summary}")
            
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_summarizer_fix()
    if success:
        print("\n✅ All tests passed! You can now run the full pipeline.")
    else:
        print("\n❌ Tests failed. Please check the error messages.") 