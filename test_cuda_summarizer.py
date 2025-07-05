#!/usr/bin/env python3
"""
Test script to verify CUDA summarizer is working correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from feature_engineering.summarizer import ReviewSummarizer

def test_cuda_summarizer():
    """Test the summarizer with CUDA"""
    
    print("Testing CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    print("\nTesting ReviewSummarizer with CUDA...")
    
    # Test with CUDA parameters
    summarizer = ReviewSummarizer(
        device=0,           # Force CUDA
        max_length=256,     # Normal length
        min_length=50,      # Normal length
        num_beams=4,        # Use beam search
        temperature=0.7,    # Normal temperature
        batch_size=8        # Normal batch size
    )
    
    print(f"Parameters:")
    print(f"  device: {summarizer.device}")
    print(f"  max_length: {summarizer.max_length}")
    print(f"  min_length: {summarizer.min_length}")
    print(f"  num_beams: {summarizer.num_beams}")
    print(f"  temperature: {summarizer.temperature}")
    print(f"  batch_size: {summarizer.batch_size}")
    
    # Test with realistic texts
    test_texts = [
        "This restaurant has amazing food! The pasta is cooked perfectly and the sauce is flavorful. The service was excellent - our waiter was attentive and friendly. The prices are reasonable for the quality of food. However, the restaurant was quite noisy and the tables were a bit cramped. Overall, I would definitely recommend this place for a nice dinner.",
        "The food was okay but the service was slow. The atmosphere was nice though. The prices were a bit high for what you get. The staff seemed overwhelmed and we had to wait a long time for our food. The decor is beautiful and the location is convenient.",
        "Great place with excellent food and service! The staff is very friendly and the atmosphere is cozy. Prices are reasonable and the portions are generous. Highly recommend this restaurant for both casual dining and special occasions."
    ]
    
    print(f"\nTesting with {len(test_texts)} realistic texts...")
    
    try:
        summaries = summarizer.summarize_batch(test_texts)
        print("✅ CUDA summarization completed successfully!")
        print(f"Generated {len(summaries)} summaries")
        
        for i, (text, summary) in enumerate(zip(test_texts, summaries)):
            print(f"\nText {i+1}: {text[:100]}...")
            print(f"Summary {i+1}: {summary}")
            
    except Exception as e:
        print(f"❌ Error during CUDA summarization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_cuda_summarizer()
    if success:
        print("\n✅ CUDA summarizer is working correctly!")
        print("You can now run the full pipeline with CUDA.")
    else:
        print("\n❌ CUDA summarizer failed. Please check the error messages.") 