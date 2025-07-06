#!/usr/bin/env python3
"""
Test script to verify GPU usage in Colab
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from feature_engineering.summarizer import ReviewSummarizer

def test_gpu_usage():
    """Test GPU usage and summarizer with GPU"""
    
    print("🔍 Checking GPU availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("❌ CUDA not available, will use CPU")
    
    print("\n🧪 Testing summarizer with GPU...")
    
    # Test with GPU parameters
    summarizer = ReviewSummarizer(
        device="cuda",        # Force GPU usage
        max_length=64,        # Short length
        min_length=10,        # Short length
        num_beams=1,          # No beam search
        temperature=0.3,      # Low temperature
        batch_size=1          # Small batch
    )
    
    print(f"Summarizer device: {summarizer.device}")
    print(f"Summarizer device type: {type(summarizer.device)}")
    
    # Test with simple texts
    test_texts = [
        "This restaurant has excellent food and great service. The atmosphere is cozy and prices are reasonable.",
        "The food was good but service was slow. The place was crowded and noisy.",
        "Amazing experience! Delicious food, friendly staff, and beautiful decor."
    ]
    
    print(f"\n📝 Testing with {len(test_texts)} texts...")
    
    try:
        summaries = summarizer.summarize_batch(test_texts)
        print("✅ GPU summarization completed successfully!")
        print(f"Generated {len(summaries)} summaries")
        
        for i, (text, summary) in enumerate(zip(test_texts, summaries)):
            print(f"\nText {i+1}: {text[:50]}...")
            print(f"Summary {i+1}: {summary}")
            
        if torch.cuda.is_available():
            print(f"\n💾 Final GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
    except Exception as e:
        print(f"❌ Error during GPU summarization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_gpu_usage()
    if success:
        print("\n🎉 GPU test passed! You can now run the full pipeline with GPU acceleration.")
    else:
        print("\n💥 GPU test failed. Please check the error messages.") 