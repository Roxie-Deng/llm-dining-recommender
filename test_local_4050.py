#!/usr/bin/env python3
"""
Test script for local RTX 3060 GPU with 6GB VRAM
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from feature_engineering.summarizer import ReviewSummarizer

def test_3060_gpu():
    """Test RTX 3060 GPU capabilities"""
    
    print("Testing RTX 3060 GPU for local processing...")
    print("=" * 50)
    
    # Check GPU status
    print("GPU Status:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Initial memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Test summarizer initialization
    print("\nTesting summarizer initialization...")
    try:
        summarizer = ReviewSummarizer(
            device="cuda",
            max_length=128,
            min_length=20,
            batch_size=2,  # Increased for 3060
            temperature=0.5
        )
        print("Summarizer initialized successfully!")
        print(f"Batch size: {summarizer.batch_size}")
        print(f"Max length: {summarizer.max_length}")
        print(f"Min length: {summarizer.min_length}")
        
        # Test with sample texts
        test_texts = [
            "This restaurant has amazing food and great service. The atmosphere is cozy and the prices are reasonable.",
            "The food was okay but the service was slow. The prices were a bit high for what you get.",
            "Excellent restaurant with fantastic food quality. The staff is friendly and the ambiance is perfect."
        ]
        
        print(f"\nTesting summarization with {len(test_texts)} sample texts...")
        summaries = summarizer.summarize_batch(test_texts, chunk_size=300)
        
        print("Summarization results:")
        for i, (text, summary) in enumerate(zip(test_texts, summaries)):
            print(f"\nText {i+1}: {text[:50]}...")
            print(f"Summary {i+1}: {summary}")
        
        # Check final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"\nFinal GPU memory: {final_memory:.1f} MB")
        
        print("\nRTX 3060 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_management():
    """Test memory management for 3060"""
    
    print("\nTesting memory management...")
    
    summarizer = ReviewSummarizer(device="cuda")
    
    # Test memory cleanup
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Force cleanup
        summarizer._clean_gpu_memory(force=True)
        
        final_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Memory after cleanup: {final_memory:.1f} MB")
        
        print("Memory management test completed!")

if __name__ == "__main__":
    success = test_3060_gpu()
    if success:
        test_memory_management()
        print("\nAll tests passed! RTX 3060 is ready for local processing.")
        print("\nNext steps:")
        print("1. Run the full pipeline: python -m src.feature_engineering.pipeline_with_review_summary")
        print("2. Monitor GPU memory usage during processing")
        print("3. The pipeline will automatically use chunk size 300 for RTX 3060")
    else:
        print("\nTests failed. Please check your CUDA installation and GPU drivers.") 