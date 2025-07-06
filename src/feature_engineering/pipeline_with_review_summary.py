import pandas as pd
import json
from pathlib import Path
from .summarizer import ReviewSummarizer
from .vectorizer_with_review import create_business_embeddings_with_summary
import yaml
import torch

class FeatureEngineeringPipelineWithReviewSummary:
    def __init__(self, config_path="configs/data_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.profile_path = self.config['feature_engineering']['output']['with_review']['extracted_features_with_text']
        self.summary_profile_path = self.config['feature_engineering']['output']['with_summary']['extracted_features_with_summary']
        self.embedding_path = self.config['feature_engineering']['output']['with_summary']['business_embeddings_with_summary']
        self.metadata_path = self.config['feature_engineering']['output']['with_summary']['business_embeddings_with_summary_metadata']
        self.summarizer_cfg = self.config.get('summarization', {})

    def _check_gpu_status(self):
        """Check and report GPU status"""
        print("Checking GPU status...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name()}")
            print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            
            # Determine optimal chunk size based on GPU
            gpu_name = torch.cuda.get_device_name()
            if "T4" in gpu_name:
                self.chunk_size = 200  # Conservative for T4
                print("Detected T4 GPU - using chunk size: 200")
            elif "V100" in gpu_name:
                self.chunk_size = 500  # More aggressive for V100
                print("Detected V100 GPU - using chunk size: 500")
            elif "A100" in gpu_name:
                self.chunk_size = 1000  # Very aggressive for A100
                print("Detected A100 GPU - using chunk size: 1000")
            else:
                self.chunk_size = 200  # Default conservative
                print("Unknown GPU - using default chunk size: 200")
        else:
            print("CUDA not available, will use CPU")
            self.chunk_size = 100  # Smaller chunks for CPU

    def run(self):
        print("Starting feature engineering pipeline with review summary...")
        print("=" * 60)
        
        # Check GPU status and determine chunk size
        self._check_gpu_status()
        print()
        
        print("Step 1: Load with_review profile...")
        df = pd.read_json(self.profile_path, encoding='utf-8')
        print(f"Loaded {len(df)} businesses.")
        print(f"Total texts to process: {len(df['review_tip'].fillna('').tolist())}")

        print("\nStep 2: Summarize review_tip...")
        # Read parameters from config file with aggressive settings
        summarizer = ReviewSummarizer(
            model_name=self.summarizer_cfg.get('model', 'facebook/bart-large-cnn'),
            device=self.summarizer_cfg.get('device', 'cuda'),  # Use CUDA for GPU acceleration
            prompt=self.summarizer_cfg.get('prompt'),
            max_length=self.summarizer_cfg.get('max_length', 512),  # Aggressive max_length
            min_length=self.summarizer_cfg.get('min_length', 50),  # Aggressive min_length
            num_beams=self.summarizer_cfg.get('num_beams', 4),  # Use beam search for quality
            temperature=self.summarizer_cfg.get('temperature', 0.7),  # Higher temperature for creativity
            batch_size=self.summarizer_cfg.get('batch_size', 1)  # Small batch for memory management
        )
        
        print(f"Summarizer parameters:")
        print(f"  device: {summarizer.device}")
        print(f"  max_length: {summarizer.max_length}")
        print(f"  min_length: {summarizer.min_length}")
        print(f"  num_beams: {summarizer.num_beams}")
        print(f"  temperature: {summarizer.temperature}")
        print(f"  batch_size: {summarizer.batch_size}")
        print(f"  max_retries: {summarizer.max_retries}")
        print(f"  chunk_size: {self.chunk_size}")
        
        # Process summarization with chunking for large datasets
        texts = df['review_tip'].fillna("").tolist()
        print(f"\nStarting summarization of {len(texts)} texts in chunks...")
        
        df['review_tip_summary'] = summarizer.summarize_batch(texts, chunk_size=self.chunk_size)
        print("Summarization complete.")

        print(f"\nStep 3: Save profile with summary to {self.summary_profile_path}")
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(self.summary_profile_path), exist_ok=True)
        df.to_json(self.summary_profile_path, orient='records', force_ascii=False, indent=2)

        print("Step 4: Vectorize all features (including summary)...")
        embeddings, business_ids, feature_names = create_business_embeddings_with_summary(
            df,
            config_path=self.config,
            output_dir=Path(self.embedding_path).parent
        )
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Final GPU status check
        if torch.cuda.is_available():
            print(f"\nFinal GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        print("Pipeline completed successfully!")

def main():
    pipeline = FeatureEngineeringPipelineWithReviewSummary()
    pipeline.run()

if __name__ == '__main__':
    main() 