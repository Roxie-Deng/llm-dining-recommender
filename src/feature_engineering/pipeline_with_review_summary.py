import pandas as pd
import json
from pathlib import Path
from .summarizer import ReviewSummarizer
from .vectorizer_with_review import create_business_embeddings_with_summary
import yaml

class FeatureEngineeringPipelineWithReviewSummary:
    def __init__(self, config_path="configs/data_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.profile_path = self.config['feature_engineering']['output']['with_review']['extracted_features_with_text']
        self.summary_profile_path = self.config['feature_engineering']['output']['with_summary']['extracted_features_with_summary']
        self.embedding_path = self.config['feature_engineering']['output']['with_summary']['business_embeddings_with_summary']
        self.metadata_path = self.config['feature_engineering']['output']['with_summary']['business_embeddings_with_summary_metadata']
        self.summarizer_cfg = self.config.get('summarization', {})

    def run(self):
        print("Step 1: Load with_review profile...")
        df = pd.read_json(self.profile_path, encoding='utf-8')
        print(f"Loaded {len(df)} businesses.")

        print("Step 2: Summarize review_tip...")
        summarizer = ReviewSummarizer(
            model_name=self.summarizer_cfg.get('model', 'facebook/bart-large-cnn'),
            device=self.summarizer_cfg.get('device', 'cuda'),  # Default to CUDA for GPU acceleration
            prompt=self.summarizer_cfg.get('prompt'),
            max_length=self.summarizer_cfg.get('max_length', 64),
            min_length=self.summarizer_cfg.get('min_length', 10),
            num_beams=self.summarizer_cfg.get('num_beams', 1),
            temperature=self.summarizer_cfg.get('temperature', 0.3),
            batch_size=self.summarizer_cfg.get('batch_size', 1)
        )
        
        print(f"Summarizer parameters:")
        print(f"  device: {summarizer.device}")
        print(f"  max_length: {summarizer.max_length}")
        print(f"  min_length: {summarizer.min_length}")
        print(f"  num_beams: {summarizer.num_beams}")
        print(f"  temperature: {summarizer.temperature}")
        print(f"  batch_size: {summarizer.batch_size}")
        
        df['review_tip_summary'] = summarizer.summarize_batch(df['review_tip'].fillna("").tolist())
        print("Summarization complete.")

        print(f"Step 3: Save profile with summary to {self.summary_profile_path}")
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
        print("Done.")

def main():
    pipeline = FeatureEngineeringPipelineWithReviewSummary()
    pipeline.run()

if __name__ == '__main__':
    main() 