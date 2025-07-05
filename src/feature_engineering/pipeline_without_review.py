"""
Feature engineering pipeline without review processing.
Main entry point for feature extraction and vectorization without review data.
"""

import pandas as pd
import json
import yaml
from pathlib import Path
import os
from .feature_extractor_without_review import FeatureExtractorWithoutReview
from .vectorizer_without_review import create_business_embeddings_without_review
import numpy as np

class FeatureEngineeringPipelineWithoutReview:
    def __init__(self, config_path="configs/data_config.yaml"):
        """
        Initialize the feature engineering pipeline without review processing
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.extractor = FeatureExtractorWithoutReview(config_path)
        self.extracted_features = None
        self.embeddings = None
        self.business_ids = None
        self.feature_names = None
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_business_data(self, input_path):
        """
        Load business data from JSON file
        
        Args:
            input_path (str): Path to business data file
            
        Returns:
            DataFrame: Loaded business data
        """
        print(f"Loading business data from: {input_path}")
        data = []
        try:
            # Try to read as jsonlines (one JSON object per line)
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            df = pd.DataFrame(data)
        except json.JSONDecodeError:
            # Fallback: try to read as a standard JSON array
            print("[Info] Fallback to standard JSON array format.")
            df = pd.read_json(input_path)
        print(f"Loaded {len(df)} businesses")
        return df
    
    def extract_features_step(self, businesses_df):
        """
        Extract features from business data (without reviews)
        
        Args:
            businesses_df (DataFrame): Raw business data
            
        Returns:
            DataFrame: DataFrame with extracted features
        """
        print("\n=== Feature Extraction Step (Without Reviews) ===")
        
        self.extracted_features = self.extractor.extract_features(businesses_df)
        
        return self.extracted_features
    
    def vectorize_features_step(self, output_path=None):
        """
        Vectorize extracted features (without reviews)
        
        Args:
            output_path (str, optional): Path to save embeddings
            
        Returns:
            tuple: (embeddings, business_ids, feature_names)
        """
        if self.extracted_features is None:
            raise ValueError("No extracted features available. Run extract_features_step first.")
        
        print("\n=== Feature Vectorization Step (Without Reviews) ===")
        
        self.embeddings, self.business_ids, self.feature_names = create_business_embeddings_without_review(
            self.extracted_features,
            output_path=output_path
        )
        
        return self.embeddings, self.business_ids, self.feature_names
    
    def save_extracted_features(self, output_path):
        """
        Save extracted features to JSON file
        
        Args:
            output_path (str): Path to save extracted features
        """
        if self.extracted_features is None:
            raise ValueError("No extracted features available.")
        
        print(f"Saving extracted features to: {output_path}")
        
        # Convert DataFrame to list of dictionaries
        features_list = self.extracted_features.to_dict('records')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features_list, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted features saved to: {output_path}")
    
    def get_feature_summary(self):
        """
        Get summary of extracted features
        
        Returns:
            dict: Feature summary statistics
        """
        if self.extracted_features is None:
            return {}
        
        summary = {
            'total_businesses': len(self.extracted_features),
            'features': {}
        }
        
        # Analyze each feature column
        for column in self.extracted_features.columns:
            if column == 'business_id':
                continue
                
            feature_data = self.extracted_features[column]
            
            if feature_data.dtype in ['int64', 'float64']:
                summary['features'][column] = {
                    'type': 'numerical',
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max(),
                    'missing': feature_data.isna().sum()
                }
            elif feature_data.dtype == 'bool':
                summary['features'][column] = {
                    'type': 'boolean',
                    'true_count': feature_data.sum(),
                    'false_count': (~feature_data).sum(),
                    'missing': feature_data.isna().sum()
                }
            else:
                summary['features'][column] = {
                    'type': 'categorical/text',
                    'unique_values': feature_data.nunique(),
                    'missing': feature_data.isna().sum()
                }
        
        return summary
    
    def run_pipeline(self, business_data_path, output_dir=None):
        """
        Run the complete feature engineering pipeline without review processing
        
        Args:
            business_data_path (str): Path to business data file
            output_dir (str): Directory to save outputs
            
        Returns:
            dict: Pipeline results
        """
        print("=== Feature Engineering Pipeline (Without Reviews) ===")
        
        # Use config default if output_dir not specified
        if output_dir is None:
            output_dir = self.config['paths']['features_output']['without_review']
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load business data
        print("\nStep 1: Loading business data...")
        businesses_df = self.load_business_data(business_data_path)
        
        # Step 2: Extract features (without reviews)
        print("\nStep 2: Extracting features (without reviews)...")
        extracted_df = self.extract_features_step(businesses_df)
        
        # Save extracted features
        extracted_features_path = self.config['feature_engineering']['output']['without_review']['extracted_features']
        self.save_extracted_features(extracted_features_path)
        
        # Step 3: Vectorize features (without reviews)
        print("\nStep 3: Vectorizing features (without reviews)...")
        embeddings_path = self.config['feature_engineering']['output']['without_review']['embeddings']
        embeddings, business_ids, feature_names = self.vectorize_features_step(embeddings_path)
        
        # Step 4: Generate summary
        print("\nStep 4: Generating feature summary...")
        feature_summary = self.get_feature_summary()
        
        # Save summary
        summary_path = self.config['feature_engineering']['output']['without_review']['summary']
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(convert_np(feature_summary), f, indent=2, ensure_ascii=False)
        
        # Print results
        print(f"\n=== Pipeline Complete (Without Reviews) ===")
        print(f"Total businesses processed: {len(businesses_df)}")
        print(f"Features extracted: {len(extracted_df.columns)}")
        print(f"Embedding dimensions: {embeddings.shape}")
        print(f"Output files:")
        print(f"  - Extracted features: {extracted_features_path}")
        print(f"  - Embeddings: {embeddings_path}_embeddings.npy")
        print(f"  - Metadata: {embeddings_path}_metadata.json")
        print(f"  - Vectorizer: {embeddings_path}_vectorizer.pkl")
        print(f"  - Summary: {summary_path}")
        
        return {
            'extracted_features': extracted_df,
            'embeddings': embeddings,
            'business_ids': business_ids,
            'feature_names': feature_names,
            'summary': feature_summary,
            'output_paths': {
                'extracted_features': extracted_features_path,
                'embeddings': f"{embeddings_path}_embeddings.npy",
                'metadata': f"{embeddings_path}_metadata.json",
                'vectorizer': f"{embeddings_path}_vectorizer.pkl",
                'summary': summary_path
            }
        }

# Utility function to convert numpy types to native Python types for JSON serialization

def convert_np(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

def main():
    """
    Main function to run the feature engineering pipeline without reviews
    """
    # Initialize pipeline
    pipeline = FeatureEngineeringPipelineWithoutReview()
    
    # Use config default paths
    config = pipeline.config
    business_data_path = config['paths']['processed']['stratified_sample']
    output_dir = config['paths']['features_output']['without_review']
    
    # Run pipeline
    results = pipeline.run_pipeline(
        business_data_path=business_data_path,
        output_dir=output_dir
    )
    
    print("\nFeature engineering pipeline (without reviews) completed successfully!")

if __name__ == "__main__":
    main() 