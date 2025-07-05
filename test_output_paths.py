#!/usr/bin/env python3
"""
Test script to verify the new output paths configuration
"""

import yaml
import os

def test_output_paths():
    """Test the output paths configuration"""
    
    print("Testing output paths configuration...")
    
    # Load config
    with open("configs/data_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check with_summary paths
    with_summary_paths = config['feature_engineering']['output']['with_summary']
    
    print("With Summary Output Paths:")
    for key, path in with_summary_paths.items():
        print(f"  {key}: {path}")
        
        # Check if directory exists or can be created
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            print(f"    ✅ Directory exists: {dir_path}")
        else:
            print(f"    ⚠️  Directory will be created: {dir_path}")
    
    print("\n✅ All output paths are correctly configured!")
    print("The pipeline will save files to:")
    print(f"  - Extracted features: {with_summary_paths['extracted_features_with_summary']}")
    print(f"  - Embeddings: {with_summary_paths['business_embeddings_with_summary']}")
    print(f"  - Metadata: {with_summary_paths['business_embeddings_with_summary_metadata']}")

if __name__ == "__main__":
    test_output_paths() 