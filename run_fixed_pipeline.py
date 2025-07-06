#!/usr/bin/env python3
"""
Run the fixed pipeline with review summary
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_engineering.pipeline_with_review_summary import FeatureEngineeringPipelineWithReviewSummary

def run_fixed_pipeline():
    """Run the fixed pipeline"""
    
    print("🚀 Starting fixed pipeline with review summary...")
    print("=" * 50)
    
    try:
        # Create pipeline instance
        pipeline = FeatureEngineeringPipelineWithReviewSummary()
        
        # Run pipeline
        pipeline.run()
        
        print("=" * 50)
        print("✅ Pipeline completed successfully!")
        print("\nOutput files:")
        print("- data/engineered/with_summary/extracted_features_with_summary.json")
        print("- data/engineered/with_summary/business_embeddings_with_summary.npy")
        print("- data/engineered/with_summary/business_embeddings_with_summary_metadata.json")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_fixed_pipeline()
    if success:
        print("\n🎉 All done! The pipeline has been completed successfully.")
    else:
        print("\n💥 Pipeline failed. Please check the error messages above.") 