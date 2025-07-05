"""
Feature engineering pipeline with review processing.
Main entry point for feature extraction and vectorization with review and tip data.
"""

import pandas as pd
import json
import yaml
from pathlib import Path
import os
from collections import defaultdict
from tqdm import tqdm
from .vectorizer_with_review import create_business_embeddings_with_review
import numpy as np

class FeatureEngineeringPipelineWithReview:
    def __init__(self, config_path="configs/data_config.yaml"):
        """
        Initialize the feature engineering pipeline with review processing
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.extracted_features_with_text = None
        self.embeddings = None
        self.business_ids = None
        self.feature_names = None

    def _load_config(self, config_path):
        """Load configuration from YAML file (force utf-8 encoding)"""
        with open(config_path, 'r', encoding='utf-8') as f:
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
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            df = pd.DataFrame(data)
        except json.JSONDecodeError:
            print("[Info] Fallback to standard JSON array format.")
            df = pd.read_json(input_path)
        print(f"Loaded {len(df)} businesses")
        return df

    def load_review_data(self, review_path, tip_path):
        """
        Load review and tip data, group by business_id
        Args:
            review_path (str): Path to review data file
            tip_path (str): Path to tip data file
        Returns:
            dict, dict: reviews_by_biz, tips_by_biz
        """
        reviews_by_biz = defaultdict(list)
        with open(review_path, 'r', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                reviews_by_biz[r['business_id']].append(r)
        tips_by_biz = defaultdict(list)
        with open(tip_path, 'r', encoding='utf-8') as f:
            for line in f:
                t = json.loads(line)
                tips_by_biz[t['business_id']].append(t)
        return reviews_by_biz, tips_by_biz

    def add_review_tip_text(self, businesses_df, reviews_by_biz, tips_by_biz, n_reviews=10, n_tips=1):
        """
        For each business, add a 'review_tip' field by concatenating latest n_reviews reviews and n_tips tips
        Args:
            businesses_df (DataFrame): Business data
            reviews_by_biz (dict): Reviews grouped by business_id
            tips_by_biz (dict): Tips grouped by business_id
            n_reviews (int): Number of latest reviews to use
            n_tips (int): Number of latest tips to use
        Returns:
            DataFrame: Updated DataFrame with 'review_tip' field
        """
        review_texts = []
        for idx, row in tqdm(businesses_df.iterrows(), total=len(businesses_df), desc='Adding review_tip'):
            bid = row['business_id']
            reviews = sorted(reviews_by_biz.get(bid, []), key=lambda x: x['date'], reverse=True)[:n_reviews]
            review_str = ' '.join([r['text'] for r in reviews])
            tips = sorted(tips_by_biz.get(bid, []), key=lambda x: x['date'], reverse=True)[:n_tips]
            tip_str = ' '.join([t['text'] for t in tips])
            review_tip = (review_str + ' ' + tip_str).strip()
            review_texts.append(review_tip)
        businesses_df = businesses_df.copy()
        businesses_df['review_tip'] = review_texts
        return businesses_df

    def save_extracted_features_with_text(self, output_path):
        """
        Save extracted features with review_tip to JSON file
        Args:
            output_path (str): Path to save extracted features
        """
        if self.extracted_features_with_text is None:
            raise ValueError("No extracted features with text available.")
        print(f"Saving extracted features with text to: {output_path}")
        features_list = self.extracted_features_with_text.to_dict('records')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features_list, f, indent=2, ensure_ascii=False)
        print(f"Extracted features with text saved to: {output_path}")

    def run_pipeline(self, business_data_path, review_path, tip_path, output_dir=None):
        """
        Run the complete feature engineering pipeline with review processing
        Args:
            business_data_path (str): Path to business data file
            review_path (str): Path to review data file
            tip_path (str): Path to tip data file
            output_dir (str): Directory to save outputs
        Returns:
            dict: Pipeline results
        """
        print("=== Feature Engineering Pipeline (With Reviews) ===")
        if output_dir is None:
            output_dir = self.config['paths']['features_output']['with_review']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("\nStep 1: Loading business data...")
        businesses_df = self.load_business_data(business_data_path)
        print("\nStep 2: Loading review and tip data...")
        reviews_by_biz, tips_by_biz = self.load_review_data(review_path, tip_path)
        print("\nStep 3: Adding review_tip field...")
        businesses_df = self.add_review_tip_text(businesses_df, reviews_by_biz, tips_by_biz)
        self.extracted_features_with_text = businesses_df
        extracted_features_path = self.config['feature_engineering']['output']['with_review']['extracted_features_with_text']
        self.save_extracted_features_with_text(extracted_features_path)
        print("\nStep 4: Vectorizing features (with reviews)...")
        self.embeddings, self.business_ids, self.feature_names = create_business_embeddings_with_review(
            self.extracted_features_with_text,
            config_path=self.config,
            output_dir=output_dir
        )
        print("Pipeline completed.")
        return {
            'features': self.extracted_features_with_text,
            'embeddings': self.embeddings,
            'business_ids': self.business_ids,
            'feature_names': self.feature_names
        }

def main():
    pipeline = FeatureEngineeringPipelineWithReview()
    config = pipeline.config
    # Read input paths from config['paths']['raw']
    business_data_path = config['paths']['raw']['business']
    review_path = config['paths']['raw']['review']
    tip_path = config['paths']['raw']['tip']
    output_dir = config['paths']['features_output']['with_review']
    pipeline.run_pipeline(business_data_path, review_path, tip_path, output_dir)

if __name__ == '__main__':
    main() 