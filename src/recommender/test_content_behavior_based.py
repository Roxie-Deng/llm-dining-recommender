"""
Test script for Content-Behavior Based Recommender System
"""

import os
import json
import numpy as np
from content_behavior_based import ContentBehaviorRecommender

def test_recommender():
    # Paths
    business_profiles_path = "data/processed/features/sample_200_business_profiles_refine.json"
    user_behavior_path = "data/processed/user_modeling/user_behavior.json"
    model_save_path = "data/processed/models/content_behavior"
    
    # Create test user behavior data if not exists
    if not os.path.exists(user_behavior_path):
        test_user_behavior = {
            "test_user_1": {
                "business_1": {
                    "timestamp": "2024-01-01T12:00:00Z",
                    "rating": 5,
                    "visit_count": 3
                },
                "business_2": {
                    "timestamp": "2024-02-01T12:00:00Z",
                    "rating": 4,
                    "visit_count": 2
                }
            }
        }
        os.makedirs(os.path.dirname(user_behavior_path), exist_ok=True)
        with open(user_behavior_path, 'w') as f:
            json.dump(test_user_behavior, f)
    
    # Initialize recommender
    print("Initializing recommender...")
    recommender = ContentBehaviorRecommender(
        business_profiles_path=business_profiles_path,
        user_behavior_path=user_behavior_path
    )
    
    # Test 1: Get similar businesses without user context
    print("\nTest 1: Getting similar businesses without user context")
    business_id = "xJyp6RLqNRv3tSu6njPKxQ"  # World of Beer
    similar_businesses = recommender.get_similar_businesses(business_id, top_n=3)
    print(f"Similar businesses to {business_id}:")
    for business in similar_businesses:
        print(f"- {business['name']} (score: {business['similarity_score']:.3f})")
    
    # Test 2: Get similar businesses with user context
    print("\nTest 2: Getting similar businesses with user context")
    user_id = "test_user_1"
    similar_businesses = recommender.get_similar_businesses(
        business_id=business_id,
        user_id=user_id,
        top_n=3
    )
    print(f"Similar businesses to {business_id} for user {user_id}:")
    for business in similar_businesses:
        print(f"- {business['name']} (score: {business['similarity_score']:.3f})")
    
    # Test 3: Get personalized recommendations
    print("\nTest 3: Getting personalized recommendations")
    recommendations = recommender.recommend_for_user(user_id, top_n=3)
    print(f"Recommendations for user {user_id}:")
    for business in recommendations:
        print(f"- {business['name']} (score: {business['similarity_score']:.3f})")
    
    # Test 4: Save and load model
    print("\nTest 4: Saving and loading model")
    recommender.save_model(model_save_path)
    
    loaded_recommender = ContentBehaviorRecommender.load_model(
        model_path=model_save_path,
        business_profiles_path=business_profiles_path,
        user_behavior_path=user_behavior_path
    )
    
    # Verify loaded model
    recommendations_loaded = loaded_recommender.recommend_for_user(user_id, top_n=3)
    print("Recommendations from loaded model:")
    for business in recommendations_loaded:
        print(f"- {business['name']} (score: {business['similarity_score']:.3f})")
    
    # Test 5: Time decay factor
    print("\nTest 5: Testing time decay factor")
    test_timestamps = [
        "2024-03-01T12:00:00Z",  # Recent
        "2024-02-01T12:00:00Z",  # 1 month ago
        "2024-01-01T12:00:00Z",  # 2 months ago
        "2023-12-01T12:00:00Z"   # 3 months ago
    ]
    
    for timestamp in test_timestamps:
        decay = recommender._get_time_decay_factor(timestamp)
        print(f"Time decay for {timestamp}: {decay:.3f}")

if __name__ == "__main__":
    test_recommender() 