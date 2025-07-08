"""
Test script for the content-based recommender using SBERT
"""

import json
import os
import time
from content_based import ContentBasedRecommender

def test_recommender(model_path=None, save_results=True):
    """Test the content-based recommender"""
    start_time = time.time()
    
    # Business profiles path
    business_profiles_path = "data/processed/features/sample_200_business_profiles_refine.json"
    
    # Create results directory
    results_dir = "data/results/content_based"
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
    
    # Load model or create new one
    if model_path and os.path.exists(os.path.join(model_path, 'similarity_matrix.npy')):
        print(f"Loading pre-computed model from {model_path}...")
        recommender = ContentBasedRecommender.load_model(model_path, business_profiles_path)
    else:
        print("Creating new recommender model...")
        # Initialize the recommender with SBERT model
        recommender = ContentBasedRecommender(
            business_profiles_path,
            model_name="paraphrase-MiniLM-L6-v2"  # Faster model for testing
        )
        
        # Save the model for future use
        if model_path:
            recommender.save_model(model_path)
    
    # Print initialization time
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f} seconds")
    
    # Test getting similar businesses
    print("\nTesting similar business recommendations:")
    test_business_ids = [
        "xJyp6RLqNRv3tSu6njPKxQ",  # World of Beer (Casual Dining)
        "V6Om7YZhlRQRU7WfuxHq8Q",  # Beck's Cajun Cafe (Specialty Dining)
        "DytKODMqcvQ7MWA0NN2uNw",  # Lloyd Whiskey Bar (Bars & Lounges)
        "KgozvZ1UFfXuJToqe8CfQg"   # Fortune Chinese Restaurant (Asian Cuisines)
    ]
    
    # Dictionary to store all item-based recommendation results
    item_recommendations = {}
    
    for business_id in test_business_ids:
        print(f"\nFinding similar businesses to: {business_id}")
        similar_businesses = recommender.get_similar_businesses(business_id, top_n=3)
        
        # Store results in dictionary
        item_recommendations[business_id] = similar_businesses
        
        if similar_businesses:
            for idx, business in enumerate(similar_businesses):
                print(f"{idx+1}. {business['name']} (Score: {business['similarity_score']:.4f})")
                print(f"   Description: {business['description']}")
                print(f"   Categories: {business['categories']}")
        else:
            print(f"No similar businesses found for {business_id}")
    
    # Save item-based recommendation results
    if save_results:
        item_results_path = os.path.join(results_dir, "item_recommendations.json")
        with open(item_results_path, 'w', encoding='utf-8') as f:
            json.dump(item_recommendations, f, indent=2)
        print(f"\nItem-based recommendations saved to: {item_results_path}")
    
    # Test user recommendations
    print("\nTesting user recommendations:")
    test_user_histories = [
        # User interested in Asian cuisine
        ["KgozvZ1UFfXuJToqe8CfQg", "mDd3GFkYQKXhTlpm7SHtCQ"],
        # User interested in bars/drinks
        ["xJyp6RLqNRv3tSu6njPKxQ", "DytKODMqcvQ7MWA0NN2uNw", "_msFUghqBbYNmO8grFJNAQ"],
        # Mixed interests
        ["V6Om7YZhlRQRU7WfuxHq8Q", "DytKODMqcvQ7MWA0NN2uNw", "4y5nCB6NWXRpjL8Spo1iuw"]
    ]
    
    # Dictionary to store all user-based recommendation results
    user_recommendations = {}
    
    for i, user_history in enumerate(test_user_histories):
        user_id = f"user_{i+1}"
        print(f"\nRecommendations for User {i+1} (History: {len(user_history)} businesses):")
        
        # Get business names for history
        history_names = []
        for bid in user_history:
            for business in recommender.business_profiles:
                if business['business_id'] == bid:
                    history_names.append(business['name'])
                    break
        
        print(f"User history: {', '.join(history_names)}")
        
        # Get recommendations
        recommendations = recommender.recommend_for_user(user_history, top_n=3)
        
        # Store results for this user
        user_recommendations[user_id] = {
            'history': user_history,
            'history_names': history_names,
            'recommendations': recommendations
        }
        
        for idx, rec in enumerate(recommendations):
            print(f"{idx+1}. {rec['name']} (Score: {rec['similarity_score']:.4f})")
            print(f"   Description: {rec['description']}")
            print(f"   Categories: {rec['categories']}")
    
    # Save user-based recommendation results
    if save_results:
        user_results_path = os.path.join(results_dir, "user_recommendations.json")
        with open(user_results_path, 'w', encoding='utf-8') as f:
            json.dump(user_recommendations, f, indent=2)
        print(f"\nUser-based recommendations saved to: {user_results_path}")
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Return results for potential further processing
    return {
        'item_recommendations': item_recommendations,
        'user_recommendations': user_recommendations
    }

if __name__ == "__main__":
    # Optional: specify a path to save/load the model
    model_path = "data/models/content_based"
    test_recommender(model_path) 