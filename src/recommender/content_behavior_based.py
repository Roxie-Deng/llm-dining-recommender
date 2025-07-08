"""
Content-Behavior Based Recommender System
Combines content features with user behavior patterns for personalized recommendations
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import json
import torch
import os
from datetime import datetime
import math

class ContentBehaviorRecommender:
    def __init__(self, 
                 business_profiles_path: str,
                 user_behavior_path: str = None,
                 model_name: str = "paraphrase-MiniLM-L6-v2"):
        """
        Initialize the recommender system
        
        Args:
            business_profiles_path: Path to business profiles JSON file
            user_behavior_path: Path to user behavior data JSON file
            model_name: SBERT model name to use
        """
        # Load business profiles
        with open(business_profiles_path, 'r', encoding='utf-8') as f:
            self.business_profiles = json.load(f)
            
        # Convert to DataFrame for easier processing
        self.business_df = pd.DataFrame(self.business_profiles)
        
        # Load user behavior data if provided
        self.user_behavior = {}
        if user_behavior_path and os.path.exists(user_behavior_path):
            with open(user_behavior_path, 'r', encoding='utf-8') as f:
                self.user_behavior = json.load(f)
        
        # Check CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load SBERT model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize feature processors
        self.scaler = StandardScaler()
        
        # Process features and compute similarity matrices
        self.content_features = self._process_content_features()
        self.content_similarity = self._compute_similarity_matrix(self.content_features)
        
        # Create business ID to index mapping
        self.business_id_to_idx = {
            business['business_id']: i for i, business in enumerate(self.business_profiles)
        }
        
        # Initialize user profiles cache
        self.user_profiles_cache = {}
        
    def _process_content_features(self) -> np.ndarray:
        """Process and combine all content features"""
        # 1. Prepare text data for semantic encoding
        text_data = []
        for _, row in self.business_df.iterrows():
            # Combine name, description, and categories
            combined_text = f"{row['name']} - {row['description']}"
            
            # Add categories if available
            if 'categories' in row and row['categories']:
                combined_text += f" - {row['categories']}"
                
            text_data.append(combined_text)
        
        # 2. Encode text using SBERT
        print("Encoding text data using SBERT...")
        text_embeddings = self.model.encode(text_data, show_progress_bar=True)
        
        # 3. Process numeric features
        numeric_columns = ['stars', 'review_count', 'price_normalized']
        numeric_features = self.business_df[numeric_columns].values
        scaled_numeric = self.scaler.fit_transform(numeric_features)
        
        # 4. Process boolean features (convert 0/1 to float)
        boolean_columns = ['reservations', 'outdoor_seating']
        boolean_features = self.business_df[boolean_columns].values.astype(float)
        
        # 5. Combine features with weights
        text_weight = 0.7
        numeric_weight = 0.2
        boolean_weight = 0.1
        
        combined_features = np.hstack([
            text_weight * text_embeddings,
            numeric_weight * scaled_numeric,
            boolean_weight * boolean_features
        ])
        
        return combined_features
        
    def _compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between businesses"""
        print("Computing similarity matrix...")
        return cosine_similarity(features)
        
    def _get_time_decay_factor(self, timestamp: str) -> float:
        """Calculate time decay factor based on timestamp"""
        if not timestamp:
            return 1.0
            
        try:
            visit_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.now()
            days_diff = (current_time - visit_time).days
            
            # Exponential decay with half-life of 30 days
            decay_factor = math.exp(-days_diff / 30)
            return decay_factor
        except:
            return 1.0
            
    def _build_user_profile(self, user_id: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Build user profile based on their behavior history
        
        Returns:
            Tuple of (user profile vector, business weights)
        """
        if user_id in self.user_profiles_cache:
            return self.user_profiles_cache[user_id]
            
        if user_id not in self.user_behavior:
            return None, {}
            
        user_history = self.user_behavior[user_id]
        business_weights = {}
        
        # Initialize user profile vector
        user_profile = np.zeros_like(self.content_features[0])
        
        # Process each business in user history
        for business_id, behavior in user_history.items():
            if business_id not in self.business_id_to_idx:
                continue
                
            # Calculate business weight based on behavior
            weight = 1.0
            
            # Apply time decay
            if 'timestamp' in behavior:
                weight *= self._get_time_decay_factor(behavior['timestamp'])
                
            # Apply rating weight if available
            if 'rating' in behavior:
                weight *= (behavior['rating'] / 5.0)  # Normalize rating to [0,1]
                
            # Apply visit count weight
            if 'visit_count' in behavior:
                weight *= min(behavior['visit_count'] / 5.0, 1.0)  # Cap at 5 visits
                
            business_weights[business_id] = weight
            
            # Add weighted business features to user profile
            business_idx = self.business_id_to_idx[business_id]
            user_profile += weight * self.content_features[business_idx]
            
        # Normalize user profile
        if np.any(user_profile):
            user_profile /= np.linalg.norm(user_profile)
            
        # Cache the result
        self.user_profiles_cache[user_id] = (user_profile, business_weights)
        
        return user_profile, business_weights
        
    def get_similar_businesses(self, 
                             business_id: str, 
                             user_id: str = None,
                             top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-N most similar businesses to the given business,
        optionally considering user behavior
        
        Args:
            business_id: Target business ID
            user_id: Optional user ID to consider behavior
            top_n: Number of similar businesses to return
            
        Returns:
            List of similar businesses with similarity scores
        """
        if business_id not in self.business_id_to_idx:
            print(f"Business ID {business_id} not found in the dataset")
            return []
            
        # Get business index
        business_idx = self.business_id_to_idx[business_id]
        
        # Get base similarity scores
        similarity_scores = self.content_similarity[business_idx].copy()
        
        # Apply user behavior weights if user_id provided
        if user_id:
            user_profile, business_weights = self._build_user_profile(user_id)
            if user_profile is not None:
                # Adjust similarity scores based on user behavior
                for bid, weight in business_weights.items():
                    if bid in self.business_id_to_idx:
                        idx = self.business_id_to_idx[bid]
                        similarity_scores[idx] *= (1 + weight)
        
        # Exclude the business itself
        similarity_scores[business_idx] = -1
        
        # Get top-N indices
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]
        
        # Prepare results
        similar_businesses = []
        for idx in top_indices:
            business = self.business_profiles[idx]
            similar_businesses.append({
                'business_id': business['business_id'],
                'name': business['name'],
                'similarity_score': float(similarity_scores[idx]),
                'description': business['description'],
                'cluster': business.get('cluster', 0),
                'categories': business.get('categories', '')
            })
            
        return similar_businesses
        
    def recommend_for_user(self, 
                          user_id: str,
                          top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations for a user
        
        Args:
            user_id: User ID to generate recommendations for
            top_n: Number of recommendations to generate
            
        Returns:
            List of recommended businesses with similarity scores
        """
        # Build user profile
        user_profile, business_weights = self._build_user_profile(user_id)
        if user_profile is None:
            print(f"No behavior data found for user {user_id}")
            return []
            
        # Calculate similarity scores between user profile and all businesses
        similarity_scores = np.zeros(len(self.business_profiles))
        for i, business_features in enumerate(self.content_features):
            similarity_scores[i] = cosine_similarity(
                user_profile.reshape(1, -1),
                business_features.reshape(1, -1)
            )[0][0]
            
        # Apply business weights
        for bid, weight in business_weights.items():
            if bid in self.business_id_to_idx:
                idx = self.business_id_to_idx[bid]
                similarity_scores[idx] *= (1 + weight)
                
        # Exclude businesses already in user history
        for bid in business_weights.keys():
            if bid in self.business_id_to_idx:
                idx = self.business_id_to_idx[bid]
                similarity_scores[idx] = -1
                
        # Get top-N indices
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]
        
        # Prepare recommendations
        recommendations = []
        for idx in top_indices:
            business = self.business_profiles[idx]
            recommendations.append({
                'business_id': business['business_id'],
                'name': business['name'],
                'similarity_score': float(similarity_scores[idx]),
                'description': business['description'],
                'cluster': business.get('cluster', 0),
                'categories': business.get('categories', '')
            })
            
        return recommendations
        
    def save_model(self, path: str):
        """Save the model and necessary data for later use"""
        os.makedirs(path, exist_ok=True)
        
        # Save similarity matrix
        np.save(os.path.join(path, 'content_similarity.npy'), self.content_similarity)
        
        # Save features
        np.save(os.path.join(path, 'content_features.npy'), self.content_features)
        
        # Save business ID mapping
        with open(os.path.join(path, 'business_id_mapping.json'), 'w') as f:
            json.dump(self.business_id_to_idx, f)
            
        print(f"Model saved to {path}")
        
    @classmethod
    def load_model(cls, 
                  model_path: str,
                  business_profiles_path: str,
                  user_behavior_path: str = None):
        """Load a previously saved model"""
        instance = cls.__new__(cls)
        
        # Load business profiles
        with open(business_profiles_path, 'r', encoding='utf-8') as f:
            instance.business_profiles = json.load(f)
            
        # Load user behavior if provided
        instance.user_behavior = {}
        if user_behavior_path and os.path.exists(user_behavior_path):
            with open(user_behavior_path, 'r', encoding='utf-8') as f:
                instance.user_behavior = json.load(f)
        
        # Load DataFrame
        instance.business_df = pd.DataFrame(instance.business_profiles)
        
        # Load similarity matrix
        instance.content_similarity = np.load(os.path.join(model_path, 'content_similarity.npy'))
        
        # Load features
        instance.content_features = np.load(os.path.join(model_path, 'content_features.npy'))
        
        # Load business ID mapping
        with open(os.path.join(model_path, 'business_id_mapping.json'), 'r') as f:
            instance.business_id_to_idx = json.load(f)
            
        # Initialize user profiles cache
        instance.user_profiles_cache = {}
            
        print(f"Model loaded from {model_path}")
        return instance 