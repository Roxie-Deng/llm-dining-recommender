"""
Content-based recommender system implementation using SBERT models
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import json
import torch
import os

class ContentBasedRecommender:
    def __init__(self, business_profiles_path: str, model_name: str = "paraphrase-MiniLM-L6-v2"):
        """
        Initialize the recommender system
        
        Args:
            business_profiles_path: Path to business profiles JSON file
            model_name: SBERT model name to use
        """
        # Load business profiles
        with open(business_profiles_path, 'r', encoding='utf-8') as f:
            self.business_profiles = json.load(f)
            
        # Convert to DataFrame for easier processing
        self.business_df = pd.DataFrame(self.business_profiles)
        
        # Check CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load SBERT model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize feature processors
        self.scaler = StandardScaler()
        
        # Process features and compute similarity matrix
        self.features = self._process_features()
        self.similarity_matrix = self._compute_similarity_matrix()
        
        # Create business ID to index mapping
        self.business_id_to_idx = {
            business['business_id']: i for i, business in enumerate(self.business_profiles)
        }
        
    def _process_features(self):
        """Process and combine all features"""
        # 1. Prepare text data for semantic encoding
        text_data = []
        for _, row in self.business_df.iterrows():
            # Combine name, description, categories, and review text
            combined_text = f"{row['name']} - {row['description']}"
            
            # Add categories if available
            if 'categories' in row and row['categories']:
                combined_text += f" - {row['categories']}"
                
            # Add review/tip text if available
            if 'text' in row and row['text'] and len(row['text']) > 0:
                # Truncate review text if too long (SBERT may have token limits)
                review_text = row['text'][:1000] if len(row['text']) > 1000 else row['text']
                combined_text += f" - {review_text}"
                
            text_data.append(combined_text)
        
        # 2. Encode text using SBERT
        print("Encoding text data using SBERT...")
        text_embeddings = self.model.encode(text_data, show_progress_bar=True)
        
        # 3. Process numeric features
        numeric_columns = ['stars', 'review_count', 'price_normalized']
        numeric_features = self.business_df[numeric_columns].values
        scaled_numeric = self.scaler.fit_transform(numeric_features)
        
        # 4. Process boolean features
        boolean_columns = ['reservations', 'outdoor_seating']
        boolean_features = self.business_df[boolean_columns].values
        
        # 5. Combine features with weights
        # Weights for different feature types
        text_weight = 0.7  # Give higher weight to text embeddings
        numeric_weight = 0.2
        boolean_weight = 0.1
        
        # Calculate dimensions for debugging
        text_dim = text_embeddings.shape[1]
        numeric_dim = scaled_numeric.shape[1]
        boolean_dim = boolean_features.shape[1]
        
        print(f"Feature dimensions - Text: {text_dim}, Numeric: {numeric_dim}, Boolean: {boolean_dim}")
        
        # Combine features
        combined_features = np.hstack([
            text_weight * text_embeddings,
            numeric_weight * scaled_numeric,
            boolean_weight * boolean_features
        ])
        
        return combined_features
        
    def _compute_similarity_matrix(self):
        """Compute cosine similarity matrix between businesses"""
        print("Computing similarity matrix...")
        return cosine_similarity(self.features)
        
    def get_similar_businesses(self, business_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-N most similar businesses to the given business
        
        Args:
            business_id: Target business ID
            top_n: Number of similar businesses to return
            
        Returns:
            List of similar businesses with similarity scores
        """
        # Check if business ID exists
        if business_id not in self.business_id_to_idx:
            print(f"Business ID {business_id} not found in the dataset")
            return []
            
        # Get business index
        business_idx = self.business_id_to_idx[business_id]
        
        # Get similarity scores for this business
        similarity_scores = self.similarity_matrix[business_idx].copy()
        
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
        
    def recommend_for_user(self, user_history: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user based on their history
        
        Args:
            user_history: List of business IDs the user has interacted with
            top_n: Number of recommendations to generate
            
        Returns:
            List of recommended businesses with similarity scores
        """
        # Validate user history
        if not user_history:
            print("User history is empty")
            return []
            
        # Filter valid business IDs
        valid_business_ids = []
        for bid in user_history:
            if bid in self.business_id_to_idx:
                valid_business_ids.append(bid)
            else:
                print(f"Business ID {bid} not found in the dataset")
                
        if not valid_business_ids:
            print("No valid business IDs in user history")
            return []
            
        # Get similarity scores for each business in user history
        similarity_scores = np.zeros(len(self.business_profiles))
        for bid in valid_business_ids:
            idx = self.business_id_to_idx[bid]
            similarity_scores += self.similarity_matrix[idx]
            
        # Average the scores
        similarity_scores /= len(valid_business_ids)
        
        # Exclude businesses already in user history
        for bid in valid_business_ids:
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
        """
        Save the model and necessary data for later use
        
        Args:
            path: Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save similarity matrix
        np.save(os.path.join(path, 'similarity_matrix.npy'), self.similarity_matrix)
        
        # Save features
        np.save(os.path.join(path, 'features.npy'), self.features)
        
        # Save business ID to index mapping
        with open(os.path.join(path, 'business_id_mapping.json'), 'w') as f:
            json.dump(self.business_id_to_idx, f)
            
        print(f"Model saved to {path}")
        
    @classmethod
    def load_model(cls, model_path: str, business_profiles_path: str):
        """
        Load a previously saved model
        
        Args:
            model_path: Path to the saved model directory
            business_profiles_path: Path to business profiles JSON file
            
        Returns:
            Loaded ContentBasedRecommender instance
        """
        # Create instance without processing
        instance = cls.__new__(cls)
        
        # Load business profiles
        with open(business_profiles_path, 'r', encoding='utf-8') as f:
            instance.business_profiles = json.load(f)
            
        # Load DataFrame
        instance.business_df = pd.DataFrame(instance.business_profiles)
        
        # Load similarity matrix
        instance.similarity_matrix = np.load(os.path.join(model_path, 'similarity_matrix.npy'))
        
        # Load features
        instance.features = np.load(os.path.join(model_path, 'features.npy'))
        
        # Load business ID mapping
        with open(os.path.join(model_path, 'business_id_mapping.json'), 'r') as f:
            instance.business_id_to_idx = json.load(f)
            
        print(f"Model loaded from {model_path}")
        return instance 