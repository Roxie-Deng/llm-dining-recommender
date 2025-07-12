import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class ContentBasedRecommender:
    """
    Item-item similarity recommender using precomputed features (npy) and metadata (json).
    Computes cosine similarity between all items and generates top-N recommendations.
    """
    def __init__(self, features_path: str, metadata_path: str):
        """
        Initialize the recommender by loading features and metadata.
        Args:
            features_path: Path to the .npy file containing item features.
            metadata_path: Path to the .json file containing item metadata.
        """
        # Load features (shape: [num_items, feature_dim])
        self.features = np.load(features_path)
        # Load metadata (list of dicts, one per item)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        # Check alignment
        assert len(self.features) == len(self.metadata), "Features and metadata size mismatch!"
        # Build id-to-index mapping
        self.id_to_idx = {item['business_id']: idx for idx, item in enumerate(self.metadata)}
        self.idx_to_id = {idx: item['business_id'] for idx, item in enumerate(self.metadata)}
        self.similarity_matrix = None

    def compute_similarity_matrix(self):
        """
        Compute the cosine similarity matrix for all items.
        """
        self.similarity_matrix = cosine_similarity(self.features)

    def get_top_n_similar(self, top_n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        For each item, get the top-N most similar items (excluding itself).
        Args:
            top_n: Number of similar items to recommend for each item.
        Returns:
            Dictionary mapping item_id to a list of recommended items with similarity scores.
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        num_items = len(self.metadata)
        recommendations = {}
        for idx in range(num_items):
            sim_scores = self.similarity_matrix[idx].copy()
            sim_scores[idx] = -1  # Exclude self
            top_indices = np.argsort(sim_scores)[-top_n:][::-1]
            recs = []
            for rec_idx in top_indices:
                rec_item = self.metadata[rec_idx]
                recs.append({
                    'business_id': rec_item['business_id'],
                    'name': rec_item.get('name', ''),
                    'similarity_score': float(sim_scores[rec_idx]),
                    'description': rec_item.get('description', ''),
                    'cluster': rec_item.get('cluster', 0),
                    'categories': rec_item.get('categories', '')
                })
            recommendations[self.metadata[idx]['business_id']] = recs
        return recommendations

    def save_recommendations(self, recommendations: Dict[str, List[Dict[str, Any]]], output_path: str):
        """
        Save the recommendations to a JSON file.
        Args:
            recommendations: The recommendations dictionary.
            output_path: Path to save the JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False) 