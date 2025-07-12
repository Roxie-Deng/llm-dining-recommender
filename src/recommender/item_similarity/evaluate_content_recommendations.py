import json
import numpy as np
from collections import Counter
from typing import Dict, List, Any
import os

def evaluate_item_recommendations(recommendations_path: str, metadata_path: str, output_path: str):
    """
    Evaluate item-item recommendations using cluster match rate, price difference, and category overlap.
    Args:
        recommendations_path: Path to the JSON file with item recommendations.
        metadata_path: Path to the JSON file with item metadata.
        output_path: Path to save the evaluation results (JSON).
    """
    # Load recommendations
    with open(recommendations_path, 'r', encoding='utf-8') as f:
        recommendations = json.load(f)
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    # Build business_id to metadata mapping
    business_lookup = {item['business_id']: item for item in metadata}

    # Evaluation statistics
    same_cluster = 0
    different_cluster = 0
    cluster_distances = []
    price_differences = []
    category_overlaps = []
    similarity_scores = []

    for source_id, recs in recommendations.items():
        source = business_lookup.get(source_id)
        if not source:
            continue
        source_cluster = source.get('cluster', 0)
        source_price = source.get('price_normalized', 0.5)
        source_categories = set(str(source.get('categories', '')).split(', '))
        for rec in recs:
            rec_id = rec['business_id']
            rec_item = business_lookup.get(rec_id)
            if not rec_item:
                continue
            rec_cluster = rec_item.get('cluster', 0)
            rec_price = rec_item.get('price_normalized', 0.5)
            rec_categories = set(str(rec_item.get('categories', '')).split(', '))
            # Cluster match
            if source_cluster == rec_cluster:
                same_cluster += 1
            else:
                different_cluster += 1
            # Cluster distance
            cluster_distances.append(abs(source_cluster - rec_cluster))
            # Price difference
            price_differences.append(abs(float(source_price) - float(rec_price)))
            # Category overlap (Jaccard)
            if source_categories and rec_categories:
                overlap = len(source_categories.intersection(rec_categories))
                union = len(source_categories.union(rec_categories))
                ratio = overlap / union if union > 0 else 0
                category_overlaps.append(ratio)
            # Similarity score
            similarity_scores.append(rec.get('similarity_score', 0))

    # Compute statistics
    total = same_cluster + different_cluster
    stats = {
        'similarity_score': {
            'mean': float(np.mean(similarity_scores)) if similarity_scores else 0,
            'min': float(np.min(similarity_scores)) if similarity_scores else 0,
            'max': float(np.max(similarity_scores)) if similarity_scores else 0,
            'median': float(np.median(similarity_scores)) if similarity_scores else 0
        },
        'cluster_match_rate': same_cluster / total if total > 0 else 0,
        'avg_cluster_distance': float(np.mean(cluster_distances)) if cluster_distances else 0,
        'avg_price_difference': float(np.mean(price_differences)) if price_differences else 0,
        'avg_category_overlap': float(np.mean(category_overlaps)) if category_overlaps else 0
    }

    # Save evaluation results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {output_path}") 