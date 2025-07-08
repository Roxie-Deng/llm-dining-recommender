"""
Evaluation script for content-based recommendations
"""

import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from pathlib import Path

def load_recommendations():
    """Load recommendation results"""
    item_path = "data/results/content_based/item_recommendations.json"
    user_path = "data/results/content_based/user_recommendations.json"
    
    with open(item_path, 'r', encoding='utf-8') as f:
        item_recommendations = json.load(f)
        
    with open(user_path, 'r', encoding='utf-8') as f:
        user_recommendations = json.load(f)
        
    return item_recommendations, user_recommendations

def load_business_data():
    """Load business data for additional analysis"""
    business_path = "data/processed/features/sample_200_business_profiles_refine.json"
    
    with open(business_path, 'r', encoding='utf-8') as f:
        business_data = json.load(f)
        
    # Create a lookup dictionary
    business_lookup = {
        business["business_id"]: business 
        for business in business_data
    }
    
    return business_data, business_lookup

def evaluate_item_recommendations(item_recommendations, business_lookup):
    """Evaluate item-based recommendations"""
    results = {
        'similarity_scores': [],
        'same_cluster': 0,
        'different_cluster': 0,
        'cluster_distances': [],
        'price_differences': [],
        'category_overlap': []
    }
    
    for source_id, recommendations in item_recommendations.items():
        # Skip if source business not found
        if source_id not in business_lookup:
            continue
            
        source_business = business_lookup[source_id]
        source_cluster = source_business.get('cluster', 0)
        source_price = source_business.get('price_normalized', 0.5)
        source_categories = set(source_business.get('categories', '').split(', '))
        
        for rec in recommendations:
            # Collect similarity score
            results['similarity_scores'].append(rec['similarity_score'])
            
            # Get recommendation data
            rec_id = rec['business_id']
            rec_cluster = rec.get('cluster', 0)
            rec_price = business_lookup[rec_id].get('price_normalized', 0.5)
            rec_categories = set(rec.get('categories', '').split(', '))
            
            # Check cluster match
            if source_cluster == rec_cluster:
                results['same_cluster'] += 1
            else:
                results['different_cluster'] += 1
            
            # Compute cluster distance
            cluster_distance = abs(source_cluster - rec_cluster)
            results['cluster_distances'].append(cluster_distance)
            
            # Compute price difference
            price_diff = abs(source_price - rec_price)
            results['price_differences'].append(price_diff)
            
            # Compute category overlap
            if source_categories and rec_categories:
                overlap = len(source_categories.intersection(rec_categories))
                ratio = overlap / len(source_categories.union(rec_categories))
                results['category_overlap'].append(ratio)
    
    # Calculate statistics
    stats = {
        'similarity_score': {
            'mean': np.mean(results['similarity_scores']),
            'min': min(results['similarity_scores']),
            'max': max(results['similarity_scores']),
            'median': np.median(results['similarity_scores'])
        },
        'cluster_match_rate': results['same_cluster'] / (results['same_cluster'] + results['different_cluster']),
        'avg_cluster_distance': np.mean(results['cluster_distances']),
        'avg_price_difference': np.mean(results['price_differences']),
        'avg_category_overlap': np.mean(results['category_overlap'])
    }
    
    return results, stats

def evaluate_user_recommendations(user_recommendations, business_lookup):
    """Evaluate user-based recommendations"""
    results = {
        'similarity_scores': [],
        'cluster_distributions': [],
        'price_consistency': [],
        'category_diversity': []
    }
    
    for user_id, user_data in user_recommendations.items():
        # Get user history
        history = user_data['history']
        recommendations = user_data['recommendations']
        
        # Skip if no history or recommendations
        if not history or not recommendations:
            continue
        
        # Collect user history data
        history_clusters = []
        history_prices = []
        history_categories = set()
        
        for bid in history:
            if bid in business_lookup:
                business = business_lookup[bid]
                history_clusters.append(business.get('cluster', 0))
                history_prices.append(business.get('price_normalized', 0.5))
                categories = business.get('categories', '').split(', ')
                history_categories.update(categories)
        
        # Analyze recommendations
        rec_clusters = []
        rec_prices = []
        rec_categories = set()
        
        for rec in recommendations:
            # Collect similarity score
            results['similarity_scores'].append(rec['similarity_score'])
            
            # Get recommendation data
            rec_id = rec['business_id']
            if rec_id in business_lookup:
                business = business_lookup[rec_id]
                rec_clusters.append(business.get('cluster', 0))
                rec_prices.append(business.get('price_normalized', 0.5))
                categories = business.get('categories', '').split(', ')
                rec_categories.update(categories)
        
        # Calculate cluster distribution overlap
        history_cluster_counts = Counter(history_clusters)
        rec_cluster_counts = Counter(rec_clusters)
        
        # Normalize counts to percentages
        total_history = sum(history_cluster_counts.values())
        total_rec = sum(rec_cluster_counts.values())
        
        if total_history > 0 and total_rec > 0:
            history_distribution = {k: v/total_history for k, v in history_cluster_counts.items()}
            rec_distribution = {k: v/total_rec for k, v in rec_cluster_counts.items()}
            
            # Calculate KL divergence between distributions (lower is better)
            all_clusters = set(history_distribution.keys()).union(set(rec_distribution.keys()))
            distribution_diff = 0
            for cluster in all_clusters:
                p = history_distribution.get(cluster, 0.001)  # avoid zero
                q = rec_distribution.get(cluster, 0.001)  # avoid zero
                distribution_diff += p * np.log(p/q)
            
            results['cluster_distributions'].append(distribution_diff)
        
        # Calculate price consistency
        if history_prices and rec_prices:
            avg_history_price = np.mean(history_prices)
            avg_rec_price = np.mean(rec_prices)
            price_diff = abs(avg_history_price - avg_rec_price)
            results['price_consistency'].append(price_diff)
        
        # Calculate category diversity
        if history_categories and rec_categories:
            new_categories = rec_categories.difference(history_categories)
            diversity_ratio = len(new_categories) / len(rec_categories) if rec_categories else 0
            results['category_diversity'].append(diversity_ratio)
    
    # Calculate statistics
    stats = {
        'similarity_score': {
            'mean': np.mean(results['similarity_scores']),
            'min': min(results['similarity_scores']),
            'max': max(results['similarity_scores']),
            'median': np.median(results['similarity_scores'])
        },
        'avg_cluster_distribution_diff': np.mean(results['cluster_distributions']),
        'avg_price_consistency': np.mean(results['price_consistency']),
        'avg_category_diversity': np.mean(results['category_diversity'])
    }
    
    return results, stats

def generate_visualizations(item_results, user_results, output_dir):
    """Generate visualizations of the evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for similarity scores
    plt.figure(figsize=(10, 6))
    plt.hist(item_results['similarity_scores'], bins=20, alpha=0.5, label='Item-based')
    plt.hist(user_results['similarity_scores'], bins=20, alpha=0.5, label='User-based')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'similarity_scores.png'))
    
    # Create figure for cluster matching (item-based)
    plt.figure(figsize=(8, 6))
    plt.bar(['Same Cluster', 'Different Cluster'], 
            [item_results['same_cluster'], item_results['different_cluster']])
    plt.ylabel('Count')
    plt.title('Cluster Matching in Item-based Recommendations')
    plt.savefig(os.path.join(output_dir, 'cluster_match.png'))
    
    # Create figure for price differences
    plt.figure(figsize=(8, 6))
    plt.hist(item_results['price_differences'], bins=10, alpha=0.5, label='Item-based')
    if user_results['price_consistency']:
        plt.hist(user_results['price_consistency'], bins=10, alpha=0.5, label='User-based')
    plt.xlabel('Price Difference')
    plt.ylabel('Count')
    plt.title('Distribution of Price Differences')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'price_differences.png'))
    
    # Create figure for category overlap
    plt.figure(figsize=(8, 6))
    plt.hist(item_results['category_overlap'], bins=10, alpha=0.7)
    plt.xlabel('Category Overlap Ratio')
    plt.ylabel('Count')
    plt.title('Distribution of Category Overlap in Item-based Recommendations')
    plt.savefig(os.path.join(output_dir, 'category_overlap.png'))
    
    # Create figure for category diversity in user recommendations
    if user_results['category_diversity']:
        plt.figure(figsize=(8, 6))
        plt.hist(user_results['category_diversity'], bins=10, alpha=0.7)
        plt.xlabel('Category Diversity Ratio')
        plt.ylabel('Count')
        plt.title('Distribution of Category Diversity in User-based Recommendations')
        plt.savefig(os.path.join(output_dir, 'category_diversity.png'))

def main():
    """Main evaluation function"""
    # Create output directory
    output_dir = "data/evaluation/content_based"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading recommendation data...")
    item_recommendations, user_recommendations = load_recommendations()
    
    print("Loading business data...")
    business_data, business_lookup = load_business_data()
    
    print("\nEvaluating item-based recommendations...")
    item_results, item_stats = evaluate_item_recommendations(item_recommendations, business_lookup)
    
    print("\nEvaluating user-based recommendations...")
    user_results, user_stats = evaluate_user_recommendations(user_recommendations, business_lookup)
    
    # Print statistics
    print("\n===== Item-based Recommendation Statistics =====")
    print(f"Similarity Score: Mean = {item_stats['similarity_score']['mean']:.4f}, "
          f"Min = {item_stats['similarity_score']['min']:.4f}, "
          f"Max = {item_stats['similarity_score']['max']:.4f}")
    print(f"Cluster Match Rate: {item_stats['cluster_match_rate']:.2%}")
    print(f"Average Cluster Distance: {item_stats['avg_cluster_distance']:.2f}")
    print(f"Average Price Difference: {item_stats['avg_price_difference']:.4f}")
    print(f"Average Category Overlap: {item_stats['avg_category_overlap']:.2%}")
    
    print("\n===== User-based Recommendation Statistics =====")
    print(f"Similarity Score: Mean = {user_stats['similarity_score']['mean']:.4f}, "
          f"Min = {user_stats['similarity_score']['min']:.4f}, "
          f"Max = {user_stats['similarity_score']['max']:.4f}")
    print(f"Average Cluster Distribution Difference: {user_stats['avg_cluster_distribution_diff']:.4f}")
    print(f"Average Price Consistency: {user_stats['avg_price_consistency']:.4f}")
    print(f"Average Category Diversity: {user_stats['avg_category_diversity']:.2%}")
    
    # Generate visualizations
    print("\nGenerating evaluation visualizations...")
    generate_visualizations(item_results, user_results, output_dir)
    
    # Save statistics to file
    stats = {
        'item_based': item_stats,
        'user_based': user_stats
    }
    
    stats_path = os.path.join(output_dir, 'recommendation_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")
    
if __name__ == "__main__":
    main() 