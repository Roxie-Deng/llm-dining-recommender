import argparse
import os
from .content_based_recommender import ContentBasedRecommender
from .evaluate_content_recommendations import evaluate_item_recommendations


def main():
    """
    Main script to run item-item similarity recommendation and evaluation.
    Parses command line arguments, generates recommendations, and evaluates them.
    """
    parser = argparse.ArgumentParser(description="Item-item similarity recommendation and evaluation")
    parser.add_argument('--features', type=str, required=True, help='Path to the .npy features file')
    parser.add_argument('--metadata', type=str, required=True, help='Path to the metadata .json file')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top similar items to recommend')
    # Recommended: output_dir like results/with_review/ or results/without_review/
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results (e.g., results/with_review/)')
    args = parser.parse_args()

    # Prepare output paths
    os.makedirs(args.output_dir, exist_ok=True)
    rec_path = os.path.join(args.output_dir, 'item_recommendations.json')
    eval_path = os.path.join(args.output_dir, 'recommendation_stats.json')

    # Run recommender
    print("Loading features and metadata...")
    recommender = ContentBasedRecommender(args.features, args.metadata)
    print("Computing item-item similarity and generating recommendations...")
    recommendations = recommender.get_top_n_similar(top_n=args.top_n)
    recommender.save_recommendations(recommendations, rec_path)
    print(f"Recommendations saved to {rec_path}")

    # Run evaluation
    print("Evaluating recommendations...")
    evaluate_item_recommendations(rec_path, args.metadata, eval_path)

if __name__ == '__main__':
    main() 