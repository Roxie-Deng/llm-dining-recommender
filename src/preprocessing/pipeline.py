from .category_processor import CategoryProcessor
from .category_filter import CategoryFilter
from .business_filter import BusinessFilter
from .hard_condition_filter import hard_condition_filter
from .process_business_clusters import run as process_business_clusters_run
from .sampler import stratified_sample_by_cluster
from ..utils.config import load_config, get_path
import json
import os

def main():
    """
    Simplified preprocessing pipeline: category processing, business filtering, hard condition filtering, clustering, and stratified sampling.
    1. Category processing
    2. Business filtering with refined categories (in-memory)
    3. Hard condition filtering (state, stars, review_count, etc.)
    4. Load filtered data to memory for clustering and simplification
    5. Perform clustering and create simplified entries
    6. Stratified sampling by cluster
    7. Save final sampled result
    """
    config = load_config()

    # Step 1: Category Processing
    print("Step 1: Processing categories...")
    category_processor = CategoryProcessor(config)
    category_processor.process_categories()
    
    # Step 2: Business Filtering with refined categories (in-memory)
    print("Step 2: Filtering businesses with refined categories...")
    business_filter = BusinessFilter(config)
    
    # Get refined categories in memory
    category_filter = CategoryFilter(config)
    refined_categories = category_filter.load_meaningful_categories()
    exclude_categories = category_filter.load_exclude_categories()
    
    print(f"Using {len(refined_categories)} refined categories")
    print(f"Excluding {len(exclude_categories)} categories")
    
    # Filter businesses using refined categories
    business_filter.filter_dining_businesses_with_refined_categories(refined_categories, exclude_categories)

    # Step 3: Hard condition filtering
    print("Step 3: Applying hard condition filters...")
    input_path = get_path(config, "paths", "processed", "dining_businesses")
    filtered_path = "data/processed/dining_related_businesses_filtered.json"
    hard_condition_filter(
        input_path=input_path,
        output_path=filtered_path,
        state="PA",
        is_open=1,
        min_stars=3.0,
        min_review_count=10
    )

    # Step 4: Load filtered businesses for clustering
    with open(filtered_path, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    print(f"Loaded {len(filtered_data)} filtered businesses for clustering.")

    # Step 5: Clustering and simplification (in-memory)
    print("Step 5: Performing clustering and simplification...")
    clustered_data = process_business_clusters_run(filtered_data, config, refined_categories, exclude_categories)

    # Step 6: Stratified sampling by cluster
    print("Step 6: Performing stratified sampling by cluster...")
    sample_size = config.get("feature_engineering", {}).get("sampling", {}).get("size", 1000)
    final_sample = stratified_sample_by_cluster(clustered_data, "cluster", sample_size, None)

    # Step 7: Save the final result
    output_sample_path = "data/processed/stratified_sample.json"
    os.makedirs(os.path.dirname(output_sample_path), exist_ok=True)
    with open(output_sample_path, 'w', encoding='utf-8') as f:
        json.dump(final_sample, f, indent=2, ensure_ascii=False)
    
    print(f"\nPipeline complete. Final sampled data saved to: {output_sample_path}")
    print(f"Category filtering applied:")
    print(f"  - Refined categories: {len(refined_categories)}")
    print(f"  - Excluded categories: {len(exclude_categories)}")

if __name__ == "__main__":
    main() 