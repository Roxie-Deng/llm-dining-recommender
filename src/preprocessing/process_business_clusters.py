import json
from ..utils.config import get_path

def run(businesses, config, refined_categories: set, exclude_categories: set):
    """
    Map business categories to clusters and generate simplified business entries.
    This version also cleans categories and adds the business name.
    """
    clusters_path = "data/processed/hybrid_clusters.json"

    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)
        
    category_to_cluster = {}
    for cluster_id, categories in clusters.items():
        for category in categories:
            category_to_cluster[category] = int(cluster_id)

    def process_business(business, mapping):
        # Clean categories first
        business_cats_raw = business.get("categories", "")
        if business_cats_raw is None:
            business_cats_raw = ""

        business_cats = set(cat.strip() for cat in business_cats_raw.split(","))
        
        # Filter categories to only include refined ones and exclude unwanted ones
        cleaned_cats_set = business_cats & refined_categories
        cleaned_cats_set = cleaned_cats_set - exclude_categories
        
        # Determine cluster from the original (uncleaned) categories for better mapping
        cluster_counts = {}
        for category in business_cats:
            if category in mapping:
                cluster_id = mapping[category]
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        main_cluster = None
        if cluster_counts:
            main_cluster = max(cluster_counts.items(), key=lambda x: x[1])[0]
        
        # Safely get attributes, handling missing fields
        attributes = business.get("attributes", {})
        if attributes is None:
            attributes = {}
        
        # Create simplified entry with essential fields
        simplified_entry = {
            "business_id": business["business_id"],
            "name": business.get("name"),
            "cluster": main_cluster,
            "categories": ", ".join(sorted(cleaned_cats_set)),
            "stars": business["stars"],
            "review_count": business["review_count"],
            "RestaurantsPriceRange2": attributes.get("RestaurantsPriceRange2"),
            "OutdoorSeating": attributes.get("OutdoorSeating"),
            "Ambience": attributes.get("Ambience"),
            "RestaurantsReservations": attributes.get("RestaurantsReservations")
        }
        
        return simplified_entry

    results = [process_business(business, category_to_cluster) for business in businesses]
    
    print("In-memory business clustering and simplification complete.")
    return results 