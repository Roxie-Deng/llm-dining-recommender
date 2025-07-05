import json
from typing import List, Set, Dict, Any
from collections import Counter

class CategoryFilter:
    def __init__(self, config: dict):
        self.config = config
        
    def load_meaningful_categories(self) -> Set[str]:
        """Load meaningful categories from config."""
        meaningful_categories = set()
        
        # Get meaningful categories from config
        category_config = self.config.get('category_filtering', {}).get('meaningful_categories', {})
        
        for category_type, categories in category_config.items():
            meaningful_categories.update(categories)
            
        return meaningful_categories
    
    def load_exclude_categories(self) -> Set[str]:
        """Load categories to exclude from config."""
        exclude_config = self.config.get('category_structuring', {}).get('filtering', {}).get('exclude_categories', [])
        return set(exclude_config)
    
    def filter_categories(self, categories: List[str], min_count: int = 10) -> List[str]:
        """
        Filter categories by removing:
        1. Categories unrelated to dining (e.g., Golf, Gym)
        2. Broad umbrella terms (e.g., Food, Restaurant)
        3. Categories with insufficient business count
        
        Returns refined list of specific and informative categories.
        """
        # Load configuration
        meaningful_categories = self.load_meaningful_categories()
        exclude_categories = self.load_exclude_categories()
        
        # Filter categories
        filtered_categories = []
        
        for category in categories:
            category = category.strip()
            
            # Skip if category is in exclude list
            if category in exclude_categories:
                continue
                
            # Keep if category is in meaningful list
            if category in meaningful_categories:
                filtered_categories.append(category)
                continue
                
            # Additional filtering logic for edge cases
            # Skip very broad terms that might have slipped through
            broad_terms = {"Food", "Restaurant", "Restaurants", "Dining", "Cuisine"}
            if category in broad_terms:
                continue
                
            # Skip categories that are clearly not dining-related
            non_dining_keywords = {"golf", "gym", "fitness", "sports", "shopping", "entertainment", 
                                  "nightlife", "health", "medical", "automotive", "real estate"}
            if any(keyword in category.lower() for keyword in non_dining_keywords):
                continue
        
        return sorted(list(set(filtered_categories)))
    
    def get_category_statistics(self, business_file_path: str, max_records: int = None) -> Dict[str, Any]:
        """Get statistics about categories in the dataset."""
        category_counts = Counter()
        total_businesses = 0
        
        with open(business_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_records and line_num >= max_records:
                    break
                    
                business = json.loads(line.strip())
                categories = business.get("categories", "")
                
                if categories:
                    for category in categories.split(","):
                        category = category.strip()
                        if category:
                            category_counts[category] += 1
                
                total_businesses += 1
        
        return {
            "total_businesses": total_businesses,
            "unique_categories": len(category_counts),
            "category_counts": dict(category_counts.most_common()),
            "most_common_categories": category_counts.most_common(20)
        }
    
    def process_and_filter_categories(self, input_file_path: str, output_file_path: str, 
                                    max_records: int = None, min_category_count: int = 10) -> Dict[str, Any]:
        """
        Process categories from business data and apply filtering.
        
        Args:
            input_file_path: Path to business data file
            output_file_path: Path to save filtered categories
            max_records: Maximum number of records to process
            min_category_count: Minimum business count for a category to be included
            
        Returns:
            Dictionary with processing statistics
        """
        print("Loading category statistics...")
        stats = self.get_category_statistics(input_file_path, max_records)
        
        print(f"Found {stats['unique_categories']} unique categories")
        print(f"Processing {stats['total_businesses']} businesses")
        
        # Get all categories
        all_categories = list(stats['category_counts'].keys())
        
        # Apply filtering
        print("Applying category filtering...")
        filtered_categories = self.filter_categories(all_categories, min_category_count)
        
        # Filter by minimum count
        final_categories = []
        for category in filtered_categories:
            if stats['category_counts'].get(category, 0) >= min_category_count:
                final_categories.append(category)
        
        # Save filtered categories
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_categories, f, indent=2, ensure_ascii=False)
        
        # Calculate statistics
        total_businesses_with_filtered_categories = sum(
            stats['category_counts'].get(cat, 0) for cat in final_categories
        )
        
        result_stats = {
            "original_categories": stats['unique_categories'],
            "filtered_categories": len(final_categories),
            "total_businesses": stats['total_businesses'],
            "businesses_with_filtered_categories": total_businesses_with_filtered_categories,
            "reduction_percentage": round((1 - len(final_categories) / stats['unique_categories']) * 100, 2),
            "filtered_categories": final_categories
        }
        
        print(f"Category filtering complete:")
        print(f"  Original categories: {result_stats['original_categories']}")
        print(f"  Filtered categories: {result_stats['filtered_categories']}")
        print(f"  Reduction: {result_stats['reduction_percentage']}%")
        print(f"  Categories saved to: {output_file_path}")
        
        return result_stats 