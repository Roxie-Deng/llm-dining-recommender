import json
from typing import List, Dict, Any
import os

class BusinessFilter:
    def __init__(self, config: dict):
        self.config = config
        
    def should_exclude_business(self, business: Dict[str, Any]) -> bool:
        """Check if a business should be excluded based on categories and name."""
        # Get business categories
        categories = set(cat.strip() for cat in business.get("categories", "").split(","))
        name = business.get("name", "").lower()
        
        # Check for exclusion keywords in categories
        exclude_categories = set(self.config['categories']['exclude_keywords'])
        if categories & exclude_categories:
            # If it has both restaurant and exclude categories, keep it
            restaurant_categories = set(self.config['categories']['restaurant_keywords'])
            if not (categories & restaurant_categories):
                return True
        
        # Check for exclusion keywords in name
        exclude_name_keywords = set(self.config['categories']['exclude_name_keywords'])
        if any(keyword in name for keyword in exclude_name_keywords):
            return True
            
        return False

    def filter_dining_businesses_with_refined_categories(self, refined_categories: set, exclude_categories: set) -> List[Dict[str, Any]]:
        """
        Filter dining-related businesses using refined categories in memory.
        This version only filters businesses but does not modify their data.
        
        Args:
            refined_categories: Set of refined categories to keep
            exclude_categories: Set of categories to exclude (used for checking, not direct filtering here)
        """
        dining_businesses = []
        total_records = 0
        
        # Process business file
        with open(self.config['paths']['raw']['business'], 'r', encoding='utf-8') as f:
            for line in f:
                total_records += 1
                business = json.loads(line.strip())
                
                # Skip if no categories
                if not business.get("categories"):
                    continue
                    
                # Get business categories
                business_cats = set(cat.strip() for cat in business["categories"].split(","))
                
                # Check if business has any refined categories
                if not (refined_categories & business_cats):
                    continue
                    
                # Check for exclusion criteria (e.g. name based exclusion)
                if self.should_exclude_business(business):
                    continue

                # Add the business with its original data
                dining_businesses.append(business)
        
        # Save filtered businesses
        with open(self.config['paths']['processed']['businesses']['dining'], 'w', encoding='utf-8') as f:
            json.dump(dining_businesses, f, indent=2, ensure_ascii=False)
        
        print(f"Filtered {len(dining_businesses)} businesses from {total_records} total records")
        
        return total_records, len(dining_businesses)

    def filter_dining_businesses(self, refined_categories_path: str = None) -> List[Dict[str, Any]]:
        """
        Filter dining-related businesses based on refined categories and name.
        
        Args:
            refined_categories_path: Path to refined categories file. If None, uses original dining categories.
        """
        # Load categories - use refined categories if available, otherwise fall back to original
        if refined_categories_path and os.path.exists(refined_categories_path):
            with open(refined_categories_path, 'r', encoding='utf-8') as f:
                dining_categories = set(json.load(f))
            print(f"Using refined categories: {len(dining_categories)} categories")
        else:
            # Fall back to original dining categories
            with open(self.config['paths']['processed']['categories']['dining'], 'r', encoding='utf-8') as f:
                dining_categories = set(json.load(f))
                print(f"Using original dining categories: {len(dining_categories)} categories")
        
        dining_businesses = []
        total_records = 0
        
        # Process business file
        with open(self.config['paths']['raw']['business'], 'r', encoding='utf-8') as f:
            for line in f:
                total_records += 1
                business = json.loads(line.strip())
                
                # Skip if no categories
                if not business.get("categories"):
                    continue
                    
                # Check if business has dining categories
                business_cats = set(cat.strip() for cat in business["categories"].split(","))
                if not (dining_categories & business_cats):
                    continue
                    
                # Check for exclusion criteria
                if self.should_exclude_business(business):
                    continue
                    
                # Filter categories to only include refined ones
                if refined_categories_path and os.path.exists(refined_categories_path):
                    filtered_cats = business_cats & dining_categories
                    if filtered_cats:  # Only keep if there are still categories after filtering
                        business["categories"] = ", ".join(sorted(filtered_cats))
                    else:
                        continue  # Skip if no categories remain after filtering
                
                dining_businesses.append(business)
        
        # Save filtered businesses
        with open(self.config['paths']['processed']['businesses']['dining'], 'w', encoding='utf-8') as f:
            json.dump(dining_businesses, f, indent=2, ensure_ascii=False)
        
        return total_records, len(dining_businesses) 