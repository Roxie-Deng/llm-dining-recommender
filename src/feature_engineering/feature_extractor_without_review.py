"""
Feature extraction module for business data without review processing.
Focuses on business attributes, categories, and basic information.
"""

import json
import yaml
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

def load_config(config_path="configs/data_config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_attributes(attr_str):
    """Parse attributes from string or return as is if already a dictionary"""
    if not attr_str or attr_str == "None":
        return {}
        
    if isinstance(attr_str, dict):
        return attr_str
        
    try:
        return json.loads(attr_str)
    except:
        try:
            return ast.literal_eval(attr_str)
        except:
            return {}

class FeatureExtractorWithoutReview:
    def __init__(self, config_path="configs/data_config.yaml"):
        """Initialize feature extractor with configuration"""
        self.config = load_config(config_path)
        
    def process_attributes(self, businesses_df):
        """
        Process business attributes according to configuration:
        1. Retain only specified attributes
        2. Handle missing values according to imputation strategy
        3. Format attributes to the correct type
        4. Normalize price range to 0-1 scale
        Supports both flat columns and nested 'attributes' field.
        """
        retained_attrs = self.config['feature_engineering']['retained_attributes']
        attr_config = self.config['feature_engineering']['attribute_config']
        impute_strategy = self.config['feature_engineering']['processing']['impute_strategy']
        
        processed_df = businesses_df.copy()
        has_attributes = 'attributes' in processed_df.columns
        
        # Ensure all retained attributes are present as columns
        for attr in retained_attrs:
            if attr not in processed_df.columns:
                processed_df[attr] = None
        
        # Initialize normalized_price column
        processed_df['normalized_price'] = None
        
        for idx, row in processed_df.iterrows():
            # Support both nested 'attributes' and flat columns
            if has_attributes:
                attrs = parse_attributes(row['attributes'])
            else:
                attrs = {attr: row.get(attr) for attr in retained_attrs}
            
            # Process each retained attribute
            for attr_name in retained_attrs:
                attr_value = attrs.get(attr_name)
                attr_type = attr_config[attr_name]['type']
                
                # Handle attribute based on type
                if attr_type == "numeric":
                    if attr_value is None or attr_value == "None" or pd.isna(attr_value):
                        processed_df.at[idx, attr_name] = None
                    else:
                        processed_df.at[idx, attr_name] = int(attr_value)
                        
                elif attr_type == "boolean":
                    true_values = attr_config[attr_name]['true_values']
                    
                    if attr_value is None or attr_value == "None":
                        processed_df.at[idx, attr_name] = None
                    else:
                        processed_df.at[idx, attr_name] = str(attr_value) in true_values
                        
                elif attr_type == "string" and attr_name == "Ambience":
                    if attr_value is None or attr_value == "None":
                        processed_df.at[idx, attr_name] = None
                    elif isinstance(attr_value, str) and ("{" in attr_value or "'" in attr_value):
                        try:
                            ambience_dict = ast.literal_eval(attr_value)
                            cleaned_ambience = sorted([k for k, v in ambience_dict.items() if v is True])
                            processed_df.at[idx, attr_name] = ", ".join(cleaned_ambience) if cleaned_ambience else None
                        except:
                            processed_df.at[idx, attr_name] = None
                    else:
                        processed_df.at[idx, attr_name] = attr_value
        
        # Impute missing values
        for attr in retained_attrs:
            strategy = impute_strategy[attr]
            if strategy == "median" and attr_config[attr]['type'] == "numeric":
                median_value = processed_df[attr].dropna().median()
                processed_df[attr] = processed_df[attr].fillna(median_value)
            elif strategy == "False" and attr_config[attr]['type'] == "boolean":
                processed_df[attr] = processed_df[attr].fillna(False)
            elif strategy != "leave_null":
                processed_df[attr] = processed_df[attr].fillna(strategy)
        
        # Normalize price
        if "RestaurantsPriceRange2" in processed_df.columns:
            min_val = 1  # Minimum price value
            max_val = 4  # Maximum price value
            
            # Normalize to 0-1 range
            processed_df['normalized_price'] = (processed_df['RestaurantsPriceRange2'] - min_val) / (max_val - min_val)
            
            # Handle potential out-of-range values
            processed_df['normalized_price'] = processed_df['normalized_price'].clip(0, 1)
        
        return processed_df

    def map_businesses_to_clusters(self, businesses_df, clusters_path):
        """
        Map businesses to their cuisine clusters based on their categories.
        If 'cluster' column already exists, skip mapping.
        """
        if 'cluster' in businesses_df.columns:
            print("'cluster' column already exists, skipping cluster mapping.")
            return businesses_df
        
        with open(clusters_path, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)
        
        # Create category to cluster mapping dictionary
        category_to_cluster = {}
        for cluster_id, categories in clusters_data.items():
            for category in categories:
                category_to_cluster[category] = int(cluster_id)
        
        # Function to determine the cluster for a business based on its categories
        def get_cluster(row):
            # Parse categories
            if isinstance(row['categories'], str):
                # Split categories string into list
                categories = [cat.strip() for cat in row['categories'].split(',')]
            elif isinstance(row['categories'], list):
                categories = row['categories']
            else:
                return 0  # Default cluster if no valid categories
            
            # Check each category for a match in our clusters
            matching_clusters = []
            for category in categories:
                if category in category_to_cluster:
                    matching_clusters.append(category_to_cluster[category])
            
            # Return the most common cluster if any matches found
            if matching_clusters:
                # Use Counter to find the most common cluster
                return Counter(matching_clusters).most_common(1)[0][0]
            
            return 0  # Default cluster if no matches
        
        # Apply the mapping function to create the category_cluster field
        result_df = businesses_df.copy()
        result_df['category_cluster'] = result_df.apply(get_cluster, axis=1)
        
        # Print summary of cluster mappings
        cluster_counts = result_df['category_cluster'].value_counts().sort_index()
        print("Businesses mapped to cuisine clusters:")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(result_df)) * 100
            print(f"  Cluster {cluster_id}: {count} businesses ({percentage:.2f}%)")
        
        return result_df

    def generate_business_descriptions(self, df):
        """
        Generate natural language descriptions for businesses based on their attributes
        
        Args:
            df (DataFrame): DataFrame with business data including name, normalized_price, 
                           Ambience, and category_cluster
        
        Returns:
            DataFrame: DataFrame with added 'description' field
        """
        # Price tier mapping
        price_tiers = {
            (0, 0.25): "budget-friendly",
            (0.25, 0.5): "affordable",
            (0.5, 0.75): "expensive",
            (0.75, 1.0): "luxury"
        }
        
        # Cluster labels mapping
        cluster_labels = {
            1: "Asian Cuisines",
            2: "European Cuisines",
            3: "African & Middle Eastern Cuisines",
            4: "American & Latin Cuisines",
            5: "Bars & Lounges",
            6: "Breakfast & Fast Food",
            7: "Specialty Dining",
            8: "Casual Dining",
            9: "Beverages & Desserts"
        }
        
        # Function to map normalized price to price tier
        def get_price_tier(price):
            if pd.isna(price):
                return "mid-range"  # Default if price is missing
            
            for (lower, upper), tier in price_tiers.items():
                if lower <= price < upper or (price == 1.0 and upper == 1.0):
                    return tier
            
            return "mid-range"  # Default fallback
        
        # Function to get establishment type based on cluster
        def get_establishment_type(cluster):
            if cluster == 5:
                return "bar"
            elif cluster in [6, 9]:
                return "eatery"
            else:
                return "restaurant"
        
        # Function to generate description for a single business
        def create_description(row):
            name = row['name']
            
            # Get price tier
            price_tier = get_price_tier(row.get('normalized_price'))
            
            # Get establishment type and category information
            cluster = row.get('category_cluster', 0)
            establishment = get_establishment_type(cluster)
            category_label = cluster_labels.get(cluster, "General Dining")
            
            # Get original category for more specific description if available
            specific_category = ""
            if 'categories' in row and row['categories']:
                if isinstance(row['categories'], str):
                    categories = [cat.strip() for cat in row['categories'].split(',')]
                    if categories:
                        specific_category = categories[0]  # Use first category
            
            # Create category phrase
            if specific_category:
                category_phrase = f"categorised under {specific_category}"
            else:
                category_phrase = f"offering {category_label}"
            
            # Check for ambience
            ambience = row.get('Ambience')
            ambience_phrase = ""
            if ambience and not pd.isna(ambience) and ambience != "Unknown":
                ambience_phrase = f", with {ambience} ambience"
            
            # Construct description
            description = f"{name}: {price_tier} {establishment} {category_phrase}{ambience_phrase}"
            
            return description
        
        # Apply description generation to all rows
        result_df = df.copy()
        result_df['description'] = result_df.apply(create_description, axis=1)
        
        return result_df

    def extract_features(self, businesses_df):
        """
        Main feature extraction pipeline without review processing
        
        Args:
            businesses_df (DataFrame): Raw business data
        
        Returns:
            DataFrame: DataFrame with extracted features
        """
        print("Starting feature extraction (without reviews)...")
        
        # Step 1: Process attributes
        print("Step 1: Processing attributes...")
        df = self.process_attributes(businesses_df)
        
        # Step 2: Map to clusters
        print("Step 2: Mapping to clusters...")
        clusters_path = self.config['paths']['processed']['hybrid_clusters']
        # Only map clusters if 'cluster' column does not exist
        if 'cluster' in df.columns:
            print("'cluster' column already exists, skipping cluster mapping.")
            df['category_cluster'] = df['cluster']
        else:
            df = self.map_businesses_to_clusters(df, clusters_path)
        
        # Step 3: Generate descriptions
        print("Step 3: Generating descriptions...")
        df = self.generate_business_descriptions(df)
        
        print("Feature extraction complete (without reviews)!")
        return df 