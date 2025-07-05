"""
Data loading module for the recommendation system.
Contains functions for loading and validating raw data.
"""

import pandas as pd
import json
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_json_data(file_path: str) -> pd.DataFrame:
    """
    Load JSON data into a pandas DataFrame.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        DataFrame containing the loaded data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def load_raw_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all raw data files specified in the config.
    
    Args:
        config: Configuration dictionary containing file paths
        
    Returns:
        Tuple of (business_df, review_df, user_df)
    """
    paths = config['paths']['raw_data']
    
    business_df = load_json_data(paths['business'])
    review_df = load_json_data(paths['review'])
    user_df = load_json_data(paths['user'])
    
    return business_df, review_df, user_df

def validate_data(df: pd.DataFrame, data_type: str, config: Dict[str, Any]) -> bool:
    """
    Validate loaded data against configuration parameters.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ('business', 'review', or 'user')
        config: Configuration dictionary containing validation parameters
        
    Returns:
        Boolean indicating if validation passed
    """
    params = config['processing'][data_type]
    
    if data_type == 'review':
        # Validate review ratings
        if not (df['stars'].between(params['min_rating'], params['max_rating'])).all():
            return False
            
    elif data_type == 'business':
        # Validate business review counts
        if not (df['review_count'] >= params['min_review_count']).all():
            return False
            
    elif data_type == 'user':
        # Validate user review counts and friends
        if not (df['review_count'] >= params['min_review_count']).all():
            return False
        if not (df['friends_count'] >= params['min_friends']).all():
            return False
            
    return True 