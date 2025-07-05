"""
Vectorization module for business features without review data.
Converts extracted features into numerical vectors for machine learning models.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
from pathlib import Path
import torch

class VectorizerWithoutReview:
    def __init__(self, config_path="configs/data_config.yaml", model_name='all-MiniLM-L6-v2'):
        """
        Initialize the vectorizer with configuration
        
        Args:
            config_path (str): Path to configuration file
            model_name (str): Name of the sentence transformer model
        """
        self.config = self._load_config(config_path)
        self.model_name = model_name
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
    
    def encode_text_features(self, df, text_columns=['description']):
        """
        Encode text features using sentence transformers
        
        Args:
            df (DataFrame): DataFrame with text columns
            text_columns (list): List of text column names to encode
            
        Returns:
            dict: Dictionary with encoded text features
        """
        self.load_model()
        
        text_embeddings = {}
        
        for column in text_columns:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame")
                continue
                
            print(f"Encoding text column: {column}")
            
            # Handle missing values
            texts = df[column].fillna("").tolist()
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts, 
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            text_embeddings[f"{column}_embedding"] = embeddings
            
        return text_embeddings
    
    def encode_categorical_features(self, df, categorical_columns=['category_cluster', 'Ambience']):
        """
        Encode categorical features using label encoding
        
        Args:
            df (DataFrame): DataFrame with categorical columns
            categorical_columns (list): List of categorical column names to encode
            
        Returns:
            dict: Dictionary with encoded categorical features
        """
        categorical_embeddings = {}
        
        for column in categorical_columns:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame")
                continue
                
            print(f"Encoding categorical column: {column}")
            
            # Create label encoder if not exists
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                
                # Fit on all unique values (including NaN)
                unique_values = df[column].dropna().unique()
                self.label_encoders[column].fit(unique_values)
            
            # Encode values
            encoded_values = df[column].map(
                lambda x: self.label_encoders[column].transform([x])[0] 
                if pd.notna(x) and x in self.label_encoders[column].classes_ 
                else -1
            ).values
            
            categorical_embeddings[f"{column}_encoded"] = encoded_values
            
        return categorical_embeddings
    
    def encode_boolean_features(self, df, boolean_columns=['OutdoorSeating', 'RestaurantsReservations']):
        """
        Encode boolean features as binary values
        
        Args:
            df (DataFrame): DataFrame with boolean columns
            boolean_columns (list): List of boolean column names to encode
            
        Returns:
            dict: Dictionary with encoded boolean features
        """
        boolean_embeddings = {}
        
        for column in boolean_columns:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame")
                continue
                
            print(f"Encoding boolean column: {column}")
            
            # Convert to binary (True=1, False=0, NaN=-1)
            encoded_values = df[column].map({True: 1, False: 0}).fillna(-1).values
            boolean_embeddings[f"{column}_binary"] = encoded_values
            
        return boolean_embeddings
    
    def normalize_numerical_features(self, df, numerical_columns=['stars', 'review_count', 'normalized_price']):
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df (DataFrame): DataFrame with numerical columns
            numerical_columns (list): List of numerical column names to normalize
            
        Returns:
            dict: Dictionary with normalized numerical features
        """
        numerical_features = {}
        
        for column in numerical_columns:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame")
                continue
                
            print(f"Normalizing numerical column: {column}")
            
            # Extract values and handle missing values
            values = df[column].fillna(df[column].median()).values.reshape(-1, 1)
            
            # Fit scaler if not already fitted
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                self.scaler.fit(values)
            
            # Transform values
            normalized_values = self.scaler.transform(values).flatten()
            numerical_features[f"{column}_normalized"] = normalized_values
            
        return numerical_features
    
    def create_tfidf_features(self, df, text_column='categories', max_features=100):
        """
        Create TF-IDF features from text columns
        
        Args:
            df (DataFrame): DataFrame with text column
            text_column (str): Name of text column to process
            max_features (int): Maximum number of features to extract
            
        Returns:
            dict: Dictionary with TF-IDF features
        """
        if text_column not in df.columns:
            print(f"Warning: Column '{text_column}' not found in DataFrame")
            return {}
            
        print(f"Creating TF-IDF features from column: {text_column}")
        
        # Initialize TF-IDF vectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit on the text data
            texts = df[text_column].fillna("").tolist()
            self.tfidf_vectorizer.fit(texts)
        
        # Transform the text data
        texts = df[text_column].fillna("").tolist()
        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        return {f"{text_column}_tfidf": tfidf_features}
    
    def combine_features(self, feature_dicts):
        """
        Combine all feature vectors into a single feature matrix
        
        Args:
            feature_dicts (list): List of feature dictionaries
            
        Returns:
            tuple: (feature_matrix, feature_names)
        """
        all_features = []
        feature_names = []
        
        for feature_dict in feature_dicts:
            for name, features in feature_dict.items():
                if isinstance(features, np.ndarray):
                    if features.ndim == 1:
                        features = features.reshape(-1, 1)
                    all_features.append(features)
                    feature_names.append(name)
        
        if all_features:
            feature_matrix = np.hstack(all_features)
            return feature_matrix, feature_names
        else:
            return np.array([]), []
    
    def vectorize_businesses(self, df):
        """
        Main vectorization pipeline for business data without reviews
        
        Args:
            df (DataFrame): DataFrame with extracted features
            
        Returns:
            tuple: (feature_matrix, feature_names, business_ids)
        """
        print("Starting business vectorization (without reviews)...")
        
        # Step 1: Encode text features (only description, no review_tip)
        print("Step 1: Encoding text features...")
        text_features = self.encode_text_features(df, text_columns=['description'])
        
        # Step 2: Encode categorical features
        print("Step 2: Encoding categorical features...")
        categorical_features = self.encode_categorical_features(df)
        
        # Step 3: Encode boolean features
        print("Step 3: Encoding boolean features...")
        boolean_features = self.encode_boolean_features(df)
        
        # Step 4: Normalize numerical features
        print("Step 4: Normalizing numerical features...")
        numerical_features = self.normalize_numerical_features(df)
        
        # Step 5: Create TF-IDF features
        print("Step 5: Creating TF-IDF features...")
        tfidf_features = self.create_tfidf_features(df)
        
        # Step 6: Combine all features
        print("Step 6: Combining features...")
        all_feature_dicts = [
            text_features,
            categorical_features,
            boolean_features,
            numerical_features,
            tfidf_features
        ]
        
        feature_matrix, feature_names = self.combine_features(all_feature_dicts)
        
        # Get business IDs robustly
        if 'business_id' in df.columns:
            business_ids = df['business_id'].tolist()
        else:
            print("Warning: 'business_id' column not found in input data. Using index as IDs.")
            business_ids = list(df.index)
        
        print(f"Vectorization complete! Feature matrix shape: {feature_matrix.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        return feature_matrix, feature_names, business_ids
    
    def save_vectorizer(self, output_path):
        """
        Save the trained vectorizer components
        
        Args:
            output_path (str): Path to save the vectorizer
        """
        vectorizer_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'model_name': self.model_name
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(vectorizer_data, f)
        
        print(f"Vectorizer saved to: {output_path}")
    
    def load_vectorizer(self, input_path):
        """
        Load a trained vectorizer
        
        Args:
            input_path (str): Path to load the vectorizer from
        """
        with open(input_path, 'rb') as f:
            vectorizer_data = pickle.load(f)
        
        self.label_encoders = vectorizer_data['label_encoders']
        self.scaler = vectorizer_data['scaler']
        self.tfidf_vectorizer = vectorizer_data['tfidf_vectorizer']
        self.model_name = vectorizer_data['model_name']
        
        print(f"Vectorizer loaded from: {input_path}")

def create_business_embeddings_without_review(df, config_path="configs/data_config.yaml", output_path=None):
    """
    Convenience function to create business embeddings without review data
    
    Args:
        df (DataFrame): DataFrame with business features
        config_path (str): Path to configuration file
        output_path (str, optional): Path to save embeddings
        
    Returns:
        tuple: (embeddings, business_ids, feature_names)
    """
    vectorizer = VectorizerWithoutReview(config_path)
    embeddings, feature_names, business_ids = vectorizer.vectorize_businesses(df)
    
    if output_path:
        # Save embeddings
        np.save(f"{output_path}_embeddings.npy", embeddings)
        
        # Save metadata
        metadata = {
            'business_ids': business_ids,  # already a list
            'feature_names': feature_names
        }
        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save vectorizer
        vectorizer.save_vectorizer(f"{output_path}_vectorizer.pkl")
        
        print(f"Embeddings saved to: {output_path}")
    
    return embeddings, business_ids, feature_names 