"""
Vectorization module for business features with review and tip data.
Converts extracted features and multiple text fields into numerical vectors for machine learning models.
Supports segment-wise encoding and pooling for long texts.
All comments and docstrings are in English.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from pathlib import Path
import torch

def segment_encode(text, model, max_tokens=400, pool_method='mean'):
    """
    Encode long text by segmenting and pooling segment embeddings.
    Args:
        text (str): Input text
        model (SentenceTransformer): SentenceTransformer model
        max_tokens (int): Max tokens per segment (approximate by words)
        pool_method (str): Pooling method ('mean' or 'max')
    Returns:
        np.ndarray: Pooled embedding
    """
    words = text.split()
    segments = [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
    if not segments:
        segments = [""]
    embeddings = model.encode(segments, batch_size=8, show_progress_bar=False, convert_to_numpy=True)
    if pool_method == 'mean':
        pooled = np.mean(embeddings, axis=0)
    elif pool_method == 'max':
        pooled = np.max(embeddings, axis=0)
    else:
        raise ValueError(f"Unknown pool_method: {pool_method}")
    return pooled

def create_business_embeddings_with_review(df, config_path="configs/data_config.yaml", output_dir=None):
    """
    Create business embeddings with review_tip, description, and categories text fields.
    Args:
        df (DataFrame): DataFrame with business features and text fields
        config_path (str or dict): Path to config file or config dict
        output_dir (str, optional): Directory to save outputs
    Returns:
        tuple: (embeddings, business_ids, feature_names)
    """
    # Load config
    if isinstance(config_path, str):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config_path
    if output_dir is None:
        output_dir = config['paths']['features_output']['with_review']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Structured features
    feature_columns = config['feature_engineering']['vectorizer']['with_review']['feature_columns']
    business_ids = df['business_id'].tolist()
    X_structured = df[feature_columns].values.astype(float)
    feature_names = list(feature_columns)

    # Text fields
    text_columns = config['feature_engineering']['vectorizer']['with_review'].get('text_columns', ['review_tip', 'description', 'categories'])
    text_vectors = []
    text_feature_names = []

    for text_col in text_columns:
        if text_col == 'review_tip':
            vec_type = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_vectorizer', 'paraphrase-minilm')
            if vec_type in ['paraphrase-minilm', 'bert']:
                model_name = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_bert_model', 'paraphrase-MiniLM-L6-v2')
                pool_method = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_pool_method', 'mean')
                max_tokens = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_max_tokens', 400)
                print(f"Loading SentenceTransformer model for review_tip: {model_name}")
                model = SentenceTransformer(model_name)
                if torch.cuda.is_available():
                    model = model.to('cuda')
                texts = df[text_col].fillna("").tolist()
                X_text = np.stack([
                    segment_encode(text, model, max_tokens=max_tokens, pool_method=pool_method)
                    for text in texts
                ])
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_bert_{i}" for i in range(X_text.shape[1])]
            elif vec_type == 'tfidf':
                max_features = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_max_features', 256)
                vectorizer = TfidfVectorizer(max_features=max_features)
                texts = df[text_col].fillna("").tolist()
                X_text = vectorizer.fit_transform(texts).toarray()
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_tfidf_{i}" for i in range(X_text.shape[1])]
            else:
                raise ValueError(f"Unknown review_tip vectorizer: {vec_type}")
        elif text_col == 'description':
            vec_type = config['feature_engineering']['vectorizer']['with_review'].get('description_vectorizer', 'bert')
            if vec_type == 'bert':
                model_name = config['feature_engineering']['vectorizer']['with_review'].get('description_bert_model', 'all-MiniLM-L6-v2')
                print(f"Loading SentenceTransformer model for description: {model_name}")
                model = SentenceTransformer(model_name)
                if torch.cuda.is_available():
                    model = model.to('cuda')
                texts = df[text_col].fillna("").tolist()
                X_text = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_bert_{i}" for i in range(X_text.shape[1])]
            elif vec_type == 'tfidf':
                max_features = config['feature_engineering']['vectorizer']['with_review'].get('description_max_features', 256)
                vectorizer = TfidfVectorizer(max_features=max_features)
                texts = df[text_col].fillna("").tolist()
                X_text = vectorizer.fit_transform(texts).toarray()
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_tfidf_{i}" for i in range(X_text.shape[1])]
            else:
                raise ValueError(f"Unknown description vectorizer: {vec_type}")
        elif text_col == 'categories':
            vec_type = config['feature_engineering']['vectorizer']['with_review'].get('categories_vectorizer', 'tfidf')
            if vec_type == 'tfidf':
                max_features = config['feature_engineering']['vectorizer']['with_review'].get('categories_max_features', 256)
                vectorizer = TfidfVectorizer(max_features=max_features)
                texts = df[text_col].fillna("").tolist()
                X_text = vectorizer.fit_transform(texts).toarray()
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_tfidf_{i}" for i in range(X_text.shape[1])]
            else:
                raise ValueError(f"Unknown categories vectorizer: {vec_type}")
        else:
            raise ValueError(f"Unknown text column: {text_col}")

    # Concatenate all features
    all_vectors = [X_structured] + text_vectors
    all_feature_names = feature_names + text_feature_names
    embeddings = np.concatenate(all_vectors, axis=1)

    # Save embeddings and metadata
    embeddings_path = config['feature_engineering']['output']['with_review']['business_embeddings_with_text']
    metadata_path = config['feature_engineering']['output']['with_review']['business_embeddings_with_text_metadata']
    np.save(embeddings_path, embeddings)
    metadata = {
        "business_ids": business_ids,
        "feature_names": all_feature_names,
        "structured_dim": X_structured.shape[1],
        "text_dims": [vec.shape[1] for vec in text_vectors],
        "combined_dim": embeddings.shape[1],
        "text_vectorizer_config": {k: v for k, v in config['feature_engineering']['vectorizer']['with_review'].items() if k != 'feature_columns'}
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved metadata to {metadata_path}")
    return embeddings, business_ids, all_feature_names

def create_business_embeddings_with_summary(df, config_path="configs/data_config.yaml", output_dir=None):
    """
    Create business embeddings with review_tip, description, categories, and review_tip_summary text fields.
    Args:
        df (DataFrame): DataFrame with business features and text fields including summary
        config_path (str or dict): Path to config file or config dict
        output_dir (str, optional): Directory to save outputs
    Returns:
        tuple: (embeddings, business_ids, feature_names)
    """
    # Load config
    if isinstance(config_path, str):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config_path
    if output_dir is None:
        output_dir = config['paths']['features_output']['with_review']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Structured features
    feature_columns = config['feature_engineering']['vectorizer']['with_review']['feature_columns']
    business_ids = df['business_id'].tolist()
    X_structured = df[feature_columns].values.astype(float)
    feature_names = list(feature_columns)

    # Text fields - now including review_tip_summary
    text_columns = config['feature_engineering']['vectorizer']['with_review'].get('text_columns', ['review_tip', 'description', 'categories']) + ['review_tip_summary']
    text_vectors = []
    text_feature_names = []

    for text_col in text_columns:
        if text_col == 'review_tip':
            vec_type = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_vectorizer', 'paraphrase-minilm')
            if vec_type in ['paraphrase-minilm', 'bert']:
                model_name = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_bert_model', 'paraphrase-MiniLM-L6-v2')
                pool_method = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_pool_method', 'mean')
                max_tokens = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_max_tokens', 400)
                print(f"Loading SentenceTransformer model for review_tip: {model_name}")
                model = SentenceTransformer(model_name)
                if torch.cuda.is_available():
                    model = model.to('cuda')
                texts = df[text_col].fillna("").tolist()
                X_text = np.stack([
                    segment_encode(text, model, max_tokens=max_tokens, pool_method=pool_method)
                    for text in texts
                ])
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_bert_{i}" for i in range(X_text.shape[1])]
            elif vec_type == 'tfidf':
                max_features = config['feature_engineering']['vectorizer']['with_review'].get('review_tip_max_features', 256)
                vectorizer = TfidfVectorizer(max_features=max_features)
                texts = df[text_col].fillna("").tolist()
                X_text = vectorizer.fit_transform(texts).toarray()
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_tfidf_{i}" for i in range(X_text.shape[1])]
            else:
                raise ValueError(f"Unknown review_tip vectorizer: {vec_type}")
        elif text_col == 'review_tip_summary':
            # Use BERT for summary vectorization
            model_name = config['feature_engineering']['vectorizer']['with_review'].get('description_bert_model', 'all-MiniLM-L6-v2')
            print(f"Loading SentenceTransformer model for review_tip_summary: {model_name}")
            model = SentenceTransformer(model_name)
            if torch.cuda.is_available():
                model = model.to('cuda')
            texts = df[text_col].fillna("").tolist()
            X_text = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
            text_vectors.append(X_text)
            text_feature_names += [f"{text_col}_bert_{i}" for i in range(X_text.shape[1])]
        elif text_col == 'description':
            vec_type = config['feature_engineering']['vectorizer']['with_review'].get('description_vectorizer', 'bert')
            if vec_type == 'bert':
                model_name = config['feature_engineering']['vectorizer']['with_review'].get('description_bert_model', 'all-MiniLM-L6-v2')
                print(f"Loading SentenceTransformer model for description: {model_name}")
                model = SentenceTransformer(model_name)
                if torch.cuda.is_available():
                    model = model.to('cuda')
                texts = df[text_col].fillna("").tolist()
                X_text = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_bert_{i}" for i in range(X_text.shape[1])]
            elif vec_type == 'tfidf':
                max_features = config['feature_engineering']['vectorizer']['with_review'].get('description_max_features', 256)
                vectorizer = TfidfVectorizer(max_features=max_features)
                texts = df[text_col].fillna("").tolist()
                X_text = vectorizer.fit_transform(texts).toarray()
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_tfidf_{i}" for i in range(X_text.shape[1])]
            else:
                raise ValueError(f"Unknown description vectorizer: {vec_type}")
        elif text_col == 'categories':
            vec_type = config['feature_engineering']['vectorizer']['with_review'].get('categories_vectorizer', 'tfidf')
            if vec_type == 'tfidf':
                max_features = config['feature_engineering']['vectorizer']['with_review'].get('categories_max_features', 256)
                vectorizer = TfidfVectorizer(max_features=max_features)
                texts = df[text_col].fillna("").tolist()
                X_text = vectorizer.fit_transform(texts).toarray()
                text_vectors.append(X_text)
                text_feature_names += [f"{text_col}_tfidf_{i}" for i in range(X_text.shape[1])]
            else:
                raise ValueError(f"Unknown categories vectorizer: {vec_type}")
        else:
            raise ValueError(f"Unknown text column: {text_col}")

    # Concatenate all features
    all_vectors = [X_structured] + text_vectors
    all_feature_names = feature_names + text_feature_names
    embeddings = np.concatenate(all_vectors, axis=1)

    # Save embeddings and metadata
    embeddings_path = config['feature_engineering']['output']['with_summary']['business_embeddings_with_summary']
    metadata_path = config['feature_engineering']['output']['with_summary']['business_embeddings_with_summary_metadata']
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    
    np.save(embeddings_path, embeddings)
    metadata = {
        "business_ids": business_ids,
        "feature_names": all_feature_names,
        "structured_dim": X_structured.shape[1],
        "text_dims": [vec.shape[1] for vec in text_vectors],
        "combined_dim": embeddings.shape[1],
        "text_vectorizer_config": {k: v for k, v in config['feature_engineering']['vectorizer']['with_review'].items() if k != 'feature_columns'}
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved metadata to {metadata_path}")
    return embeddings, business_ids, all_feature_names

# Configuration Notes:
# 1. You can configure in configs/data_config.yaml under feature_engineering.vectorizer.with_review:
#    text_vectorizer: "paraphrase-minilm"  # or "bert"/"tfidf"
#    bert_model: "paraphrase-MiniLM-L6-v2"
#    pool_method: "mean"  # or "max"
#    max_tokens: 400
# 2. To change models or pooling methods later, just modify the config file, no code changes needed. 