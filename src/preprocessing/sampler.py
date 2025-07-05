import json
import numpy as np
from collections import Counter
import os

def stratified_sample_by_cluster(data, cluster_key, sample_size, output_path=None):
    """
    Perform stratified sampling by cluster, operating in-memory.
    Args:
        data (list): A list of business dictionaries.
        cluster_key (str): The key indicating cluster id (e.g., 'main_cluster').
        sample_size (int): Total number of samples to draw.
        output_path (str, optional): If provided, saves the result to this path.
    Returns:
        list: The list of sampled business dictionaries.
    """
    if not data:
        return []

    clusters = [entry.get(cluster_key) for entry in data if entry.get(cluster_key) is not None]
    if not clusters:
        return []
        
    counter = Counter(clusters)
    total = sum(counter.values())
    
    samples_per_cluster = {k: int(sample_size * v / total) for k, v in counter.items()}
    
    sampled = []
    rng = np.random.default_rng(42)
    for cluster_id, n in samples_per_cluster.items():
        cluster_entries = [entry for entry in data if entry.get(cluster_key) == cluster_id]
        if n > len(cluster_entries):
            n = len(cluster_entries)
        if n > 0:
            # Use choice on indices to avoid issues with list of dicts
            chosen_indices = rng.choice(len(cluster_entries), n, replace=False)
            sampled.extend([cluster_entries[i] for i in chosen_indices])

    print(f"In-memory stratified sampling complete. Sampled {len(sampled)} items.")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled, f, indent=2, ensure_ascii=False)
        print(f"Stratified sample saved to: {output_path}")
        
    return sampled 