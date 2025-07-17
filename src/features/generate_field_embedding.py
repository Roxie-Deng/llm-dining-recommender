import argparse
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
import json
import ast

"""
This script generates an embedding .npy file for a specified field (text or structured) in a csv file.
- For text fields, uses SentenceTransformer to encode.
- For structured fields, directly converts to float numpy array.
- All paths, field name, model, batch size are passed as arguments.
- Output: npy file with shape (n_samples, embedding_dim)
"""

def main():
    parser = argparse.ArgumentParser(description='Generate npy embedding for a specified field in a csv.')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--field', type=str, required=True, help='Field/column name to encode')
    parser.add_argument('--output_npy', type=str, required=True, help='Output npy file path')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name (for text field)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding (text field)')
    parser.add_argument('--field_type', type=str, choices=['text', 'struct'], default='text', help='Field type: text or struct')
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"[ERROR] Input file {args.input_csv} does not exist!")
        return
    df = pd.read_csv(args.input_csv)
    if args.field not in df.columns:
        print(f"[ERROR] Field '{args.field}' not found in input file!")
        return
    values = df[args.field].fillna("").tolist() if args.field_type == 'text' else df[args.field].values

    if args.field_type == 'text':
        print(f"[INFO] Loading SentenceTransformer model: {args.model}")
        model = SentenceTransformer(args.model)
        if torch.cuda.is_available():
            model = model.to('cuda')
        print(f"[INFO] Encoding {len(values)} texts...")
        embeddings = []
        for i, val in enumerate(values):
            # Try to parse as list, otherwise treat as string
            review_list = None
            if isinstance(val, list):
                review_list = val
            elif isinstance(val, str):
                try:
                    # Try json.loads first
                    review_list = json.loads(val)
                    if not isinstance(review_list, list):
                        review_list = None
                except Exception:
                    try:
                        review_list = ast.literal_eval(val)
                        if not isinstance(review_list, list):
                            review_list = None
                    except Exception:
                        review_list = None
            # If it's a list, encode each item and average pool
            if review_list is not None:
                review_list = [str(x) for x in review_list if str(x).strip()]
                if len(review_list) == 0:
                    emb = np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
                else:
                    embs = model.encode(review_list, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True)
                    emb = np.mean(embs, axis=0)
            else:
                # Single string, encode directly
                emb = model.encode(str(val), batch_size=1, show_progress_bar=False, convert_to_numpy=True)
            embeddings.append(emb)
        embeddings = np.stack(embeddings)
    else:
        print(f"[INFO] Converting structured field '{args.field}' to float array...")
        embeddings = np.array(values, dtype=np.float32).reshape(-1, 1)

    np.save(args.output_npy, embeddings)
    print(f"[INFO] Saved embedding to {args.output_npy}, shape: {embeddings.shape}")

if __name__ == '__main__':
    main() 