import argparse
import pandas as pd
import os
import json

"""
This script generates business profile tables (v1/v2/v3/v4) for different feature ablation experiments.
- Input: business_profile_base.csv (for v1~v3) or business_profile_base_with_summary.csv (for v4)
- Output: business_profile_vX.csv (X=1,2,3,4)
- All file paths and mode are passed as arguments.
- Output columns are selected/combined according to mode.
"""

def main():
    parser = argparse.ArgumentParser(description='Generate business profile table for different modes (v1/v2/v3/v4).')
    parser.add_argument('--input_base', type=str, required=True, help='Input base CSV (business_profile_base.csv or business_profile_base_with_summary.csv)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV (business_profile_vX.csv)')
    parser.add_argument('--mode', type=str, required=True, choices=['v1', 'v2', 'v3', 'v4'], help='Profile mode: v1/v2/v3/v4')
    parser.add_argument('--desc_col', type=str, default='description', help='Description column name')
    parser.add_argument('--review_col', type=str, default='review', help='Review column name')
    parser.add_argument('--summary_col', type=str, default='summary', help='Summary column name (for v4)')
    args = parser.parse_args()

    # Load input
    if not os.path.exists(args.input_base):
        print(f"[ERROR] Input file {args.input_base} does not exist!")
        return
    df = pd.read_csv(args.input_base)

    # Always keep these structure columns (adjust as needed)
    struct_cols = ['business_id', 'normalized_price', 'OutdoorSeating', 'RestaurantsReservations']
    struct_cols += [col for col in df.columns if col.startswith('category_cluster_')]

    # Mode-specific columns
    if args.mode == 'v1':
        # Only structure features
        out_cols = struct_cols
    elif args.mode == 'v2':
        # Structure + description
        out_cols = struct_cols + [args.desc_col]
    elif args.mode == 'v3':
        # Structure + description + review
        out_cols = struct_cols + [args.desc_col, args.review_col]
    elif args.mode == 'v4':
        # Structure + description + summary
        out_cols = struct_cols + [args.desc_col, args.summary_col]
    else:
        print(f"[ERROR] Unknown mode: {args.mode}")
        return

    # Check columns exist
    for col in out_cols:
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not found in input file!")
            return

    # Output
    # If review_col in out_cols, serialize review list to string for CSV compatibility
    if args.review_col in out_cols and 'review' in df.columns:
        df['review'] = df['review'].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
    df[out_cols].to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"[INFO] Saved {args.mode} profile to {args.output}")

if __name__ == '__main__':
    main() 