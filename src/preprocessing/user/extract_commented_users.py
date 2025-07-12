import argparse
import json
import os
from collections import defaultdict


def extract_commented_users(sampled_business_path, review_path, commented_users_path, user_business_map_path=None):
    # 1. Read sampled business IDs
    with open(sampled_business_path, 'r', encoding='utf-8') as f:
        if sampled_business_path.endswith('.json'):
            business_objs = json.load(f)
            # Support both list of dicts and list of ids
            if isinstance(business_objs, list) and isinstance(business_objs[0], dict) and 'business_id' in business_objs[0]:
                sampled_business_ids = set(b['business_id'] for b in business_objs)
            else:
                sampled_business_ids = set(business_objs)
        else:
            sampled_business_ids = set(line.strip() for line in f if line.strip())

    # 2. Count user comments on sampled businesses
    user_business = defaultdict(set)  # user_id -> set of business_id

    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            business_id = review.get('business_id')
            user_id = review.get('user_id')
            if business_id in sampled_business_ids:
                user_business[user_id].add(business_id)

    # 3. Output commented user statistics
    with open(commented_users_path, 'w', encoding='utf-8') as f:
        f.write('user_id,num_sampled_business_commented\n')
        for user_id, businesses in user_business.items():
            f.write(f'{user_id},{len(businesses)}\n')

    # 4. Optional: Output user-business interaction details
    if user_business_map_path:
        with open(user_business_map_path, 'w', encoding='utf-8') as f:
            f.write('user_id,business_id\n')
            for user_id, businesses in user_business.items():
                for business_id in businesses:
                    f.write(f'{user_id},{business_id}\n')


def main():
    parser = argparse.ArgumentParser(description='Extract users who commented on sampled businesses.')
    parser.add_argument('--sampled_business_path', type=str, required=True, help='Path to sampled business id list (txt or json)')
    parser.add_argument('--review_path', type=str, required=True, help='Path to Yelp review data (jsonl)')
    parser.add_argument('--commented_users_path', type=str, required=True, help='Output path for commented users csv')
    parser.add_argument('--user_business_map_path', type=str, default=None, help='(Optional) Output path for user-business map csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.commented_users_path), exist_ok=True)
    if args.user_business_map_path:
        os.makedirs(os.path.dirname(args.user_business_map_path), exist_ok=True)

    extract_commented_users(
        args.sampled_business_path,
        args.review_path,
        args.commented_users_path,
        args.user_business_map_path
    )

if __name__ == '__main__':
    main() 