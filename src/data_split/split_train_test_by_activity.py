import argparse
import pandas as pd
import json
import os

def split_train_test(
    user_business_path,
    business_sample_path,
    output_dir,
    n=4
):
    """
    Split user-business interactions into train/test sets by user activity threshold (n).
    Train set = A-class users' train (leave-one-out) + all B-class users' interactions.
    Test set = A-class users' test (last interaction).
    All paths and parameters are configurable.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load sampled business IDs
    with open(business_sample_path, 'r', encoding='utf-8') as f:
        business_sample = set([item['business_id'] for item in json.load(f)])

    # Load user-business interaction details
    # Assume columns: user_id, business_id, date
    interactions = pd.read_csv(user_business_path)
    # Keep only interactions with sampled businesses
    interactions = interactions[interactions['business_id'].isin(business_sample)]

    # Count user activity (number of unique businesses each user interacted with)
    user_activity = interactions.groupby('user_id')['business_id'].nunique().reset_index()
    user_activity.columns = ['user_id', 'activity']

    # Define A-class (active) and B-class (inactive) users
    active_users = set(user_activity[user_activity['activity'] >= n]['user_id'])
    inactive_users = set(user_activity[user_activity['activity'] < n]['user_id'])

    active_inter = interactions[interactions['user_id'].isin(active_users)]
    inactive_inter = interactions[interactions['user_id'].isin(inactive_users)]

    # Leave-one-out split for A-class users
    a_train_rows = []
    a_test_rows = []
    for user_id, user_df in active_inter.groupby('user_id'):
        user_df_sorted = user_df.sort_values('date')
        if len(user_df_sorted) < 2:
            a_train_rows.append(user_df_sorted)
            continue
        test_row = user_df_sorted.iloc[[-1]]  # last interaction as test
        train_row = user_df_sorted.iloc[:-1]  # all but last as train
        a_test_rows.append(test_row)
        a_train_rows.append(train_row)
    if a_train_rows:
        a_train_df = pd.concat(a_train_rows)
    else:
        a_train_df = pd.DataFrame(columns=active_inter.columns)
    if a_test_rows:
        a_test_df = pd.concat(a_test_rows)
    else:
        a_test_df = pd.DataFrame(columns=active_inter.columns)

    # All B-class users' interactions go to train
    b_train_df = inactive_inter

    # Merge A-class train and B-class train to form the final train set
    train_df = pd.concat([a_train_df, b_train_df], ignore_index=True)
    test_df = a_test_df

    # Save train and test sets
    train_path = os.path.join(output_dir, f'train_interactions_n{n}.csv')
    test_path = os.path.join(output_dir, f'test_interactions_n{n}.csv')
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    # Save statistics
    stat = {
        'n': n,
        'num_active_users': len(active_users),
        'num_inactive_users': len(inactive_users),
        'num_active_interactions': len(active_inter),
        'num_inactive_interactions': len(inactive_inter),
        'total_interactions': len(interactions),
        'num_train_interactions': len(train_df),
        'num_test_interactions': len(test_df)
    }
    stat_path = os.path.join(output_dir, f'split_stat_n{n}.json')
    with open(stat_path, 'w', encoding='utf-8') as f:
        json.dump(stat, f, indent=2)

    print(f"Train set saved to {train_path}")
    print(f"Test set saved to {test_path}")
    print(f"Statistics saved to {stat_path}")

def main():
    parser = argparse.ArgumentParser(description='Split train/test sets by user activity threshold (n).')
    parser.add_argument('--user_business_path', type=str, required=True, help='Path to user-business interaction CSV (must include user_id, business_id, date)')
    parser.add_argument('--business_sample_path', type=str, required=True, help='Path to sampled business id list (json)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save split results')
    parser.add_argument('--n', type=int, default=4, help='User activity threshold for A-class users (default: 4)')
    args = parser.parse_args()

    split_train_test(
        user_business_path=args.user_business_path,
        business_sample_path=args.business_sample_path,
        output_dir=args.output_dir,
        n=args.n
    )

if __name__ == '__main__':
    main() 