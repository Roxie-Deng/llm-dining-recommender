import pandas as pd
import json
import os

# Dynamically determine project root and data paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
USER_BUSINESS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'user', 'user_business_map.csv')
BUSINESS_SAMPLE_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'stratified_sample.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'n_user_stat.csv')

# Adjustable n threshold range
N_LIST = list(range(3, 6))  # You can modify this as needed

# Load sampled business IDs
with open(BUSINESS_SAMPLE_PATH, 'r', encoding='utf-8') as f:
    business_sample = set([item['business_id'] for item in json.load(f)])

# Load user-business interaction details
# Assume columns: user_id, business_id, date
interactions = pd.read_csv(USER_BUSINESS_PATH)
# Keep only interactions with sampled businesses
interactions = interactions[interactions['business_id'].isin(business_sample)]

# Count user activity (number of unique businesses each user interacted with)
user_activity = interactions.groupby('user_id')['business_id'].nunique().reset_index()
user_activity.columns = ['user_id', 'activity']

stat_list = []
for n in N_LIST:
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
            # Should not happen for active users, but just in case
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

    # Stat: business train interaction count (after merging B-class)
    business_train_count = train_df.groupby('business_id').size().reset_index(name='train_interaction_count')
    business_train_count.to_csv(os.path.join(SCRIPT_DIR, f'business_train_interaction_count_n{n}.csv'), index=False, encoding='utf-8-sig')

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
    stat_list.append(stat)

# Output statistics table
stat_df = pd.DataFrame(stat_list)
stat_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print(f'Statistics completed. Results saved to {OUTPUT_PATH}')
print(f'Business train interaction counts saved as business_train_interaction_count_n{{n}}.csv for each n.') 