import argparse
import pandas as pd
import json
import numpy as np
import ast

def normalize_price(price):
    try:
        price = int(price)
        return (price - 1) / 3  # Normalize 1-4 to 0-1
    except:
        return np.nan  # 用于后续中位数填充

def binarize_bool(val):
    if str(val).lower() == 'true':
        return 1
    elif str(val).lower() == 'false':
        return 0
    else:
        return np.nan  # 用于后续0填充

def process_ambience(val):
    if pd.isna(val) or val is None or val == '' or val == 'None':
        return ''
    if isinstance(val, dict):
        true_keys = [k for k, v in val.items() if v is True]
        return ', '.join(sorted(true_keys))
    if isinstance(val, str):
        # 尝试解析为dict
        if ('{' in val and '}' in val) or ("'" in val):
            try:
                d = ast.literal_eval(val)
                if isinstance(d, dict):
                    true_keys = [k for k, v in d.items() if v is True]
                    return ', '.join(sorted(true_keys))
            except:
                pass
        # 不是dict，直接返回字符串
        return val.strip()
    return ''

def aggregate_reviews(reviews_df, business_id, n=10):
    reviews = reviews_df[reviews_df['business_id'] == business_id]
    reviews = reviews.sort_values('date')
    texts = reviews['text'].fillna('').tolist()[-n:]
    return ' '.join(texts)

def generate_description(row):
    name = row.get('name', '')
    price = row.get('normalized_price', np.nan)
    # price_tier
    if pd.isna(price):
        price_tier = 'mid-range'
    elif price < 0.25:
        price_tier = 'budget-friendly'
    elif price < 0.5:
        price_tier = 'affordable'
    elif price < 0.75:
        price_tier = 'expensive'
    else:
        price_tier = 'luxury'
    # categories
    categories = row.get('categories', '')
    if pd.isna(categories):
        categories = ''
    category_phrase = f"categorised under {categories}" if categories else ''
    # ambience
    ambience = row.get('ambience', '')
    ambience_phrase = f", with {ambience} ambience" if ambience else ''
    # 拼接
    desc = f"{name}: {price_tier} eatery {category_phrase}{ambience_phrase}"
    return desc.strip()

def main():
    parser = argparse.ArgumentParser(description='Generate business profile base table with categories, ambience, description, and review aggregation.')
    parser.add_argument('--business_json', type=str, required=True, help='Path to stratified_sample.json (business metadata)')
    parser.add_argument('--train_interactions', type=str, required=True, help='Path to train_interactions_nX.csv (train set interactions)')
    parser.add_argument('--review_data', type=str, required=True, help='Path to review data (jsonl or csv)')
    parser.add_argument('--output_path', type=str, required=True, help='Output CSV path for business_profile_base.csv')
    parser.add_argument('--review_top_n', type=int, default=10, help='Number of latest reviews to aggregate per business (default: 10)')
    args = parser.parse_args()

    # Load business metadata
    with open(args.business_json, 'r', encoding='utf-8') as f:
        business_list = json.load(f)
    business_df = pd.DataFrame(business_list)

    # 处理categories缺失
    if 'categories' not in business_df.columns:
        business_df['categories'] = ''
    business_df['categories'] = business_df['categories'].fillna('').astype(str)

    # 处理ambience缺失和格式
    if 'Ambience' in business_df.columns:
        business_df['ambience'] = business_df['Ambience'].apply(process_ambience)
    else:
        business_df['ambience'] = ''

    # 结构化特征
    business_df['normalized_price'] = business_df['RestaurantsPriceRange2'].apply(normalize_price)
    # 缺失用中位数填充
    median_price = business_df['normalized_price'].median()
    business_df['normalized_price'] = business_df['normalized_price'].fillna(median_price)

    for col in ['OutdoorSeating', 'RestaurantsReservations']:
        if col in business_df.columns:
            business_df[col] = business_df[col].apply(binarize_bool)
            business_df[col] = business_df[col].fillna(0).astype(int)
        else:
            business_df[col] = 0

    # One-hot encode category_cluster if present
    if 'cluster' in business_df.columns:
        one_hot = pd.get_dummies(business_df['cluster'], prefix='category_cluster').astype(int)
        business_df = pd.concat([business_df, one_hot], axis=1)

    # Load train interactions to get valid business_ids
    train_inter = pd.read_csv(args.train_interactions)
    train_business_ids = set(train_inter['business_id'].unique())
    business_df = business_df[business_df['business_id'].isin(train_business_ids)].copy()

    # Load review data
    if args.review_data.endswith('.json') or args.review_data.endswith('.jsonl'):
        reviews = []
        with open(args.review_data, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    reviews.append(json.loads(line))
        reviews_df = pd.DataFrame(reviews)
    else:
        reviews_df = pd.read_csv(args.review_data)

    # Only keep reviews for businesses in train set
    reviews_df = reviews_df[reviews_df['business_id'].isin(train_business_ids)]

    # 聚合review和生成description
    agg_reviews = []
    descriptions = []
    for idx, row in business_df.iterrows():
        business_id = row['business_id']
        agg_reviews.append(aggregate_reviews(reviews_df, business_id, n=args.review_top_n))
        descriptions.append(generate_description(row))
    business_df['review'] = agg_reviews
    business_df['description'] = descriptions

    # 输出列
    output_columns = ['business_id', 'categories', 'ambience', 'normalized_price', 'OutdoorSeating', 'RestaurantsReservations']
    output_columns += [col for col in business_df.columns if col.startswith('category_cluster_')]
    output_columns += ['description', 'review']
    business_df[output_columns].to_csv(args.output_path, index=False, encoding='utf-8-sig')
    print(f"Business profile base table saved to {args.output_path}")

if __name__ == '__main__':
    main() 