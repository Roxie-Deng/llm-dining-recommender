import json

def hard_condition_filter(
    input_path: str,
    output_path: str,
    state: str = "PA",
    is_open: int = 1,
    min_stars: float = 3.0,
    min_review_count: int = 10
):
    """
    Filter businesses by hard conditions (state, is_open, stars, review_count).
    
    Args:
        input_path: Path to input JSON file containing business data
        output_path: Path to save filtered business data
        state: Target state for filtering (default: "PA")
        is_open: Business open status (default: 1 for open)
        min_stars: Minimum star rating (default: 3.0)
        min_review_count: Minimum number of reviews (default: 10)
    """
    filtered = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        businesses = json.load(f)
    
    for business in businesses:
        if (
            business.get("state") == state
            and business.get("is_open") == is_open
            and business.get("stars", 0) >= min_stars
            and business.get("review_count", 0) >= min_review_count
        ):
            filtered.append(business)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    
    print(f"Hard condition filtering complete: {len(filtered)} businesses saved to {output_path}")
    print(f"Filter conditions: state={state}, is_open={is_open}, min_stars={min_stars}, min_review_count={min_review_count}") 