# Feature Engineering Module

This module handles feature extraction and vectorization for the LLMRec recommendation system. Currently supports feature engineering without review data, with plans to add review-based features.

## File Structure

```
src/feature_engineering/
├── __init__.py                           # Module initialization
├── feature_extractor_without_review.py   # Feature extractor (without reviews)
├── vectorizer_without_review.py          # Vectorizer (without reviews)
├── pipeline_without_review.py            # Main pipeline (without reviews)
└── README.md                             # Documentation
```

## Planned Extensions

Future versions will include:
- `feature_extractor_with_review.py`      # Feature extractor (with reviews)
- `vectorizer_with_review.py`             # Vectorizer (with reviews)
- `pipeline_with_review.py`               # Main pipeline (with reviews)

## Usage

```python
from src.feature_engineering import FeatureEngineeringPipelineWithoutReview

pipeline = FeatureEngineeringPipelineWithoutReview()
results = pipeline.run_pipeline("data/processed/stratified_sample.json")
```

*Documentation will be updated as the module is completed.* 