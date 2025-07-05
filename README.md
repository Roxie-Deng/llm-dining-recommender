# LLMRec - Yelp Data Feature Engineering Pipeline

A comprehensive feature engineering and vectorization pipeline for Yelp restaurant data, supporting both review-based and non-review-based feature extraction.

## Features

- **Dual Pipeline Support**: 
  - `without_review`: Processes structured features, descriptions, and categories
  - `with_review`: Adds review and tip text processing
  - `with_summary`: Includes AI-powered review summarization using BART-large-CNN

- **Text Vectorization**: 
  - BERT embeddings using SentenceTransformer models
  - TF-IDF vectorization for categories
  - Segment-wise encoding for long texts

- **AI Summarization**: 
  - Automatic review summarization with configurable parameters
  - Support for custom prompts and generation parameters

## Project Structure

```
├── src/
│   ├── feature_engineering/
│   │   ├── pipeline_with_review_summary.py  # Main pipeline with summarization
│   │   ├── summarizer.py                    # BART-based text summarization
│   │   ├── vectorizer_with_review.py        # Text vectorization utilities
│   │   └── README.md
│   ├── preprocessing/
│   └── utils/
├── configs/
│   └── data_config.yaml                     # Configuration file
├── data/
│   ├── processed/                           # Preprocessed data (included)
│   └── engineered/                          # Feature engineered data (included)
│   └── raw/                                # Raw Yelp data (excluded - too large)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `configs/data_config.yaml` to customize:
- Model parameters (BERT models, batch sizes, etc.)
- Summarization settings (max_length, temperature, etc.)
- Output paths and file locations
- Text processing parameters

## Usage

### Local Environment

```bash
# Run the complete pipeline with summarization
python -m src.feature_engineering.pipeline_with_review_summary
```

### Google Colab

1. Clone this repository in Colab:
```python
!git clone https://github.com/your-username/LLMRec.git
%cd LLMRec
```

2. Install dependencies:
```python
!pip install -r requirements.txt
```

3. Run the pipeline (uses pre-processed data):
```python
!python -m src.feature_engineering.pipeline_with_review_summary
```

**Note**: The repository includes pre-processed data (`data/processed/` and `data/engineered/`), so you can run the pipeline directly without needing the raw Yelp dataset.

## Output Files

The pipeline generates:
- `data/engineered/with_summary/extracted_features_with_summary.json` - Features with AI summaries
- `data/engineered/with_summary/business_embeddings_with_summary.npy` - Vectorized features
- `data/engineered/with_summary/business_embeddings_with_summary_metadata.json` - Metadata

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- SentenceTransformers
- Other dependencies listed in requirements.txt

## License

[Your License Here] 