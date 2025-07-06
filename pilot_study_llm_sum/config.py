"""
Configuration file for the restaurant review summarization experiment.

This script contains all constants and configuration parameters used across the experiment:
- Model names and parameters
- Prompt templates
- Evaluation metrics settings
- File paths
- Other experiment parameters

Input: None
Output: None (This is a configuration file)
"""

# Model configurations
MODEL_NAMES = {
    't5': 'google/flan-t5-base',
    'bart': 'facebook/bart-large-cnn'
}

# Model parameters
MODEL_PARAMS = {
    'max_length': 512,
    'min_length': 50,
    'num_beams': 4,
    'temperature': 0.7
}

# Prompt templates
PROMPT_TEMPLATES = {
    "basic": """Summarize the following restaurant reviews.\n\n[REVIEW_TEXT_HERE]""",
    
    "zero_shot": """Summarize the following restaurant reviews. Include both positive and negative aspects of:
- Food quality and taste
- Service experience
- Price and value
- Overall impression
Keep the summary factual and balanced.\n\n[REVIEW_TEXT_HERE]""",

   "role_zero_shot": """You are a professional restaurant critic. Your job is to write concise, objective summaries based on customer reviews. Include both positive and negative aspects of:
- Food quality and taste
- Service experience
- Price and value
- Overall impression
Keep the summary factual and balanced.\n\n[REVIEW_TEXT_HERE]""",
    
    "few_shot": """Summarize the following restaurant reviews. Include both positive and negative aspects.

Example 1:
Input: "The steak was juicy but a bit salty. Service was quick but the waiter was cold. Prices were okay."
Output: "Juicy steak but slightly oversalted. Fast service with a lack of friendliness. Reasonable pricing."

Example 2:
Input: "Loved the dessert, but the pasta was undercooked. The staff were attentive. It's a bit pricey though."
Output: "Delicious desserts but undercooked pasta. Attentive service. Slightly expensive."

Now summarize this review:
[REVIEW_TEXT_HERE]""",
    
    "role_few_shot": """You are a professional restaurant critic. Your job is to write concise, objective summaries based on customer reviews.

Example 1:
Input: "The steak was juicy but a bit salty. Service was quick but the waiter was cold. Prices were okay."
Output: "Juicy steak but slightly oversalted. Fast service with a lack of friendliness. Reasonable pricing."

Example 2:
Input: "Loved the dessert, but the pasta was undercooked. The staff were attentive. It's a bit pricey though."
Output: "Delicious desserts but undercooked pasta. Attentive service. Slightly expensive."

Please summarize the following:
[REVIEW_TEXT_HERE]""",
    
    "cot": """Let's summarize this restaurant review step-by-step:

1. Identify the positive and negative aspects of the food.
2. Identify the service experience (good and bad).
3. Mention price and overall value.
4. Conclude with a short overall impression.

Review:
[REVIEW_TEXT_HERE]

Now, begin step-by-step:""",

    "role_cot": """You are a professional restaurant critic. Your job is to write concise, objective summaries based on customer reviews.

1. Identify the positive and negative aspects of the food.
2. Identify the service experience (good and bad).
3. Mention price and overall value.
4. Conclude with a short overall impression.

Review:
[REVIEW_TEXT_HERE]

Now, begin step-by-step:"""
}

# Evaluation metrics settings
EVALUATION_METRICS = {
    'rouge': ['rouge1', 'rougeL'],
    'bleu': True,
    'length': True,
    'repetition': True,
    'sentiment': True
}

# File paths
DATA_PATH = '5_sumy_gold.json'
RESULTS_DIR = 'results'

# Human evaluation settings
NUM_SAMPLES_FOR_HUMAN_EVAL = 3 