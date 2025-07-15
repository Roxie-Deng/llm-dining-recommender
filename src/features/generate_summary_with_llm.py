import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import yaml

"""
This script generates business-level summaries using an LLM (e.g., BART, T5, etc.).
- The prompt template is read from the config file (summarization.prompt), and can be overridden by command line argument.
- All other parameters (model, device, max_length, etc.) are also configurable via config or CLI.
- Input: a CSV with a text column (e.g., review), Output: a CSV with an added summary column.
"""

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_device(device_str=None):
    """Determine device (cuda/cpu) based on user input and availability."""
    if device_str:
        if device_str.lower() == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser(description='Generate business summaries with LLM (for V4 profile).')
    parser.add_argument('--input_path', type=str, required=True, help='Input CSV (e.g., business_profile_base.csv)')
    parser.add_argument('--output_path', type=str, required=True, help='Output CSV (with summary column)')
    parser.add_argument('--config', type=str, default='configs/data_config.yaml', help='Config YAML path')
    parser.add_argument('--text_col', type=str, default='review', help='Column to summarize (default: review)')
    parser.add_argument('--summary_col', type=str, default='summary', help='Output summary column name')
    parser.add_argument('--model', type=str, default=None, help='LLM model name (overrides config)')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda, overrides config)')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt template (overrides config)')
    parser.add_argument('--max_length', type=int, default=None, help='Max summary length')
    parser.add_argument('--min_length', type=int, default=None, help='Min summary length')
    parser.add_argument('--num_beams', type=int, default=None, help='Beam search width')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    summarizer_cfg = config.get('summarization', {})

    # Priority: CLI > config > default
    model_name = args.model or summarizer_cfg.get('model', 'facebook/bart-large-cnn')
    device = get_device(args.device or summarizer_cfg.get('device', None))
    prompt = args.prompt or summarizer_cfg.get('prompt', None)
    max_length = args.max_length or summarizer_cfg.get('max_length', 512)
    min_length = args.min_length or summarizer_cfg.get('min_length', 50)
    num_beams = args.num_beams or summarizer_cfg.get('num_beams', 4)
    temperature = args.temperature or summarizer_cfg.get('temperature', 0.7)
    batch_size = args.batch_size or summarizer_cfg.get('batch_size', 1)

    print(f"Model: {model_name}, Device: {device}, Batch size: {batch_size}")
    if prompt:
        print(f"Prompt (first 80 chars): {prompt[:80]} ...")
    else:
        print("Prompt: [default, plain text]")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # Read input data
    df = pd.read_csv(args.input_path)
    texts = df[args.text_col].fillna("").tolist()
    summaries = []

    for text in tqdm(texts, desc='Generating summary'):
        # Build prompt
        if prompt:
            input_text = prompt.replace('{text}', text)
        else:
            input_text = text
        # Tokenize
        inputs = tokenizer([input_text], max_length=1024, truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    df[args.summary_col] = summaries
    df.to_csv(args.output_path, index=False, encoding='utf-8-sig')
    print(f"Saved with summary to {args.output_path}")

if __name__ == '__main__':
    main() 