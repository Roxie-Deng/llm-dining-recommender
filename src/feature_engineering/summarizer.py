import torch
from transformers import pipeline
from typing import List

class ReviewSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn', device=None, prompt=None, 
                 max_length=512, min_length=50, num_beams=4, temperature=0.7, batch_size=8):
        self.model_name = model_name
        
        # Force CUDA if available, unless explicitly set to CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = 0  # Force CUDA
                print("CUDA available, using GPU")
            else:
                self.device = -1
                print("CUDA not available, using CPU")
        else:
            self.device = device
            if device == 0:
                print("Using GPU as specified")
            else:
                print("Using CPU as specified")
        
        # Load model with proper device setting
        self.summarizer = pipeline('summarization', model=self.model_name, device=self.device)
        self.prompt = prompt or (
            "Summarize the following restaurant reviews. Include both positive and negative\n"
            "aspects of:\n"
            "• Food quality and taste\n"
            "• Service experience\n"
            "• Price and value\n"
            "• Overall impression\n"
            "Keep the summary factual and balanced.\n"
            "{text}"
        )
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.batch_size = batch_size

    def summarize_batch(self, texts: List[str]) -> List[str]:
        prompts = [self.prompt.replace('{text}', t) for t in texts]
        summaries = []
        
        # Use original batch size for better performance
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i+self.batch_size]
            
            # Filter out empty or very short texts
            valid_prompts = []
            valid_indices = []
            for idx, prompt in enumerate(batch_prompts):
                # Check if text has meaningful content (not just prompt template)
                text_content = prompt.replace(self.prompt.replace('{text}', ''), '').strip()
                if text_content and len(text_content) > 20:  # Require more content
                    valid_prompts.append(prompt)
                    valid_indices.append(idx)
            
            if not valid_prompts:
                # Return empty summaries for invalid texts
                batch_summaries = [""] * len(batch_prompts)
            else:
                try:
                    # Use original parameters for better quality
                    batch_outputs = self.summarizer(
                        valid_prompts, 
                        max_length=self.max_length,
                        min_length=self.min_length,
                        num_beams=self.num_beams,
                        do_sample=True,  # Enable sampling for temperature
                        temperature=self.temperature,
                        truncation=True,
                        pad_token_id=self.summarizer.tokenizer.eos_token_id
                    )
                    
                    # Reconstruct full batch with empty summaries for invalid texts
                    batch_summaries = [""] * len(batch_prompts)
                    for idx, output in zip(valid_indices, batch_outputs):
                        batch_summaries[idx] = output['summary_text']
                        
                except Exception as e:
                    print(f"Error in summarization batch {i//self.batch_size + 1}: {e}")
                    # Return empty summaries on error
                    batch_summaries = [""] * len(batch_prompts)
            
            summaries.extend(batch_summaries)
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available() and i % (self.batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return summaries 