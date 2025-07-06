import torch
from transformers import pipeline
from typing import List
import time

class ReviewSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn', device=None, prompt=None, 
                 max_length=512, min_length=50, num_beams=4, temperature=0.7, batch_size=8):
        self.model_name = model_name
        
        # Improved device setting logic
        if device is None:
            if torch.cuda.is_available():
                self.device = 0  # Force CUDA
                print("CUDA available, using GPU")
            else:
                self.device = -1
                print("CUDA not available, using CPU")
        else:
            # Handle string device parameters
            if isinstance(device, str):
                if device.lower() == 'cpu':
                    self.device = -1
                    print("Using CPU as specified")
                elif device.lower() == 'cuda':
                    if torch.cuda.is_available():
                        self.device = 0
                        print("Using GPU as specified")
                    else:
                        self.device = -1
                        print("CUDA requested but not available, falling back to CPU")
                else:
                    self.device = -1
                    print(f"Unknown device '{device}', using CPU")
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
        
        # GPU memory management
        self.error_count = 0
        self.max_retries = 3

    def _clean_gpu_memory(self, force=False):
        """Intelligent GPU memory cleanup"""
        if not torch.cuda.is_available():
            return
            
        try:
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_threshold = 1000  # 1GB threshold
            
            if force or memory_used > memory_threshold:
                print(f"Cleaning GPU memory (used: {memory_used:.1f} MB)")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all CUDA operations
                time.sleep(0.1)  # Small delay to ensure cleanup
                
                # Verify cleanup
                memory_after = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU memory after cleanup: {memory_after:.1f} MB")
                
        except Exception as e:
            print(f"Warning: GPU memory cleanup failed: {e}")

    def _validate_text(self, text):
        """Validate and clean input text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit text length to prevent GPU memory issues
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        # Remove problematic characters
        text = text.replace('\x00', '').replace('\ufffd', '')
        
        return text

    def summarize_batch(self, texts: List[str]) -> List[str]:
        prompts = [self.prompt.replace('{text}', self._validate_text(t)) for t in texts]
        summaries = []
        
        # Use more conservative batch processing
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i+self.batch_size]
            
            # Filter out empty or very short texts
            valid_prompts = []
            valid_indices = []
            for idx, prompt in enumerate(batch_prompts):
                # Check if text has meaningful content (not just prompt template)
                text_content = prompt.replace(self.prompt.replace('{text}', ''), '').strip()
                if text_content and len(text_content) > 10:  # Lower content requirement
                    valid_prompts.append(prompt)
                    valid_indices.append(idx)
            
            if not valid_prompts:
                # Return empty summaries for invalid texts
                batch_summaries = [""] * len(batch_prompts)
            else:
                retry_count = 0
                while retry_count < self.max_retries:
                    try:
                        # Use more conservative parameter settings
                        batch_outputs = self.summarizer(
                            valid_prompts, 
                            max_length=self.max_length,
                            min_length=self.min_length,
                            num_beams=self.num_beams,
                            do_sample=True,  # Enable sampling for temperature
                            temperature=self.temperature,
                            truncation=True,
                            pad_token_id=self.summarizer.tokenizer.eos_token_id,
                            early_stopping=False,  # Disable early_stopping
                            length_penalty=1.0  # Use default length_penalty
                        )
                        
                        # Reconstruct full batch with empty summaries for invalid texts
                        batch_summaries = [""] * len(batch_prompts)
                        for idx, output in zip(valid_indices, batch_outputs):
                            batch_summaries[idx] = output['summary_text']
                        
                        # Reset error count on success
                        self.error_count = 0
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        self.error_count += 1
                        
                        print(f"Error in summarization batch {i//self.batch_size + 1} (attempt {retry_count}): {e}")
                        
                        # Force GPU memory cleanup on CUDA errors
                        if "CUDA" in str(e) and torch.cuda.is_available():
                            print("CUDA error detected, forcing GPU memory cleanup...")
                            self._clean_gpu_memory(force=True)
                            time.sleep(1)  # Wait before retry
                        
                        # If max retries reached, return empty summaries
                        if retry_count >= self.max_retries:
                            print(f"Max retries reached for batch {i//self.batch_size + 1}, returning empty summaries")
                            batch_summaries = [""] * len(batch_prompts)
                            break
            
            summaries.extend(batch_summaries)
            
            # Intelligent GPU memory management
            if torch.cuda.is_available() and i % (self.batch_size * 10) == 0:
                self._clean_gpu_memory()
            
            # Force cleanup if too many errors
            if self.error_count > 5 and torch.cuda.is_available():
                print("Too many errors detected, forcing GPU cleanup...")
                self._clean_gpu_memory(force=True)
                self.error_count = 0
        
        return summaries 