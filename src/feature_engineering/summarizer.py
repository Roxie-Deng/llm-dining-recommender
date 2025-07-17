import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List
import time

class ReviewSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn', device=None, prompt=None, 
                 max_length=512, min_length=50, num_beams=4, temperature=0.7, batch_size=8):
        self.model_name = model_name
        
        # Set attributes first before using them in pipeline initialization
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.batch_size = batch_size
        
        # Improved device setting logic
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("CUDA available, using GPU")
            else:
                self.device = "cpu"
                print("CUDA not available, using CPU")
        else:
            # Handle string device parameters
            if isinstance(device, str):
                if device.lower() == 'cpu':
                    self.device = "cpu"
                    print("Using CPU as specified")
                elif device.lower() == 'cuda':
                    if torch.cuda.is_available():
                        self.device = "cuda"
                        print("Using GPU as specified")
                    else:
                        self.device = "cpu"
                        print("CUDA requested but not available, falling back to CPU")
                else:
                    self.device = "cpu"
                    print(f"Unknown device '{device}', using CPU")
            else:
                self.device = "cuda" if device == 0 else "cpu"
            if device == 0:
                print("Using GPU as specified")
            else:
                print("Using CPU as specified")
        
        # Load model and tokenizer directly (matching pilot study approach)
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")
        
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
        
        # Enhanced GPU memory management
        self.error_count = 0
        self.max_retries = 3
        
        # Record initial memory usage for baseline
        if torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"Model loaded. Initial GPU memory: {self.initial_memory:.1f} MB")
        else:
            self.initial_memory = 0

    def _clean_gpu_memory(self, force=False):
        """Intelligent GPU memory cleanup with baseline tracking"""
        if not torch.cuda.is_available():
            return
            
        try:
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_threshold = 2000  # 2GB threshold (increased from 1GB)
            
            # Only clean if memory usage is significantly above baseline
            baseline_threshold = self.initial_memory + 500  # 500MB above baseline
            
            if force or (memory_used > memory_threshold and memory_used > baseline_threshold):
                print(f"Cleaning GPU memory (used: {memory_used:.1f} MB, baseline: {self.initial_memory:.1f} MB)")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all CUDA operations
                time.sleep(0.1)  # Small delay to ensure cleanup
                
                # Verify cleanup
                memory_after = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU memory after cleanup: {memory_after:.1f} MB")
                
        except Exception as e:
            print(f"Warning: GPU memory cleanup failed: {e}")

    def _validate_text(self, text):
        """Validate and clean input text (matching pilot study approach)"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic text cleaning (matching pilot study)
        text = ' '.join(text.split())  # Remove excessive whitespace
        text = text.replace('\x00', '').replace('\ufffd', '')  # Remove problematic characters
        
        # Let tokenizer handle truncation during encoding (like pilot study)
        return text

    def _generate_summary(self, text: str) -> str:
        """Generate summary for a single text (matching pilot study approach)"""
        try:
            # Prepare input with truncation at tokenizer level (like pilot study)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            # Generate summary
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                do_sample=True,
                temperature=self.temperature,
                early_stopping=False,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                repetition_penalty=1.0
            )
            # Decode and return
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""

    def summarize_batch(self, texts: List[str], chunk_size=200) -> List[str]:
        """
        Summarize texts in chunks to handle large datasets on T4 GPU
        
        Args:
            texts: List of texts to summarize
            chunk_size: Number of texts to process in each chunk (default: 200 for T4)
        """
        print(f"Processing {len(texts)} texts in chunks of {chunk_size}")
        
        all_summaries = []
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(texts))
            chunk_texts = texts[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} (texts {start_idx + 1}-{end_idx})")
            
            # Process this chunk
            chunk_summaries = self._process_chunk(chunk_texts)
            all_summaries.extend(chunk_summaries)
            
            # Force memory cleanup after each chunk
            if torch.cuda.is_available():
                print(f"Cleaning memory after chunk {chunk_idx + 1}")
                self._clean_gpu_memory(force=True)
                time.sleep(0.5)  # Give GPU time to settle
        
        print(f"Completed processing all {len(texts)} texts")
        return all_summaries

    def _process_chunk(self, texts: List[str]) -> List[str]:
        """Process a single chunk of texts using pilot study approach"""
        prompts = [self.prompt.replace('{text}', self._validate_text(t)) for t in texts]
        summaries = []
        # Process texts one by one (like pilot study) to avoid batch issues
        for i, prompt in enumerate(prompts):
            # Check if text has meaningful content
            text_content = prompt.replace(self.prompt.replace('{text}', ''), '').strip()
            if text_content and len(text_content) > 10:
                retry_count = 0
                while retry_count < self.max_retries:
                    try:
                        summary = self._generate_summary(prompt)
                        summaries.append(summary)
                        self.error_count = 0
                        break
                    except Exception as e:
                        retry_count += 1
                        self.error_count += 1
                        print(f"Error in summarization text {i+1} (attempt {retry_count}): {e}")
                        # Force GPU memory cleanup on CUDA errors
                        if "CUDA" in str(e) and torch.cuda.is_available():
                            print("CUDA error detected, forcing GPU memory cleanup...")
                            self._clean_gpu_memory(force=True)
                            time.sleep(1)  # Wait before retry
                        # If max retries reached, return empty summary
                        if retry_count >= self.max_retries:
                            print(f"Max retries reached for text {i+1}, returning empty summary")
                            summaries.append("")
                            break
            else:
                summaries.append("")
            # Less frequent cleanup
            if torch.cuda.is_available() and i % 10 == 0:
                self._clean_gpu_memory()
            # Force cleanup only if too many errors
            if self.error_count > 5 and torch.cuda.is_available():
                print("Too many errors detected, forcing GPU cleanup...")
                self._clean_gpu_memory(force=True)
                self.error_count = 0
        return summaries 