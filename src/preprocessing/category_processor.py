import json
from typing import Set, List
from sentence_transformers import SentenceTransformer, util
from accelerate import Accelerator

class CategoryProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = Accelerator()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = self.accelerator.prepare(self.model)
        
    def extract_unique_categories(self, file_path: str, max_records: int) -> Set[str]:
        """Extract unique categories from business dataset."""
        unique_categories = set()
        record_count = 0

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if record_count >= max_records:
                    break
                business = json.loads(line.strip())
                if "categories" in business and business["categories"]:
                    categories = [cat.strip() for cat in business["categories"].split(",")]
                    unique_categories.update(categories)
                record_count += 1

        return unique_categories

    def find_dining_categories(self, categories: List[str]) -> List[str]:
        """Find dining-related categories using semantic similarity."""
        # Prepare data
        dining_keywords = self.config['categories']['dining_keywords']
        
        # Generate embeddings
        category_embeddings = self.model.encode(
            categories, 
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=self.config['parameters']['batch_size']
        )
        keyword_embeddings = self.model.encode(
            dining_keywords, 
            convert_to_tensor=True
        )
        
        # Move tensors to device
        category_embeddings = self.accelerator.prepare(category_embeddings)
        keyword_embeddings = self.accelerator.prepare(keyword_embeddings)
        
        # Calculate similarities
        similarities = util.cos_sim(keyword_embeddings, category_embeddings)
        
        # Extract categories above threshold
        threshold = self.config['parameters']['similarity_threshold']
        dining_related_indices = (similarities > threshold).nonzero(as_tuple=True)[1]
        dining_related = [categories[idx] for idx in dining_related_indices]
        
        return list(set(dining_related))

    def process_categories(self):
        """Main processing pipeline for categories."""
        # Extract unique categories
        unique_categories = self.extract_unique_categories(
            self.config['paths']['raw']['business'],
            self.config['parameters']['max_scan_records']
        )
        
        # Save unique categories
        with open(self.config['paths']['processed']['categories']['unique'], 'w', encoding='utf-8') as f:
            json.dump(sorted(unique_categories), f, indent=2, ensure_ascii=False)
        
        # Find dining-related categories
        dining_categories = self.find_dining_categories(list(unique_categories))
        
        # Save dining categories
        with open(self.config['paths']['processed']['categories']['dining'], 'w', encoding='utf-8') as f:
            json.dump(sorted(dining_categories), f, indent=2, ensure_ascii=False)
        
        return len(unique_categories), len(dining_categories) 